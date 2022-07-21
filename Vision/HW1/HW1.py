import cv2
import numpy as np
from matplotlib import pyplot as plt


def MatchSIFT(loc1, des1, loc2, des2):
    """
    Find the matches of SIFT features between two images

    Parameters
    ----------
    loc1 : ndarray of shape (n1, 2)
        Keypoint locations in image 1
    des1 : ndarray of shape (n1, 128)
        SIFT descriptors of the keypoints image 1
    loc2 : ndarray of shape (n2, 2)
        Keypoint locations in image 2
    des2 : ndarray of shape (n2, 128)
        SIFT descriptors of the keypoints image 2

    Returns
    -------
    x1 : ndarray of shape (n, 2)
        Matched keypoint locations in image 1
    x2 : ndarray of shape (n, 2)
        Matched keypoint locations in image 2
    """

    from sklearn.neighbors import NearestNeighbors

    x2y = []
    y2x = []

    # x2y[i] : x[i] -> y[f(i)]
    # y2x[i] : y[i] -> x[f(i)]

    # Nearest Neighbor Index from image 1 to image 2
    nn = NearestNeighbors(n_neighbors=2)
    dist, index = nn.fit(des2).kneighbors(des1)
    # Ratio Test
    for i in range(len(dist)):
        if dist[i][0] / dist[i][1] < 0.8:
            x2y.append(index[i][0])
        else:
            x2y.append(-1)

    # Nearest Neighbor Index from image 2 to image 1
    dist, index = nn.fit(des1).kneighbors(des2)
    # Ratio Test
    for i in range(len(dist)):
        if dist[i][0] / dist[i][1] < 0.8:
            y2x.append(index[i][0])
        else:
            y2x.append(-1)

    # Bi-Directional consistency
    x1 = []
    x2 = []

    for i in range(len(x2y)):
        if i < 0 or x2y[i] < 0:
            continue
        if y2x[x2y[i]] == i and i not in x1 and x2y[i] not in x2:
            # x1.append(loc1[i])
            # x2.append(loc2[x2y[i]])
            x1.append(i)
            x2.append(x2y[i])

    x1 = [loc1[i] for i in x1]
    x2 = [loc2[i] for i in x2]

    x1 = np.array(x1)
    x2 = np.array(x2)
    return x1, x2


def EstimateH(x1, x2, ransac_n_iter, ransac_thr):
    """
    Estimate the homography between images using RANSAC

    Parameters
    ----------
    x1 : ndarray of shape (n, 2)
        Matched keypoint locations in image 1
    x2 : ndarray of shape (n, 2)
        Matched keypoint locations in image 2
    ransac_n_iter : int
        Number of RANSAC iterations
    ransac_thr : float
        Error threshold for RANSAC

    Returns
    -------
    H : ndarray of shape (3, 3)
        The estimated homography
    inlier : ndarray of shape (k,)
        The inlier indices
    """

    H = None
    inlier = []

    # Homogeous coordinate
    x1_ = np.array([x1[:, 0], x1[:, 1], np.ones(x1.shape[0])]).T
    x2_ = np.array([x2[:, 0], x2[:, 1], np.ones(x2.shape[0])]).T

    for _ in range(ransac_n_iter):
        # Randomly sample 4 points
        idx = np.random.choice(len(x1), 4, replace=False)
        x1_sample = x1[idx]
        x2_sample = x2[idx]

        # Estimate homography by DLT
        A = np.zeros((8, 9))
        for j in range(4):
            A[2 * j] = [
                x1_sample[j][0],
                x1_sample[j][1],
                1,
                0,
                0,
                0,
                -x2_sample[j][0] * x1_sample[j][0],
                -x2_sample[j][0] * x1_sample[j][1],
                -x2_sample[j][0],
            ]
            A[2 * j + 1] = [
                0,
                0,
                0,
                x1_sample[j][0],
                x1_sample[j][1],
                1,
                -x2_sample[j][1] * x1_sample[j][0],
                -x2_sample[j][1] * x1_sample[j][1],
                -x2_sample[j][1],
            ]

        _, _, vh = np.linalg.svd(A)
        H_ = vh[-1].reshape(3, 3)
        # Compute the error
        # err = np.linalg.norm(x2 - np.matmul(H, x1), axis=1)

        err = np.zeros(x1.shape[0])
        for i in range(x1.shape[0]):
            xx = np.matmul(H_, x1_[i])
            xx = xx / xx[2]
            err[i] = np.linalg.norm(x2_[i] - xx)

        # Find inliers
        inlier_ = np.where(err < ransac_thr)[0]

        # If we have more inliers than before, keep the homography
        if len(inlier_) > len(inlier):
            inlier = inlier_
            H = H_

    # DLT with all inliers
    A = np.zeros((2 * len(inlier), 9))
    for j in range(len(inlier)):
        A[2 * j] = [
            x1[inlier[j]][0],
            x1[inlier[j]][1],
            1,
            0,
            0,
            0,
            -x2[inlier[j]][0] * x1[inlier[j]][0],
            -x2[inlier[j]][0] * x1[inlier[j]][1],
            -x2[inlier[j]][0],
        ]
        A[2 * j + 1] = [
            0,
            0,
            0,
            x1[inlier[j]][0],
            x1[inlier[j]][1],
            1,
            -x2[inlier[j]][1] * x1[inlier[j]][0],
            -x2[inlier[j]][1] * x1[inlier[j]][1],
            -x2[inlier[j]][1],
        ]

    _, _, vh = np.linalg.svd(A)
    H = vh[-1].reshape(3, 3)

    return H, inlier


def EstimateR(H, K):
    """
    Compute the relative rotation matrix

    Parameters
    ----------
    H : ndarray of shape (3, 3)
        The estimated homography
    K : ndarray of shape (3, 3)
        The camera intrinsic parameters

    Returns
    -------
    R : ndarray of shape (3, 3)
        The relative rotation matrix from image 1 to image 2
    """

    # H = K R K^-1
    # R = K^-1 H K
    Kinv = np.linalg.inv(K)
    R = Kinv @ H @ K
    if np.linalg.det(R) < 0:
        R = -R
    return R


def ConstructCylindricalCoord(Wc, Hc, K):
    """
    Generate 3D points on the cylindrical surface

    Parameters
    ----------
    Wc : int
        The width of the canvas
    Hc : int
        The height of the canvas
    K : ndarray of shape (3, 3)
        The camera intrinsic parameters of the source images

    Returns
    -------
    p : ndarray of shape (Hc, Hc, 3)
        The 3D points corresponding to all pixels in the canvas
    """

    # (w, h) -> (f sin (w * 2pi / Wc), Hc / 2 - h, f cos (w * 2pi / Wc))
    x = np.linspace(0, Wc - 1, Wc)
    y = np.linspace(0, Hc - 1, Hc)
    X, Y = np.meshgrid(x, y)
    X = X.astype(np.uint32)
    Y = Y.astype(np.uint32)
    f = K[0, 0]

    p = np.zeros((Hc, Wc, 3))
    for i in range(Hc):
        for j in range(Wc):
            p[i, j, 0] = -f * np.cos(X[i, j] * 2 * np.pi / Wc)
            p[i, j, 1] = Y[i, j] - Hc / 2
            p[i, j, 2] = f * np.sin(X[i, j] * 2 * np.pi / Wc)

    return p


def Projection(p, K, R, W, H):
    """
    Project the 3D points to the camera plane

    Parameters
    ----------
    p : ndarray of shape (Hc, Wc, 3)
        A set of 3D points that correspond to every pixel in the canvas image
    K : ndarray of shape (3, 3)
        The camera intrinsic parameters
    R : ndarray of shape (3, 3)
        The rotation matrix
    W : int
        The width of the source image
    H : int
        The height of the source image

    Returns
    -------
    u : ndarray of shape (Hc, Wc, 2)
        The 2D projection of the 3D points
    mask : ndarray of shape (Hc, Wc)
        The corresponding binary mask indicating valid pixels
    """

    Hc = p.shape[0]
    Wc = p.shape[1]
    u = np.zeros((Hc, Wc, 2))
    mask = np.zeros((Hc, Wc))

    # (w, h) -> X=(f sin (w * 2pi / Wc), Hc / 2 - h, f cos (w * 2pi / Wc)) -> PX = KR[I|0] X = KRX
    X = p[0, 0]
    X = np.matmul(K, R) @ X
    for i in range(Hc):
        for j in range(Wc):
            X = p[i, j]
            X = np.matmul(K, R) @ X
            if X[2] != 0:
                X = X / X[2]
                u[i, j, 0] = X[0]
                u[i, j, 1] = X[1]
                if (
                    0 > u[i, j, 0]
                    or u[i, j, 0] > W - 1
                    or u[i, j, 1] < 0
                    or u[i, j, 1] > H - 1
                ):
                    mask[i, j] = 0
                elif (R @ p[i, j])[2] < 0:
                    mask[i, j] = 0
                else:
                    mask[i, j] = 1
            else:
                mask[i, j] = 0

    return u, mask


def WarpImage2Canvas(image_i, u, mask_i):
    """
    Warp the image to the cylindrical canvas

    Parameters
    ----------
    image_i : ndarray of shape (H, W, 3)
        The i-th image with width W and height H
    u : ndarray of shape (Hc, Wc, 2)
        The mapped 2D pixel locations in the source image for pixel transport
    mask_i : ndarray of shape (Hc, Wc)
        The valid pixel indicator

    Returns
    -------
    canvas_i : ndarray of shape (Hc, Wc, 3)
        the canvas image generated by the i-th source image
    """

    Hc = u.shape[0]
    Wc = u.shape[1]
    H = image_i.shape[0]
    W = image_i.shape[1]
    canvas_i = np.zeros((Hc, Wc, 3))

    for i in range(Hc):
        for j in range(Wc):
            if mask_i[i, j] == 1:
                x, y = int(u[i, j, 0]), int(u[i, j, 1])
                canvas_i[i, j] = image_i[y, x]

    return canvas_i


def UpdateCanvas(canvas, canvas_i, mask_i):
    """
    Update the canvas with the new warped image

    Parameters
    ----------
    canvas : ndarray of shape (Hc, Wc, 3)
        The previously generated canvas
    canvas_i : ndarray of shape (Hc, Wc, 3)
        The i-th canvas
    mask_i : ndarray of shape (Hc, Wc)
        The mask of the valid pixels on the i-th canvas

    Returns
    -------
    canvas : ndarray of shape (Hc, Wc, 3)
        The updated canvas image
    """

    Hc = canvas.shape[0]
    Wc = canvas.shape[1]
    for i in range(Hc):
        for j in range(Wc):
            if mask_i[i, j] == 1:
                canvas[i, j] = canvas_i[i, j]
    return canvas


if __name__ == "__main__":
    ransac_n_iter = 500
    ransac_thr = 3
    K = np.asarray([[320, 0, 480], [0, 320, 270], [0, 0, 1]])

    # Read all images
    im_list = []
    for i in range(1, 9):
        im_file = "{}.jpg".format(i)
        im = cv2.imread(im_file)
        im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
        im_list.append(im)

    rot_list = []
    rot_list.append(np.eye(3))
    for i in range(len(im_list) - 1):
        # Load consecutive images I_i and I_{i+1}
        img_1 = im_list[i]
        img_2 = im_list[i + 1]

        # Extract SIFT features
        sift = cv2.xfeatures2d.SIFT_create()
        kp1, des1 = sift.detectAndCompute(img_1, None)
        kp2, des2 = sift.detectAndCompute(img_2, None)

        loc1 = [kp.pt for kp in kp1]
        loc2 = [kp.pt for kp in kp2]

        loc1 = np.array(loc1)
        loc2 = np.array(loc2)

        # Find the matches between two images (x1 <--> x2)
        x1, x2 = MatchSIFT(loc1, des1, loc2, des2)

        # Estimate the homography between images using RANSAC
        H, inlier = EstimateH(x1, x2, ransac_n_iter, ransac_thr)

        # Compute the relative rotation matrix R
        R = EstimateR(H, K)

        # Compute R_new (or R_i+1)
        R_new = R @ rot_list[-1]
        rot_list.append(R_new)

    Him = im_list[0].shape[0]
    Wim = im_list[0].shape[1]

    Hc = Him
    Wc = len(im_list) * Wim // 2

    canvas = np.zeros((Hc, Wc, 3), dtype=np.uint8)
    p = ConstructCylindricalCoord(Wc, Hc, K)

    fig = plt.figure("HW1")
    plt.axis("off")
    plt.ion()
    plt.show()
    for i, (im_i, rot_i) in enumerate(zip(im_list, rot_list)):
        # Project the 3D points to the i-th camera plane
        u, mask_i = Projection(p, K, rot_i, Wim, Him)
        # Warp the image to the cylindrical canvas
        canvas_i = WarpImage2Canvas(im_i, u, mask_i)
        # Update the canvas with the new warped image
        canvas = UpdateCanvas(canvas, canvas_i, mask_i)
        plt.imshow(canvas)
        plt.savefig(
            "output_{}.png".format(i + 1), dpi=600, bbox_inches="tight", pad_inches=0
        )
