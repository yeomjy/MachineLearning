import cv2
import numpy as np



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
    x1 : ndarray of shape (m, 2)
        The matched keypoint locations in image 1
    x2 : ndarray of shape (m, 2)
        The matched keypoint locations in image 2
    ind1 : ndarray of shape (m,)
        The indices of x1 in loc1
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

    ind1 = np.array(x1)

    x1 = [loc1[i] for i in x1]
    x2 = [loc2[i] for i in x2]

    x1 = np.array(x1)
    x2 = np.array(x2)
    return x1, x2, ind1





def EstimateE(x1, x2):
    """
    Estimate the essential matrix, which is a rank 2 matrix with singular values
    (1, 1, 0)

    Parameters
    ----------
    x1 : ndarray of shape (n, 2)
        Set of correspondences in the first image
    x2 : ndarray of shape (n, 2)
        Set of correspondences in the second image

    Returns
    -------
    E : ndarray of shape (3, 3)
        The essential matrix
    """
    

    n = x1.shape[0]

    A = np.zeros((n, 9))

    for i in range(n):
        x = x1[i][0]
        y = x1[i][1]
        xx = x2[i][0]
        yy = x2[i][1]
        A[i][0] = xx * x
        A[i][1] = xx * y
        A[i][2] = xx
        A[i][3] = yy * x
        A[i][4] = yy * y
        A[i][5] = yy
        A[i][6] = x
        A[i][7] = y
        A[i][8] = 1

    U, S, V = np.linalg.svd(A)
    E = V[-1].reshape(3, 3)

    U, S, V = np.linalg.svd(E)
    S = np.eye(3)
    S[2, 2] = 0
    E = U @ S @ V

    return E



def EstimateE_RANSAC(x1, x2, ransac_n_iter, ransac_thr):
    """
    Estimate the essential matrix robustly using RANSAC

    Parameters
    ----------
    x1 : ndarray of shape (n, 2)
        Set of correspondences in the first image
    x2 : ndarray of shape (n, 2)
        Set of correspondences in the second image
    ransac_n_iter : int
        Number of RANSAC iterations
    ransac_thr : float
        Error threshold for RANSAC

    Returns
    -------
    E : ndarray of shape (3, 3)
        The essential matrix
    inlier : ndarray of shape (k,)
        The inlier indices
    """

    E = None
    inlier = []
    
    for _ in range(ransac_n_iter):
        idx = np.random.choice(x1.shape[0], 8, replace=False)
        x1_ = x1[idx]
        x2_ = x2[idx]
        E_ = EstimateE(x1_, x2_)
        inlier_ = []

        err = np.zeros(x1.shape[0])
        for i in range(x1.shape[0]):
            x22 = np.array([x2[i][0], x2[i][1], 1])
            x11 = np.array([x1[i][0], x1[i][1], 1])
            err[i] = x22 @ E_ @ x11

        inlier_ = np.where(err < ransac_thr)

        if len(inlier_) > len(inlier):
            inlier = inlier_
            E = E_

    return E, inlier



def BuildFeatureTrack(Im, K):
    """
    Build feature track

    Parameters
    ----------
    Im : ndarray of shape (N, H, W, 3)
        Set of N images with height H and width W
    K : ndarray of shape (3, 3)
        Intrinsic parameters

    Returns
    -------
    track : ndarray of shape (N, F, 2)
        The feature tensor, where F is the number of total features
    """
    
    # TODO Your code goes here
    ransac_n_iter = 200
    ransac_thr = 0.01
    sift = cv2.xfeatures2d.SIFT_create()

    N = len(Im)

    track_ = []

    kp = []
    des = []

    for img in Im:
        kp1, des1 = sift.detectAndCompute(img, None)
        kp.append(kp1)
        des.append(des1)

    for i in range(N):
        F = len(kp[i])

        # track_i = -np.ones((N - i, F, 2))
        track_i = -np.ones((N, F, 2))
        for j in range(i+1, N):
            kp1 = kp[i]
            kp2 = kp[j]
            des1 = des[i]
            des2 = des[j]
            loc1 = np.array([kp.pt for kp in kp1])
            loc2 = np.array([kp.pt for kp in kp2])

            # Normalizing coordinates
            # Shape: (3, n)
            loc1 = np.array([loc1[:, 0], loc1[:, 1], np.ones(loc1.shape[0])])
            loc2 = np.array([loc2[:, 0], loc2[:, 1], np.ones(loc2.shape[0])])

            inv = np.linalg.inv(K)
            loc1 = inv @ loc1
            loc2 = inv @ loc2

            loc1 = loc1 / loc1[2]
            loc2 = loc2 / loc2[2]
            loc1 = loc1[:2]
            loc2 = loc2[:2]
            loc1 = loc1.T
            loc2 = loc2.T

            x1, x2, ind1 = MatchSIFT(loc1, des1, loc2, des2)
            _, inlier = EstimateE_RANSAC(x1, x2, ransac_n_iter, ransac_thr)

            # kp1 = np.array([kp.pt for kp in kp1])
            # kp2 = np.array([kp.pt for kp in kp2])

            track_i[i, inlier, :] = x1[inlier, :]
            track_i[j, inlier, :] = x2[inlier, :]


        # remove not matched points
        not_matched = []

        for i in range(F):
            if np.all(track_i[:, i, :] == -1):
                not_matched.append(i)

        track_i = np.delete(track_i, not_matched, 1)

        track_.append(track_i)

    F = 0

    idx_list = []
    for i in range(N):
        idx_list.append((F, F + track_[i].shape[1]))
        F = F + track_[i].shape[1]


    assert F == idx_list[-1][1]

    track = -np.ones((N, F, 2))

    # for i in range(N-1):
    #     start = idx_list[i][0]
    #     end = idx_list[i][1]
    #     track_i = track_[i]
    #     assert end - start == track_i.shape[1]
    #     for j in range(i, N-1):
    #         for k in range(start, end):
    #             track[j, k, :] = track_i[j - i, k - start, :]

    for i, track_i in enumerate(track_):
        start, end = idx_list[i]
        for j in range(N):
            for k in range(start, end):
                if track_i[j, k-start, 0] != -1 and track_i[j, k-start, 1] != -1:
                    track[j, k, :] = track_i[j, k-start, :]


    return track