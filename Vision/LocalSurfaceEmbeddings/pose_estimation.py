import open3d as o3d
import numpy as np
import cv2
from pathlib import Path
from sklearn.neighbors import NearestNeighbors
from lse import compute_lse
import yaml


def mask_to_pts(O):
    # input : mask - boolean array with shape (H,W)
    # output : points - matrix with shape (n,2)
    # return (x,y) coord of True in mask array
    return np.asarray(np.where(O)).T

def reprojection(K, R, t, H, W, points, obj_name, img_name):
    canvas = np.zeros((H, W, 1), dtype=np.uint8)
    P = np.zeros((3, 4))
    P[:3, :3] = R
    P[:3, 3] = t.squeeze()
    P = K @ P


    print(f"{obj_name}_{img_name} reprojection start", flush=True)
    for i in range(len(points)):
        p = np.array([points[i, 0], points[i, 1], points[i, 2], 1])
        pp = P @ p
        pp = pp / pp[2]
        pp = pp[:2]
        x, y= pp
        x, y = int(x), int(y)
        if 0 <= y < H and 0 <= x < W:
            canvas[y, x] = 255
        

    cv2.imwrite(f"{obj_name}_{img_name}_reprojection.png", canvas)
    print(f"{obj_name}_{img_name} reprojected", flush=True)

def score_estimation(rvec, tvec, O, C, lse_O, lse_C, K, weight=[0.97, 0.03]):
    # rvec, tvec : rotation and translation vector (model coord to camera frame), (3,) each
    # n : number of points in pointcloud ; m : dimension of lse
    # C : pointcloud, (n, 3)
    # lse_O : predicted lse value in each pixel of the image, (H, W, m)
    # lse_C : known lse of CAD model, (n, m)
    # K : camera intrinsic, (3, 3)

    H, W, _ = lse_O.shape
    # threshold = 0.1  # TODO : threshold value optimizing
    lse_projection = np.zeros((H, W, 11))


    # projection of the 3d points in image space
    projection, _ = cv2.projectPoints(C, rvec, tvec, K, 0)
    # projection = np.around(np.asarray(projection))
    projection = projection.squeeze()

    # filter points out of the image
    in_the_image = (0 <= projection[:, 0]) * (projection[:, 0] < W) * (0 <= projection[:, 1]) * (projection[:, 1] < H)
    # projection_short = projection[in_the_image]
    # lse_C_short = lse_C[in_the_image]
    projection = projection[in_the_image].astype(np.uint8)
    projection_mask = np.zeros((H, W))
    projection_mask[projection[:, 1], projection[:, 0]] = 1
    lse_projection[projection[:, 1], projection[:, 0]] = lse_C[in_the_image]

    inter = np.sum((projection_mask == 1) * (O == 1))
    union = np.sum((projection_mask == 1) + (O == 1))
    iou =  inter / union

    lse_err = np.linalg.norm(lse_O - lse_projection)
    print(f"IOU:{iou}, lse_err: {lse_err}", flush=True)
    score = weight[0] * iou + weight[1] * -lse_err
    return score




    # calculate error and count inliers
    # error = np.sum(np.absolute(lse_O[projection_short[:, 0], projection_short[:, 1]] - lse_C_short), axis=1)
    # score = np.sum(error < threshold)
    # return score


def pose_refinement(O, C, rvec, tvec):
    rvec_refined = rvec  # TODO
    tvec_refined = tvec
    return rvec_refined, tvec_refined


def pose_estimation_O_C(O, C, lse_O, lse_C, K):
    # O : a boolean mask (H, W)
    # C : a CAD model (pointcloud) (n, 3)
    # lse_O : predicted lse value in each pixel of the image, (H, W, m)
    # lse_C : known lse of CAD model, (n, m), m is dimension of lse
    # K: Camera Matrix

    # Initialize variables
    # K = np.eye(3)  # TODO : check camera matrix of t-less dataset
    print("Pose Estimation O_C started", flush=True)

    img_pts = mask_to_pts(O)
    cad_pts = np.asarray(C)

    ransac_iter = 100

    n_img_pts = len(img_pts)
    # n_cad_pts = len(cad_pts)

    rvec_best = None
    tvec_best = None
    score_best = -np.inf
    n = 6
    nn = NearestNeighbors(n_neighbors=n).fit(lse_C)
    # _, matches  = nn.kneighbors(lse_O[O])

    for i in range(ransac_iter):
        # randomly select n points
        # img_ind = np.random.choice(n_img_pts, size=n, replace=False)  # replace=False == 비복원추출
        # cad_ind = np.random.choice(n_cad_pts, size=n, replace=False)
        # img_idx = np.random.choice(n_img_pts, size=n, replace=False)
        # idx_ = nn.kneighbors(lse_O[img_idx])
        print(f"Pose estimation RANSAC: {i+1}th loop", flush=True)
        # n_pts = matches.shape[0]
        n_pts = lse_O[O].shape[0]
        img_idx = np.random.choice(n_pts, size=n, replace=False)
        _, matches = nn.kneighbors(lse_O[O][img_idx])
        # idx_ = matches[img_idx]

        cad_idx = []
        for match_idx in range(matches.shape[0]):
            m = matches[match_idx]
            for ind in range(len(m)):
                if m[ind] not in cad_idx:
                    cad_idx.append(m[ind])
                    break
            # for k in range(len(j)):
            #     if j[k] not in cad_idx:
            #         cad_idx.append(j[k])

        cad_idx = np.array(cad_idx)

        img_coord = img_pts[img_idx, :].astype(np.float32)
        img_coord = np.array([img_coord[:, 1], img_coord[:, 0]]).T # (y, x) -> (x, y)
        print(img_coord.shape)
        print(cad_pts[cad_idx, :].shape)
        # use pnp algorithm to estimate pose
        success, rvec, tvec = cv2.solvePnP(cad_pts[cad_idx, :], img_coord, K, 0,
        flags=cv2.SOLVEPNP_EPNP)

        if success:
            # calculate score using lse
            score = score_estimation(rvec, tvec, O, C, lse_O, lse_C, K)

            # update the best result
            if score >= score_best:
                rvec_best = rvec
                tvec_best = tvec
                score_best = score

    #rvec, tvec = pose_refinement(O, C, rvec_best, tvec_best)
    #score = score_estimation(rvec, tvec, O, C, lse_O, lse_C, K)

    return rvec_best, tvec_best, score_best


def pose_estimation(masks, models):
    pass

if __name__ == "__main__":
    base_dir = Path.home() / "data" / "t-less_v2" 
    np.random.seed(0)
    pcd_dir = Path("./lse_pcd")
    if not pcd_dir.exists():
        pcd_dir.mkdir()

    for obj_dir in [base_dir / "train_primesense" / "01"]:
        # lse_C = np.load(base_dir  / "lse" / f"obj_{obj_dir.name}.npy")
        # points = lse_C[:, :3]
        # lse_C = lse_C[:, 4:]
        # print(points.shape)
        # print(lse_C.shape)
        if (pcd_dir / f"{obj_dir.name}_lse.npy").exists():
            lse_C = np.load(pcd_dir / f"{obj_dir.name}_lse.npy")
            points = np.load(pcd_dir / f"{obj_dir.name}_pcd.npy")
            print(f"Model {obj_dir.name}_ loaded", flush=True)
        else:
            cad_model = o3d.io.read_triangle_mesh(str(base_dir / "models_cad" / f"obj_{id:02}.ply"))
            cad_model_t = o3d.t.geometry.TriangleMesh.from_legacy(cad_model)
            cad_model.compute_triangle_normals()
            n_sample = len(cad_model.triangle_normals) if len(cad_model.triangle_normals) > 10000 else 10000
            pcd = cad_model.sample_points_uniformly(n_sample)
            pcd.estimate_normals()
            lse_C = np.zeros((n_sample, 11))
            points = np.asarray(pcd.points)
            normals = np.asarray(pcd.normals)
            print(f"Model {obj_dir.name} computing started: {n_sample}", flush=True)
            for j in range(n_sample):
                p = points[j]
                n = normals[j]
                lse_C[j] = compute_lse(points, p, n)
            print(f"Model {obj_dir.name} computed", flush=True)
        
            np.save(pcd_dir / f"{obj_dir.name}_lse.npy", lse_C)
            np.save(pcd_dir / f"{obj_dir.name}_pcd.npy", points)
        with open(obj_dir / "gt.yml") as f:
            gt = yaml.load(f, Loader=yaml.FullLoader)
        with open(obj_dir / "info.yml") as f:
            info = yaml.load(f, Loader=yaml.FullLoader)

        img_list = sorted(list((obj_dir / "rgb").glob("*.png")))
        n_img = len(img_list)

        imgs = np.random.choice(n_img, 5, replace=False)
        # for i in range(n_img):
        for i in imgs:
            objs = gt[i]
            img = cv2.imread(str(img_list[i]))
            
            for obj in objs:
                id = int(obj['obj_id'])
                R = np.asarray(obj['cam_R_m2c']).reshape((3, 3))
                t = np.asarray(obj['cam_t_m2c'])
                K = np.asarray(info[i]['cam_K']).reshape((3, 3))
                lse_O = np.load(obj_dir / "lse_embeddings" / f"lse_{id:02}_{i:04}.npy")
                H, W, _ = lse_O.shape
                # mask = np.zeros((H, W))
                mask = np.any(lse_O != 0, axis=2)

                rvec, tvec, score = pose_estimation_O_C(mask, points, lse_O, lse_C, K)
                print(rvec)
                R_estimated, _ = cv2.Rodrigues(rvec)

                reprojection(K, R_estimated, tvec, H, W, points, obj_dir.name, f"{i:04}")
                

