import open3d as o3d
import cv2
import numpy as np
import yaml


model = o3d.io.read_triangle_mesh("/home/yeomjy/data/t-less_v2/models_cad/obj_01.ply")

path = "/home/yeomjy/data/t-less_v2/train_primesense/01"



with open(path + "/gt.yml") as f:
    gt = yaml.load(f, Loader=yaml.FullLoader)
with open(path + "/info.yml") as f:
    info = yaml.load(f, Loader=yaml.FullLoader)

pcd = model.sample_points_uniformly(50000)
img = cv2.imread(path + "/rgb/0000.png")
for idx in range(1296):


    R = gt[idx][0]['cam_R_m2c']
    t = gt[idx][0]['cam_t_m2c']
    K = info[idx]['cam_K']

    R = np.asarray(R)
    t = np.asarray(t)
    K = np.asarray(K)
    R = R.reshape((3, 3))
    K = K.reshape((3, 3))
    canvas = np.zeros((img.shape[0], img.shape[1], 1))


    Rt = np.concatenate([R, t.reshape((3, 1))], axis=1)
    P = K @ Rt
    points = np.asarray(pcd.points)

    for i in range(50000):
        p = np.concatenate([points[i], [1]])
        p = P @ p
        p = p / p[2]
        p = p[:2]
        x, y = int(p[0]), int(p[1])
        canvas[y, x] = 255

    cv2.imwrite(f"test_{idx}.png", canvas)

        



    

