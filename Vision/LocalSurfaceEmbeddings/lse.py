# LSE and LSE prediction
import glob
import cv2
import os
import time
import yaml
from pathlib import Path

import numpy as np
from numpy.linalg import norm, svd
from sklearn.neighbors import NearestNeighbors
from torchvision.transforms.functional import to_tensor
from nn_util import *
from PIL import Image
# from torchvision.models.segmentation import deeplabv3_resnet50 as deeplabv3

import torch
import torch.nn as nn
import open3d as o3d
# import open3d.cpu.pybind.geometry.PointCloud as PointCloud
# import open3d.cpu.pybind.geometry.TriangleMesh as TriangleMesh

from torch.utils.data import Dataset, DataLoader


def compute_lse(cad_points: np.ndarray, p: np.ndarray, normal: np.ndarray, r: float = 3.,
                sigma: float = 5.) -> np.ndarray:
    """
    cad_points: (n, 3) point cloud
    point: (3,) point
    normal (3,) normal vector of point

    """

    i_idx = [0, 2]
    j_idx = [0, 2]
    k_idx = [0, 1, 2]

    lse = np.zeros((11,))

    # cad_points: np.ndarray = np.asarray(cad_model.vertices)
    # p = cad_points[p_idx]

    # convert unit (r = 3cm, cad model uses mm)
    nbhd = cad_points[norm(cad_points - p, axis=1) < r]
    nbhd = nbhd - p

    n = nbhd.shape[0]

    C = np.zeros((3, 3))

    for i in range(n):
        C += np.outer(nbhd[i], nbhd[i])

    L, S, R = svd(C)

    # normal = cad_model.vertex_normals[p_idx]

    r1 = R[0]
    r3 = R[2] if R[2] @ normal > 0 else -R[2]
    r2 = np.cross(r1, r3)

    # Adjusted rotation matrix
    Rbar = np.array([r1, r2, r3])

    idx = 0
    for i in i_idx:
        for j in j_idx:
            for k in k_idx:
                if i == j == k == 0:
                    continue
                for l in range(n):
                    v = Rbar @ nbhd[l]
                    x, y, z = v
                    lse[idx] += np.exp(-norm(v) ** 2 / sigma ** 2) * (x ** i) * (y ** j) * (z ** k)
                idx += 1

    mean = lse.mean()
    std = lse.std()

    if not np.any(std == 0):
        lse = (lse - mean) / std
    else:
        lse = lse - mean

    return lse


# UNet-like architecture
class LSEPredictor(nn.Module):

    def __init__(self, c_in=3, num_classes=11):
        super(LSEPredictor, self).__init__()

        self.down1 = StackEncoder(c_in, 24, kernel_size=3)  # 128
        self.down2 = StackEncoder(24, 64, kernel_size=3)  # 64
        self.down3 = StackEncoder(64, 128, kernel_size=3)  # 32
        self.down4 = StackEncoder(128, 256, kernel_size=3)  # 16
        self.down5 = StackEncoder(256, 512, kernel_size=3)  # 8

        self.up5 = StackDecoder(512, 512, 256, kernel_size=3, output_padding=1)  # 16
        self.up4 = StackDecoder(256, 256, 128, kernel_size=3)  # 32
        self.up3 = StackDecoder(128, 128, 64, kernel_size=3)  # 64
        self.up2 = StackDecoder(64, 64, 24, kernel_size=3)  # 128
        self.up1 = StackDecoder(24, 24, 24, kernel_size=3)  # 256

        self.classify = nn.Conv2d(24, num_classes, kernel_size=1, bias=True)

        self.center = nn.Sequential(ConvBnRelu2d(512, 512, kernel_size=3, padding=1))

    def forward(self, x: torch.Tensor):
        # x: (batch_size, W, H, 3): Batch of RGB images
        # output: (batch_size, W, H, 11): LSE for each pixel
        out = x
        down1, out = self.down1(out)
        down2, out = self.down2(out)
        down3, out = self.down3(out)
        down4, out = self.down4(out)
        down5, out = self.down5(out)

        out = self.center(out)

        out = self.up5(out, down5)
        out = self.up4(out, down4)
        out = self.up3(out, down3)
        out = self.up2(out, down2)
        out = self.up1(out, down1)

        out = self.classify(out)

        return out


class LSEDataSet(Dataset):
    def __init__(self, mode="train"):
        base_dir = (Path.home() / "data" / "t-less_v2").resolve()
        #cameras = ["primesense", "kinect", "canon"]
        cameras = ["primesense"]
        self.img_list = []
        self.lse_list = []

        for camera in cameras:
            for obj_dir in (base_dir / f"train_{camera}").iterdir():
                img_list_ = [i for i in (obj_dir / "rgb").glob("*.png")]
                lse_list_ = [i for i in (obj_dir / "lse_embeddings").glob("*.npy")]
                img_list_.sort()
                lse_list_.sort()

                self.img_list += img_list_
                self.lse_list += lse_list_
        """
        exist = False
        for lse_path in self.lse_list:
            lse = np.load(str(lse_path))
            if np.any(np.isnan(lse)):
                exist = True


        if exist:
            raise ValueError()    
        else:
            print("Data File Okay")
        """
        self.len = len(self.img_list)
        self.transform = Compose([RandomHorizontalFlip(0.5), RandomVerticalFlip(0.5)])

    def __getitem__(self, index):
        #img = cv2.imread(str(self.img_list[index]), cv2.IMREAD_COLOR)
        img = Image.open(str(self.img_list[index]))
        lse = np.load(str(self.lse_list[index]))
        img = to_tensor(img)
        lse = torch.as_tensor(lse, dtype=torch.float32)
        lse = lse.reshape((11, 400, 400))

        return self.transform(img, lse)

    def __len__(self):
        return self.len


if __name__ == '__main__':
    # mode = "LSE_TRAINING_SET_GENERATE"
    # mode = "LSE"
    mode = "LSE_PREDICTOR_TRAIN"

    base_dir = (Path.home() / "data/t-less_v2").resolve()
    model_dir = (base_dir / "models_cad").resolve()
    save_dir = (base_dir / "lse").resolve()
    if mode == "LSE":
        if not save_dir.exists():
            save_dir.mkdir(parents=True)

        nan_list = ['06', '07', '08', '09', '23', '27', '29']
        models = [i for i in model_dir.glob("*.ply") if i.stem[-2:] in nan_list]

        #for filename in model_dir.glob("*.ply"):
        for filename in models:
            cad_model: o3d.geometry.TriangleMesh = o3d.io.read_triangle_mesh(str(filename))

            cad_model.compute_triangle_normals()
            n_sample = len(cad_model.triangle_normals) if len(cad_model.triangle_normals) > 10000 else 10000
            cad_model_pcd: o3d.geometry.PointCloud = cad_model.sample_points_uniformly(n_sample)
            cad_model_pcd.estimate_normals()
            cad_points = np.asarray(cad_model_pcd.points)

            lse_i = np.zeros((cad_points.shape[0], 15))
            checkpoint = time.time()
            for point_idx in range(cad_points.shape[0]):
                p = cad_points[point_idx]
                normal = cad_model_pcd.normals[point_idx]

                lse_i[point_idx][:3] = p
                lse_i[point_idx][3] = 1
                lse_i[point_idx][4:] = compute_lse(cad_points, p, normal)
                if point_idx % 100 == 0:
                    t = time.time()
                    print(f"{filename}: {point_idx}/{cad_points.shape[0]}, {t - checkpoint:.2f}sec")
                    checkpoint = time.time()

            save_name = filename.stem
            np.save(str(save_dir / save_name), lse_i)

    elif mode == "LSE_PREDICTOR_TRAIN":
        save_dir = Path("./lse_predictor")
        predictor = LSEPredictor()
        # predictor = deeplabv3(num_classes=11)
        predictor.train()
        optimizer = torch.optim.Adam(predictor.parameters(), lr=0.01)
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        predictor = predictor.to(device)
        
        if (save_dir / "lse_predictor.pt").exists():
            state_dict = torch.load(save_dir / "lse_predictor.pt")
            predictor.load_state_dict(state_dict["model_state_dict"])
            optimizer.load_state_dict(state_dict["optim_state_dict"])
            already_trained = state_dict["epoch"]
        else:
            already_trained = 0
        if not save_dir.exists(): 
            save_dir.mkdir(parents=True)

        loss_fn = nn.MSELoss()
        batch_size = 16
        n_epoch = 1
        dataset = LSEDataSet()
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        f = open("lse_training.txt", "a")

        for epoch in range(already_trained, n_epoch + already_trained):
            f.write(f"Epoch {epoch+1} start\n")
            f.flush()
            start = time.time()
            n = len(dataloader)
            for index, (image, mask) in enumerate(dataloader):
                image = image.to(device)
                mask = mask.to(device)
                optimizer.zero_grad()
                output = predictor(image)
                loss = loss_fn(output, mask)
                loss.backward()
                optimizer.step()
            end = time.time()
            f.write(f"Epoch {epoch+1}/{n_epoch+already_trained}: loss: {loss.item()}, time: {end-start:.05}sec\n")
            f.flush()

            state_dict = {
                "model_state_dict": predictor.state_dict(),
                "optim_state_dict": optimizer.state_dict()
            }
            torch.save(state_dict, save_dir / f"epoch_{epoch}.pt")

        state_dict = {
                "model_state_dict": predictor.state_dict(),
                "optim_state_dict": optimizer.state_dict(),
                "epoch": already_trained + n_epoch
        }
        torch.save(state_dict, save_dir / "lse_predictor.pt")
        f.close()

    elif mode == "LSE_TRAINING_SET_GENERATE":
        cameras = ["primesense"]
        nan_list = ['06', '07', '08', '09', '23', '27', '29']
        for camera in cameras:
            for obj_dir in (base_dir / f"train_{camera}").iterdir():
                # models = [i for i in model_dir.glob("*.ply") if i.stem[-2:] in nan_list]
                if obj_dir.name not in nan_list:
                    continue

                # load info and ground truth
                with open(obj_dir / "gt.yml") as f:
                    gt = yaml.load(f, Loader=yaml.FullLoader)
                with open(obj_dir / "info.yml") as f:
                    info = yaml.load(f, Loader=yaml.FullLoader)

                # load CAD model
                mesh_name = f"obj_{obj_dir.name}.ply"
                cad_model: o3d.geometry.TriangleMesh = o3d.io.read_triangle_mesh(str(model_dir / mesh_name))
                cad_model.compute_triangle_normals()

                # load CAD model pointcloud
                if (model_dir / "pointclouds" / f"{obj_dir.name}.ply").exists():
                    pcd = o3d.io.read_point_cloud(str(model_dir / "pointclouds" / f"{obj_dir.name}.ply"))
                else:
                    n_sample = len(cad_model.triangle_normals) if len(cad_model.triangle_normals) > 10000 else 10000
                    cad_model_pcd: o3d.geometry.PointCloud = cad_model.sample_points_uniformly(n_sample)
                    cad_model_pcd.estimate_normals()

                    if not (model_dir / "pointclouds").exists():
                        (model_dir / "pointclouds").mkdir(parents=True)

                    o3d.io.write_point_cloud(str(model_dir / "pointclouds" / f"{obj_dir.name}.ply"), cad_model_pcd)

                cad_model_t = o3d.t.geometry.TriangleMesh.from_legacy(cad_model)

                # lse_list: [x, y, z, 1, lse] : (n, 15)
                lse_list = np.load(str(save_dir / f"obj_{obj_dir.name}.npy"))
                lse_points = lse_list[:, :3]
                lse_values = lse_list[:, 4:]

                img_list = list((obj_dir / "rgb").glob("*.png"))
                img_list.sort()
                n_images = len(img_list)

                for i in range(n_images):
                    start = time.time()
                    # load image
                    img = cv2.imread(str(img_list[i]))

                    # load camera parameters
                    R = gt[i][0]['cam_R_m2c']
                    t = gt[i][0]['cam_t_m2c']
                    K = info[i]['cam_K']

                    R = np.asarray(R).reshape((3, 3))
                    t = np.asarray(t)
                    K = np.asarray(K).reshape((3, 3))
                    lse_embedding = np.zeros((img.shape[0], img.shape[1], 11))

                    H, W = img.shape[0], img.shape[1]

                    extrinsic = np.zeros((4, 4))
                    extrinsic[:3, :3] = R
                    extrinsic[:3, 3] = t
                    extrinsic[3, :3] = 0
                    extrinsic[3, 3] = 1

                    rays = o3d.t.geometry.RaycastingScene.create_rays_pinhole(
                        o3d.core.Tensor(K),
                        o3d.core.Tensor(extrinsic),
                        W,
                        H
                    )

                    # Cast ray to cad model
                    scene = o3d.t.geometry.RaycastingScene()
                    mesh_id = scene.add_triangles(cad_model_t)

                    cast = scene.cast_rays(rays)
                    nn = NearestNeighbors(n_neighbors=1).fit(lse_points)
                    point_list = []
                    coord_list = []

                    for y in range(H):
                        for x in range(W):
                            # non-zero pixel and hit
                            if np.any(img[y, x] != 0) and cast['t_hit'][y, x] != np.inf:
                                normal = cast['primitive_normals'][y, x].numpy()
                                point = rays[y, x][0:3] + cast['t_hit'][y, x] * rays[y, x][3:]
                                point = point.numpy()
                                point_list.append(point)
                                coord_list.append([y, x])

                    point_list = np.array(point_list)
                    coord_list = np.array(coord_list)
                    _, indices = nn.kneighbors(point_list)

                    lse_embedding[coord_list[:, 0], coord_list[:, 1]] = lse_values[indices].reshape((-1, 11))

                    end = time.time()
                    print(f'{camera}:{obj_dir}: {i}/{n_images}: {end - start}')

                    lse_save = obj_dir / "lse_embeddings"
                    if not lse_save.exists():
                        lse_save.mkdir(parents=True)

                    np.save(str(lse_save / f"lse_{obj_dir.name}_{i:04}"), lse_embedding)
