import torch
import numpy as np
import open3d as o3d
import yaml
import cv2
import time
from pathlib import Path

def generate_mask():
    train_dir = Path.home() / "data" / "t-less_v2" / "train_primesense"
    model_dir = Path.home() / "data" / "t-less_v2" / "models_cad"
    camera = "primesense"

    # for obj_dir in [Path.home() / "data" / "t-less_v2" / "train_primesense" / "01"]:
    for obj_dir in train_dir.iterdir():
        with open(obj_dir / "gt.yml", "r") as f:
            gt = yaml.load(f, Loader = yaml.FullLoader)
        with open(obj_dir / "info.yml", "r") as f:
            info = yaml.load(f, Loader = yaml.FullLoader)
        

        mesh_name = f"obj_{obj_dir.name}.ply"
        cad_model: o3d.geometry.TriangleMesh = o3d.io.read_triangle_mesh(str(model_dir / mesh_name))
        cad_model.compute_triangle_normals()

        cad_model_t = o3d.t.geometry.TriangleMesh.from_legacy(cad_model)


        img_list = list((obj_dir / "rgb").glob("*.png"))
        img_list.sort()
        n_images = len(img_list)

        for i in range(n_images):

            # for sample image
            #if i % 100 != 0:
            #    continue

            # start = time.time()
            # load image
            img = cv2.imread(str(img_list[i]))

            # load camera parameters
            R = gt[i][0]['cam_R_m2c']
            t = gt[i][0]['cam_t_m2c']
            K = info[i]['cam_K']

            R = np.asarray(R).reshape((3, 3))
            t = np.asarray(t)
            K = np.asarray(K).reshape((3, 3))

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
            canvas = np.zeros((1, H, W), dtype=bool)

            for y in range(H):
                for x in range(W):
                    # non-zero pixel and hit
                    canvas[0, y, x] = np.any(img[y, x] != 0) and cast['t_hit'][y, x] != np.inf


            # end = time.time()
            # print(f'{camera}:{obj_dir}: {i}/{n_images}: {end - start}')

            save_dir = obj_dir / "mask"
            if not save_dir.exists():
                save_dir.mkdir(parents=True)

            torch.save(torch.as_tensor(canvas), str(save_dir / f"mask_{obj_dir.name}_{i:04}.pt") )
            # write as image
            # cv2.imwrite(str(save_dir / f"mask_{obj_dir.name}_{i:04}.png"), canvas.reshape((H, W, 1)) * 255)
