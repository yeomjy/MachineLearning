# MaskRCNN Object Detection
from nn_util import *
import torch
import torch.nn as nn
import time
import yaml
from torch.utils.data import DataLoader, Dataset
from torchvision.models.detection import maskrcnn_resnet50_fpn
from pathlib import Path
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
import cv2
from torchvision.transforms import functional as F
from PIL import Image
from torch import utils


class MaskRCNNDataset(Dataset):
    def __init__(self):
        base_dir = (Path.home() / "data" / "t-less_v2").resolve()
        cameras = ["primesense"]
        self.img_list = []
        self.target_list = []
        self.mask_list = []

        for camera in cameras:
            for obj_dir in (base_dir / f"train_{camera}").iterdir():
                img_list_ = [i for i in (obj_dir / "rgb").glob("*.png")]
                img_list_.sort()
                with open(obj_dir / "gt.yml", "r") as f:
                    gt_ = yaml.load(f, Loader=yaml.FullLoader)

                target_list_ = []
                for i in range(len(gt_)):
                    box = gt_[i][0]["obj_bb"]
                    box[2] += box[0]
                    box[3] += box[1]
                    label = torch.ones(1, dtype=torch.int64)
                    label[0] = gt_[i][0]["obj_id"]
                    target_i = {
                        "boxes": torch.tensor(box),
                        "labels": label,
                    }
                    target_list_.append(target_i)
                
                mask_list_ = [i for i in (obj_dir / "mask").glob("*.pt")]
                mask_list_.sort()

                self.img_list += img_list_
                self.target_list += target_list_
                self.mask_list += mask_list_

        self.len = len(self.img_list)

    def __getitem__(self, index):

        img = Image.open(str(self.img_list[index]))
        img = F.to_tensor(img)
        target = self.target_list[index]

        # In training set, only one object exists
        # therefore, non-zero pixel is mask
        # mask = torch.any(img != 0, axis = 0)
        # mask = mask.unsqueeze(0)
        mask_path = self.mask_list[index]
        mask = torch.load(mask_path)
        return img, target["boxes"], target["labels"], mask

    def __len__(self):
        return self.len


if __name__ == '__main__':
    # training loop

    # If no mask
    from mask import generate_mask
    generate_mask()

    model = maskrcnn_resnet50_fpn(pretrained=True)
    num_classes = 31
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    model.to(device)
    model.train()
    dataset = MaskRCNNDataset()
    dataloader = DataLoader(dataset, batch_size=4, shuffle=True)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    n_epoch = 3
    save_dir = Path("./mask_rcnn")

    if not save_dir.exists():
        save_dir.mkdir(parents=True)

    if not (save_dir / "latest.pt").exists():
        already_trained = 0

    else:
        # load trained state
        state_dict = torch.load(save_dir / "latest.pt")
        already_trained = state_dict["epoch"]
        model.load_state_dict(state_dict["model_state_dict"])
        optimizer.load_state_dict(state_dict["optim_state_dict"])

    for epoch in range(already_trained, n_epoch + already_trained):
        for i, (img, boxes, labels, mask) in enumerate(dataloader):
            start = time.time()
            print(f"Epoch: {epoch}, Iteration: {i} Start")
            img = img.to(device)

            b = img.shape[0]
            targets = [
                {
                    "boxes": boxes[j].reshape((-1, 4)).to(device),
                    "labels": labels[j].to(device),
                    "masks": mask[j].to(device),
                }
                for j in range(b)
            ]

            optimizer.zero_grad()
            output = model(img, targets)
            loss = sum(loss for loss in output.values())
            loss.backward()
            end = time.time()
            print(f"Epoch: {epoch}, Iteration: {i}, Loss: {loss}, Time: {end - start}")

        state_dict = {
            "model_state_dict": model.state_dict(),
            "optim_state_dict": optimizer.state_dict()
        }

        torch.save(state_dict, save_dir / f"epoch_{epoch+1}.pt")

    # 마지막 상태 저장
    state_dict = {
        "model_state_dict": model.state_dict(),
        "optim_state_dict": optimizer.state_dict(),
        "epoch": n_epoch + already_trained
    }
    torch.save(state_dict, save_dir / "latest.pt")

    model.eval()
    n = len(dataset)
    i, j, k, l = np.random.randint(0, n, 4)
    img1 = Image.open(dataset.img_list[i])
    img2 = Image.open(dataset.img_list[j])
    img3 = Image.open(dataset.img_list[k])
    img4 = Image.open(dataset.img_list[l])
    img1 = F.to_tensor(img1)
    img2 = F.to_tensor(img2)
    img3 = F.to_tensor(img3)
    img4 = F.to_tensor(img4)

    mask1 = model(img1)[0]["masks"][0]
    mask2 = model(img2)[0]["masks"][0]
    mask3 = model(img3)[0]["masks"][0]
    mask4 = model(img4)[0]["masks"][0]

    mask1 = mask1.detach().cpu().numpy()
    mask2 = mask2.detach().cpu().numpy()
    mask3 = mask3.detach().cpu().numpy()
    mask4 = mask4.detach().cpu().numpy()

    cv2.imwrite("mask1.png", mask1)
    cv2.imwrite("mask2.png", mask2)
    cv2.imwrite("mask3.png", mask3)
    cv2.imwrite("mask4.png", mask4)
