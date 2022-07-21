from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from CustomDataSet import TestSetWithOrigMask

COLORMAP = [
    [0, 0, 0],
    [255, 0, 0],
    [0, 255, 0],
    [255, 255, 0]
]


def mask_to_colormap(mask):
    mask_shape = mask.shape
    colormap = torch.empty((mask_shape[0], mask_shape[1], 3))
    for i in range(mask_shape[0]):
        for j in range(mask_shape[1]):
            colormap[i][j] = torch.Tensor(COLORMAP[mask[i][j]])
    return colormap.type(torch.LongTensor)


# def batch_iou(pred, mask, cl, device):
#     union = ((pred == cl) + (mask == cl)).sum(dim=(1, 2))
#     inter = ((pred == cl) * (mask == cl)).sum(dim=(1, 2))
#     iou = torch.full(union.size(), -1.0).to(device)
#     iou[union > 0] = (inter[union > 0]) / union[union > 0]
#     return iou


def save_figure(model, idx, device):
    testset = TestSetWithOrigMask(idx)
    test_loader = DataLoader(testset, shuffle=False, batch_size=8)
    path = Path() / 'test_result' / f'{idx}'
    if not path.exists():
        path.mkdir(parents=True)
    with torch.no_grad():
        import matplotlib.pyplot as plt
        model.eval()
        for idx, (img, mask) in enumerate(test_loader):
            batch_size = img.size()[0]
            img, mask = img.to(device), mask.to(device)
            output = model(img)
            output = torch.argmax(output, dim=1)
            fig, axs = plt.subplots(4, 6)
            fig.set_size_inches(12, 8)
            for i, ax in enumerate(axs.ravel()):
                ax.grid(False)
                ax.set_xticks([])
                ax.set_yticks([])

                if i // 3 >= batch_size:
                    ax.imshow(np.zeros((512, 512, 1)), cmap='gray')
                    continue

                if i % 3 == 0:
                    ax.set_title("Image", fontsize=10)
                    ax.imshow(img[i // 3].permute(1, 2, 0).detach().cpu(), cmap='gray')
                if i % 3 == 1:
                    ax.set_title("True Colormap", fontsize=10)
                    ax.imshow(mask_to_colormap(mask[i // 3]))
                if i % 3 == 2:
                    colormap = mask_to_colormap(output[i // 3])
                    ax.set_title("Pred Colormap", fontsize=10)
                    ax.imshow(colormap)
            plt.savefig(path / f'{idx + 1}th_result')


def calculate_iou(model, device, test_loader):
    with torch.no_grad():
        model.eval()
        iou_tensors = []
        dice_tensors = []
        for cl in range(0, 4):
            iou_tensors.append(torch.tensor([], device=device))
            dice_tensors.append(torch.tensor([], device=device))
        for img, mask in test_loader:
            img, mask = img.to(device), mask.to(device)
            output = model(img)
            pred = torch.argmax(output, dim=1)
            for cl in range(0, 4):
                # iou_cl = batch_iou(pred, mask, cl, device)
                # iou_cl = iou_cl[iou_cl >= 0]
                union = ((pred == cl) + (mask == cl)).sum(dim=(1, 2))
                inter = ((pred == cl) * (mask == cl)).sum(dim=(1, 2))
                sumel = (pred == cl).sum(dim=(1, 2)) + (mask == cl).sum(dim=(1, 2))

                iou = torch.full(union.size(), -1.0).to(device)
                iou[union > 0] = (inter[union > 0]) / union[union > 0]
                iou = iou[iou >= 0]
                iou_tensors[cl] = torch.cat((iou_tensors[cl], iou))

                dice = torch.full(sumel.size(), -1.0).to(device)
                dice[sumel > 0] = (2 * inter[sumel > 0]) / sumel[sumel > 0]
                dice = dice[dice >= 0]
                dice_tensors[cl] = torch.cat((dice_tensors[cl], dice))

        with open('test_result/statistics.txt', 'a') as f:
            for cl in range(0, 4):
                iou_cl = iou_tensors[cl].cpu()
                average = iou_cl.mean().item()
                std = iou_cl.std().item()
                max = iou_cl.max().item()
                min = iou_cl.min().item()
                print(f'Class {cl + 1} IOU Statistics: {average=}, {std=}, {max=}, {min=}', file=f)
                torch.save(iou_cl, f'./test_result/iou_{cl}.pt')

                dice_cl = dice_tensors[cl].cpu()
                average = dice_cl.mean().item()
                std = dice_cl.std().item()
                max = dice_cl.max().item()
                min = dice_cl.min().item()
                print(f'Class {cl + 1} DICE Statistics: {average=}, {std=}, {max=}, {min=}', file=f)
                torch.save(dice_cl, f'./test_result/dice_{cl}.pt')


def Test(model, path, test_loader, device, idx=-1, mode='all'):
    model = model.to(device)
    loaded = torch.load(path)
    model.load_state_dict(loaded['model_state_dict'])

    if mode == 'all':
        if idx == -1:
            idx = np.random.randint(low=0, high=44)
        save_figure(model, idx, device)
        calculate_iou(model, device, test_loader)
    elif mode == 'iou':
        calculate_iou(model, device, test_loader)
    elif mode == 'person':
        if idx == -1:
            idx = np.random.randint(low=0, high=44)
        save_figure(model, idx, device)

# with torch.no_grad():
#     import matplotlib.pyplot as plt
#     model.eval()
#     img, mask = next(iter(test_loader))
#     img, mask = img.to(device), mask.to(device)
#     output = model(img)
#     output = torch.argmax(output, dim=1)
#     fig, axs = plt.subplots(4, 3)
#     fig.set_size_inches(6, 8)
#     for i, ax in enumerate(axs.ravel()):
#         ax.grid(False)
#         ax.set_xticks([])
#         ax.set_yticks([])
#
#         if i % 3 == 0:
#             ax.set_title("Image", fontsize=10)
#             ax.imshow(img[i // 3].permute(1, 2, 0).detach().cpu(), cmap='gray')
#         if i % 3 == 1:
#             ax.set_title("True Colormap", fontsize=10)
#             ax.imshow(mask_to_colormap(mask[i // 3]))
#         if i % 3 == 2:
#             colormap = mask_to_colormap(output[i // 3])
#             ax.set_title("Pred Colormap", fontsize=10)
#             ax.imshow(colormap)
#     plt.show()
