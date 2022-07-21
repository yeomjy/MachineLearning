from CustomDataSet import TestSet
import numpy as np
from pathlib import Path
from torch.utils.data import DataLoader
from UNet import UNet
from Test import mask_to_colormap
from torchvision.transforms.functional import to_pil_image
import torch
import glob
def make_mask(path):
    class_map = {1: 'Liver', 2:'Spleen', 3:'Kidney'}
    base_dir = Path()
    device = torch.device('cuda:0')
    model = UNet().to(device)
    loaded = torch.load(base_dir / 'best_acc.pt')
    model.load_state_dict(loaded['model_state_dict'])
    #dataset = TestSetWithOrigMask(people_idx=idx)
    dataset = TestSet(path)
    dataloader = DataLoader(dataset, shuffle=False, batch_size=24, num_workers=4)

    result_dir = path / 'generated'
    if not result_dir.exists():
        result_dir.mkdir(exist_ok=True, parents=True)

    if not (result_dir / 'Liver').exists():
        (result_dir / 'Liver').mkdir(parents=True)
    if not (result_dir / 'Spleen').exists():
        (result_dir / 'Spleen').mkdir(parents=True)
    if not (result_dir / 'Kidney').exists():
        (result_dir / 'Kidney').mkdir(parents=True)

    num = 1
    with torch.no_grad():
        model.eval()
        for img in dataloader:
            b = img.shape[0]
            img = img.to(device)
            output = model(img)
            pred = torch.argmax(output, dim=1)
            for cl in range(1, 4):
                cloned_img = img.clone().detach()
                cloned_img[(pred!=cl).unsqueeze(1)] = 0
                for i in range(b):
                    if cloned_img[i].sum() > 0:
                        class_name = class_map[cl]
                        save_img = to_pil_image(cloned_img[i])
                        save_img.save(result_dir / class_name / f'{num}.png')
                        num += 1
                    



if __name__ == '__main__':
    
    # data = data/Test/Liver_cancer/*.jpg
    dir_list = glob.glob('/csehome/wnsdud/classification/data/*/*')
    for i in dir_list:
        make_mask(Path(i))
