import glob
from enum import Enum

from PIL import Image
from torch.utils.data import Dataset
from torch.utils.data.dataset import T_co
from pathlib import Path

from Transform import T_val


class CustomDataSet(Dataset):

    def __init__(self, mode, organ, transform):
        path = Path('data') / f'{mode}_{organ}.txt'
        self.transform = transform

        self.img_list = list()
        self.label_list = list()
        
        with open(path, 'r') as f:
            for i in f.readlines():
                img, label = i.strip().split(' ')
                self.img_list.append(img)
                label = 0 if label == 'normal' else 1
                self.label_list.append(label)

    def __len__(self):
        return len(self.img_list)

    def __getitem__(self, index):
        image = Image.open(self.img_list[index]).convert('RGB')
        label = self.label_list[index]
        return self.transform(image), label
