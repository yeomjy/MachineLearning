import glob
from enum import Enum

from PIL import Image
from torch.utils.data import Dataset
from torch.utils.data.dataset import T_co
from torchvision.transforms.functional import to_tensor

from Transform import T_val


class Mode(Enum):
    TRAIN = 0
    TEST = 1
    VAL = 2


class SegmentationDataSet(Dataset):
    def __init__(self, transforms, mode=Mode.TRAIN):
        self.transforms = transforms
        if mode == Mode.TRAIN:
            img_f = open('./unet_data/train_image_list', 'r')
            mask_f = open('./unet_data/train_mask_list', 'r')
        elif mode == Mode.TEST:
            img_f = open('./unet_data/test_image_list', 'r')
            mask_f = open('./unet_data/test_mask_list', 'r')
        else:
            img_f = open('./unet_data/val_image_list', 'r')
            mask_f = open('./unet_data/val_mask_list', 'r')

        self.image_paths = [i.strip() for i in img_f.readlines()]
        self.target_paths = [i.strip() for i in mask_f.readlines()]
        self.len = len(self.image_paths)
        print(f'{len(self.image_paths)} images and {len(self.target_paths)} masks for {mode.name} Dataset')

        img_f.close()
        mask_f.close()

    def __len__(self):
        return self.len

    def __getitem__(self, index):
        image = Image.open(self.image_paths[index]).convert('L')
        mask = Image.open(self.target_paths[index])
        return self.transforms(image, mask)

    def set_transform(self, transform):
        self.transforms = transform


class TestSetWithOrigMask(Dataset):
    def __init__(self, people_idx=0):
        img_dir = glob.glob('./unet_data/image/1800_01_01*')
        mask_dir = glob.glob('./unet_data/masks/1800_01_01*')
        img_dir.sort()
        mask_dir.sort()
        self.img_dir = glob.glob(img_dir[people_idx] + '/*.jpg')
        self.mask_dir = glob.glob(mask_dir[people_idx] + '/*.png')
        self.img_dir.sort()
        self.mask_dir.sort()
        self.len = len(self.img_dir)
        self.transforms = T_val

    def __len__(self):
        return self.len

    def __getitem__(self, index):
        image = Image.open(self.img_dir[index]).convert('L')
        mask = Image.open(self.mask_dir[index])
        return self.transforms(image, mask)

class TestSet(Dataset):
    def __init__(self, path):
        self.list = glob.glob(str(path / '*.jpg'))
        self.len = len(self.list)

    def __len__(self):
        return self.len

    def __getitem__(self, index):
        image = self.list[index]
        image = Image.open(image).convert('L')
        return to_tensor(image)

