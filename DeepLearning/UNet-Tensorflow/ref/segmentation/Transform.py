import random

import numpy as np
import torch
from PIL import Image
from torchvision.transforms import functional as TF


class Compose(object):
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, image, target):
        for t in self.transforms:
            image, target = t(image, target)
        return image, target


class Resize(object):
    def __init__(self, size):
        self.size = size

    def __call__(self, image, target):
        image = TF.resize(image, self.size)
        target = TF.resize(target, self.size, interpolation=TF.InterpolationMode.NEAREST)
        return image, target


class RandomHorizontalFlip(object):
    def __init__(self, flip_prob):
        self.flip_prob = flip_prob

    def __call__(self, image, target):
        if random.random() < self.flip_prob:
            image = TF.hflip(image)
            target = TF.hflip(target)
        return image, target


class RandomVerticalFlip(object):
    def __init__(self, flip_prob):
        self.flip_prob = flip_prob

    def __call__(self, image, target):
        if random.random() < self.flip_prob:
            image = TF.vflip(image)
            target = TF.vflip(target)
        return image, target


class ToTensor(object):
    def __call__(self, image, target):
        image = TF.to_tensor(image)
        # name = target.filename
        target = torch.as_tensor(np.array(target), dtype=torch.int64)
        target[target == 255] = 0
        # if len(target.shape) > 2:
        #     raise ValueError(target.shape)

        return image, target


T_train = Compose([Resize((512, 512)), RandomVerticalFlip(0.5), RandomHorizontalFlip(0.5), ToTensor()])
T_val = Compose([Resize((512, 512)), ToTensor()])
