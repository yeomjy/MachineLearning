import random

import numpy as np
import torch
from PIL import Image
from torchvision.transforms import functional as TF


class Compose(object):
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, image):
        for t in self.transforms:
            image = t(image)
        return image


class Resize(object):
    def __init__(self, size):
        self.size = size

    def __call__(self, image):
        image = TF.resize(image, self.size)
        return image


class RandomHorizontalFlip(object):
    def __init__(self, flip_prob):
        self.flip_prob = flip_prob

    def __call__(self, image):
        if random.random() < self.flip_prob:
            image = TF.hflip(image)
        return image


class RandomVerticalFlip(object):
    def __init__(self, flip_prob):
        self.flip_prob = flip_prob

    def __call__(self, image):
        if random.random() < self.flip_prob:
            image = TF.vflip(image)
        return image


class ToTensor(object):
    def __call__(self, image):
        image = TF.to_tensor(image)
        return image


T_train = Compose([Resize((512, 512)), RandomVerticalFlip(0.5), RandomHorizontalFlip(0.5), ToTensor()])
T_val = Compose([Resize((512, 512)), ToTensor()])
