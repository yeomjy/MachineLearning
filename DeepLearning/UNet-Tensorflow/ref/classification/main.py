import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from CustomDataSet import CustomDataSet
# from Test import Test
from Trainer import train
from Transform import T_train, T_val
import torchvision.models as models
import torch.nn as nn

if __name__ == '__main__':

    model = models.resnet50(pretrained=True)
    for param in model.parameters():
        param.requires_grad = False # 사전 학습된 gradient를 유지합니다. 

    # from collections import OrderedDict
    # classifier = nn.Sequential(OrderedDict([ #모델의 마지막 부분을 다시 정의하여 학습합니다.
    #                       ('fc1', nn.Linear(2048, 3)), #Fully conneted layer부분인데 입력단은 원래 개수로 맞춰주시고 마지막단의 개수는 class의 개수로 맞춰주시면 됩니다.앞서 구조의 맨 마지막 부분인 (fc): Linear(in_features=2048, out_features=1000, bias=True)를 보시면 됩니다. 
    #                       ('output', nn.LogSoftmax(dim=1))
    #                       ]))
    # model.fc = classifier
    model.fc = nn.Linear(2048, 2)


    train_set = CustomDataSet(transform=T_train, mode='Train', organ='kidney')
    val_set = CustomDataSet(transform=T_val, mode='Valid', organ='kidney')

    train_loader = DataLoader(train_set, batch_size=16, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_set, batch_size=16, shuffle=True, num_workers=4)
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

    train(model, device, train_loader, val_loader, 'kidney_trained.pt', 'kidney' ,epochs=2)


