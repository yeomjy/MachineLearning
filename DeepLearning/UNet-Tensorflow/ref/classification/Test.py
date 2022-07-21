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

    model = models.resnet50(pretrained=False)
    for param in model.parameters():
        param.requires_grad = False # 사전 학습된 gradient를 유지합니다. 

    # from collections import OrderedDict
    # classifier = nn.Sequential(OrderedDict([ #모델의 마지막 부분을 다시 정의하여 학습합니다.
    #                       ('fc1', nn.Linear(2048, 3)), #Fully conneted layer부분인데 입력단은 원래 개수로 맞춰주시고 마지막단의 개수는 class의 개수로 맞춰주시면 됩니다.앞서 구조의 맨 마지막 부분인 (fc): Linear(in_features=2048, out_features=1000, bias=True)를 보시면 됩니다. 
    #                       ('output', nn.LogSoftmax(dim=1))
    #                       ]))
    # model.fc = classifier
    model.fc = nn.Linear(2048, 2)
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    model = model.to(device)
    model.load_state_dict(torch.load('liver_best_acc.pt')['model_state_dict'])


    test_set = CustomDataSet(transform=T_val, mode='Test', organ='liver')

    test_loader = DataLoader(test_set, batch_size=16, shuffle=False, num_workers=4)

    #train(model, device, train_loader, val_loader, 'kidney_trained.pt', 'kidney' ,epochs=2)
    tp = 0
    tn = 0
    fp = 0
    fn = 0
    with torch.no_grad():
        model.eval()

        for img, label in test_loader:
            img = img.to(device)
            output=model(img)
            pred = torch.argmax(output, dim=1)

            p = pred.cpu()
            tp += (p[label==1]==1).sum().item()
            fn += (p[label==1]==0).sum().item()
            tn += (p[label==0]==0).sum().item()
            fp += (p[label==0]==1).sum().item()
            

    with open('liver.txt', 'w') as f:
        f.write(f'True Positive: {tp}')
        f.write(f'True Negative: {tn}')
        f.write(f'False Positive: {fp}')
        f.write(f'False Negative: {fn}')

            


