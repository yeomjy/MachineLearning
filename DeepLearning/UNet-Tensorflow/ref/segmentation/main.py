import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from CustomDataSet import Mode, SegmentationDataSet, TestSetWithOrigMask
from Test import Test
from Trainer import train
from Transform import T_train, T_val
from UNet import UNet

if __name__ == '__main__':

    if len(sys.argv) == 1:
        print("options:\n"
              "\t--train\n"
              "\t--train epoch={# of epoch}\n"
              "\t\t if you don't specify epoch, default value is 10\n"
              "\t--test\n"
              "\t--test all|iou|person={index}\n"
              "\t\t all           : do iou test and person test for a random index\n"
              "\t\t iou           : calculate average iou for test set\n"
              "\t\t person={index}: test whole images of person {index} and save predicted mask to image\n"
              "\t\t if you don't specify test mode, default value is all\n"
              "\t--file {path_to_file}\n"
              "\t\t to specify trained model file (*.pt)\n\n"
              "\t You should select at least one mode")
        exit(0)

    epoch = 10
    person_idx = -1
    test_on = False
    train_on = False
    test_mode = 'all'
    path = './best_acc.pt'

    for idx, arg in enumerate(sys.argv):
        if arg.startswith('--train'):
            train_on = True
            if idx + 1 < len(sys.argv):
                next_arg = sys.argv[idx + 1]
                if 'epoch' in next_arg:
                    epoch_string = next_arg.split('=')[-1]
                    try:
                        epoch = int(epoch_string)
                    except ValueError:
                        print(f'Epoch value is not Integer: {epoch_string}')
                        exit(1)

        elif arg.startswith('--test'):
            test_on = True
            if idx + 1 < len(sys.argv):
                next_arg = sys.argv[idx + 1]
                if next_arg == 'iou':
                    test_mode = 'iou'
                elif 'person' in next_arg:
                    person_idx_str = next_arg.split('=')[-1]
                    test_mode = 'person'
                    try:
                        person_idx = int(person_idx_str)
                    except ValueError:
                        print(f'Person index is not Integer: {person_idx_str}')
                elif next_arg == 'all':
                    test_mode = 'all'
        elif arg.startswith('--file'):
            if idx + 1 >= len(sys.argv):
                print(f'Error: no path')
                exit(1)
            elif not Path(sys.argv[idx + 1]).exists():
                print(f'Error: {Path(sys.argv[idx + 1])} not exists')
                exit(1)
            else:
                path = Path(sys.argv[idx + 1])
        else:
            if 'epoch' in arg or 'person' in arg or arg == 'iou' or arg == 'all' \
                    or Path(arg).exists():
                continue
            else:
                print(f'Invalid Argument: {arg}')
                exit(1)

    train_set = SegmentationDataSet(transforms=T_train, mode=Mode.TRAIN)
    test_set = SegmentationDataSet(transforms=T_val, mode=Mode.TEST)
    val_set = SegmentationDataSet(transforms=T_val, mode=Mode.VAL)

    train_loader = DataLoader(train_set, batch_size=10, shuffle=True)
    test_loader = DataLoader(test_set, batch_size=10, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=10, shuffle=True)
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

    model = UNet()

    if train_on:
        train(model, device, train_loader, val_loader, path, epochs=epoch)

    elif test_on:
        Test(model, path, test_loader, device, person_idx, mode=test_mode)
