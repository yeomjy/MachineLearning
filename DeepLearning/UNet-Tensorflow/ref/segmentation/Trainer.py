import time
from pathlib import Path

import torch.optim


def train(model, device, train_loader, val_loader, path, epochs=10):
    model = model.to(device)
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    model_path = Path(path)
    already_trained = 0
    if model_path.exists():
        loaded = torch.load(model_path)
        model.load_state_dict(loaded['model_state_dict'])
        optimizer.load_state_dict(loaded['optim_state_dict'])
        already_trained = loaded['epoch']

    best_accuracy = 0
    # loss_list = []
    f = open('result.txt', 'a')
    start = time.time()
    for epoch in range(already_trained, already_trained + epochs):
        ep_start = time.time()
        batch_loss_list = []
        model.train()
        for img, mask in train_loader:
            img, mask = img.to(device), mask.to(device)
            output = model(img)

            optimizer.zero_grad()
            loss = criterion(output, mask)
            loss.backward()
            optimizer.step()
            batch_loss_list.append(loss.item())

        print('Epoch {:4d}/{} Loss {:.6f}'.format(epoch + 1, epochs + already_trained,
                                                  sum(batch_loss_list) / len(batch_loss_list)), file=f)

        torch.save({
            'model_state_dict': model.state_dict(),
            'optim_state_dict': optimizer.state_dict(),
            'epoch'           : epoch + 1
        }, f'trained/epoch_{epoch + 1}.pt')

        model.eval()
        batch_loss_list = []

        correct = 0
        count = 0
        with torch.no_grad():
            #    average_iou = torch.zeros(3).to(device)
            #    iou_nums = torch.zeros(3).to(device)
            for img, mask in val_loader:
                img, mask = img.to(device), mask.to(device)
                output = model(img)
                loss = criterion(output, mask)
                batch_loss_list.append(loss.item())

                pred = torch.argmax(output, dim=1)
                correct += (pred == mask).sum()
                count += mask.numel()

            accuracy = correct / count * 100
            print('Epoch {:4d}/{} Val Loss {:.6f}'.format(epoch + 1, epochs + already_trained,
                                                          sum(batch_loss_list) / len(batch_loss_list)),file=f)
            print('Epoch {:4d}/{} Val Pixelwise Accuracy {}%'.format(epoch + 1, epochs + already_trained, accuracy), file=f)

            if accuracy > best_accuracy:
                print('Saving Model')
                best_accuracy = accuracy
                torch.save({
                    'model_state_dict': model.state_dict(),
                    'optim_state_dict': optimizer.state_dict(),
                    'epoch'           : epoch + 1
                }, 'best_acc.pt')
        ep_end = time.time()
        print('Training Time for Epoch {:4d}: {:.4f}\n'.format(epoch + 1, ep_end - ep_start), file=f)
    end = time.time()
    print('Total Training Time: {:.4f}'.format(end - start), file=f)
    torch.save({
        'model_state_dict': model.state_dict(),
        'optim_state_dict': optimizer.state_dict(),
        'epoch'           : epochs
    }, 'unet_trained.pt')
    f.close()
