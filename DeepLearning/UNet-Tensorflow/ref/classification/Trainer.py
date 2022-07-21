import time
from pathlib import Path
import sys

import torch.optim


def train(model, device, train_loader, val_loader, path, organ, epochs=10):
    model = model.to(device)
    # criterion = torch.nn.NLLLoss()
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
    # f = sys.stdout
    print(f'Model for {organ} training start...', file=f)
    start = time.time()
    for epoch in range(already_trained, already_trained + epochs):
        ep_start = time.time()
        batch_loss_list = []
        model.train()
        for img, label in train_loader:
            optimizer.zero_grad()
            img, label = img.to(device), label.to(device)
            output = model(img)

            loss = criterion(output, label)
            loss.backward()
            optimizer.step()
            batch_loss_list.append(loss.item())

        print('Epoch {:4d}/{} Loss {:.6f}'.format(epoch + 1, epochs + already_trained,
                                                  sum(batch_loss_list) / len(batch_loss_list)), file=f)

        torch.save({
            'model_state_dict': model.state_dict(),
            'optim_state_dict': optimizer.state_dict(),
            'epoch'           : epoch + 1
        }, f'trained/{organ}/epoch_{epoch + 1}.pt')

        model.eval()
        batch_loss_list = []

        correct = 0
        count = 0
        with torch.no_grad():
            for img, label in val_loader:
                img, label = img.to(device), label.to(device)
                output = model(img)
                loss = criterion(output, label)
                batch_loss_list.append(loss.item())

                pred = torch.argmax(output, dim=1)
                correct += (pred == label).sum().item()
                count += label.size()[0]

            accuracy = correct / count * 100
            print('Epoch {:4d}/{} Val Loss {:.6f}'.format(epoch + 1, epochs + already_trained,
                                                          sum(batch_loss_list) / len(batch_loss_list)),file=f)
            print('Epoch {:4d}/{} Val Accuracy {}%'.format(epoch + 1, epochs + already_trained, accuracy), file=f)

            if accuracy > best_accuracy:
                print('Saving Model')
                best_accuracy = accuracy
                torch.save({
                    'model_state_dict': model.state_dict(),
                    'optim_state_dict': optimizer.state_dict(),
                    'epoch'           : epoch + 1
                }, f'{organ}_best_acc.pt')
        ep_end = time.time()
        print('Training Time for Epoch {:4d}: {:.4f}\n'.format(epoch + 1, ep_end - ep_start), file=f)
    end = time.time()
    print('Total Training Time: {:.4f}'.format(end - start), file=f)
    print(f'Model for {organ} training finised...', file=f)
    torch.save({
        'model_state_dict': model.state_dict(),
        'optim_state_dict': optimizer.state_dict(),
        'epoch'           : epochs
    }, f'{organ}_trained.pt')
    f.close()
