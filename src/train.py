import torch
import torch.nn as nn
import torch.optim as optim
import dataloader
from torch.utils.data import DataLoader
import numpy as np
from sklearn.metrics import accuracy_score
from torchvision import transforms

from finetuned_alexnet import FinetunedAlexNet

start_unfreeze_epoch = 7

def train(data_dir="../data", save_dir="../save/", batch_size=64, epochs=10):
    model = FinetunedAlexNet()
    model.cuda()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    mean=[0.485, 0.456, 0.406]
    std=[0.229, 0.224, 0.225]

    train_tranform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
        transforms.Resize((386, 468)),
    ])

    train_dataset = dataloader.CubImageDataset(data_dir, 0, transform=train_tranform)
    train_loader = DataLoader(train_dataset, batch_size=batch_size)
    valid_dataset = dataloader.CubImageDataset(data_dir, 1, transform=train_tranform)
    valid_loader = DataLoader(valid_dataset, batch_size=batch_size)

    best_valid_acc = 0

    for epoch in range(epochs):
        
        if epoch == start_unfreeze_epoch:
            model.unfreezeAll()
            optimizer = optim.Adam(squeezenet.parameters(), lr=4e-5)
            
        model.train()
        print(f"Epoch {epoch+1}")
        train_loss = 0
        valid_loss = 0
        for i, data in enumerate(train_loader):
            x, y = data
            x = x.cuda()
            y = y.cuda()
            optimizer.zero_grad()
            pred = model(x)
            loss = criterion(pred, y)
            loss.backward()

            optimizer.step()
            train_loss += loss.item() * x.size(0)
            print('.', end='', flush=True)
        print()
        model.eval()

        valid_acc = 0
        for i, data in enumerate(valid_loader):
            x, y = data
            x = x.cuda()
            y = y.cuda()
            pred = model(x)
            loss = criterion(pred, y)
            valid_loss += loss.item() * x.size(0)
            y_actual = y.data.cpu().numpy()
            pred = torch.max(pred,1)[1]
            y_pred = pred.detach().cpu().numpy()
            valid_acc += accuracy_score(y_actual, y_pred, normalize=False)
            print('*', end="", flush=True)
        train_loss = train_loss / len(train_dataset)
        valid_loss = valid_loss / len(valid_dataset)
        valid_acc = valid_acc / len(valid_dataset)

        print()
        print(f'Epch {epoch+1}, Training Loss: {train_loss}, Validation Loss: {valid_loss}, Validation Accuracy: {valid_acc}')

        if valid_acc > best_valid_acc:
            print(f"New best validation accuracy ({best_valid_acc} -> {valid_acc})")
            print("Saving model")
            torch.save(model.state_dict(), save_dir +'best_alexnet_baseline.pt')
            print("Saved")
            best_valid_acc = valid_acc
    print("Finished Training")
    
if __name__ == '__main__':
    train()