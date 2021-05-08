import torch
import torch.nn as nn
import torch.optim as optim
from CUB_200_2011 import dataloader
from torch.utils.data import DataLoader
import numpy as np
from sklearn.metrics import accuracy_score
from torchvision import transforms

def train(m, dir, batch_size, epochs):
    model = m
    model.cuda()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    transform = transforms(xxx)

    train_dataset = dataloader.CubImageDataset(dir, 0, transform=transform)
    train_loader = DataLoader(train_dataset, batch_size=batch_size)
    valid_dataset = dataloader.CubImageDataset(dir, 1, transform=transform)
    valid_loader = DataLoader(valid_dataset, batch_size=batch_size)

    best_valid_acc = 0

    for epoch in range(epochs):
        model.train()
        print(f"Epoch {epoch+1}")
        train_loss = 0
        valid_loss = 0
        for i, data in enumerate(train_loader):
            x, y = data
            x = x.cuda()
            y = y.cuda()
            y = y.view(-1, 1)
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
            y = y.view(-1,1)
            pred = model(x)
            loss = criterion(pred, y)
            valid_loss += loss.item() * x.size(0)
            y_actual = y.data.cpu().numpy()
            y_pred = pred.detach().cpu().numpy()
            y_pred = np.argmax(y_pred, axis=0)
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
            torch.save(model.state_dict(), 'best_xxx.pt')
            print("Saved")
            best_valid_acc = valid_acc