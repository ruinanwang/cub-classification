import torch
import torch.nn as nn
import torch.optim as optim
import dataloader
from torch.utils.data import DataLoader
import numpy as np
from sklearn.metrics import accuracy_score
from torchvision import transforms

from finetuned_alexnet import FinetunedAlexNet
from finetuned_alexnet import FinetunedResNet

start_unfreeze_epoch = 7

def train(data_dir="../data", save_dir="../save/", batch_size=64, epochs=10):

    transformation_train = transforms.Compose([
        transforms.Resize((256,256)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.RandomAffine(0, shear=10, scale=(0.8,1.2)),
        transforms.ColorJitter(brightness=1, contrast=1, saturation=1),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    
    transformation_valid = transforms.Compose([
        transforms.Resize((256,256)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    
    mean=[0.485, 0.456, 0.406]
    std=[0.229, 0.224, 0.225]

    train_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
        transforms.Resize((386, 468)),
    ])

    train_dataset = dataloader.CubImageDataset(data_dir, 0, transform=train_transform)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    valid_dataset = dataloader.CubImageDataset(data_dir, 1, transform=train_transform)
    valid_loader = DataLoader(valid_dataset, batch_size=batch_size)
    
    model = FinetunedResNet()
    model.cuda()
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)

    best_valid_acc = 0

    for epoch in range(epochs):
            
        model.train()
        print(f"Epoch {epoch+1}")
        train_loss = 0
        train_acc = 0
#         min_label_predicted = 10000
        for i, data in enumerate(train_loader):
            x, y = data
            x = x.cuda()
            y = y.cuda()
            
            pred = model(x)
            loss = criterion(pred, y)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            pred = torch.max(pred,1)[1]
#             if min(pred) < min_label_predicted:
#                 min_label_predicted = min(pred)
#                 print(min_label_predicted)
#             print(pred)
            
            train_loss += loss.item()
            train_acc += torch.sum( pred == y )

            print('.', end='', flush=True)
        print()
        model.eval()
           
        valid_loss = 0
        valid_acc = 0
        for i, data in enumerate(valid_loader):
            x, y = data
            x = x.cuda()
            y = y.cuda()
            
            pred = model(x)
            loss = criterion(pred, y)
            pred = torch.max(pred,1)[1]
            
            valid_loss += loss.item()
            valid_acc += torch.sum( pred == y )
            
            print('*', end="", flush=True)
        train_loss = train_loss / len(train_dataset)
        train_acc = train_acc / len(train_dataset)
        valid_loss = valid_loss / len(valid_dataset)
        valid_acc = valid_acc / len(valid_dataset)

        print()
        print(f'Epch {epoch+1}, Training Loss: {train_loss}, Training Accuracy: {train_acc}, Validation Loss: {valid_loss}, Validation Accuracy: {valid_acc}')

        if valid_acc > best_valid_acc:
            print(f"New best validation accuracy ({best_valid_acc} -> {valid_acc})")
            print("Saving model")
            torch.save(model.state_dict(), save_dir +'best_resnet_baseline.pt')
            print("Saved")
            best_valid_acc = valid_acc
    print("Finished Training")
    
if __name__ == '__main__':
    train()