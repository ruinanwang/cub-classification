import torch
import torch.nn as nn
import torch.optim as optim
import dataloader
from torch.utils.data import DataLoader
import numpy as np
from sklearn.metrics import accuracy_score
from torchvision import transforms

from ete_models import FinetunedAlexNet
from ete_models import FinetunedResNet
from ete_models import FinetunedVggNet
from ete_models import FinetunedDenseNet

from plot import *

import argparse
from torch.utils.tensorboard import SummaryWriter

parser = argparse.ArgumentParser(description='PUB training args')
parser.add_argument("-n", type=str, required=True)
args = parser.parse_args()
train_writer = SummaryWriter(log_dir="../runs/"+args.n+"/train")
val_writer = SummaryWriter(log_dir="../runs/"+args.n+"/val")


start_unfreeze_epoch = 7

def train(args, train_writer, val_writer, data_dir="../data", save_dir="../save/", batch_size=64, epochs=20):
    model = FinetunedDenseNet()
    model.cuda()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.01)

    mean=[0.485, 0.456, 0.406]
    std=[0.229, 0.224, 0.225]

    transformation_train = transforms.Compose([
        transforms.Resize((256,256)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.RandomAffine(0, shear=10, scale=(0.8,1.2)),
        transforms.ColorJitter(brightness=1, contrast=1, saturation=1),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std)
    ])
    
    transformation_valid = transforms.Compose([
        transforms.Resize((256,256)),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std)
    ])

    train_dataset = dataloader.CubImageDataset(data_dir, 0, False, transform=transformation_train)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    valid_dataset = dataloader.CubImageDataset(data_dir, 1, False, transform=transformation_valid)
    valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=True)

    best_valid_acc = 0

    for epoch in range(epochs):
        
#         if epoch == start_unfreeze_epoch:
#             model.unfreezeAll()
#             optimizer = optim.SGD(model.parameters(), lr=0.01)
            
        model.train()
        print(f"Epoch {epoch+1}")
        train_loss = 0
        train_acc = 0
        valid_loss = 0
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
            train_acc += torch.sum(pred==y)
            train_loss += loss.item()
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
            valid_loss += loss.item()
            pred = torch.max(pred,1)[1]
            valid_acc += torch.sum(pred==y)
            print('*', end="", flush=True)
        train_loss = train_loss / len(train_dataset)
        valid_loss = valid_loss / len(valid_dataset)
        valid_acc = valid_acc / len(valid_dataset)
        train_acc = train_acc / len(train_dataset)
        
        train_writer.add_scalar("Loss", train_loss, epoch)
        val_writer.add_scalar("Loss", valid_loss, epoch)
        train_writer.add_scalar("Accuracy", train_acc, epoch)
        val_writer.add_scalar("Accuracy", valid_acc, epoch)

        print()
        print(f'Epch {epoch+1}, Training Loss: {train_loss}, Training Accuracy: {train_acc}, Validation Loss: {valid_loss}, Validation Accuracy: {valid_acc}')

        if valid_acc > best_valid_acc:
            print(f"New best validation accuracy ({best_valid_acc} -> {valid_acc})")
            print("Saving model")
            torch.save(model.state_dict(), save_dir + args.n + '.pt')
            print("Saved")
            best_valid_acc = valid_acc
    print("Finished Training")
    
if __name__ == '__main__':
    train(args, train_writer, val_writer)