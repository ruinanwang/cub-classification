import torch
import torch.nn as nn
import torch.optim as optim
import dataloader
from torch.utils.data import DataLoader
import numpy as np
from sklearn.metrics import accuracy_score
from torchvision import transforms
from tqdm import tqdm

from finetuned_alexnet import FinetunedAlexNet1
from finetuned_alexnet import FinetunedResNet1
from FullyConnectedModel import FullyConnectedModel

from plot import plot

import argparse

parser = argparse.ArgumentParser(description='PUB training args')
parser.add_argument("-n", type=str, required=True)
args = parser.parse_args()


def train_first_model(args, data_dir="../data/", save_dir="../save/", batch_size=64, epochs=10):
    model = FinetunedAlexNet1()
#     model = FinetunedResNet1()
    model.cuda()
#     print(model)
    criterion = nn.CrossEntropyLoss() #nn.MultiLabelMarginLoss() #nn.BCEWithLogitsLoss() #nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.1)

    mean=[0.485, 0.456, 0.406]
    std=[0.229, 0.224, 0.225]
#     train_transform = transforms.Compose([
#         transforms.ToTensor(),
#         transforms.Normalize(mean=mean, std=std),
#         transforms.Resize((386, 468)),
#     ])
#     train_transform = transforms.Compose([
#         transforms.ColorJitter(brightness=1, contrast=1, saturation=1),
#         transforms.RandomHorizontalFlip(),
#         transforms.RandomRotation(15),
#         transforms.Resize((386, 468)),
#         transforms.ToTensor(),
#         transforms.Normalize(mean=mean, std=std),
#     ])
    
#     valid_transform = transforms.Compose([
#         transforms.Resize((386, 468)),
#         transforms.ToTensor(),
#         transforms.Normalize(mean=mean, std=std),
#     ])

    train_transform = transforms.Compose([
        transforms.Resize((256,256)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.RandomAffine(0, shear=10, scale=(0.8,1.2)),
        transforms.ColorJitter(brightness=1, contrast=1, saturation=1),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    valid_transform = transforms.Compose([
        transforms.Resize((256,256)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    train_dataset = dataloader.CubImageDataset(data_dir, 0, True, transform=train_transform)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    valid_dataset = dataloader.CubImageDataset(data_dir, 1, True, transform=valid_transform)
    valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=True)

    best_valid_acc = 0
    train_acc_list = []
    train_loss_list = []
    valid_acc_list = []
    valid_loss_list = []

    train_prediction_output = np.array([])
    validation_prediction_output = np.array([])

    for epoch in range(epochs):
        model.train()
        print(f"Epoch {epoch+1}")
        train_loss = 0
        train_acc = 0
        valid_loss = 0
        print("Training...")
        for i, data in tqdm(enumerate(train_loader)):
            x, y = data
            x = x.cuda()
            y = y.cuda()
            
            pred = model(x)
           
            loss = None
            y += 1

            for ind, p in enumerate(pred):
                if loss==None:
                    loss = criterion(p, y[:, ind])
                else:
                    loss += criterion(p, y[:, ind])
                p = torch.max(p, 1)[1]
                train_acc += torch.sum(p==y[:, ind])
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
        model.eval()

        valid_acc = 0
        print("Validating...")
        for i, data in tqdm(enumerate(valid_loader)):
            x, y = data
            x = x.cuda()
            y = y.cuda()
            pred = model(x)
            loss = None
            y += 1
            for ind, p in enumerate(pred):
                if loss==None:
                    loss = criterion(p, y[:, ind])
                else:
                    loss += criterion(p, y[:, ind])

                p = torch.max(p, 1)[1]
                valid_acc += torch.sum(p==y[:, ind])

            valid_loss += loss.item()

        train_loss = train_loss / len(train_dataset)
        valid_loss = valid_loss / len(valid_dataset)
        valid_acc = valid_acc / (len(valid_dataset)*312)
        train_acc = train_acc / (len(train_dataset)*312)
        
        train_acc_list.append(train_acc)
        train_loss_list.append(train_loss)
        valid_acc_list.append(valid_acc)
        valid_loss_list.append(valid_loss)

        print()
        print(f'Epoch {epoch+1}, Training Loss: {train_loss}, Training Accuracy: {train_acc}, Validation Loss: {valid_loss}, Validation Accuracy: {valid_acc}\n')

        if valid_acc > best_valid_acc:
            print(f"New best validation accuracy ({best_valid_acc} -> {valid_acc})")
            print("Saving model...")
            torch.save(model.state_dict(), save_dir + args.n + '.pt')
            print("Saved\n")
            best_valid_acc = valid_acc
            
    plot(train_loss_list, train_acc_list, valid_loss_list, valid_acc_list, save_dir + args.n)
    
    return train_prediction_output, validation_prediction_output

def train_second_model(args, data_dir="../data/", save_dir="../save/", batch_size=64, epochs=500):
    model = FullyConnectedModel(input_size=85, hidden_size=150, num_classes=200)
    model.cuda()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.01)


    train_dataset = dataloader.CubImageDataset(data_dir, 0, True, part=1)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    valid_dataset = dataloader.CubImageDataset(data_dir, 1, True, part=1)
    valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=True)

    best_valid_acc = 0
    train_acc_list = []
    train_loss_list = []
    valid_acc_list = []
    valid_loss_list = []

    for epoch in range(epochs):
        model.train()
        print(f"Epoch {epoch+1}")
        train_loss = 0
        train_acc = 0
        valid_loss = 0
        valid_acc = 0
        print("Training...")
        for i, data in tqdm(enumerate(train_loader)):
            x, y = data
            x = x.type(torch.FloatTensor)
#             x = x[:, :100]
#             print(x.shape)
            x = x.cuda()
            y = y.cuda()
            pred = model(x)
            loss = criterion(pred, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            pred = torch.max(pred, 1)[1]
#             if epoch >= 5:
#                 print("Epoch ", epoch, ": TRAINING WRONG CLASSIFICATION - should be: ", y, " but predicted: ", pred)
            train_acc += torch.sum(pred==y)
            train_loss += loss.item()
#             print('.', end='', flush=True)
        print()
        model.eval()

        print("Validating...")
        for i, data in tqdm(enumerate(valid_loader)):
            x, y = data
            x = x.type(torch.FloatTensor)
#             x = x[:, :100]
            x = x.cuda()
            y = y.cuda()
            pred = model(x)
            loss = criterion(pred, y)
            valid_loss += loss.item()
            pred = torch.max(pred, 1)[1]
#             if epoch >= 5:
#                 print("Epoch ", epoch, ": VALIDATION WRONG CLASSIFICATION - should be: ", y, " but predicted: ", pred)
            valid_acc += torch.sum(pred==y)
#             print('*', end="", flush=True)
        print(valid_acc, len(valid_dataset))
        train_loss = train_loss / len(train_dataset)
        valid_loss = valid_loss / len(valid_dataset)
        valid_acc = valid_acc / len(valid_dataset)
        train_acc = train_acc / len(train_dataset)
        
        train_acc_list.append(train_acc)
        train_loss_list.append(train_loss)
        valid_acc_list.append(valid_acc)
        valid_loss_list.append(valid_loss)

        print()
        print(f'Epoch {epoch+1}, Training Loss: {train_loss}, Training Accuracy: {train_acc}, Validation Loss: {valid_loss}, Validation Accuracy: {valid_acc}')

        if valid_acc > best_valid_acc:
            print(f"New best validation accuracy ({best_valid_acc} -> {valid_acc})")
            print("Saving model")
            torch.save(model.state_dict(), save_dir + args.n + '.pt')
            print("Saved")
            best_valid_acc = valid_acc
        print(f"Current Best Valid Accuracy: {best_valid_acc}\n")
    plot(train_loss_list, train_acc_list, valid_loss_list, valid_acc_list, save_dir + args.n)

            
if __name__=='__main__':
#     train_first_model(args)
    train_second_model(args)