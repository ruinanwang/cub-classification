import torch
import torch.nn as nn
import torch.optim as optim
import dataloader
from torch.utils.data import DataLoader
import numpy as np
from sklearn.metrics import accuracy_score
from torchvision import transforms
from tqdm import tqdm

from cb_models import FinetunedAlexNet1
from cb_models import FinetunedResNet1
from cb_models import FinetunedInceptionV3_1
from FullyConnectedModel import FullyConnectedModel

from plot import plot

import argparse
from torch.utils.tensorboard import SummaryWriter

parser = argparse.ArgumentParser(description='PUB training args')
parser.add_argument("-m", type=int, required=True)
parser.add_argument("-model_name", type=str, required=False)
parser.add_argument("-n", type=str, required=True)
args = parser.parse_args()
train_writer = SummaryWriter(log_dir="../runs/"+args.n+"/train")
val_writer = SummaryWriter(log_dir="../runs/"+args.n+"/val")


def train_first_model(args, train_writer, val_writer, data_dir="../data/", save_dir="../save/", batch_size=64, epochs=15, num_attributes=85):
    if args.model_name.lower() in ['alexnet', 'alex']:
        model = FinetunedAlexNet1(num_attributes)
        size = (256, 256)
    elif args.model_name.lower() in ['resnet', 'res']:
        model = FinetunedResNet1(num_attributes)
        size = (256, 256)
    elif args.model_name.lower() in ['inception']:
        model = FinetunedInceptionV3_1(num_attributes)
        size = (299, 299)
        if batch_size == 64: batch_size = 32
    model.cuda()
    criterion = nn.BCELoss() #nn.CrossEntropyLoss() #nn.MultiLabelMarginLoss() #nn.BCEWithLogitsLoss() #nn.CrossEntropyLoss()
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
        transforms.Resize(size),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.RandomAffine(0, shear=10, scale=(0.8,1.2)),
        transforms.ColorJitter(brightness=1, contrast=1, saturation=1),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    valid_transform = transforms.Compose([
        transforms.Resize(size),
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
            y = y.type(torch.FloatTensor)
            y = y.cuda()
            pred = model(x)
            loss = None
            print("new pred: ", pred)
            for ind, p in enumerate(pred):
                target = y[:, ind].reshape(-1, 1)
                if loss==None:
                    loss = criterion(p, target)
                else:
                    loss += criterion(p, target)
                p = (p>0.5).type(torch.cuda.FloatTensor)
                train_acc += torch.sum(p==target)
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
            y = y.type(torch.FloatTensor)
            y = y.cuda()
            pred = model(x)
            loss = None
            for ind, p in enumerate(pred):
                target = y[:, ind].reshape(-1, 1)
                if loss==None:
                    loss = criterion(p, target)
                else:
                    loss += criterion(p, target)

                p = (p>0.5).type(torch.cuda.FloatTensor)
                valid_acc += torch.sum(p==target)

            valid_loss += loss.item()

        train_loss = train_loss / len(train_dataset)
        valid_loss = valid_loss / len(valid_dataset)
        valid_acc = valid_acc / (len(valid_dataset)*num_attributes)
        train_acc = train_acc / (len(train_dataset)*num_attributes)
        
        train_writer.add_scalar("Loss", train_loss, epoch)
        val_writer.add_scalar("Loss", valid_loss, epoch)
        train_writer.add_scalar("Accuracy", train_acc, epoch)
        val_writer.add_scalar("Accuracy", valid_acc, epoch)
        
        train_acc_list.append(train_acc)
        train_loss_list.append(train_loss)
        valid_acc_list.append(valid_acc)
        valid_loss_list.append(valid_loss)

        print()
        print(f'Epoch {epoch+1}, Training Loss: {train_loss}, Training Accuracy: {train_acc}, Validation Loss: {valid_loss}, Validation Accuracy: {valid_acc}')

        if valid_acc > best_valid_acc:
            print(f"New best validation accuracy ({best_valid_acc} -> {valid_acc})")
            print("Saving model...")
            torch.save(model.state_dict(), save_dir + args.n + '.pt')
            print("Saved")
            best_valid_acc = valid_acc
        print(f"Current Best Valid Accuracy: {best_valid_acc}\n")
            
    plot(train_loss_list, train_acc_list, valid_loss_list, valid_acc_list, save_dir + args.n)
    
    return train_prediction_output, validation_prediction_output

def train_second_model(args, train_writer, val_writer, data_dir="../data/", save_dir="../save/", batch_size=64, epochs=70, num_attributes=85):
    model = FullyConnectedModel(input_size=num_attributes, hidden_size=150, num_classes=200, num_layers=1)
    model.cuda()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.1)


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
            x = x.cuda()
            y = y.cuda()
            pred = model(x)
            loss = criterion(pred, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            pred = torch.max(pred, 1)[1]
            train_acc += torch.sum(pred==y)
            train_loss += loss.item()
        print()
        
        model.eval()
        print("Validating...")
        for i, data in tqdm(enumerate(valid_loader)):
            x, y = data
            x = x.type(torch.FloatTensor)
            x = x.cuda()
            y = y.cuda()
            pred = model(x)
            loss = criterion(pred, y)
            valid_loss += loss.item()
            pred = torch.max(pred, 1)[1]
            valid_acc += torch.sum(pred==y)
        print(valid_acc, len(valid_dataset))
        train_loss = train_loss / len(train_dataset)
        valid_loss = valid_loss / len(valid_dataset)
        train_acc = train_acc / len(train_dataset)
        valid_acc = valid_acc / len(valid_dataset)
        
        train_writer.add_scalar("Loss", train_loss, epoch)
        val_writer.add_scalar("Loss", valid_loss, epoch)
        train_writer.add_scalar("Accuracy", train_acc, epoch)
        val_writer.add_scalar("Accuracy", valid_acc, epoch)
        
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
    if args.m == 1:
        train_first_model(args, train_writer, val_writer)
        train_writer.flush()
        val_writer.flush()
    elif args.m == 2:
        train_second_model(args, train_writer, val_writer)
        train_writer.flush()
        val_writer.flush()