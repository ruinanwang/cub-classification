import torch
import torch.nn as nn
import torch.optim as optim
import dataloader
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm

# from cb_models import FinetunedAlexNet2
from cb_models import FinetunedResNet2
from cb_models import FinetunedInceptionV3_2
from cb_models import FinetunedVggNet
from cb_models import FinetunedDenseNet
from FullyConnectedModel import FullyConnectedModel

from plot import plot

import argparse
from torch.utils.tensorboard import SummaryWriter

parser = argparse.ArgumentParser(description='PUB training args')
parser.add_argument("-n1", type=str, required=True)
parser.add_argument("-n2", type=str, required=True)
parser.add_argument("-model_name", type=str, required=True)
# parser.add_argument("-n", type=str, required=True)
parser.add_argument("-hidden", type=int, required=True)
parser.add_argument("-layers", type=int, required=True, default=2)
args = parser.parse_args()
n = args.n1 + " " + args.n2 + " combine"
train_writer = SummaryWriter(log_dir="../runs/"+n+"/train")
val_writer = SummaryWriter(log_dir="../runs/"+n+"/val")

def train(args, train_writer, val_writer, data_dir="../data/", save_dir="../save/", batch_size=64, epochs=25, num_attributes=89):
    if args.model_name.lower() in ['alexnet', 'alex']:
        model1 = FinetunedAlexNet2(num_attributes) #haven't created yet
        size = (256, 256)
    elif args.model_name.lower() in ['resnet', 'res']:
        model1 = FinetunedResNet2(num_attributes)
        size = (256, 256)
    elif args.model_name.lower() in ['inception']:
        model1 = FinetunedInceptionV3_2(num_attributes)
        size = (299, 299)
        if batch_size == 64: batch_size = 32
    elif args.model_name.lower() in ['vgg', 'vggnet']:
        model1 = FinetunedVggNet(num_attributes)
        size = (256, 256)
        if batch_size == 64: batch_size = 32
    elif args.model_name.lower() in ['dense', 'densenet']:
        model1 = FinetunedDenseNet(num_attributes)
        size = (256, 256)

    model1.load_state_dict(torch.load(save_dir + args.n1 + '.pt'))
    model1.cuda()
    
    model2 = FullyConnectedModel(input_size=num_attributes, hidden_size=args.hidden, num_classes=200, num_layers=args.layers)
    model2.load_state_dict(torch.load(save_dir + args.n2 + '.pt'))
    model2.cuda()
    
    print('Successfully loaded in models')

    criterion = nn.CrossEntropyLoss()
    params = list(model1.parameters()) + list(model2.parameters())
    optimizer = optim.SGD(params, lr=0.01)

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

    train_dataset = dataloader.CubImageDataset(data_dir, 0, False, transform=train_transform)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    valid_dataset = dataloader.CubImageDataset(data_dir, 1, False, transform=valid_transform)
    valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=True)

    best_valid_acc = 0
    train_acc_list = []
    train_loss_list = []
    valid_acc_list = []
    valid_loss_list = []

    for epoch in range(epochs):
        model1.train()
        model2.train()
        print(f"Epoch {epoch+1}")
        train_loss = 0
        train_acc = 0
        valid_loss = 0
        valid_acc = 0
        print("Training...")
        for x, y in tqdm(train_loader):
            x = x.cuda()
            y = y.cuda()
            out1 = model1(x)
            pred = model2(out1)
            loss = criterion(pred, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            pred = torch.max(pred, 1)[1]
            train_acc += torch.sum(pred==y)
            train_loss += loss.item()
        
        model1.eval()
        model2.eval()
        print("Validating...")
        for x, y in tqdm(valid_loader):
            x = x.cuda()
            y = y.cuda()
            out1 = model1(x)
            pred = model2(out1)
            loss = criterion(pred, y)
            valid_loss += loss.item()
            pred = torch.max(pred, 1)[1]
            valid_acc += torch.sum(pred==y)
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
            torch.save(model1.state_dict(), save_dir + args.n1 + '_combine.pt')
            torch.save(model2.state_dict(), save_dir + args.n2 + '_combine.pt')
            print("Saved")
            best_valid_acc = valid_acc
        print(f"Current Best Valid Accuracy: {best_valid_acc}\n")
    plot(train_loss_list, train_acc_list, valid_loss_list, valid_acc_list, save_dir + args.n1 + " " + args.n2 + " combine")


if __name__ == '__main__':
    train(args, train_writer, val_writer)