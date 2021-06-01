import torch
import torch.nn as nn
import torch.optim as optim
import dataloader
from torch.utils.data import DataLoader
import numpy as np
from sklearn.metrics import accuracy_score
from torchvision import transforms
from tqdm import tqdm

# from cb_models import FinetunedAlexNet2
from cb_models import FinetunedResNet2
from cb_models import FinetunedInceptionV3_2
from FullyConnectedModel import FullyConnectedModel

import argparse

parser = argparse.ArgumentParser(description='PUB training args')
parser.add_argument("-n1", type=str, required=True)
parser.add_argument("-n2", type=str, required=True)
parser.add_argument("-model_name", type=str, required=True)
args = parser.parse_args()

def test(args, data_dir="../data", save_dir="../save/", batch_size=64, num_attributes=89):
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

    model1.load_state_dict(torch.load(save_dir + args.n1 + '.pt'))
    model1.cuda()
    model1.eval()
    
    model2 = FullyConnectedModel(input_size=num_attributes, hidden_size=150, num_classes=200)
    model2.load_state_dict(torch.load(save_dir + args.n2 + '.pt'))
    model2.cuda()
    model2.eval()

    mean=[0.485, 0.456, 0.406]
    std=[0.229, 0.224, 0.225]
    test_transform = transforms.Compose([
        transforms.Resize(size),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])

    test_dataset = dataloader.CubImageDataset(data_dir, 2, False, transform=test_transform)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)

    test_acc = 0.0
    prediction = np.array([])
    
    print("Testing...")
    
    for x, y in tqdm(test_loader):
        x = x.cuda()
        y = y.cuda()
        pred = model2(model1(x))
        pred = torch.max(pred, 1)[1]
        test_acc += torch.sum(pred==y)
    test_acc /= len(test_dataset)
    
    print(f"TEST Accuracy: {test_acc}")
    return prediction

if __name__ == '__main__':
    test(args)