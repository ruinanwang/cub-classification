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

import argparse

parser = argparse.ArgumentParser(description='PUB training args')
parser.add_argument("-n", type=str, required=True)
args = parser.parse_args()

def test(args, data_dir="../data", save_dir="../save/", batch_size=64):
    model = FinetunedAlexNet()
#     model = FinetunedResNet()
    model.load_state_dict(torch.load(save_dir + args.n + '.pt'))
    model.cuda()
    model.eval()

#     test_transform = transforms.Compose([
#         transforms.ToTensor(),
#         transforms.Resize((386, 468)),
#     ])
    mean=[0.485, 0.456, 0.406]
    std=[0.229, 0.224, 0.225]
    
    test_transform = transforms.Compose([
        transforms.Resize((386, 468)),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std)
    ])

    test_dataset = dataloader.CubImageDataset(data_dir, 2, False, transform=test_transform)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)

    test_acc = 0.0
    prediction = np.array([])
    
    for x, y in test_loader:
        x = x.cuda()
        y = y.cuda()
        pred = model(x)
        pred = torch.max(pred,1)[1]
        test_acc += torch.sum( pred == y )
    test_acc /= len(test_dataset)
    
    print(f"TEST Accuracy: {test_acc}")
    return prediction

if __name__ == '__main__':
    test(args)