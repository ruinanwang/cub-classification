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

import argparse

parser = argparse.ArgumentParser(description='PUB training args')
parser.add_argument("-n", type=str, required=True)
args = parser.parse_args()

def test(args, data_dir="../data", save_dir="../save/", batch_size=64):
    model = FinetunedAlexNet1()
#     model = FinetunedResNet1()
    model.load_state_dict(torch.load(save_dir + args.n + '.pt'))
    model.cuda()
    model.eval()

    mean=[0.485, 0.456, 0.406]
    std=[0.229, 0.224, 0.225]
    test_transform = transforms.Compose([
        transforms.Resize((386, 468)),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
    ])

    test_dataset = dataloader.CubImageDataset(data_dir, 2, True, transform=test_transform)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)

    test_acc = 0.0
    prediction = np.array([])
    
    print("Testing...")
    
    for x, y in tqdm(test_loader):
        x = x.cuda()
        y = y.cuda()
        pred = model(x)
#         y[y>0] = 1
#         y[y<0] = -1
        y += 1
#         y = y.type(torch.cuda.LongTensor)
        for ind, p in enumerate(pred):
            p = torch.max(p, 1)[1]
            test_acc += torch.sum( p == y[:, ind] )
    test_acc /= len(test_dataset)*312
    
    print(f"TEST Accuracy: {test_acc}")
    return prediction

if __name__ == '__main__':
    test(args)