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

def test(data_dir="../data", save_dir="../save/", batch_size=64):
    model = FinetunedAlexNet()
#     model = FinetunedResNet()
    model.load_state_dict(torch.load(save_dir+'best_alexnet_baseline.pt'))
    model.cuda()
    model.eval()

    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((386, 468)),
    ])

    test_dataset = dataloader.CubImageDataset(data_dir, 2, transform=test_transform)
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
    test()