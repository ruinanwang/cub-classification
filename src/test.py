import torch
import torch.nn as nn
import torch.optim as optim
from CUB_200_2011 import dataloader
from torch.utils.data import DataLoader
import numpy as np
from sklearn.metrics import accuracy_score
from torchvision import transforms

def test(m, dir, batch_size):
    model = m
    model.load_state_dict(torch.load('best_xxx.pt'))
    model.cuda()
    model.eval()

    test_transform = transforms(xxx)

    test_dataset = dataloader.CubImageDataset(dir, 2, transform=test_transform)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)

    test_acc = 0.0
    prediction = np.array([])
    
    for x, y in test_loader:
        x = x.cuda()
        pred = model(x)
        y_pred = pred.detach().cpu().numpy()
        y_pred = np.argmax(y_pred, axis=0)
        test_acc += accuracy_score(y.data.cpu().numpy(), y_pred, normalize=False)
        prediction = np.concatenate(prediction, y_pred)
    test_acc /= len(test_dataset)
    
    print(f"TEST Accuracy: {test_acc}")
    return prediction