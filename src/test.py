import torch
import torch.nn as nn
import torch.optim as optim
from CUB_200_2011 import dataloader
from torch.utils.data import DataLoader
import numpy as np
from sklearn.metrics import accuracy_score
from torchvision import transforms

from finetuned_alexnet import FinetunedAlexNet

def test(data_dir="../data", save_dir="../save/", batch_size=64):
    model = FineTunedAlexNet()
    model.load_state_dict(torch.load(save_dir+'best_alexnet_baseline.pt'))
    model.cuda()
    model.eval()

    test_transform = transforms.Compose([
        transforms.ToTensor(),
#         transforms.Normalize(mean=mean, std=std),
        transforms.Resize((386, 468)),
    ])

    test_dataset = dataloader.CubImageDataset(data_dir, 2, transform=test_transform)
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

if __name__ == '__main__':
    test()