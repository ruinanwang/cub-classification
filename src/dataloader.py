import os
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
import matplotlib.pyplot as plt
from torchvision.datasets.folder import default_loader

class CubImageDataset(Dataset):
    def __init__(self, root_dir, train_val_test_mode, use_annotation, transform=None, part=0):
        self.root_dir = root_dir
        self.train_val_test_mode = train_val_test_mode
        self.transform = transform
        self.use_annotation = use_annotation
        self.part = part
        img_path = pd.read_csv(
            os.path.join(self.root_dir, 'CUB_200_2011', 'images.txt'), 
            sep=' ', 
            names=['img_id', 'img_path'])
        img_labels = pd.read_csv(
            os.path.join(self.root_dir, 'CUB_200_2011', 'image_class_labels.txt'), 
            sep=' ', 
            names=['img_id', 'label'])
        train_val_test_split = pd.read_csv(
            os.path.join(self.root_dir, 'CUB_200_2011', 'train_val_test_split.txt'), 
            sep=' ', 
            names=['img_id', 'train_val_test'])
        self.annotations = pd.read_csv(
            os.path.join(self.root_dir, 'CUB_200_2011', 'attributes', 'attributes_majority_votes.txt'), 
            sep=',',
            header=None)
        images = img_path.merge(img_labels, on='img_id').merge(train_val_test_split, on='img_id')
        if self.train_val_test_mode == 0:
            self.data = images[images.train_val_test == 0]
        if self.train_val_test_mode == 1:
            self.data = images[images.train_val_test == 1]
        if self.train_val_test_mode == 2:
            self.data = images[images.train_val_test == 2]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        if self.use_annotation == False:
            sample = self.data.iloc[idx]
            path = os.path.join(self.root_dir, 'CUB_200_2011', 'images', sample[1])
            image = default_loader(path)
            if self.transform:
                image = self.transform(image)
            label = sample[2] - 1 # make labels start from index 0
#             print(image, label)
            return image, label
        if self.use_annotation == True:
            if self.part == 0:
                sample = self.data.iloc[idx]
                path = os.path.join(self.root_dir, 'CUB_200_2011', 'images', sample[1])
                image = default_loader(path)
                if self.transform:
                      image = self.transform(image)
                label = self.annotations.iloc[idx].to_numpy()
#                 print(image, label)
                return image, label
            if self.part == 1:
                sample = self.data.iloc[idx]
                annotation = self.annotations.iloc[idx].to_numpy()
                label = sample[2] - 1
#                 print(annotation, label)
                return annotation, label