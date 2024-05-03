import os
import PIL
import re
from PIL import Image
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import sys
sys.path.append("..")
from utils.utils_Peter import convertRGB2CIELab




class Dataset_for_training(Dataset):
    """
    Dataset class for training the DFE model.

    Args:
        file_paths (list): List of file paths to the training images.
        window_size (int): Size of the window for resizing the images.
        train_transform (torchvision.transforms): Data augmentation transforms for training.

    Methods:
        __len__(): Returns the number of images in the dataset.
        __getitem__(index): Returns the image and its corresponding label at the given index.

    Returns:
        tuple: A tuple containing the transformed image and its corresponding label.
    """
    def __init__(self, file_paths, window_size, train_transform):
        self.file_paths = file_paths
        self.window_size = window_size
        self.train_transform = train_transform

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, index):
        
        image = Image.open(self.file_paths[index])

        transform = transforms.Compose([
            transforms.Resize((self.window_size, self.window_size)),
            transforms.ToTensor(),
            # RGBtoLAB(),
            # transforms.Normalize(mean=[48.4597,  1.4077,  8.0963], std=[27.2366, 12.0433, 17.0808]),    #5Imagenet CIELAB
            # transforms.Normalize(mean=[0.1681, 0.0326, 0.0564], std=[0.0424, 0.0198, 0.0190]),   #hand4M CIELAB
        ])
        image = transform(image)
        image = image.permute(1, 2, 0)
        image = convertRGB2CIELab(image)
        image = torch.tensor(image).permute(2, 0, 1)
        label = image.clone()
        if self.train_transform is not None:
            image = self.train_transform(image)
        
        # print("image.shape: ", image.shape)
        return image, label
    

class Dataset_for_tracking(Dataset):
    """
    Dataset class for tracking using the DFE model.

    Args:
        data (list): List of images for tracking.

    Methods:
        __len__(): Returns the number of images in the dataset.
        __getitem__(index): Returns the image at the given index.

    Returns:
        torch.Tensor: The transformed image tensor.
    """
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        
        image = self.data[index]
        
        transform = transforms.Compose([
            transforms.ToTensor(),
            # transforms.Normalize(mean=[64.9681, 13.9973, 21.2239], std=[14.8747,  7.4034,  6.9481])
        ])
        image = transform(image)
        image = image.permute(1, 2, 0)
        image = convertRGB2CIELab(image)
        image = torch.tensor(image).permute(2, 0, 1)
        
        return image