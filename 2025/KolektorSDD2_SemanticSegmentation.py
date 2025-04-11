import numpy as np
import torch
import numpy
import cv2
import os
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn

from scipy.ndimage import histogram

root = "data/KolektorSDD2"

def explore_data():
    histogram = np.zeros(256)
    for file_name in os.listdir(os.path.join(root, "train")):
        if file_name.__contains__("_GT"):
            mask = cv2.imread(os.path.join(root, "train", file_name), cv2.IMREAD_UNCHANGED)
            his, _ = np.histogram(mask, bins=256, range=(0, 255))
            histogram += his

    print(histogram)
    return

class KolektorSDD2Dataset(Dataset)
    def __init__(self, transforms, data_path):
        super(KolektorSDD2Dataset, self).__init__()
        self.transforms = transforms
        self.image_files = [os.path.join(data_path, file) for file in data_path if not file.__contains__("_GT")]
        self.mask_files = [file.splitext()[0] + "_GT.png" for file in self.image_files]

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, index):
        image_file = self.image_files[index]
        mask_file = self.mask_files[index]
        image = cv2.imread(image_file, cv2.IMREAD_UNCHANGED)
        mask = cv2.imread(mask_file, cv2.IMREAD_UNCHANGED)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        mask[mask == 255] = 1
        augmented = self.transforms(image=image, mask=mask)
        augmentd_image = augmented['image']
        augmentd_mask = augmented['mask']
        return augmentd_image, augmentd_mask

class DoubleConv(torch.nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DoubleConv, self).__init__()
        self.network = torch.nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=(3, 3), padding=1, stride=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.Conv2d(out_channels, out_channels, kernel_size=(3, 3), padding=1, stride=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
        )

    def forward(self, x):
        return self.network(x)

class UNET(torch.nn.Module):
    def __init__(self):
        super(UNET, self).__init__()
        pass

    def forward(self, x):
        pass



    


if __name__ == '__main__':
    explore_data()