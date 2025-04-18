import numpy as np
import torch
import numpy
import cv2
import os
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import albumentations as A
from albumentations.pytorch import ToTensorV2


from scipy.ndimage import histogram

epochs = 10
learning_rate = 1e-3
batch_size = 16
image_width = 176
image_height = 592

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

class KolektorSDD2Dataset(Dataset):
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
    def __init__(self, in_channels, out_channels, channels=[64, 128, 256, 512]):
        super(UNET, self).__init__()
        self.up = nn.ModuleList()
        self.down = nn.ModuleList()

        for idx, channel in enumerate(channels):
            self.down.append(DoubleConv(in_channels, channel))
            self.down.append(nn.MaxPool2d(kernel_size=(2, 2), stride=2, padding=2))

            in_channels = channels

        self.bottleneck = nn.ModuleList()
        self.bottleneck.append(DoubleConv(channels[-1], channels[-1] * 2))
        self.bottleneck.append(nn.ConvTranspose2d(channels[-1] * 2, channels[-1], kernel_size=2, stride=2, padding=2))

        # [512, 256, 128, 64]
        for idx, channel in enumerate(channels[::-1]):
            self.up.append(DoubleConv(channel * 2, channel))

            # if idx == len(channels) - 1:
            if channel != channels[::-1][-1]:
                self.up.append(nn.ConvTranspose2d(channel, channel // 2, kernel_size=2, stride=2, padding=2))

        self.out = nn.Conv2d(channels[0], out_channels, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        skip_connection = []
        for idx, m in enumerate(self.down):
            x = m(x)
            if idx % 2 == 0:
                skip_connection.append(x)

        for idx, m in enumerate(self.bottleneck):
            x = m(x)

        for idx, m in enumerate(self.up):
            if idx % 2 == 0:
                skip_feature = skip_connection[::-1][idx // 2]
                x = m(torch.cat([skip_feature, x], dim=1))
            else:
                x = m(x)

        x = self.out(x)
        x = nn.Sigmoid(x)
        return x

def train(model, dataloader):

    criterion = torch.nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters())

    for epoch in enumerate(range(epochs)):
        for step, batch in enumerate(dataloader):
            image, mask = batch
            pred_mask = model(image)
            loss = criterion(pred_mask.flatten(), mask.flatten())

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if step % 100 == 0:
                print("epoch: {}, step: {}, loss: {}".format(epoch, step, loss.item()))

if __name__ == '__main__':
    explore_data()
    transform = A.Compose([
        A.RandomCrop(width=image_width, height=image_height),
        A.HorizontalFlip(p=0.5),
        A.Normalize(mean=(0.0,), std=(1.0,)),
        ToTensorV2(),
    ])

    model = UNET(1, 1)

    train_dataset = KolektorSDD2Dataset(transform, os.path.join(root, "train"))
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    train(model, train_dataloader)







