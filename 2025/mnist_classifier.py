import torch
import numpy as np
from sklearn.metrics import f1_score
from sklearn.model_selection import learning_curve
from torchvision.datasets import MNIST
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import torch.nn as nn

# hyper parameters
batch_size = 32
epochs = 10
learning_rate = 1e-3

# dataset
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
 ])

train_dataset = MNIST(
    root='./data/mnist',
    train=True,
    transform=transform,
    download=True
)
val_dataset = MNIST(
    root='./data/mnist',
    train=False,
    transform=transform,
    download=True
)

train_loader = DataLoader(
    dataset=train_dataset,
    batch_size=batch_size,
    shuffle=True
)
val_loader = DataLoader(
    dataset=val_dataset,
    batch_size=batch_size,
    shuffle=False
)

# model
class FullyConnectedModel(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim):
        super(FullyConnectedModel, self).__init__()

        self.fc1 = torch.nn.Linear(input_dim, hidden_dim)
        self.fc2 = torch.nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        '''
        :param x: [B, 1, 28, 28]
        :return:
        '''
        batch_size = x.size()[0] # batch_size = x.shape[0]
        x = x.view(batch_size, -1) # [B, 784]
        x = self.fc1(x) # [B, h]
        x = torch.relu(x)
        x = self.fc2(x) # [B, 10] logits

        return x

class ConvolutionModel(nn.Module):
    def __init__(self):
        super(ConvolutionModel, self).__init__()
        self.conv1 = nn.Conv2d(
            in_channels=1,
            out_channels=20,
            kernel_size=(3, 3),
            padding=1,
            stride=1,
        )
        self.conv2 = nn.Conv2d(
            in_channels=20,
            out_channels=40,
            kernel_size=(3, 3),
            padding=1,
            stride=1,
        )
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.fc = nn.Linear(40 * 7 * 7, 10)

    def forward(self, x):
        '''
        :param x: [B, 1, 28, 28]
        :return:
        '''
        batch_size = x.shape[0]
        x = self.conv1(x) # [B, 20, 28, 28]
        x = self.pool1(x) # [B, 20, 14, 14]
        x = self.conv2(x) # [B, 40, 14, 14]
        x = self.pool2(x) # [B, 40, 7, 7]
        x = x.view(batch_size, -1) # [B, 40 * 7 * 7]
        x = self.fc(x) # [B, 10]

        return x

model = FullyConnectedModel(784, 10, 100)
# model = ConvolutionModel()

criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# train
def train(model, dataloader):
    for epoch in range(epochs):
        for step, batch in enumerate(dataloader):
            batch_x, batch_y = batch
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)

            pred_logits = model(batch_x)
            loss = criterion(pred_logits, batch_y) # pred_logits: [B, 10], batch_y: [B]

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if step % 10 == 0:
                print("epoch: {}, step: {}, loss: {:.4f}".format(epoch, step, loss.item()))

if __name__ == '__main__':
    train(model, train_loader)












































