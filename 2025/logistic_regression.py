import numpy as np
import torch
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader

batch_size = 4
learning_rate = 0.001

# 1.prepare data, preprocessing data
x1 = np.repeat([1, 4], 100)
x2 = np.repeat([1, 4], 100)
# x1 = np.array([1] * 100 + [4] * 100)
# print(x1)
y = np.repeat([0, 1], 100)

x1 = x1 + np.random.randn(len(x1)) * 1.0
x2 = x2 + np.random.randn(len(x2)) * 1.0

def split_data(x1, x2, y, train_ratio=0.8):
    index = np.arange(0, len(x1))
    np.random.shuffle(index)
    x1 = x1[index]
    x2 = x2[index]
    y = y[index]

    train_count = int(len(x1) * train_ratio)
    train_x1 = x1[:train_count]
    train_x2 = x2[:train_count]
    train_y = y[:train_count]

    val_x1 = x1[train_count:]
    val_x2 = x2[train_count:]
    val_y = y[train_count:]

    return train_x1, train_x2, train_y, val_x1, val_x2, val_y


train_x1, train_x2, train_y, val_x1, val_x2, val_y = split_data(x1, x2, y, train_ratio=0.8)
print()

# plt.scatter(x1, x2, c=y)
# plt.show()

# 9093452087
# 2. dataset
class MyDataset(Dataset):
    def __init__(self, x1, x2, y):
        super(MyDataset, self).__init__()
        # x1: np[200], x2: np[200], y: np[200]
        x1 = torch.tensor(x1) # tensor[200]
        x2 = torch.tensor(x2) # tensor[200]
        self.y = torch.tensor(y) # tensor[200]
        # [200] [200] -> [200, 2]
        self.x = torch.stack([x1, x2], dim=1)

    def __len__(self):
        # return self.x.size()[0]
        return self.x.shape[0] # self.x.shape: [200, 2]

    # callback
    def __getitem__(self, index):
        return self.x[index], self.y[index]

# test dataset
dataset = MyDataset(x1, x2, y)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

for step, batch in enumerate(dataloader):
    print(step, batch)
    break

# 3. build model
# 4. loss function, optimizer
# 5. train the model
# 6. test











