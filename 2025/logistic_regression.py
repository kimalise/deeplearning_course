import numpy as np
import torch
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score

batch_size = 4
learning_rate = 0.001
epochs = 100
log_freq = 10

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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
        return self.x[index].to(torch.float32), self.y[index]

# test dataset
# dataset = MyDataset(x1, x2, y)
# dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
#
# for step, batch in enumerate(dataloader):
#     print(step, batch)
#     break

# 3. build model
class MyModel(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, output_dim)

    def forward(self, x):
        '''
        :param x: [B, 2]
        :return: [B]
        '''
        x = self.fc1(x) # [B, 2] x [2, 1] = [B, 1]
        x = torch.sigmoid(x) # [B, 1]
        return x

# x = torch.randn(4, 2)
# model = MyModel(2, 1)
# y = model(x)
# print(y)

# 4. loss function, optimizer
model = MyModel(2, 1)
model.to(device)

criterion = torch.nn.BCELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

def validation(model, val_dataloader):
    model.eval()

    predicted_labels = []
    golden_labels = []
    with torch.no_grad():
        for step, batch in enumerate(val_dataloader):
            batch_x, batch_y = batch
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)

            pred_y = model(batch_x) # tensor [4, 1]
            pred_y = torch.squeeze(pred_y) # tensor [4]
            pred_y = (pred_y > 0.5).to(torch.long)
            predicted_labels.extend(pred_y.cpu().numpy().tolist())
            golden_labels.extend(batch_y.cpu().numpy().tolist())

    model.train()

    f1 = f1_score(golden_labels, predicted_labels, average="macro")
    return f1


# 5. train the model
def train(model, train_dataloader, val_dataloader):
    best_f1 = -1

    for epoch in range(epochs):
        for step, batch in enumerate(train_dataloader):
            batch_x, batch_y = batch
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)

            pred_y = model(batch_x)
            pred_y = pred_y.squeeze()
            loss = criterion(pred_y, batch_y.to(torch.float32))

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if step % log_freq == 0:
                print("epoch: {}, step: {}, loss: {:.4f}".format(epoch, step, loss.item()))

        f1 = validation(model, val_dataloader)
        print("validation f1: {:.4f}".format(f1))
        if f1 > best_f1:
            torch.save(model.state_dict(), "best.pt")
            best_f1 = f1


# 6. test

if __name__ == '__main__':
    train_dataset = MyDataset(train_x1, train_x2, train_y)
    val_dataset = MyDataset(val_x1, val_x2, val_y)

    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    train(model, train_dataloader, val_dataloader)










