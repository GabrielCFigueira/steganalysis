import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor, Lambda
import pandas as pd
from torch.utils.data import Dataset
import math
import torch.nn.functional as F

def fileName(string):
    split = string.split("_")
    return split[0], int(split[1][1:])

def normalization(t, m, s):
    return (t - m) / s

class NeuralNetwork(nn.Module):

    def __init__(self, n_frames, n_features):
        super(NeuralNetwork, self).__init__()
        self.pre1 = nn.Linear(n_features, n_features)
        self.pre2 = nn.Linear(n_features, n_features)
        self.lstm = nn.LSTM(input_size=n_features, hidden_size=n_features * 2, num_layers=3, batch_first=True)
        self.fc1 = nn.Linear(n_features * 2 * n_frames, n_features * n_frames)
        self.fc2 = nn.Linear(n_features * n_frames, 1)

    def forward(self, x):
        x = self.pre1(x)
        x = self.pre2(x)
        batch_size = x.size()[0]
        out = self.lstm(x)
        x = out[0].contiguous()
        x = x.view(batch_size, -1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        return x

class StegoDataset(Dataset):
    def __init__(self, features, labels, transform=None, target_transform=None):
        self.features = features
        self.labels = labels
        
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        feature = self.features[idx]
        if self.transform:
            feature = self.transform(feature, 0.5, 0.01)
        label = self.labels[idx].unsqueeze(0)
        return feature, label


def train_loop(dataloader, model, loss_fn, optimizer):
    model.train()
    size = len(dataloader.dataset)
    for batch, (X, y) in enumerate(dataloader):
        # Compute prediction and loss
        pred = model(X)
        loss = loss_fn(pred, y)

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if batch % 2 == 0:
            loss, current = loss.item(), batch * len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")


def test_loop(dataloader, model, loss_fn):
    model.eval()
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    test_loss, correct = 0, 0

    with torch.no_grad():
        for X, y in dataloader:
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            correct += (torch.nn.functional.sigmoid(pred) > 0.5).type(torch.float).sum().item()

    test_loss /= num_batches
    correct /= size
    print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")

csv = pd.read_csv("vid-features.csv")
file_name, _ = fileName(csv['file_name'][0])
current_file_name = file_name

n_videos = 1
n_frames = 300 #TODO improve this
oldI = 300
for idx in range(len(csv.index)):
    current_file_name, i = fileName(csv['file_name'][idx])
    
    if current_file_name != file_name:
        n_videos += 1
        file_name = current_file_name
        if oldI < n_frames:
            n_frames = oldI

    oldI = i
    

print("n_videos: " + str(n_videos))
print("n_frames: " + str(n_frames))

n_columns = len(csv.iloc[0]) - 2
        
features = torch.empty((n_videos, n_frames, n_columns))
labels = torch.empty((n_videos,))

file_name, _ = fileName(csv['file_name'][0])
current_file_name = file_name

labels[0] = csv['class'][0]

video_id = 0
for idx in range(len(csv.index)):
    current_file_name, i = fileName(csv['file_name'][idx])

    if current_file_name != file_name:
        video_id += 1
        labels[video_id] = csv['class'][idx]
        file_name = current_file_name
    elif i > n_frames:
        continue

    frame = torch.tensor(csv.loc[idx].drop(['file_name', 'class']))
    
    features[video_id][i - 1] = frame

labels = labels.type(torch.FloatTensor)
test_frac = 0.2

lengths = [math.floor(len(features) * (1 - test_frac)), math.ceil(len(features) * test_frac)]

training_features, testing_features = torch.utils.data.random_split(features, lengths, generator=torch.Generator().manual_seed(52))
training_labels, testing_labels = torch.utils.data.random_split(labels, lengths, generator=torch.Generator().manual_seed(52))

training_data = StegoDataset(training_features, training_labels)
test_data = StegoDataset(training_features, training_labels)

model = NeuralNetwork(n_frames, n_columns)

train_dataloader = DataLoader(training_data, batch_size=8, shuffle=True)
test_dataloader = DataLoader(test_data, batch_size=8, shuffle=True)
learning_rate = 1e-3
loss_fn = nn.BCEWithLogitsLoss()
#loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

epochs = 40
for t in range(epochs):
    print(f"Epoch {t+1}\n-------------------------------")
    train_loop(train_dataloader, model, loss_fn, optimizer)
    test_loop(test_dataloader, model, loss_fn)
    print("Done!")
