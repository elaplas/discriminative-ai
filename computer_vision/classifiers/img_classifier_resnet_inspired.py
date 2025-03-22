import torch
import torch.nn as nn
import torch.optim as optim
import torchvision 
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader, random_split
from torchsummary import summary

###
### This classifier (resnet34 inspired) reaches around ... accuracy with less than ... trainable parameters
### Hint: In resnet, a block of two residual layers containing two conv3x3 with stride=n and padding =1 and 
### a shortcut is followed by another block of two residual layers and this patterns is repeated.
### The n for stride=n in residual layer is not set to 1 after each 6 residual layers to reducce 2D dimesions (hxw) by n.
### In n for stride=n is not set to 1, the first conv layer of the next residual doubles the number of output 
### channels to compensate for the reduced hxw dimensions. 
###

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Training parameters
batch_size = 128
num_epochs = 20
learning_rate = 0.01

transform = transforms.Compose([
    transforms.RandomRotation(15),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor()
])

train_dataset = torchvision.datasets.FashionMNIST(root = "./fashionmnist_data", train=True, download = True, transform = transform)
test_dataset = torchvision.datasets.FashionMNIST(root="./fashionmnist_data", train=False, download=True, transform = transform)
print(test_dataset)
train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, num_workers=4)
test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, num_workers=4)

class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.cov1 = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, stride=stride, padding=1)
        self.cov2 = nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=3, stride=1, padding=1)
        self.bn = nn.BatchNorm2d(out_channels)
        self.shortcut = nn.Sequential()
        # To addapt input either channel numbers or hxw dimenstions 
        if in_channels != out_channels or stride!=1:
            self.shortcut = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=1, stride=stride, padding=0)
        self.relu = nn.ReLU()
        

    def forward(self, x):
        y = self.cov1(x)
        y = self.cov2(y)
        y = self.shortcut(x) + y
        y = self.bn(self.relu(y))
        return y
    
    def __call__(self, x):
        return self.forward(x)


class Classifier(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.num_classes = num_classes
        self.conv_layers = nn.Sequential(
            # Block 1: 1x28x28 -> 16x28x28
            ResidualBlock(in_channels=1, out_channels=16, stride=1),
            # Block 2: 16x28x28 -> 32x28x28
            ResidualBlock(in_channels=16, out_channels=32, stride=1),
            # Block 3: 32x28x28 -> 64x28x28
            ResidualBlock(in_channels=32, out_channels=64, stride=1),
            # Block 4: 64x28x28-> 128x14x14 (Replaced max pooling with strided convolutions for learnable downsampling)
            ResidualBlock(in_channels=64, out_channels=128, stride=2)
        )

        self.fc = nn.Sequential(
            # Reduce to 128x1x1
            nn.AdaptiveAvgPool2d(1), 
            nn.Flatten(), 
            nn.Linear(128, num_classes)
        )

        self.softmax = nn.Softmax(dim=1)
    
    def forward(self, x: torch.tensor) -> torch.tensor:
        x = self.conv_layers(x)
        x = self.fc(x)
        x = self.softmax(x)
        x = x.view(x.shape[0], self.num_classes, 1)
        return x

    def __call__(self, x):
        return self.forward(x)

def convertLabelToProbability(labels):
    x = torch.zeros((labels.shape[0], 10, 1)).float()
    for i in range(labels.shape[0]):
        x[i][labels[i]] = 1.0
    return x

def cal_accuracy(x: torch.tensor, y: torch.tensor):
    res = 0
    for i in range(x.size()[0]):
        if sum(abs(x[i] - y[i])) <= 0.1:
            res += 1
    res /= float(x.size()[0])
    return res


# Create and check model
model = Classifier(10).to(device)
print(summary(model, (1,28,28)))
x = torch.randint(-5,5, (1,1,28,28)).float().to(device)
y = model(x)
print(y.shape)
print(y)

# Config training
loss_fn = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate) 

for i in range(num_epochs):
    for x_batch, y_batch in train_loader:
        y_batch = convertLabelToProbability(y_batch)
        optimizer.zero_grad()
        x_batch = x_batch.to(device)
        y_batch = y_batch.to(device)
        prediction = model(x_batch)
        loss = loss_fn(prediction, y_batch)
        loss.backward()
        optimizer.step()
        print(f"epoch: {i}, training loss: {loss}")
        print(f"epoch: {i}, training accuracy: {cal_accuracy(prediction, y_batch)}")

        # Reduce learning rate after 10 epochs 
        if (i != 0 and i%10 == 0):
            for param_group in optimizer.param_groups:
                param_group['lr'] *= 0.1
    
