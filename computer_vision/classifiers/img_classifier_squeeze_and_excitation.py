import torch
import torch.nn as nn
import torch.optim as optim
import torchvision 
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader, random_split
from torchsummary import summary

###
### This classifier (vgg16 inspired) with squeeze-and-excitation layer (channel attention)
### Important channels are amplified, less useful ones suppressed 
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

class ChannelAttention(nn.Module):
    """
    Input: Feature map tensor of shape (B, C, H, W)
    Output: Attention-weighted feature map of same dimensions
    Squeeze Phase (Global Information)
    Uses AdaptiveAvgPool2d(1) to reduce spatial dimensions to 1x1:
    (B, C, H, W) → (B, C, 1, 1)
    Captures global context by averaging across spatial dimensions
    Excitation Phase (Channel Relationships)
    Squeezed features are fed through two fully connected layers:
    First FC reduces dimensionality by reduction ratio (e.g., 16:1)
    Second FC restores original channel count
    Sigmoid activation produces channel-wise attention weights between 0-1
    Feature Recalibration
    Original feature map is multiplied by attention weights:
    x * y.expand_as(x)
    Important channels are amplified, less useful ones suppressed, so
    it can be used to select relevant channels for each head/task in multi-task networks
    """
    def __init__(self, channels, reduction=2):
        super().__init__()
        self.fc = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(), 
            nn.Linear(channels, channels//reduction),
            nn.Linear(channels//reduction, channels),
            nn.Sigmoid()
        )

    def forward(self, x):
        y = self.fc(x)
        x = x * y.view(y.size()[0], y.size()[1],1,1).expand_as(x)
        return x
    
    def __call__(self, x):
        return self.forward(x)


class Classifier(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.num_classes = num_classes
        self.conv_layers = nn.Sequential(
            # Block 1: 1x28x28 ->32x28x28
            nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.1),
            # Block 2: 32x28x28 -> 32x28x28
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.1),
            # Block 3: 32x28x28 -> 64x14x14
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.1),
            ChannelAttention(64),
            # Block 4: 64x14x14-> 64x14x14
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.1),
            # Block 5: 64x14x14-> 128x7x7
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0),
            nn.BatchNorm2d(128)
        )

        self.fc = nn.Sequential(
            # Reduce to 512x1x1
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
    
