import torch
import torch.nn as nn
import torch.optim as optim
import torchvision 
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader, random_split
from torchsummary import summary
from backbones.resnet.resnet32 import ResNet32
from utility.metrics_calc import convertLabelToProbability, cal_accuracy

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
learning_rate = 0.0001

transform = transforms.Compose([
    transforms.RandomRotation(15),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Resize((56, 56))
])

train_dataset = torchvision.datasets.FashionMNIST(root = "./fashionmnist_data", train=True, download = True, transform = transform)
test_dataset = torchvision.datasets.FashionMNIST(root="./fashionmnist_data", train=False, download=True, transform = transform)
print(test_dataset)
train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, num_workers=4)
test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, num_workers=4)


class Classifier(nn.Module):
    def __init__(self, num_classes, in_channels=3):
        super().__init__()

        self.num_classes = num_classes

        self.feature_extractor = ResNet32(num_classes=num_classes, in_channels=in_channels)
        self.fc = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(512, num_classes)
        )

        self.softmax = nn.Softmax(dim=1)

    def forward(self, x: torch.tensor) -> torch.tensor:
        x = self.feature_extractor(x)
        x = self.fc(x)
        x = self.softmax(x)
        return x

    def __call__(self, x):
        return self.forward(x)


# Create and check model
model = Classifier(num_classes=10, in_channels=1).to(device)
x = torch.randint(-5,5, (1,1,56,56)).float().to(device)
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
        if (i != 0 and i%5 == 0):
            for param_group in optimizer.param_groups:
                param_group['lr'] *= 0.2
    
