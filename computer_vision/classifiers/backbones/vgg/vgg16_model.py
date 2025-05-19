import torch.nn as nn

class ConvBlock2(nn.Module):
    def __init__(self, in_channels, out_channels):
        self.conv_block = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=3, stride=1, padding=1),
            nn.MaxPool2d(stride=2, padding=0)
        )
    def forward(self, X):
        X = self.conv_block(X)
    def __call__(self, X):
        return self.forward(X)
    
class ConvBlock3(nn.Module):
    def __init__(self, in_channels, out_channels):
        self.conv_block = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=3, stride=1, padding=1),
            nn.MaxPool2d(stride=2, padding=0)
        )
    def forward(self, X):
        X = self.conv_block(X)
    def __call__(self, X):
        return self.forward(X)

class VGG16(nn.Mdule):
    def __init__(self, num_classes):
        self.conv_blocks = nn.Sequential(
            ConvBlock2(in_channels=3, out_channels=64),
            ConvBlock2(in_channels=64, out_channels=128),
            ConvBlock3(in_channels=128, out_channels=256),
            ConvBlock3(in_channels=256, out_channels=512),
            ConvBlock3(in_channels=512, out_channels=512),
        )

        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(4096, 4096),
            nn.Linear(4096, 4096),
            nn.Linear(4096, num_classes),
        )
        self.sofmax = nn.Softmax(dim=1)
    def forward(self, X):
        Y = self.conv_blocks(X)
        Y = self.fc(Y)
        Y = self.sofmax(Y)
        return Y
    def __call__(self, X):
        return self.forward(X)