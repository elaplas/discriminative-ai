import torch.nn as nn

class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.res = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, stride=stride, padding=1),
            nn.ReLU(), 
            nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=3, stride=1, padding=1),
        )
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=1, stride=stride, padding=0)
        self.relu = nn.ReLU()  
    def forward(self, X):
        Y = self.res(X)
        Y = self.shortcut(X) + Y
        Y = self.relu(Y)
        return Y
    def __call__(self, X):
        return self.forward(X)

class ResBlock3(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.res_block = nn.Sequential(
            ResidualBlock(in_channels=in_channels, out_channels=out_channels, stride=stride),
            ResidualBlock(in_channels=out_channels, out_channels=out_channels, stride=1),
            ResidualBlock(in_channels=out_channels, out_channels=out_channels, stride=1),
        )
    def forward(self, X):
        Y = self.res_block(X)
        return Y
    def __call__(self, X):
        return self.forward(X)

class ResBlock4(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.res_block = nn.Sequential(
            ResidualBlock(in_channels=in_channels, out_channels=out_channels, stride=stride),
            ResidualBlock(in_channels=out_channels, out_channels=out_channels, stride=1),
            ResidualBlock(in_channels=out_channels, out_channels=out_channels, stride=1),
            ResidualBlock(in_channels=out_channels, out_channels=out_channels, stride=1),
        )
    def forward(self, X):
        Y = self.res_block(X)
        return Y
    def __call__(self, X):
        return self.forward(X)
    
class ResBlock6(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.res_block = nn.Sequential(
            ResidualBlock(in_channels=in_channels, out_channels=out_channels, stride=stride),
            ResidualBlock(in_channels=out_channels, out_channels=out_channels, stride=1),
            ResidualBlock(in_channels=out_channels, out_channels=out_channels, stride=1),
            ResidualBlock(in_channels=out_channels, out_channels=out_channels, stride=1),
            ResidualBlock(in_channels=out_channels, out_channels=out_channels, stride=1),
            ResidualBlock(in_channels=out_channels, out_channels=out_channels, stride=1),
        )
    def forward(self, X):
        Y = self.res_block(X)
        return Y
    def __call__(self, X):
        return self.forward(X)

class ResNet32(nn.Module):
    def __init__(self, num_classes, in_channels=3):
        super().__init__()
        self.res_blocks = nn.Sequential(
            ResBlock3(in_channels=in_channels, out_channels=64, stride=1),
            ResBlock4(in_channels=64, out_channels=128, stride=2),
            ResBlock6(in_channels=128, out_channels=256, stride=2),
            ResBlock3(in_channels=256, out_channels=512, stride=2)
        )
        self.maxpool2d = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)

    def forward(self, X):
        Y =  self.maxpool2d(X)
        Y = self.res_blocks(Y)
        
        return Y
    def __call__(self, X):
        return self.forward(X)   