import torch.nn as nn

class ConvBlock2(nn.Module):
    """A stack of two conv layers according to paper "very deep convolutional networks for large scale image recognition"

    Args:
        nn (nn.Module): Helper type provided by pytorch for implementing a neural network model
    """
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv_block = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=3, stride=1, padding=1),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        )
    def forward(self, X):
        X = self.conv_block(X)
        return X
    def __call__(self, X):
        return self.forward(X)
    
class ConvBlock3(nn.Module):
    """A stack of three conv layers according to paper "very deep convolutional networks for large scale image recognition"

    Args:
        nn (nn.Module): Helper type provided by pytorch for implementing a neural network model
    """
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv_block = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=3, stride=1, padding=1),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        )
    def forward(self, X):
        X = self.conv_block(X)
        return X
    def __call__(self, X):
        return self.forward(X)

class VGG13(nn.Module):
    """It implements vgg16 exculding the last three layers to be used as a backbone or a feature extractor.
       The conv blocks double the number of output channels to compensate for
       the reduced hxw dimensions by maxpooling.  

    Args:
        nn (nn.Module): Helper type provided by pytorch for implementing a neural network model
    """
    def __init__(self, in_channels=3):
        super().__init__()
        self.conv_blocks = nn.Sequential(
            ConvBlock2(in_channels=in_channels, out_channels=64),
            ConvBlock2(in_channels=64, out_channels=128),
            ConvBlock3(in_channels=128, out_channels=256),
            ConvBlock3(in_channels=256, out_channels=512),
            ConvBlock3(in_channels=512, out_channels=512),
        )

    def forward(self, X):
        Y = self.conv_blocks(X)
        return Y
    def __call__(self, X):
        return self.forward(X)