import torch.nn as nn

class ResidualBlock(nn.Module):
    """it implements the residual block introduced in paper "deep residual learning for image recognition".
       The input tensor passed to a block of two conv layers is added to the output of the block. The addition
       operation will forward the gradient it recieves from the parent node to the child node in the backpropogation, 
       which helps the gradient flow in deep architectures.
       If there is mismatch between the resolution of the input and the output of the block, it is handeled by striding 
       in the convolution oprtation. If there is mismatch between the number of 
       channels of the input and the number of channels of the output block, it is handeled by point-wise convolutation (
       convolution with kernel size one).  


    Args:
        nn (nn.Module): Helper type provided by pytorch for implementing a neural network model
    """
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
    """ A stack of three residual layers

    Args:
        nn (nn.Module): Helper type provided by pytorch for implementing a neural network model
    """
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
    """A stack of four residual layers

    Args:
        nn (nn.Module): Helper type provided by pytorch for implementing a neural network model
    """
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
    """A stack of six residual layers

    Args:
        nn (nn.Module): Helper type provided by pytorch for implementing a neural network model
    """
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
    """It implements resnet34 exculding the last two layers to be used as a backbone or a feature extractor.
       The n for stride=n in residual blocks is not set to 1 after each stack of residual blocks to reducce 2D dimesions 
       (hxw) by n (learnable down sampling). The residual blocks double the number of output channels to compensate for
        the reduced hxw dimensions.  

    Args:
        nn (nn.Module): Helper type provided by pytorch for implementing a neural network model
    """
    def __init__(self, in_channels=3):
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