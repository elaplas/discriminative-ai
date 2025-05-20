import torch.nn as nn

# It reduces the computation time by factor 10
class DepthwiseSeperableConv2D(nn.Module):
    """It implements the depthwise seperable convolution introduced in the paper "MobileNets: efficient convolution
       neural networks for mobile vision application". In a common convolution operation, to generate one output channel, 
       a kernel with shape kxkxin_channels (3d kernel) is applided to in_channels number of channels, which results in 
       one output channel, the ith kxk kernel (2d kernel) is applied to ith channel to generate ith feature map; then the 
       feature maps are summped up along depth yielding only one output channel. This process is repeated as many as
       out_channels to generate out_channels number of output channels.  

       In depthwise seperable operation, the ith kxk kernel (2d kernel) is applied to ith channel to generate ith feature map, 
       which results in in_channels number of feature maps. Then a 3d kernel with shape in_channelsx1x1 is applied to 
       in_channels number of channels. In this way, we can reduce the computation cost by a factor of 10 times.    

    Args:
        nn (nn.Module): Helper type provided by pytorch for implementing a neural network model
    """
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.depthwise_conv_block = nn.Sequential(
            ## If "groups" are set to "in_channels", in_channelsx3x3 number of kernels are applied to in_channels 
            # number of input channels and in_channels number of features are produced and returned as output without 
            # summing the respective feature pixels along depth, which is done in normal convolution with groups=1
            nn.Conv2d(in_channels=in_channels, out_channels=in_channels, kernel_size=3, stride=stride, padding=1, groups=in_channels),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(),
            # Pointwise convolution: normal convolution with kernel_size=1 to extend/shrik the number of 
            # channels of intermediate results
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
        )
    def forward(self, X):
        Y = self.depthwise_conv_block(X)
        return Y
    def __call__(self, X):
        return self.forward(X)

class Mobilenet26(nn.Module):
    """It implements mobilenet28 exculding the last two layers to be used as a backbone or a feature extractor.
       The n for stride=n in some depthwise conv blocks is not set to 1 to reducce 2D dimesions 
       (hxw) by n (learnable down sampling). The depthwise conv blocks double the number of output channels to compensate for
        the reduced hxw dimensions.   

    Args:
        nn (nn.Module): Helper type provided by pytorch for implementing a neural network model
    """
    def __init__(self, in_channels=3):
        super().__init__()
        self.conv_blocks = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=64, kernel_size=3, stride=2, padding=1),
            DepthwiseSeperableConv2D(in_channels=64, out_channels=64, stride=1),
            DepthwiseSeperableConv2D(in_channels=64, out_channels=128, stride=2),
            DepthwiseSeperableConv2D(in_channels=128, out_channels=128, stride=1),
            DepthwiseSeperableConv2D(in_channels=128, out_channels=256, stride=2),
            DepthwiseSeperableConv2D(in_channels=256, out_channels=256, stride=1),
            DepthwiseSeperableConv2D(in_channels=256, out_channels=512, stride=2),

            DepthwiseSeperableConv2D(in_channels=512, out_channels=512, stride=1),
            DepthwiseSeperableConv2D(in_channels=512, out_channels=512, stride=1),
            DepthwiseSeperableConv2D(in_channels=512, out_channels=512, stride=1),
            DepthwiseSeperableConv2D(in_channels=512, out_channels=512, stride=1),
            DepthwiseSeperableConv2D(in_channels=512, out_channels=512, stride=1),

            DepthwiseSeperableConv2D(in_channels=512, out_channels=1024, stride=2),
            DepthwiseSeperableConv2D(in_channels=1024, out_channels=1024, stride=2),
        )

    def forward(self, X):
        Y = self.conv_blocks(X)
        return Y
    
    def __call__(self, X):
        return self.forward(X)