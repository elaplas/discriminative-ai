import torch.nn as nn

# It reduces the computation time by factor 10
class DepthwiseSeperableConv2D(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        self.depthwise_conv_block = nn.Sequential(
            ## If "groups" are set to "in_channels", in_channelsx3x3 are applied to in_channels number of input channels
            # and in_channels number of features are produced and returned as output without summing the respective 
            # feature pixels along depth, which is done in normal convolution with groups=1
            nn.Conv2d(in_channels=in_channels, out_channels=in_channels, kernel_size=3, stride=stride, padding=1, groups=in_channels),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(),
            # Pointwise convolution: normal convolution with kernel_size=1
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
        )
    def forward(self, X):
        Y = self.depthwise_conv_block(X)
        return Y
    def __call__(self, X):
        return self.forward(X)

class Mobilenet28(nn.Module):
    def __init__(self, num_classes):
        self.conv_blocks = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, stride=2, padding=1),
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

        self.fc = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Linear(1024, num_classes)
        )

        self.softmax = nn.Softmax(dim=1)

    def forward(self, X):
        Y = self.conv_blocks(X)
        Y = self.fc(Y)
        Y = self.softmax(Y)
        return Y
    def __call__(self, X):
        return self.forward(X)