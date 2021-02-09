import torch
from torch import Tensor
import torch.nn as nn

'''
basic block of resnet, consist of:
    Conv2d
    BatchNorm
    Relu

in_channels is the out_channels of previous layer/block
'''
# 1*1 convolution
def conv1x1(in_channels, out_channels,  stride=1) -> nn.Conv2d:
    return nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride)

# 3*3 convolution
def conv3x3(in_channels, out_channels,  stride=1) -> nn.Conv2d:
    return nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride)


class ResBlock(nn.Module):

    def __init__(self, in_channels, out_channels, stride):
        super(ResBlock, self).__init__()
        
        self.conv1 = conv3x3(in_channels, out_channels,  stride)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv1x1(out_channels, out_channels)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        self.stride = stride

    def forward(self, x) ->Tensor:
        identity = x

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        # use a 1*1 conv for downsampling 
        x = self.conv2(x)
        x = self.bn2(x)

        # add the identity
        x += identity
        x = self.relu(x)

        return x

class ResNet(nn.Module):
    
    def __init__(self):
        super(ResNet, self).__init__()

        # init the in_channels for ResBlocks
        self.channels = 64

        # input of the resnet is 3 channels
        self.conv1 = nn.Conv2d(3, self.channels, kernel_size=7, stride=2)
        self.bn1 = nn.BatchNorm2d(self.channels)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2)

        self.block1 = ResBlock(self.channels, 64, 1)
        self.block2 = ResBlock(64, 128, 2)
        self.block3 = ResBlock(128, 256, 2)
        self.block4 = ResBlock(256, 512, 2)
        # global max pooling, output is N*C*1*1
        self.avgpool = nn.AdaptiveAvgPool2d((1,1))

        # flatten change shape to N*C
        self.flatten = nn.Flatten()
        self.fc = nn.Linear(512, 2)
        self.sigm = nn.Sigmoid()

    def forward(self, x) -> Tensor:
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.block1.forward(x)
        x = self.block2.forward(x)
        x = self.block3.forward(x)
        x = self.block4.forward(x)

        x = self.avgpool(x)
        x = self.flatten(x)
        x = self.fc(x)
        x = self.sigm(x)

        return x


