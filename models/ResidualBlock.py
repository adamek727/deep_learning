import torch.nn as nn
import torch.nn.functional as F

class ResidualBlock(nn.Module):
    
    def __init__(self, channels, kernel_size, maxpooling=False):
        
        super(ResidualBlock, self).__init__()
        
        self.channels = channels
        self.kernel_size = kernel_size
        self.maxpooling = maxpooling
        self.padding = int((kernel_size-1)/2)
        
        self.conv1 = nn.Conv2d(self.channels, self.channels, kernel_size, padding=self.padding)
        self.conv2 = nn.Conv2d(self.channels, self.channels, kernel_size, padding=self.padding)
        
        self.bn1 = nn.BatchNorm2d(self.channels)
        self.bn2 = nn.BatchNorm2d(self.channels)
        
        self.relu = nn.LeakyReLU(negative_slope=0.1)
        
        self.maxpool = nn.MaxPool2d(2, 2)
        
    def forward(self, x):

        residue = x
        
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)
        
        x = x + residue
        
        if(self.maxpooling):
            return x
        else:
            return self.maxpool(x) 