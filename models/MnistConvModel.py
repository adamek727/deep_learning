import torch
import torch.nn as nn
import torch.nn.functional as F

from . import MnistFcModel as mm

class MnistConvModel(mm.MnistFcModel):
    
    def __init__(self, model_name, num_of_outputs=10 ):
        
        super(MnistConvModel, self).__init__(model_name)
        
        self.num_of_outputs = num_of_outputs
        self.model_name = model_name
 
        self.conv1 = nn.Conv2d(1, 16, 3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, 3, padding=1)
        self.conv3 = nn.Conv2d(32, 64, 3, padding=1)
        self.conv4 = nn.Conv2d(64, 128, 3, padding=1)

        self.maxpool = nn.MaxPool2d(2, 2)
        
        self.fc1 = nn.Linear(3*3*128, self.num_of_outputs)
        self.fc2 = None
        self.fc3 = None

    def forward(self, x):
           
        x = x.float()        #28x28x1
        
        x = self.conv1(x)    #28x28x16
        x = self.maxpool(x)  #14x14x16
        x = self.conv2(x)    #14x14x32
        x = self.maxpool(x)  #7x7x32
        x = self.conv3(x)    #7x7x64
        x = self.maxpool(x)  #3x3x64
        x = self.conv4(x)    #3x3x128
        
        x = x.view(-1, 3*3*128)
        x = F.sigmoid(self.fc1(x))
        
        return x