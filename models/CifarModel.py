import torch
import torch.nn as nn
import torch.nn.functional as F

from . import ResidualBlock as rb

class CifarModel(nn.Module):
    
    def __init__(self, model_name, num_of_outputs=2 ):
        
        super(CifarModel, self).__init__()
        
        self.num_of_outputs = num_of_outputs
        self.model_name = model_name
        
        self.conv1 = nn.Conv2d(3, 16, 5, padding=2)
        
        self.block1 = rb.ResidualBlock(16,3,False) # out: 32x32x16
        self.block2 = rb.ResidualBlock(16,3,True)  # out: 16x16x16
        
        self.bottleneck1 = nn.Conv2d(16,32,1)
        
        self.block3 = rb.ResidualBlock(32,3,False) # out: 16x16x32
        self.block4 = rb.ResidualBlock(32,3,True)  # out: 8x8x32
        
        self.bottleneck2 = nn.Conv2d(32,64,1)
        
        self.block5 = rb.ResidualBlock(64,3,False) # out: 8x8x64
        self.block6 = rb.ResidualBlock(64,3,True)  # out: 4x4x64
        
        self.fc1 = nn.Linear(4*4*64, 200)
        self.fc2 = nn.Linear(200, self.num_of_outputs)

    def forward(self, x):
        
        x = self.conv1(x)
        x = self.block1(x)
        x = self.block2(x)
        
        x = self.bottleneck1(x)
        x = self.block3(x)
        x = self.block4(x)
        
        x = self.bottleneck2(x)
        x = self.block5(x)
        x = self.block6(x)
        
        x = x.view(-1, 4*4*64)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        
        return x
    
    def save_model(self, state, is_best, directory='models/',filename='checkpoint.pth.tar'):
        torch.save(state, directory+self.model_name+'_'+filename)
        if is_best:
            torch.save(state, directory+self.model_name+'_'+'model_best.pth.tar')