import torch
import torch.nn as nn
import torch.nn.functional as F

from . import ResidualBlock as rb

class FlowerModel(nn.Module):
    
    def __init__(self, model_name, num_of_outputs=17 ):
        
        super(FlowerModel, self).__init__()
        
        self.num_of_outputs = num_of_outputs
        self.model_name = model_name
        
        self.conv1 = nn.Conv2d(3, 16, 5, padding=2)
        
        self.block1 = rb.ResidualBlock(16,3) # out: 100x100x16
        self.block2 = rb.ResidualBlock(16,3) # out: 100x100x16
        self.block3 = rb.ResidualBlock(16,3,maxpooling=True)  # out: 50x50x16
        
        self.bottleneck1 = nn.Conv2d(16,32,1)
        
        self.block4 = rb.ResidualBlock(32,3) # out: 50x50x32
        self.block5 = rb.ResidualBlock(32,3) # out: 50x50x32
        self.block6 = rb.ResidualBlock(32,3,maxpooling=True)  # out: 25x25x32
        
        self.bottleneck2 = nn.Conv2d(32,64,1)
        
        self.block7 = rb.ResidualBlock(64,3) # out: 25x25x64
        self.block8 = rb.ResidualBlock(64,3) # out: 25x25x64
        self.block9 = rb.ResidualBlock(64,3,maxpooling=True)  # out: 12x12x64
        
        self.bottleneck3 = nn.Conv2d(64,128,1)
        
        self.block10 = rb.ResidualBlock(128,3) # out: 12x12x128
        self.block11 = rb.ResidualBlock(128,3) # out: 12x12x128
        self.block12 = rb.ResidualBlock(128,3,maxpooling=True)  # out: 6x6x128
        
        self.bottleneck4 = nn.Conv2d(128,32,1)
        
        self.fc1 = nn.Linear(6*6*32, 200)
        self.fc2 = nn.Linear(200, self.num_of_outputs)

    def forward(self, x):
        
        x = self.conv1(x)
        
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        
        x = self.bottleneck1(x)
        
        x = self.block4(x)
        x = self.block5(x)
        x = self.block6(x)
        
        x = self.bottleneck2(x)
        
        x = self.block7(x)
        x = self.block8(x)
        x = self.block9(x)
        
        x = self.bottleneck3(x)
        
        x = self.block10(x)
        x = self.block11(x)
        x = self.block12(x)
        
        x = self.bottleneck4(x)
        
        x = x.view(-1, 6*6*32)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        
        return x
    
    def save_model(self, state, is_best, directory='models/',filename='checkpoint.pth.tar'):
        torch.save(state, directory+self.model_name+'_'+filename)
        if is_best:
            torch.save(state, directory+self.model_name+'_'+'model_best.pth.tar')