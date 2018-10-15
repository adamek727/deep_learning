import torch
import torch.nn as nn
import torch.nn.functional as F

class MnistFcModel(nn.Module):
    
    def __init__(self, model_name, num_of_outputs=10 ):
        
        super(MnistFcModel, self).__init__()
        
        self.num_of_outputs = num_of_outputs
        self.model_name = model_name
 
        self.fc1 = nn.Linear(28*28, 500)
        self.fc2 = nn.Linear(500, 200)
        self.fc3 = nn.Linear(200, num_of_outputs)

    def forward(self, x):
           
        x = x.float()
        x = F.sigmoid(self.fc1(x))
        x = F.sigmoid(self.fc2(x))
        x = F.sigmoid(self.fc3(x))
        
        return x
    
    def save_model(self, state, is_best, directory='checkpoints/',filename='checkpoint.pth.tar'):
        torch.save(state, directory+self.model_name+'_'+filename)
        if is_best:
            torch.save(state, directory+self.model_name+'_'+'model_best.pth.tar')
            
    def load_model(self, load_best=False, directory='checkpoints/', filename='checkpoint.pth.tar'):
        
        path = directory + self.model_name + '_' + filename
        if load_best == True:
            path = directory + self.model_name + '_' + 'model_best.pth.tar'
        checkpoint = torch.load(path)
        self.load_state_dict(checkpoint['state_dict'])