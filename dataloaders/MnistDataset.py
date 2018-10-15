import torch
from torch.utils import data
import numpy as np
import matplotlib.pyplot as plt
from mlxtend.data import loadlocal_mnist


class MnistDataset(data.Dataset):
    
    def __init__(self, folder_path, dataset_type='train'):
        
        self.dataset_type = dataset_type
        
        if dataset_type == 'train':
            self.x, self.y = loadlocal_mnist(
                images_path=folder_path+'/train-images.idx3-ubyte', 
                labels_path=folder_path+'/train-labels.idx1-ubyte'
            )
        else:
            self.x, self.y = loadlocal_mnist(
                images_path=folder_path+'/t10k-images.idx3-ubyte', 
                labels_path=folder_path+'/t10k-labels.idx1-ubyte'
            )
        
    def __len__(self):
        
        return self.x.shape[0]

    
    def length(self):
        
        return self.__len__()
    
    
    def __getitem__(self, index):

        return self.x[index].astype(np.int), self.y[index].astype(np.int)