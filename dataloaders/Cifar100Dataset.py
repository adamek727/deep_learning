import torch
from torch.utils import data
import numpy as np
import matplotlib.pyplot as plt
import scipy
import scipy.ndimage
import skimage
import math
import random
import pickle
import dataloaders.Cifar10Dataset as c10

class Cifar100Dataset(c10.Cifar10Dataset):
    
    def __init__(self, folder_path, dataset_type='train',device='cpu'):
        
        self.folder_path = folder_path
        self.images = []
        self.labels = []
        
        self.images_per_batch = 50000
        
        if dataset_type == 'train':
            self.all_files = ['train']
        if dataset_type == 'val' or dataset_type == 'test':
            self.all_files = ['test']
            self.images_per_batch = 10000
        
        self.metadata = self.__unpickle__(folder_path+'meta')
        self.label_names = self.metadata[b'fine_label_names']
        self.image_channels = 3
        self.channel_size = 1024 
        self.image_side_size = 32
        self.dataset_type = dataset_type
        
        self.rotation = False
        self.crop = False
        self.flip = False
        self.blur = False
        self.noise = False
        
        self.mean = 0.5
        self.std_dev = 0.1
        self.normalization = False
        
        for file in self.all_files:
            
            binary_data = self.__unpickle__(folder_path+file)
            images = binary_data[b'data']
            labels = binary_data[b'fine_labels']
            
            for i in range(self.images_per_batch):
                image = np.zeros((self.image_side_size, self.image_side_size, self.image_channels), dtype=np.uint8)

                for j in range(self.image_channels):
                    image[:,:,j] = images[i, j*self.channel_size:(j+1)*self.channel_size].reshape(self.image_side_size, self.image_side_size)
                
                self.images.append(np.rot90(image))
                self.labels.append(labels[i])
                