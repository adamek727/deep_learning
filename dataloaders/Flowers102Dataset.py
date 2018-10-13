import torch
from torch.utils import data
import numpy as np
import matplotlib.pyplot as plt
import scipy
import scipy.ndimage
import scipy.io
import skimage
import math
import random
import glob
import dataloaders.Flowers17Dataset as fl17

class Flowers102Dataset(fl17.Flowers17Dataset):
    
    def __init__(self, folder_path, dataset_type='train',device='cpu', image_size=100):
        
        super(Flowers102Dataset, self).__init__(folder_path, dataset_type,device, image_size)
        
        self.folder_path = folder_path
        self.images = []
        self.labels = []
        self.images_per_class = 80
        self.image_channels = 3
        self.image_side_size = image_size
        self.dataset_type = dataset_type
        
        files = glob.glob(folder_path+'imgs/'+'*.jpg')
        files.sort(key=lambda x: x.lower())
        
        labels = scipy.io.loadmat(folder_path+'imagelabels.mat')['labels']
        labels = labels.reshape(labels.shape[1])
       
        
        for i, image_path in enumerate(files):
            
            image_type = 'train'
            if (i % 10 == 8):
                image_type = 'val'
            if (i % 10 == 9):
                image_type = 'test'
                
                
            if dataset_type == image_type:
                self.images.append(image_path)
                self.labels.append(labels[i]-1) 
             
            
                
        self.rotation = False
        self.crop = False
        self.flip = False
        self.blur = False
        self.noise = False
        
        self.mean = 0.5
        self.std_dev = 0.1
        self.normalization = False