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


class Flowers17Dataset(data.Dataset):
    
    def __init__(self, folder_path, dataset_type='train',device='cpu', image_size=100):
        
        super(Flowers17Dataset, self).__init__()
        
        self.folder_path = folder_path
        self.images = []
        self.labels = []
        self.images_per_class = 80
        self.image_channels = 3
        self.image_side_size = image_size
        self.dataset_type = dataset_type
        
        files = glob.glob(folder_path+'*.jpg')
        files.sort(key=lambda x: x.lower())
        
        for i, image_path in enumerate(files):
            
            image_type = 'train'
            if (i % 10 == 8):
                image_type = 'val'
            if (i % 10 == 9):
                image_type = 'test'
                
                
            if dataset_type == image_type:
                self.images.append(image_path)
                self.labels.append(int(i/self.images_per_class))    
        
        self.rotation = False
        self.crop = False
        self.flip = False
        self.blur = False
        self.noise = False
        
        self.mean = 0.5
        self.std_dev = 0.1
        self.normalization = False
        
    def __len__(self):
        
        return len(self.images)

    
    def length(self):
        
        return len(self.images)
    
    
    def __getitem__(self, index):
        
        img = scipy.misc.imread(self.images[index])
        
        if img.shape[0] > img.shape[1]:
            img = np.rot90(img)
            
        img = scipy.misc.imresize(img, (self.image_side_size,self.image_side_size))
        
        if(self.normalization):
            img = self.normalize(img)
            
        img = self.augment(img)
        
        img = np.transpose(img, (2, 1, 0))
        return img.astype(np.float32)/255, self.labels[index]
    
    
    def getNameForLabel(self, label):
        
        return self.label_names[label]
    
    
    def augment(self, image):
        
        if( random.random() > 0.5 and self.rotation):
            angle = random.random()*360
            image = scipy.ndimage.interpolation.rotate(image, angle,reshape=False, order=0,mode='reflect')
        
        if( random.random() > 0.5 and self.crop):
            
            cropFactor = int(random.random()*5+1)
            image = image[cropFactor:-cropFactor,cropFactor:-cropFactor]
            image = scipy.misc.imresize(image, (self.image_side_size, self.image_side_size))
        
        if( random.random() > 0.5 and self.flip):
            if( random.random() > 0.5 ):
                image = np.flip(image,0).copy()
            else:
                image = np.flip(image,1).copy()
        
        if( random.random() > 0.5 and self.blur):
            image = scipy.ndimage.gaussian_filter(image, sigma=0.25)
        
        if( random.random() > 0.5 and self.noise):
            
            r = np.random.normal(0,5,image.shape).astype(image.dtype)
            image += r
            image = np.clip(image, 0, 255)
            
        return image
    
    
    def normalize(self, image):
        
        image = image.astype(np.float32)/255
        
        
        for i in range(self.image_channels):
            image[:,:,i] *= self.std_dev/np.std(image[:,:,i])
            image[:,:,i] -= np.ones((self.image_side_size, self.image_side_size)) * np.mean(image[:,:,i])-self.mean
        
        return (image*255).astype(np.uint8)
    
    
    def setRotation(self, enable):
        self.rotation = enable
        
        
    def setCrop(self, enable):
        self.crop = enable
        
        
    def setFlip(self, enable):
        self.flip = enable
        
        
    def setBlur(self, enable):
        self.blur = enable
        
        
    def setNoise(self, enable):
        self.noise = enable
        
        
    def setNormalization(self, enable, mean, std_dev):
        self.normalization = enable
        self.mean = 0.5
        self.std_dev = 0.1
        
    def plotImage(self, image, label):
                             
        plt.imshow( np.transpose(image[:,:,:].numpy(), (1, 2, 0)))
        plt.show()
        print('label: ',label)