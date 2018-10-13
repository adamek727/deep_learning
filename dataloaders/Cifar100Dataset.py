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


class Cifar100Dataset(data.Dataset):
    
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
        
    def __len__(self):
        
        return len(self.images)

    
    def length(self):
        
        return len(self.images)
    
    
    def __getitem__(self, index):
        
        img = self.images[index].copy()
        
        if(self.normalization):
            img = self.normalize(img)
            
        img = self.augment(img)
        
        img = np.transpose(img, (2, 1, 0))
        return img.astype(np.float32)/255, self.labels[index]
    
    
    def __unpickle__(self, file):
        
        with open(file, 'rb') as fo:
            dict = pickle.load(fo, encoding='bytes')
        return dict
    
    
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
        print(self.getNameForLabel(label))