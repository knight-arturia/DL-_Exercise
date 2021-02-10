from torch.utils.data import Dataset
import torch
import pandas as pd
from pathlib import Path
from skimage.io import imread
from skimage.color import gray2rgb
import numpy as np
import torchvision as tv
from torchvision.transforms.transforms import Normalize, ToPILImage, ToTensor

train_mean = [0.59685254, 0.59685254, 0.59685254]
train_std = [0.16043035, 0.16043035, 0.16043035]

'''
data is a pandas.dataframe
mode is a string, value is 'val' or 'train'
'''
class ChallengeDataset(Dataset):
    
    # constructor
    def __init__(self, data, mode):
        
        super(ChallengeDataset, self).__init__()
        
        self.data_frame = data
        self.flag = mode
        # set different transform strategies for training and validation
        if mode == 'train':
            self._transform = tv.transforms.Compose([
                ToPILImage(),
                ToTensor(),
                Normalize(train_mean, train_std)
            ])
        elif mode == 'val':
            self._transform = tv.transforms.Compose([
                ToPILImage(),
                ToTensor(),
                Normalize(train_mean, train_std)
            ])
    
    def __len__(self):
        return len(self.data_frame)
    
    def __getitem__(self, index):
        
        # if index is tensor type, change to list 
        if torch.is_tensor(index):
            index = index.tolist()

        # get image name from first col in data_frame
        img_name = self.data_frame.iloc[index,0]
        image = imread(img_name)
        image = gray2rgb(image)
        
        # print('image name = ', format(img_name))

        # feature is all cols except first one
        labels = self.data_frame.iloc[index, 1:]
        labels = np.array(labels, dtype = int)
        labels = torch.from_numpy(labels).float()

        # print('labels = ', labels)
        # print('type = ', labels.dtype)

        if self._transform:
            image = self._transform(image)
            # labels = self._transform(labels)
        
        sample = (image, labels)
        
        return sample 
