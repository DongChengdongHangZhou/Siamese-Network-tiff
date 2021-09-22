import os
import random
import tifffile as tiff
from torch.utils.data import Dataset
import torch


class SiameseNetworkDataset(Dataset):
    def __init__(self,imageFolderDataset,transform=None):
        self.imageFolderDataset = imageFolderDataset    
        self.transform = transform
        
    def __getitem__(self,index):
        anchor_label = random.choice(os.listdir(self.imageFolderDataset.root))
        positive_dir = self.imageFolderDataset.root + '/' + anchor_label
        anchor_dir = positive_dir + '/' + random.choice(os.listdir(positive_dir))
        positive_dir = positive_dir + '/' + random.choice(os.listdir(positive_dir))
        while True:
            #keep looping till a different class image is found
            negative_tuple = random.choice(self.imageFolderDataset.imgs)
            if str(negative_tuple[1]) !=anchor_label:
                break

        anchor = tiff.imread(anchor_dir)
        positive = tiff.imread(positive_dir)
        negative = tiff.imread(negative_tuple[0])
        if self.transform is not None:
            anchor = self.transform(anchor)
            positive = self.transform(positive)
            negative = self.transform(negative)
            anchor = torch.reshape(anchor,(2,8,8))
            positive = torch.reshape(positive,(2,8,8))
            negative = torch.reshape(negative,(2,8,8))
            anchor = anchor.type(torch.FloatTensor)
            positive = positive.type(torch.FloatTensor)
            negative = negative.type(torch.FloatTensor)
        return anchor, positive, negative
    
    def __len__(self):
        return len(self.imageFolderDataset.imgs)