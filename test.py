import os
import torchvision.datasets as dset
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
import random
from PIL import Image
import torch
from torch.utils.data import DataLoader,Dataset
import tifffile as tiff   
import torch.nn as nn
from face_network import simpleCNN
from customDataset import SiameseNetworkDataset

def imshow(img,text=None,should_save=False):
    npimg = img.numpy()
    plt.axis("off")
    if text:
        plt.text(75, 8, text, style='italic',fontweight='bold',
            bbox={'facecolor':'white', 'alpha':0.8, 'pad':10})
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()    

def show_plot(iteration,loss):
    plt.plot(iteration,loss)
    plt.show()


class Config():
    training_dir = "./data/train"
    testing_dir = "./data/test"
    train_batch_size = 1
    train_number_epochs = 200


class ContrastiveLoss(torch.nn.Module):

    def __init__(self, margin=1.0):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin

    def forward(self, anchor, positive, negative):
        distance_anchor_positive = (anchor - positive).pow(2).sum(1)
        distance_anchor_negative = (anchor - negative).pow(2).sum(1)
        loss_triplet  = torch.relu(distance_anchor_positive - distance_anchor_negative + self.margin)
        return loss_triplet.mean()


if __name__ == '__main__':
    folder_dataset = dset.ImageFolder(root=Config.testing_dir)
    siamese_dataset = SiameseNetworkDataset(imageFolderDataset=folder_dataset,transform=transforms.ToTensor())
    test_dataloader = DataLoader(siamese_dataset,
                        shuffle=True,
                        num_workers=4,
                        batch_size=Config.train_batch_size)

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    net = simpleCNN().to(device)
    checkpoint = torch.load('./checkpoints/inceptionNet0003.pth')
    net.load_state_dict(checkpoint['inceptionNet'])
    net.eval()
    criterion = ContrastiveLoss()


    counter = []
    loss_history = [] 
    iteration_number= 0




    transform=transforms.ToTensor()
    for i in range(100):
        path = './data/test/'+'0/'+str(i)+'.tiff'
        img = tiff.imread(path)
        img = transform(img)
        img = torch.reshape(img,(2,8,8)).unsqueeze(0)
        img = img.type(torch.FloatTensor)
        img = img.cuda()
        output0 = net(img)
        standard = tiff.imread('ave.tiff')
        standard = transform(standard)
        standard = torch.reshape(standard,(2,8,8)).unsqueeze(0)
        standard = standard.type(torch.FloatTensor)
        standard = standard.cuda()
        output1 = net(standard)
        print(torch.sum((output0-output1)*(output0-output1)).item())