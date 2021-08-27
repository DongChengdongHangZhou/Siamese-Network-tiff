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
    training_dir = './data/train'
    testing_dir = './data/test'
    train_batch_size = 32
    train_number_epochs = 200


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
            anchor = anchor.type(torch.FloatTensor)
            positive = positive.type(torch.FloatTensor)
            negative = negative.type(torch.FloatTensor)
        return anchor, positive, negative
    
    def __len__(self):
        return len(self.imageFolderDataset.imgs)


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
    folder_dataset = dset.ImageFolder(root=Config.training_dir)
    siamese_dataset = SiameseNetworkDataset(imageFolderDataset=folder_dataset,transform=transforms.ToTensor())
    train_dataloader = DataLoader(siamese_dataset,
                        shuffle=True,
                        num_workers=4,
                        batch_size=Config.train_batch_size)

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    net = simpleCNN().to(device)
    # checkpoint = torch.load('./checkpoints/inceptionNet0005.pth')
    # net.load_state_dict(checkpoint['inceptionNet'])
    net.train()
    criterion = ContrastiveLoss()
    optimizer = torch.optim.Adam(net.parameters(), lr=0.0005, weight_decay=0.0005)

    counter = []
    loss_history = [] 
    iteration_number= 0

    for epoch in range(0,Config.train_number_epochs):
        for i, data in enumerate(train_dataloader,0):
            anchor, positive, negative = data
            anchor, positive, negative = anchor.cuda(), positive.cuda(), negative.cuda()
            optimizer.zero_grad()
            output_anchor = net(anchor)
            output_positive = net(positive)
            output_negative = net(negative)
            loss_triplet = criterion(output_anchor, output_positive, output_negative)
            loss_triplet.backward()
            optimizer.step()
            print("step number {} Epoch number {} Current loss {}\n".format(i, epoch, loss_triplet.item()))
            f = open('logger.txt', 'a')
            f.write('epoch: ')
            f.write(str(epoch))
            f.write('    step: ')
            f.write(str(i))
            f.write('    loss: ')
            f.write(str(loss_triplet.item()))
            f.write('\n')
            iteration_number +=1
            counter.append(iteration_number)
            loss_history.append(loss_triplet.item())
        if True:
            torch.save({
                'inceptionNet': net.state_dict()
            }, './checkpoints/' + 'inceptionNet' + '%04d.pth' % epoch
            )
    f.close()
    show_plot(counter,loss_history)

