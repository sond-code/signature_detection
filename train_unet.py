import os
from PIL import Image
import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms
import os
import matplotlib.pyplot as plt
import numpy as np
import copy
import random
import shutil
import zipfile
from math import atan2, cos, sin, sqrt, pi, log
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
from numpy import linalg as LA
from torch import optim, nn
from torch.utils.data import DataLoader, random_split
from torchvision import transforms
from tqdm import tqdm



class SegmentationModel(Dataset):
    def __init__(self , img_folder, masks_folder, transform = None):
        self.images_dir =  img_folder
        self.masks_dir = masks_folder
        self.transform = transform
        self.images = os.listdir(img_folder)
        self.masks = os.listdir(masks_folder)
        self.create_mask_map()
    def create_mask_map(self):
        self.mask_map = {}
        for idx,mask in enumerate(self.masks):
            mask_core = mask.split('.')[0][:-5]
            self.mask_map[mask_core] = idx
        print("mask" , self.mask_map)

    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        image_name = self.images[idx]
        image_path = os.path.join(self.images_dir, image_name)
        image_name_m = image_name.split('.')[0]
        mask_idx = self.mask_map[image_name_m]
        mask_path = os.path.join(self.masks_dir, self.masks[mask_idx])
        image = Image.open(image_path).convert('RGB')
        mask = Image.open(mask_path).convert('L')
        mask = mask.resize((512,192), resample = Image.Resampling.BILINEAR)
        image = image.resize((512,192), resample = Image.Resampling.BILINEAR)
        mask = np.array(mask)
        mask = np.where(mask > 127, 0, 1)  
        mask = torch.from_numpy(mask).long()
        
        self.show_image(image)
        self.show_image(mask)
        if self.transform:
            image = self.transform(image)
        return image, mask

    def show_image(self,img):
        plt.imshow(img, cmap = 'gray')
        plt.show()
        
class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv_op = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.conv_op(x)
    
class DownSample(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = DoubleConv(in_channels, out_channels)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(self, x):
        down = self.conv(x)
        p = self.pool(down)

        return down, p
    


transform = transforms.Compose([
    transforms.ToTensor()
])

dataset = SegmentationModel('images','masks',transform)
train_size = int(0.8 * len(dataset))
test_size = len(dataset) - train_size
train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

##f##or i in range(10):
 ##   dataset[np.random.randint(10)]


class UpSample(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_channels, in_channels//2, kernel_size=2, stride=2)
        self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        x = torch.cat([x1, x2], 1)
        return self.conv(x)
    
class UNet(nn.Module):
    def __init__(self, in_channels, num_classes):
        super().__init__()
        self.down_convolution_1 = DownSample(in_channels, 64)
        self.down_convolution_2 = DownSample(64, 128)
        self.down_convolution_3 = DownSample(128, 256)
        self.down_convolution_4 = DownSample(256, 512)

        self.bottle_neck = DoubleConv(512, 1024)

        self.up_convolution_1 = UpSample(1024, 512)
        self.up_convolution_2 = UpSample(512, 256)
        self.up_convolution_3 = UpSample(256, 128)
        self.up_convolution_4 = UpSample(128, 64)

        self.out = nn.Conv2d(in_channels=64, out_channels=num_classes, kernel_size=1)

    def forward(self, x):
        down_1, p1 = self.down_convolution_1(x)
        down_2, p2 = self.down_convolution_2(p1)
        down_3, p3 = self.down_convolution_3(p2)
        down_4, p4 = self.down_convolution_4(p3)

        b = self.bottle_neck(p4)

        up_1 = self.up_convolution_1(b, down_4)
        up_2 = self.up_convolution_2(up_1, down_3)
        up_3 = self.up_convolution_3(up_2, down_2)
        up_4 = self.up_convolution_4(up_3, down_1)

        out = self.out(up_4)
        return out


def dice(prediction, ground_truth, epsilon=1e-07):
    prediction_copy = torch.sigmoid(prediction).squeeze(1)  
    prediction_copy = (prediction_copy > 0.5).float()  
    
    inter = (prediction_copy * ground_truth).sum()
    union = prediction_copy.sum() + ground_truth.sum()
    dice = (2. * inter + epsilon) / (union + epsilon)
    return dice

if __name__ == '__main__':

    BATCH_SIZE = 8
    LEARNING_RATE = 3e-4

    device = 'cpu'
    num_workers = 2
    if torch.cuda.is_available():
        device = 'cuda'
        num_workers = torch.cuda.device_count() * 4

    train_loader = DataLoader(train_dataset, batch_size = 8 , num_workers = num_workers , shuffle = True)
    test_loader = DataLoader(test_dataset, batch_size = 8 , num_workers = num_workers , shuffle = False)

    model = UNet(in_channels=3, num_classes=1).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE)
    criterion = nn.BCEWithLogitsLoss()
    torch.cuda.empty_cache()



Epochs = 50
train_losses = []
dc_coeffs = []

for epoch in tqdm(range(Epochs)):
    model.train()
    epoch_train_loss = 0
    dice_total = 0
    for idx,data_point  in enumerate(tqdm(train_loader, position=0 ,leave= True)):
        images = data_point[0].float().to(device)
        masks = data_point[1].float().to(device)
        y_pred  = model(images)
        optimizer.zero_grad()
        loss = criterion(y_pred,masks.unsqueeze(1))
        dc_coeff = dice(y_pred,masks)
        epoch_train_loss += loss.item()
        dice_total += dc_coeff.item()
        loss.backward() 
        optimizer.step()

    avg_loss = epoch_train_loss / (idx+1)
    avg_dc = dice_total / (idx+1)
    train_losses.append(avg_loss)
    dc_coeffs.append(avg_dc)
    
    print("-" * 30)
    print(f"Training Loss EPOCH {epoch + 1}: {avg_loss:.4f}")
    print(f"Training DICE EPOCH {epoch + 1}: {avg_dc:.4f}")

