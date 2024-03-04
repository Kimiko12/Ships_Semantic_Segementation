import sys
import platform
import torch
import sklearn as s
import numpy as np
import pandas as pd
import os
from torchvision.io import read_image
from PIL import Image
import torchvision.transforms as transforms
import cv2
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from tqdm import tqdm
import torch.nn as nn 
import torch.optim as optim
import matplotlib.pyplot as plt
from skimage.io import imread
from skimage.morphology import binary_opening, disk, label
from torch.utils.data import random_split



has_gpu = torch.cuda.is_available()
has_mps = torch.backends.mps.is_built()
device = "mps" if has_mps else "cuda" if torch.cuda.is_available() else "cpu"

print(f"Python Platform: {platform.platform()}")
print(f"PyTorch Version: {torch.__version__}")
print()
print(f"Python {sys.version}")
print(f"Pandas {pd.__version__}")
print(f"Scikit-Learn {sk.__version__}")
print("NVIDIA/CUDA GPU is", "available" if has_gpu else "NOT AVAILABLE")
print("MPS (Apple Metal) is", "AVAILABLE" if has_mps else "NOT AVAILABLE")
print(f"Target device is {device}")


train_data = os.listdir('D:/загрузки/train_v2')
test_data = os.listdir('D:/загрузки/test_v2')


image_path_train = 'D:/загрузки/train_v2'
image_path_test = 'D:/загрузки/test_v2'

submission = pd.read_csv('D:/загрузки/sample_submission_v2.csv')
mask = pd.read_csv('D:/загрузки/train_ship_segmentations_v2.csv')

empty_ship = mask['EncodedPixels'].isna().sum()

Containing_ship = mask['EncodedPixels'].notna().sum()

categories = ['Empty images', 'Images containing ships']
counts = [empty_ship, Containing_ship]

df = mask.groupby('ImageId').agg({'EncodedPixels': 'count'})
df = df.rename(columns={'EncodedPixels' : 'ships'})
df['has_ship'] = df['ships'].map(lambda x: 1 if x > 0 else 0)

df_without_ship = mask[mask['EncodedPixels'].notna()]

def rle_decode(mask_rle, shape=(768, 768)):
    '''
    mask_rle: run-length as string formated (start length)
    shape: (height,width) of array to return 
    Returns numpy array, 1 - mask, 0 - background
    '''
    s = mask_rle.split()
    starts, lengths = [np.asarray(x, dtype=int) for x in (s[0:][::2], s[1:][::2])]
    starts -= 1
    ends = starts + lengths
    img = np.zeros(shape[0]*shape[1], dtype=np.uint8)
    for lo, hi in zip(starts, ends):
        img[lo:hi] = 1
    return img.reshape(shape).T

def masks_as_image(in_mask_list):
    all_masks = np.zeros((768, 768), dtype = np.uint8)
    for mask in in_mask_list:
        if isinstance(mask, str):
            all_masks |= rle_decode(mask)
    return all_masks

def rle_decode(mask_rle, shape=(768, 768)):
    s = mask_rle.split()
    starts, lengths = [np.asarray(x, dtype=int) for x in (s[0:][::2], s[1:][::2])]
    starts -= 1
    ends = starts + lengths
    img = np.zeros(shape[0] * shape[1], dtype=np.uint8)
    for lo, hi in zip(starts, ends):
        img[lo:hi] = 1
    return img.reshape(shape).T

def combine_masks(mask_list):
    combined_mask = None
    for mask_rle in mask_list:
        if isinstance(mask_rle, str):
            mask = rle_decode(mask_rle)
            if combined_mask is None:
                combined_mask = mask
            else:
                combined_mask |= mask  
    return combined_mask

combined_masks = df_without_ship.groupby('ImageId')['EncodedPixels'].apply(lambda x: combine_masks(x.tolist()))
mask['ships'] = mask['EncodedPixels'].map(lambda x: 1 if not pd.isna(x) else 0)
df = mask.groupby('ImageId').agg({'EncodedPixels': 'count'})
df = df.rename(columns={'EncodedPixels' : 'ships'})
df['has_ship'] = df['ships'].map(lambda x: 1 if x > 0 else 0)

df_combined_masks = pd.DataFrame(combined_masks)
df_combined_masks

class CustomDataset(Dataset):
    def __init__(self, df, image_dir, transform=None):
        self.df = df
        self.image_dir = image_dir
        self.transform = transform
        self.image_names = df['ImageId'].tolist()
    
    def __len__(self):
        return len(self.image_names)
    
    def __getitem__(self, idx):
        image_name = self.image_names[idx]
        image_path = os.path.join(self.image_dir, image_name)
        image = Image.open(image_path).convert('RGB')
        
        mask = self.df['EncodedPixels'].iloc[idx]
        mask = Image.fromarray(mask) 

        if self.transform is not None:
            image = self.transform(image)
            mask = self.transform(mask)
            mask = (mask > 0).float()

        return image, mask


transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
])
batch_size = 20

df_combined_masks.reset_index(inplace=True)
df_combined_masks.rename(columns={'index': 'ImageId'}, inplace=True)

dataset = CustomDataset(df=df_combined_masks, image_dir=image_path_train, transform=transform)


class CustomDataset_for_test_data(Dataset):
    def __init__(self, image_dir, transform=None):
        self.image_dir = image_dir
        self.transform = transform
        
        self.image_names = os.listdir(self.image_dir)
    
    def __len__(self):
        return len(self.image_names)
    
    def __getitem__(self, idx):
        image_name = self.image_names[idx]
        image_path = os.path.join(self.image_dir, image_name)
        image = Image.open(image_path).convert('RGB')

        if self.transform is not None:
            image = self.transform(image)
                 
        return image
    
transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
])

dataset_test = CustomDataset_for_test_data(image_dir=image_path_test, transform = transform)

train_size = int(0.8 * len(dataset)) 
test_size = len(dataset) - train_size


train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

train_data_loader = DataLoader(train_dataset, batch_size = 10, shuffle = True, num_workers= 4)
test_data_loader = DataLoader(test_dataset, batch_size = 10, shuffle = True, num_workers= 4)

class DoubleConv(nn.Module):
    def __init__(self, input_channels, output_channels):
        super(DoubleConv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(input_channels, output_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(output_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(output_channels, output_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(output_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.conv(x)

class UNet(nn.Module):
    def __init__(self, input_channels=3, output_channels=1):
        super(UNet, self).__init__()
        self.downs = nn.ModuleList([
            DoubleConv(input_channels, 32),
            DoubleConv(32, 64),
            DoubleConv(64, 128),
            DoubleConv(128, 256),
            DoubleConv(256, 512),
            DoubleConv(512, 1024)
        ])
        self.ups = nn.ModuleList([
            nn.ConvTranspose2d(1024, 512, kernel_size=2, stride=2),
            DoubleConv(1024, 512),
            nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2),
            DoubleConv(512, 256),
            nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2),
            DoubleConv(256, 128),
            nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2),
            DoubleConv(128, 64),
            nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2),
            DoubleConv(64, 32)
        ])
        self.final_conv = nn.Conv2d(32, output_channels, kernel_size=1)

    def forward(self, x):
        skip_connections = []
        for down in self.downs:
            x = down(x)
            skip_connections.append(x)
            x = nn.MaxPool2d(kernel_size=2)(x)

        x = skip_connections.pop()
        for idx in range(0, len(self.ups), 2):
            x = self.ups[idx](x)
            skip_connection = skip_connections.pop()
            x = torch.cat((x, skip_connection), dim=1)
            x = self.ups[idx + 1](x)

        return torch.sigmoid(self.final_conv(x))

input_channels = 3
output_channels = 1
model = UNet(input_channels, output_channels)

def check_accuracy(loader, model):
    num_correct = 0
    num_pixels = 0
    dice_score = 0

    model.eval()

    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            preds = model(x)
            preds = (preds > 0.5).float()
            num_correct += (preds == y).sum()
            num_pixels += torch.numel(preds)
            
            # Преобразуем массивы NumPy обратно в тензоры PyTorch
            dice_score += dice_coef(torch.tensor(y.cpu().detach().numpy()), torch.tensor(preds.cpu().detach().numpy()))

    print(f'Got {num_correct}/{num_pixels} with accuracy {num_correct/num_pixels*100:.2f}')
    print(f'Dice score: {dice_score/len(loader)}')
    model.train()

def dice_coef(y_true, y_pred, smooth=1):
    y_true_f = y_true.flatten()  
    y_pred_f = y_pred.flatten()
    intersection = torch.sum(y_true_f * y_pred_f)
    union = torch.sum(y_true_f) + torch.sum(y_pred_f) - intersection
    return (2. * intersection + smooth) / (union + smooth)

def bce_dice_loss(y_true, y_pred, smooth=1):
    bce = nn.BCELoss()
    bce_loss = bce(y_pred, y_true) 
    dice_loss = (1 - dice_coef(y_true, y_pred, smooth))
    return 0.2 * bce_loss + 0.8 * dice_loss

def binary_accuracy(y_true, y_pred):
    y_pred = (y_pred > 0.5).float()
    return (y_pred == y_true).float().mean()


def dice_coef(y_true, y_pred, smooth=1):
    y_true_f = y_true.view(-1)
    y_pred_f = y_pred.view(-1)
    intersection = torch.sum(y_true_f * y_pred_f)
    dice = (2. * intersection + smooth) / (torch.sum(y_true_f) + torch.sum(y_pred_f) + smooth)
    return dice

def bce_dice_coef_loss(y_true, y_pred, smooth=1):
    bce = torch.nn.BCELoss()
    bce_loss = bce(y_pred, y_true)
    dice_loss = (1 - dice_coef(y_true, y_pred, smooth))
    return 0.2 * bce_loss + 0.8 * dice_loss

model.load_state_dict(torch.load('D:/загрузки/ships/working_weights.pth'))

def check_accuracy(images, model):
    model.eval()
    all_predictions = []
    
    with torch.no_grad():
        preds = model(images)
        preds = (preds > 0.5).float()
        all_predictions.append(preds.cpu().detach().numpy())
    
    model.train()
    return np.concatenate(all_predictions, axis=0)

def visualize_segmentation(images, mask_preds, mask_trues):
    num_images = images.shape[0]
    
    for i in range(num_images):
        image = images[i]
        mask_pred = mask_preds[i].squeeze()
        mask_true = mask_trues[i].squeeze()  # Squeeze the mask_true tensor
        
        plt.figure(figsize=(15, 5))
        
        plt.subplot(1, 3, 1)
        plt.imshow(image)
        plt.title('Image')
        plt.axis('off')
        
        plt.subplot(1, 3, 2)
        plt.imshow(mask_pred, cmap='gray')
        plt.title('Predicted Mask')
        plt.axis('off')
        
        plt.subplot(1, 3, 3)
        plt.imshow(mask_true, cmap='gray')
        plt.title('True Mask')
        plt.axis('off')
        
        plt.show()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

for idx, (image, mask) in enumerate(test_data_loader):
    if idx >= 10:
        break
    
    image = image.to(device)
    
    mask_preds = check_accuracy(image, model) 
    mask_trues = mask
    
    visualize_segmentation(image.cpu().numpy().transpose(0, 2, 3, 1), mask_preds, mask_trues)
