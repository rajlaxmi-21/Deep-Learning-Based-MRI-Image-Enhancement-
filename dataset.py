import h5py
import numpy as np
import os

import torch
import torchvision
import cv2
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader

input_folder = '/kaggle/input/fastmri-knee-val'
output_folder = 'normalized_h5'

os.makedirs(output_folder, exist_ok=True)

for file_name in os.listdir(input_folder):
    if file_name.endswith('.h5'):
        file_path = os.path.join(input_folder, file_name)
        with h5py.File(file_path, 'r') as f:
            data = f['reconstruction_rss'][:]
        
        data_normalized = data / np.max(data)
        
        # Save normalized data to new .h5
        out_path = os.path.join(output_folder, file_name)
        with h5py.File(out_path, 'w') as f_out:
            f_out.create_dataset('reconstruction_rss', data=data_normalized)
        
print("Done")

all_data = []
input_fol = '/kaggle/working/normalized_h5'

for f in os.listdir(input_fol):
    file_path = os.path.join(input_fol, f)
    with h5py.File(file_path, 'r') as fi:
        slicex  = fi['reconstruction_rss'][:]
        all_data.append(slicex)
        
all_data = np.concatenate(all_data, axis=0)

class LowQualGen:
    def __init__(self, downscale_fact = 2, noise= 0.01, motion_blur = 0):
        self.downscale_fact = downscale_fact
        self.noise = noise
        self.motion_blur = motion_blur

    def __call__(self, img):
        h, w= img.shape
        small = cv2.resize(img, (w//self.downscale_fact, h//self.downscale_fact), interpolation = cv2.INTER_AREA)
        upscaled = cv2.resize(small, (w,h), interpolation=cv2.INTER_LINEAR)

        noise = np.random.normal(0, self.noise, img.shape)
        noisy = upscaled + noise

        if self.motion_blur > 0:
            noisy = cv2.GaussianBlur(noisy, (self.motion_blur, self.motion_blur), 0)

        noisy = np.clip(noisy, 0.0, 1.0)
        return noisy
    
class MRIdataset(Dataset):
    def __init__(self, slices, lq_generator=None, transform=None):
        self.slices = slices
        self.lq_generator = lq_generator
        self.transform = transform
        
    def __len__(self):
        return len(self.slices)
        
    def __getitem__(self, idx):
        hq = self.slices[idx]
        if self.lq_generator:
            lq = self.lq_generator(hq)
        else:
            lq = hq.copy()
        
        hq = torch.tensor(hq, dtype=torch.float).unsqueeze(0)
        lq = torch.tensor(lq, dtype=torch.float).unsqueeze(0)
        
        if self.transform:
            hq = self.transform(hq)
            lq = self.transform(lq)
            
        return lq, hq

# split into train and val    
train_slices, val_slices = train_test_split(all_data, test_size=0.2, random_state=42)


lq_gen = LowQualGen(downscale_fact=2, noise=0.01, motion_blur=1)

train_dataset = MRIdataset(train_slices, lq_generator=lq_gen)
val_dataset = MRIdataset(val_slices, lq_generator=lq_gen)

train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
val_loader   = DataLoader(val_dataset, batch_size=8, shuffle=False)

lq, hq = next(iter(train_loader))
print("LQ shape:", lq.shape, "HQ shape:", hq.shape)
