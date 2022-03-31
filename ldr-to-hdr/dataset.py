from torch.utils.data import Dataset
from torchvision import transforms
import os
import numpy as np
import PIL
from tqdm import tqdm 
import torch


class HDRDataset(Dataset):
    def __init__(self, all_paths, mode=None):
        self.eval = eval
        self.mode = mode

        self.width = 256
        self.height = 256
        self.transform = transforms.Resize((self.height,self.width))
        self.tensor_folder = 'Tensor_data/'
        

        if mode == 'train':
            self.hdr_paths = all_paths['train']['hdr']
            self.ldr_paths = all_paths['train']['ldr']
            
        elif mode == 'test':
            self.hdr_paths = all_paths['test']['hdr']
            self.ldr_paths = all_paths['test']['ldr']

    
    def __len__(self):
        return len(self.hdr_paths)
    
    def __getitem__(self, idx):        
       
        hdr_tensor = torch.load(self.tensor_folder+self.hdr_paths[idx])
        ldr_tensor = torch.load(self.tensor_folder+self.ldr_paths[idx])
        if self.transform:
            hdr_tensor = self.transform(hdr_tensor)
            ldr_tensor = self.transform(ldr_tensor)
            
        return ldr_tensor,hdr_tensor