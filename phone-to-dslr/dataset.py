from torch.utils.data import Dataset
from torchvision import transforms
import os
import numpy as np
import PIL
from tqdm import tqdm 


class DPEDDataset(Dataset):
    def __init__(self, data_folder, eval=False, mode=None, image_source = None):
        self.data_folder = data_folder + image_source
        self.eval = eval
        self.mode = mode

        self.transform = transforms.Compose([transforms.ToPILImage(), transforms.ToTensor(),
                                              transforms.Normalize((0.5, 0.5, 0.5), (0.25, 0.25, 0.25)),
                                              ])
        self.width = 100
        self.height = 100

        self.input_folder = os.path.join(self.data_folder, 'training_data/' + image_source)
        self.output_folder = os.path.join(self.data_folder, 'training_data/canon')

        if self.eval:
            self.input_folder = os.path.join(self.data_folder, 'test_data/patches/' + image_source)
            self.output_folder = os.path.join(self.data_folder, 'test_data/patches/canon')
        
        image_names = os.listdir(self.input_folder)
            
        self.paths = [(os.path.join(self.input_folder, i), os.path.join(self.output_folder, i)) for i in image_names]
        
        if self.mode == 'val': 
            self.paths = self.paths[:len(image_names)//2]
            
        elif self.mode == 'test': 
            self.paths = self.paths[len(image_names)//2:]

    
    def __len__(self):
        return len(self.paths)
    
    def __getitem__(self, idx): 
        phone_image = np.asarray(PIL.Image.open(self.paths[idx][0]).resize((64, 64)))
        dslr_image = np.asarray(PIL.Image.open(self.paths[idx][1]).resize((64, 64)))
        if self.transform:
            phone_image = self.transform(phone_image).float()
            dslr_image = self.transform(dslr_image).float()
        return phone_image, dslr_image