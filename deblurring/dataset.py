from torch.utils.data import Dataset
from torchvision import transforms
import os
import numpy as np
import PIL
from tqdm import tqdm 
import torch

import matplotlib.pyplot as plt
from PIL import Image
import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from diffaugment import DiffAugment


import copy

class DPEDDataset(Dataset):
    def __init__(self, data_folder, eval=False, mode=None, image_source = None):
        self.data_folder = data_folder 
        self.dump_root_folder = r"./dped/dump"
        self.eval = eval
        self.mode = mode

        # You can use any valid transformations here

        # The following transformation normalizes each channel using the mean and std provided
        self.transform = transforms.Compose([transforms.ToPILImage(), transforms.ToTensor(),
                                              transforms.Normalize((0.5, 0.5, 0.5), (0.25, 0.25, 0.25))
                                              ])

        self.to_tensor = transforms.ToTensor()
        self.blur_transform= transforms.GaussianBlur(kernel_size=9, sigma=1.0)

        # we will use the following width and height to resize
        self.width = 100
        self.height = 100

        self.input_folder = os.path.join(self.data_folder, 'training_data/'+image_source)
        if not os.path.exists(self.input_folder):
            os.makedirs(self.input_folder)

        self.output_folder = os.path.join(self.data_folder, 'training_data/canon')
        if not os.path.exists(self.output_folder):
            os.makedirs(self.output_folder)

        if self.eval:
            self.input_folder = os.path.join(self.data_folder, 'test_data/patches/' + image_source)
            if not os.path.exists(self.input_folder):
                os.makedirs(self.input_folder)
            self.output_folder = os.path.join(self.data_folder, 'test_data/patches/canon')
            if not os.path.exists(self.output_folder):
                os.makedirs(self.output_folder)

        self.dump_input_folder = os.path.join(self.dump_root_folder, 'training_data/'+image_source)
        if not os.path.exists(self.dump_input_folder):
            os.makedirs(self.dump_input_folder)
        self.dump_output_folder = os.path.join(self.dump_root_folder, 'training_data/canon')
        if not os.path.exists(self.dump_output_folder):
            os.makedirs(self.dump_output_folder)

        if self.eval:
            self.dump_input_folder = os.path.join(self.dump_root_folder, 'test_data/patches/' + image_source)
            if not os.path.exists(self.dump_input_folder):
                os.makedirs(self.dump_input_folder)
            self.dump_output_folder = os.path.join(self.dump_root_folder, 'test_data/patches/canon')
            if not os.path.exists(self.dump_output_folder):
                os.makedirs(self.dump_output_folder)

        self.phone_img_names = set(os.listdir(self.input_folder))
        self.dslr_img_names = set(os.listdir(self.output_folder))
        image_names = list(self.phone_img_names.intersection(self.dslr_img_names))


        self.paths = [(os.path.join(self.input_folder, i), os.path.join(self.output_folder, i)) for i in image_names]
        self.dump_paths = [(os.path.join(self.dump_input_folder, i), os.path.join(self.dump_output_folder, i)) for i in image_names]

        if self.mode == 'val': # use first 50 images for validation
            self.paths = self.paths[:50]
            self.dump_paths = self.paths[:50]

        elif self.mode == 'test': # use last 50 images for test
            self.paths = self.paths[len(image_names)//2:]
            self.dump_paths = self.paths[len(image_names)//2:]

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):

        dslr_image = np.asarray(PIL.Image.open(self.paths[idx][1]).resize((256, 256)))

        dslr_image = self.transform(dslr_image).float()
        r = np.random.random_integers(0, 1, size=1)
        if r == 1:
            dslr_image = DiffAugment(dslr_image, policy='', channels_first=True)

        dslr_image_blur = copy.deepcopy(dslr_image)
        dslr_image_blur = self.blur_transform(dslr_image_blur).float()

        return dslr_image_blur, dslr_image

class PairedImageDataset(Dataset):
	def __init__(self, root_dir, dir_name, phone_dir_name, dslr_dir_name, resize=(256, 256)):
		self.phone_imgs_dir=os.path.join(root_dir, dir_name, phone_dir_name)
		self.dslr_imgs_dir = os.path.join(root_dir, dir_name, dslr_dir_name)
		self.notransform = transforms.ToTensor()
		self.size=resize
		self.transforms= transforms.Compose([transforms.ToTensor(), \
											 transforms.Normalize((0.5, 0.5, 0.5), (0.25, 0.25, 0.25)), \
											 transforms.Resize((256, 256))])

		self.phone_img_names = set(os.listdir(self.phone_imgs_dir))
		self.dslr_img_names = set(os.listdir(self.dslr_imgs_dir))
		self.img_names = list(self.phone_img_names.intersection(self.dslr_img_names))


	def __len__(self):
		return len(self.img_names)


	def __getitem__(self, item):
		phone_img_names=os.listdir(self.phone_imgs_dir)
		phone_img_name=phone_img_names[item]

		phone_img_path=os.path.join(self.phone_imgs_dir, phone_img_name)
		dslr_img_path = os.path.join(self.dslr_imgs_dir, phone_img_name)

		phone_img=Image.open(phone_img_path).resize(self.size)
		dslr_img=Image.open(dslr_img_path).resize(self.size)

		phone_img_t = self.transforms(phone_img) 
		dslr_img_t = self.transforms(dslr_img) 

		return phone_img_t, dslr_img_t

class PairedImageTensorDataset(Dataset):
	def __init__(self, root_dir, dir_name, phone_dir_name, dslr_dir_name, resize=(256, 256)):
		self.phone_imgs_dir=os.path.join(root_dir, dir_name, phone_dir_name)
		self.dslr_imgs_dir = os.path.join(root_dir, dir_name, dslr_dir_name)
		self.notransform = transforms.ToTensor()
		self.size=resize
		self.phone_img_names = set(os.listdir(self.phone_imgs_dir))
		self.dslr_img_names = set(os.listdir(self.dslr_imgs_dir))
		self.img_names = list(self.phone_img_names.intersection(self.dslr_img_names))
		self.blur_transform = transforms.GaussianBlur(kernel_size=9, sigma=1.0)

	def __len__(self):
		return len(self.img_names)

	def __getitem__(self, item):
		img_name=self.img_names[item]

		phone_img_path=os.path.join(self.phone_imgs_dir, img_name)
		dslr_img_path = os.path.join(self.dslr_imgs_dir, img_name)
		dslr_img_t = torch.load(dslr_img_path)

		r = np.random.random_integers(0, 1, size=1)

		dslr_image_blur_t = copy.deepcopy(dslr_img_t)
		dslr_image_blur_t = self.blur_transform(dslr_image_blur_t).float()


		return dslr_image_blur_t, dslr_img_t

	def normalize(self, img):
		return (img - np.min(img)) / (np.max(img) - np.min(img))

	def visualize(self, phone_img, dslr_img, figsize=(10,5)):
		phone_img = phone_img.cpu().permute(1, 2, 0).numpy()
		dslr_img = dslr_img.cpu().permute(1, 2, 0).numpy()

		fig, ax= plt.subplots(1, 2, figsize=figsize)
		ax[0].imshow(self.normalize(phone_img))
		ax[0].set_title('iPhone')
		ax[1].imshow(self.normalize(dslr_img))
		ax[1].set_title('DSLR')
		plt.show()

