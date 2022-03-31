import matplotlib.pyplot as plt
import numpy as np
import torch
from datetime import datetime
import pickle

from file_utils import *
from utils import *

import torch.nn as nn
import string
from PIL import Image
import torchvision
import torchvision.transforms as transforms
import torch
import torch.nn as nn
import torch.optim as optim
from dataset import HDRDataset
from generator_model import Generator
# from discriminator_model import Discriminator
from discriminator_model import PatchGAN
from torch.utils.data import DataLoader
from tqdm import tqdm
from torchvision.utils import save_image
from metrics import *
torch.backends.cudnn.benchmark = True
import time
from PerceptualLoss import *
    

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
class Experiment(object):
    def __init__(self, name='config'):
        config_data = read_file_in_dir('./', name + '.json')
        self.ROOT_STATS_DIR = './experiment_data'
        if config_data is None:
            raise Exception("Configuration file doesn't exist: ", name)

        self.__name = config_data['experiment_name']
        self.__experiment_dir = os.path.join(self.ROOT_STATS_DIR, self.__name)
        self.config_data = config_data
        
        #Loading all image paths
        all_paths = None
        with open('all_paths.pickle', 'rb') as handle:
            all_paths = pickle.load(handle)
          
        #Train validation and test Data
        Dataset = HDRDataset(all_paths,mode='train')
        train_size = int(len(Dataset)*0.8)
        val_size = len(Dataset)-train_size
        self.train_dataset, self.val_dataset = torch.utils.data.random_split(Dataset,[train_size,val_size])
        self.test_dataset = HDRDataset(all_paths,mode='test')
        
        #Train validation and test Dataloadaers
        self.__train_loader = DataLoader(self.train_dataset,batch_size=config_data['dataset']['batch_size'],shuffle=True)
        self.__val_loader = DataLoader(self.val_dataset,batch_size=config_data['dataset']['batch_size'],shuffle=True)
        self.__test_loader = DataLoader(self.test_dataset,batch_size=config_data['dataset']['batch_size'],shuffle=True)
   
        # Setup Experiment
        
        self.__epochs = config_data['experiment']['num_epochs']
        self.__current_epoch = 0
        self.__training_losses_gen = []
        self.__val_losses_gen = []
        self.__training_losses_disc = []
        self.__val_losses_disc = []
        self.__best_model = None  # Save your best model in this field and use this in test method.
        self.__best_val_loss = 100000


        self.discriminator = PatchGAN(in_channels=3).to(device)
        self.generator = Generator(in_channels=3, features=64).to(device)
        self.disc_opt = optim.Adam(self.discriminator.parameters(), lr=config_data['experiment']['learning_rate'], betas=(0.5, 0.999),)
        self.gen_opt = optim.Adam(self.generator.parameters(), lr=config_data['experiment']['learning_rate'], betas=(0.5, 0.999))
        self.BCE = nn.BCEWithLogitsLoss()
        if config_data['model']['type_of_loss']['Perceptual']:
            self.loss = VGGPerceptualLoss().cuda().float()
        else:
            self.loss = nn.L1Loss()

        #percep loss
#         self.percep_loss= VGGPerceptualLoss()
#         self.percep_loss=self.percep_loss.cuda().float()
        
        self.g_scaler = torch.cuda.amp.GradScaler()
        self.d_scaler = torch.cuda.amp.GradScaler()
        #self.__init_model()

        # Load Experiment Data if available
        self.__load_experiment()

    # Loads the experiment data if exists to resume training from last saved checkpoint.
    def __load_experiment(self):
        os.makedirs(self.ROOT_STATS_DIR, exist_ok=True)

        
        if os.path.exists(self.__experiment_dir) and False:
            self.__training_losses_gen = read_file_in_dir(self.__experiment_dir, 'training_losses_gen.txt')
            self.__val_losses_gen = read_file_in_dir(self.__experiment_dir, 'val_losses_gen.txt')
            self.__training_losses_disc = read_file_in_dir(self.__experiment_dir, 'training_losses_disc.txt')
            self.__val_losses_disc = read_file_in_dir(self.__experiment_dir, 'val_losses_disc.txt')
            self.__current_epoch = len(self.__training_losses)

            state_dict = torch.load(os.path.join(self.__experiment_dir, 'latest_model.pt'))
            self.generator.load_state_dict(state_dict['generator'])
            self.discriminator.load_state_dict(state_dict['discriminator'])
            self.gen_opt.load_state_dict(state_dict['optimizer_gen'])
            self.disc_opt.load_state_dict(state_dict['optimizer_disc'])
            

        else:
            os.makedirs(self.__experiment_dir,exist_ok=True)

    def __init_model(self):
        if torch.cuda.is_available():
            self.discriminator = self.discriminator.cuda().float() 
            self.generator = self.generator.cuda.float()
            self.BCE = self.BCE.cuda().float()
            self.L1_loss = self.L1_loss.cuda().float()


    # Main method to run your experiment. Should be self-explanatory.
    def run(self):
        start_epoch = self.__current_epoch
        for epoch in range(start_epoch, self.__epochs):  # loop over the dataset multiple times
            start_time = datetime.now()
            self.__current_epoch = epoch
            D_train_loss, G_train_loss = self.__train()
            D_val_loss, G_val_loss = self.__val()
            '''
            self.__record_stats(D_train_loss, D_val_loss, 'disc')
            self.__record_stats(G_train_loss, G_val_loss, 'gen')
            self.__log_epoch_stats(start_time)
            '''
            if(self.__epochs % 1 == 0):
                self.__save_model()
                
    def train_one_step(self, loader):
        tqdm_loop = tqdm(loader, leave=True)
        D_loss_total = 0
        G_loss_total = 0
        for i, (x, y) in enumerate(tqdm_loop):
            x = x.to(device)
            y = y.to(device)
            # Train Discriminator
            with torch.cuda.amp.autocast():
                y_fake = self.generator(x)
                D_real = self.discriminator(x, y)
                D_real_loss = self.BCE(D_real, torch.ones_like(D_real))
                D_fake =self.discriminator(x, y_fake.detach())
                D_fake_loss = self.BCE(D_fake, torch.zeros_like(D_fake))
                D_loss = (D_real_loss + D_fake_loss) / 2
                

            self.discriminator.zero_grad()
            self.d_scaler.scale(D_loss).backward()
            self.d_scaler.step(self.disc_opt)
            self.d_scaler.update()

            # Train generator
            with torch.cuda.amp.autocast():
                D_fake = self.discriminator(x, y_fake)
                G_fake_loss = self.BCE(D_fake, torch.ones_like(D_fake))
#                 L1 = self.L1_loss(y_fake, y) * self.config_data['model']['lambda']
#                 G_loss = G_fake_loss + L1
                weighted_loss = self.loss(y_fake, y) * self.config_data['model']['lambda']
                G_loss = G_fake_loss + weighted_loss
#                 G_loss = G_fake_loss + self.loss(y_fake,y)

            self.gen_opt .zero_grad()
            self.g_scaler.scale(G_loss).backward()
            self.g_scaler.step(self.gen_opt)
            self.g_scaler.update()
            D_loss_total += D_loss
            G_loss_total += G_loss

            if i % 10 == 0:
                tqdm_loop.set_postfix(
                    D_real=torch.sigmoid(D_real).mean().item(),
                    D_fake=torch.sigmoid(D_fake).mean().item(),
                )
        return (D_loss_total/loader.__len__()).item(), (G_loss_total/loader.__len__()).item()
    # TODO: Perform one training iteration on the whole dataset and return loss value
    def __train(self):
        self.discriminator.train()
        self.generator.train()
        D_loss, G_loss = self.train_one_step(self.__train_loader)
        
        return D_loss, G_loss

    # TODO: Perform one Pass on the validation set and return loss value. You may also update your best model here.
    def __val(self):
        self.generator.eval()
        self.discriminator.eval()
        D_loss_total = 0
        G_loss_total = 0
        ssim_total = 0
        uqi_total = 0
        psnr_total = 0
        no_of_iter = 0
        with torch.no_grad():
            for i, (x, y) in enumerate(self.__val_loader):
                x = x.to(device)
                y = y.to(device)

                # Train Discriminator
                with torch.cuda.amp.autocast():
                    y_fake = self.generator(x)
                    D_real = self.discriminator(x, y)
                    D_real_loss = self.BCE(D_real, torch.ones_like(D_real))
                    D_fake =self.discriminator(x, y_fake.detach())
                    D_fake_loss = self.BCE(D_fake, torch.zeros_like(D_fake))
                    D_loss = (D_real_loss + D_fake_loss) / 2
                    G_fake_loss = self.BCE(D_fake, torch.ones_like(D_fake))
                    weighted_loss = self.loss(y_fake, y) * self.config_data['model']['lambda']
                    G_loss = G_fake_loss + weighted_loss

                    D_loss_total += D_loss
                    G_loss_total += G_loss
                    ssim_total += ssim(y,y_fake)
                    uqi_total += uqi(y,y_fake)
                    psnr_total += psnr(y,y_fake)
                    no_of_iter = i
                    
                    if i==0:
                        ldr_ = x[0].cpu().detach().permute(1,2,0).numpy()
                        hdr_ = y[0].cpu().detach().permute(1,2,0).numpy()
                        generated_ = y_fake[0].cpu().detach().permute(1,2,0).numpy()
                        visualize(ldr_,hdr_,generated_,self.__current_epoch)
   
        print("SSIM = ", (ssim_total)/(no_of_iter + 1))
        print("PSNR = ", (psnr_total)/(no_of_iter + 1))
        print("UQI = ", (uqi_total)/(no_of_iter + 1))
        self.generator.train()
        self.discriminator.train()
                                          
        return (D_loss_total/self.val_dataset.__len__()).item(), (G_loss_total/self.val_dataset.__len__()).item()

    def test(self):
        pass
    
    def __save_model(self):
        root_model_path = os.path.join(self.__experiment_dir, 'latest_model.pt')    
        torch.save({
            'generator': self.generator.state_dict(),
            'discriminator': self.discriminator.state_dict(),
            'optimizer_gen': self.gen_opt.state_dict(),
            'optimizer_disc': self.disc_opt.state_dict(),
            }, root_model_path)
                                          


    def __record_stats(self, train_loss, val_loss, loss_type):
                                          
        if(loss_type == 'gen'):
            self.__training_losses_gen.append(train_loss)
            self.__val_losses_gen.append(val_loss)

            self.plot_stats(loss_type)
                                          

            write_to_file_in_dir(self.__experiment_dir, 'training_losses_'+loss_type +'.txt', self.__training_losses_gen)
                                          
            write_to_file_in_dir(self.__experiment_dir, 'val_losses_'+loss_type+'.txt', self.__val_losses_gen)
                                          
                                          
        elif(loss_type == 'disc'):
            self.__training_losses_disc.append(train_loss)
            self.__val_losses_disc.append(val_loss)

            self.plot_stats(loss_type)

            write_to_file_in_dir(self.__experiment_dir, 'training_losses_'+loss_type +'.txt', self.__training_losses_disc)
            write_to_file_in_dir(self.__experiment_dir, 'val_losses_'+loss_type+'.txt', self.__val_losses_disc)
                                          

    def __log(self, log_str, file_name=None):
        print(log_str)
        log_to_file_in_dir(self.__experiment_dir, 'all.log', log_str)
        if file_name is not None:
            log_to_file_in_dir(self.__experiment_dir, file_name, log_str)

    def __log_epoch_stats(self, start_time):
        time_elapsed = datetime.now() - start_time
        time_to_completion = time_elapsed * (self.__epochs - self.__current_epoch - 1)
        train_loss = self.__training_losses_gen[self.__current_epoch]
        val_loss = self.__val_losses_disc[self.__current_epoch]
        summary_str = "Epoch: {}, Train Loss: {}, Gen Val Loss: {}, Took {}, ETA: {}\n"
        summary_str = summary_str.format(self.__current_epoch + 1, train_loss, val_loss, str(time_elapsed),
                                         str(time_to_completion))
        self.__log(summary_str, 'epoch.log')
        train_loss = self.__training_losses_disc[self.__current_epoch]
        val_loss = self.__val_losses_disc[self.__current_epoch]
        summary_str = "Epoch: {}, Train Loss: {}, Disc Val Loss: {}, Took {}, ETA: {}\n"
        summary_str = summary_str.format(self.__current_epoch + 1, train_loss, val_loss, str(time_elapsed),
                                         str(time_to_completion))
        self.__log(summary_str, 'epoch.log')
        
    def plot_stats(self, loss_type):
        e = self.__current_epoch + 1
        x_axis = np.arange(1, e + 1, 1)
        plt.figure()
        if(loss_type == 'disc'):
            plt.title('Discriminator Loss Plot')
            training_loss = self.__training_losses_disc
            validation_loss = self.__val_losses_disc
        elif(loss_type == 'gen'):
            plt.title('Generator Loss Plot')
            training_loss = self.__training_losses_gen
            validation_loss = self.__val_losses_gen
        plt.plot(x_axis, training_loss, label="Training Loss")
        plt.plot(x_axis, validation_loss, label="Validation Loss")
        plt.xlabel("Epochs")
        plt.legend(loc='best')
        #plt.title(self.__name + " Stats Plot")

        plt.savefig(os.path.join(self.__experiment_dir, "stat_plot_"+ loss_type +".png"))
        plt.show()


if __name__ == "__main__":
    exp_name = 'default'

    print("Running Experiment: ", exp_name)
    exp = Experiment(exp_name)
    exp.run()
    exp.test()


        






        
        




