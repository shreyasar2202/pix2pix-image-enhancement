#!/usr/bin/python
import matplotlib.pyplot as plt
import numpy as np
import torch
from datetime import datetime

from file_utils import *
import torch.nn as nn
import string
from PIL import Image
import torchvision.transforms as transforms
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.models as models
from dataset import DPEDDataset
from generator_model import Generator
from discriminator_model import PatchGAN
from torch.utils.data import DataLoader
from tqdm import tqdm
import torchvision
from vgg import *
from color_loss import *
import torch.cuda as cutorch
from torchvision.utils import save_image
from metrics import *

CRITIC_ITERATIONS = 5
LAMBDA_GP = 10
WEIGHT_CLIP = 0.01
torch.backends.cudnn.benchmark = True
device = torch.device(('cuda:0' if torch.cuda.is_available() else 'cpu'
                      ))


class Experiment(object):

    def __init__(self, name='config'):
        config_data = read_file_in_dir('./', name + '.json')
        self.ROOT_STATS_DIR = './experiment_data'
        if config_data is None:
            raise Exception("Configuration file doesn't exist: ", name)

        self.__name = config_data['experiment_name']
        self.__experiment_dir = os.path.join(self.ROOT_STATS_DIR,
                self.__name)
        self.config_data = config_data
        self.train_dataset = DPEDDataset(config_data['dataset'
                ]['data_dir'], image_source='sony')
        self.__train_loader = DataLoader(self.train_dataset,
                batch_size=config_data['dataset']['batch_size'],
                shuffle=True)

        # Load Datasets

        self.val_dataset = DPEDDataset(config_data['dataset']['data_dir'
                ], eval=True, mode='val', image_source='sony')
        self.__val_loader = DataLoader(self.val_dataset,
                config_data['dataset']['batch_size'], shuffle=True)
        self.test_dataset = DPEDDataset(config_data['dataset'
                ]['data_dir'], eval=True, mode='test',
                image_source='sony')
        self.__test_loader = DataLoader(self.test_dataset,
                batch_size=1, shuffle=False)

        # Setup Experiment

        self.__epochs = config_data['experiment']['num_epochs']
        self.__current_epoch = 0
        self.__training_losses_gen = []
        self.__val_losses_gen = []
        self.__training_losses_disc = []
        self.__val_losses_disc = []
        self.__best_model = None
        self.__best_val_ssim = 0
        self.__ssim = []
        self.__psnr = []

        self.discriminator = PatchGAN(in_channels=3).to(device)
        self.generator = Generator(in_channels=3,
                                   features=64).to(device)
        self.disc_opt = optim.Adam(self.discriminator.parameters(),
                                   lr=config_data['experiment'
                                   ]['learning_rate'], betas=(0.5,
                                   0.999))
        self.gen_opt = optim.Adam(self.generator.parameters(),
                                  lr=config_data['experiment'
                                  ]['learning_rate'], betas=(0.5,
                                  0.999))
        self.BCE = nn.BCEWithLogitsLoss()
        self.L1_loss = nn.L1Loss()
        self.gen_opt = optim.RMSprop(self.generator.parameters(),
                lr=config_data['experiment']['learning_rate'])
        self.disc_opt = optim.RMSprop(self.discriminator.parameters(),
                lr=config_data['experiment']['learning_rate'])

        # percep loss

        self.percep_loss = VGGPerceptualLoss()
        self.percep_loss = self.percep_loss.cuda().float()
        self.tex_con = models.vgg19(pretrained=True).features
        self.tex_con = self.tex_con.cuda().float()

        self.g_scaler = torch.cuda.amp.GradScaler()
        self.d_scaler = torch.cuda.amp.GradScaler()
        self.__load_experiment()

    # Loads the experiment data if exists to resume training from last saved checkpoint.

    def __load_experiment(self):
        os.makedirs(self.ROOT_STATS_DIR, exist_ok=True)

        if os.path.exists(self.__experiment_dir):
            self.__training_losses_gen = \
                read_file_in_dir(self.__experiment_dir,
                                 'training_losses_gen.txt')
            self.__val_losses_gen = \
                read_file_in_dir(self.__experiment_dir,
                                 'val_losses_gen.txt')
            self.__training_losses_disc = \
                read_file_in_dir(self.__experiment_dir,
                                 'training_losses_disc.txt')
            self.__val_losses_disc = \
                read_file_in_dir(self.__experiment_dir,
                                 'val_losses_disc.txt')
            self.__current_epoch = len(self.__training_losses)

            state_dict = torch.load(os.path.join(self.__experiment_dir,
                                    'latest_model.pt'))
            self.generator.load_state_dict(state_dict['generator'])
            self.discriminator.load_state_dict(state_dict['discriminator'
                    ])
            self.gen_opt.load_state_dict(state_dict['optimizer_gen'])
            self.disc_opt.load_state_dict(state_dict['optimizer_disc'])
        else:

            os.makedirs(self.__experiment_dir)

    def __init_model(self):
        if torch.cuda.is_available():
            self.discriminator = self.discriminator.cuda().float()
            self.generator = self.generator.cuda().float()
            self.BCE = self.BCE.cuda().float()
            self.L1_loss = self.L1_loss.cuda().float()

    def gram_matrix(self, y):
        (b, ch, h, w) = y.size()
        features = y.view(b, ch, w * h)
        features_t = features.transpose(1, 2)
        gram = features.bmm(features_t) / (ch * h * w)
        return gram

    def run(self):
        start_epoch = self.__current_epoch

        for epoch in range(start_epoch, self.__epochs):  # loop over the dataset multiple times
            start_time = datetime.now()
            self.__current_epoch = epoch
            (D_train_loss, G_train_loss) = self.__train()

            (D_val_loss, G_val_loss) = self.__val()
            if self.__current_epoch % 20 == 0:
                self.test()
            self.__record_stats(D_train_loss, D_val_loss, 'disc')
            self.__record_stats(G_train_loss, G_val_loss, 'gen')

            # self.__log_epoch_stats(start_time)

            if self.__epochs % 10 == 0:
                self.__save_model()

    def train_one_step(self, loader):
        tqdm_loop = tqdm(loader, leave=True)
        D_loss_total = 0
        G_loss_total = 0
        percep_loss_total = 0
        color_loss_total = 0
        text_loss_total = 0
        cl = ColorLoss()
        blur_rgb = Blur(3)
        for (i, (x, y)) in enumerate(tqdm_loop):
            x = x.to(device)
            y = y.to(device)

            # Train Discriminator

            with torch.cuda.amp.autocast():
                for _ in range(CRITIC_ITERATIONS):
                    y_fake = self.generator(x)
                    D_real = self.discriminator(x, y).reshape(-1)
                    D_fake = self.discriminator(x, y_fake.detach()).reshape(-1)
                    D_loss = -(torch.mean(D_real) - torch.mean(D_fake))
                    self.discriminator.zero_grad()
                    self.d_scaler.scale(D_loss).backward(retain_graph=True)
                    self.d_scaler.step(self.disc_opt)
                    self.d_scaler.update()
                    

                # clip discriminator weights between -0.01, 0.01 to enforce Lipschitz constraint
                for p in self.discriminator.parameters():
                    p.data.clamp_(-WEIGHT_CLIP, WEIGHT_CLIP)

            # Train generator

            with torch.cuda.amp.autocast():
                D_fake = self.discriminator(x, y_fake)
                y_fake_blur = blur_rgb(y_fake)
                y_dslr_blur = blur_rgb(y)
                color_loss = cl(y_fake_blur, y_dslr_blur)
                color_loss_total += color_loss.item()

                # TEXTURE LOSS

                gen_features = self.tex_con(y_fake)
                orig_feautes = self.tex_con(x)
                style_featues = self.tex_con(y)

                tex_con_loss = self.calculate_loss(gen_features,
                        orig_feautes, style_featues)

                percep_loss = self.config_data['model']['lambda'] * (3
                        * self.percep_loss.forward(y_fake, y) + 0.015
                        * color_loss + 17 * tex_con_loss) \
                    - torch.mean(D_fake)

            self.gen_opt.zero_grad()
            self.g_scaler.scale(percep_loss).backward()
            self.g_scaler.step(self.gen_opt)
            self.g_scaler.update()
            D_loss_total += D_loss.item()

            # G_loss_total += G_loss

            G_loss_total += percep_loss.item()

            if i % 10 == 0:
                tqdm_loop.set_postfix(D_real=torch.sigmoid(D_real).mean().item(),
                        D_fake=torch.sigmoid(D_fake).mean().item())

        return (D_loss_total / loader.__len__(), G_loss_total
                / loader.__len__())

    def __train(self):
        self.discriminator.train()
        self.generator.train()
        (D_loss, G_loss) = self.train_one_step(self.__train_loader)

        return (D_loss, G_loss)

    def __val(self):
        self.generator.eval()
        self.discriminator.eval()
        D_loss_total = 0
        G_loss_total = 0
        ssim_total = 0
        uqi_total = 0
        psnr_total = 0
        no_of_iter = 0
        percep_loss_total = 0
        color_loss_total = 0
        text_loss_total = 0
        cl = ColorLoss()
        blur_rgb = Blur(3)
        with torch.no_grad():
            for (i, (x, y)) in enumerate(self.__val_loader):
                x = x.to(device)
                y = y.to(device)
                with torch.cuda.amp.autocast():

                    y_fake = self.generator(x)
                    D_real = self.discriminator(x, y).reshape(-1)
                    D_fake = self.discriminator(x,
                            y_fake.detach()).reshape(-1)
                    D_loss = -(torch.mean(D_real) - torch.mean(D_fake))
                    D_fake = self.discriminator(x, y_fake)
                    y_fake_blur = blur_rgb(y_fake)
                    y_dslr_blur = blur_rgb(y)
                    color_loss = cl(y_fake_blur, y_dslr_blur)
                    color_loss_total += color_loss.item()

                    gen_features = self.tex_con(y_fake)
                    orig_feautes = self.tex_con(y)
                    style_featues = self.tex_con(y)
                    tex_con_loss = self.calculate_loss(gen_features,
                            orig_feautes, style_featues)
                    percep_loss = self.config_data['model']['lambda'] \
                        * (3 * self.percep_loss.forward(y_fake, y)
                           + 0.015 * color_loss + 17 * tex_con_loss) \
                        - torch.mean(D_fake)
                    D_loss_total += D_loss.item()
                    G_loss_total += percep_loss.item()
                    ssim_total += ssim(y, y_fake)
                    uqi_total += uqi(y, y_fake)
                    psnr_total += psnr(y, y_fake)
                    no_of_iter = i
                    if i == 0 and self.__current_epoch % 5 == 0:
                        phone = x[0].detach().cpu().permute(1, 2,
                                0).numpy()
                        fake = y_fake[0].detach().cpu().permute(1, 2,
                                0).numpy().astype(np.float32)
                        real_dslr = y[0].detach().cpu().permute(1, 2,
                                0).numpy()
                        (fig, ax) = plt.subplots(1, 3, figsize=(10, 5))
                        ax[0].imshow(phone * 0.25 + 0.5)
                        ax[0].set_title('phone')
                        ax[1].imshow(real_dslr * 0.25 + 0.5)
                        ax[1].set_title('dslr')
                        ax[2].imshow(fake * 0.25 + 0.5)
                        ax[2].set_title('generated')
                        plt.show()
        self.__ssim.append(ssim_total / (no_of_iter + 1))
        self.__psnr.append(psnr_total / (no_of_iter + 1))
        print ('SSIM = ', ssim_total / (no_of_iter + 1))
        print ('PSNR = ', psnr_total / (no_of_iter + 1))
        print ('UQI = ', uqi_total / (no_of_iter + 1))
        self.generator.train()
        self.discriminator.train()

        if ssim_total > self.__best_val_ssim:
            self.__save_model('best_model.pt')
            self.__best_val_ssim = ssim_total

        return (D_loss_total / self.__val_loader.__len__(),
                G_loss_total / self.__val_loader.__len__())

    def test(self):
        self.generator.eval()
        self.discriminator.eval()
        D_loss_total = 0
        G_loss_total = 0
        ssim_total = 0
        uqi_total = 0
        psnr_total = 0
        no_of_iter = 0
        cl = ColorLoss()
        blur_rgb = Blur(3)
        percep_loss_total = 0
        color_loss_total = 0
        with torch.no_grad():
            for (i, (x, y)) in enumerate(self.__test_loader):
                x = x.to(device)
                y = y.to(device)

                # Train Discriminator

                with torch.cuda.amp.autocast():
                    y_fake = self.generator(x)
                    D_real = self.discriminator(x, y).reshape(-1)
                    D_fake = self.discriminator(x,
                            y_fake.detach()).reshape(-1)
                    D_loss = -(torch.mean(D_real) - torch.mean(D_fake))
                    D_fake = self.discriminator(x, y_fake)
                    y_fake_blur = blur_rgb(y_fake)
                    y_dslr_blur = blur_rgb(y)
                    color_loss = cl(y_fake_blur, y_dslr_blur)
                    color_loss_total += color_loss.item()

                    gen_features = self.tex_con(y_fake)
                    orig_feautes = self.tex_con(y)
                    style_featues = self.tex_con(y)
                    tex_con_loss = self.calculate_loss(gen_features,
                            orig_feautes, style_featues)
                    percep_loss = self.config_data['model']['lambda'] \
                        * (3 * self.percep_loss.forward(y_fake, y)
                           + 0.015 * color_loss + 17 * tex_con_loss) \
                        - torch.mean(D_fake)
                    D_loss_total += D_loss.item()
                    G_loss_total += percep_loss.item()
                    ssim_total += ssim(y, y_fake)
                    uqi_total += uqi(y, y_fake)
                    psnr_total += psnr(y, y_fake)
                    no_of_iter = i

                    if i == 0 and self.__current_epoch % 5 == 0:
                        phone = x[0].detach().cpu().permute(1, 2,
                                0).numpy()
                        fake = y_fake[0].detach().cpu().permute(1, 2,
                                0).numpy().astype(np.float32)
                        real_dslr = y[0].detach().cpu().permute(1, 2,
                                0).numpy()
                        (fig, ax) = plt.subplots(1, 3, figsize=(10, 5))
                        ax[0].imshow(phone * 0.25 + 0.5)
                        ax[0].set_title('phone')
                        ax[1].imshow(real_dslr * 0.25 + 0.5)
                        ax[1].set_title('dslr')
                        ax[2].imshow(fake * 0.25 + 0.5)
                        ax[2].set_title('generated')
                        plt.show()
        print ('SSIM = ', ssim_total / (no_of_iter + 1))
        print ('PSNR = ', psnr_total / (no_of_iter + 1))
        print ('UQI = ', uqi_total / (no_of_iter + 1))
        self.generator.train()
        self.discriminator.train()
        return (D_loss_total / self.val_dataset.__len__(), G_loss_total
                / self.val_dataset.__len__())

    def __save_model(self, name='latest_model.pt'):
        root_model_path = os.path.join(self.__experiment_dir, name)
        torch.save({
            'generator': self.generator.state_dict(),
            'discriminator': self.discriminator.state_dict(),
            'optimizer_gen': self.gen_opt.state_dict(),
            'optimizer_disc': self.disc_opt.state_dict(),
            }, root_model_path)

    def __record_stats(
        self,
        train_loss,
        val_loss,
        loss_type,
        ):

        if loss_type == 'gen':
            self.__training_losses_gen.append(train_loss)
            self.__val_losses_gen.append(val_loss)

            self.plot_stats(loss_type)

            write_to_file_in_dir(self.__experiment_dir,
                                 'training_losses_' + loss_type + '.txt'
                                 , self.__training_losses_gen)

            write_to_file_in_dir(self.__experiment_dir, 'val_losses_'
                                 + loss_type + '.txt',
                                 self.__val_losses_gen)
        elif loss_type == 'disc':

            self.__training_losses_disc.append(train_loss)
            self.__val_losses_disc.append(val_loss)

            self.plot_stats(loss_type)

            write_to_file_in_dir(self.__experiment_dir,
                                 'training_losses_' + loss_type + '.txt'
                                 , self.__training_losses_disc)
            write_to_file_in_dir(self.__experiment_dir, 'val_losses_'
                                 + loss_type + '.txt',
                                 self.__val_losses_disc)
        write_to_file_in_dir(self.__experiment_dir, 'ssim.txt',
                             self.__ssim)
        write_to_file_in_dir(self.__experiment_dir, 'psnr.txt',
                             self.__psnr)

    def __log(self, log_str, file_name=None):
        print(log_str)
        log_to_file_in_dir(self.__experiment_dir, 'all.log', log_str)
        if file_name is not None:
            log_to_file_in_dir(self.__experiment_dir, file_name,
                               log_str)

    def __log_epoch_stats(self, start_time):
        time_elapsed = datetime.now() - start_time
        time_to_completion = time_elapsed * (self.__epochs
                - self.__current_epoch - 1)
        train_loss = self.__training_losses_gen[self.__current_epoch]
        val_loss = self.__val_losses_disc[self.__current_epoch]
        summary_str = \
            'Epoch: {}, Train Loss: {}, Gen Val Loss: {}, Took {}, ETA: {}\n'
        summary_str = summary_str.format(self.__current_epoch + 1,
                train_loss, val_loss, str(time_elapsed),
                str(time_to_completion))
        self.__log(summary_str, 'epoch.log')
        train_loss = self.__training_losses_disc[self.__current_epoch]
        val_loss = self.__val_losses_disc[self.__current_epoch]
        summary_str = \
            'Epoch: {}, Train Loss: {}, Disc Val Loss: {}, Took {}, ETA: {}\n'
        summary_str = summary_str.format(self.__current_epoch + 1,
                train_loss, val_loss, str(time_elapsed),
                str(time_to_completion))
        self.__log(summary_str, 'epoch.log')

    def plot_stats(self, loss_type):
        e = self.__current_epoch + 1
        x_axis = np.arange(1, e + 1, 1)
        plt.figure()
        if loss_type == 'disc':
            plt.title('Discriminator Loss Plot')
            training_loss = self.__training_losses_disc
            validation_loss = self.__val_losses_disc
        elif loss_type == 'gen':
            plt.title('Generator Loss Plot')
            training_loss = self.__training_losses_gen
            validation_loss = self.__val_losses_gen
        plt.plot(x_axis, training_loss, label='Training Loss')
        plt.plot(x_axis, validation_loss, label='Validation Loss')
        plt.xlabel('Epochs')
        plt.legend(loc='best')
        plt.savefig(os.path.join(self.__experiment_dir, 'stat_plot_'
                    + loss_type + '.png'))
        plt.close()

        # plt.show()

    def calc_content_loss(self, gen_feat, orig_feat):

            # calculating the content loss of each layer by calculating the MSE between the content and generated features and adding it to content loss

        content_l = torch.mean((gen_feat - orig_feat) ** 2)
        return content_l

    def calc_style_loss(self, gen, style):

        # Calculating the gram matrix for the style and the generated image

        (channel, height, width) = gen.shape
        G = torch.mm(gen.view(channel, height * width),
                     gen.view(channel, height * width).t())
        A = torch.mm(style.view(channel, height * width),
                     style.view(channel, height * width).t())

        # Calcultating the style loss of each layer by calculating the MSE between the gram matrix of the style image and the generated image and adding it to style loss

        style_l = torch.mean((G - A) ** 2)
        return style_l

    def calculate_loss(
        self,
        gen_features,
        orig_feautes,
        style_featues,
        ):
        style_loss = content_loss = 0
        for (gen, cont, style) in zip(gen_features, orig_feautes,
                style_featues):

            # extracting the dimensions from the generated image

            content_loss += self.calc_content_loss(gen, cont)
            style_loss += self.calc_style_loss(gen, style)
        alpha = 0.2
        beta = 0.2

        # calculating the total loss of e th epoch

        total_loss = alpha * content_loss + beta * style_loss
        return total_loss


if __name__ == '__main__':
    exp_name = 'default'

    print ('Running Experiment: ', exp_name)
    exp = Experiment(exp_name)
    exp.run()
    exp.test()
