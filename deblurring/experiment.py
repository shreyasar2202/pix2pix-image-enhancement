import matplotlib.pyplot as plt
import numpy as np
import torch
from datetime import datetime
import torchvision
import torchvision.models as models
from file_utils import *

import torch.nn as nn
import string
from PIL import Image
import torchvision.transforms as transforms
import torch
import torch.nn as nn
import torch.optim as optim
from dataset import DPEDDataset, PairedImageTensorDataset
from generator_model import Generator, ResnetGenerator
from discriminator_model import PatchGAN
from torch.utils.data import DataLoader
from tqdm import tqdm
from torchvision.utils import save_image
from metrics import *
import gc

torch.backends.cudnn.benchmark = True
import os
from diffaugment import DiffAugment
from vgg import VGGPerceptualLoss


device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


class Experiment(object):

	def calc_content_loss(self, gen_feat, orig_feat):
		# calculating the content loss of each layer by calculating the MSE between the content and generated features and adding it to content loss
		content_l = torch.mean((gen_feat - orig_feat) ** 2)
		return content_l

	def calc_style_loss(self, gen, style):
		# Calculating the gram matrix for the style and the generated image
		channel, height, width = gen.shape

		G = torch.mm(gen.view(channel, height * width), gen.view(channel, height * width).t())
		A = torch.mm(style.view(channel, height * width), style.view(channel, height * width).t())

		# Calcultating the style loss of each layer by calculating the MSE between the gram matrix of the style image and the generated image and adding it to style loss
		style_l = torch.mean((G - A) ** 2)
		return style_l

	def calculate_loss(self, gen_features, orig_feautes, style_featues):
		style_loss = content_loss = 0
		for gen, cont, style in zip(gen_features, orig_feautes, style_featues):
			# extracting the dimensions from the generated image
			content_loss += self.calc_content_loss(gen, cont)
			style_loss += self.calc_style_loss(gen, style)
		alpha = 0.2
		beta = 0.2
		# calculating the total loss of e th epoch
		total_loss = alpha * content_loss + beta * style_loss
		return total_loss


	def __init__(self, name='config'):
		config_data = read_file_in_dir('./', name + '.json')
		self.ROOT_STATS_DIR = './experiment_data'
		if config_data is None:
			raise Exception("Configuration file doesn't exist: ", name)

		self.__name = config_data['experiment_name']
		self.__experiment_dir = os.path.join(self.ROOT_STATS_DIR, self.__name)
		self.config_data = config_data

		self.train_dataset = PairedImageTensorDataset(root_dir='./tensor_data_zipped', dir_name='training_data',phone_dir_name='iphone', dslr_dir_name='canon')
		self.__train_loader = DataLoader(
			self.train_dataset,
			batch_size=config_data['dataset']['batch_size'],
			shuffle=True
		)

		self.val_dataset = PairedImageTensorDataset(root_dir='./tensor_data_zipped', dir_name='test_data\\patches',phone_dir_name='iphone', dslr_dir_name='canon')
		self.__val_loader = DataLoader(self.val_dataset, batch_size=1, shuffle=True)

		self.__epochs = config_data['experiment']['num_epochs']
		self.__current_epoch = 0
		self.__training_losses_gen = []
		self.__training_losses_gen_cgan = []
		self.__val_losses_gen = []
		self.__training_losses_disc = []
		self.__val_losses_gen_cgan = []
		self.__val_losses_disc = []
		self.__best_model = None  # Save your best model in this field and use this in test method.
		self.__best_val_loss = 100000

		self.discriminator = PatchGAN(in_channels=3).to(device)
		self.generator = Generator(in_channels=3, features=64).to(device)
		self.disc_opt = optim.Adam(self.discriminator.parameters(), lr=config_data['experiment']['learning_rate'], betas=(0.5, 0.999),)
		self.gen_opt = optim.Adam(self.generator.parameters(), lr=config_data['experiment']['learning_rate'], betas=(0.5, 0.999))

		self.BCE = nn.BCEWithLogitsLoss()
		self.L1_loss = nn.L1Loss()
		self.VGGPerceptual_loss = VGGPerceptualLoss().to(torch.device('cuda:0'))

		self.g_scaler = torch.cuda.amp.GradScaler()
		self.d_scaler = torch.cuda.amp.GradScaler()
		# self.__init_model()

		self.tex_con = models.vgg19(pretrained=True).features
		self.tex_con = self.tex_con.cuda().float()


		self.__save_samples_dir= config_data['dataset']['save_dir']
		self.__test_out_dir = config_data['dataset']['test_out_dir']

		self.denorm = transforms.Compose([ transforms.Normalize(mean = [ 0., 0., 0. ],
                                                     std = [ 1/0.25, 1/0.25, 1/0.25 ]),
                                transforms.Normalize(mean = [ -0.5, -0.5, -0.5 ],
                                                     std = [ 1., 1., 1. ]),
                               ])

	# Loads the experiment data if exists to resume training from last saved checkpoint.
	def __load_experiment(self):
		os.makedirs(self.ROOT_STATS_DIR, exist_ok=True)

		if os.path.exists(self.__experiment_dir):
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
			os.makedirs(self.__experiment_dir)

	def __init_model(self):
		if torch.cuda.is_available():
			self.discriminator = self.discriminator.cuda().float()
			self.generator = self.generator.cuda.float()
			self.BCE = self.BCE.cuda().float()
			self.L1_loss = self.L1_loss.cuda().float()

	# Main method to run your experiment. Should be self-explanatory.
	def run(self):
		start_epoch = self.__current_epoch
		for epoch in range(start_epoch, self.__epochs):  
			start_time = datetime.now()
			self.__current_epoch = epoch
			D_train_loss, G_train_loss = self.__train()
			D_val_loss, G_val_loss = self.__val()

			self.__training_losses_gen.append(G_train_loss)
			self.__val_losses_gen.append(G_val_loss)
			self.__training_losses_disc.append(D_train_loss)
			self.__val_losses_disc.append(D_val_loss)

			# self.__log_epoch_stats(start_time)
			if (self.__current_epoch % 20 == 0):
				self.__save_model()
				self.test()
				self.plot_stats('gen')
				self.plot_stats('disc')
				self.plot_stats('cgan')

	def train_one_step(self, loader):
		tqdm_loop = tqdm(loader, leave=True)
		D_loss_total = 0
		G_loss_total = 0
		D_real_loss_total = 0
		D_fake_loss_total = 0
		color_loss_total = 0
		G_fake_loss_total = 0
		texture_loss_total = 0
		l1_loss_total = 0
		percep_loss_total = 0
		print("Current Epoch : " + str(self.__current_epoch))

		for i, (x, y) in enumerate(tqdm_loop):
			x = x.to(device)
			y = y.to(device)
			# Train Discriminator
			with torch.cuda.amp.autocast():
				y_fake = self.generator(x)

				D_real = self.discriminator(x, y)
				D_real_loss = self.BCE(D_real, torch.ones_like(D_real))
				D_fake = self.discriminator(x, y_fake.detach())
				D_fake_loss = self.BCE(D_fake, torch.zeros_like(D_fake))
				D_loss = (1 * (1 * D_real_loss + 1 * D_fake_loss) / 2)
				D_real_loss_total += D_real_loss
				D_fake_loss_total += D_fake_loss

			self.discriminator.zero_grad()
			self.d_scaler.scale(D_loss).backward()
			self.d_scaler.step(self.disc_opt)
			self.d_scaler.update()

			G_loss = torch.tensor([0])
			D_loss_total += D_loss

			# Train generator
			with torch.cuda.amp.autocast():
				D_fake = self.discriminator(x, y_fake)
				G_fake_loss = self.BCE(D_fake, torch.ones_like(D_fake))
				L1 = self.L1_loss(y_fake, y) * self.config_data['model']['lambda']
				p_loss = self.VGGPerceptual_loss(y_fake, y) * self.config_data['model']['lambda']
				G_loss = 1 * (1 * G_fake_loss + 100*L1 + 1 * p_loss)
				G_fake_loss_total += G_fake_loss
				l1_loss_total += 0
				percep_loss_total += 0

			self.gen_opt.zero_grad()
			self.g_scaler.scale(G_loss).backward()
			self.g_scaler.step(self.gen_opt)
			self.g_scaler.update()
			G_loss_total += G_loss


			if i % 10 == 0:
				tqdm_loop.set_postfix(
					D_real=torch.sigmoid(D_real).mean().item(),
					D_fake=torch.sigmoid(D_fake).mean().item(),
				)

		self.__save_generated_images(self.__current_epoch, self.__save_samples_dir)

		print('Average D_real_loss : ' + str((D_real_loss_total/loader.__len__()).item()) + ' | Average D_fake_loss: ' + \
			  str((D_fake_loss_total/loader.__len__()).item()) + ' | Average D_loss :  ' + str((D_loss/loader.__len__()).item()) + '\n')

		print('Average G_fake_loss : ' + str((G_fake_loss_total/ loader.__len__()).item()) + '  | Average p_loss :  ' + str(
			0) + ' | Average G_loss :  ' + str((G_loss_total/loader.__len__()).item()) + '\n')

		self.__training_losses_gen_cgan.append((G_fake_loss_total/ loader.__len__()).item())
		print((color_loss_total / loader.__len__()))
		return (D_loss_total / loader.__len__()).item(), (G_loss_total / loader.__len__()).item()

	def gradient_penalty(self, critic, x, real, fake, device="cuda"):

		BATCH_SIZE, C, H, W = real.shape
		alpha = torch.rand((BATCH_SIZE, 1, 1, 1)).repeat(1, C, H, W).to(device)
		interpolated_images = real * alpha + fake * (1 - alpha)

		mixed_scores = torch.sigmoid(critic(x, interpolated_images)).mean()

		gradient = torch.autograd.grad(
			inputs=interpolated_images,
			outputs=mixed_scores,
			grad_outputs=torch.ones_like(mixed_scores),
			create_graph=True,
			retain_graph=True,
		)[0]
		gradient = gradient.view(gradient.shape[0], -1)
		gradient_norm = gradient.norm(2, dim=1)
		gradient_penalty = torch.mean((gradient_norm - 1) ** 2)
		return gradient_penalty

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
		G_fake_total = 0

		with torch.no_grad():
			for i, (x, y) in enumerate(self.__val_loader):
				x = x.to(device)
				y = y.to(device)

				# Train Discriminator
				with torch.cuda.amp.autocast():
					y_fake = self.generator(x)
					D_real = self.discriminator(x, y)
					D_real_loss = self.BCE(D_real, torch.ones_like(D_real))
					D_fake = self.discriminator(x, y_fake.detach())
					D_fake_loss = self.BCE(D_fake, torch.zeros_like(D_fake))

					D_loss = 1 * (D_real_loss + D_fake_loss) / 2
					G_fake_loss = self.BCE(D_fake, torch.ones_like(D_fake))

					L1 = self.L1_loss(y_fake, y) * self.config_data['model']['lambda']
					p_loss = self.VGGPerceptual_loss(y_fake, y) * self.config_data['model']['lambda']
					G_loss = 1 * (1 * G_fake_loss + 100*L1 + 1* p_loss) # + 1 * p_loss

					G_fake_total += G_fake_loss
					D_loss_total += D_loss
					G_loss_total += G_loss

					if (self.__current_epoch % 10 == 0):
						ssim_total += ssim(y, y_fake)
						uqi_total += uqi(y, y_fake)
						psnr_total += psnr(y, y_fake)
					no_of_iter = i

		if (self.__current_epoch % 10 == 0):
			print("SSIM = ", (ssim_total) / (no_of_iter + 1))
			print("PSNR = ", (psnr_total) / (no_of_iter + 1))
			print("UQI = ", (uqi_total) / (no_of_iter + 1))
		self.generator.train()
		self.discriminator.train()

		self.__val_losses_gen_cgan.append( (G_fake_total/ self.val_dataset.__len__()).item() )
		return (D_loss_total / self.val_dataset.__len__()).item(), (G_loss_total / self.val_dataset.__len__()).item()

	def test(self):
		self.generator.eval()
		self.discriminator.eval()
		D_loss_total = 0
		G_loss_total = 0
		ssim_total = 0
		uqi_total = 0
		psnr_total = 0
		ssim_input_total = 0
		uqi_input_total = 0
		psnr_input_total = 0
		no_of_iter = 0

		if not os.path.exists(self.__test_out_dir):
			os.mkdir(self.__test_out_dir)


		with torch.no_grad():
			for i, (x, y) in enumerate(self.__val_loader):
				x = x.to(device)
				y = y.to(device)

				with torch.cuda.amp.autocast():
					y_fake = self.generator(x)
					D_real = self.discriminator(x, y)
					D_real_loss = self.BCE(D_real, torch.ones_like(D_real))
					D_fake = self.discriminator(x, y_fake.detach())
					D_fake_loss = self.BCE(D_fake, torch.zeros_like(D_fake))
					D_loss = 1 * (D_real_loss + D_fake_loss) / 2
					G_fake_loss = self.BCE(D_fake, torch.ones_like(D_fake))
					L1 = self.L1_loss(y_fake, y) * self.config_data['model']['lambda']
					p_loss = self.VGGPerceptual_loss(y_fake, y) * self.config_data['model']['lambda']
					G_loss = 1 * (1 * G_fake_loss + 100*L1 + 1 * p_loss)
					D_loss_total += D_loss
					G_loss_total += G_loss
					ssim_input_total += ssim(y, x)
					uqi_input_total += uqi(y, x)
					psnr_input_total += psnr(y, x)

					ssim_total += ssim(y, y_fake)
					uqi_total += uqi(y, y_fake)
					psnr_total += psnr(y, y_fake)
					no_of_iter = i

					y_fake = self.generator(x)
					y_fake = self.denorm(y_fake)
					save_image(y_fake, self.__test_out_dir + f"\\y_gen_{self.__current_epoch}_{i}.png")
					save_image(self.denorm(x), self.__test_out_dir + f"\\input_{self.__current_epoch}_{i}.png")
					if self.__current_epoch == 1:
						save_image(self.denorm(y), self.__test_out_dir + f"\\label_{self.__current_epoch}_{i}.png")

		print("Current Epoch : " + str(self.__current_epoch))
		print("SSIM = ", (ssim_total) / (no_of_iter + 1))
		print("PSNR = ", (psnr_total) / (no_of_iter + 1))
		print("UQI = ", (uqi_total) / (no_of_iter + 1))

		print("SSIM Input = ", (ssim_input_total) / (no_of_iter + 1))
		print("PSNR Input = ", (psnr_input_total) / (no_of_iter + 1))
		print("UQI Input = ", (uqi_input_total) / (no_of_iter + 1))

		self.generator.train()
		self.discriminator.train()

	def __save_generated_images(self, epoch, folder):
		x, y = next(iter(self.__val_loader))
		x, y = x.to(device), y.to(device)

		if not os.path.exists(folder):
			os.mkdir(folder)

		self.generator.eval()
		with torch.no_grad():
			y_fake = self.generator(x)
			y_fake = self.denorm(y_fake)
			save_image(y_fake, folder + f"\\y_gen_{epoch}.png")
			save_image(self.denorm(x), folder + f"\\input_{epoch}.png")
			if epoch == 1:
				save_image(self.denorm(y), folder + f"\\label_{epoch}.png")
		self.generator.train()

	def __save_model(self):
		root_model_path = os.path.join('.\\', 'latest_model.pt')
		torch.save({
			'generator': self.generator.state_dict(),
			'discriminator': self.discriminator.state_dict(),
			'optimizer_gen': self.gen_opt.state_dict(),
			'optimizer_disc': self.disc_opt.state_dict(),
		}, root_model_path)

	def __record_stats(self, train_loss, val_loss, loss_type):

		if (loss_type == 'gen'):
			self.__training_losses_gen.append(train_loss)
			self.__val_losses_gen.append(val_loss)

			self.plot_stats(loss_type)


		elif (loss_type == 'disc'):
			self.__training_losses_disc.append(train_loss)
			self.__val_losses_disc.append(val_loss)

			self.plot_stats(loss_type)


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
		if (loss_type == 'disc'):
			plt.title('Discriminator Loss Plot')
			training_loss = self.__training_losses_disc
			validation_loss = self.__val_losses_disc
		elif (loss_type == 'gen'):
			plt.title('Generator Loss Plot')
			training_loss = self.__training_losses_gen
			validation_loss = self.__val_losses_gen
		elif (loss_type == 'cgan'):
			plt.title('Generator cGAN Loss Plot')
			training_loss = self.__training_losses_gen_cgan
			validation_loss = self.__val_losses_gen_cgan

		plt.plot(x_axis, training_loss, label="Training Loss")
		plt.plot(x_axis, validation_loss, label="Validation Loss")
		plt.xlabel("Epochs")
		plt.legend(loc='best')
		plt.savefig(os.path.join("stat_plot_" + loss_type + ".png"))


if __name__ == "__main__":
	exp_name = 'default'

	print("Running Experiment: ", exp_name)
	exp = Experiment(exp_name)
	exp.run()
	exp.test()