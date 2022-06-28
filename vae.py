import argparse
import parser
import pathlib
import os

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np

import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchvision.utils import make_grid
from torchvision.datasets import MNIST

import matplotlib.pyplot as plt

def add_args(parser):
    parser.add_argument("--e_hidden", type=int, default=512)
    parser.add_argument("--d_hidden", type=int, default=512)
    parser.add_argument("--latent_dim", type=int, default=2)
    parser.add_argument("--learning_rate", type=float, default=0.001)
    parser.add_argument("--batch_size", type=int, default=100)
    parser.add_argument("--weight_decay", type=float, default=1e-5)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--decoder_prob", choices=["b", "g"],default='b')
    parser.add_argument("--n_samples", type=int, default=1000)
    parser.add_argument("--step", type=str, default='train')
    parser.add_argument("--data_dir", type=pathlib.Path, default="./data")
    parser.add_argument("--model_dir", type=pathlib.Path, default="./model_data")

'''
Encoder
'''
class Encoder(nn.Module):

	def __init__(self,latent_dim):
		super(Encoder, self).__init__()

		self.latent_dim = latent_dim

		self.encoder = nn.Sequential(
				nn.Linear(784, cfg.e_hidden),
				nn.ReLU(),
				nn.Linear(cfg.e_hidden, 2 * self.latent_dim),
			)

	def forward(self, x):
		x = self.encoder(x)

		return x[:,:self.latent_dim], x[:,self.latent_dim:]

'''
Bernoulli Decoder
'''
class BernoulliDecoder(nn.Module):
	def __init__(self,latent_dim):
		super(BernoulliDecoder, self).__init__()

		self.latent_dim = latent_dim

		self.decoder = nn.Sequential(
				nn.Linear(self.latent_dim,cfg.d_hidden),
				nn.ReLU(),
				nn.Linear(cfg.d_hidden,784),
				nn.Sigmoid()
			)

	def forward(self ,x):
		return self.decoder(x)

'''
Gaussian Decoder
'''
class GaussianDecoder(nn.Module):
	def __init__(self,latent_dim):
		super(GaussianDecoder, self).__init__()

		self.latent_dim = latent_dim

		self.encoder = nn.Sequential(
				nn.Linear(self.latent_dim, cfg.e_hidden),
				nn.ReLU(),
				nn.Linear(cfg.d_hidden, 2 * self.latent_dim),
			)
	def forward(self, x):

		x = self.encoder(x)

		return x[:,:self.latent_dim], x[:,self.latent_dim:]

class VAE(nn.Module):
	def __init__(self,decoder_prob = 'b'):
		super(VAE, self).__init__()

		self.decoder_prob = decoder_prob

		self.latent_dim = cfg.latent_dim if self.decoder_prob == 'b' else 784
		
		print(self.latent_dim)

		# encoder layers
		self.encoder = Encoder(self.latent_dim)

		# decoder layers
		self.decoder = BernoulliDecoder(self.latent_dim) if self.decoder_prob == 'b' else GaussianDecoder(self.latent_dim)

	def forward(self, x):

		x = x.view(-1, 784)

		# encode
		encoder_mu, encoder_logvar = self.encoder(x)

		# 训练时重参数化采样
		if self.training:
			z = encoder_mu + torch.randn_like(encoder_mu)* torch.exp(0.5 * encoder_logvar)
		else:
			# 评估时，模型已稳定，均值可作为隐变量
		 	z = encoder_mu

		# decode
		decoded_x = self.decoder(z)

		return encoder_mu, encoder_logvar, decoded_x

	def loss(self,x, decoded_x, encoder_mu, encoder_logvar):

		x = x.view(-1,784)

		loglikelihood = self.bernoulli_loglikelihood(x, decoded_x) if self.decoder_prob == 'b' else self.gaussian_loglikelihood(x, *decoded_x) 

		return loglikelihood - self.KL(encoder_mu,encoder_logvar)

	r'''
	Bernoulli distribution log-likelihood(AEVE Page11 C.1)

	Bernoulli的log似然与二分类交叉熵只差一个负号，decoder与其他生成模型一致
	'''
	def bernoulli_loglikelihood(self, x, decoded_x):
		return F.binary_cross_entropy(input=decoded_x.view(-1, 784), target=x.view(-1, 784), reduction='sum')

	'''
	Gaussian distribution log-likelihood(AEVB Page11 C.2)

	AEVB中提到，少量的隐变量可以收敛，隐变量过多不收敛。高斯分布需要设置与输入同等维度的隐变量，
	经过训练过程并不收敛，具体表现为均值方差的参数很快变为nan；把网络设计的更复杂也可能收敛
	'''
	def gaussian_loglikelihood(self,x, mu, logvar):
		return - 0.5 * torch.matmul((x - mu) * torch.exp( -logvar) , (x - mu).T).sum() - 0.5* 0.5 * 784 * torch.abs(torch.sum(logvar))

	'''
	KL Divergence(AEVB Page5)

	KL计算仅依赖均值方差
	'''
	def KL(self, mu, logvar):
		return 0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

def train():

	train_set = MNIST(root=cfg.data_dir, train=True, download=True, transform=transforms.ToTensor())
	test_set  = MNIST(root=cfg.data_dir, train=False, download=True, transform=transforms.ToTensor())

	train_loader = DataLoader(train_set, batch_size=cfg.batch_size, shuffle=True)
	test_loader  = DataLoader(test_set, batch_size=cfg.batch_size, shuffle=True)

	vae = VAE(cfg.decoder_prob)

	vae = vae.to(device)

	optimizer = optim.Adam(params=vae.parameters(), lr=cfg.learning_rate, weight_decay=cfg.weight_decay)

	vae.train()

	# Train
	losses = []

	for epoch in range(cfg.epochs):
		losses.append(0)

		number_of_batches = 0

		# Grab the batch, we are only interested in images not on their labels
		for images, _ in train_loader:

			x = images.to(device)

			optimizer.zero_grad()

			encoder_mu, encoder_logvar, decoded_x = vae(x)

			loss = vae.loss(x, decoded_x, encoder_mu, encoder_logvar)

			loss.backward()

			optimizer.step()

			# Add loss to the cumulative sum
			losses[-1] += loss.item()

			number_of_batches += 1

			# print("iter: ",number_of_batches,'loss=',loss.item())
  
		losses[-1] /= number_of_batches
		
		print('Epoch [%d / %d] average reconstruction error: %f' % (epoch+1, cfg.epochs, losses[-1]))

		model_path = os.path.join(cfg.model_dir,"vae-{}.model".format(epoch+1))

		torch.save(vae.state_dict(),model_path)

def generate():

	vae = VAE(cfg.decoder_prob)

	model_path = os.path.join(cfg.model_dir,"vae-{}.model".format(50))

	vae.load_state_dict(torch.load(model_path))

	print("Loaded model from " + model_path)

	vae.eval()

	with torch.no_grad():
		z = torch.randn(50, cfg.latent_dim, device=device)

		# Reconstruct images from sampled latent vectors
		recon_images = vae.decoder(z)
		recon_images = recon_images.view(recon_images.size(0), 1, 28, 28)
		recon_images = recon_images.cpu()
		recon_images = recon_images.clamp(0, 1)

		# Plot Generated Images
		plt.imshow(np.transpose(make_grid(recon_images, 10, 5).numpy(), (1, 2, 0)))	

		plt.show()

if __name__ == '__main__':
	
	parser = argparse.ArgumentParser()
	add_args(parser)
	cfg = parser.parse_args()

	device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

	print(device)

	if cfg.step == 'train':
		train()

	elif cfg.step == 'gen':
		generate()

