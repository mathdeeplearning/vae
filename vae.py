import argparse
import parser
import pathlib

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
    parser.add_argument("--data_dir", type=pathlib.Path, default="./data")

'''
	1、分布假设：
		- p(z)  隐变量z的先验分布，假设为N(0,1)
		- p(x|z) z的似然分布，也称为生成模型。可选择多变量伯努利分布或多变量高斯分布
		- q(z|x) z的后验变分分布，假设为多变量高斯分布

	2、ELBO = -DL(q(z|x)||p(z)) + sum_{l=1}^L log p(x|z_l)/L
		- 第一项散度计算，与变量无关；第二项期望计算需通过采样完成
		- 根据p(x|z)不同的假设，有不同的计算形式

	3、Encoder：计算ELBO中的散度，结果与z无关
		- 输入为784维度的手写图片，输出为q(z|x)的方差、均值

	3、Decoder：计算ELBO中的期望
		- 先从N(0,1)中采样，然后通过重参数化，转变为需要的分布，再计算log p(x|z)期望
	
	4、重参数化：从N(0,1)中采样z，通过平移、放缩变换 z' = m + s*z，变换为q(z|x)分布，其中m，s为q(z|x)的方差均值
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
		# 一层DNN得到均值方差
		return x[:,:self.latent_dim], x[:,self.latent_dim:]

'''
Bernoulli分布假设，实际输出的是重建后的图片
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
高斯分布的似然概率，与Encoder相同，学习均值方差
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

		# 一层DNN得到均值方差
		encoder_mu, encoder_logvar = self.encoder(x)

		# 训练的时候，根据每个样本，从正态分布采样，作重参数化
		if self.training:
			z = encoder_mu + torch.randn_like(encoder_mu)* torch.exp(0.5*encoder_logvar)
		else:
			# 测试的时候，模型已经训练完毕，隐编码即为对应的期望
		 	z = encoder_mu

		decoded_x = self.decoder(z)

		return decoded_x, encoder_mu, encoder_logvar

	def loss(self,x, decoded_x, encoder_mu, encoder_logvar):

		# print("mu = ",encoder_mu)
		# print("var = ",encoder_logvar)
		x = x.view(-1,784)

		loglikelihood = self.bernoulli_loglikelihood(x, decoded_x) if self.decoder_prob == 'b' else self.gaussian_loglikelihood(x, *decoded_x) 

		return loglikelihood - self.KL(encoder_mu,encoder_logvar)

	'''
	Bernoulli分布log-likelihood等价于计算交叉熵
	'''
	def bernoulli_loglikelihood(self, x, decoded_x):
		return F.binary_cross_entropy(input=decoded_x.view(-1, 784), target=x.view(-1, 784), reduction='sum')

	'''
	Gaussian分布log-likelihood需要x的维度与mu、logvar的维度一致才能计算，
	原论文中提到，只有z的维度很小才能收敛，而x的维度是784，所以实际过程中难以收敛。
	'''
	def gaussian_loglikelihood(self,x, mu, logvar):
		print("mu = ",mu)
		print("var = ",logvar)

		return - 0.5 * torch.matmul((x - mu) * torch.exp( -logvar) , (x - mu).T).sum() - 0.5* 0.5 * 784 * torch.abs(torch.sum(logvar))

	'''
	ELBO中KL距离
	'''
	def KL(self, mu, logvar):
		return 0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

def plot(vae,device):
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

if __name__ == '__main__':
	
	parser = argparse.ArgumentParser()
	add_args(parser)
	cfg = parser.parse_args()

	device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
	print(device)

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

			decoded_x, encoder_mu, encoder_logvar = vae(x)

			loss = vae.loss(x, decoded_x, encoder_mu, encoder_logvar)

			loss.backward()

			optimizer.step()

			# Add loss to the cumulative sum
			losses[-1] += loss.item()

			number_of_batches += 1

			# print("iter: ",number_of_batches,'loss=',loss.item())
  
		losses[-1] /= number_of_batches
		
		print('Epoch [%d / %d] average reconstruction error: %f' % (epoch+1, cfg.epochs, losses[-1]))

		plot(vae,device)
