from __future__ import print_function

import argparse
import pathlib
import matplotlib.pyplot as plt

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torch.optim as optim

import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torchvision.utils import make_grid

def add_args(parser):
	parser.add_argument("--embedding_dim", type=int, default=256)
	parser.add_argument("--num_embeddings", type=int, default=512)
	parser.add_argument("--hidden_size", type=int, default=256)
	parser.add_argument("--learning_rate", type=float, default=1e-3)
	parser.add_argument("--batch_size", type=int, default=16)
	parser.add_argument("--epochs", type=int, default=50)
	parser.add_argument("--vq_type", type=str, choices=["vq", "ema"], default='ema')
	parser.add_argument("--step", type=str, default='train')
	parser.add_argument("--decay", type=float, default=0.99)
	parser.add_argument("--data_dir", type=pathlib.Path, default="./data/CIFAR10")
	parser.add_argument("--model_dir", type=pathlib.Path, default="./model_data/vqvae")

class VectorQuantizer(nn.Module):
	def __init__(self, num_embeddings, embedding_dim, decay=0.99):
		super(VectorQuantizer, self).__init__()

		self.num_embeddings = num_embeddings
		self.embedding_dim = embedding_dim
		self.decay = decay

		self.embedding = nn.Embedding(self.num_embeddings, self.embedding_dim)
		self.embedding.weight.data.uniform_(-1/self.num_embeddings, 1/self.num_embeddings)

		self.embedding_num = torch.zeros(self.num_embeddings)

	def forward(self, inputs):

		emb_shape = self.embedding.weight.shape

		'''
		 Calculate L2 norm between inputs and embedding(E = embedding_dim, N = num_embeddings)
		 1. reshape inputs to 16 * 256 * 8 * 8 * 1 (BCHW*1)
		 2. permute embedding 512 * 256 -> 256 * 512 (NE -> EN)
		 3. reshape embedding to 256 * 1 * 1 * 512 (E * 1 * 1 * N)
		 4. boradcast embedding to 16 * 256 * 8 * 8 * 512 (BEHWN)
		 5. calculate L2 norm for dim=1, the resulting shape 16 * 8 * 8 * 512 (BHWN)
		'''
		distances = torch.norm(inputs.unsqueeze(-1) - self.embedding.weight.permute(1,0).view(emb_shape[-1],1,1,emb_shape[0]), 2, 1)

		# the argmin indices for last dim (16 * 8 * 8) * 1 => 1024 * 1
		encoding_indices = torch.argmin(distances, -1).view(-1,1).long()

		# EMA updating
		if self.training and self.decay > 0: self.ema_update(inputs, encoding_indices)

		# shape 16 * 8 * 8 * 256(BHWC)
		shifted_shape = inputs.permute(0, 2, 3, 1).shape

		'''
		1. index_select shape 1024 * 256 (B*H*W) x C
		2. reshape to 16 * 8 * 8 * 256(BHWC)
		3. permute to 16 * 256 * 8 * 8(BCHW)
		'''
		quantized = self.embedding.weight.index_select(0,encoding_indices.view(-1)).view(shifted_shape).permute(0, 3, 1, 2)

		# straight-through estimator
		return inputs + (quantized - inputs).detach(), quantized

	def ema_update(self, inputs, encoding_indices):
		'''
		For the compution efficiency, the implementation is slightly differ from VQ-VAE papaer, 
		that it updates all embeddings in one minibatch. It makes sense for the large batch size,
		where all embeddings will be selected with high probability. 
		'''			
		# 1024 * 512 one-hot
		ema_onehot = torch.zeros(encoding_indices.numel(), self.num_embeddings).long()

		ema_onehot.scatter_(1,encoding_indices,torch.ones_like(encoding_indices))

		self.embedding_num = self.decay * self.embedding_num  + (1 - self.decay) * ema_onehot.sum(0)

		# broadcast: (1024 * 1 * 512) x (1024 * 256 * 1) => 1024 * 512 * 256
		new_embedding = (inputs.permute(0, 2, 3, 1).reshape(-1,1,self.embedding_dim) * ema_onehot.unsqueeze(-1))

		ema_embedding = self.decay * self.embedding.weight + (1 - self.decay) * new_embedding.sum(0)

		self.embedding.weight = nn.Parameter(ema_embedding / (self.embedding_num.unsqueeze(-1) + 1e-6))	

class ResBlock(nn.Module):
	def __init__(self, in_channels, out_channels, bn=False):
		super(ResBlock, self).__init__()

		layers = [
			nn.ReLU(),
			nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1),
			nn.ReLU(),
			nn.Conv2d(out_channels, out_channels, kernel_size=1, stride=1, padding=0)
		]
		if bn:
			layers.insert(2, nn.BatchNorm2d(out_channels))

		self.convs = nn.Sequential(*layers)

	def forward(self, x):
		return x + self.convs(x)

class VQ_VAE(nn.Module):
	def __init__(self, hidden_size=256, num_embeddings=512, embedding_dim=256, vq_type='ema' ,bn=True, vq_coef=1, commit_coef=0.5, num_channels=3):
		super(VQ_VAE,self).__init__()

		# encoder
		self.encoder = nn.Sequential(
			nn.Conv2d(num_channels, hidden_size, kernel_size=4, stride=2, padding=1),
			nn.BatchNorm2d(hidden_size),
			nn.ReLU(inplace=True),
			nn.Conv2d(hidden_size, hidden_size, kernel_size=4, stride=2, padding=1),
			nn.BatchNorm2d(hidden_size),
			nn.ReLU(inplace=True),
			ResBlock(hidden_size, hidden_size, bn=bn),
			nn.BatchNorm2d(hidden_size),
			ResBlock(hidden_size, hidden_size, bn=bn),
			nn.BatchNorm2d(hidden_size),
		)

		# decoder
		self.decoder = nn.Sequential(
			ResBlock(hidden_size, hidden_size),
			nn.BatchNorm2d(hidden_size),
			ResBlock(hidden_size, hidden_size),
			nn.ConvTranspose2d(hidden_size, hidden_size, kernel_size=4, stride=2, padding=1),
			nn.BatchNorm2d(hidden_size),
			nn.ReLU(inplace=True),
			nn.ConvTranspose2d(hidden_size, num_channels, kernel_size=4, stride=2, padding=1),
			nn.Tanh()
		)

		self.quantizer = VectorQuantizer(num_embeddings,embedding_dim, cfg.decay) 

		self.vq_coef = vq_coef
		self.commit_coef = commit_coef


	def forward(self, x):
		z_e = self.encoder(x)

		z_q, quantized = self.quantizer(z_e)

		decoded_x = self.decoder(z_q)

		return decoded_x, z_e, quantized

	def loss(self, x, decoded_x, z_e, quantized):

		mse_loss = F.mse_loss(decoded_x, x)

		vq_loss = torch.mean(torch.norm(quantized - z_e, 2, 1))

		# the loss is identical to that introduced in origin VQVAE paper
		loss = mse_loss + (self.vq_coef + self.commit_coef) * vq_loss 

		return loss, mse_loss, vq_loss

def train():
	training_data = datasets.CIFAR10(root=cfg.data_dir, train=True, download=True,
								transform=transforms.Compose([
									transforms.ToTensor(),
									transforms.Normalize((0.5,0.5,0.5), (1.0,1.0,1.0))
								]))

	validation_data = datasets.CIFAR10(root=cfg.data_dir, train=False, download=True,
								transform=transforms.Compose([
									transforms.ToTensor(),
									transforms.Normalize((0.5,0.5,0.5), (1.0,1.0,1.0))
								]))

	train_loader = DataLoader(training_data, batch_size=cfg.batch_size,shuffle=True,pin_memory=True)	

	validation_loader = DataLoader(validation_data,batch_size=cfg.batch_size,shuffle=True,pin_memory=True)

	model = VQ_VAE(hidden_size=cfg.hidden_size, num_embeddings=cfg.num_embeddings, embedding_dim=cfg.embedding_dim, vq_type = cfg.vq_type)

	model.to(device)

	optimizer = optim.Adam(model.parameters(), lr=cfg.learning_rate, amsgrad=False)

	model.train()

	# Train
	losses = []

	for epoch in range(cfg.epochs):
		losses.append(0)

		number_of_batches = 0

		# Grab the batch, we are only interested in images not on their labels
		for images, _ in train_loader:

			x = images.to(device)

			optimizer.zero_grad()

			decoded_x, z_e, quantized = model(x)

			loss, mse, vq_loss = model.loss(x, decoded_x, z_e, quantized)

			loss.backward()

			optimizer.step()

			# Add loss to the cumulative sum
			losses[-1] += loss.item()

			number_of_batches += 1

			print("iter: ",number_of_batches,'loss=%.3f(%.3f,%.3f)'%(loss.item(),mse.item(),vq_loss.item()))
  
		losses[-1] /= number_of_batches
		
		print('Epoch [%d / %d] average reconstruction error: %f' % (epoch+1, cfg.epochs, losses[-1]))

		model_path = os.path.join(cfg.model_dir,"vqvae-{}.model".format(epoch+1))

		torch.save(model.state_dict(),model_path)	

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

