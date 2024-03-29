from __future__ import print_function

import os
import re
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
from torch.utils.tensorboard import SummaryWriter
import torchviz

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class VectorQuantizer(nn.Module):
	def __init__(self, num_embeddings, embedding_dim, decay=0.99):
		super(VectorQuantizer, self).__init__()

		self.num_embeddings = num_embeddings
		self.embedding_dim = embedding_dim
		self.decay = decay

		self.embedding = nn.Embedding(self.num_embeddings, self.embedding_dim).to(device)
		# self.embedding.weight.data.uniform_(-1/self.num_embeddings, 1/self.num_embeddings)
		self.embedding.weight.data.normal_(0, 0.2)

		if decay >0: self.init_ema()

	def init_ema(self):
		self.epsilon = 1e-6
		self.embedding_num_v = torch.arange(self.num_embeddings).unsqueeze(-1).to(device)
		self.embedding.weight.requires_grad_(False)
		self.register_buffer('shadow_size', torch.zeros(self.num_embeddings))
		self.register_buffer('shadow_embedding', torch.Tensor(self.num_embeddings, self.embedding_dim))

	def forward(self, inputs):

		emb_shape = self.embedding.weight.shape

		r'''
		 Calculate L2 norm between inputs and embedding

		 B: Batch Size, C: Channel, H: Height, W: Width, E: embedding_dim, N: num_embeddings

		 1. reshape inputs to 		(B, C, H, W, 1)	 => (16, 256, 8, 8, 1) 
		 2. permute embedding 		(N, E) -> (E, N) => (512, 256) -> (256, 512) 
		 3. reshape embedding to 	(E, 1, 1, N)	 => (256, 1, 1, 512) 
		 4. boradcast embedding to 	(B, E, H, W, N)	 => (16, 256, 8, 8, 512) 
		 5. L2 norm shape 			(B, H, W, N)	 => (16, 8, 8, 512) 
		'''
		distances = torch.norm(inputs.unsqueeze(-1) - self.embedding.weight.permute(1,0).view(emb_shape[-1],1,1,emb_shape[0]), 2, 1)

		# the argmin indices (16 , 8 , 8) => (1024,)
		encoding_indices = torch.argmin(distances, -1).view(-1).long()

		# 1024 * 512 one-hot
		ema_onehot = F.one_hot(encoding_indices, self.num_embeddings).type_as(inputs).to(device)

		# (1024, 512) * (512, 256) -> (1024, 256) -> (16, 8, 8, 256) -> (16, 256, 8, 8)
		quantized = torch.matmul(ema_onehot, self.embedding.weight).view(inputs.shape[0], *inputs.shape[2:], inputs.shape[1]).permute(0, 3, 1, 2).contiguous()

		# EMA updating
		if self.training and self.decay > 0: self.ema_update(inputs, torch.unique(encoding_indices), ema_onehot)
		
		perplexity = torch.exp(torch.special.entr(ema_onehot.sum(0)/(ema_onehot.sum() + 1e-10)).sum())

		# straight-through estimator
		return inputs + (quantized - inputs).detach(), quantized, perplexity

	@torch.no_grad()
	def ema_update(self, inputs, unique_indices, ema_onehot):

		# (512, 1) == (1, N) =>(512, N) => (512,) bool tensor
		need_indices = (self.embedding_num_v == unique_indices.view(1,-1)).sum(1).bool()

		need_new_size = torch.sum(ema_onehot, 0)[need_indices]

		self.shadow_size[need_indices] = self.decay * self.shadow_size[need_indices] + (1 - self.decay) * need_new_size

		# (512, 256)
		need_new_embedding = torch.matmul(ema_onehot.t(), inputs.view(-1, self.embedding_dim))

		self.shadow_embedding[need_indices] = self.decay * self.shadow_embedding[need_indices] + (1 - self.decay) * need_new_embedding[need_indices]

		# updating weight
		self.embedding.weight[need_indices] = self.shadow_embedding[need_indices] / need_new_size.unsqueeze(-1)

class VQ_VAE(nn.Module):
	def __init__(self, hidden_size=256, num_embeddings=512, embedding_dim=256, vq_type='ema' ,bn=True, vq_coef=1, commit_coef=0.25, num_channels=3):
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

		self.quantizer = VectorQuantizer(num_embeddings,embedding_dim, cfg.decay if vq_type =='ema' else 0)

		self.vq_type = vq_type
		self.vq_coef = 0 if vq_type=='ema' else vq_coef
		self.commit_coef = commit_coef
		self.model_dir = cfg.model_dir

		for l in self.modules():
			if isinstance(l, nn.Linear) or isinstance(l, nn.Conv2d):
				l.weight.detach().normal_(0, 0.2)
				nn.init.constant_(l.bias, 0)

		# self.encoder[-1].weight.detach().fill_(1 / 40)


	def forward(self, x):
		z_e = self.encoder(x)

		# vq
		z_q, quantized, perplexity = self.quantizer(z_e)

		# decode
		decoded_x = self.decoder(z_q)

		return decoded_x, z_e, quantized, perplexity

	def loss(self, x, decoded_x, z_e, quantized):

		mse_loss = F.mse_loss(decoded_x, x)
		vq_loss = F.mse_loss(quantized , z_e.detach())
		commit_loss = F.mse_loss(quantized.detach() , z_e)

		loss = mse_loss + self.vq_coef * vq_loss + self.commit_coef * commit_loss
		return loss, mse_loss, vq_loss

	def sample(self, size):
		indices = torch.randint(self.quantizer.num_embeddings, (size * 64,))
		embeddings = self.quantizer.embedding.weight.index_select(0, indices)

		return embeddings.view(size, 8, 8, -1).permute(0, 3, 1, 2).contiguous()

	def save(self, epoch):
		model_path = os.path.join(self.model_dir,"vq_vae-{}.model".format(epoch+1))
		model_dir = os.path.dirname(model_path)
		
		if not os.path.exists(model_dir): os.makedirs(model_dir)

		torch.save(model.state_dict(),model_path)

		print("Saved model to ", model_path)

	def load(self):
		model_num = self.last_model_num()

		assert model_num is not None, "Not found models from {}".format(self.model_dir)

		model_path = os.path.join(self.model_dir,"vq_vae-{}.model".format(model_num))
		model.load_state_dict(torch.load(model_path))

		print("Loaded the last model: " + model_path)

	def last_model_num(self):

		pattern = r'vq_vae-(\d+)\.model'
		model_names = filter(lambda fn: len(re.findall(pattern,fn)) == 1, os.listdir(self.model_dir))
		nums = [int(re.findall(pattern,fn)[0]) for fn in model_names]

		return max(nums) if len(nums) > 0 else None

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

class Trainer(object):

	def __init__(self, model, train_loader = None, test_loader = None, from_prtrained = False):
		self.model = model
		self.train_loader = train_loader
		self.test_loader = test_loader
		self.from_prtrained = from_prtrained
		self.writer = SummaryWriter(cfg.log_dir)

	def train(self):
		
		if self.from_prtrained: self.model.load()

		optimizer = optim.Adam(self.model.parameters(), lr=cfg.learning_rate, amsgrad=False)

		# Train
		losses = [0]

		for epoch in range(cfg.epochs):
			losses.append(0)
			# train
			self.do_train(epoch, optimizer, losses)

			# test
			self.test()

		print("Losses for {} epochs {}".format(cfg.epochs, losses))

	def do_train(self, epoch, optimizer, losses):

		self.model.train()

		number_of_batches = 0

		# Grab the batch, we are only interested in images not on their labels
		for images, _ in self.train_loader:
			x = images.to(device)
			optimizer.zero_grad()

			decoded_x, z_e, quantized, perplexity = self.model(x)

			loss, mse, vq_loss = self.model.loss(x, decoded_x, z_e, quantized)
								
			loss.backward()

			# torchviz.make_dot(loss, params=dict(self.model.named_parameters())).render("attached", format="png")

			optimizer.step()

			# Add loss to the cumulative sum
			losses[-1] += loss.item()
			
			number_of_batches += 1

			if number_of_batches % 100 == 0:
				print(f'Tranin epoch: {epoch} iter: {number_of_batches}','perplexity=%.3f, loss=%.3f(%.3f,%.3f)'% (perplexity,loss.item(),mse.item(),vq_loss.item()))

		losses[-1] /= number_of_batches

		print('Epoch [%d / %d] average error: %f' % (epoch+1, cfg.epochs, losses[-1]))

		self.model.save(epoch)

	def test(self, load_model = False):

		if load_model: self.model.load()

		model.eval()

		number_of_batches = 0

		write_image = lambda label, x: self.writer.add_image(label,make_grid(x[:64]))

		losses = np.zeros(3)

		reconstructed = False

		with torch.no_grad():
			for data, _ in self.test_loader:

				data = data.to(device)

				decoded_x = None
				
				decoded_x, z_e, quantized,perplexity = self.model(data)
				outputs = self.model.loss(data, decoded_x, z_e, quantized)

				if not reconstructed:
					write_image('original/test', data.mul(0.5).add(0.5))
					write_image('reconstructed/test', decoded_x.mul(0.5).add(0.5))
					reconstructed = True

				outputs = [l.cpu().item() for l in outputs]

				losses += np.array(outputs)

				number_of_batches += 1

				# print("Test iter: ",number_of_batches,'loss=%.3f(%.3f,%.3f)'% tuple(outputs))

		losses /= number_of_batches

		print('Test average error: %.3f(%.3f,%.3f)' % tuple(losses.tolist()))

	def generate(self, size):

		self.model.load()
		self.model.eval()

		write_image = lambda label, x: self.writer.add_image(label,make_grid(x.mul(0.5).add(0.5)))

		sample_embeddings = self.model.sample(size)

		decoded_x = self.model.decoder(sample_embeddings).cpu()

		write_image('reconstructed/generate', decoded_x)



def init_args(parser):
	parser.add_argument("--embedding_dim", type=int, default=256)
	parser.add_argument("--num_embeddings", type=int, default=512)
	parser.add_argument("--hidden_size", type=int, default=256)
	parser.add_argument("--learning_rate", type=float, default=2e-4)
	parser.add_argument("--batch_size", type=int, default=64)
	parser.add_argument("--test_batch_size", type=int, default=64)
	parser.add_argument("--epochs", type=int, default=1)
	parser.add_argument("--vq_type", type=str, choices=["vq", "ema"], default='vq')
	parser.add_argument("--step", type=str, default='train')
	parser.add_argument("--decay", type=float, default=0.99)
	parser.add_argument("--data_dir", type=pathlib.Path, default="./data/CIFAR10")
	parser.add_argument("--model_dir", type=pathlib.Path, default="./model_data/vq_vae")
	parser.add_argument("--log_dir", type=pathlib.Path, default="./log/vq_vae")

def data_loader(data_dir, batch_size, train):
	dataset = datasets.CIFAR10(root= data_dir, 
								train=train, 
								download=True, 
								transform= transforms.Compose([
									transforms.ToTensor(),
									transforms.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5))
								]))
	return DataLoader(dataset, batch_size=batch_size, shuffle=True, pin_memory=True)	

if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	init_args(parser)
	cfg = parser.parse_args()

	# device = torch.device('mps')

	print('device={}, vq_type={}'.format(device,cfg.vq_type))

	model = VQ_VAE(hidden_size=cfg.hidden_size, num_embeddings=cfg.num_embeddings, embedding_dim=cfg.embedding_dim, vq_type = cfg.vq_type)

	model.to(device)

	if cfg.step == 'train':
		trainer = Trainer(model, train_loader = data_loader(cfg.data_dir, cfg.batch_size, True), 
			test_loader = data_loader(cfg.data_dir, cfg.test_batch_size, False),
			from_prtrained = False)
		trainer.train()

	elif cfg.step == 'test':
		trainer = Trainer(model, test_loader = data_loader(cfg.data_dir, cfg.test_batch_size, False))
		trainer.test(True)

	elif cfg.step == 'gen':
		trainer = Trainer(model)
		trainer.generate(16)