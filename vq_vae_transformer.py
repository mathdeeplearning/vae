from __future__ import print_function
import math

import os
import re
import argparse
import pathlib
import matplotlib.pyplot as plt
import shutil

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

class Discriminator(nn.Module):
	def __init__(self):
		super(Discriminator, self).__init__()

		self.model = nn.Sequential(
			nn.Conv2d(3, 64, 4, 2, 1),
			nn.LeakyReLU(0.2),
			nn.Conv2d(64, 128, 4, 2, 1),
			nn.BatchNorm2d(128),
			nn.LeakyReLU(0.2, True),
			nn.Conv2d(128, 1, 1, 1, 0),
			nn.Sigmoid(),
		)
		
		for l in self.modules():
			if isinstance(l, nn.Conv2d):
				l.weight.detach().normal_(0.0, 0.02)
			elif isinstance(l, nn.BatchNorm2d):
				l.weight.detach().normal_(1.0, 0.02)
				nn.init.constant_(l.bias, 0)

	def forward(self, x):
		return self.model(x)

class VectorQuantizer(nn.Module):
	def __init__(self, num_hiddens, num_embeddings, embedding_dim):
		super(VectorQuantizer, self).__init__()

		self.num_embeddings = num_embeddings
		self.embedding_dim = embedding_dim

		self.embedding = nn.Embedding(self.num_embeddings, self.embedding_dim).to(device)
		self.embedding.weight.data.normal_(0, 0.2)

		self.logits_proj = nn.Conv2d(num_hiddens, self.num_embeddings, 1)

	def forward(self, inputs):

		# (16, 256, 8, 8) -> (16, 512, 8, 8)
		# (B, C, H, W) -> (B, N, H, W)
		logits = self.logits_proj(inputs)

		# (B, N, H, W)
		logits_onehot = F.gumbel_softmax(logits, tau=cfg.tau, hard=cfg.hard)

		# (16, 512, 8, 8) * (512, 256) -> (16, 256, 8, 8)
		quantized = torch.einsum('B N H W, N E -> B E H W',logits_onehot, self.embedding.weight)

		# (512,)
		dict_refs = logits_onehot.view(-1, self.num_embeddings).sum(0)

		perplexity = torch.exp(torch.special.entr(F.normalize(dict_refs, p=1, dim=0)).sum())

		return quantized, dict_refs, perplexity

class Transformer(nn.Module):
	def __init__(self, hidden_size, num_attention_heads) -> None:
		super(Transformer, self).__init__()
		self.num_attention_heads = num_attention_heads
		self.attention_head_size = hidden_size // num_attention_heads
		self.all_head_size = self.num_attention_heads * self.attention_head_size

		self.query = nn.Conv2d(hidden_size, self.all_head_size, 1)
		self.key = nn.Conv2d(hidden_size, self.all_head_size, 1)
		self.value = nn.Conv2d(hidden_size, self.all_head_size, 1)

	def transpose_for_scores(self, x):
		# (16, 256, 8, 8) => (16, 8, 8, 256) =>  (16, 64, 256) => (16, 64, 8, 32) => (16, 8, 64, 32)
		return x.permute(0, 2, 3, 1).contiguous().view(x.shape[0], -1, self.num_attention_heads, self.attention_head_size).permute(0, 2, 1, 3)

	def forward(self, inputs):
		
		# (16, 8, 64, 32)
		query_layer = self.transpose_for_scores(self.query(inputs))

		# (16, 8, 64, 32)
		key_layer = self.transpose_for_scores(self.key(inputs))

		# (16, 8, 64, 32)
		value_layer = self.transpose_for_scores(self.value(inputs))

		# (16, 8, 64, 32) * (16, 8, 32, 64) => (16, 8, 64, 64)
		attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
		
		# Normalize the attention scores to probabilities.
		# (16, 8, 64, 64) 
		attention_probs = nn.Softmax(dim=-1)(attention_scores)

		# (16, 8, 64, 64) * (16, 8, 64, 32) => (16, 8, 64, 32)
		context_layer = torch.matmul(attention_probs, value_layer)

		# (16, 64, 8, 32) => (16, 8, 8, 256) => (16, 256, 8, 8)
		return context_layer.permute(0, 2, 1, 3).contiguous().view(context_layer.shape[0],*inputs.shape[2:], -1).permute(0, 3, 1, 2).contiguous()

class PositionalEncoding(nn.Module):

	def __init__(self, hidden_size, encoding_shape):
		super(PositionalEncoding, self).__init__()

		max_len = encoding_shape.numel()

		# position (64,)
		position = torch.arange(max_len).unsqueeze(1)

		div_term = torch.exp(torch.arange(0, hidden_size, 2) * (-math.log(10000.0) / hidden_size))

		# (64, 256)
		pe = torch.zeros(max_len, hidden_size)

		pe[:, 0::2] = torch.sin(position * div_term)
		pe[:, 1::2] = torch.cos(position * div_term)

		# (64, 256) => (256, 64) => (1, 256, 8, 8)
		self.register_buffer('pe', pe.permute(1, 0).contiguous().view(1, -1, *encoding_shape))

	def forward(self, x):
		# (16, 256, 8, 8) + (1, 256, 8, 8) => (16, 256, 8, 8)
		return x + self.pe

class VQ_VAE(nn.Module):
	def __init__(self, hidden_size=256, num_embeddings=512, embedding_dim=256,bn=True, vq_coef=1, commit_coef=0.25, num_channels=3):
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
			# PositionalEncoding(hidden_size, torch.Size((8,8))),
			Transformer(hidden_size, 8),
			nn.BatchNorm2d(hidden_size),
		)

		# decoder
		self.decoder = nn.Sequential(
			# PositionalEncoding(hidden_size, torch.Size((8,8))),
			Transformer(hidden_size, 8),
			nn.BatchNorm2d(hidden_size),
			ResBlock(hidden_size, hidden_size),
			nn.BatchNorm2d(hidden_size),
			ResBlock(hidden_size, hidden_size),
			nn.ConvTranspose2d(hidden_size, hidden_size, kernel_size=4, stride=2, padding=1),
			nn.BatchNorm2d(hidden_size),
			nn.ReLU(inplace=True),
			nn.ConvTranspose2d(hidden_size, num_channels, kernel_size=4, stride=2, padding=1),
			nn.Tanh()
		)

		self.quantizer = VectorQuantizer(hidden_size,num_embeddings,embedding_dim)
        
		self.discriminator = Discriminator().to(device)

		self.vq_coef = vq_coef
		self.commit_coef = commit_coef
		self.model_dir = cfg.model_dir

		for l in self.modules():
			if isinstance(l, nn.Linear) or isinstance(l, nn.Conv2d):
				l.weight.detach().normal_(0, 0.2)
				# nn.init.constant_(l.bias, 0)

	def forward(self, x):
		z_e = self.encoder(x)

		# vq
		quantized, dict_refs, perplexity = self.quantizer(z_e)

		# decode
		decoded_x = self.decoder(quantized)

		return decoded_x, z_e, quantized, dict_refs, perplexity

	def loss(self, x, decoded_x, disc_factor):

		# real images scores
		x_pred = self.discriminator(x)

		# reconstructed images scores
		decoded_x_pred = self.discriminator(decoded_x)

		# total discriminator loss
		d_loss = disc_factor * 0.5 *torch.mean(2 - x_pred + decoded_x_pred)

		# generate loss
		g_loss = - decoded_x_pred.mean()

		# reconstruct loss
		r_loss = F.mse_loss(decoded_x, x) + 0.8 * F.l1_loss(decoded_x, x)

		λ = self.calculate_lambda(r_loss, d_loss, decoded_x)

		# final q loss with G loss
		q_loss = r_loss + disc_factor * λ * g_loss

		return q_loss, r_loss, - g_loss, d_loss

	def calculate_lambda(self, r_loss, d_loss, decoded_x):
		perceptual_loss_grads = torch.autograd.grad(r_loss, decoded_x, retain_graph=True)[0]
		gan_loss_grads = torch.autograd.grad(d_loss, decoded_x, retain_graph=True)[0]

		λ = torch.norm(perceptual_loss_grads) / (torch.norm(gan_loss_grads) + 1e-4)
		λ = torch.clamp(λ, 0, 1e4).detach()
		return 0.8 * λ

	def clamp_weights(self, clamp_min=-0.5, clamp_max=0.5):
		for p in self.discriminator.parameters():
			p.data.clamp_(clamp_min, clamp_max)
			
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

class LossesTrick(object):
	def __init__(self, num_embeddings):
		self.train_losses = []
		self.test_losses = []

		self.train_dict_refs = torch.zeros(num_embeddings,device='cpu')
		self.test_dict_refs = torch.zeros(num_embeddings,device='cpu')

	def init_epoch(self):
		self.test_losses = []
		self.test_dict_refs.data.fill_(0)

	def mean_error(self, mode = 'train'):
		losses = self.train_losses if mode == 'train' else self.test_losses
		return np.array(losses).mean(0) 

	def update_train(self, epoch, number_of_batches, dict_refs, perplexity, v_losses, print_count = 10):
		self.train_losses.append([v.cpu().item() for v in v_losses])
		self.train_dict_refs += dict_refs.cpu().detach()

		if number_of_batches % print_count == 0:
			print(f'Tranin epoch: {epoch} iter: {number_of_batches}','perplexity=%.3f, q_loss=%.3f(%.3f, %.3f), discriminator_loss=%.3f' % (perplexity,*(l.item() for l in v_losses)))

	def stat_train(self, epoch, epochs):
		perplexity = torch.exp(torch.special.entr(F.normalize(self.train_dict_refs, p=1, dim=0)).sum())
		print(f'Epoch [{epoch} / {epochs}]','perplexity=%.3f, average error: q_loss=%.3f(%.3f, %.3f), discriminator_loss=%.3f' % (perplexity, * self.mean_error()))

	def update_test(self, dict_refs, v_losses):
		self.test_losses.append([v.cpu().item() for v in v_losses])
		self.test_dict_refs += dict_refs.cpu().detach()

	def stat_test(self):
		perplexity = torch.exp(torch.special.entr(F.normalize(self.test_dict_refs, p=1, dim=0)).sum())
		print('Test perplexity = %.3f, Test Average error: q_loss=%.3f(%.3f, %.3f), discriminator_loss=%.3f' % (perplexity,* self.mean_error('test')))


class Trainer(object):

	def __init__(self, model, train_loader = None, test_loader = None, from_prtrained = False):
		self.model = model
		self.train_loader = train_loader
		self.test_loader = test_loader
		self.from_prtrained = from_prtrained
		self.writer = SummaryWriter(cfg.log_dir)
		self.write_image = lambda label, x, step: self.writer.add_image(label,make_grid(x[:16]),step)
		self.losses_trick = LossesTrick(cfg.num_embeddings)

	def train(self):
		
		if self.from_prtrained: self.model.load()

		q_optimizer = optim.Adam(self.model.parameters(), lr=cfg.learning_rate, amsgrad=False)

		discriminator_optimizer = torch.optim.Adam(self.model.discriminator.parameters(), lr=cfg.learning_rate, eps=1e-08, betas=(0.5, 0.9))

		for epoch in range(1, cfg.epochs + 1):
			
			self.losses_trick.init_epoch()

			# train
			self.do_train(epoch, q_optimizer, discriminator_optimizer)

			# save
			self.model.save(epoch)

	def do_train(self, epoch, q_optimizer, discriminator_optimizer):

		self.model.train()

		number_of_batches = 0

		# Grab the batch, we are only interested in images not on their labels
		for images, _ in self.train_loader:

			disc_factor = 0 if epoch < 2 else 1

			x = images.to(device)

			decoded_x, _, _, dict_refs, perplexity = self.model(x)

			losses = self.model.loss(x, decoded_x, disc_factor)
			
			# train generator
			q_optimizer.zero_grad()
			losses[0].backward(retain_graph=True)

			# train discriminator
			discriminator_optimizer.zero_grad()
			losses[-1].backward()

			q_optimizer.step()
			discriminator_optimizer.step()

			self.model.clamp_weights()

			if number_of_batches % 300 == 0:
				self.write_image(f'original/{number_of_batches}', x.mul(0.5).add(0.5), epoch)
				self.write_image(f'reconstructed/{number_of_batches}', decoded_x.mul(0.5).add(0.5), epoch)

			self.losses_trick.update_train(epoch, number_of_batches, dict_refs, perplexity, losses, 10)

			number_of_batches += 1

		self.losses_trick.stat_train(epoch, cfg.epochs)

	def test(self, epoch, load_model = False):

		if load_model: self.model.load()

		model.eval()

		write_image = lambda label, x, step: self.writer.add_image(label,make_grid(x[:64]),step)

		number_of_batches = 0

		with torch.no_grad():
			for data, _ in self.test_loader:

				data = data.to(device)

				decoded_x = None
				
				decoded_x = self.model(data)[0]

				write_image(f'original/{number_of_batches}', data.mul(0.5).add(0.5), epoch)
				write_image(f'reconstructed{number_of_batches}', decoded_x.mul(0.5).add(0.5), epoch)

				print(f'[{epoch}-{number_of_batches}]-original')
				
				number_of_batches += 1

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
	parser.add_argument("--batch_size", type=int, default=16)
	parser.add_argument("--test_batch_size", type=int, default=64)
	parser.add_argument("--epochs", type=int, default=4)
	parser.add_argument("--step", type=str, default='train')
	parser.add_argument("--decay", type=float, default=0.99)
	parser.add_argument("--data_dir", type=pathlib.Path, default="./data/CIFAR10")
	parser.add_argument("--model_dir", type=pathlib.Path, default="./model_data/vq_vae")
	parser.add_argument("--log_dir", type=pathlib.Path, default="./log/vq_vae")
	parser.add_argument("--tau", type=float, default=0.9)
	parser.add_argument("--hard", type=bool, default=True)
	parser.add_argument("--vgg_lpips_path", type=str, default='./model_data/vgg_lpips/vgg.pth')
	parser.add_argument("--use_gan", type=bool, default=True)
	parser.add_argument("--version", type=pathlib.Path, default="1.0")

def data_loader(data_dir, batch_size, train, shuffle = True):
	dataset = datasets.CIFAR10(root= data_dir, 
								train=train, 
								download=True, 
								transform= transforms.Compose([
									transforms.ToTensor(),
									transforms.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5))
								]))
	return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, pin_memory=True)	

def init_env(clean = True):
	script_name = os.path.splitext(os.path.basename(__file__))[0]
	cfg.model_dir = os.path.join(cfg.model_dir, script_name, cfg.version)
	cfg.log_dir = os.path.join(cfg.log_dir, script_name, cfg.version)

	clean_dir = lambda d: shutil.rmtree(d) if os.path.exists(d) else None

	clean_dir(cfg.model_dir)
	print("Clean ", cfg.model_dir)

	clean_dir(cfg.log_dir)
	print("Clean ", cfg.log_dir)

if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	init_args(parser)
	cfg = parser.parse_args()

	init_env()

	# device = torch.device('mps')

	print('device={}, log_dir={}, model_dir={}'.format(device,cfg.log_dir,cfg.log_dir))

	model = VQ_VAE(hidden_size=cfg.hidden_size, num_embeddings=cfg.num_embeddings, embedding_dim=cfg.embedding_dim)

	model.to(device)

	if cfg.step == 'train':
		trainer = Trainer(model, train_loader = data_loader(cfg.data_dir, cfg.batch_size, True), 
			test_loader = data_loader(cfg.data_dir, cfg.test_batch_size, False, shuffle=False),
			from_prtrained = False)
		trainer.train()

	elif cfg.step == 'test':
		trainer = Trainer(model, test_loader = data_loader(cfg.data_dir, cfg.test_batch_size, False))
		trainer.test(True)

	elif cfg.step == 'gen':
		trainer = Trainer(model)
		trainer.generate(16)