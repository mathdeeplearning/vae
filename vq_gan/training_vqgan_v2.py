import os
import argparse
import pathlib
from bleach import clean
import torch
import torch.nn.functional as F
from torchvision import utils as vutils
from discriminator import Discriminator
from lpips import LPIPS
from vqgan import VQGAN
from utils import  weights_init
from torch.utils.data import DataLoader
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torchvision.utils import make_grid
from PIL import Image
import shutil
import timeit
from torch.utils.tensorboard import SummaryWriter

class TrainVQGAN:
    def __init__(self,train_loader, args):
        self.train_loader = train_loader
        self.vqgan = VQGAN(args).to(device=args.device)
        self.discriminator = Discriminator(args).to(device=args.device)
        self.discriminator.apply(weights_init)
        self.perceptual_loss = LPIPS(args.vgg_lpips).eval().to(device=args.device) if args.use_percept_loss else None
        self.opt_vq, self.opt_disc = self.configure_optimizers(args)
        self.writer = SummaryWriter(args.log_dir)

        self.prepare_training()

        self.train(args)

    def configure_optimizers(self, args):
        lr = args.learning_rate
        opt_vq = torch.optim.Adam(
            list(self.vqgan.encoder.parameters()) +
            list(self.vqgan.decoder.parameters()) +
            list(self.vqgan.codebook.parameters()) +
            list(self.vqgan.quant_conv.parameters()) +
            list(self.vqgan.post_quant_conv.parameters()),
            lr=lr, eps=1e-08, betas=(args.beta1, args.beta2)
        )
        opt_disc = torch.optim.Adam(self.discriminator.parameters(), lr=lr, eps=1e-08, betas=(args.beta1, args.beta2))

        return opt_vq, opt_disc

    @staticmethod
    def prepare_training():
        os.makedirs("results", exist_ok=True)
        os.makedirs("checkpoints", exist_ok=True)

    def train(self, args):

        write_image = lambda label, x: self.writer.add_image(label,make_grid(x[:64]))
        steps_per_epoch = len(self.train_loader)
        for epoch in range(args.epochs):
            
            number_of_batches = 0

            for imgs, _ in self.train_loader:
                s1 = timeit.default_timer()

                imgs = imgs.to(device=args.device)
                decoded_images, _, q_loss = self.vqgan(imgs)

                disc_real = self.discriminator(imgs)
                disc_fake = self.discriminator(decoded_images)

                disc_factor = self.vqgan.adopt_weight(args.disc_factor, epoch*steps_per_epoch+number_of_batches, threshold=args.disc_start)

                s2 = timeit.default_timer()
                rec_loss = torch.abs(imgs - decoded_images)

                perceptual_loss = self.perceptual_loss(imgs, decoded_images) if args.use_percept_loss else 0

                perceptual_rec_loss = args.perceptual_loss_factor * perceptual_loss + args.rec_loss_factor * rec_loss
                
                perceptual_rec_loss = perceptual_rec_loss.mean()
                g_loss = -torch.mean(disc_fake)

                s3 = timeit.default_timer()

                λ = self.vqgan.calculate_lambda(perceptual_rec_loss, g_loss, decoded_images)
                vq_loss = perceptual_rec_loss + q_loss + disc_factor * λ * g_loss

                d_loss_real = torch.mean(F.relu(1. - disc_real))
                d_loss_fake = torch.mean(F.relu(1. + disc_fake))
                gan_loss = disc_factor * 0.5*(d_loss_real + d_loss_fake)

                s4 = timeit.default_timer()

                self.opt_vq.zero_grad()
                vq_loss.backward(retain_graph=True)

                self.opt_disc.zero_grad()
                gan_loss.backward()

                self.opt_vq.step()
                self.opt_disc.step()

                s5 = timeit.default_timer()

                # print("T1:%.3fs,T2:%.3fs,T3:%.3fs,T4:%.3fs"%(s2-s1,s3-s2,s4-s3,s5-s4))
                if number_of_batches % 10 == 0:
                    print("vq_loss={},gan_loss={}".format(vq_loss.cpu().item(),gan_loss.cpu().item()))

                if number_of_batches % 1000 == 0:
                    with torch.no_grad():
                        write_image('[{}-{}]original'.format(epoch, number_of_batches), imgs.mul(0.5).add(0.5))
                        write_image('[{}-{}]reconstructed'.format(epoch, number_of_batches), decoded_images.mul(0.5).add(0.5))
                        print("[{}-{}] Saved...".format(epoch, number_of_batches))

                number_of_batches += 1

            torch.save(self.vqgan.state_dict(), os.path.join("checkpoints", f"vqgan_epoch_{epoch}.pt"))

def data_loader(data_dir, batch_size, train, size):
	dataset = datasets.CIFAR10(root= data_dir, 
								train=train, 
								download=True, 
								transform= transforms.Compose([
                                    # transforms.Pad((size - 32)//2),
                                    transforms.Resize(size),
									transforms.ToTensor(),
									transforms.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5))
								]))
	return DataLoader(dataset, batch_size=batch_size, shuffle=True, pin_memory=True)

def init_env(cfg, clean = True):
	script_name = os.path.splitext(os.path.basename(__file__))[0]
	log_name = '%s_%s_%s'%(cfg.version, cfg.batch_size, cfg.use_percept_loss)

	cfg.log_dir = os.path.join(cfg.log_dir, script_name, log_name)

	if clean:
		shutil.rmtree(cfg.log_dir, ignore_errors=True)
		print("Clean ", cfg.log_dir)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="VQGAN")
    parser.add_argument('--latent-dim', type=int, default=256, help='Latent dimension n_z (default: 256)')
    parser.add_argument('--image-size', type=int, default=256, help='Image height and width (default: 256)')
    parser.add_argument('--num-codebook-vectors', type=int, default=1024, help='Number of codebook vectors (default: 256)')
    parser.add_argument('--beta', type=float, default=0.25, help='Commitment loss scalar (default: 0.25)')
    parser.add_argument('--image-channels', type=int, default=3, help='Number of channels of images (default: 3)')
    parser.add_argument('--dataset-path', type=str, default='../data/celeba/img_align_celeba', help='Path to data (default: /data)')
    # parser.add_argument('--device', type=str, default="cuda", help='Which device the training is on')
    parser.add_argument('--batch_size', type=int, default=2, help='Input batch size for training (default: 6)')
    parser.add_argument('--epochs', type=int, default=4, help='Number of epochs to train (default: 50)')
    parser.add_argument('--learning-rate', type=float, default=2.25e-05, help='Learning rate (default: 0.0002)')
    parser.add_argument('--beta1', type=float, default=0.5, help='Adam beta param (default: 0.0)')
    parser.add_argument('--beta2', type=float, default=0.9, help='Adam beta param (default: 0.999)')
    parser.add_argument('--disc-start', type=int, default=10000, help='When to start the discriminator (default: 0)')
    parser.add_argument('--disc-factor', type=float, default=1., help='')
    parser.add_argument('--rec-loss-factor', type=float, default=1., help='Weighting factor for reconstruction loss.')
    parser.add_argument('--perceptual-loss-factor', type=float, default=1., help='Weighting factor for perceptual loss.')
    parser.add_argument("--data-dir", type=pathlib.Path, default="../data/CIFAR10")
    parser.add_argument("--vgg-lpips", type=pathlib.Path, default="../vgg_lpips/vgg.pth")
    parser.add_argument("--log_dir", type=pathlib.Path, default="../log/vq_gan")
    parser.add_argument("--version", type=pathlib.Path, default="1.0")
    parser.add_argument("--use-percept-loss", type=lambda s: s.lower()=='true', default=True)

    args = parser.parse_args()

    init_env(args)

    print("Version={}, percept-loss={}, batch size:{}".format(args.version, args.use_percept_loss, args.batch_size))
    args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_loader = data_loader(args.data_dir, args.batch_size, True, args.image_size)

    train_vqgan = TrainVQGAN(train_loader, args)



