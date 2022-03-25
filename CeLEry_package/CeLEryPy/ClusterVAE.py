import torch
# from models import BaseVAE
from torch import nn
from torch.nn import functional as F
from . types_ import *


class ClusterVAE(nn.Module):

	def __init__(self,  
				 # in_channels: int,
				 latent_dim: int,
				 total_cluster: int,
				 hidden: List = None,
				 fgx = 2, fgy = 2,
				 **kwargs) -> None:
		super(ClusterVAE, self).__init__()

		self.latent_dim = latent_dim
		self.total_cluster = total_cluster
		
		scanx = fgx % 4 + 3
		scany = fgy % 4 + 3
		
		if hidden is None:
			hidden = [16, 8, 4, 8, 8]
		
		self.hidden = hidden
		# encoder
		self.encoderl1 = nn.Sequential( # like the Composition layer you built
			nn.Conv2d(1, hidden[0], [scanx,scany]),  # 76,  116		   178 208   # 80, 86
			nn.ReLU())
		# self.encoderl2 = nn.Sequential(nn.MaxPool2d(2, stride=2))  #38, 58		 # 78, 82
		self.encoderl3 = nn.Sequential(
			nn.Conv2d(hidden[0], hidden[1], 4, stride=2),
			nn.ReLU())	# 18, 28         # 38, 40
		self.encoderl4 = nn.Sequential(
			nn.Conv2d(hidden[1], hidden[2], 4, stride=2),     #18, 19
			nn.ReLU())	# 15, 25
		# decoder
		self.decoderl4 = nn.Sequential(
			nn.ConvTranspose2d(hidden[2], hidden[3], 4, stride=2),
			nn.ReLU())	# 35, 54
		self.decoderl3 = nn.Sequential(
			nn.ConvTranspose2d(hidden[3], hidden[4], 4, stride=2),  
			nn.ReLU())  # 38,57
		# self.decoderl2 = nn.Sequential(
		# 	nn.ConvTranspose2d(16, 8, 2, stride=2),  
		# 	nn.ReLU())	 #76, 114
		self.decoderl1 = nn.Sequential(
			nn.ConvTranspose2d(hidden[4], 1, [scanx,scany]) #,
			#nn.ReLU()
			#nn.Sigmoid()
			)
			
		self.enbedimx = int(((fgx - scanx + 1)/2-1)/2 -1)
		self.enbedimy = int(((fgy - scany + 1)/2-1)/2 -1)
		node_int = int(self.enbedimx * self.enbedimy * hidden[2])
		self.fc_mu = nn.Linear(node_int, latent_dim)
		self.fc_var = nn.Linear(node_int, latent_dim)
		self.decoder_input = nn.Linear(self.latent_dim + self.total_cluster + 1, node_int)
		
		
		if 'KLDw' in kwargs:
			self.kld_weight = kwargs['KLDw']
		else:
			self.kld_weight = 1
		
		self.seed = 0

	def encode(self, input: Tensor) -> List[Tensor]:
		"""
		Encodes the input by passing through the encoder network
		and returns the latent codes.
		:param input: (Tensor) Input tensor to encoder [N x C x H x W]
		:return: (Tensor) List of latent codes
		"""
		result = self.encoderl1(input)
		# result = self.encoderl2(result)
		result = self.encoderl3(result)
		result = self.encoderl4(result)
		result = torch.flatten(result, start_dim=1)

		# Split the result into mu and var components
		# of the latent Gaussian distribution
		mu = self.fc_mu(result)
		log_var = self.fc_var(result)

		return [mu, log_var]

	def decode(self, z: Tensor) -> Tensor:
		"""
		Maps the given latent codes
		onto the image space.
		:param z: (Tensor) [B x D]
		:return: (Tensor) [B x C x H x W]
		"""
		result = self.decoder_input(z)
		result = result.view(-1, self.hidden[2], self.enbedimx, self.enbedimy)
		result = self.decoderl4(result)
		result = self.decoderl3(result)
		# result = self.decoderl2(result)
		result = self.decoderl1(result)
		return result

	def reparameterize(self, mu: Tensor, logvar: Tensor) -> Tensor:
		"""
		Reparameterization trick to sample from N(mu, var) from
		N(0,1).
		:param mu: (Tensor) Mean of the latent Gaussian [B x D]
		:param logvar: (Tensor) Standard deviation of the latent Gaussian [B x D]
		:return: (Tensor) [B x D]
		"""
		std = torch.exp(0.5 * logvar)
		torch.manual_seed(self.seed)
		eps = torch.randn_like(std)
		return eps * std + mu

	def forward(self, input: Tensor, **kwargs) -> List[Tensor]:
		mu, log_var = self.encode(input[0])
		z = self.reparameterize(mu, log_var)
		zplus = torch.cat((z, input[1]), dim = 1)
		return  [self.decode(zplus), input, mu, log_var]

	def loss_function(self,
					  *args,
					  **kwargs) -> dict:
		"""
		Computes the VAE loss function.
		KL(N(\mu, \sigma), N(0, 1)) = \log \frac{1}{\sigma} + \frac{\sigma^2 + \mu^2}{2} - \frac{1}{2}
		:param args:
		:param kwargs:
		:return:
		"""
		recons = args[0]
		input = args[1]
		mu = args[2]
		log_var = args[3]

		kld_weight = self.kld_weight  # Account for the minibatch samples from the dataset
		
		
		recons_loss = F.mse_loss(recons, input[0])


		kld_loss = torch.mean(-0.5 * torch.sum(1 + log_var - mu ** 2 - log_var.exp(), dim = 1), dim = 0)

		loss = recons_loss + kld_weight * kld_loss
		return {'loss': loss, 'Reconstruction_Loss':recons_loss, 'KLD':-kld_loss}

		
		
class ClusterVAEmask(ClusterVAE):
	def __init__(self,  
		# in_channels: int,
		latent_dim: int,
		total_cluster: int,
		hidden: List = None,
		fgx = 2, fgy = 2,
		**kwargs) -> None:
		super(ClusterVAEmask, self).__init__(latent_dim, total_cluster, hidden, fgx, fgy,  **kwargs)
	
	def forward(self, input: Tensor, **kwargs) -> List[Tensor]:
		mu, log_var = self.encode(input[0])
		z = self.reparameterize(mu, log_var)
		zplus = torch.cat((z, input[1]), dim = 1)
		mask = (input[0] != 0) * 1
		return  [self.decode(zplus), input, mu, log_var, mask.float()]

	def loss_function(self,
					*args,
					**kwargs) -> dict:
		"""
		Computes the VAE loss function.
		KL(N(\mu, \sigma), N(0, 1)) = \log \frac{1}{\sigma} + \frac{\sigma^2 + \mu^2}{2} - \frac{1}{2}
		:param args:
		:param kwargs:
		:return:
		"""
		recons = args[0]
		input = args[1]
		mu = args[2]
		log_var = args[3]
		mask = args[4]

		kld_weight = self.kld_weight  # Account for the minibatch samples from the dataset
		
		
		recons_loss = F.mse_loss(recons * mask, input[0] * mask)


		kld_loss = torch.mean(-0.5 * torch.sum(1 + log_var - mu ** 2 - log_var.exp(), dim = 1), dim = 0)

		loss = recons_loss + kld_weight * kld_loss
		return {'loss': loss, 'Reconstruction_Loss':recons_loss, 'KLD':-kld_loss}
 

