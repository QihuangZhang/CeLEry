import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from tqdm import tqdm

class SpaCluster(object):
	def __init__(self):
		super(SpaCluster, self).__init__()
		self.l=None

	def set_l(self, l):
		self.l=l
        
	def train(self, model, train_loader, 
		num_epochs=5, learning_rate=1e-3, annealing = False, KLDwinc = 0.02, n_incr =50, RCcountMax = 40):
		self.learning_rate = 1e-2
		if (self.learning_rate > learning_rate):
			self.learning_rate = learning_rate
		self.model = model
		
		optimizer = optim.Adam(self.model.parameters(),
									lr=self.learning_rate, 
									weight_decay=1e-5) 
		RCcount = 0
		loss_min = 99999999
		for epoch in range(num_epochs):
			total_loss = 0
			for i, img in enumerate(tqdm(train_loader)):
				recon = self.model(img)
				loss = self.model.loss_function(*recon)
				loss.get("loss").backward()
				optimizer.step()
				optimizer.zero_grad()
				total_loss += loss.get("loss").data
			print('Epoch:{}, Loss:{:.4f}'.format(epoch+1, float(total_loss)))
			if (total_loss>loss_min):
				RCcount = RCcount + 1
				if (RCcount == RCcountMax):
					RCcount = 0
					self.learning_rate = self.learning_rate/2
					optimizer.param_groups[0]['lr'] = self.learning_rate
					loss_min = loss_min + 10
					print('New learning rate:{}'.format(float(self.learning_rate)))
			else:
				loss_min = total_loss
			if annealing:
				self.model.seed = epoch
				if epoch % n_incr == (n_incr-1):
					self.model.kld_weight = self.model.kld_weight + KLDwinc
					print('KLD weight annealing: increase {}. Now is :{:.4f}'.format(KLDwinc, float(self.model.kld_weight)))
					loss_min = loss_min + 500
			if (self.learning_rate < 1e-7):
				break
				
	def get_predict(self,train_loader):
		output = []
		for i, img in enumerate(train_loader):
			recon = self.model(img)
			output.append(recon[0].detach().numpy()[0,0,:,:])
		return(np.stack(output))
	
	def get_hidecode(self,train_loader):
		output = []
		for i, img in enumerate(tqdm(train_loader)):
			embedding1 = self.model.encoderl1(img.float())
			embedding2 = self.model.encoderl2(embedding1)
			embedding3 = self.model.encoderl3(embedding2)
			embedding4 = self.model.encoderl4(embedding3)
			output.append(embedding4)
		return(output)
		
	def deep_reshape(self, data, refer):
		"""
		Given generated data for a sample and a reference coordinates data, reshape the data by (location) X (Gene)
		:param data: (Numpy) [nsample X Gene X location_x X location_y]
		:return: (Numpy) [nsample X Gene X location(x X y filtered)]
		"""
		x = refer.iloc[:,0]
		y = refer.iloc[:,1]
		xmin = x.min()
		xmax = x.max()
		ymin = y.min()
		ymax = y.max()
		xlen = xmax - xmin + 1
		ylen = ymax - ymin + 1
		marker = np.zeros(xlen*ylen, dtype = bool)
		for i in range(refer.shape[0]):
			marker[(refer.iloc[i,0]-xmin)*ylen + refer.iloc[i,1] - ymin] = True
		final = data[:,:,marker]
		return(final)
	
	def fast_generation(self,train_loader, nsample):
		"""
		Given original gene-image data and the number of samples to be sampled
		:param train_loader
			   nsample: (Int) the number of samples
		:return: (Numpy) [nsample X Gene X location(x X y filtered)]
		"""
		output = []
		for i, img in enumerate(tqdm(train_loader)):
			outputinside = []
			self.model.seed = 0
			mu, log_var = self.model(img)[2:4]
			for j in range(nsample):
				self.model.seed = j
				z = self.model.reparameterize(mu, log_var)
				zplus = torch.cat((z, img[1]), dim = 1)
				outputi = self.model.decode(zplus)
				outputinside.append(outputi.detach().numpy()[0,0,:,:])
			output.append(np.stack(outputinside))
		final = np.stack(output)
		final2 = np.swapaxes( final,0,1)
		final3 = final2.reshape((final2.shape[0], final2.shape[1],-1) )
		return(final3)
		