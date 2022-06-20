import numpy as np
import torch
from torch.utils.data import TensorDataset



class datasetgenemap(TensorDataset):
	"""Dataset wrapping unlabeled data tensors.
	No longer used.
	Each sample will be retrieved by indexing tensors along the first
	dimension.

	Arguments:
		datainput (numpy array): contains sample data.
	"""
	def __init__(self, datainput):
		self.data_tensor = torch.from_numpy(datainput).float()

	def __getitem__(self, index):
		return self.data_tensor[index].astype(np.float32)

	def __len__(self):
		return len(self.data_tensor)



class datagenemapclust(TensorDataset):
	"""Dataset wrapping labeled (cluster label) data tensors with cluster information.
	Used in data augmentation models
	Each sample will be retrieved by indexing tensors along the first
	dimension.

	Arguments:
		datainput (numpy array): contains sample data.
	"""
	def __init__(self, datainput, label):
		self.data_tensor = torch.from_numpy(datainput).float()
		self.maxnum = label.max()
		self.clustempty = np.zeros(self.maxnum + 1,'float32')
		self.label = label

	def __getitem__(self, index):
		image = self.data_tensor[index]
		cluster = self.clustempty.copy()
		cluster[self.label[index]] = 1
		return image, torch.from_numpy(cluster).float()

	def __len__(self):
		return len(self.data_tensor)



class wrap_gene_location(TensorDataset):
	"""Dataset wrapping labeled (cluster label) data tensors with cluster information.
	Used in data prediction models
	Each sample will be retrieved by indexing tensors along the first
	dimension.

	Arguments:
		datainput (numpy array): contains sample data.
	"""
	def __init__(self, datainput, label):
		self.data_tensor = torch.from_numpy(datainput).float()
		cord = label.to_numpy().astype('float32')
		cordx = cord[:,0]
		cordy = cord[:,1]
		self.xmin = cordx.min()-1
		self.ymin = cordy.min()-1
		self.xmax = cordx.max()+1
		self.ymax = cordy.max()+1
		self.cordx_norm = (cordx - self.xmin)/(self.xmax-self.xmin)
		self.cordy_norm = (cordy - self.ymin)/(self.ymax-self.ymin)
		self.imagedimension = self.data_tensor.shape
	def __getitem__(self, index):
		indexsample = index // self.imagedimension[2]
		indexspot = index % self.imagedimension[2]
		geneseq = self.data_tensor[indexsample,:,indexspot]
		cordinates = torch.tensor([self.cordx_norm[indexspot],self.cordy_norm[indexspot]])
		return geneseq, cordinates
	def __len__(self):
		return self.imagedimension[0] * self.imagedimension[2]


class wrap_gene_layer(TensorDataset):
	"""Dataset wrapping labeled (cluster label) data tensors with cluster information.
	Used in data prediction models
	Each sample will be retrieved by indexing tensors along the first
	dimension.

	Arguments:
		datainput (numpy array): contains sample data.
		layer (boolean): T if layer information is contained
		layerkey: the keyword for layer. Default is "Layer"
	"""
	def __init__(self, datainput, label, layerkey = "layer"):
		self.data_tensor = torch.from_numpy(datainput).float()
		getlayer = label[layerkey].to_numpy()
		self.layer = getlayer.astype('float32')
		self.layersunq = np.sort(np.unique(self.layer))
		self.nlayers = len(self.layersunq)
		self.imagedimension = self.data_tensor.shape
	def __getitem__(self, index):
		indexsample = index // self.imagedimension[2]
		indexspot = index % self.imagedimension[2]
		geneseq = self.data_tensor[indexsample,:,indexspot]
		layeri = int(self.layer[indexspot]) - 1
		layerv = np.zeros(self.nlayers-1)
		layerv[:layeri] = 1
		return geneseq, layerv
	def __len__(self):
		return self.imagedimension[0] * self.imagedimension[2]


class wrap_gene_domain(TensorDataset):
	"""Dataset wrapping labeled (cluster label) data tensors with cluster information.
	Used in data prediction models
	Each sample will be retrieved by indexing tensors along the first
	dimension.

	Arguments:
		datainput (numpy array): contains sample data.
		layer (boolean): T if layer information is contained
		layerkey: the keyword for layer. Default is "Layer"
	"""
	def __init__(self, datainput, label, layerkey = "layer"):
		self.data_tensor = torch.from_numpy(datainput).float()
		getlayer = label[layerkey].to_numpy()
		self.layer = getlayer.astype('float32')
		self.layersunq = np.sort(np.unique(self.layer))
		self.nlayers = len(self.layersunq)
		self.imagedimension = self.data_tensor.shape
	def __getitem__(self, index):
		indexsample = index // self.imagedimension[2]
		indexspot = index % self.imagedimension[2]
		geneseq = self.data_tensor[indexsample,:,indexspot]
		layeri = self.layer[indexspot].astype('int64')
		return geneseq, layeri
	def __len__(self):
		return self.imagedimension[0] * self.imagedimension[2]
