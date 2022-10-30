import torch

from torch import nn
from torch.nn import functional as F
from . types_ import *
import math

class DNN(nn.Module):

	def __init__(self,  
				 in_channels: int,
				 hidden_dims: List = None,			 
				 **kwargs) -> None:
		super(DNN, self).__init__()
		
		if hidden_dims is None:
			hidden_dims = [200, 100, 50]

		self.fclayer1 = nn.Sequential( 
			nn.Linear(in_channels, hidden_dims[0]),
			# nn.BatchNorm1d(hidden_dims[0]),
			nn.ReLU())
		self.fclayer2 = nn.Sequential( 
			nn.Linear(hidden_dims[0], hidden_dims[1]),
			# nn.BatchNorm1d(hidden_dims[1]),
			nn.ReLU())
		self.fclayer3 = nn.Sequential( 
			nn.Linear(hidden_dims[1], hidden_dims[2]),
			# nn.BatchNorm1d(hidden_dims[2]),
			nn.ReLU())
		self.fclayer4 = nn.Sequential(
			nn.Linear(hidden_dims[2], 2),
			# nn.BatchNorm1d(2),
			nn.Sigmoid())

	def forward(self, input: Tensor, **kwargs) -> List[Tensor]:
		z = self.fclayer1(input[0])
		z = self.fclayer2(z)
		z = self.fclayer3(z)
		z = self.fclayer4(z)
		return  [z,input]

	def loss_function(self,
					  *args,
					  **kwargs) -> dict:
		"""
		Computes the spatial coordinates loss function
		:param args: results data and input matrix
		:return:
		"""
		cord_pred = args[0]
		input = args[1]

		loss = F.mse_loss(cord_pred, input[1])
		
		return {'loss': loss}


class DNN5(DNN):

	def __init__(self,  
				 in_channels: int,
				 hidden_dims: List = None,			 
				 **kwargs) -> None:
		super(DNN, self).__init__()
		
		if hidden_dims is None:
			hidden_dims = [200, 100, 50, 20, 10]

		self.fclayer1 = nn.Sequential( 
			nn.Linear(in_channels, hidden_dims[0]),
			nn.ReLU())
		self.fclayer2 = nn.Sequential( 
			nn.Linear(hidden_dims[0], hidden_dims[1]),
			nn.ReLU())
		self.fclayer3 = nn.Sequential( 
			nn.Linear(hidden_dims[1], hidden_dims[2]),
			nn.ReLU())
		self.fclayer4 = nn.Sequential( 
			nn.Linear(hidden_dims[2], hidden_dims[3]),
			nn.ReLU())
		self.fclayer5 = nn.Sequential( 
			nn.Linear(hidden_dims[3], hidden_dims[4]),
			nn.ReLU())
		self.fclayer6 = nn.Sequential(
			nn.Linear(hidden_dims[4], 2),
			nn.Sigmoid())

	def forward(self, input: Tensor, **kwargs) -> List[Tensor]:
		z = self.fclayer1(input[0])
		z = self.fclayer2(z)
		z = self.fclayer3(z)
		z = self.fclayer4(z)
		z = self.fclayer5(z)
		z = self.fclayer6(z)
		return  [z,input]



class DNNordinal_v2(DNN):
	"""
	 This model seperate the white matters from the grey matters (L1-L6)
	"""
	def __init__(self,  
		# in_channels: int,
		in_channels: int,
		num_classes: int,
		hidden_dims: List = None,			 
		**kwargs) -> None:
		super(DNNordinal, self).__init__(in_channels, hidden_dims, **kwargs)
		
		if hidden_dims is None:
			hidden_dims = [200, 100, 50]

		self.fclayer1 = nn.Sequential( 
			nn.Linear(in_channels, hidden_dims[0]),
			nn.ReLU())
		self.fclayer2 = nn.Sequential( 
			nn.Linear(hidden_dims[0], hidden_dims[1]),
			nn.ReLU())
		self.fclayer3 = nn.Sequential( 
			nn.Linear(hidden_dims[1], hidden_dims[2]),
			nn.ReLU())
		self.fclayer4 = nn.Sequential(
			nn.Linear(hidden_dims[2], 2))
		
		self.coral_bias = torch.nn.Parameter(
                torch.arange(num_classes - 1, 0, -1).float() / (num_classes-1))
	
	def forward(self, input: Tensor, **kwargs) -> List[Tensor]:
		"""
        Computes forward pass.
        Parameters
        -----------
        x : torch.tensor, shape=(num_examples, num_features)
            Input features.
        Returns
        -----------
        logits : torch.tensor, shape=(num_examples, num_classes-1)
        """
		z = self.fclayer1(input[0])
		z = self.fclayer2(z)
		z = self.fclayer3(z)
		z = self.fclayer4(z)
		logits = z[0,1] + self.coral_bias
		logitWM = z[0,0]
		return  [logits, logitWM, input]

	def loss_function(self,
					*args,
					**kwargs) -> dict:
		"""Computes the CORAL loss described in
		Cao, Mirjalili, and Raschka (2020)
		*Rank Consistent Ordinal Regression for Neural Networks
		   with Application to Age Estimation*
		Pattern Recognition Letters, https://doi.org/10.1016/j.patrec.2020.11.008
		Parameters
		----------
		logits : torch.tensor, shape(num_examples, num_classes-1)
			Outputs of the CORAL layer.
		levels : torch.tensor, shape(num_examples, num_classes-1)
			True labels represented as extended binary vectors
			(via `coral_pytorch.dataset.levels_from_labelbatch`).
		importance_weights : torch.tensor, shape=(num_classes-1,) (default=None)
			Optional weights for the different labels in levels.
			A tensor of ones, i.e.,
			`torch.ones(num_classes-1, dtype=torch.float32)`
			will result in uniform weights that have the same effect as None.
		reduction : str or None (default='mean')
			If 'mean' or 'sum', returns the averaged or summed loss value across
			all data points (rows) in logits. If None, returns a vector of
			shape (num_examples,)

		"""
		logits = args[0]
		logitWM = args[1]
		levelALL = args[2][1]
		
		levels = levelALL[0,:(levelALL.shape[1]-1)]
		levelWM = levelALL[0,levelALL.shape[1]-1]
		
		if not logits.shape == levels.shape:
			raise ValueError("Please ensure that logits (%s) has the same shape as levels (%s). "
							% (logits.shape, levels.shape))
		term1 = (F.logsigmoid(logits)*levels  + (F.logsigmoid(logits) - logits)*(1-levels))
		term2 = F.logsigmoid(logitWM)*levelWM  + (F.logsigmoid(logitWM) - logitWM + term1)*(1-levelWM)

		val = (-torch.sum(term2, dim=0))

		# loss = torch.sum(val)
		return {'loss': val}



class DNNordinal(DNN):
	def __init__(self,  
		# in_channels: int,
		in_channels: int,
		num_classes: int,
		hidden_dims: List = None,	
		importance_weights: List=None,
		**kwargs) -> None:
		super(DNNordinal, self).__init__(in_channels, hidden_dims, **kwargs)
		
		if hidden_dims is None:
			hidden_dims = [200, 100, 50]

		self.fclayer1 = nn.Sequential( 
			nn.Linear(in_channels, hidden_dims[0]),
			nn.Dropout(0.25),
			nn.ReLU())
		self.fclayer2 = nn.Sequential( 
			nn.Linear(hidden_dims[0], hidden_dims[1]),
			nn.ReLU())
		self.fclayer3 = nn.Sequential( 
			nn.Linear(hidden_dims[1], hidden_dims[2]),
			nn.ReLU())
		self.fclayer4 = nn.Sequential(
			nn.Linear(hidden_dims[2], 1))
		
		self.coral_bias = torch.nn.Parameter(
                torch.arange(num_classes - 1, 0, -1).float() / (num_classes-1))
				
		self.importance_weights = importance_weights
	
	def forward(self, input: Tensor, **kwargs) -> List[Tensor]:
		"""
        Computes forward pass.
        Parameters
        -----------
        x : torch.tensor, shape=(num_examples, num_features)
            Input features.
        Returns
        -----------
        logits : torch.tensor, shape=(num_examples, num_classes-1)
        """
		z = self.fclayer1(input[0])
		z = self.fclayer2(z)
		z = self.fclayer3(z)
		z = self.fclayer4(z)
		logits = z + self.coral_bias
		return  [logits, input]

	def loss_function(self,
					*args,
					**kwargs) -> dict:
		"""Computes the CORAL loss described in
		Cao, Mirjalili, and Raschka (2020)
		*Rank Consistent Ordinal Regression for Neural Networks
		   with Application to Age Estimation*
		Pattern Recognition Letters, https://doi.org/10.1016/j.patrec.2020.11.008
		Parameters
		----------
		logits : torch.tensor, shape(num_examples, num_classes-1)
			Outputs of the CORAL layer.
		levels : torch.tensor, shape(num_examples, num_classes-1)
			True labels represented as extended binary vectors
			(via `coral_pytorch.dataset.levels_from_labelbatch`).
		importance_weights : torch.tensor, shape=(num_classes-1,) (default=None)
			Optional weights for the different labels in levels.
			A tensor of ones, i.e.,
			`torch.ones(num_classes, dtype=torch.float32)`
			will result in uniform weights that have the same effect as None.
		reduction : str or None (default='mean')
			If 'mean' or 'sum', returns the averaged or summed loss value across
			all data points (rows) in logits. If None, returns a vector of
			shape (num_examples,)

		"""
		logits = args[0]
		levels = args[1][1]
		
		if not logits.shape == levels.shape:
			raise ValueError("Please ensure that logits (%s) has the same shape as levels (%s). "
							% (logits.shape, levels.shape))
		term1 = (F.logsigmoid(logits)*levels  + (F.logsigmoid(logits) - logits)*(1-levels))
		layerid = torch.sum(levels, dim = 1)
		
		if self.importance_weights is not None:
			term2 =  torch.mul(self.importance_weights[layerid.numpy()], term1.transpose(0,1))
		else:
			term2 =  term1.transpose(0,1)
		
		val = (-torch.sum(term2, dim=0))

		loss = torch.mean(val)
		return {'loss': loss}



class DNNregion(DNN):
	def __init__(self,  
		# in_channels: int,
		in_channels: int,
		alpha,
		hidden_dims: List = None,	
		**kwargs) -> None:
		super(DNNregion, self).__init__(in_channels, hidden_dims, **kwargs)
		
		if hidden_dims is None:
			hidden_dims = [200, 100, 50]

		self.fclayer1 = nn.Sequential( 
			nn.Linear(in_channels, hidden_dims[0]),
			nn.Dropout(0.25),
			nn.ReLU())
		self.fclayer2 = nn.Sequential( 
			nn.Linear(hidden_dims[0], hidden_dims[1]),
			nn.ReLU())
		self.fclayer3 = nn.Sequential( 
			nn.Linear(hidden_dims[1], hidden_dims[2]),
			nn.ReLU())
		self.fclayer4 = nn.Sequential(
			nn.Linear(hidden_dims[2], 5))
		
		self.alpha = alpha
			
	def forward(self, input: Tensor, **kwargs) -> List[Tensor]:
		"""
        Computes forward pass.
        Parameters
        -----------
        x : torch.tensor, shape=(num_examples, num_features)
            Input features.
        Returns
        -----------
        logits : torch.tensor, shape=(num_examples, num_classes-1)
        """
		z = self.fclayer1(input[0])
		z = self.fclayer2(z)
		z = self.fclayer3(z)
		z = self.fclayer4(z)
		
		cord = F.sigmoid( z[:,0:2] )
		r = F.softplus( z[:,2:4] )
		theta = F.sigmoid( z[:,4] ) * math.pi
		return  [cord, r, theta, input]

	def loss_function(self,
					*args,
					**kwargs) -> dict:
		"""Computes the loss described in Justin's
		Parameters
		----------
		logits : torch.tensor, shape(num_examples, num_classes-1)
			Outputs of the CORAL layer.
		levels : torch.tensor, shape(num_examples, num_classes-1)
			True labels represented as extended binary vectors
			(via `coral_pytorch.dataset.levels_from_labelbatch`).
		importance_weights : torch.tensor, shape=(num_classes-1,) (default=None)
			Optional weights for the different labels in levels.
			A tensor of ones, i.e.,
			`torch.ones(num_classes, dtype=torch.float32)`
			will result in uniform weights that have the same effect as None.
		reduction : str or None (default='mean')
			If 'mean' or 'sum', returns the averaged or summed loss value across
			all data points (rows) in logits. If None, returns a vector of
			shape (num_examples,)

		"""
		cord_pred = args[0]
		r_pred = args[1]
		theta_pred = args[2]
		input = args[3]
		
		roration_x = torch.cat((torch.cos(theta_pred).unsqueeze(1), torch.sin(theta_pred).unsqueeze(1)), 1)
		roration_y = torch.cat((torch.sin(theta_pred).unsqueeze(1), -torch.cos(theta_pred).unsqueeze(1)), 1)

		
		# MSE_Adjust = (cord_pred - input[1]) / (r_pred + 1e-7): old version - without considering rotation
		cord_decenter = cord_pred - input[1]
		semi_x = torch.sum(cord_decenter * roration_x , dim = 1)
		semi_y = torch.sum(cord_decenter * roration_y , dim = 1)
		cord_trans = torch.cat( (semi_x.unsqueeze(1), semi_y.unsqueeze(1)), 1)
		MSE_Adjust = cord_trans / (r_pred + 1e-7)
		
		area = torch.prod(r_pred)
		
		MSE_sum = torch.sum(torch.square(MSE_Adjust), dim = 1)
		Si =  (MSE_sum <= 1) * 1

		val = (self.alpha * (1-Si)+(1-self.alpha)*Si) * torch.abs(MSE_sum - 1)

		loss = torch.mean(val)

		return {'loss': loss, 'MSE_pure': MSE_sum, 'Inside_indic':Si, 'Area':area.detach().numpy()}



class DNNdomain(DNN):
	def __init__(self,  
		# in_channels: int,
		in_channels: int,
		num_classes: int,
		hidden_dims: List = None,	
		importance_weights: List=None,
		**kwargs) -> None:
		super(DNNdomain, self).__init__(in_channels, hidden_dims, **kwargs)
		
		if hidden_dims is None:
			hidden_dims = [200, 100, 50]

		self.fclayer1 = nn.Sequential( 
			nn.Linear(in_channels, hidden_dims[0]),
			nn.Dropout(0.25),
			nn.ReLU())
		self.fclayer2 = nn.Sequential( 
			nn.Linear(hidden_dims[0], hidden_dims[1]),
			nn.ReLU())
		self.fclayer3 = nn.Sequential( 
			nn.Linear(hidden_dims[1], hidden_dims[2]),
			nn.ReLU())
		self.fclayer4 = nn.Sequential(
			nn.Linear(hidden_dims[2], num_classes))
		self.importance_weights = importance_weights
	
	def forward(self, input: Tensor, **kwargs) -> List[Tensor]:
		"""
        Computes forward pass.
        Parameters
        -----------
        x : torch.tensor, shape=(num_examples, num_features)
            Input features.
        Returns
        -----------
        logits : torch.tensor, shape=(num_examples, num_classes-1)
        """
		z = self.fclayer1(input[0])
		z = self.fclayer2(z)
		z = self.fclayer3(z)
		logits = self.fclayer4(z)
		return  [logits, input]

	def loss_function(self,
					*args,
					**kwargs) -> dict:
		"""Computes the CORAL loss described in
		Cao, Mirjalili, and Raschka (2020)
		*Rank Consistent Ordinal Regression for Neural Networks
		   with Application to Age Estimation*
		Pattern Recognition Letters, https://doi.org/10.1016/j.patrec.2020.11.008
		Parameters
		----------
		logits : torch.tensor, shape(num_examples, num_classes-1)
			Outputs of the CORAL layer.
		levels : torch.tensor, shape(num_examples, num_classes-1)
			True labels represented as extended binary vectors
			(via `coral_pytorch.dataset.levels_from_labelbatch`).
		importance_weights : torch.tensor, shape=(num_classes-1,) (default=None)
			Optional weights for the different labels in levels.
			A tensor of ones, i.e.,
			`torch.ones(num_classes, dtype=torch.float32)`
			will result in uniform weights that have the same effect as None.
		reduction : str or None (default='mean')
			If 'mean' or 'sum', returns the averaged or summed loss value across
			all data points (rows) in logits. If None, returns a vector of
			shape (num_examples,)
		"""
		logits = args[0]
		levels = args[1][1]
		
		# if not logits.shape == levels.shape:
		# 	raise ValueError("Please ensure that logits (%s) has the same shape as levels (%s). "
		# 					% (logits.shape, levels.shape))
				
		if self.importance_weights is not None:
			loss = nn.CrossEntropyLoss(weight = self.importance_weights)
		else:
			loss = nn.CrossEntropyLoss()

		# layerid = torch.sum(levels, dim = 1)
		
		output = loss(logits, levels)

		return {'loss': output}

