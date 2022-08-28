# import scanpy as sc
import pandas as pd
import numpy as np
import scipy
import os
import scanpy as sc
from tqdm import tqdm
from skimage.metrics import structural_similarity as ssim
import pickle
from math import floor
import json
from . util import *

import matplotlib.pyplot as plt
import matplotlib.colors as clr

def get_SSIM (coords, referadata, referlocation, trainAnn, genelist):
	"""
		Compute the 
		:coords: numpy [Length_Locations x 2]: the predicted coordinates. Each cell in [0,1]
		:referadata: AnnData: the in the annotated data format
		:referlocation: dataframe [Length_Locations x 2]: the true location in the data
		:trainAnn: AnnData: the annotated data in the training set
		:return: float: the calculated SSIM statistics  
	"""
	referx = referlocation.iloc[:,0]
	refery = referlocation.iloc[:,1]
	trainx = trainAnn.obs.iloc[:,0]
	trainy = trainAnn.obs.iloc[:,1]
	referxmin = min(referx.min(), trainx.min())
	referxmax = max(referx.max(), trainx.max())
	referymin = min(refery.min(), trainy.min())
	referymax = max(refery.max(), trainy.max())
	xlen = referxmax - referxmin + 1
	ylen = referymax - referymin + 1
	SSIM_list = []
	cor_list = []
	GMSE_list = np.zeros(len(genelist))
	img_predict = np.array(np.ones((len(genelist),xlen,ylen,2))*0)
	img_truth = np.array(np.ones((len(genelist),xlen,ylen))*0)
	for i in tqdm(genelist):  
		## z2 = trainAnn.X[:,i] 
		## Add in the true training data in the prediction 
		#for inp in range(trainAnn.X.shape[0]):
			# if (z2[inp]!=0):
				# img_truth[i, trainx.iloc[inp]-referxmin,trainy.iloc[inp]-referymin] = z2[inp]
				# img_predict[i, trainx.iloc[inp]-referxmin,trainy.iloc[inp]-referymin,0] = z2[inp]
		z = referadata.X[:,i] 
		for inp in range(coords.shape[0]):
			if (z[inp]!=0):
				x_pixel_pred = floor(coords[inp,0]*xlen)
				y_pixel_pred = floor(coords[inp,1]*ylen)
				img_truth[i, referx.iloc[inp]-referxmin,refery.iloc[inp]-referymin] = z[inp]
				# img_predict[i, referx.iloc[inp]-referxmin,refery.iloc[inp]-referymin,0] = z[inp]
				## If a spot has multiple predictions, take the average
				if (img_predict[i,x_pixel_pred, y_pixel_pred,1] == 0):
					img_predict[i,x_pixel_pred, y_pixel_pred,0] = z[inp]
				else:
					img_predict[i,x_pixel_pred, y_pixel_pred,0] = (img_predict[i,x_pixel_pred, y_pixel_pred,0] * img_predict[i,x_pixel_pred, y_pixel_pred,1] + z[inp]) / (img_predict[i,x_pixel_pred, y_pixel_pred,1] + 1)
					img_predict[i,x_pixel_pred, y_pixel_pred,1] = img_predict[i,x_pixel_pred, y_pixel_pred,1] + 1
		img_truth_i = centralize2(img_truth[i,:,:])
		img_predict_i = centralize2(img_predict[i,:,:,0])
		ssim_i = ssim(img_truth_i, img_predict_i, data_range= 1)
		SSIM_list.append(ssim_i)
		corr_i = scipy.stats.pearsonr(img_truth_i.flatten(), img_predict_i.flatten())
		cor_list.append(corr_i[0])
		GMSE_list[i] = (np.square(img_truth_i.flatten() - img_predict_i.flatten())).mean()
	return([np.array(SSIM_list),img_predict,img_truth,np.array(cor_list),GMSE_list])

# def get_genemap_MSE (coords, referadata, referlocation, trainAnn, genelist):
	# """
		# Compute the 
		# :coords: numpy [Length_Locations x 2]: the predicted coordinates. Each cell in [0,1]
		# :referadata: AnnData: the in the annotated data format
		# :referlocation: dataframe [Length_Locations x 2]: the true location in the data
		# :trainAnn: AnnData: the annotated data in the training set
		# :return: float: the calculated SSIM statistics  
	# """
	# referx = referlocation.iloc[:,0]
	# refery = referlocation.iloc[:,1]
	# trainx = trainAnn.obs.iloc[:,0]
	# trainy = trainAnn.obs.iloc[:,1]
	# referxmin = min(referx.min(), trainx.min())
	# referxmax = max(referx.max(), trainx.max())
	# referymin = min(refery.min(), trainy.min())
	# referymax = max(refery.max(), trainy.max())
	# xlen = referxmax - referxmin + 1
	# ylen = referymax - referymin + 1
	# SSIM_list = []
	# cor_list = []
	# img_predict = np.array(np.ones((len(genelist),xlen,ylen,2))*0)
	# img_truth = np.array(np.ones((len(genelist),xlen,ylen))*0)
	# MSE_GE = np.zeros(len(genelist))
	# # for i in tqdm(genelist):  
	# i = 1
	# z2 = trainAnn.X[:,i] 
	# for inp in range(trainAnn.X.shape[0]):
		# if (z2[inp]!=0):
			# img_truth[i, trainx.iloc[inp]-referxmin,trainy.iloc[inp]-referymin] = z2[inp]
	# z = referadata.X[:,i] 
	# for inp in range(coords.shape[0]):
		# if (z[inp]!=0):
			# img_truth[i, referx.iloc[inp]-referxmin,refery.iloc[inp]-referymin] = z[inp]
	# MSE_GEi = np.zeros(coords.shape[0])
	# z_truth = np.zeros(coords.shape[0])
	# for inp in range(coords.shape[0]):
		# if (z[inp]!=0):
			# x_pixel_pred = floor(coords[inp,0]*xlen)
			# y_pixel_pred = floor(coords[inp,1]*ylen)
			# ## Compute MSE for the predicted points
			# z_truth[inp] = img_truth[i, x_pixel_pred, y_pixel_pred]
			# MSE_GEi[inp] = (z[inp] - z_truth[inp]) * (z[inp] - z_truth[inp])
	# MSE_GE[i] = np.mean(MSE_GEi)
	# return ([MSE_GE, MSE_GEi, z, z_truth])


def Pred_Density (coords, referadata, referlocation):
	"""
		Report the prediction distribution regarding the number of prediction fall in each spot/pixel
		:coords: numpy [Length_Locations x 2]: the predicted coordinates. Each cell in [0,1]
		:referadata: AnnData: the in the annotated data format
		:referlocation: dataframe [Length_Locations x 2]: the true location in the data
		:return: float: the calculated SSIM statistics  
	"""
	referx = referlocation.iloc[:,0]
	refery = referlocation.iloc[:,1]
	referxmin = referx.min()
	referxmax = referx.max()
	referymin = refery.min()
	referymax = refery.max()
	xlen = referxmax - referxmin + 1
	ylen = referymax - referymin + 1
	
	img_predict = np.array(np.ones((xlen,ylen))*0)
	img_truth = np.array(np.ones((xlen,ylen))*0)
	for inp in range(coords.shape[0]):
		x_pixel_pred = floor(coords[inp,0]*xlen)
		y_pixel_pred = floor(coords[inp,1]*ylen)
		img_predict[x_pixel_pred, y_pixel_pred] = img_predict[x_pixel_pred, y_pixel_pred] + 1
	return(img_predict)
	

def RelocationPlot (coords, referlocation, filename = None):
	"""
		Report the prediction distribution regarding the number of prediction fall in each spot/pixel
		:coords: numpy [Length_Locations x 2]: the predicted coordinates. Each cell in [0,1]
		:referlocation: dataframe [Length_Locations x 2]: the true location in the data
		:return: float: the calculated SSIM statistics  
	"""
	plt.figure()
	referx = referlocation.iloc[:,0]
	refery = referlocation.iloc[:,1]
	referxmin = referx.min()
	referxmax = referx.max()
	referymin = refery.min()
	referymax = refery.max()
	xlen = referxmax - referxmin + 1
	ylen = referymax - referymin + 1
	#	
	for inp in range(coords.shape[0]):
		x_pixel_pred = floor(coords[inp,0]*xlen)
		y_pixel_pred = floor(coords[inp,1]*ylen)
		xvalues = [referx[inp]- referxmin,x_pixel_pred]
		yvalues = [refery[inp]- referymin,y_pixel_pred]
		plt.plot(yvalues, xvalues, color = "#555b6e", alpha = 0.5, lw = 0.8)
	#
	plt.scatter(refery - referymin, referx - referxmin,  marker='^', color= "#bee3db") # 
	plt.scatter(coords[:,1]*ylen, coords[:,0]*xlen, marker='o', color= "#ffd6ba") #, color= "ffd6ba"
	#
	plt.gca().invert_yaxis()
	if filename is None:
		plt.show()
	else:
		plt.savefig(filename + '.pdf')




def UncertaintyPlot (cords, filename, hist):
	"""
		Produce an overlay plot of the area of the uncertainty region on the top of the histology map
	"""
	cords = cords.copy()
	plt.figure()
	plt.imshow(hist)
	color_self = clr.LinearSegmentedColormap.from_list('pink_green', ['#3AB370',"#EAE7CC","#FD1593"], N=256)
	cx = cords.obs["x"].to_numpy() * 50
	cxm = cx - cx.min()
	cy = cords.obs["y"].to_numpy() * 50
	cym = cy - cy.min()
	plt.scatter(cym, cxm, c=cords.X, s=15, cmap=color_self)
	plt.colorbar()
	plt.savefig(filename + '.pdf')



def report_prop_method (folder, name, dataSection2, traindata, Val_loader, outname = ""):
	"""
		Report the results of the proposed methods in comparison to the other method
		:folder: string: specified the folder that keep the proposed DNN method
		:name: string: specified the name of the DNN method, also will be used to name the output files
		:dataSection2: AnnData: the data of Section 2
		:traindata: AnnData: the data used in training data. This is only needed for compute SSIM
		:Val_loader: Dataload: the validation data from dataloader
		:outname: string: specified the name of the output, default is the same as the name
	"""
	if outname == "":
		outname = name
	filename2 = "{folder}/{name}.obj".format(folder = folder, name = name)
	filehandler = open(filename2, 'rb') 
	DNNmodel = pickle.load(filehandler)
	#
	total_loss_org = []
	coords_predict = np.zeros((dataSection2.obs.shape[0],2))
	for i, img in enumerate(Val_loader):
		recon = DNNmodel(img)
		coords_predict[i,:] = recon[0].detach().numpy()
		loss = DNNmodel.loss_function(*recon)
		total_loss_org.append(loss.get("loss").data)
	#
	losstotal_itemize = [x.item() for x in total_loss_org]
	losstotal = sum(losstotal_itemize)
	print('Loss for enhancement data only:{:.4f}'.format(float(losstotal)))
	#
	RelocationPlot (coords_predict, referadata = dataSection2, referlocation = dataSection2.obs[["x","y"]], filename = "{folder}/{name}_location".format(folder = folder, name = outname))
	#
	Density = Pred_Density(coords_predict, referadata = dataSection2, referlocation = dataSection2.obs[["x","y"]])
	plotGeneImg(Density, filename = "{folder}/{name}_density".format(folder = folder, name = outname), plotcolor = "Oranges")
	#
	SSIM_result =  get_SSIM(coords_predict, referadata = dataSection2, trainAnn = traindata, referlocation = dataSection2.obs[["x","y"]], genelist = range(dataSection2.X.shape[1]))
	plotGeneImg(centralize2(SSIM_result[1][0,:,:,0]), filename = "{folder}/{name}_SSIM_1".format(folder = folder, name = outname))
	plotGeneImg(centralize2(SSIM_result[2][0,:,:]), filename = "{folder}/{name}_SSIM_anchor_1".format(folder = folder, name = outname))
	plotGeneImg(centralize2(SSIM_result[1][1,:,:,0]), filename = "{folder}/{name}_SSIM_2".format(folder = folder, name = outname))
	plotGeneImg(centralize2(SSIM_result[2][1,:,:]), filename = "{folder}/{name}_SSIM_anchor_2".format(folder = folder, name = outname))
	plotGeneImg(centralize2(SSIM_result[1][2,:,:,0]), filename = "{folder}/{name}_SSIM_3".format(folder = folder, name = outname))
	plotGeneImg(centralize2(SSIM_result[2][2,:,:]), filename = "{folder}/{name}_SSIM_anchor_3".format(folder = folder, name = outname))
	np.save("{folder}/{name}_SSIM.npy".format(folder = folder, name = outname), np.array(SSIM_result[0]))
	np.save("{folder}/{name}_cor.npy".format(folder = folder, name = outname), np.array(SSIM_result[3]))
	np.save("{folder}/{name}_MSE.npy".format(folder = folder, name = outname), np.array(SSIM_result[4]))
	del SSIM_result
	result_org = sc.AnnData(X= np.expand_dims(np.array(losstotal_itemize),  axis = 1))
	result_org.obs = dataSection2.obs
	getGeneImg(result_org, emptypixel = -0.1)
	plotGeneImg(result_org.GeneImg[0,:,:],filename = "{folder}/{name}_MSE".format(folder = folder, name = outname), range = (-0.05,0.3))
	results = {"MSE":str(round(losstotal,4))}
	with open("{folder}/{name}_results.txt".format(folder = folder, name = outname), 'w') as file:
		file.write(json.dumps(results))
	return losstotal



def report_region (folder, name, dataSection2, traindata, Val_loader, hist, outname = ""):
	"""
		Report the results of the proposed methods in comparison to the other method
		:folder: string: specified the folder that keep the proposed DNN method
		:name: string: specified the name of the DNN method, also will be used to name the output files
		:dataSection2: AnnData: the data of Section 2
		:traindata: AnnData: the data used in training data. This is only needed for compute SSIM
		:Val_loader: Dataload: the validation data from dataloader
		:outname: string: specified the name of the output, default is the same as the name
		:ImageSec2: Numpy: the image data that are refering to
	"""
	if outname == "":
		outname = name
	filename2 = "{folder}/{name}.obj".format(folder = folder, name = name)
	filehandler = open(filename2, 'rb') 
	DNNmodel = pickle.load(filehandler)
	#
	total_loss_org = []
	area_record = []
	region_predict = np.zeros((dataSection2.obs.shape[0],5))
	for i, img in enumerate(Val_loader):
		recon = DNNmodel(img)
		region_predict[i,0:2] = recon[0].detach().numpy()
		region_predict[i,2:4] = recon[1].detach().numpy()
		region_predict[i,4] = recon[2].detach().numpy()
		loss = DNNmodel.loss_function(*recon)
		total_loss_org.append(loss.get("Inside_indic").data)
		area_record.append(loss.get("Area").item())
	#
	losstotal_itemize = [x.item() for x in total_loss_org]
	losstotal = np.mean(losstotal_itemize)
	result_area = sc.AnnData(X= np.expand_dims(np.array(area_record),  axis = 1))
	result_area.obs = dataSection2.obs
	getGeneImg(result_area, emptypixel = -0.1)
	plotGeneImg(result_area.GeneImg[0,:,:],filename = "{folder}/{name}_AREA".format(folder = folder, name = outname))
	UncertaintyPlot(result_area,filename = "{folder}/{name}_AREA_overlay".format(folder = folder, name = outname), hist = hist)
	return losstotal
