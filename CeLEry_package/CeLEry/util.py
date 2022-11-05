import pandas as pd
import numpy as np
import scipy
import os
import scanpy as sc
from tqdm import tqdm
# from skimage.metrics import structural_similarity as ssim
# import pickle
from math import floor
import json
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
# from seaborn import heatmap as seaheatmap
from scipy.sparse import issparse


import matplotlib.pyplot as plt
# from anndata import AnnData,read_csv,read_text,read_mtx
# from scipy.sparse import issparse

def prefilter_cells(adata,min_counts=None,max_counts=None,min_genes=200,max_genes=None):
	if min_genes is None and min_counts is None and max_genes is None and max_counts is None:
		raise ValueError('Provide one of min_counts, min_genes, max_counts or max_genes.')
	id_tmp=np.asarray([True]*adata.shape[0],dtype=bool)
	id_tmp=np.logical_and(id_tmp,sc.pp.filter_cells(adata.X,min_genes=min_genes)[0]) if min_genes is not None  else id_tmp
	id_tmp=np.logical_and(id_tmp,sc.pp.filter_cells(adata.X,max_genes=max_genes)[0]) if max_genes is not None  else id_tmp
	id_tmp=np.logical_and(id_tmp,sc.pp.filter_cells(adata.X,min_counts=min_counts)[0]) if min_counts is not None  else id_tmp
	id_tmp=np.logical_and(id_tmp,sc.pp.filter_cells(adata.X,max_counts=max_counts)[0]) if max_counts is not None  else id_tmp
	adata._inplace_subset_obs(id_tmp)
	adata.raw=sc.pp.log1p(adata,copy=True) #check the rowname 
	print("the var_names of adata.raw: adata.raw.var_names.is_unique=:",adata.raw.var_names.is_unique)
   

def prefilter_genes(adata,min_counts=None,max_counts=None,min_cells=10,max_cells=None):
	if min_cells is None and min_counts is None and max_cells is None and max_counts is None:
		raise ValueError('Provide one of min_counts, min_genes, max_counts or max_genes.')
	id_tmp=np.asarray([True]*adata.shape[1],dtype=bool)
	id_tmp=np.logical_and(id_tmp,sc.pp.filter_genes(adata.X,min_cells=min_cells)[0]) if min_cells is not None  else id_tmp
	id_tmp=np.logical_and(id_tmp,sc.pp.filter_genes(adata.X,max_cells=max_cells)[0]) if max_cells is not None  else id_tmp
	id_tmp=np.logical_and(id_tmp,sc.pp.filter_genes(adata.X,min_counts=min_counts)[0]) if min_counts is not None  else id_tmp
	id_tmp=np.logical_and(id_tmp,sc.pp.filter_genes(adata.X,max_counts=max_counts)[0]) if max_counts is not None  else id_tmp
	adata._inplace_subset_var(id_tmp)


def prefilter_specialgenes(adata,Gene1Pattern="ERCC",Gene2Pattern="MT-"):
	id_tmp1=np.asarray([not str(name).startswith(Gene1Pattern) for name in adata.var_names],dtype=bool)
	id_tmp2=np.asarray([not str(name).startswith(Gene2Pattern) for name in adata.var_names],dtype=bool)
	id_tmp=np.logical_and(id_tmp1,id_tmp2)
	adata._inplace_subset_var(id_tmp)


def centralize (data):
	datanew = data.copy()
	for i in tqdm(range(datanew.shape[0])):
		z = datanew[i,0,:,:]
		zmin = z.min()
		zmax = z.max()
		if (zmax != zmin):
			datanew[i,0,:,:] = (z-zmin)/(zmax-zmin)
		else:
			datanew[i,0,:,:] =  z / (zmax + 1)
	return datanew
	
def centralize2 (data):
	datanew = data.copy()
	mask = (datanew != 0) * 1
	zmin = datanew.min()
	zmax = datanew.max()
	if (zmax != zmin):
		datanew = (datanew-zmin)/(zmax-zmin) * mask
	else:
		datanew =  datanew / (zmax + 1) * mask
	return datanew 


def getGeneImg (datainput, emptypixel, obsset = None):
	# Transform the AnnData file into Genes of images
	# datainput: the input data of AnnData object
	# obsset: the set of location column names if they are not x_cord, y_cord
	# emptypixel: a float that indicate the value on the missing pixel
	adata = (datainput.X.A if issparse(datainput.X) else datainput.X)
	if obsset is None:
		x = datainput.obs["x_cord"]
		y = datainput.obs["y_cord"]
	else:
		x = datainput.obs[obsset[0]]
		y = datainput.obs[obsset[1]]  	
	xmin = x.min()
	xmax = x.max()
	ymin = y.min()
	ymax = y.max()
	## Append a one-side padding if the axis is odd
	## if ((xmax-xmin+1) % 2 == 0):
	## 	xlim = xmax-xmin+2
	## else:
	xlim = xmax-xmin+1
	## if ((ymax-ymin+1) % 2 == 0):
	## 	ylim = ymax-ymin+2
	## else:
	ylim = ymax-ymin+1
	shape = (xlim,ylim)
	all_arr = []
	firstIteration = True
	for i in tqdm(range(adata.shape[1])):
		z = adata[:,i] 
		zmin = z.min()
		zmax = z.max()
		# create array for image : zmax+1 is the default value
		img = np.array(np.ones(shape)*emptypixel)
		for inp in range(x.shape[0]):
			if (z[inp]!=emptypixel):
				img[x.iloc[inp]-xmin,y.iloc[inp]-ymin]=z[inp]
		all_arr.append(img)
	datainput.GeneImg = np.stack(all_arr)			

# def getGeneImgSparse (adata, emptypixel):
# 	# Transform the AnnData file into Genes of images for sparse matrix format
# 	# adata: the input data of AnnData object
# 	# emptypixel: a float that indicate the value on the missing pixel
# 	x = adata.obs.iloc[:,0]
# 	y = adata.obs.iloc[:,0]
# 	xmin = x.min().iloc[0]
# 	xmax = x.max().iloc[0]
# 	ymin = y.min().iloc[0]
# 	ymax = y.max().iloc[0]
# 	## Append a one-side padding if the axis is odd
# 	if ((xmax-xmin+1) % 2 == 0):
# 		xlim = xmax-xmin+2
# 	else:
# 		xlim = xmax-xmin+1
# 	if ((ymax-ymin+1) % 2 == 0):
# 		ylim = ymax-ymin+2
# 	else:
# 		ylim = ymax-ymin+1 
# 	shape = (xlim,ylim)
# 	all_arr = []
# 	firstIteration = True
# 	for i in tqdm(range(adata.X.shape[1])):
# 		z = adata.X[:,i] 
# 		zmin = z.min()
# 		zmax = z.max()
# 		# create array for image : zmax+1 is the default value
# 		img = np.array(np.ones(shape)*emptypixel)
# 		for inp in range(x.shape[0]):
# 			if (z[inp,0]!=emptypixel):
# 				img[x.iloc[inp]-xmin,y.iloc[inp]-ymin]=z[inp,0]
# 		all_arr.append(img)
# 	adata.GeneImg = np.stack(all_arr)		


def plotGeneImg (img, filename = None, range = None, plotcolor = 'YlGnBu'):
	# set mask on default value
	# img.mask = (img==0)
	plt.figure()
	shape = img.shape
	# set a gray background for test
	img_bg_test =  np.zeros(shape)
	cmap_bg_test = plt.get_cmap('gray')
	plt.imshow(img_bg_test,cmap=cmap_bg_test,interpolation='none',vmin=0,vmax=6)
	# plot
	cmap = plt.get_cmap(plotcolor)
	plt.imshow(img,cmap=cmap,interpolation='none')
	if range is not None:
		plt.clim(range[0], range[1])
	plt.colorbar()
	if filename is None:
		plt.show()
	else:
		plt.savefig(filename + '.pdf')


def plotarrangefile (bdataexpand, foldername, label, Path = "../output/"):
	try:
		os.mkdir( Path + foldername)
	except FileExistsError:
		print("Folder already exists")
	except FileNotFoundError:
		print("The path before foldername is not found.")
	else:
		print ("Folder {foldername} is successfully created".format(foldername = foldername))
	ncategory = np.bincount(label).shape[0]
	for i in range(ncategory):
		try:
			os.mkdir(Path + "{foldername}/{labels}/".format(foldername = foldername, labels=i))
		except FileExistsError:
			print("Folder for group {i} already exists".format(i = i))
		except FileNotFoundError:
			print("The path before foldername is not found.")
		else:
			print ("Folder group {i} is successfully created".format(i = i))
	for i in tqdm(range(bdataexpand.shape[0])):
		plotGeneImg(bdataexpand[i,0,:,:], filename = Path + "{foldername}/{labels}/fig{i}".format(foldername = foldername, labels = label[i], i = i))


def get_zscore (adata, mean = None, sd = None ):
	genotypedata = (adata.X.A if issparse(adata.X) else adata.X)
	if mean is None:
		genemean = np.mean(genotypedata, axis =0)
		genesd = np.std(genotypedata, axis = 0)
	else:
		genemean = mean
		genesd = sd
	try:
		if adata.standardize is not True:
				datatransform = (genotypedata - genemean) / genesd
				adata.X = datatransform
				adata.genemean = genemean
				adata.genesd = genesd
				adata.standardize = True
		else:
			print("Data has already been z-scored")
	except AttributeError:
		datatransform = (genotypedata - genemean) / genesd
		adata.X = datatransform
		adata.genemean = genemean
		adata.genesd = genesd
		adata.standardize = True


def get_histlgy_color (coords, refer, histimage, beta = 49):
	"""
		According to the predicted coordinates. Get histology image from the coordinates.
		:coords: numpy [Length_Locations x 2]: the predicted coordinates. Each cell in [0,1]
		:refer: dataframe [Length_Locations x 2]: the true location in the data
		:beta: int [1]: to control the range of neighbourhood when calculate grey vale for one spot
		:histimage: numpy [xlen x ylen]: Histology data
		:return: numpy [ xlen x ylen x 3 (RGB values)]
	"""
	beta_half=round(beta/2)
	
	imageshape = histimage.shape
	maxx = imageshape[0]
	maxy = imageshape[1]
	
	referx = refer.iloc[:,0]
	refery = refer.iloc[:,1]
	referxmin = referx.min()
	referxmax = referx.max()
	referymin = refery.min()
	referymax = refery.max()
	xlen = referxmax - referxmin + 1
	ylen = referymax - referymin + 1
	canvus = np.array(np.ones((xlen,ylen,3))*255)  ## backgroud color white: 255 black: 0
			
	for i in range(coords.shape[0]):
		# Step 1: Capture the corresponding from the histology information
		x_pixel_pred = round(coords[i,0]*maxx)
		y_pixel_pred = round(coords[i,1]*maxy)
		subimage = histimage[max(0,x_pixel_pred-beta_half):min(maxx,x_pixel_pred+beta_half+1), max(0,y_pixel_pred-beta_half):min(maxy,y_pixel_pred+beta_half+1)]
		subimage_mean = np.mean(np.mean(subimage, axis = 0), axis = 0)
		# Place the color on the canvus of original map
		referx_current = refer.iloc[i,0]
		refery_current = refer.iloc[i,1]
		canvus[referx_current - referxmin, refery_current - referymin,:] = subimage_mean
	return(canvus)	
		
def printimage (image, path):
	"""
		According to the predicted coordinates. Get histology image from the coordinates.
		:image:  numpy [xlen x ylen x 3 (RGB values)]: the image
		:path: string [1]: the path to print the plot
	"""
	plt.imshow(image/255)
	plt.savefig(path + '.pdf')
	
	
def closest_node(node, nodes):
    dist_2 = np.sum((nodes - node)**2, axis=1)
    return np.argmin(dist_2)
	
def get_Layer_LIBD (adata, coords_predict, referann):
	"""
		Get the layer label of the LIBD data
		:adata: the main adata that are working with
		:coords_predict: Numpy [n x 2]: the predicted coordinates based on deep neural network
		:referann: AnnData: the AnnData for the reference data. Usually the training data
	"""
	referlocation = referann.obs.copy()
	referx = referlocation.iloc[:,0]
	refery = referlocation.iloc[:,1]
	referxmin = referx.min()
	referxmax = referx.max()
	referymin = refery.min()
	referymax = refery.max()
	xlen = referxmax - referxmin + 1
	ylen = referymax - referymin + 1
	# Normalized the dictionary coordinates
	referlocation.iloc[:,0] = (referlocation.iloc[:,0] - referxmin) / xlen
	referlocation.iloc[:,1] = (referlocation.iloc[:,1] - referymin) / ylen
	reloc_np = referlocation.to_numpy()
	reloc_np = reloc_np[:,0:2]
	# Find the closet points in the dictionary for each predicted cords
	pred = np.zeros(coords_predict.shape[0])
	for i in range(coords_predict.shape[0]):
		pred[i] = closest_node(coords_predict[i,:] , reloc_np)
	# map the coordinates
	pred_layer = referlocation.iloc[pred,2]
	adata.obs["pred_layer"] = pred_layer.to_numpy()
	return pred_layer


def plot_layer (adata, folder, name, coloruse):
	if coloruse is None:
		colors_use = ['#46327e', '#365c8d', '#277f8e', '#1fa187', '#4ac16d', '#a0da39', '#fde725', '#ffbb78', '#2ca02c', '#ff7f0e', '#1f77b4', '#800080', '#959595', '#ffff00', '#014d01', '#0000ff', '#ff0000', '#000000']
	else:
		colors_use = coloruse
	# colors_use = ['#111010', '#FFFF00', '#4a6fe3', '#bb7784', '#bec1d4', '#ff9896', '#98df8a', '#ffbb78', '#2ca02c', '#ff7f0e', '#1f77b4', '#800080', '#959595', '#ffff00', '#014d01', '#0000ff', '#ff0000', '#000000']
	num_celltype = 7 # len(adata.obs["pred_layer"].unique())
	adata.uns["pred_layer_str_colors"]=list(colors_use[:num_celltype])
	cdata = adata.copy()
	cdata.obs["x4"] = cdata.obs["x2"]*50
	cdata.obs["x5"] = cdata.obs["x3"]*50
	fig=sc.pl.scatter(cdata, alpha = 1, x = "x5", y = "x4", color = "pred_layer_str", palette = colors_use, show = False, size = 50)
	fig.set_aspect('equal', 'box')
	fig.figure.savefig("{path}/{name}_Layer_pred.pdf".format(path = folder, name = name), dpi = 300)
	cdata.obs["Layer"] = cdata.obs["Layer"].astype(int).astype('str')
	fig2=sc.pl.scatter(cdata, alpha = 1, x = "x5", y = "x4", color = "Layer", palette = colors_use, show = False, size = 50)
	fig2.set_aspect('equal', 'box')
	fig2.figure.savefig("{path}/{name}_Layer_ref.pdf".format(path = folder, name = name), dpi = 300)


def plot_confusion_matrix (referadata, filename, nlayer = 7):
	""" Plot the confusion matrix
		:referadata: the main adata that are working with
		:filename: Numpy [n x 2]: the predicted coordinates based on deep neural network
	"""
	labellist = [i+1 for  i in range(nlayer)]
	conf_mat = confusion_matrix(referadata.obs[["Layer"]], referadata.obs[["pred_layer"]], labels = labellist)
	conf_mat_perc = conf_mat / conf_mat.sum(axis=1, keepdims=True)   # transform the matrix to be row percentage
	conf_mat_CR = classification_report(referadata.obs[["Layer"]], referadata.obs[["pred_layer"]], output_dict=True, labels = labellist)
	np.savetxt('{filename}.csv'.format(filename = filename), conf_mat_perc, delimiter=',')
	with open('{filename}_Classification_Metric.json'.format(filename = filename), 'w') as fp:
		json.dump(conf_mat_CR, fp)
	# plt.figure()
	# conf_mat_fig = seaheatmap(conf_mat_perc, annot=True, cmap='Blues')
	# confplot = conf_mat_fig.get_figure()    
	# confplot.savefig("{filename}.png".format(filename = filename), dpi=400)

def make_annData_spatial (adata, spatial, min_cells = 3, filtered = False):
    """ 
    adata: an annData file for the transcriptomics data
    spatial: an pandas dataframe recording the location information for each spot
    """
    if filtered == False:
        adata.obs["select"] = spatial[1]
        adata.obs["x_cord"] = spatial[2]
        adata.obs["y_cord"] = spatial[3]
        adata.obs["x_pixel"] = spatial[4]
        adata.obs["y_pixel"] = spatial[5]
        # Select captured samples
        adata = adata[adata.obs["select"] == 1]
    else:
        spatialsub = spatial[spatial.iloc[:,0] == 1]
        adata.obs = adata.obs.join(spatialsub)
        adata.obs.columns = ['select', 'x_cord', 'y_cord', 'x_pixel', 'y_pixel']
    adata.var_names = [i.upper() for i in list(adata.var_names)]
    adata.var["genename"] = adata.var.index.astype("str")
    #
    adata.var_names_make_unique()
    prefilter_genes(adata, min_cells=min_cells) # avoiding all genes are zeros
    prefilter_specialgenes(adata)
    #Normalize and take log for UMI
    sc.pp.normalize_per_cell(adata)
    sc.pp.log1p(adata)
    return adata

def make_annData_query (adata):
    """ 
    adata: an annData file for the scRNA data
    """
    adata.var_names = [i.upper() for i in list(adata.var_names)]
    adata.var["genename"] = adata.var.index.astype("str")
    #
    adata.var_names_make_unique()
    prefilter_genes(adata, min_cells=3) # avoiding all genes are zeros
    prefilter_specialgenes(adata)
    #Normalize and take log for UMI
    sc.pp.normalize_per_cell(adata)
    sc.pp.log1p(adata)
    return adata

