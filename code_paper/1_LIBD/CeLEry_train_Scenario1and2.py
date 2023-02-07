#!-### Note: Need to run "preprocess.py" first to obtain the available datasets.


## In this version of Cell Location discovEry (LIBD) we consider region of a tissue   under Scenarios 1 and 2

# Application to LIBD data

import os,csv,re
import pandas as pd
import numpy as np
import scanpy as sc
import math

from skimage import io, color
from sklearn.cluster import KMeans

from scipy.sparse import issparse
import random, torch
import warnings
warnings.filterwarnings("ignore")
import matplotlib.colors as clr
import matplotlib.pyplot as plt
import pickle

#Read original data and save it to h5ad
from scanpy import read_10x_h5
#import SpaGCN as spg
import CeLEry as cel

from data.LIBD.LIBD_gene_select import d_g

# import tangram as tg

##  1. Data Preperation --------------------------------------------------------------------------
### Load MouseBarin Data Section 1: Regarded as Spatial Transcriptomic Data
dataSection1 = sc.read("../data/LIBD/data_151673.h5ad")


## Conduct clustering
cdata = dataSection1.copy()
cel.getGeneImg(cdata,emptypixel = 0)
cdataexpand =  np.expand_dims(cdata.GeneImg, axis=1) 

cdatacentral = cel.centralize(cdataexpand.copy())
direclust = [cdatacentral[x,0,:,:] for x in range(cdatacentral.shape[0])]
direflat = [x.flat for x in direclust]
direflatnp = np.stack(direflat)

# implementing k-means clustering
kmeansmodel =  KMeans(n_clusters=100, random_state=0)
kmeans = kmeansmodel.fit(direflatnp)
np.save("../output/LIBD/cluster_673.npy", kmeans.labels_)


## Calculating z-score
cel.get_zscore(dataSection1)

# get sorted indeces

dataSection1sort = dataSection1.obs.sort_values (by = ['x2','x3'])
dataSection1 = dataSection1[dataSection1sort.index]

def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)

##  2. Data Augmentation --------------------------------------------------------------------------
cdata = dataSection1.copy()
cel.getGeneImg(cdata,emptypixel = 0)
cdataexpand =  np.expand_dims(cdata.GeneImg, axis=1) 
np.save("../output/LIBD/full_geneimg.npy", cdataexpand)

# Read in gene expression and spatial location
cdataexp_full = np.load("../output/LIBD/full_geneimg.npy")


# Load Clustering Results
Kmeans_cluster = np.load("../output/LIBD/cluster_673.npy")

full = cel.datagenemapclust(cdataexp_full,Kmeans_cluster)


## Step 1: Model Fitting of CAVE------------------------------------------------------------------------------------

def FitGenModel (cdataexpand, beta, learning_rate = 1e-3):
    g = torch.Generator()
    g.manual_seed(2020)
    trainloader = torch.utils.data.DataLoader(full, batch_size=1, num_workers = 4, shuffle = True, worker_init_fn=seed_worker, generator=g)
    random.seed(2020)
    torch.manual_seed(2020)
    np.random.seed(2020)
    #
    ## Set up Autoencoder
    CVAEmodel = cel.ClusterVAEmask(latent_dim = 511-Kmeans_cluster.max(), total_cluster = Kmeans_cluster.max(), fgx = cdataexpand.shape[2], fgy = cdataexpand.shape[3], KLDw = 0, hidden = [8,4,2,4,4])
    CVAEmodel = CVAEmodel.float()
    filename = "../output/LIBD/Generation/CVAE_{beta}.obj".format(beta = beta)
    #
    ## Run Autoencoder 
    clg=cel.SpaCluster()
    clg.train(model = CVAEmodel, train_loader = trainloader, num_epochs= 249, annealing = True, KLDwinc = beta/4, n_incr =50, RCcountMax = 30, learning_rate = 0.001)
    # Save the model to a local folder
    filehandler = open(filename, 'wb') 
    pickle.dump(CVAEmodel, filehandler)
    print('save model to: {filename}'.format(filename=filename))
    CVAEmodel.filename = filename
    return CVAEmodel, clg

CVAEmodel_e5, clg_e5 = FitGenModel(cdataexpand = cdataexp_full, beta = 1e-5)
# CVAEmodel_e2, clg_e2 = FitGenModel(cdataexpand = cdataexp_full, beta = 1e-2)


# ## if still converging
def FitGenModel_continue (model, clg, cdataexpand, beta):
    g = torch.Generator()
    g.manual_seed(2020)
    trainloader= torch.utils.data.DataLoader(full, batch_size=1, num_workers = 4, shuffle = True, worker_init_fn=seed_worker, generator=g)
    filename = "../output/LIBD/Generation/CVAE_{beta}.obj".format(beta = beta)
    clg.train(model = model, train_loader = trainloader, num_epochs= 150, annealing = False, RCcountMax = 30, learning_rate = clg.learning_rate)
    # Save the model to a local folder
    filehandler = open(filename, 'wb') 
    pickle.dump(model, filehandler)
    print('save model to: {filename}'.format(filename=filename))
    model.filename = filename
    return model, clg

CVAEmodel_e5, clg_e5 = FitGenModel_continue(model = CVAEmodel_e5, clg = clg_e5, cdataexpand = cdataexp_full, beta = 1e-5)


## Step 2: Data Generation  ------------------------------------------------------------------------------------

## Glimpse of generate model
def GeneratePlot(beta, traindata):
    trainloader= torch.utils.data.DataLoader(traindata, batch_size=1, num_workers = 4)
    filename = "../output/LIBD/Generation/CVAE_{beta}.obj".format(beta = beta)
    # 
    filehandler = open(filename, 'rb') 
    CVAEmodel = pickle.load(filehandler)
    #
    clg=cel.SpaCluster()
    clg.model = CVAEmodel
    try:
        os.makedirs("../output/LIBD/Generation/Glimps/Gen{beta}".format(beta = beta))
    except FileExistsError:
        print("Folder already exists")
    for j, img in enumerate(trainloader):
        # img = next(dataloader_iterator)
        cel.plotGeneImg(img[0][0,0,:,:], filename = "../output/LIBD/Generation/Glimps/Gen{beta}/img{j}".format(beta = beta, j = j))
        omin = img[0].min()
        omax = img[0].max()
        for i in range(10):
            result = CVAEmodel(img) 
            outputimg = result[0][0,0,:,:].detach().numpy() * result[4][0,0,:,:].detach().numpy()
            cel.plotGeneImg( outputimg , filename = "../output/LIBD/Generation/Glimps/Gen{beta}/img{j}var{i}".format(beta = beta, j = j, i = i), range = (omin.item(), omax.item()))

GeneratePlot(beta = 1e-5, traindata = full)
GeneratePlot(beta = 1e-2, traindata = full)


def Data_Generation(beta, dataSection1, traindata, nrep):
    trainloader= torch.utils.data.DataLoader(traindata, batch_size=1, num_workers = 4)
    random.seed(2021)
    torch.manual_seed(2021)
    np.random.seed(2021)
    #
    filename = "../output/LIBD/Generation/CVAE_{beta}.obj".format(beta = beta) 
    filehandler = open(filename, 'rb') 
    CVAEmodel = pickle.load(filehandler)
    #
    clg=cel.SpaCluster()
    clg.model = CVAEmodel
    data_gen=clg.fast_generation(trainloader, nrep)
    # data_gen=np.load("../output/{folder}/data_gen.npy".format(folder = folder))
    data_gen_rs = clg.deep_reshape (data = data_gen, refer = dataSection1.obs)
    try:
        os.makedirs("../output/LIBD/DataGen")
    except FileExistsError:
        print("Folder already exists")
    np.save("../output/LIBD/DataGen/data_gen_{beta}_n{nrep}.npy".format(beta = beta, nrep = nrep), data_gen_rs)


Data_Generation(beta = 1e-5, nrep = 2, dataSection1 = dataSection1, traindata = full)
Data_Generation(beta = 1e-5, nrep = 4, dataSection1 = dataSection1, traindata = full)
Data_Generation(beta = 1e-5, nrep = 6, dataSection1 = dataSection1, traindata = full)
Data_Generation(beta = 1e-5, nrep = 8, dataSection1 = dataSection1, traindata = full)
Data_Generation(beta = 1e-5, nrep = 10, dataSection1 = dataSection1, traindata = full)



## Step 3** (weighted regression model): Prediction Model  ------------------------------------------------------------------------------------
## Count the number of spots on each layer
layer_count =  dataSection1.obs["Layer"].value_counts().sort_index()
layer_weight = layer_count[7]/layer_count[0:7]
layer_weights = torch.tensor(layer_weight.to_numpy())



def FitPredModel (beta, nrep, dataSection1):
    #
    random.seed(2020)
    torch.manual_seed(2020)
    np.random.seed(2020)
    g = torch.Generator()
    g.manual_seed(2021)
    # Original Version
    data_gen_rs = np.load("../output/LIBD/DataGen/data_gen_{beta}_n{nrep}.npy".format(beta = beta, nrep = nrep))
    # Attach the original
    tdatax = np.expand_dims(dataSection1.X, axis = 0)
    tdata_rs = np.swapaxes(tdatax, 1, 2)
    datacomp = np.concatenate((data_gen_rs, tdata_rs), axis=0)
    #
    dataDNN = cel.wrap_gene_layer(datacomp, dataSection1.obs, "Layer")
    CoReg_loader = torch.utils.data.DataLoader(dataDNN, batch_size=4, num_workers = 4, shuffle = True, worker_init_fn=seed_worker, generator=g)
    # Create Deep Neural Network for Coordinate Regression
    DNNmodel = cel.DNNordinal( in_channels = data_gen_rs.shape[1], num_classes = 7, hidden_dims = [50, 10, 5], importance_weights = layer_weights )
    DNNmodel = DNNmodel.float()
    #
    CoReg=cel.SpaCluster()
    CoReg.train(model = DNNmodel, train_loader = CoReg_loader, num_epochs= 250, RCcountMax = 5, learning_rate = 0.001)
    #
    filename2 = "../output/LIBD/Prediction/data_gen_layer_{beta}_n{nrep}.obj".format(beta = beta, nrep = nrep)
    filehandler2 = open(filename2, 'wb') 
    pickle.dump(DNNmodel, filehandler2)

# temp
# beta = 1e-5
# nrep = 10
# dataSection1 = dataSection1

FitPredModel(beta = 1e-5, nrep = 2, dataSection1 = dataSection1)
FitPredModel(beta = 1e-5, nrep = 4, dataSection1 = dataSection1)
FitPredModel(beta = 1e-5, nrep = 6, dataSection1 = dataSection1)
FitPredModel(beta = 1e-5, nrep = 8, dataSection1 = dataSection1)
FitPredModel(beta = 1e-5, nrep = 10, dataSection1 = dataSection1)

def FitPredModel_continue (holdoff, beta, nrep, dataSection1, learning_rate):
    filename2 = "../output/LIBD/Prediction/data_gen_layer_{holdoff}_{beta}_n{nrep}.obj".format(holdoff = holdoff, beta = beta, nrep = nrep)
    filehandler2 = open(filename2, 'rb')
    DNNmodel = pickle.load(filehandler2)
    #
    data_gen_rs = np.load("../output/LIBD/DataGen/data_gen_layer_{holdoff}_{beta}_n{nrep}.npy".format(holdoff = holdoff, beta = beta, nrep = nrep))
    #
    dataDNN = cel.wrap_gene_layer(data_gen_rs, dataSection1.obs)
    CoReg_loader = torch.utils.data.DataLoader(dataDNN, batch_size=1, num_workers = 4, shuffle = True)
    # Create Deep Neural Network for Coordinate Regression
    #
    random.seed(2021)
    torch.manual_seed(2021)
    np.random.seed(2021)
    #
    CoReg=cel.SpaCluster()
    CoReg.train(model = DNNmodel, train_loader = CoReg_loader, num_epochs= 60, RCcountMax = 1, learning_rate = learning_rate)
    #
    filehandler2 = open(filename2, 'wb') 
    pickle.dump(DNNmodel, filehandler2)


FitPredModel_continue(holdoff = 50 , beta = 1e-5, nrep = 10, dataSection1 = Section1train50, learning_rate = 3.125e-06)

FitPredModel_continue(holdoff = 50 , beta = 1e-9, nrep = 10, dataSection1 = Section1train50, learning_rate = 1.220703125e-08)


## Step 3**.2: Prediction Model of the case without data augmentation  ------------------------------------------------------------------------------------

def FitPredModelNE (dataSection1):
    tdatax = np.expand_dims(dataSection1.X, axis = 0)
    tdata_rs = np.swapaxes(tdatax, 1, 2)
    DataTra = cel.wrap_gene_layer(tdata_rs, dataSection1.obs, "Layer")
    t_loader= torch.utils.data.DataLoader(DataTra, batch_size=1, num_workers = 4, shuffle = True)
    # Create Deep Neural Network for Coordinate Regression  # 10, 4, 2
    DNNmodel = cel.DNNordinal( in_channels = DataTra[1][0].shape[0], num_classes = 7, hidden_dims = [10, 4, 2], importance_weights = layer_weights ) # [100,50,25] )
    DNNmodel = DNNmodel.float()
    #
    CoOrg=cel.SpaCluster()
    CoOrg.train(model = DNNmodel, train_loader = t_loader, num_epochs= 150, RCcountMax = 15, learning_rate = 0.001)
    #
    filename3 = "../output/LIBD/Prediction/layer_PreOrg.obj"
    filehandler2 = open(filename3, 'wb') 
    pickle.dump(DNNmodel, filehandler2)


FitPredModelNE (dataSection1 = dataSection1)

