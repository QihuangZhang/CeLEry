## In this version of Cell Location discovEry (LIBD) we consider region of a tissue 

# Application to LIBD data

import os,csv,re
import pandas as pd
import numpy as np
import scanpy as sc
import math

from skimage import io, color
from sklearn.cluster import KMeans
from sklearn.preprocessing import normalize

from scipy.sparse import issparse
import random, torch
import warnings
warnings.filterwarnings("ignore")
import matplotlib.colors as clr
import matplotlib.pyplot as plt
import pickle

#Read original data and save it to h5ad
from scanpy import read_10x_h5
os.chdir("SpaClusterPython")
#import SpaGCN as spg
import CeLEry as cel


def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


##  1. Data Preperation --------------------------------------------------------------------------
### Load MouseBarin Data Section 1: Regarded as Spatial Transcriptomic Data
dataSectionT1 = sc.read("../data/Alzheimer/data_151673.h5ad")
dataSectionT2 = sc.read("../data/Alzheimer/data_151674.h5ad")
dataSectionT3 = sc.read("../data/Alzheimer/data_151675.h5ad")
dataSectionT4 = sc.read("../data/Alzheimer/data_151676.h5ad")
dataSectionM = sc.read("../data/Alzheimer/MergeTrains73747576.h5ad")

# # ## Conduct clustering
# cdata = dataSectionM.copy()
# # implementing k-means clustering
# kmeansmodel =  KMeans(n_clusters=150, random_state=0)
# kmeans = kmeansmodel.fit(normalize(np.transpose(cdata.X), norm = 'max', axis = 0))
# np.save("../output/Alzheimer/cluster_Merge.npy", kmeans.labels_)


## Calculating z-score
cel.get_zscore(dataSectionT1)
cel.get_zscore(dataSectionT2)
cel.get_zscore(dataSectionT3)
cel.get_zscore(dataSectionT4)
cel.get_zscore(dataSectionM)


# get sorted indeces

dataSectionT1sort = dataSectionT1.obs.sort_values (by = ['x2','x3'])
dataSectionT1 = dataSectionT1[dataSectionT1sort.index]
dataSectionT2sort = dataSectionT2.obs.sort_values (by = ['x2','x3'])
dataSectionT2 = dataSectionT2[dataSectionT2sort.index]
dataSectionT3sort = dataSectionT3.obs.sort_values (by = ['x2','x3'])
dataSectionT3 = dataSectionT3[dataSectionT3sort.index]
dataSectionT4sort = dataSectionT4.obs.sort_values (by = ['x2','x3'])
dataSectionT4 = dataSectionT4[dataSectionT4sort.index]
dataSectionMsort = dataSectionM.obs.sort_values (by = ['x2','x3'])
dataSectionM = dataSectionM[dataSectionMsort.index]

##  2. Data Augmentation --------------------------------------------------------------------------
cdata = dataSectionT1.copy()
cel.getGeneImg(cdata,emptypixel = 0)
cdataexpand1 =  np.expand_dims(cdata.GeneImg, axis=1) 

cdata = dataSectionT2.copy()
cel.getGeneImg(cdata,emptypixel = 0)
cdataexpand2 =  np.expand_dims(cdata.GeneImg, axis=1) 

cdata = dataSectionT3.copy()
cel.getGeneImg(cdata,emptypixel = 0)
cdataexpand3 =  np.expand_dims(cdata.GeneImg, axis=1) 

cdata = dataSectionT4.copy()
cel.getGeneImg(cdata,emptypixel = 0)
cdataexpand4 =  np.expand_dims(cdata.GeneImg, axis=1) 

# np.save("../output/LIBD/full_geneimg.npy", cdataexpand)

# Read in gene expression and spatial location
# cdataexp_full= np.load("../output/LIBD/full_geneimg.npy")


# Load Clustering Results
Kmeans_cluster = np.load("../output/Alzheimer/cluster_Merge.npy")

full_T1 = cel.datagenemapclust(cdataexpand1, Kmeans_cluster)
full_T2 = cel.datagenemapclust(cdataexpand2, Kmeans_cluster)
full_T3 = cel.datagenemapclust(cdataexpand3, Kmeans_cluster)
full_T4 = cel.datagenemapclust(cdataexpand4, Kmeans_cluster)

g = torch.Generator()
g.manual_seed(2021)

train_full_T1= torch.utils.data.DataLoader(full_T1, batch_size=1, num_workers = 4, shuffle = True, worker_init_fn=seed_worker, generator=g)
train_full_T2= torch.utils.data.DataLoader(full_T2, batch_size=1, num_workers = 4, shuffle = True, worker_init_fn=seed_worker, generator=g)
train_full_T3= torch.utils.data.DataLoader(full_T3, batch_size=1, num_workers = 4, shuffle = True, worker_init_fn=seed_worker, generator=g)
train_full_T4= torch.utils.data.DataLoader(full_T4, batch_size=1, num_workers = 4, shuffle = True, worker_init_fn=seed_worker, generator=g)

## Step 1: Model Fitting of CAVE------------------------------------------------------------------------------------

def FitGenModel (trainloader, cdataexpand, beta, batch, learning_rate = 1e-3):
    #
    ## Run Autoencoder 
    random.seed(2021)
    torch.manual_seed(2021)
    np.random.seed(2021)
    ## Set up Autoencoder
    CVAEmodel = cel.ClusterVAEmask(latent_dim = 511-Kmeans_cluster.max(), total_cluster = Kmeans_cluster.max(), fgx = cdataexpand.shape[2], fgy = cdataexpand.shape[3], KLDw = 0, hidden = [8,4,2,4,4])
    CVAEmodel = CVAEmodel.float()
    filename = "../output/Alzheimer/Generation/CVAE_T{batch}_{beta}.obj".format(beta = beta, batch = batch)
    #
    clg=cel.SpaCluster()
    clg.train(model = CVAEmodel, train_loader = trainloader, num_epochs= 249, annealing = True, KLDwinc = beta/4, n_incr =50, RCcountMax = 40, learning_rate = 0.001)
    # Save the model to a local folder
    try:
        os.makedirs("../output/Alzheimer/Generation")
    except FileExistsError:
        print("Folder already exists")
    filehandler = open(filename, 'wb') 
    pickle.dump(CVAEmodel, filehandler)
    print('save model to: {filename}'.format(filename=filename))
    CVAEmodel.filename = filename
    return CVAEmodel, clg

## if still converging
def FitGenModel_continue (model, clg, trainloader, cdataexpand, beta, batch):
    filename = "../output/Alzheimer/Generation/CVAE_T{batch}_{beta}.obj".format(beta = beta, batch = batch)
    clg.train(model = model, train_loader = trainloader, num_epochs= 125, annealing = False, RCcountMax = 15, learning_rate = clg.learning_rate)
    # Save the model to a local folder
    filehandler = open(filename, 'wb') 
    pickle.dump(model, filehandler)
    print('save model to: {filename}'.format(filename=filename))
    model.filename = filename
    return model, clg

g = torch.Generator()
g.manual_seed(2021)
CVAEmodel_1, clg_1 = FitGenModel(trainloader = train_full_T1, cdataexpand = cdataexpand1, beta = 1e-5, batch = 1)
CVAEmodel_1, clg_1 = FitGenModel_continue(model = CVAEmodel_1, clg = clg_1, trainloader = train_full_T1, cdataexpand = cdataexpand1, beta = 1e-5, batch = 1)

g = torch.Generator()
g.manual_seed(2021)
CVAEmodel_2, clg_2 = FitGenModel(trainloader = train_full_T2, cdataexpand = cdataexpand2, beta = 1e-5, batch = 2)
CVAEmodel_2, clg_2 = FitGenModel_continue(model = CVAEmodel_2, clg = clg_2, trainloader = train_full_T2, cdataexpand = cdataexpand2, beta = 1e-5, batch = 2)

g = torch.Generator()
g.manual_seed(2021)
CVAEmodel_3, clg_3 = FitGenModel(trainloader = train_full_T3, cdataexpand = cdataexpand3, beta = 1e-5, batch = 3)
CVAEmodel_3, clg_3 = FitGenModel_continue(model = CVAEmodel_3, clg = clg_3, trainloader = train_full_T3, cdataexpand = cdataexpand3, beta = 1e-5, batch = 3)

g = torch.Generator()
g.manual_seed(2021)
CVAEmodel_4, clg_4 = FitGenModel(trainloader = train_full_T4, cdataexpand = cdataexpand4, beta = 1e-5, batch = 4)
CVAEmodel_4, clg_4 = FitGenModel_continue(model = CVAEmodel_4, clg = clg_4, trainloader = train_full_T4, cdataexpand = cdataexpand4, beta = 1e-5, batch = 4)






## Step 2: Data Generation  ------------------------------------------------------------------------------------

## Glimpse of generate model
# def GeneratePlot(beta, traindata):
#     trainloader= torch.utils.data.DataLoader(traindata, batch_size=1, num_workers = 4)
#     filename = "../output/Alzheimer/Generation/CVAE_{beta}.obj".format(beta = beta)
#     # 
#     filehandler = open(filename, 'rb') 
#     CVAEmodel = pickle.load(filehandler)
#     #
#     clg=cel.SpaCluster()
#     clg.model = CVAEmodel
#     try:
#         os.makedirs("../output/Alzheimer/Generation/Glimps/Gen{beta}".format(beta = beta))
#     except FileExistsError:
#         print("Folder already exists")
#     for j, img in enumerate(trainloader):
#         # img = next(dataloader_iterator)
#         cel.plotGeneImg(img[0][0,0,:,:], filename = "../output/Alzheimer/Generation/Glimps/Gen{beta}/img{j}".format(beta = beta, j = j))
#         omin = img[0].min()
#         omax = img[0].max()
#         for i in range(10):
#             result = CVAEmodel(img) 
#             outputimg = result[0][0,0,:,:].detach().numpy() * result[4][0,0,:,:].detach().numpy()
#             cel.plotGeneImg( outputimg , filename = "../output/Alzheimer/Generation/Glimps/Gen{beta}/img{j}var{i}".format(beta = beta, j = j, i = i), range = (omin.item(), omax.item()))

# GeneratePlot(beta = 1e-5, traindata = full)
# GeneratePlot(beta = 1e-2, traindata = full)


def Data_Generation(beta, dataSection1, traindata, nrep, batch):
    random.seed(2021)
    torch.manual_seed(2021)
    np.random.seed(2021)
    trainloader= torch.utils.data.DataLoader(traindata, batch_size=1, num_workers = 4)
    #
    filename = "../output/Alzheimer/Generation/CVAE_T{batch}_{beta}.obj".format(batch = batch, beta = beta) 
    filehandler = open(filename, 'rb') 
    CVAEmodel = pickle.load(filehandler)
    #
    clg=cel.SpaCluster()
    clg.model = CVAEmodel
    data_gen=clg.fast_generation(trainloader, nrep)
    # data_gen=np.load("../output/{folder}/data_gen.npy".format(folder = folder))
    data_gen_rs = clg.deep_reshape (data = data_gen, refer = dataSection1.obs)
    try:
        os.makedirs("../output/Alzheimer/DataGen")
    except FileExistsError:
        print("Folder already exists")
    np.save("../output/Alzheimer/DataGen/data_gen_T{batch}_{beta}_n{nrep}.npy".format(batch = batch, beta = beta, nrep = nrep), data_gen_rs)


Data_Generation(beta = 1e-5, nrep = 2, dataSection1 = dataSectionT1, traindata = full_T1, batch = 1)
Data_Generation(beta = 1e-5, nrep = 10, dataSection1 = dataSectionT1, traindata = full_T1, batch = 1)
Data_Generation(beta = 1e-5, nrep = 2, dataSection1 = dataSectionT2, traindata = full_T2, batch = 2)
Data_Generation(beta = 1e-5, nrep = 10, dataSection1 = dataSectionT2, traindata = full_T2, batch = 2)
Data_Generation(beta = 1e-5, nrep = 2, dataSection1 = dataSectionT3, traindata = full_T3, batch = 3)
Data_Generation(beta = 1e-5, nrep = 10, dataSection1 = dataSectionT3, traindata = full_T3, batch = 3)
Data_Generation(beta = 1e-5, nrep = 2, dataSection1 = dataSectionT4, traindata = full_T4, batch = 4)
Data_Generation(beta = 1e-5, nrep = 10, dataSection1 = dataSectionT4, traindata = full_T4, batch = 4)


## Step 3** (Weighted ordinal regression model): Prediction Model  ------------------------------------------------------------------------------------

g2 = torch.Generator()
g2.manual_seed(2021)


## Count the number of spots on each layer
layer_count =  dataSectionM.obs["Layer"].value_counts().sort_index()
layer_weight = layer_count[7]/layer_count[0:7]
layer_weights = torch.tensor(layer_weight.to_numpy())


def FitPredModel_Load (beta, nrep, dataSection1, batch):
    # Original Version
    data_gen_rs = np.load("../output/Alzheimer/DataGen/data_gen_T{batch}_{beta}_n{nrep}.npy".format(batch = batch, beta = beta, nrep = nrep))
    # Attach the original
    tdatax = np.expand_dims(dataSection1.X, axis = 0)
    tdata_rs = np.swapaxes(tdatax, 1, 2)
    datacomp = np.concatenate((data_gen_rs, tdata_rs), axis=0)
    #
    dataDNN = cel.wrap_gene_layer(datacomp, dataSection1.obs, layerkey = "Layer")
    return dataDNN


def FitPredModel (beta, nrep, dataSection1, dataSection2, dataSection3, dataSection4):
    random.seed(2021)
    torch.manual_seed(2021)
    np.random.seed(2021)
    #
    dataDNN1 = FitPredModel_Load (beta, nrep, dataSection1, batch = 1)
    dataDNN2 = FitPredModel_Load (beta, nrep, dataSection2, batch = 2)
    dataDNN3 = FitPredModel_Load (beta, nrep, dataSection3, batch = 3)
    dataDNN4 = FitPredModel_Load (beta, nrep, dataSection4, batch = 4)
    dataDNN = torch.utils.data.ConcatDataset([dataDNN1, dataDNN2, dataDNN3, dataDNN4])
    CoReg_loader = torch.utils.data.DataLoader(dataDNN, batch_size=16, num_workers = 4, shuffle = True, worker_init_fn=seed_worker, generator=g2)
    # Create Deep Neural Network for Coordinate Regression
    DNNmodel = cel.DNNordinal_v1( in_channels = len(dataDNN[0][0]), num_classes = 7, hidden_dims = [16, 4, 2], importance_weights = layer_weights)
    DNNmodel = DNNmodel.float()
    #
    random.seed(2020)
    torch.manual_seed(2020)
    np.random.seed(2020)
    #
    CoReg=cel.SpaCluster()
    CoReg.train(model = DNNmodel, train_loader = CoReg_loader, num_epochs= 160, RCcountMax = 5, learning_rate = 0.01)
    #
    filename2 = "../output/Alzheimer/Prediction/data_gene_All_layerv2_{beta}_n{nrep}.obj".format(beta = beta, nrep = nrep)
    filehandler2 = open(filename2, 'wb') 
    pickle.dump(DNNmodel, filehandler2)

# temp
# beta = 1e-5
# nrep = 10
# dataSection1 = dataSection1

FitPredModel(beta = 1e-5, nrep = 2, dataSection1 = dataSectionT1, dataSection2 = dataSectionT2, dataSection3 = dataSectionT3, dataSection4 = dataSectionT4)
FitPredModel(beta = 1e-5, nrep = 10, dataSection1 = dataSectionT1, dataSection2 = dataSectionT2, dataSection3 = dataSectionT3, dataSection4 = dataSectionT4)



def FitPredModel_continue (beta, nrep, dataSection1, dataSection2, dataSection3, learning_rate):
    filename2 = "../output/Alzheimer/Prediction/data_gene_All_layerv2_{beta}_n{nrep}.obj".format(beta = beta, nrep = nrep)
    filehandler2 = open(filename2, 'rb')
    DNNmodel = pickle.load(filehandler2)
    #
    dataDNN1 = FitPredModel_Load (beta, nrep, dataSection1, batch = 1)
    dataDNN2 = FitPredModel_Load (beta, nrep, dataSection2, batch = 2)
    dataDNN3 = FitPredModel_Load (beta, nrep, dataSection3, batch = 3)
    dataDNN = torch.utils.data.ConcatDataset([dataDNN1, dataDNN2, dataDNN3])
    CoReg_loader = torch.utils.data.DataLoader(dataDNN, batch_size=4, num_workers = 4, shuffle = True)
    # Create Deep Neural Network for Coordinate Regression
    #
    random.seed(2021)
    torch.manual_seed(2021)
    np.random.seed(2021)
    #
    CoReg=cel.SpaCluster()
    CoReg.train(model = DNNmodel, train_loader = CoReg_loader, num_epochs= 60, RCcountMax = 4, learning_rate = learning_rate)
    #
    filehandler2 = open(filename2, 'wb') 
    pickle.dump(DNNmodel, filehandler2)


FitPredModel_continue(beta = 1e-5, nrep = 2, dataSection1 = dataSectionT1, dataSection2 = dataSectionT2, dataSection3 = dataSectionT3, learning_rate = 0.000625)

FitPredModel_continue(beta = 1e-5, nrep = 10, dataSection1 = dataSectionT1, dataSection2 = dataSectionT2, dataSection3 = dataSectionT3, learning_rate = 1.220703125e-08)


## Step 3**.2: Prediction Model of the case without data augmentation  ------------------------------------------------------------------------------------

def FitPredModelNE (dataSection1):
    tdatax = np.expand_dims(dataSection1.X, axis = 0)
    tdata_rs = np.swapaxes(tdatax, 1, 2)
    DataTra = cel.wrap_gene_layer(tdata_rs, dataSection1.obs)
    t_loader= torch.utils.data.DataLoader(DataTra, batch_size=4, num_workers = 4, shuffle = True)
    # Create Deep Neural Network for Coordinate Regression
    DNNmodel = cel.DNNordinal_v1( in_channels = DataTra[1][0].shape[0], num_classes = 6, hidden_dims = [8, 4, 2], importance_weights = layer_weights) # [100,50,25] )
    DNNmodel = DNNmodel.float()
    #
    CoOrg=cel.SpaCluster()
    CoOrg.train(model = DNNmodel, train_loader = t_loader, num_epochs= 120, RCcountMax = 5, learning_rate = 0.01)
    #
    filename3 = "../output/Alzheimer/Prediction/layer_PreOrgv2.obj"
    filehandler2 = open(filename3, 'wb') 
    pickle.dump(DNNmodel, filehandler2)


FitPredModelNE (dataSection1 = dataSectionM)

