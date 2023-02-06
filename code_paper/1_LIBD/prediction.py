## In this version of Cell Location discovEry (LIBD) we consider region of a tissue and we hold off a partial

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
os.chdir("SpaClusterPython")
#import SpaGCN as spg
import CeLEry as cel

from data.LIBD.LIBD_gene_select import d_g

# import tangram as tg

##  1. Data Preperation --------------------------------------------------------------------------
### Load MouseBarin Data Section 1: Regarded as Spatial Transcriptomic Data
dataSection1 = sc.read("../data/LIBD/data_151673.h5ad")
dataSection2 = sc.read("../data/LIBD/data_151676.h5ad")
dataSection3 = sc.read("../data/LIBD/data_151507.h5ad")

# Obtain the number of counts in each layer
layer_count =  dataSection2.obs["Layer"].value_counts().sort_index()
layer_count =  dataSection3.obs["Layer"].value_counts().sort_index()


## Conduct clustering
# cdata = dataSection1.copy()
# cel.getGeneImg(cdata,emptypixel = 0)
#cdataexpand =  np.expand_dims(cdata.GeneImg, axis=1) 

#cdatacentral = cel.centralize(cdataexpand.copy())
#direclust = [cdatacentral[x,0,:,:] for x in range(cdatacentral.shape[0])]
#direflat = [x.flat for x in direclust]
#direflatnp = np.stack(direflat)

## implementing k-means clustering
#kmeansmodel =  KMeans(n_clusters=20, random_state=0)
#kmeans = kmeansmodel.fit(direflatnp)
#np.save("../output/LIBD/cluster.npy", kmeans.labels_)


## Calculating z-score
cel.get_zscore(dataSection1)
cel.get_zscore(dataSection2)
cel.get_zscore(dataSection3)

class_num = 7

##  2*. Test (layer ordinal logistic regression) --------------------------------------------------------------------------

def report_prop_method_LIBD (folder, tissueID, name, dataSection2, traindata, Val_loader, coloruse, outname = ""):
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
    coords_predict = np.zeros(dataSection2.obs.shape[0])
    payer_prob = np.zeros((dataSection2.obs.shape[0],class_num+2))
    for i, img in enumerate(Val_loader):
        recon = DNNmodel(img)
        logitsvalue = np.squeeze(torch.sigmoid(recon[0]).detach().numpy(), axis = 0)
        if (logitsvalue[class_num-2] == 1):
            coords_predict[i] = class_num
            payer_prob[i,(class_num + 1)] = 1
        else:
            logitsvalue_min = np.insert(logitsvalue, 0, 1, axis=0)
            logitsvalue_max = np.insert(logitsvalue_min, class_num, 0, axis=0) 
            prb = np.diff(logitsvalue_max)
            # prbfull = np.insert(-prb[0], 0, 1 -logitsvalue[0,0], axis=0)
            prbfull = -prb.copy() 
            coords_predict[i] = np.where(prbfull == prbfull.max())[0].max() + 1
            payer_prob[i,2:] = prbfull
    #
    dataSection2.obs["pred_layer"] = coords_predict.astype(int)
    payer_prob[:,0] = dataSection2.obs["Layer"]
    payer_prob[:,1] = dataSection2.obs["pred_layer"]
    dataSection2.obs["pred_layer_str"] = coords_predict.astype(int).astype('str')
    cel.plot_layer(adata = dataSection2, folder = "{folder}{tissueID}".format(folder = folder, tissueID = tissueID), name = name, coloruse = coloruse)
    cel.plot_confusion_matrix ( referadata = dataSection2, filename = "{folder}{tissueID}/{name}conf_mat_fig".format(folder = folder, tissueID = tissueID, name = name))
    np.savetxt("{folder}{tissueID}/{name}_probmat.csv".format(folder = folder, tissueID = tissueID, name = name), payer_prob, delimiter=',')


    

def Evaluate (testdata, tissueID, traindata, beta, nrep, coloruse = None):
    ## Wrap up Validation data in to dataloader
    vdatax = np.expand_dims(testdata.X, axis = 0)
    vdata_rs = np.swapaxes(vdatax, 1, 2)
    DataVal = cel.wrap_gene_layer(vdata_rs, testdata.obs, "Layer")
    Val_loader= torch.utils.data.DataLoader(DataVal, batch_size=1, num_workers = 4)
    #
    report_prop_method_LIBD(folder = "../output/LIBD/Prediction", tissueID = tissueID,
                       name = "data_gen_layer_{beta}_n{nrep}".format(beta = beta, nrep = nrep),
                       dataSection2 = testdata, traindata = traindata,
                       Val_loader = Val_loader, coloruse = coloruse)


Evaluate(testdata = dataSection2, tissueID = 151676, traindata = dataSection1, beta = 1e-5, nrep = 2)
Evaluate(testdata = dataSection2, tissueID = 151676, traindata = dataSection1, beta = 1e-5, nrep = 4)
Evaluate(testdata = dataSection2, tissueID = 151676, traindata = dataSection1, beta = 1e-5, nrep = 6)
Evaluate(testdata = dataSection2, tissueID = 151676, traindata = dataSection1, beta = 1e-5, nrep = 8)
Evaluate(testdata = dataSection2, tissueID = 151676, traindata = dataSection1, beta = 1e-5, nrep = 10)


Evaluate(testdata = dataSection3, tissueID = 151507, traindata = dataSection1, beta = 1e-5, nrep = 2)
Evaluate(testdata = dataSection3, tissueID = 151507, traindata = dataSection1, beta = 1e-5, nrep = 4)
Evaluate(testdata = dataSection3, tissueID = 151507, traindata = dataSection1, beta = 1e-5, nrep = 6)
Evaluate(testdata = dataSection3, tissueID = 151507, traindata = dataSection1, beta = 1e-5, nrep = 8)
Evaluate(testdata = dataSection3, tissueID = 151507, traindata = dataSection1, beta = 1e-5, nrep = 10)


def EvaluateOrg (testdata, tissueID, traindata, coloruse = None):
    ## Wrap up Validation data in to dataloader
    vdatax = np.expand_dims(testdata.X, axis = 0)
    vdata_rs = np.swapaxes(vdatax, 1, 2)
    DataVal = cel.wrap_gene_layer(vdata_rs, testdata.obs, "Layer")
    Val_loader= torch.utils.data.DataLoader(DataVal, batch_size=1, num_workers = 4)
    #
    report_prop_method_LIBD(folder = "../output/LIBD/Prediction", tissueID = tissueID,
                       name = "layer_PreOrg",
                       dataSection2 = testdata, traindata = traindata,
                       Val_loader = Val_loader, coloruse = coloruse)

EvaluateOrg (testdata = dataSection2, tissueID = 151676, traindata = dataSection1)
EvaluateOrg (testdata = dataSection3, tissueID = 151507, traindata = dataSection1)



