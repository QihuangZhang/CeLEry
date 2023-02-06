## In this study, we use LIBD data as the training set and evaluate the performance of the results on the Alzheimer data

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
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

# from data.LIBD.LIBD_gene_select import d_g

# import tangram as tg

##  1. Data Preperation --------------------------------------------------------------------------
### Load MouseBarin Data Section 1: Regarded as Spatial Transcriptomic Data
dataSection1 = sc.read("../data/Alzheimer/MergeTrains73747576.h5ad")
dataSection2 = sc.read("../data/Alzheimer/Alzheimer_spa_DE_snRNA_py.h5ad")

## Conduct clustering
#cdata = dataSection1.copy()
#cel.getGeneImg(cdata,emptypixel = 0)
#cdataexpand =  np.expand_dims(cdata.GeneImg, axis=1) 

#cdatacentral = cel.centralize(cdataexpand.copy())
#direclust = [cdatacentral[x,0,:,:] for x in range(cdatacentral.shape[0])]
#direflat = [x.flat for x in direclust]
#direflatnp = np.stack(direflat)

## implementing k-means clustering
#kmeansmodel =  KMeans(n_clusters=20, random_state=0)
#kmeans = kmeansmodel.fit(direflatnp)
#np.save("../output/Alzheimer/cluster.npy", kmeans.labels_)


## Calculating z-score
cel.get_zscore(dataSection1)
cel.get_zscore(dataSection2)

## global parameters
class_num = 7
pca = PCA(n_components=50)


## Compute PCA of cells
principalComponents = pca.fit_transform(dataSection2.X)
PCs = ['PC_{i}'.format(i=i) for i in range(1,51)]
principalDf = pd.DataFrame(data = principalComponents, columns = PCs)
principalDf.to_csv("../output/Alzheimer/PCA_selectedGenes.csv")


tsne = TSNE(n_components=2, verbose=1, perplexity=40, n_iter=300)
tsne_results = tsne.fit_transform(principalDf)
tSNEDf = pd.DataFrame(data = tsne_results)
tSNEDf.to_csv("../output/Alzheimer/tSNEDf.csv")

## PCA of subcategories
# neuron
cellneuron = dataSection2[[(i in ["In", "Ex"]) for i in dataSection2.obs["final_celltype"] ] ]
principalComponents = pca.fit_transform(cellneuron.X)
PCs = ['PC_{i}'.format(i=i) for i in range(1,51)]
principalDf = pd.DataFrame(data = principalComponents, columns = PCs)
principalDf["names"] = cellneuron.obs["cellname"]
principalDf.to_csv("../output/Alzheimer/PCA_neuron.csv")


tsne = TSNE(n_components=2, verbose=1, perplexity=40, n_iter=300)
tsne_results = tsne.fit_transform(principalDf)
tSNEDf = pd.DataFrame(data = tsne_results)
tSNEDf.to_csv("../output/Alzheimer/tSNEDf_neuron.csv")

# oli
celloli = dataSection2[dataSection2.obs["final_celltype"] == "Oli" ]
principalComponents = pca.fit_transform(celloli.X)
PCs = ['PC_{i}'.format(i=i) for i in range(1,51)]
principalDf = pd.DataFrame(data = principalComponents, columns = PCs)
principalDf.to_csv("../output/Alzheimer/PCA_oli.csv")


tsne = TSNE(n_components=2, verbose=1, perplexity=40, n_iter=300)
tsne_results = tsne.fit_transform(principalDf)
tSNEDf = pd.DataFrame(data = tsne_results)
tSNEDf.to_csv("../output/Alzheimer/tSNEDf_oli.csv")


##  2**. Test (layer 2-Stage ordinal logistic regression ) --------------------------------------------------------------------------


def report_prop_method_Alzheimer (folder, name, dataSection2, traindata, Val_loader, outname = ""):
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
    payer_prob = np.zeros((dataSection2.obs.shape[0],class_num+1))
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
            payer_prob[i,1:] = prbfull
    #
    dataSection2.obs["pred_layer"] = coords_predict.astype(int)
    dataSection2.obs["pred_layer_str"] = coords_predict.astype(int).astype('str')
    payer_prob[:,0] = dataSection2.obs["pred_layer"]
    np.savetxt("{folder}/{name}_probmat.csv".format(folder = folder, name = name), payer_prob, delimiter=',')
    sc.tl.rank_genes_groups(dataSection2, 'pred_layer_str', method = 'wilcoxon', key_added = "wilcoxon")
    sc.pl.rank_genes_groups(dataSection2, n_genes = 50, sharey = False, key="wilcoxon", save = 'Alzheimer_DE.pdf')


def Evaluate (testdata, traindata, beta, nrep):
    ## Wrap up Validation data in to dataloader
    vdatax = np.expand_dims(testdata.X, axis = 0)
    vdata_rs = np.swapaxes(vdatax, 1, 2)
    DataVal = cel.wrap_gene_layer(vdata_rs, testdata.obs)
    Val_loader= torch.utils.data.DataLoader(DataVal, batch_size=1, num_workers = 4)
    #
    report_prop_method_Alzheimer(folder = "../output/Alzheimer/Prediction",
                       name = "data_gene_All_layerv2_{beta}_n{nrep}".format(beta = beta, nrep = nrep),
                       dataSection2 = testdata, traindata = traindata,
                       Val_loader = Val_loader)


## Assing a dummy layers for the cells since it is not known
dataSection2.obs["layer"] = 0
dataSection2.obs["layer"][0:7] = [0,1,2,3,4,5,6]
Evaluate(testdata = dataSection2, traindata = dataSection1, beta = 1e-5, nrep = 2)






Evaluate(testdata = dataSection2, traindata = dataSection1, beta = 1e-5, nrep = 10)

def EvaluateOrg (testdata, traindata, coloruse = None):
    ## Wrap up Validation data in to dataloader
    vdatax = np.expand_dims(testdata.X, axis = 0)
    vdata_rs = np.swapaxes(vdatax, 1, 2)
    DataVal = cel.wrap_gene_layer(vdata_rs, testdata.obs)
    Val_loader= torch.utils.data.DataLoader(DataVal, batch_size=1, num_workers = 4)
    #
    report_prop_method_Alzheimer(folder = "../output/Alzheimer/Prediction",
                       name = "layer_PreOrgv2",
                       dataSection2 = testdata, traindata = traindata,
                       Val_loader = Val_loader, coloruse = coloruse)

EvaluateOrg (testdata = dataSection2, traindata = dataSection1)

