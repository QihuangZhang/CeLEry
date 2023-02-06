import os,csv,re
import pandas as pd
import numpy as np
import scanpy as sc
import math
from skimage import io, color

from scipy.sparse import issparse
import random, torch
import warnings
warnings.filterwarnings("ignore")
import matplotlib.colors as clr
import matplotlib.pyplot as plt
import pickle

#Read original data and save it to h5ad
from scanpy import read_10x_h5
# import SpaGCN as spg

import CeLEry as cel
from data.MouseBrain.MP1_SVG import d_g
import json
# import cv2 as cv


### ------------------------------------------------------------------------------------------------------- ###
###        Preprocessing for MouseSC Data
### ------------------------------------------------------------------------------------------------------- ###

MouseSC = sc.read("../data/Seurat/MouseSC_scRNA_SeuratMouseSC.h5ad")

dataSection1full = sc.read("../data/MouseBrain/MP1_sudo.h5ad")
genename = dataSection1full.var['genename']


# # Get the gene list from the pre-screening
# genelistlist = [d_g[i] for i in  range(len(d_g))]  # transform dictionary to a list of lists
# genelist = sum(genelistlist, [])  # merge the list of lists
# genelistuni = list( dict.fromkeys(genelist) )   # remove duplicates

# genelistindex = [genename[genename == i].index[0] for i in genelistuni if  len(genename[genename == i])>0]

#Read in hitology image
ImageSec1=io.imread("../data/MouseBrain/V1_Mouse_Brain_Sagittal_Posterior_image.tif")
ImageSec1sub = ImageSec1[3000:7000,6200:10500,:]
# cel.printimage (ImageSec1sub, "../output/CeLEry/imageselect")

imgray = cv.cvtColor(ImageSec1sub, cv.COLOR_BGR2GRAY)
imgray2 = imgray.copy()
imgray2[imgray2<160] = 0
imgray2[imgray2>160] = 255

## Take the subset of dataSection1
xcords = dataSection1full.obs["x"].to_numpy()
ycords = dataSection1full.obs["y"].to_numpy()

Section1Sub = dataSection1full[(xcords>=3000) & (xcords<7000) & (ycords>=6200) & (ycords<10500), MouseSC.var_names]
Section1Sub.obs = Section1Sub.obs/50
Section1Sub.obs = Section1Sub.obs.astype(int)
Section1Sub.obs["inner"] = 0

## Quality Control

for i in range(Section1Sub.obs.shape[0]):
    xi = Section1Sub.obs["x"][i]
    yi = Section1Sub.obs["y"][i]
    subarea = np.mean(imgray2[(xi*50-3000):(xi*50+50-3000), (yi*50-6200):(yi*50+50-6200)])
    if subarea<140 or xi*50>6000:
         Section1Sub.obs["inner"].iloc[i] = 1
    if yi*50>10200 or xi*50<1000:
         Section1Sub.obs["inner"].iloc[i] = 0

Section1Sub = Section1Sub[Section1Sub.obs["inner"] == 1, ]

## Calculating z-score
cel.get_zscore(Section1Sub)
cel.get_zscore(MouseSC)

### ------------------------------------------------------------------------------------------------------- ###
###        Perform CeLEry analysis
### ------------------------------------------------------------------------------------------------------- ###

def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)

def FitPredModelNE (dataSection1):
    #
    random.seed(2021)
    torch.manual_seed(2021)
    np.random.seed(2021)
    g = torch.Generator()
    g.manual_seed(2021)
    #
    tdatax = np.expand_dims(dataSection1.X, axis = 0)
    tdata_rs = np.swapaxes(tdatax, 1, 2)
    DataTra = cel.wrap_gene_location(tdata_rs, dataSection1.obs)
    t_loader= torch.utils.data.DataLoader(DataTra, batch_size=4, num_workers = 4, shuffle = True, worker_init_fn=seed_worker, generator=g)
    # Create Deep Neural Network for Coordinate Regression
    DNNmodel = cel.DNN( in_channels = DataTra[1][0].shape[0], hidden_dims = [30, 25, 15] ) # [100,50,25] )
    DNNmodel = DNNmodel.float()
    #
    CoOrg=cel.SpaCluster()
    CoOrg.train(model = DNNmodel, train_loader = t_loader, num_epochs= 500, RCcountMax = 15, learning_rate = 0.0001)
    #
    filename3 = "../output/CeLEry/Mousesc/PreOrg_Mousesc.obj"
    filehandler2 = open(filename3, 'wb') 
    pickle.dump(DNNmodel, filehandler2)

FitPredModelNE (dataSection1 = Section1Sub)



### ------------------------------------------------------------------------------------------------------- ###
###        Present Results
### ------------------------------------------------------------------------------------------------------- ###


def report_prop_method_sc (folder, name, dataSection2, Val_loader, outname = ""):
    """
        Report the results of the proposed methods in comparison to the other method
        :folder: string: specified the folder that keep the proposed DNN method
        :name: string: specified the name of the DNN method, also will be used to name the output files
        :dataSection2: AnnData: the data of Section 2
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
    #
    for i, img in enumerate(Val_loader):
        recon = DNNmodel(img)
        coords_predict[i,:] = recon[0].detach().numpy()
    np.savetxt("{folder}/{name}_predmatrix.csv".format(folder = folder, name = name), coords_predict, delimiter=",")

def EvaluateOrg (testdata):
    ## Wrap up Validation data in to dataloader
    vdatax = np.expand_dims(testdata.X, axis = 0)
    vdata_rs = np.swapaxes(vdatax, 1, 2)
    DataVal = cel.wrap_gene_location(vdata_rs, testdata.obs[["sex_id","region_id"]])
    Val_loader= torch.utils.data.DataLoader(DataVal, batch_size=1, num_workers = 4)
    #
    report_prop_method_sc(folder = "../output/CeLEry/Mousesc/",
                        name = "PreOrg_Mousesc", dataSection2 = testdata,
                        Val_loader = Val_loader)


EvaluateOrg(testdata = MouseSC)

### ------------------------------------------------------------------------------------------------------- ###
###        Perform Tangram analysis
### ------------------------------------------------------------------------------------------------------- ###
# import tangram as tg

# tg.pp_adatas(MouseSC, Section1Sub, genes=None)
# map = tg.map_cells_to_space(MouseSC, Section1Sub, device='cpu')
# map.write_h5ad('../output/CeLEry/Mousesc/tangram.h5ad')

S1_xmax = Section1Sub.obs['x'].max() + 1
S1_xmin = Section1Sub.obs['x'].min() - 1
S1_ymax = Section1Sub.obs['y'].max() + 1
S1_ymin = Section1Sub.obs['y'].min() - 1

map = sc.read("../output/CeLEry/Mousesc/tangram.h5ad")


## Normalize the coordinates of both Sections
spx = (Section1Sub.obs.iloc[:,0] - S1_xmin) / (S1_xmax - S1_xmin)
spy = (Section1Sub.obs.iloc[:,1] - S1_ymin) / (S1_ymax - S1_ymin)

coords_predict_tangram = np.zeros((MouseSC.obs.shape[0],2))
for i in range(map.X.shape[0]):
    bestindex = np.argmax(map.X[i,:])
    pred = torch.FloatTensor([spx[bestindex],spy[bestindex]])
    coords_predict_tangram[i,:] = pred


np.savetxt("{folder}/{name}_predmatrix.csv".format(folder = "../output/CeLEry/Mousesc/", name = "Tangram_Mousesc"), coords_predict_tangram, delimiter=",")

