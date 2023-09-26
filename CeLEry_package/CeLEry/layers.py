import pandas as pd
import numpy as np
import scipy

import math
import torch
from torch.nn.parameter import Parameter
from torch.nn.modules.module import Module
from torch.utils.data import DataLoader

import matplotlib.pyplot as plt


class ConvolutionNN(Module):
    """
    Simple CNN layer
    """
    def __init__(self, , )

def getGeneImg (adata, geneset = None):
    # Transform the AnnData file into Genes of images
    # adata: the input data of AnnData object
    # geneset: the set of gene considered
    if geneset is None:
        x = adata.obs[["x2"]]
        y = adata.obs[["x3"]]
        xmin = x.min().iloc[0]
        xmax = x.max().iloc[0]
        ymin = y.min().iloc[0]
        ymax = y.max().iloc[0]
        # i = 12
        for i in range(adata.X.shape[1]):
            z = adata.X[:,i] 
            zmin = z.min()
            zmax = z.max()
            # create array for image : zmax+1 is the default value
            shape = (xmax-xmin+1,ymax-ymin+1)
            img = np.ma.array(np.ones(shape)*0)
            for inp in range(x.shape[0]):
                img[x.iloc[inp,0]-xmin,y.iloc[inp,0]-ymin]=z[inp,0]
            # set mask on default value
            img.mask = (img==0)
            # set a gray background for test
            img_bg_test =  np.zeros(shape)
            cmap_bg_test = plt.get_cmap('gray')
            plt.imshow(img_bg_test,cmap=cmap_bg_test,interpolation='none')
            # plot
            cmap = plt.get_cmap('jet')
            plt.imshow(img,cmap=cmap,interpolation='none',vmin=zmin,vmax=zmax)
            plt.colorbar()
            plt.show()







