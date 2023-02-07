### Datasets of this study can be download from  http://spatial.libd.org/spatialLIBD/


import os,csv,re
import pandas as pd
import numpy as np
import scanpy as sc
import math
from skimage import io, color

from scipy.sparse import issparse
import warnings
warnings.filterwarnings("ignore")
import matplotlib.colors as clr
import matplotlib.pyplot as plt


#Read original data and save it to h5ad
from scanpy import read_10x_h5
import SpaGCN as spg

os.chdir("SpaClusterPython")
from data.LIBD.LIBD_gene_select import d_g
import json





### ------------------------------------------------------------------------------------------------------- ###
###        Process the genelist
### ------------------------------------------------------------------------------------------------------- ###
def get_LIBD_top_DEgenes (studyID):
    """
    Preprocess the spatial transcriptomic raw data and obtain the optimal DE genes
    Parameters
        -----------
        studyID : string. the study ID of the LIBD datasets
    Returns
    -----------
        gene_topDE_list: the list of gene set that contains the highest DE genes between layers
    """
    adata = read_10x_h5("../data/LIBD/{studyID}/{studyID}_raw_feature_bc_matrix.h5".format(studyID = studyID))
    spatial = pd.read_csv("../data/LIBD/{studyID}/tissue_positions_list.txt".format(studyID = studyID),sep=",", header = None, na_filter = False, index_col = 0) 
    adata.obs["x1"] = spatial[1]
    adata.obs["x2"] = spatial[2]
    adata.obs["x3"] = spatial[3]
    # Select captured samples
    adata = adata[adata.obs["x1"] == 1]
    adata.var_names = [i.upper() for i in list(adata.var_names)]
    adata.var["genename"] = adata.var.index.astype("str")
    #
    del adata.obs["x1"]
    #
    adata.obs["Layer"] = 0
    LayerName =["L1","L2","L3","L4","L5","L6","WM"] #
    for i in range(7):
        Layer = pd.read_csv("../data/LIBD/{studyID}/{studyID}_{Lname}_barcodes.txt".format(studyID = studyID, Lname = LayerName[i]), sep=",", header = None, na_filter = False, index_col = 0)
        adata.obs.loc[Layer.index, "Layer"] = int(i+1)
        adata.obs.loc[Layer.index, "Layer_character"] = LayerName[i]
    data = adata[adata.obs["Layer"]!=0]    # Newly added on May 25 #Remove the spots without any layer label
    #
    #  Preprocessing
    adata.var_names_make_unique()
    spg.prefilter_genes(adata,min_cells=3) # avoiding all genes are zeros
    spg.prefilter_specialgenes(adata)
    #Normalize and take log for UMI-------
    sc.pp.normalize_per_cell(adata)
    sc.pp.log1p(adata)
    #
    sc.tl.rank_genes_groups(adata, 'Layer_character', method = 'wilcoxon', key_added = "wilcoxon")
    # sc.pl.rank_genes_groups(adata, n_genes = 200, sharey = False, key="wilcoxon", save = '{studyID}.pdf'.format(studyID = studyID))
    gene_topDE_list = []
    for layer_i in ['L1', 'L2', 'L3', 'L4', 'L5', 'L6', 'WM']:
        gene_rank = sc.get.rank_genes_groups_df (adata,  group = layer_i, key = 'wilcoxon')
        top_gene_list = list(gene_rank["names"].iloc[0:200])
        gene_topDE_list.append( top_gene_list  )
    return gene_topDE_list

genelist73 = get_LIBD_top_DEgenes (151673)
genelist74 = get_LIBD_top_DEgenes (151674)
genelist75 = get_LIBD_top_DEgenes (151675)

with open('../data/Hodge/gene_selected_151673.json') as json_file:
    gene73 = json.load(json_file)

with open('../data/Hodge/gene_selected_151674.json') as json_file:
    gene74 = json.load(json_file)

with open('../data/Hodge/gene_selected_151675.json') as json_file:
    gene75 = json.load(json_file)

# Get the gene list from the pre-screening
genelistlist73 = [gene73[str(i)] for i in  range(len(gene73))]  # transform dictionary to a list of lists
genelistlist74 = [gene74[str(i)] for i in  range(len(gene74))]  # transform dictionary to a list of lists
genelistlist75 = [gene75[str(i)] for i in  range(len(gene75))]  # transform dictionary to a list of lists
genelistlist = genelistlist73 + genelistlist74 + genelistlist75 + genelist73 + genelist74 + genelist75
genelist = sum(genelistlist, [])  # merge the list of lists
genelistuni = list( dict.fromkeys(genelist) )   # remove duplicates




### ------------------------------------------------------------------------------------------------------- ###
###    Preprocessing for spatial transcriptomics data
### ------------------------------------------------------------------------------------------------------- ###

def Preprocess_SpTrans (studyID):
    """
    Preprocess the spatial transcriptomic raw data and package into Annnoted Dataset
    Parameters
        -----------
        studyID : string. the study ID of the LIBD datasets
    Returns
    -----------
        No returns. File automatically generated.
    """
    adata = read_10x_h5("../data/LIBD/{studyID}/{studyID}_raw_feature_bc_matrix.h5".format(studyID = studyID))
    spatial = pd.read_csv("../data/LIBD/{studyID}/tissue_positions_list.txt".format(studyID = studyID),sep=",", header = None, na_filter = False, index_col = 0) 
    adata.obs["x1"] = spatial[1]
    adata.obs["x2"] = spatial[2]
    adata.obs["x3"] = spatial[3]
    # Select captured samples
    adata = adata[adata.obs["x1"] == 1]
    adata.var_names = [i.upper() for i in list(adata.var_names)]
    adata.var["genename"] = adata.var.index.astype("str")
    #
    del adata.obs["x1"]
    #
    adata.obs["Layer"] = 0
    LayerName =["L1","L2","L3","L4","L5","L6","WM"]
    for i in range(7):
        Layer = pd.read_csv("../data/LIBD/{studyID}/{studyID}_{Lname}_barcodes.txt".format(studyID = studyID, Lname = LayerName[i]), sep=",", header = None, na_filter = False, index_col = 0)
        adata.obs.loc[Layer.index, "Layer"] = int(i+1)
        adata.obs.loc[Layer.index, "Layer_character"] = LayerName[i]
    #  Remove the spots without any layer label      
    adata=adata[adata.obs["Layer"]!=0]    # Newly added on May 25
    #
    #  Preprocessing
    adata.var_names_make_unique()
    spg.prefilter_genes(adata,min_cells=3) # avoiding all genes are zeros
    spg.prefilter_specialgenes(adata)
    #Normalize and take log for UMI-------
    sc.pp.normalize_per_cell(adata)
    sc.pp.log1p(adata)
    #
    sc.tl.rank_genes_groups(adata, 'Layer_character', method = 'wilcoxon', key_added = "wilcoxon")
    sc.pl.rank_genes_groups(adata, n_genes = 50, sharey = False, key="wilcoxon", save = '{studyID}.pdf'.format(studyID = studyID))
    #
    #  Filter the Genes that are selected by SpaGCN
    genename = adata.var['genename']
    # Get the gene list from the pre-screening
    # genelistlist = [d_g[i] for i in  range(len(d_g))]  # transform dictionary to a list of lists
    # genelist = sum(genelistlist, [])  # merge the list of lists
    # genelistuni = list( dict.fromkeys(genelist) )   # remove duplicates
    genelistindex = [genename[genename == i].index[0] for i in genelistuni if  len(genename[genename == i])>0]  # only keep the genes that exists in SpT data
    # Filter the genelist and output the results
    bdata = adata[:,genelistindex]
    cdata = sc.AnnData(X = bdata.X.toarray(), obs = bdata.obs, var = bdata.var, uns =bdata.uns, obsm = bdata.obsm)
    cdata.write_h5ad("../data/LIBD/data_{studyID}.h5ad".format(studyID = studyID))
    return genelistindex

## Training Data
genelistuni = Preprocess_SpTrans(151673)
genelistuni = Preprocess_SpTrans(151674)
genelistuni = Preprocess_SpTrans(151675)
genelistuni = Preprocess_SpTrans(151676)
genelistuni = Preprocess_SpTrans(151507)

Preprocess_SpTrans(151673)
Preprocess_SpTrans(151674)
Preprocess_SpTrans(151675)


## Testing Data
Preprocess_SpTrans(151676)
Preprocess_SpTrans(151507)


### ------------------------------------------------------------------------------------------------------- ###
###    Create a merged data set for 73, 74 and 75
### ------------------------------------------------------------------------------------------------------- ###

dataSection1 = sc.read("../data/LIBD/data_{studyID}.h5ad".format(studyID = 151673)) 
print(dataSection1)
dataSection2 = sc.read("../data/LIBD/data_{studyID}.h5ad".format(studyID = 151674)) 
print(dataSection2)
dataSection3 = sc.read("../data/LIBD/data_{studyID}.h5ad".format(studyID = 151675)) 
print(dataSection3)

dataSection = dataSection1.concatenate(dataSection2, dataSection3)
print(dataSection)

dataSection.write_h5ad("../data/LIBD/MergeTrains737475.h5ad")



datakankan = sc.read("../data/LIBD/MergeTrains737475.h5ad") 




dataSection1 = sc.read("../data/LIBD/data_{studyID}.h5ad".format(studyID = 151673)) 
print(dataSection1)


dataSection4 = sc.read("../data/LIBD/data_{studyID}.h5ad".format(studyID = 151676)) 
dataSection4.obs.to_csv ("../data/LIBD/visualization_151676.csv", sep = ",")

dataSection5 = sc.read("../data/LIBD/data_{studyID}.h5ad".format(studyID = 151507)) 
dataSection5.obs.to_csv ("../data/LIBD/visualization_151507.csv", sep = ",")

print(dataSection4)
