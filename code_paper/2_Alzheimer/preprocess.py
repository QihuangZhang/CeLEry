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


#Read original data and save it to h5ad
from scanpy import read_10x_h5
import SpaGCN as spg

os.chdir("SpaClusterPython")
from data.LIBD.LIBD_gene_select import d_g
import json

### ------------------------------------------------------------------------------------------------------- ###
###        Preprocessing for Alzheimer Data
### ------------------------------------------------------------------------------------------------------- ###

### Load the csv gene expression data
# df = pd.read_csv('../data/Alzheimer/genotype.csv',  header=None)
# print(df)

# Alzheimer = sc.AnnData(X= np.array(df.transpose()))
# # Alzheimer.var = pd.DataFrame(index= df.columns[1:])
# # Alzheimer.obs["sample_name"] = np.array(df.iloc[:,0])

# phenotype = pd.read_csv('../data/Alzheimer/phenotype.csv')
# Alzheimer.obs["sample"] =  np.array(phenotype.iloc[:,0])
# Alzheimer.obs["cluster"] =  np.array(phenotype.iloc[:,1])


# Alzheimer_genelist  = pd.read_csv('../data/Alzheimer/feature.csv',  header=None)
# Alzheimer.var_names=[i.upper() for i in sum(Alzheimer_genelist.values.tolist(), [])]



Alzheimer = sc.read("../data/Alzheimer/BrainData.h5ad")
Alzheimer.var["genename"]=Alzheimer.var.index.astype("str")

Alzheimer.var_names_make_unique()
spg.prefilter_genes(Alzheimer,min_cells=3) # avoiding all genes are zeros
spg.prefilter_specialgenes(Alzheimer)
#Normalize and take log for UMI-------
sc.pp.normalize_per_cell(Alzheimer)
sc.pp.log1p(Alzheimer)


# index10 = random.sample(range(Alzheimer.shape[0]), int(3000))
# AlzheimerToy = Alzheimer[list(set(index10)), ]
# AlzheimerToy.write_h5ad("../tutorial/data/AlzheimerToy.h5ad")


Alzheimer.write_h5ad("../data/Alzheimer/Alzheimer_snRNA_py.h5ad")


### ------------------------------------------------------------------------------------------------------- ###
###        Process the genelist
### ------------------------------------------------------------------------------------------------------- ###

Alzheimer = sc.read("../data/Alzheimer/Alzheimer_snRNA_py.h5ad")


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
    # sc.pl.rank_genes_groups(adata, n_genes = 200, sharey = False, key="wilcoxon", save = '{studyID}.pdf'.format(studyID = studyID))
    gene_topDE_list = []
    for layer_i in ['L1', 'L2', 'L3', 'L4', 'L5', 'L6',"WM"]:
        gene_rank = sc.get.rank_genes_groups_df (adata,  group = layer_i, key = 'wilcoxon')
        top_gene_list = list(gene_rank["names"].iloc[0:150])
        gene_topDE_list.append( top_gene_list  )
    return gene_topDE_list

genelist73 = get_LIBD_top_DEgenes (151673)
genelist74 = get_LIBD_top_DEgenes (151674)
genelist75 = get_LIBD_top_DEgenes (151675)
genelist76 = get_LIBD_top_DEgenes (151676)


with open('../data/Alzheimer/gene_selected_151673.json') as json_file:
    gene73 = json.load(json_file)

with open('../data/Alzheimer/gene_selected_151674.json') as json_file:
    gene74 = json.load(json_file)

with open('../data/Alzheimer/gene_selected_151675.json') as json_file:
    gene75 = json.load(json_file)

with open('../data/Alzheimer/gene_selected_151676.json') as json_file:
    gene76 = json.load(json_file)



# Get the gene list from the pre-screening
genelistlist73 = [gene73[str(i)] for i in  range(len(gene73))]  # transform dictionary to a list of lists
genelistlist74 = [gene74[str(i)] for i in  range(len(gene74))]  # transform dictionary to a list of lists
genelistlist75 = [gene75[str(i)] for i in  range(len(gene75))]  # transform dictionary to a list of lists
genelistlist76 = [gene75[str(i)] for i in  range(len(gene76))]  # transform dictionary to a list of lists
genelistlist = genelistlist73 + genelistlist74 + genelistlist75 + genelistlist76 + genelist73 + genelist74 + genelist75 + genelist76
genelistlist = genelist73 + genelist74 + genelist75 + genelist76



genelist = sum(genelistlist, [])  # merge the list of lists
genelistuni = list( dict.fromkeys(genelist) )   # remove duplicates


genelist_Alzheimer = [i for i in genelistuni if i in list(Alzheimer.var_names)]



### ------------------------------------------------------------------------------------------------------- ###
###        Screen the Trainning Data
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
    LayerName =["L1","L2","L3","L4","L5","L6","WM"] #,"WM"
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
    genelist_Alzheimer2 = [value for value in genelist_Alzheimer if value in list( adata.var_names )]
    bdata = adata[:,genelist_Alzheimer2]
    cdata = sc.AnnData(X = bdata.X.toarray(), obs = bdata.obs, var = bdata.var, uns =bdata.uns, obsm = bdata.obsm)
    cdata.write_h5ad("../data/Alzheimer/data_{studyID}.h5ad".format(studyID = studyID))
    return genelist_Alzheimer2

genelist_Alzheimer = Preprocess_SpTrans(151673)
genelist_Alzheimer = Preprocess_SpTrans(151674)
genelist_Alzheimer = Preprocess_SpTrans(151675)
genelist_Alzheimer = Preprocess_SpTrans(151676)
Preprocess_SpTrans(151673)
Preprocess_SpTrans(151674)
Preprocess_SpTrans(151675)
Preprocess_SpTrans(151676)




### ------------------------------------------------------------------------------------------------------- ###
###    Create a merged data set for 73, 74 and 75
### ------------------------------------------------------------------------------------------------------- ###

dataSection1 = sc.read("../data/Alzheimer/data_{studyID}.h5ad".format(studyID = 151673)) 
print(dataSection1)
dataSection2 = sc.read("../data/Alzheimer/data_{studyID}.h5ad".format(studyID = 151674)) 
print(dataSection2)
dataSection3 = sc.read("../data/Alzheimer/data_{studyID}.h5ad".format(studyID = 151675)) 
print(dataSection3)
dataSection4 = sc.read("../data/Alzheimer/data_{studyID}.h5ad".format(studyID = 151676)) 
print(dataSection4)


dataSection = dataSection1.concatenate(dataSection2, dataSection3, dataSection4)
print(dataSection)

dataSection.write_h5ad("../data/Alzheimer/MergeTrains73747576.h5ad")


### ------------------------------------------------------------------------------------------------------- ###
###    Export the Alzheimer data with subsampling
### ------------------------------------------------------------------------------------------------------- ###

Alzheimer = sc.read("../data/Alzheimer/Alzheimer_snRNA_py.h5ad")

Alzheimersub = Alzheimer[:,genelist_Alzheimer]
# # Remove the White Matters
# Alzheimersub = Alzheimersub [ Alzheimersub.obs["Layer"]!=7 ]

# del Alzheimersub.obs['class_label']

Alzheimersub.write_h5ad("../data/Alzheimer/Alzheimer_spa_DE_snRNA_py.h5ad")



GWASgene = pd.read_csv('../data/Alzheimer/GWASgene.csv',  header=None)
GWASlist = GWASgene.values.tolist()[0]
GWASlistclean = [x.strip(' ') for x in GWASlist if x.strip(' ') in Alzheimer.var_names]
AlzheimerGWAS = Alzheimer[:,GWASlistclean]

d = pd.DataFrame(AlzheimerGWAS.X.A)
d.columns = GWASlistclean


d.to_csv("../data/Alzheimer/geno_selected2.csv", AlzheimerGWAS.X.A, delimiter=",")

# sc.tl.rank_genes_groups(Alzheimersub, 'Layer_Ch', method = 'wilcoxon', key_added = "wilcoxon")
# sc.pl.rank_genes_groups(Alzheimersub, n_genes = 50, sharey = False, key="wilcoxon", save = 'Alzheimersub.pdf')



### ------------------------------------------------------------------------------------------------------- ###
###    Export the Alzheimer data for tutorial
### ------------------------------------------------------------------------------------------------------- ###

Alzheimer_tutorial = sc.read("../data/Alzheimer/Alzheimer_spa_DE_snRNA_py.h5ad")

random.seed(2021)
torch.manual_seed(2021)
np.random.seed(2021)

index10 = random.sample(range(Alzheimer_tutorial.shape[0]), int(Alzheimer_tutorial.shape[0]*0.02))


AD_select = Alzheimer_tutorial[list(set(index10)),]
AD_select.write_h5ad("/project/mingyaolpc/Qihuang/CeLEryTest/data/tutorial/AlzheimerToy.h5ad")
