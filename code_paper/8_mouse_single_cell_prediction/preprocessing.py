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
import SpaCluster as spc

#Read original data and save it to h5ad
from scanpy import read_10x_h5
import SpaGCN as spg

from data.MouseBrain.MP1_SVG import d_g
import json


### ------------------------------------------------------------------------------------------------------- ###
###        Preprocessing for MouseSC Data
### ------------------------------------------------------------------------------------------------------- ###

dataSection1full = sc.read("../data/MouseBrain/MP1_sudo.h5ad")
genename = dataSection1full.var['genename']


# Get the gene list from the pre-screening
genelistlist = [d_g[i] for i in  range(len(d_g))]  # transform dictionary to a list of lists
genelist = sum(genelistlist, [])  # merge the list of lists
genelistuni = list( dict.fromkeys(genelist) )   # remove duplicates

genelistindex = [genename[genename == i].index[0] for i in genelistuni if  len(genename[genename == i])>0]



### Load the csv gene expression data
df = pd.read_csv('../data/MouseBrain/reference/metadata.csv')
select_region = ["RSP", "VIS", "ENT"]

dfsub = df[df["region_label"].isin(select_region)]
selected_sample = df["region_label"].isin(select_region) # list(dfsub["sample_name"])
nsamplefull = selected_sample.shape[0]

chunksize = 50000

geneexpr_reader = pd.read_csv('../data/MouseBrain/reference/matrix.csv', chunksize=chunksize)
df2 = geneexpr_reader.get_chunk(2)
headers = list(df2.keys())
gene_mouse = [gene.upper() for gene in headers]
genelistindex2 = [i for i in genelistindex if  i in gene_mouse]
gene_select_mark = [(i.upper() in genelistindex2) for i in  gene_mouse]
gene_select_mark[0] = True

datamouse = []

for i, chunk in enumerate(tqdm(geneexpr_reader)):
    # chunk_sample = list(selected_sample[i*chunksize:min( (i+1)*chunksize, nsamplefull)])
    data = chunk.iloc[:, gene_select_mark]
    datamouse.append(data) 

mouseRNAgeneexpres = pd.concat(datamouse)
mouseRNAgeneexpressub = mouseRNAgeneexpres[mouseRNAgeneexpres["sample_name"].isin(dfsub["sample_name"])]

mouseRNAgeneexpressort= mouseRNAgeneexpressub.sort_values('sample_name')
dfsubsort = dfsub.sort_values('sample_name')
del mouseRNAgeneexpressort["sample_name"]

MouseSC = sc.AnnData(X= np.array(mouseRNAgeneexpressort), obs = dfsubsort)
MouseSC.var_names=[i.upper() for i in list(MouseSC.var_names)]
MouseSC.var["genename"]=MouseSC.var.index.astype("str")
MouseSC.obs_names = MouseSC.obs["sample_name"]
del MouseSC.obs["sample_name"]

#Normalize and take log for UMI-------
sc.pp.normalize_per_cell(MouseSC)
sc.pp.log1p(MouseSC)



### ------------------------------------------------------------------------------------------------------- ###
###        Demographic Information about MouseSC Data
### ------------------------------------------------------------------------------------------------------- ###

MouseSC = sc.read("../data/MouseBrain/MouseSC_scRNA.h5ad")
LIBD = sc.read("../data/LIBD/MergeTrains737475.h5ad")

gene_comp_index = [x for x in list(LIBD.var.index) if x in list(MouseSC.var.index)]
print('The percentage of LIBD (selected by SpaGCN) that are coesists in MouseSC is {percentage}'.format(percentage = len(gene_comp_index)/len(list(LIBD.var.index))))


adata = read_10x_h5("../data/LIBD/151673/151673_raw_feature_bc_matrix.h5")
visium_comp_index = [x for x in list(adata.var.index) if x in list(MouseSC.var.index)]
print('The percentage of Visium 10X data that are coesists in MouseSC is {percentage}'.format(percentage = len(visium_comp_index)/len(list(adata.var.index))))

for i in range(1,8):
    ndata = MouseSC[MouseSC.obs["Layer"] == int(i)].copy()
    percent_gene = np.mean(ndata.X != 0, axis = 0)
    expressed_gene_percentage = np.sum(percent_gene > 0.05)/len(percent_gene)
    print('The layer {ni} (nspot = {nspot}) has the percentage: {percent}'.format(ni = i, nspot = ndata.X.shape[0], percent = expressed_gene_percentage))

### ------------------------------------------------------------------------------------------------------- ###
###    Check the percentage of DE genes between L3/L4 or L4/L5
### ------------------------------------------------------------------------------------------------------- ###

MouseSC = sc.read("../data/MouseSC/MouseSC_snRNA.h5ad")

sc.tl.rank_genes_groups(MouseSC, 'Layer_Ch', method = 'wilcoxon', key_added = "wilcoxon")
sc.pl.rank_genes_groups(MouseSC, n_genes = 50, sharey = False, key="wilcoxon", save = True)

sc.tl.rank_genes_groups(MouseSC, 'Layer_Ch', groups=["L4"], reference = "L5",  method='wilcoxon')
sc.pl.rank_genes_groups(MouseSC, groups=["L4"], n_genes=50, save = 'L4vsL5.pdf')

sc.tl.rank_genes_groups(MouseSC, 'Layer_Ch', groups=["L4"], reference = "L3",  method='wilcoxon')
sc.pl.rank_genes_groups(MouseSC, groups=["L4"], n_genes=50, save = 'L4vsL3.pdf')


#### Add by 2021-06-26 try adding DE genes in the analysis
MouseSC = sc.read("../data/MouseSC/MouseSC_snRNA.h5ad")

## Summarize the list of top DE genes
sc.tl.rank_genes_groups(MouseSC, 'Layer_Ch', method = 'wilcoxon', key_added = "wilcoxon")
gene_topDE_list = []
for layer_i in ['L1', 'L2', 'L3', 'L4', 'L5', 'L6']:
    gene_rank = sc.get.rank_genes_groups_df (MouseSC,  group = layer_i, key = 'wilcoxon')
    top_gene_list = list(gene_rank["names"].iloc[0:50])
    gene_topDE_list.append( top_gene_list  )





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
    LayerName =["L1","L2","L3","L4","L5","L6"] #,"WM"
    for i in range(6):
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
    for layer_i in ['L1', 'L2', 'L3', 'L4', 'L5', 'L6']:
        gene_rank = sc.get.rank_genes_groups_df (adata,  group = layer_i, key = 'wilcoxon')
        top_gene_list = list(gene_rank["names"].iloc[0:200])
        gene_topDE_list.append( top_gene_list  )
    return gene_topDE_list

genelist73 = get_LIBD_top_DEgenes (151673)
genelist74 = get_LIBD_top_DEgenes (151674)
genelist75 = get_LIBD_top_DEgenes (151675)
genelist76 = get_LIBD_top_DEgenes (151676)


with open('../data/MouseSC/gene_selected_151673.json') as json_file:
    gene73 = json.load(json_file)

with open('../data/MouseSC/gene_selected_151674.json') as json_file:
    gene74 = json.load(json_file)

with open('../data/MouseSC/gene_selected_151675.json') as json_file:
    gene75 = json.load(json_file)

with open('../data/MouseSC/gene_selected_151676.json') as json_file:
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


genelist_MouseSC = [i for i in genelistuni if i in list(MouseSC.var_names)]



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
    LayerName =["L1","L2","L3","L4","L5","L6"] #,"WM"
    for i in range(6):
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
    #  genename = adata.var['genename']
    # Get the gene list from the pre-screening
    # genelistlist = [d_g[i] for i in  range(len(d_g))]  # transform dictionary to a list of lists
    # genelist = sum(genelistlist, [])  # merge the list of lists
    # genelistuni = list( dict.fromkeys(genelist) )   # remove duplicates
    #  genelistindex = [genename[genename == i].index[0] for i in genelistuni if  len(genename[genename == i])>0]  # only keep the genes that exists in SpT data
    # Filter the genelist and output the results
    genelist_MouseSC2 = [value for value in genelist_MouseSC if value in list( adata.var_names )]
    bdata = adata[:,genelist_MouseSC2]
    cdata = sc.AnnData(X = bdata.X.toarray(), obs = bdata.obs, var = bdata.var, uns =bdata.uns, obsm = bdata.obsm)
    cdata.write_h5ad("../data/MouseSC/data_{studyID}.h5ad".format(studyID = studyID))
    return genelist_MouseSC2

genelist_MouseSC = Preprocess_SpTrans(151673)
genelist_MouseSC = Preprocess_SpTrans(151674)
genelist_MouseSC = Preprocess_SpTrans(151675)
genelist_MouseSC = Preprocess_SpTrans(151676)
Preprocess_SpTrans(151673)
Preprocess_SpTrans(151674)
Preprocess_SpTrans(151675)
Preprocess_SpTrans(151676)




### ------------------------------------------------------------------------------------------------------- ###
###    Create a merged data set for 73, 74 and 75
### ------------------------------------------------------------------------------------------------------- ###

dataSection1 = sc.read("../data/MouseSC/data_{studyID}.h5ad".format(studyID = 151673)) 
print(dataSection1)
dataSection2 = sc.read("../data/MouseSC/data_{studyID}.h5ad".format(studyID = 151674)) 
print(dataSection2)
dataSection3 = sc.read("../data/MouseSC/data_{studyID}.h5ad".format(studyID = 151675)) 
print(dataSection3)
dataSection4 = sc.read("../data/MouseSC/data_{studyID}.h5ad".format(studyID = 151676)) 
print(dataSection4)


dataSection = dataSection1.concatenate(dataSection2, dataSection3, dataSection4)
print(dataSection)

dataSection.write_h5ad("../data/MouseSC/MergeTrains73747576.h5ad")


### ------------------------------------------------------------------------------------------------------- ###
###    Export the MouseSC data with subsampling
### ------------------------------------------------------------------------------------------------------- ###

MouseSC = sc.read("../data/MouseSC/MouseSC_snRNA.h5ad")

MouseSCsub = MouseSC[:,genelist_MouseSC]
# Remove the White Matters
MouseSCsub = MouseSCsub [ MouseSCsub.obs["Layer"]!=7 ]

del MouseSCsub.obs['class_label']


MouseSCsub.write_h5ad("../data/MouseSC/MouseSC_spa_DE_snRNA.h5ad")


sc.tl.rank_genes_groups(MouseSCsub, 'Layer_Ch', method = 'wilcoxon', key_added = "wilcoxon")
sc.pl.rank_genes_groups(MouseSCsub, n_genes = 50, sharey = False, key="wilcoxon", save = 'MouseSCsub.pdf')


### ------------------------------------------------------------------------------------------------------- ###
###    Produce the tSNE data
### ------------------------------------------------------------------------------------------------------- ###
dataSection1 = sc.read("../data/MouseSC/MouseSC_spa_DE_snRNA.h5ad")
spc.get_zscore(dataSection1)

pca = PCA(n_components=50)

## Compute PCA of cells
principalComponents = pca.fit_transform(dataSection1.X)
PCs = ['PC_{i}'.format(i=i) for i in range(1,51)]
principalDf = pd.DataFrame(data = principalComponents, columns = PCs)


tsne = TSNE(n_components=2, verbose=1, perplexity=40, n_iter=300)
tsne_results = tsne.fit_transform(principalDf)
tSNEDf = pd.DataFrame(data = tsne_results)
tSNEDf["Layer"] = dataSection1.obs["Layer"].to_numpy()
tSNEDf.to_csv("../output/MouseSC/tSNEDf.csv")


principalDf["Layer"] = dataSection1.obs["Layer"].to_numpy()
principalDf.to_csv("../output/MouseSC/PCA_selectedGenes.csv")
