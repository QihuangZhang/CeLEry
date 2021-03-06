
<h1><center>CeLEry Tutorial</center></h1>


Author: Qihuang Zhang*, Jian Hu, Kejie Li, Baohong Zhang, David Dai,
Edward B. Lee, Rui Xiao, Mingyao Li*


### Outline

1.  Installation
2.  Import modules
3.  Data Loading
4.  Run CeLEry
5.  Data Augmentation

### 1. Installation 

To install CeLEry package you must make sure that your python version is
over 3.5. If you don't know the version of python you can check it by:
``` {.python}
import platform
platform.python_version()
```

Note: Because CeLery depends on pytorch, you should make sure torch is
correctly installed. Now you can install the current
release of CeLEry by the following three ways:

#### 1.1 PyPI: Directly install the package from PyPI. 
``` 
pip3 install CeLEryPy
#Note: you need to make sure that the pip is for python3, or we should install CeLEry by
python3 -m pip install CeLEryPy
pip3 install CeLEryPy
#If you do not have permission (when you get a permission denied error), you should install CeLEry by
pip3 install --user CeLEryPy
```

#### 1.2 Github

Download the package from Github and install it locally:
``` 
git clone https://github.com/QihuangZhang/CeLEry
cd CeLEry/CeLEry_package/
python3 setup.py install --user
```

#### 1.3 Anaconda 

If you do not have Python3.5 or Python3.6 installed, consider installing
Anaconda (see Installing Anaconda). After installing Anaconda, you can
create a new environment, for example, CeLEry (you can change to any
name you like).
``` {.python}
#create an environment called CeLEry
conda create -n CeLEry python=3.7.9
#activate your environment 
conda activate CeLEry
git clone https://github.com/QihuangZhang/CeLEry
cd CeLEry/CeLEry_package/
python3 setup.py build
python3 setup.py install
conda deactivate
```

### 2. Import python modules

```
import scanpy as sc
import torch
import CeLEry as cel

import os,csv,re
import pandas as pd
import numpy as np
import math
from skimage import io, color

from scipy.sparse import issparse
import random, torch
import warnings
warnings.filterwarnings("ignore")
import pickle

import json

```

``` {.python}
cel.__version__
```

### 3. Load-in data 

The current version of CeLEry takes two input data, the reference data
and the query data. The reference data is used to trained the model and
the query data is the dataset where predictions (or classifications) are
made.

1.  The Reference Data (the spatial transcriptomics data): AnnData
    format including:

-   the gene expression matrix spot by gene ($n_{spot}$ by $k$);
-   the spot-specific information (e.g., coordinates, layer, etc.)
    

2.  The Query Data (the scRNA-seq data): AnnData format including:

-   the gene expression matrix cell by gene ($n_{cell}$ by $k$);
-   the demographic information for each cell (e.g., cell type, layer,
    etc.) 

AnnData stores a data matrix `.X` together with annotations of
observations `.obs`, variables `.var` and unstructured annotations
`.uns`.

``` {.python}
"""
#Read original 10x_h5 data and save it to h5ad
from scanpy import read_10x_h5
adata = read_10x_h5("../tutorial/data/151673/expression_matrix.h5")
spatial = pd.read_csv("../tutorial/data/151673/positions.txt",sep=",",header=None,na_filter=False,index_col=0) 
adata.obs["x1"] = spatial[1]
adata.obs["x2"] = spatial[2]
adata.obs["x3"] = spatial[3]
adata.obs["x4"] = spatial[4]
adata.obs["x5"] = spatial[5]
adata.obs["x_array"] = adata.obs["x2"]
adata.obs["y_array"] = adata.obs["x3"]
adata.obs["x_pixel"] = adata.obs["x4"]
adata.obs["y_pixel"] = adata.obs["x5"]
#Select captured samples
adata = adata[adata.obs["x1"]==1]
adata.var_names = [i.upper() for i in list(adata.var_names)]
adata.var["genename"] = adata.var.index.astype("str")
adata.write_h5ad("../tutorial/data/151673/sample_data.h5ad")
"""
#Read in gene expression and spatial location
Qdata = sc.read("data/tutorial/MouseSCToy.h5ad")
Rdata = sc.read("data/tutorial/MousePosteriorToy.h5ad")

Rdata
```

```
>>> Rdata
AnnData object with n_obs ?? n_vars = 5824 ?? 356
    obs: 'x', 'y', 'inner'
    var: 'genename'
```

Here, `Qdata` stores the annodated query data (scRNA-seq/snRNA-seq data) and `Rdata` is the annoated reference data collected from spatial transcriptomics.

Before inplementing our methods, we often normalize both the reference
data and the query data:

``` {.python}
cel.get_zscore(Qdata)
cel.get_zscore(Rdata)
```

### 4. Run CeLEry 

We demonstrate the implemnetation of CeLEry in two tasks. In the first
task, CeLEry is implemented to predict the 2D coordinates for the cells.
In the second task, we classify the cells into different layers.

#### 4.1 Analysis Task 1: Coordinates Recovery

In the first task, we train a deep neural network using the reference
data, and then apply the trained model to predict the location of the
cells (or spots) in the query data.

##### Training 

First, we train the model using spatial transcriptomic data. The trained
model will also automately save as an `.obj` file in the specified path. This step can take an hour depending on the structure of the neural network.

``` {.python}
model_train = cel.Fit_cord (data_train = Rdata, hidden_dims = [30, 25, 15], num_epochs_max = 500, path = "output/tutorial", filename = "Org_Mousesc")
```


The fitting function `Fit_cord` involves the following parameters:

-   data_train (an annotated matrix): the input data

-   hidden_dims (a list of length three): the width of the neural
    network in each layer. In total, three layers are considered in the
    neural network.

-   num_epochs_max: maximum number of epochs considered in the training
    procedure.

-   path: the directory that saving the model object

-   filename: the name of the model object to be saved to the path.

##### Prediction

Then, we apply the trained model to the query data to predict the
coordinates of the cells.


The prediction function `Predict_cord` contains three arguments:

-   data_test (an annotated matrix): the input query dat

-   path: the directory that saving the model object

-   filename: the name of the model object to be saved

The method implementation outputs the 2D coordinates in `pred_cord`. A
`.csv` file will also saved with the name \"predmatrix\".


Example code:

``` {.python}
pred_cord = cel.Predict_cord (data_test = Qdata, path = "output/tutorial", filename = "Org_Mousesc")

pred_cord
```

Output:

```
array([[0.67726576, 0.49435037],
       [0.42489582, 0.51810944],
       [0.07367212, 0.4977431 ],
       ...,
       [0.72734278, 0.43093637],
       [0.63597023, 0.10852443],
       [0.3674576 , 0.50103331]])
```

Each row of the output matrix represents a 2D coordinates of the predicted cells.

#### 4.2 Analysis Task 2: Layer Recovery

In the second task, we use CeLEry to classify the cells into different
layers. First, we load the spatial transcriptomics data with annotation
for layers together with a single cell RNA data collected from an
Alzheimer\'s study.

``` {.python}
Qdata = sc.read("data/tutorial/AlzheimerToy.h5ad")
Rdata = sc.read("data/tutorial/DataLayerToy.h5ad")

cel.get_zscore(Qdata)
cel.get_zscore(Rdata)
```

For many times, the gene set in query data (`Qdata`) is different from the reference data (`Rdata`). To ensure the model trained by reference data is applicable to the query data, it is essential to the genes sets are identical for both datasets.

``` {.python}
common_gene = list(set(Qdata.var_names) & set(Rdata.var_names))
#
Query_select = Qdata[:,common_gene]
Reference_select = Rdata[:,common_gene]
```

Output of comparison after gene filtering:
``` {.python}
>>> Qdata
AnnData object with n_obs ?? n_vars = 2452 ?? 1134
    obs: 'cellname', 'sample', 'groupid', 'final_celltype', 'maxprob', 'imaxprob', 'trem2', 'atscore', 'apoe', 'sampleID', 'n_counts'
    var: 'Ensembl', 'genename', 'n_cells'
>>> Query_select
View of AnnData object with n_obs ?? n_vars = 2452 ?? 1134
    obs: 'cellname', 'sample', 'groupid', 'final_celltype', 'maxprob', 'imaxprob', 'trem2', 'atscore', 'apoe', 'sampleID', 'n_counts'
    var: 'Ensembl', 'genename', 'n_cells'
```


The sample size of the spots in each layer could be very different, leading to the poor performance of the classification in some layers. We consider weighting the sample from each layer. A typical way to choose weight is to use $1/sample size$.

``` {.python}
layer_count =  Reference_select.obs["Layer"].value_counts().sort_index()
layer_weight = layer_count[7]/layer_count[0:7]
layer_weights = torch.tensor(layer_weight.to_numpy())
layer_weights
```

Output:
```
tensor([1.8791, 2.0277, 0.5187, 2.3532, 0.7623, 0.7413, 1.0000],
       dtype=torch.float64)
```


We train the model using the function `Fit_layer`. The model will
returned and also save as an `.obj` object to be loaded later. This step can take an hour according to the structure of the neural network.

``` {.python}
model_train = cel.Fit_layer (data_train = Reference_select, layer_weights = layer_weights, layerkey = "Layer", 
                             hidden_dims = [30, 25, 15], num_epochs_max = 500, path = "output/tutorial", filename = "Org_layer")
```

Then, we apply the trained model to the scRNA-seq/snRNA-seq data:

```
pred_layer = cel.Predict_layer(data_test = Query_select, class_num = 7, path = "output/tutorial", filename = "Org_layer", predtype = "deterministic")
pred_layer
```

Output:
```
array([4., 4., 6., ..., 6., 5., 4.])
```

```
probability_each_layer = cel.Predict_layer(data_test = Query_select, class_num = 7, path = "output/tutorial", filename = "Org_layer", predtype = "probabilistic")
probability_each_layer
```

Output:
```
array([[ 2.27034092e-04,  1.87861919e-02,  3.39182794e-01, ...,
         6.73830733e-02,  1.59902696e-03,  5.36697644e-06],
       [ 3.27110291e-04,  2.68441439e-02,  4.18585479e-01, ...,
         4.77917679e-02,  1.11017306e-03,  3.72435829e-06],
       [-0.00000000e+00,  4.52995300e-06,  1.26361847e-04, ...,
         1.24190569e-01,  8.50280046e-01,  2.23746095e-02],
       ...,
       [ 1.19209290e-07,  9.77516174e-06,  2.75790691e-04, ...,
         2.34772027e-01,  7.47992218e-01,  1.03732459e-02],
       [ 2.26497650e-06,  1.93536282e-04,  5.41210175e-03, ...,
         7.42786407e-01,  1.36679530e-01,  5.30853984e-04],
       [ 7.78675079e-04,  6.15803003e-02,  5.94597340e-01, ...,
         2.06700731e-02,  4.66533442e-04,  1.56409408e-06]])

```

<!-- ### 5. Data Augmentation 

Due to the limit sample size of the reference data (i.e, spatial
transcriptomic data), we can also implementment an augmentation
procedure to enlarge the sample size before implementing CeLEry.

(UNDER CONSTRUCTION)
::: -->
