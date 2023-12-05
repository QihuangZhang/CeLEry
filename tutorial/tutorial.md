
<h1><center>CeLEry Tutorial</center></h1>


Author: Qihuang Zhang*, Shunzhou Jiang, Amelia Schroeder, Jian Hu, Kejie Li, Baohong Zhang, David Dai,
Edward B. Lee, Rui Xiao, Mingyao Li*


### Outline

1.  Installation
2.  Import modules
3.  Data Loading
4.  Run CeLEry

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
AnnData object with n_obs × n_vars = 5824 × 356
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
`.csv` file will also saved with the name `predmatrix`. The prediction results is also saved in `Qdata.obs`.


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

Each row of the output matrix represents a 2D coordinates of the predicted cells. The results is also appearing in the updaded `Qdata.obs`.

``` {.python}
Qdata.obs
```


<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>exp_component_name</th>
      <th>platform_label</th>
      <th>cluster_color</th>
      <th>cluster_order</th>
      <th>cluster_label</th>
      <th>class_color</th>
      <th>class_order</th>
      <th>class_label</th>
      <th>subclass_color</th>
      <th>subclass_order</th>
      <th>...</th>
      <th>injection_roi_label</th>
      <th>injection_type_color</th>
      <th>injection_type_id</th>
      <th>injection_type_label</th>
      <th>cortical_layer_label</th>
      <th>outlier_call</th>
      <th>outlier_type</th>
      <th>n_counts</th>
      <th>x_cord_pred</th>
      <th>y_cord_pred</th>
    </tr>
    <tr>
      <th>sample_name</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>CAGCGACAGAGACTTA-L8TX_180115_01_F11</th>
      <td>CAGCGACAGAGACTTA-17L8TX_180115_01_F11</td>
      <td>10x</td>
      <td>#07C6D9</td>
      <td>196.0</td>
      <td>196_L4/5 IT CTX</td>
      <td>#00ADEE</td>
      <td>2.0</td>
      <td>Glutamatergic</td>
      <td>#00E5E5</td>
      <td>17.0</td>
      <td>...</td>
      <td>NaN</td>
      <td>#FF7373</td>
      <td>1</td>
      <td>NaN</td>
      <td>All</td>
      <td>False</td>
      <td>nan</td>
      <td>602.0</td>
      <td>0.700048</td>
      <td>0.534017</td>
    </tr>
    <tr>
      <th>AGCTCTCCATCGACGC-L8TX_180115_01_F09</th>
      <td>AGCTCTCCATCGACGC-23L8TX_180115_01_F09</td>
      <td>10x</td>
      <td>#00FFFF</td>
      <td>188.0</td>
      <td>188_L4/5 IT CTX</td>
      <td>#00ADEE</td>
      <td>2.0</td>
      <td>Glutamatergic</td>
      <td>#00E5E5</td>
      <td>17.0</td>
      <td>...</td>
      <td>NaN</td>
      <td>#FF7373</td>
      <td>1</td>
      <td>NaN</td>
      <td>All</td>
      <td>False</td>
      <td>nan</td>
      <td>1597.0</td>
      <td>0.427784</td>
      <td>0.516523</td>
    </tr>
    <tr>
      <th>AGAGCGACACCTCGTT-L8TX_180406_01_C08</th>
      <td>AGAGCGACACCTCGTT-7L8TX_180406_01_C08</td>
      <td>10x</td>
      <td>#297F98</td>
      <td>331.0</td>
      <td>331_L6 CT ENTm</td>
      <td>#00ADEE</td>
      <td>2.0</td>
      <td>Glutamatergic</td>
      <td>#174596</td>
      <td>34.0</td>
      <td>...</td>
      <td>NaN</td>
      <td>#FF7373</td>
      <td>1</td>
      <td>NaN</td>
      <td>0</td>
      <td>False</td>
      <td>nan</td>
      <td>869.0</td>
      <td>0.071109</td>
      <td>0.467619</td>
    </tr>
    <tr>
      <th>TGGCGCAAGTACACCT-L8TX_180115_01_C11</th>
      <td>TGGCGCAAGTACACCT-11L8TX_180115_01_C11</td>
      <td>10x</td>
      <td>#28758B</td>
      <td>329.0</td>
      <td>329_L6 CT CTX</td>
      <td>#00ADEE</td>
      <td>2.0</td>
      <td>Glutamatergic</td>
      <td>#2D8CB8</td>
      <td>33.0</td>
      <td>...</td>
      <td>NaN</td>
      <td>#FF7373</td>
      <td>1</td>
      <td>NaN</td>
      <td>All</td>
      <td>False</td>
      <td>nan</td>
      <td>1876.0</td>
      <td>0.900394</td>
      <td>0.176353</td>
    </tr>
    <tr>
      <th>ACCTTTAGTTATCACG-L8TX_180115_01_D11</th>
      <td>ACCTTTAGTTATCACG-12L8TX_180115_01_D11</td>
      <td>10x</td>
      <td>#00FFFF</td>
      <td>188.0</td>
      <td>188_L4/5 IT CTX</td>
      <td>#00ADEE</td>
      <td>2.0</td>
      <td>Glutamatergic</td>
      <td>#00E5E5</td>
      <td>17.0</td>
      <td>...</td>
      <td>NaN</td>
      <td>#FF7373</td>
      <td>1</td>
      <td>NaN</td>
      <td>All</td>
      <td>False</td>
      <td>nan</td>
      <td>2068.0</td>
      <td>0.345863</td>
      <td>0.418795</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>TTTGTCATCGTACCGG-L8TX_180221_01_B11</th>
      <td>TTTGTCATCGTACCGG-1L8TX_180221_01_B11</td>
      <td>10x</td>
      <td>#B36C76</td>
      <td>13.0</td>
      <td>13_Lamp5</td>
      <td>#F05A28</td>
      <td>1.0</td>
      <td>GABAergic</td>
      <td>#DA808C</td>
      <td>3.0</td>
      <td>...</td>
      <td>NaN</td>
      <td>#FF7373</td>
      <td>1</td>
      <td>NaN</td>
      <td>All</td>
      <td>False</td>
      <td>nan</td>
      <td>966.0</td>
      <td>0.222131</td>
      <td>0.285191</td>
    </tr>
    <tr>
      <th>ACCTTTAGTGATGATA-L8TX_180221_01_D11</th>
      <td>ACCTTTAGTGATGATA-3L8TX_180221_01_D11</td>
      <td>10x</td>
      <td>#30E6BA</td>
      <td>131.0</td>
      <td>131_L2 IT RSPv</td>
      <td>#00ADEE</td>
      <td>2.0</td>
      <td>Glutamatergic</td>
      <td>#2DB38A</td>
      <td>10.0</td>
      <td>...</td>
      <td>NaN</td>
      <td>#FF7373</td>
      <td>1</td>
      <td>NaN</td>
      <td>All</td>
      <td>False</td>
      <td>nan</td>
      <td>833.0</td>
      <td>0.248778</td>
      <td>0.135483</td>
    </tr>
    <tr>
      <th>AGCTCTCCATAGAAAC-L8TX_180115_01_D11</th>
      <td>AGCTCTCCATAGAAAC-12L8TX_180115_01_D11</td>
      <td>10x</td>
      <td>#02F970</td>
      <td>183.0</td>
      <td>183_L2/3 IT CTX</td>
      <td>#00ADEE</td>
      <td>2.0</td>
      <td>Glutamatergic</td>
      <td>#0BE652</td>
      <td>15.0</td>
      <td>...</td>
      <td>NaN</td>
      <td>#FF7373</td>
      <td>1</td>
      <td>NaN</td>
      <td>All</td>
      <td>False</td>
      <td>nan</td>
      <td>1423.0</td>
      <td>0.726116</td>
      <td>0.442133</td>
    </tr>
    <tr>
      <th>AGAGCGACACAACGTT-L8TX_180406_01_H01</th>
      <td>AGAGCGACACAACGTT-1L8TX_180406_01_H01</td>
      <td>10x</td>
      <td>#299337</td>
      <td>141.0</td>
      <td>141_L3 IT ENTm</td>
      <td>#00ADEE</td>
      <td>2.0</td>
      <td>Glutamatergic</td>
      <td>#65CA2F</td>
      <td>12.0</td>
      <td>...</td>
      <td>NaN</td>
      <td>#FF7373</td>
      <td>1</td>
      <td>NaN</td>
      <td>0</td>
      <td>False</td>
      <td>nan</td>
      <td>911.0</td>
      <td>0.632229</td>
      <td>0.108702</td>
    </tr>
    <tr>
      <th>AGCTCTCCATATACGC-L8TX_180406_01_B06</th>
      <td>AGCTCTCCATATACGC-5L8TX_180406_01_B06</td>
      <td>10x</td>
      <td>#297F98</td>
      <td>331.0</td>
      <td>331_L6 CT ENTm</td>
      <td>#00ADEE</td>
      <td>2.0</td>
      <td>Glutamatergic</td>
      <td>#174596</td>
      <td>34.0</td>
      <td>...</td>
      <td>NaN</td>
      <td>#FF7373</td>
      <td>1</td>
      <td>NaN</td>
      <td>0</td>
      <td>False</td>
      <td>nan</td>
      <td>708.0</td>
      <td>0.330912</td>
      <td>0.482509</td>
    </tr>
  </tbody>
</table>
<p>3000 rows × 59 columns</p>
</div>


##### Confidence Score

To quantify the uncertainty of the prediction, we can produce confidence score for each predicted subject. We first train the deep neural network using `cel.Fit_region`.

The usage of `Fit_region` is similar to Fit_cord. An extra parmameter `alpha` is needed to indicate the confidence level.

``` {.python}
model_train = cel.Fit_region (data_train = Rdata, alpha = 0.95, hidden_dims = [30, 25, 15], num_epochs_max = 500, path = "output/example", filename = "ConfRegion_Mousesc")

model_train
```

Then, we use `Predict_region` to evaluate the confidence score for each prediction subject. This function produce two new columns in the object of query data: area and conf_score.

The `area` records the area of the predicted circle, which will cover the truth with probability `alpha`. The confidence score measures the uncertainty of the prediction, which is defined as `1 - area`. Higher confidence level represents a lower uncertainty in the prediction.

``` {.python}
pred_region = cel.Predict_region (data_test = Qdata, path = "output/example", filename = "ConfRegion_Mousesc")

Qdata.obs
```

Output:

<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>exp_component_name</th>
      <th>platform_label</th>
      <th>cluster_color</th>
      <th>cluster_order</th>
      <th>cluster_label</th>
      <th>class_color</th>
      <th>class_order</th>
      <th>class_label</th>
      <th>subclass_color</th>
      <th>subclass_order</th>
      <th>...</th>
      <th>injection_roi_label</th>
      <th>injection_type_color</th>
      <th>injection_type_id</th>
      <th>injection_type_label</th>
      <th>cortical_layer_label</th>
      <th>outlier_call</th>
      <th>outlier_type</th>
      <th>n_counts</th>
      <th>area_record</th>
      <th>conf_score</th>
    </tr>
    <tr>
      <th>sample_name</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>CAGCGACAGAGACTTA-L8TX_180115_01_F11</th>
      <td>CAGCGACAGAGACTTA-17L8TX_180115_01_F11</td>
      <td>10x</td>
      <td>#07C6D9</td>
      <td>196.0</td>
      <td>196_L4/5 IT CTX</td>
      <td>#00ADEE</td>
      <td>2.0</td>
      <td>Glutamatergic</td>
      <td>#00E5E5</td>
      <td>17.0</td>
      <td>...</td>
      <td>NaN</td>
      <td>#FF7373</td>
      <td>1</td>
      <td>NaN</td>
      <td>All</td>
      <td>False</td>
      <td>nan</td>
      <td>602.0</td>
      <td>0.521508</td>
      <td>0.478492</td>
    </tr>
    <tr>
      <th>AGCTCTCCATCGACGC-L8TX_180115_01_F09</th>
      <td>AGCTCTCCATCGACGC-23L8TX_180115_01_F09</td>
      <td>10x</td>
      <td>#00FFFF</td>
      <td>188.0</td>
      <td>188_L4/5 IT CTX</td>
      <td>#00ADEE</td>
      <td>2.0</td>
      <td>Glutamatergic</td>
      <td>#00E5E5</td>
      <td>17.0</td>
      <td>...</td>
      <td>NaN</td>
      <td>#FF7373</td>
      <td>1</td>
      <td>NaN</td>
      <td>All</td>
      <td>False</td>
      <td>nan</td>
      <td>1597.0</td>
      <td>0.464650</td>
      <td>0.535350</td>
    </tr>
    <tr>
      <th>AGAGCGACACCTCGTT-L8TX_180406_01_C08</th>
      <td>AGAGCGACACCTCGTT-7L8TX_180406_01_C08</td>
      <td>10x</td>
      <td>#297F98</td>
      <td>331.0</td>
      <td>331_L6 CT ENTm</td>
      <td>#00ADEE</td>
      <td>2.0</td>
      <td>Glutamatergic</td>
      <td>#174596</td>
      <td>34.0</td>
      <td>...</td>
      <td>NaN</td>
      <td>#FF7373</td>
      <td>1</td>
      <td>NaN</td>
      <td>0</td>
      <td>False</td>
      <td>nan</td>
      <td>869.0</td>
      <td>0.478806</td>
      <td>0.521194</td>
    </tr>
    <tr>
      <th>TGGCGCAAGTACACCT-L8TX_180115_01_C11</th>
      <td>TGGCGCAAGTACACCT-11L8TX_180115_01_C11</td>
      <td>10x</td>
      <td>#28758B</td>
      <td>329.0</td>
      <td>329_L6 CT CTX</td>
      <td>#00ADEE</td>
      <td>2.0</td>
      <td>Glutamatergic</td>
      <td>#2D8CB8</td>
      <td>33.0</td>
      <td>...</td>
      <td>NaN</td>
      <td>#FF7373</td>
      <td>1</td>
      <td>NaN</td>
      <td>All</td>
      <td>False</td>
      <td>nan</td>
      <td>1876.0</td>
      <td>0.533365</td>
      <td>0.466635</td>
    </tr>
    <tr>
      <th>ACCTTTAGTTATCACG-L8TX_180115_01_D11</th>
      <td>ACCTTTAGTTATCACG-12L8TX_180115_01_D11</td>
      <td>10x</td>
      <td>#00FFFF</td>
      <td>188.0</td>
      <td>188_L4/5 IT CTX</td>
      <td>#00ADEE</td>
      <td>2.0</td>
      <td>Glutamatergic</td>
      <td>#00E5E5</td>
      <td>17.0</td>
      <td>...</td>
      <td>NaN</td>
      <td>#FF7373</td>
      <td>1</td>
      <td>NaN</td>
      <td>All</td>
      <td>False</td>
      <td>nan</td>
      <td>2068.0</td>
      <td>0.419726</td>
      <td>0.580274</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>TTTGTCATCGTACCGG-L8TX_180221_01_B11</th>
      <td>TTTGTCATCGTACCGG-1L8TX_180221_01_B11</td>
      <td>10x</td>
      <td>#B36C76</td>
      <td>13.0</td>
      <td>13_Lamp5</td>
      <td>#F05A28</td>
      <td>1.0</td>
      <td>GABAergic</td>
      <td>#DA808C</td>
      <td>3.0</td>
      <td>...</td>
      <td>NaN</td>
      <td>#FF7373</td>
      <td>1</td>
      <td>NaN</td>
      <td>All</td>
      <td>False</td>
      <td>nan</td>
      <td>966.0</td>
      <td>0.599244</td>
      <td>0.400756</td>
    </tr>
    <tr>
      <th>ACCTTTAGTGATGATA-L8TX_180221_01_D11</th>
      <td>ACCTTTAGTGATGATA-3L8TX_180221_01_D11</td>
      <td>10x</td>
      <td>#30E6BA</td>
      <td>131.0</td>
      <td>131_L2 IT RSPv</td>
      <td>#00ADEE</td>
      <td>2.0</td>
      <td>Glutamatergic</td>
      <td>#2DB38A</td>
      <td>10.0</td>
      <td>...</td>
      <td>NaN</td>
      <td>#FF7373</td>
      <td>1</td>
      <td>NaN</td>
      <td>All</td>
      <td>False</td>
      <td>nan</td>
      <td>833.0</td>
      <td>0.490772</td>
      <td>0.509228</td>
    </tr>
    <tr>
      <th>AGCTCTCCATAGAAAC-L8TX_180115_01_D11</th>
      <td>AGCTCTCCATAGAAAC-12L8TX_180115_01_D11</td>
      <td>10x</td>
      <td>#02F970</td>
      <td>183.0</td>
      <td>183_L2/3 IT CTX</td>
      <td>#00ADEE</td>
      <td>2.0</td>
      <td>Glutamatergic</td>
      <td>#0BE652</td>
      <td>15.0</td>
      <td>...</td>
      <td>NaN</td>
      <td>#FF7373</td>
      <td>1</td>
      <td>NaN</td>
      <td>All</td>
      <td>False</td>
      <td>nan</td>
      <td>1423.0</td>
      <td>0.445069</td>
      <td>0.554931</td>
    </tr>
    <tr>
      <th>AGAGCGACACAACGTT-L8TX_180406_01_H01</th>
      <td>AGAGCGACACAACGTT-1L8TX_180406_01_H01</td>
      <td>10x</td>
      <td>#299337</td>
      <td>141.0</td>
      <td>141_L3 IT ENTm</td>
      <td>#00ADEE</td>
      <td>2.0</td>
      <td>Glutamatergic</td>
      <td>#65CA2F</td>
      <td>12.0</td>
      <td>...</td>
      <td>NaN</td>
      <td>#FF7373</td>
      <td>1</td>
      <td>NaN</td>
      <td>0</td>
      <td>False</td>
      <td>nan</td>
      <td>911.0</td>
      <td>0.674069</td>
      <td>0.325931</td>
    </tr>
    <tr>
      <th>AGCTCTCCATATACGC-L8TX_180406_01_B06</th>
      <td>AGCTCTCCATATACGC-5L8TX_180406_01_B06</td>
      <td>10x</td>
      <td>#297F98</td>
      <td>331.0</td>
      <td>331_L6 CT ENTm</td>
      <td>#00ADEE</td>
      <td>2.0</td>
      <td>Glutamatergic</td>
      <td>#174596</td>
      <td>34.0</td>
      <td>...</td>
      <td>NaN</td>
      <td>#FF7373</td>
      <td>1</td>
      <td>NaN</td>
      <td>0</td>
      <td>False</td>
      <td>nan</td>
      <td>708.0</td>
      <td>0.442143</td>
      <td>0.557857</td>
    </tr>
  </tbody>
</table>
<p>3000 rows × 59 columns</p>
</div>

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

``` {.python}
Qdata
```

```
    AnnData object with n_obs × n_vars = 3000 × 26423
        obs: 'cellname', 'sample', 'groupid', 'final_celltype', 'maxprob', 'imaxprob', 'trem2', 'atscore', 'apoe', 'sampleID', 'n_counts'
        var: 'Ensembl', 'genename', 'n_cells'
```


``` {.python}
Rdata
```


```    
    AnnData object with n_obs × n_vars = 3611 × 1134
        obs: 'x2', 'x3', 'Layer', 'Layer_character', 'n_counts'
        var: 'gene_ids', 'feature_types', 'genome', 'genename'
        uns: 'wilcoxon'
```


One common error that can happen is that after normalization, the gene
expression can have NAs due to the lack of variability. (Usually because the data is not well preprocessed.) This may cause
the problem when training the data or evaluate the query. It is a good
practice to check whether NA exists in the data:


``` {.python}
np.isnan(Rdata.X).any()
```

```    
ArrayView(False)
```
The training data is good



``` {.python}
np.isnan(Qdata.X).any()
```

```
    ArrayView(True)
```
The testing data has NaN.


``` {.python}
Qdata = cel.drop_NaN(Qdata)
Qdata
```

```
    AnnData object with n_obs × n_vars = 3000 × 23796
        obs: 'cellname', 'sample', 'groupid', 'final_celltype', 'maxprob', 'imaxprob', 'trem2', 'atscore', 'apoe', 'sampleID', 'n_counts'
        var: 'Ensembl', 'genename', 'n_cells'
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
AnnData object with n_obs × n_vars = 2452 × 1134
    obs: 'cellname', 'sample', 'groupid', 'final_celltype', 'maxprob', 'imaxprob', 'trem2', 'atscore', 'apoe', 'sampleID', 'n_counts'
    var: 'Ensembl', 'genename', 'n_cells'
>>> Query_select
    View of AnnData object with n_obs × n_vars = 3611 × 1120
        obs: 'x2', 'x3', 'Layer', 'Layer_character', 'n_counts'
        var: 'gene_ids', 'feature_types', 'genome', 'genename'
        uns: 'wilcoxon'
```


The sample size of the spots in each layer could be very different, leading to the poor performance of the classification in some layers. We consider weighting the sample from each layer. A typical way to choose weight is to use $1/sample size$.

``` {.python}
layer_count =  Reference_select.obs["Layer"].value_counts().sort_index()
layer_weight = layer_count[7]/layer_count[0:7]
layer_weight
```

Output:
```
[1.8791, 2.0277, 0.5187, 2.3532, 0.7623, 0.7413, 1.0000]
```


We train the model using the function `Fit_layer`. The model will
returned and also save as an `.obj` object to be loaded later. This step can take an hour according to the structure of the neural network.

``` {.python}
model_train = cel.Fit_layer (data_train = Reference_select, layer_weights = layer_weight, layerkey = "Layer", 
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

#### 4.3 Analysis Task 3:  Domain Recovery

In the third task, we use CeLEry to classify the cells into different domains without layer structures. i.e., we don't assume the domain having an ordinal relationship.

```
Qdata = sc.read("tutorial/data/AlzheimerToy.h5ad")
Rdata = sc.read("tutorial/data/DataLayerToy.h5ad")

cel.get_zscore(Qdata)
cel.get_zscore(Rdata)

common_gene = list(set(Qdata.var_names) & set(Rdata.var_names))
#
Query_select = Qdata[:,common_gene]
Reference_select = Rdata[:,common_gene]
```

Note that the classes need to span from 0 to N-1. #!# Important

```
Reference_select.obs["domain_id"] = Rdata.obs["Layer"] -1
domain_count =  Reference_select.obs["domain_id"].value_counts().sort_index()
domain_weight = domain_count[len(domain_count)-1]/domain_count[0:len(domain_count)]

domain_weights = torch.tensor(domain_weight.to_numpy(), dtype=torch.float32)
domain_weights
```

We train the model using the function ``Fit_domain``. The fitted model will be returned and also save as an ``.obj`` object to be loaded later. 

```
model_train = cel.Fit_domain (data_train = Reference_select, domain_weights = domain_weight, domainkey = "domain_id", 
                             hidden_dims = [30, 25, 15], num_epochs_max = 500, path = "output/example", filename = "PreOrg_domain")

model_train
```

To predict the results, we implement `Predict_domain` function. A probabilitic classificatoin matrix will be returned and the deterministic domain prediction will be attached to `.obs`.

```
pred_domain = cel.Predict_domain (data_test = Query_select, class_num = 7,   path = "output/example", filename = "PreOrg_domain")

Query_select.obs
```

Output:

<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>cellname</th>
      <th>sample</th>
      <th>groupid</th>
      <th>final_celltype</th>
      <th>maxprob</th>
      <th>imaxprob</th>
      <th>trem2</th>
      <th>atscore</th>
      <th>apoe</th>
      <th>sampleID</th>
      <th>n_counts</th>
      <th>pred_domain</th>
      <th>pred_domain_str</th>
      <th>domain_cel_pred</th>
    </tr>
    <tr>
      <th>index</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>GGGAATGGTTATTCTC-1-C2</th>
      <td>GGGAATGGTTATTCTC-1</td>
      <td>C2</td>
      <td>C</td>
      <td>Oli</td>
      <td>0.827</td>
      <td>827</td>
      <td>WT</td>
      <td>A-T-</td>
      <td>E3/E3</td>
      <td>1</td>
      <td>1277.0</td>
      <td>6</td>
      <td>6</td>
      <td>6.0</td>
    </tr>
    <tr>
      <th>ATCATGGCAGTGAGTG-1-C3</th>
      <td>ATCATGGCAGTGAGTG-1</td>
      <td>C3</td>
      <td>C</td>
      <td>Ast</td>
      <td>0.822</td>
      <td>822</td>
      <td>WT</td>
      <td>A-T-</td>
      <td>E3/E3</td>
      <td>2</td>
      <td>1961.0</td>
      <td>6</td>
      <td>6</td>
      <td>6.0</td>
    </tr>
    <tr>
      <th>CGACCTTAGTGTCCCG-1-I4</th>
      <td>CGACCTTAGTGTCCCG-1</td>
      <td>I4</td>
      <td>I</td>
      <td>Oli</td>
      <td>0.827</td>
      <td>827</td>
      <td>WT</td>
      <td>A+T-</td>
      <td>E3/E3</td>
      <td>3</td>
      <td>706.0</td>
      <td>6</td>
      <td>6</td>
      <td>6.0</td>
    </tr>
    <tr>
      <th>ATCACGAGTTGGAGGT-1-I1</th>
      <td>ATCACGAGTTGGAGGT-1</td>
      <td>I1</td>
      <td>I</td>
      <td>Ex</td>
      <td>0.779</td>
      <td>779</td>
      <td>WT</td>
      <td>A+T-</td>
      <td>E3/E4</td>
      <td>6</td>
      <td>2541.0</td>
      <td>6</td>
      <td>6</td>
      <td>6.0</td>
    </tr>
    <tr>
      <th>CGACCTTCACAGATTC-1-I4</th>
      <td>CGACCTTCACAGATTC-1</td>
      <td>I4</td>
      <td>I</td>
      <td>In</td>
      <td>0.775</td>
      <td>775</td>
      <td>WT</td>
      <td>A+T-</td>
      <td>E3/E3</td>
      <td>3</td>
      <td>4185.0</td>
      <td>6</td>
      <td>6</td>
      <td>6.0</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>GGAATAACATACGCTA-1-T3</th>
      <td>GGAATAACATACGCTA-1</td>
      <td>T3</td>
      <td>T</td>
      <td>Ast</td>
      <td>0.811</td>
      <td>811</td>
      <td>R47H</td>
      <td>A+T+</td>
      <td>E3/E3</td>
      <td>9</td>
      <td>583.0</td>
      <td>6</td>
      <td>6</td>
      <td>6.0</td>
    </tr>
    <tr>
      <th>TTTCCTCGTCGGGTCT-1-I2</th>
      <td>TTTCCTCGTCGGGTCT-1</td>
      <td>I2</td>
      <td>I</td>
      <td>Ex</td>
      <td>0.545</td>
      <td>545</td>
      <td>WT</td>
      <td>A+T-</td>
      <td>E3/E4</td>
      <td>5</td>
      <td>4217.0</td>
      <td>6</td>
      <td>6</td>
      <td>6.0</td>
    </tr>
    <tr>
      <th>TCTGAGAAGAATAGGG-T1COMB</th>
      <td>TCTGAGAAGAATAGGG</td>
      <td>T1</td>
      <td>T</td>
      <td>Ast</td>
      <td>0.821</td>
      <td>821</td>
      <td>R47H</td>
      <td>A+T+</td>
      <td>E4/E4</td>
      <td>15</td>
      <td>1455.0</td>
      <td>6</td>
      <td>6</td>
      <td>6.0</td>
    </tr>
    <tr>
      <th>TTTCCTCTCGTCCAGG-1-I2</th>
      <td>TTTCCTCTCGTCCAGG-1</td>
      <td>I2</td>
      <td>I</td>
      <td>End</td>
      <td>0.721</td>
      <td>721</td>
      <td>WT</td>
      <td>A+T-</td>
      <td>E3/E4</td>
      <td>5</td>
      <td>1166.0</td>
      <td>6</td>
      <td>6</td>
      <td>6.0</td>
    </tr>
    <tr>
      <th>GCAGCCATCTGGCGTG-1-T2</th>
      <td>GCAGCCATCTGGCGTG-1</td>
      <td>T2</td>
      <td>T</td>
      <td>Ex</td>
      <td>0.355</td>
      <td>355</td>
      <td>R47H</td>
      <td>A+T+</td>
      <td>E3/E4</td>
      <td>11</td>
      <td>2434.0</td>
      <td>6</td>
      <td>6</td>
      <td>6.0</td>
    </tr>
  </tbody>
</table>
<p>3000 rows × 14 columns</p>
</div>

<!-- ### 5. Data Augmentation 

Due to the limit sample size of the reference data (i.e, spatial
transcriptomic data), we can also implementment an augmentation
procedure to enlarge the sample size before implementing CeLEry.

(UNDER CONSTRUCTION)
::: -->
