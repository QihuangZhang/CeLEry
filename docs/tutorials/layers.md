# Tissue layers

This tutorial trains CeLEry on a spatial transcriptomics section annotated with seven layers and assigns each cell of a single-cell RNA-seq dataset to one of these layers.

Run the code in an interactive Python session, such as Jupyter, started at the root of a clone of the repository, with CeLEry installed as described in [Installation](../install.md). To run it as a script on Windows or macOS, follow [Running CeLEry in a script](../reading-output.md#running-celery-in-a-script).

## Data

The two input files are in `tutorial/data/`:

- `DataLayerToy.h5ad` is the reference, a spatial transcriptomics section with 3,611 spots and 1,134 genes. The column `Layer` of its `.obs` gives the layer of each spot as a number from 1 to 7.
- `AlzheimerToy.h5ad` is the query, 3,000 cells with 26,423 genes from a single-cell RNA-seq study of Alzheimer's disease.

## Load and standardize the data

```python
import scanpy as sc
import torch
import CeLEry as cel

import os
import numpy as np
import warnings
warnings.filterwarnings("ignore")
```

```python
Qdata = sc.read("tutorial/data/AlzheimerToy.h5ad")
Rdata = sc.read("tutorial/data/DataLayerToy.h5ad")

cel.get_zscore(Qdata)
cel.get_zscore(Rdata)
```

[get_zscore](../reference.md#get_zscore) standardizes each gene to mean 0 and standard deviation 1, separately within each dataset.

```python
Qdata
```

```text
AnnData object with n_obs × n_vars = 3000 × 26423
    obs: 'cellname', 'sample', 'groupid', 'final_celltype', 'maxprob', 'imaxprob', 'trem2', 'atscore', 'apoe', 'sampleID', 'n_counts'
    var: 'Ensembl', 'genename', 'n_cells'
```

```python
Rdata
```

```text
AnnData object with n_obs × n_vars = 3611 × 1134
    obs: 'x2', 'x3', 'Layer', 'Layer_character', 'n_counts'
    var: 'gene_ids', 'feature_types', 'genome', 'genename'
    uns: 'wilcoxon'
```

## Remove genes without variation

A gene with the same value in every cell has a standard deviation of 0, so its z-scores are NaN. Such genes usually point to incomplete preprocessing. A NaN in the input makes the output of the network NaN. Check both datasets before training and prediction.

```python
np.isnan(Rdata.X).any()
```

```python
np.isnan(Qdata.X).any()
```

The check returns `False` for the reference and `True` for the query.

```python
Qdata = cel.drop_NaN(Qdata)
Qdata
```

```text
AnnData object with n_obs × n_vars = 3000 × 23796
    obs: 'cellname', 'sample', 'groupid', 'final_celltype', 'maxprob', 'imaxprob', 'trem2', 'atscore', 'apoe', 'sampleID', 'n_counts'
    var: 'Ensembl', 'genename', 'n_cells'
```

[drop_NaN](../reference.md#drop_nan) returns a new AnnData object without the genes that contain NaN. The new object keeps `.obs` and `.var` but no other annotation, such as `.uns`.

## Match the genes

The network reads the genes by their position in the expression matrix. The query must therefore contain the genes of the reference in the same order. The code below keeps the genes that both datasets share, in sorted order.

```python
common_gene = sorted(set(Qdata.var_names) & set(Rdata.var_names))
#
Query_select = Qdata[:,common_gene]
Reference_select = Rdata[:,common_gene]
```

The list is sorted because the order of a Python set changes from one session to the next, and the trained model changes with the gene order.

## Weight the layers

Layers with few spots contribute little to an unweighted training loss. [Fit_layer](../reference.md#fit_layer) therefore multiplies the loss of each spot by the weight of its layer. Here the weight of a layer is the number of spots in layer 7 divided by the number of spots in that layer. The weights are thus inversely proportional to the size of the layer, with weight 1 for layer 7.

```python
layer_count =  Reference_select.obs["Layer"].value_counts().sort_index()
layer_weight = layer_count[7]/layer_count[0:7]
layer_weight
```

`layer_weight` is a pandas Series with one weight per layer, in layer order. `Fit_layer` needs a pandas Series here, because it converts the weights with `.to_numpy()`. The length of the Series sets the number of layers of the model.

## Train the layer model

`Fit_layer` does not create its output folder, so the first line creates it.

```python
os.makedirs("output/tutorial", exist_ok=True)
model_train = cel.Fit_layer (data_train = Reference_select, layer_weights = layer_weight, layerkey = "Layer", 
                             hidden_dims = [30, 25, 15], num_epochs_max = 500, path = "output/tutorial", filename = "Org_layer")
```

[Fit_layer](../reference.md#fit_layer) trains a network with an ordinal loss, which treats the layers as ordered from 1 to 7. Its arguments are:

- `layer_weights`: the weight of each layer, as a pandas Series.
- `layerkey`: the column of `.obs` that holds the layer of each spot (default `"layer"`). The layers must be numbered from 1 to the number of layers.
- `hidden_dims`: the number of nodes in each of the three hidden layers of the network (default `[10, 5, 2]`).
- `num_epochs_max`: the maximum number of epochs (default 500).
- `path` and `filename`: the folder of the model file and its name without the extension.

`Fit_layer` returns `None`, so `model_train` holds no model. The trained network is saved with pickle to `output/tutorial/Org_layer.obj`. Training for 500 epochs can take an hour or more. [Tuning the training](coordinates.md#tuning-the-training) describes the batch size, the learning rate and `number_error_try`.

## Predict the layer of each cell

[Predict_layer](../reference.md#predict_layer) applies the saved network to the query. Its argument `class_num` is the number of layers of the model, here 7. `torch.manual_seed(0)` precedes each call because dropout stays active at prediction, so repeated calls differ unless the seed is set.

```python
torch.manual_seed(0)
pred_layer = cel.Predict_layer(data_test = Query_select, class_num = 7, path = "output/tutorial", filename = "Org_layer", predtype = "deterministic")
pred_layer
```

With `predtype = "deterministic"`, `Predict_layer` returns a NumPy array with the predicted layer of each cell, from 1 to 7.

```python
torch.manual_seed(0)
probability_each_layer = cel.Predict_layer(data_test = Query_select, class_num = 7, path = "output/tutorial", filename = "Org_layer", predtype = "probabilistic")
probability_each_layer
```

With `predtype = "probabilistic"`, the default, `Predict_layer` returns a matrix with one row per cell and one column per layer. Each row holds the probabilities of layers 1 to 7 for one cell. Any other value of `predtype` returns both results, as the list `[probabilities, layers]`. Since both calls start from the same seed, the predicted layer of each cell is the layer with the highest probability in its row.

Each call also adds the predicted layer to `Query_select.obs`, in three columns:

- `pred_layer`: the layer as an integer.
- `pred_layer_str`: the layer as a string.
- `layer_cel_pred`: the layer as a floating-point number.

Each call writes the predicted layers and the probabilities to `output/tutorial/Org_layer_probmat.csv`, without a header. The second call overwrites the file of the first. The first column of the file holds the predicted layer. Columns 2 to 8 hold the probabilities of layers 1 to 7. [Reading the output](../reading-output.md#layers-and-domains) explains how to read these results.
