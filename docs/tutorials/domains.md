# Spatial domains

This tutorial assigns each cell of a single-cell RNA-seq dataset to a spatial domain, using the seven layers of the [tissue layers](layers.md) data as domains without an order.

Run the code in an interactive Python session, such as Jupyter, started at the root of a clone of the repository, with CeLEry installed as described in [Installation](../install.md). To run it as a script on Windows or macOS, follow [Running CeLEry in a script](../reading-output.md#running-celery-in-a-script).

Unlike [Fit_layer](../reference.md#fit_layer), [Fit_domain](../reference.md#fit_domain) treats the classes as unordered. It therefore also applies to domains that do not follow each other in space.

## Data

The input files are those of [Tissue layers](layers.md), in `tutorial/data/`:

- `DataLayerToy.h5ad` is the reference, a spatial transcriptomics section with 3,611 spots and 1,134 genes. The column `Layer` of its `.obs` gives the layer of each spot as a number from 1 to 7.
- `AlzheimerToy.h5ad` is the query, 3,000 cells from a single-cell RNA-seq study of Alzheimer's disease.

## Load the data and match the genes

```python
import scanpy as sc
import torch
import CeLEry as cel

import os
import warnings
warnings.filterwarnings("ignore")
```

```python
Qdata = sc.read("tutorial/data/AlzheimerToy.h5ad")
Rdata = sc.read("tutorial/data/DataLayerToy.h5ad")

cel.get_zscore(Qdata)
cel.get_zscore(Rdata)
Qdata = cel.drop_NaN(Qdata)

common_gene = sorted(set(Qdata.var_names) & set(Rdata.var_names))
#
Query_select = Qdata[:,common_gene]
Reference_select = Rdata[:,common_gene]
```

[get_zscore](../reference.md#get_zscore) standardizes each gene to mean 0 and standard deviation 1, separately within each dataset. Fourteen of the reference genes do not vary across the 3,000 query cells, so their z-scores are NaN, and [drop_NaN](../reference.md#drop_nan) removes them. Without this step, the network would return NaN for every query cell, which `Predict_domain` turns into a probability of 1.0 for every domain, and every cell would be assigned to the last domain without an error. The original tutorial in `tutorial/tutorial.md` skips this step. `Query_select` and `Reference_select` keep the genes that both datasets share, in the same order. The list is sorted because the order of a Python set changes from one session to the next, and the trained model changes with the gene order. [Tissue layers](layers.md) explains these steps.

## Define the domains and their weights

`Fit_domain` requires the domain labels to be integers from 0 to N-1, where N is the number of domains. The layers run from 1 to 7, so the first line below subtracts 1 to give the domains 0 to 6. As in [Tissue layers](layers.md), the weight of a domain is the number of spots in the last domain divided by the number of spots in that domain.

```python
Reference_select.obs["domain_id"] = Rdata.obs["Layer"] -1
domain_count =  Reference_select.obs["domain_id"].value_counts().sort_index()
domain_weight = domain_count[len(domain_count)-1]/domain_count[0:len(domain_count)]

domain_weights = torch.tensor(domain_weight.to_numpy(), dtype=torch.float32)
domain_weights
```

`domain_weights` shows the seven weights as a PyTorch tensor. `Fit_domain` receives the pandas Series `domain_weight` instead, because it converts the weights with `.to_numpy()`.

## Train the domain model

`Fit_domain` does not create its output folder, so the first line creates it.

```python
os.makedirs("output/example", exist_ok=True)
model_train = cel.Fit_domain (data_train = Reference_select, domain_weights = domain_weight, domainkey = "domain_id", 
                             hidden_dims = [30, 25, 15], num_epochs_max = 500, path = "output/example", filename = "PreOrg_domain")

model_train
```

`Fit_domain` trains a classifier with a weighted cross-entropy loss, which does not assume any order among the domains. Its arguments are:

- `domain_weights`: the weight of each domain, as a pandas Series. Its length sets the number of domains.
- `domainkey`: the column of `.obs` that holds the domain of each spot (default `"layer"`).
- `hidden_dims`: the number of nodes in each of the three hidden layers of the network (default `[10, 5, 2]`).
- `num_epochs_max`: the maximum number of epochs (default 500).
- `path` and `filename`: the folder of the model file and its name without the extension.

`Fit_domain` returns `None`, so `model_train` displays nothing. The trained network is saved with pickle to `output/example/PreOrg_domain.obj`. Training for 500 epochs can take an hour or more. [Tuning the training](coordinates.md#tuning-the-training) describes the batch size, the learning rate and `number_error_try`.

## Predict the domain of each cell

[Predict_domain](../reference.md#predict_domain) applies the saved network to the query. Its argument `class_num` is the number of domains of the model, here 7. `torch.manual_seed(0)` precedes the call because dropout stays active at prediction, so repeated calls differ unless the seed is set.

```python
torch.manual_seed(0)
pred_domain = cel.Predict_domain (data_test = Query_select, class_num = 7,   path = "output/example", filename = "PreOrg_domain")

Query_select.obs
```

With the default `predtype = "probabilistic"`, `Predict_domain` returns a matrix with one row per cell and one column per domain. Each row holds the probabilities of domains 0 to 6 for one cell. With `predtype = "deterministic"`, it returns the predicted domain of each cell instead. Any other value returns both results, as the list `[probabilities, domains]`. The predicted domain of a cell is the domain with the highest probability.

The call adds the predicted domain, from 0 to 6, to `Query_select.obs` in three columns:

- `pred_domain`: the domain as an integer.
- `pred_domain_str`: the domain as a string.
- `domain_cel_pred`: the domain as a floating-point number.

Domain d is layer d + 1 of the reference. The call also writes the predicted domains and the probabilities to `output/example/PreOrg_domain_probmat.csv`, without a header. The first column of the file holds the predicted domain. Columns 2 to 8 hold the probabilities of domains 0 to 6. [Reading the output](../reading-output.md#layers-and-domains) explains how to read these results.
