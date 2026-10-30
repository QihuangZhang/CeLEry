# Data augmentation

This tutorial enlarges a small spatial reference with replicates that a variational autoencoder generates from it. It then trains a domain classifier on the original and the generated spots together and applies the classifier to single-cell RNA-seq data.

Run the code in an interactive Python session, such as Jupyter, started at the root of a clone of the repository, with CeLEry installed as described in [Installation](../install.md). To run it as a script on Windows or macOS, follow [Running CeLEry in a script](../reading-output.md#running-celery-in-a-script). The guard is needed here even with the arguments below, because [DataAugmentation](../reference.md#dataaugmentation) and [Predict_domain](../reference.md#predict_domain) always start worker processes.

The paper describes the procedure in the Methods subsection "Data augmentation for the training ST data". The Results subsection "Data augmentation generates realistic replicates of ST data" compares generated and observed gene maps in Fig. 2. Fig. 3b shows how the number of replicates changes the accuracy of layer recovery.

## When to use it

The autoencoder treats each gene as an image over the spot grid, so the procedure suits references whose spots lie on a grid, such as 10x Visium. The Discussion of the paper states that references without a grid layout, such as MERSCOPE, MERFISH or 10x Xenium data, would not benefit from it. The paper also states that references with large numbers of spots or cells do not need it.

## Data

The notebook [CeLEry-data-agumentation.ipynb](https://github.com/QihuangZhang/CeLEry/blob/main/code_paper/7_data_augmentation/CeLEry-data-agumentation.ipynb) augments a reference of the mouse posterior brain whose resolution was enhanced with TESLA. This reference is not in the repository. The notebook also stops before any classifier is trained. This tutorial therefore uses two files in `tutorial/data/`, the same as [Tissue layers](layers.md):

- `DataLayerToy.h5ad` is the reference, a spatial transcriptomics section with 3,611 spots and 1,134 genes. The columns `x2` and `x3` of its `.obs` hold the integer grid coordinates of each spot. The column `Layer` gives the layer of each spot as a number from 1 to 7, for layers L1 to L6 and the white matter.
- `AlzheimerToy.h5ad` is the query, 3,000 cells from a single-cell RNA-seq study of Alzheimer's disease.

`DataLayerToy.h5ad` is the only reference in the repository that has both grid coordinates and domain labels.

## Prepare the reference and the query

```python
import scanpy as sc
import torch
import CeLEry as cel
```

```python
Rdata = sc.read("tutorial/data/DataLayerToy.h5ad")
Qdata = sc.read("tutorial/data/AlzheimerToy.h5ad")

genes = sorted(set(Rdata.var_names) & set(Qdata.var_names))
Rdata = Rdata[:, genes].copy()
Qdata = Qdata[:, genes].copy()

cel.get_zscore(Rdata)
cel.get_zscore(Qdata)
Qdata = cel.drop_NaN(Qdata)
Rdata = Rdata[:, Qdata.var_names].copy()
```

The trained classifier accepts only the genes of its reference, in the same order. The gene set is therefore fixed here, before augmentation. The list is sorted because the order of a Python set changes from one session to the next. [get_zscore](../reference.md#get_zscore) standardizes each gene separately, so selecting the genes first gives the same values with less memory. Fourteen of the shared genes are zero in every cell of the query, so their standardized values are undefined. [drop_NaN](../reference.md#drop_nan) removes these genes from the query. The last line removes them from the reference. The reference and the query keep 1,120 genes.

```python
Rdata = Rdata[Rdata.obs.sort_values(by = ["x2", "x3"]).index].copy()

Rdata.obs["domain_id"] = Rdata.obs["Layer"] - 1
domain_count = Rdata.obs["domain_id"].value_counts().sort_index()
domain_weights = torch.tensor((domain_count.iloc[-1] / domain_count).to_numpy(), dtype = torch.float32)
```

The first line sorts the spots by their grid coordinates. `DataAugmentation` sorts its own copy of the reference in this way and writes the replicates in that order. [AugFit_domain](../reference.md#augfit_domain) takes the original expression and the labels in the order of its input. Sorting the reference here makes the two orders agree. Without it, the generated spots would receive the labels of other spots.

`AugFit_domain` requires the domain labels to be integers from 0 to N-1, where N is the number of domains, so layers 1 to 7 become domains 0 to 6. As in [Spatial domains](domains.md), the weight of a domain is the number of spots in the last domain divided by the number of spots in that domain. `AugFit_domain` passes the weights to the model as given, so they must be a float32 PyTorch tensor. [Fit_domain](../reference.md#fit_domain) accepts a pandas Series instead, because it converts the weights itself.

## Generate replicates

```python
cel.DataAugmentation(RefDataOrigin = Rdata, obs_location = ["x2", "x3"], path = "output/augmentation",
                     filename = "DLPFC", n_clusters = 100, beta = 1e-5, nrep = 2, generateplot = False)
```

`DataAugmentation` runs these steps, whose functions are in `CeLEry_package/CeLEry/data_augmentation.py`:

1. It turns each gene into an image over the grid given by the columns `obs_location`, with the value 0 where the grid has no spot.
2. It groups the gene images into `n_clusters` clusters with k-means. With `clusterready = True`, it reads the clusters of an earlier run instead.
3. It trains the autoencoder with `FitGenModel` and then `FitGenModel_continue`.
4. With `generateplot = True`, it draws the image of every gene and ten generated versions of it with `GeneratePlot`.
5. It generates `nrep` replicates of every gene with `Data_Generation` and keeps the values at the positions of the spots.

`DataAugmentation` returns `None`. It writes these files under `<path>/DataAugmentation/`. In the file names, Python writes `beta = 1e-5` as `1e-05`.

| Step | File under `<path>/DataAugmentation/` | Content | File in this tutorial |
|---|---|---|---|
| Clustering, only with `clusterready = False` | `<filename>_cluster.npy` | The cluster of each gene | `DLPFC_cluster.npy` |
| Training | `<filename>_CVAE_<beta>.obj` | The autoencoder as a pickle file, written after the first 249 epochs and overwritten after the next 200 | `DLPFC_CVAE_1e-05.obj` |
| Plots, only with `generateplot = True` | `<filename>_Generation/Glimps/Gen<beta>/img<j>.pdf` and `img<j>var<i>.pdf` | The image of gene `j`, counted from 0 in the gene order of the reference, and its generated versions, `i` from 0 to 9 | Not written |
| Generation | `DataGen/<filename>_data_gen_<beta>_n<nrep>.npy` | The replicates, an array of shape (`nrep`, genes, spots) | `DataGen/DLPFC_data_gen_1e-05_n2.npy`, shape (2, 1120, 3611) |

The training lengths are fixed in the code. `FitGenModel` trains for 249 epochs with Kullback-Leibler (KL) annealing. The weight of the KL term starts at 0 and rises by `beta / 4` every 50 epochs until it reaches `beta`. `FitGenModel_continue` then trains for 200 more epochs at this weight. `DataAugmentation` has no argument that changes these numbers. Every one of the 449 epochs passes each of the 1,120 genes through the autoencoder, so the time of this step grows with the number of genes. A complete run has not been timed.

With `generateplot = True`, `GeneratePlot` writes 11 PDF files per gene, 12,320 files for this reference. It also leaves every figure open, so its memory use grows with the number of genes. This tutorial therefore sets `generateplot = False`.

Since scikit-learn 1.4, the `KMeans` call in `DataAugmentation` (`data_augmentation.py` line 39) runs one initialization instead of ten, which on the 356 gene images of `tutorial/data/MousePosteriorToy.h5ad` gave clusters that agreed with a ten-initialization run at an adjusted Rand index of only 0.44 ([RENOVATION_LOG.md](https://github.com/QihuangZhang/CeLEry/blob/main/RENOVATION_LOG.md)). The gene clusters, and with them the generated replicates, therefore differ from those that the 2022 to 2023 environment of the paper produced (see [FUTURE_WORK.md](https://github.com/QihuangZhang/CeLEry/blob/main/FUTURE_WORK.md)).

`FitGenModel` and `Data_Generation` seed Python, NumPy and PyTorch with 2021. The k-means step uses `random_state = 0`. Whether two complete runs give identical replicates has not been tested.

## Train the domain classifier

```python
cel.AugFit_domain(RefDataOrigin = Rdata, domain_weights = domain_weights, domainkey = "domain_id",
                  hidden_dims = [50, 10, 5], num_epochs_max = 500, beta = 1e-5, nrep = 2,
                  path = "output/augmentation", filename = "DLPFC", num_workers = 0)
```

`AugFit_domain` reads the replicates from `<path>/DataAugmentation/DataGen/<filename>_data_gen_<beta>_n<nrep>.npy`, so `path`, `filename`, `beta` and `nrep` must match the call to `DataAugmentation`. It adds the original spots to the two replicates, which gives three copies of the 3,611 spots for training. It trains the classifier of `Fit_domain`, with three hidden layers of 50, 10 and 5 nodes, the default network that the Methods of the paper give. Training stops after `num_epochs_max` epochs at the latest.

`num_workers = 0` loads the training data in the main process. The default of 4 starts four new worker processes in every epoch, which on Windows and macOS adds their start-up time to every epoch.

`AugFit_domain` returns `None`. The trained classifier is saved with pickle to `<path>/DataAugmentation/PredictionModel/<filename>_domain_<beta>_n<nrep>.obj`, here `output/augmentation/DataAugmentation/PredictionModel/DLPFC_domain_1e-05_n2.obj`.

## Predict the domain of each cell

```python
torch.manual_seed(0)
pred_domain = cel.Predict_domain(data_test = Qdata, class_num = 7,
                                 path = "output/augmentation/DataAugmentation/PredictionModel",
                                 filename = "DLPFC_domain_1e-05_n2")

layers = {0: "L1", 1: "L2", 2: "L3", 3: "L4", 4: "L5", 5: "L6", 6: "WM"}
print(Qdata.obs["pred_domain"].map(layers).value_counts())
```

`Predict_domain` reads `<path>/<filename>.obj`, so `path` and `filename` point to the file that `AugFit_domain` wrote. It returns a matrix with one row per cell and one column per domain, which holds the probabilities of domains 0 to 6. It adds the domain with the highest probability to `Qdata.obs` as `pred_domain`. It also writes the predicted domains and the probabilities to `DLPFC_domain_1e-05_n2_probmat.csv` in the same folder, with the predicted domain in the first column. Domain d is layer d + 1 of the reference. The last line counts the cells assigned to each layer, where WM is the white matter.

`torch.manual_seed(0)` precedes the call because dropout stays active at prediction, so repeated calls differ unless the seed is set. [Reading the output](../reading-output.md#layers-and-domains) explains how to read these results.

## Settings in the paper

- `beta` weighs the KL term against the reconstruction error. The Methods of the paper give 1e-5, which is also the default of `DataAugmentation`. The notebook in `code_paper/7_data_augmentation` uses 1e-6.
- `nrep` is the number of replicates. The Discussion of the paper reports that two replicates generally gave the best performance. Fig. 3b compares several numbers of replicates.
- `n_clusters` is the number of gene clusters, 100 by default. The Methods of the paper do not give the number used in the analyses.
- The section "Paper and code (report only, nothing changed)" of [RENOVATION_LOG.md](https://github.com/QihuangZhang/CeLEry/blob/main/RENOVATION_LOG.md) lists further differences between the code and the Methods, such as the numbers of channels in the autoencoder.
