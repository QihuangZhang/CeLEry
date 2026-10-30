# Guide for the data augmentation study

## Paper

Results subsection: "Data augmentation generates realistic replicates of ST data".

Methods subsection: "Data augmentation for the training ST data".

Figures: Fig. 2, Supplementary Fig. 1 and Supplementary Fig. 2.

Supplementary Fig. 3 compares generated and biological replicates of the LIBD data. This folder does not contain its code.

## Pipeline

CeLEry-data-agumentation.ipynb (one notebook, run from top to bottom)

The notebook reads files that it does not create:

- `data/MouseBrain/MP1_sudo.h5ad` and `data/MouseBrain/V1_Mouse_Brain_Sagittal_Posterior_image.tif`, described in `3_Mouse_10x_Visium/README.md`;
- `data/Mouse2D/MP1_SVG.py`, the gene list, which is in the repository as `tutorial/data/Mouse2D/MP1_SVG.py`;
- the hold-out indices `output/CeLEry/index10.npy`, `index30.npy` and `index50.npy` of `3_Mouse_10x_Visium`;
- the gene clusters `output/CeLEry/cluster.npy` and `output/CeLEry/DataAugmentation/full_cluster.npy`.

No file in the repository writes the two cluster files. With `clusterready = False`, `cel.DataAugmentation` runs the K-means clustering itself and writes `full_cluster.npy`.

## What the notebook does

Section "1. Data Preperation" builds the reference of `3_Mouse_10x_Visium` in the steps that `3_Mouse_10x_Visium/README.md` lists, including the split by the hold-out indices.

Section "2. Data Augmentation" has four parts.

- Its first cell saves the gene images of the full reference and of the three training sets to `output/CeLEry/full_geneimg.npy`, `train10_geneimg.npy`, `train30_geneimg.npy` and `train50_geneimg.npy`.
- The cell that calls `cel.DataAugmentation` augments the full reference with `beta = 1e-6` and `clusterready = True`. With the current package, this call reads the clusters from `output/CeLEry/DataAugmentation/full_cluster.npy`. It then trains the variational autoencoder for 249 epochs and 200 more. It saves the image of every gene and ten generated versions of it as PDF files in `output/CeLEry/DataAugmentation/full_Generation/Glimps/Gen1e-06/`. It saves two generated replicates to `output/CeLEry/DataAugmentation/DataGen/full_data_gen_1e-06_n2.npy`. Fig. 2b, Fig. 2c and Supplementary Fig. 2 show such images for selected genes.
- The two cells that call `cel.DataAugmentation2` would redraw these images for `beta = 1e-6` and `beta = 1e-7`. The package has no function of this name. An earlier cell of the notebook defines `DataAugmentation2`, which draws the images of a trained model with `cel.GeneratePlot`.
- The last cell reads the clusters from `output/CeLEry/cluster.npy`. For each of 50 clusters, it maps the mean, the standard deviation and the log coefficient of variation of the genes in the cluster at every super-pixel. It saves the maps as PDF files in `output/CeLEry/DataAugmentation/clustermean/`. Supplementary Fig. 1 shows the mean and the standard deviation for six clusters.

The notebook also defines its own copy of `DataAugmentation`. The analysis calls the package function `cel.DataAugmentation` instead. `cel.DataAugmentation` creates 100 clusters by default (`n_clusters`). The last cell plots 50 clusters.

The tutorial [docs/tutorials/augmentation.md](../../docs/tutorials/augmentation.md) runs `DataAugmentation` and `AugFit_domain` on data that are in the repository.

## Datasets

10x Visium mouse posterior brain, section V1_Mouse_Brain_Sagittal_Posterior (Data availability statement of the paper):
https://support.10xgenomics.com/spatial-gene-expression/datasets/1.0.0/V1_Mouse_Brain_Sagittal_Posterior

The TESLA output `MP1_sudo.h5ad` is not in the repository. `tutorial/data/MousePosteriorToy.h5ad` has the columns that the notebook creates (`x`, `y` and `inner`), with 5,824 super-pixels and 356 genes.
