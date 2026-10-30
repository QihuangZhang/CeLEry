# Guide for the mouse brain 10x Visium study

## Paper

Results subsection: "Benchmark evaluation for 2D location recovery in mouse brain with 10x Visium data as spatial reference".

Figures: Fig. 5a to 5c, Supplementary Fig. 4a and Supplementary Fig. 5.

This folder does not contain the code for Fig. 5d (noise added to the test data), Supplementary Fig. 4b (reference without resolution enhancement) or Supplementary Fig. 6 (certainty scores of the predicted locations).

## Pipeline

hold-out indices  ->
Mouse_CeLEry.ipynb, Mouse_Tangram.ipynb, Mouse_novosparc.ipynb and Mouse_spotOTsc.ipynb (in any order)

The four notebooks share the hold-out indices `output/CeLEry/index10.npy`, `index30.npy` and `index50.npy`, which are not in the repository. Commented code in `Mouse_Tangram.ipynb`, `Mouse_novosparc.ipynb` and `Mouse_spotOTsc.ipynb` created them with `random.seed(2021)` and `random.sample`, after `cel.get_zscore`. That code saves the indices to `../output/CeLEry/`, while the four notebooks load them from `output/CeLEry/`.

## What each notebook does

Each notebook builds the same reference in these steps:

- read the expression at the super-pixel level that TESLA produced, `data/MouseBrain/MP1_sudo.h5ad`, and keep the genes listed in `data/Mouse2D/MP1_SVG.py`;
- keep the region with `x` from 3000 to 7000 and `y` from 6200 to 10500, in pixels of the histology image, and divide the coordinates by 50 to place the super-pixels on a grid;
- keep the super-pixels inside the tissue, found with a gray-level threshold on the histology image `data/MouseBrain/V1_Mouse_Brain_Sagittal_Posterior_image.tif`;
- standardize each gene with `cel.get_zscore`;
- hold out 10%, 30% or 50% of the super-pixels as the test set.

`Mouse_CeLEry.ipynb` trains `cel.Fit_cord` on each training set and saves the models `HighReso_10`, `HighReso_30` and `HighReso_50` in `output/CeLEry/2022Nov`. Its function `evaluate` predicts the locations of the held-out super-pixels. It writes the predicted and true coordinates to `location_HighReso_<rate>_predict.csv`, where `<rate>` is 10, 30 or 50. It also draws the recovered and the true expression maps of three genes (`CeLEry_HighReso_<rate>_SSIM_<k>.pdf` and `CeLEry_HighReso_<rate>_SSIM_anchor_<k>.pdf`, with `<k>` from 1 to 3). For each gene, it saves the structural similarity index (SSIM), the Pearson correlation and the mean squared error between the recovered and the true map (`CeLEry_HighReso_<rate>_SSIM.npy`, `_cor.npy` and `_MSE.npy`).

`Mouse_Tangram.ipynb`, `Mouse_novosparc.ipynb` and `Mouse_spotOTsc.ipynb` write the same files for Tangram, novoSpaRc and spaOTsc to `output/CeLEry/Tangram/`, `output/CeLEry/novosparc/` and `output/CeLEry/spaOTsc/`.

The predicted and true coordinates give the correlations of pairwise distances in Fig. 5a, Fig. 5b and Supplementary Fig. 4a. The recovered gene maps correspond to Fig. 5c. The SSIM values correspond to Supplementary Fig. 5. The code that draws these panels is not in this folder.

`evaluate` and the functions `Tangram_run`, `novosparc_run` and `spaOTsc_run` call `cel.get_SSIM`. With current scikit-image releases, `cel.get_SSIM` raises a `ValueError` (break 5 in `RENOVATION_LOG.md` at the root of the repository).

## Datasets

10x Visium mouse posterior brain, section V1_Mouse_Brain_Sagittal_Posterior (Data availability statement of the paper):
https://support.10xgenomics.com/spatial-gene-expression/datasets/1.0.0/V1_Mouse_Brain_Sagittal_Posterior

The histology image `V1_Mouse_Brain_Sagittal_Posterior_image.tif` is part of this dataset.

The TESLA output `MP1_sudo.h5ad` is not in the repository. `tutorial/data/MousePosteriorToy.h5ad` has the columns that these notebooks create (`x`, `y` and `inner`), with 5,824 super-pixels and 356 genes.

The gene list is in the repository as `tutorial/data/Mouse2D/MP1_SVG.py`.
