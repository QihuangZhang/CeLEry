# Guide for the breast cancer 10x Xenium study

## Paper

Both subfolders analyze Replicate 1 of the 10x Xenium breast cancer data.

`Domain_prediction` recovers the cell type of half of the cells after training on the other half. The cell types are the clusters of a K-means clustering with K = 10. The Discussion of the paper reports this analysis. Supplementary Fig. 14 shows it.

`2D_locationRecovery` recovers the coordinates of 10% of the cells after training on the other 90%. It uses the data of the Results subsection "Benchmark evaluation for 2D location recovery in breast cancer with 10x Xenium data as spatial reference". That subsection, Fig. 10 and Supplementary Figs. 11 to 13 train on Replicate 1 and test on Replicate 2 instead. No figure of the paper shows the 10% hold-out within Replicate 1. The code for Fig. 10 and Supplementary Figs. 11 to 13 is not in this folder.

The notebook names call the two designs Scheme 2 (`Domain_prediction`) and Scheme 4 (`2D_locationRecovery`).

## Pipeline

2D_locationRecovery/Xenium_BreastCancer_CELERY_Rep1_Scheme4_2DRecovery.ipynb  ->
2D_locationRecovery/Xenium_BreastCancer_Tangram_Rep1_Scheme4_2DRecovery.ipynb, Xenium_BreastCancer_novosparc_Rep1_Scheme4_2DRecovery.ipynb and Xenium_BreastCancer_spaOTsc_Rep1_Scheme4_2DRecovery.ipynb (in any order)

Domain_prediction/Xenium_BreastCancer_CELEREY_Scheme2_domainPred.ipynb  ->
Domain_prediction/Xenium_BreastCancer_tangram_Scheme2_domainPred.ipynb, Xenium_BreastCancer_novosparc_Rep1_Scheme2_domainPred.ipynb and Xenium_BreastCancer_spaOTsc_Rep1_Scheme2_domainPred.ipynb (in any order)

In `2D_locationRecovery`, the CeLEry notebook writes the training and test cells to `trainCell_index.csv` and `testCell_index.csv`, and the other three notebooks read these files, so all four methods use the same cells. In `Domain_prediction`, the test sets differ between the methods: the Tangram notebook repeats the split of the CeLEry notebook with `train_test_split` and `random_state = 1`, and the novoSpaRc and spaOTsc notebooks draw random subsets of 20,000 cells.

## What each notebook does

`Xenium_BreastCancer_CELERY_Rep1_Scheme4_2DRecovery.ipynb`:

- removes the cells listed in `cell_ID_toRemove_filtered_75_25.csv` and standardizes each gene with `cel.get_zscore`;
- draws 10% of the cells as the test set after `random.seed(2023)`;
- trains `cel.Fit_cord` on the pixel coordinates (`x_pixel` and `y_pixel`) of the other cells and predicts the test cells with `cel.Predict_cord`;
- draws the predicted against the true locations with `cel.RelocationPlot`.

The Tangram notebook maps the test cells onto the training cells with `tg.map_cells_to_space` and saves the mapping with `write_h5ad`. The novoSpaRc and spaOTsc notebooks save their mapping probabilities to `location_pred_probs.csv`.

`Xenium_BreastCancer_CELEREY_Scheme2_domainPred.ipynb`:

- reads the K-means clusters from `clusters.csv` and removes the cells listed in `cell_ID_toRemove_filtered_25_25.csv`;
- splits the cells in half, stratified by cluster (`train_test_split` with `random_state = 1`);
- trains `cel.Fit_domain` and predicts the clusters of the test cells with `cel.Predict_domain`;
- prints the accuracy and draws the true clusters, the predicted clusters and a contingency heatmap.

The Tangram, novoSpaRc and spaOTsc notebooks repeat this analysis with their method. The novoSpaRc and spaOTsc notebooks use 20,000 training cells and 20,000 test cells. The three notebooks save their contingency heatmaps as PNG files, for example `xenium_rep1_scheme2_tangram_domain_contigency.png`. Supplementary Fig. 14 reports accuracies of 96.3% for CeLEry, 88.4% for Tangram, 85.1% for spaOTsc and 86.3% for novoSpaRc.

## Datasets

10x Xenium FFPE human breast cancer data (Data availability statement of the paper):
https://www.10xgenomics.com/welcome?closeUrl=%2F&lastTouchOfferName=Xenium+Preprint+Dataset&lastTouchOfferType=Dataset&product=xenium&redirectUrl=%2Fproducts%2Fxenium-in-situ%2Fpreview-dataset-human-breast

The notebooks read `Xenium_FFPE_Human_Breast_Cancer_Rep1_cell_feature_matrix.h5` and `Xenium_FFPE_Human_Breast_Cancer_Rep1_cells.csv` from this dataset.

The notebooks also read three files that are not in the repository and that no notebook here creates:

- `clusters.csv`, the K-means clusters with K = 10;
- `cell_ID_toRemove_filtered_75_25.csv` and `cell_ID_toRemove_filtered_25_25.csv`, the cells removed for a low UMI count or a low number of expressed genes.

The notebooks call `os.chdir("")` and pass empty strings as output paths. Each empty string must be replaced with a local folder before a run.
