# Guide for the mouse scRNA-seq prediction study

## Paper

Results subsection: "Application to recover 2D locations for scRNA-seq data in mouse brain".

Figures: none of the published figures.

Fig. 8 shows this application with two references, the MERSCOPE mouse brain data (sample S1R1) and the MERFISH mouse brain data of Zhang et al. (slice co2_slice80). The scripts in this folder use the 10x Visium mouse posterior brain data of `3_Mouse_10x_Visium`, enhanced with TESLA, as the reference instead. The code for Fig. 8 is not in the repository.

## Pipeline

preprocessing.py  ->
Mouse_sc_analysis.py (or analysis-results.py)  ->
Mouse_sc_analysis_spaOTsc_novosparc.ipynb

## What each file does

`preprocessing.py`:

- Its first section builds the scRNA-seq data from the Allen Brain Map files `../data/MouseBrain/reference/metadata.csv` and `../data/MouseBrain/reference/matrix.csv`. It keeps the cells from the regions RSP, VIS and ENT and the genes of `MP1_SVG.py`. It then normalizes and log-transforms the counts.
- No line saves this object. The next section and `Mouse_sc_analysis.py` read it from `../data/MouseBrain/MouseSC_scRNA.h5ad`.
- The later sections repeat the LIBD preprocessing of `2_Alzheimer/preprocess.py` with output in `../data/MouseSC/`. The analysis files of this folder do not read that output.

`Mouse_sc_analysis.py` builds the reference in the steps that `3_Mouse_10x_Visium/README.md` lists, without a hold-out. It trains the coordinate network for 500 epochs on all super-pixels. It writes the predicted coordinates of the cells to `../output/CeLEry/Prediction/PreOrg_Mousesc_predmatrix.csv`.

`analysis-results.py` repeats this analysis with a query processed in Seurat, `../data/Seurat/MouseSC_scRNA_SeuratMouseSC.h5ad`. It restricts the reference to the genes of this query. It writes `../output/CeLEry/Mousesc/PreOrg_Mousesc_predmatrix.csv`. It also converts a Tangram mapping, `../output/CeLEry/Mousesc/tangram.h5ad`, into `Tangram_Mousesc_predmatrix.csv`. The code that creates this mapping is commented out in the script.

`Mouse_sc_analysis_spaOTsc_novosparc.ipynb` reads the same query and takes the first 100 cells of each of seven subclasses (L2 IT ENTl, L2 IT RHP, L2/3 IT CTX-1, L2/3 IT CTX-2, L2/3 IT ENTl, L3 IT ENT and L3 RSP-ACA). It maps these 700 cells with spaOTsc and novoSpaRc. It writes `spaOTsc_Mousesc_predmatrix.csv` and `novosparc_Mousesc_predmatrix.csv` to `output/CeLEry/Mousesc/`.

All four methods give coordinates rescaled to the range 0 to 1 on each axis of the reference. No file in this folder draws the cell maps.

`Mouse_sc_analysis.py` and `analysis-results.py` train with `cel.SpaCluster`, which the current package does not provide. The training class of the current package is `TrainerExe`. `analysis-results.py` calls `cv.cvtColor`, but its line `import cv2 as cv` is commented out. `preprocessing.py` imports the modules `SpaCluster` and `SpaGCN`, which `environment.yml` does not install.

## Datasets

Query: Allen Brain Map, mouse whole cortex and hippocampus 10x data (Data availability statement of the paper):
https://portal.brain-map.org/atlases-and-data/rnaseq/mouse-whole-cortex-and-hippocampus-10x

Reference: the 10x Visium mouse posterior brain data, its TESLA output and its histology image, described in `3_Mouse_10x_Visium/README.md`.

The scripts import the gene list as `data.MouseBrain.MP1_SVG`. The repository holds it as `tutorial/data/Mouse2D/MP1_SVG.py`.

The Seurat analysis that produced `MouseSC_scRNA_SeuratMouseSC.h5ad` is not in the repository.
