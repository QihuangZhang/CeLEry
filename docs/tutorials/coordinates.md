# Cell coordinates

This tutorial trains CeLEry on a spatial transcriptomics section of the mouse brain, predicts the 2D location of single cells from scRNA-seq data, and gives each prediction a confidence score.

Run the code in an interactive Python session, such as Jupyter, started at the root of a clone of the repository, with CeLEry installed as described in [Installation](../install.md). To run it as a script on Windows or macOS, follow [Running CeLEry in a script](../reading-output.md#running-celery-in-a-script).

## Data

The two input files are in `tutorial/data/`:

- `MousePosteriorToy.h5ad` is the reference: 5,824 super-pixels of a 10x Visium section of the mouse posterior brain, whose resolution was enhanced with TESLA, as in the paper, with 356 genes. This page calls the super-pixels spots. The columns `x` and `y` of its `.obs` hold the position of each spot, where one unit is 50 pixels of the histology image.
- `MouseSCToy.h5ad` is the query, 3,000 cells of a mouse brain scRNA-seq dataset with the same 356 genes.

The network reads the genes by their position in the expression matrix. The query must therefore contain the genes of the reference in the same order, as these two files do.

## Load and standardize the data

```python
import scanpy as sc
import CeLEry as cel

import warnings
warnings.filterwarnings("ignore")
```

```python
#Read in gene expression and spatial location
Qdata = sc.read("tutorial/data/MouseSCToy.h5ad")
Rdata = sc.read("tutorial/data/MousePosteriorToy.h5ad")

Rdata
```

```text
AnnData object with n_obs × n_vars = 5824 × 356
    obs: 'x', 'y', 'inner'
    var: 'genename'
```

`Qdata` holds the query. `Rdata` holds the reference.

```python
cel.get_zscore(Qdata)
cel.get_zscore(Rdata)
```

[get_zscore](../reference.md#get_zscore) standardizes each gene to mean 0 and standard deviation 1, separately within each dataset. It changes the AnnData object in place and leaves `.X` as a dense array of these z-scores.

## Train the coordinate model

```python
model_train = cel.Fit_cord (data_train = Rdata, hidden_dims = [30, 25, 15], num_epochs_max = 500, path = "output/tutorial", filename = "Org_Mousesc")
```

[Fit_cord](../reference.md#fit_cord) trains a neural network that predicts the location of a spot from its expression. Its arguments are:

- `data_train`: the reference. Without `location_data`, `Fit_cord` takes the coordinates from the first two columns of `data_train.obs`, here `x` and `y`. It converts that whole table to floating point, so every column of `data_train.obs` must hold numbers. Otherwise, pass a table with the coordinates in its first two columns as `location_data`.
- `hidden_dims`: the number of nodes in each of the three hidden layers (default `[30, 25, 15]`).
- `num_epochs_max`: the maximum number of epochs, or passes over the training data (default 500).
- `path`: the folder of the model file, which `Fit_cord` creates if it does not exist. Always set it, because the default empty string does not name a folder.
- `filename`: the name of the model file without its extension.

`Fit_cord` returns the trained network as a PyTorch module. It also saves the network with pickle to `output/tutorial/Org_Mousesc.obj`. Training for 500 epochs can take an hour or more. `Fit_cord` seeds Python, NumPy and PyTorch with `seednum` (default 2021). A repeated run with the same package versions on the same platform therefore gives the same network. [Tuning the training](#tuning-the-training) describes the settings that control the optimization.

## Predict the location of each cell

```python
pred_cord = cel.Predict_cord (data_test = Qdata, path = "output/tutorial", filename = "Org_Mousesc")

pred_cord
```

[Predict_cord](../reference.md#predict_cord) loads `output/tutorial/Org_Mousesc.obj` and applies the network to each cell of `Qdata`. It returns a NumPy array with one row per cell, with the predicted x in the first column and the predicted y in the second. It adds the same values to `Qdata.obs` as the columns `x_cord_pred` and `y_cord_pred`. It also writes them, without a header, to `output/tutorial/Org_Mousesc_predmatrix.csv`.

The predictions are on the scale on which the network was trained, not in the units of the reference. `Fit_cord` rescales each axis of the reference coordinates separately to the interval (0, 1), with a margin of one unit beyond the smallest and the largest value. A predicted value `p` on the first axis corresponds to `x = (x_min - 1) + p * (x_max - x_min + 2)` in the units of the reference, where `x_min` and `x_max` are the smallest and largest x of the reference spots. The second axis converts in the same way. The function `to_grid` of the [quick start](../get-started.md#check-the-predictions-on-the-held-out-spots) applies this conversion. Since the two axes are rescaled separately, distances on the (0, 1) scale do not keep the aspect ratio of the section. [Reading the output](../reading-output.md#coordinates) gives more detail.

```python
Qdata.obs
```

The last two columns of `Qdata.obs` now hold the predicted coordinates, `x_cord_pred` and `y_cord_pred`.

## Confidence score

[Fit_region](../reference.md#fit_region) trains a second network that predicts, for each spot, a location and an ellipse centered on it. The axes of the ellipse are parallel to the coordinate axes. Training aims for ellipses that contain the true location of a spot with probability `alpha` (default 0.95). `Fit_region` takes the same arguments as `Fit_cord`, plus `alpha`.

```python
model_train = cel.Fit_region (data_train = Rdata, alpha = 0.95, hidden_dims = [30, 25, 15], num_epochs_max = 500, path = "output/example", filename = "ConfRegion_Mousesc")

model_train
```

`Fit_region` returns the trained network, which `model_train` displays layer by layer. It saves the network to `output/example/ConfRegion_Mousesc.obj`, creating the folder if needed.

```python
pred_region = cel.Predict_region (data_test = Qdata, path = "output/example", filename = "ConfRegion_Mousesc")

Qdata.obs
```

[Predict_region](../reference.md#predict_region) applies this network to each cell and adds two columns to `Qdata.obs`:

- `area_record`: the product of the two semi-axes of the ellipse, on the (0, 1) scale of the coordinates. The area of the ellipse is π times this product.
- `conf_score`: the confidence score, equal to 1 minus `area_record`. A higher score means a smaller ellipse and a less uncertain prediction.

The value stored in `pred_region` is the proportion of cells whose ellipse contains their location in the argument `location_data`. Without `location_data`, `Predict_region` gives every cell the same placeholder location, so this proportion has no meaning for query cells. [Reading the output](../reading-output.md#confidence-regions) explains when it measures the coverage of the ellipses on held-out reference spots. `Predict_region` writes no file here, because it draws the areas only when `.obs` has the columns `x_cord` and `y_cord`.

## Tuning the training

The four training functions, `Fit_cord`, `Fit_region`, [Fit_layer](../reference.md#fit_layer) and [Fit_domain](../reference.md#fit_domain), share the settings below. Each epoch prints its total loss, which is the sum of the losses of its batches.

- `batch_size` (default 4): the number of spots in each gradient step. If the printed loss becomes `nan`, a sign of numerical instability, try another batch size, such as `batch_size = 8`.
- `initial_learning_rate` (default 1e-4): the learning rate of the Adam optimizer at the start of training. Values above 0.01 are reduced to 0.01. A larger rate, such as `initial_learning_rate = 0.01`, makes larger updates to the network early in training.
- `number_error_try` (default 15): the number of poor epochs after which the learning rate is halved. An epoch is poor when its total loss exceeds the lowest total loss so far. Poor epochs need not be consecutive. After each halving, the lowest loss is raised by 10 and the count restarts. CeLEry prints each new learning rate. Training ends before `num_epochs_max` epochs if the learning rate falls below 1e-7. A run with more epochs can use a larger value, such as `number_error_try = 60`, so that the learning rate is halved less often.
- `num_workers` (default 1): the number of processes that load the training data. On Windows and macOS, the default starts a new process in every epoch, which makes training much slower; `num_workers = 0` loads the data in the main process and gave bitwise identical weights in the tests of October 2026. [Reading the output](../reading-output.md#settings-that-matter) gives the measured times.
