# Quick start

This page trains CeLEry on part of a mouse brain section profiled with 10x Visium and places 300 single cells on that section. It follows the steps of `examples/quickstart.py` and runs in about a minute on a laptop CPU. Run the code from the root of a clone of the repository, in the environment described in [Installation](install.md). In a script on Windows or macOS, put the code inside a main guard, as [Reading the output](reading-output.md#running-celery-in-a-script) explains.

## The data

The two files in `examples/data` are subsets of the tutorial data. The reference holds 1,000 super-pixels of a 10x Visium section of the mouse posterior brain, whose resolution was enhanced with TESLA, as in the paper; this page calls them spots. Each spot has the expression of 352 genes and its position in the columns `x` and `y` of `obs`, where one unit is 50 pixels of the histology image. The query holds 300 cells of a mouse brain scRNA-seq dataset, with the same 352 genes.

```python
import os

import numpy as np
import pandas as pd
import scanpy as sc

import CeLEry as cel

ref = sc.read_h5ad("examples/data/MousePosteriorQuick.h5ad")
query = sc.read_h5ad("examples/data/MouseSCQuick.h5ad")
print(ref.shape, query.shape)
print(ref.obs.head(3))
```

CeLEry uses the columns of the expression matrix by position, so the reference and the query must hold the same genes in the same order. The two files here already do. With your own data, subset both to their common genes first, as the [layer tutorial](tutorials/layers.md) shows.

## Standardize the expression

[get_zscore](reference.md#get_zscore) standardizes each gene to mean 0 and standard deviation 1. Each dataset is standardized with its own means and standard deviations, which puts the spots and the cells on a common scale.

```python
cel.get_zscore(ref)
cel.get_zscore(query)
```

## Hold out part of the reference

The query cells have no known location, so the quality of the predictions is checked on reference spots that the model does not see in training. One spot in five is held out.

```python
held_out = np.random.RandomState(0).rand(ref.n_obs) < 0.2
train, test = ref[~held_out].copy(), ref[held_out].copy()
print(train.n_obs, "spots for training,", test.n_obs, "held out")
```

## Train the model

[Fit_cord](reference.md#fit_cord) trains a neural network that predicts the coordinates of a spot from its expression. The coordinates are taken from the first two columns of `obs`. Here the training runs for 50 epochs, against 500 by default, and the trained model is saved as `output/quickstart/quickstart_cord.obj`.

```python
model = cel.Fit_cord(data_train=train, hidden_dims=[30, 25, 15], num_epochs_max=50,
                     num_workers=0, path="output/quickstart", filename="quickstart_cord")
```

## Check the predictions on the held-out spots

[Predict_cord](reference.md#predict_cord) returns coordinates rescaled to the interval (0, 1), which span the training coordinates plus a margin of one grid unit on each side. The function `to_grid` converts them back to grid units.

```python
def to_grid(pred, obs):
    x, y = obs["x"].to_numpy(), obs["y"].to_numpy()
    xmin, xmax, ymin, ymax = x.min() - 1, x.max() + 1, y.min() - 1, y.max() + 1
    return np.column_stack([xmin + pred[:, 0] * (xmax - xmin), ymin + pred[:, 1] * (ymax - ymin)])


pred = to_grid(cel.Predict_cord(data_test=test, path="output/quickstart", filename="quickstart_cord"),
               train.obs)
truth = test.obs[["x", "y"]].to_numpy()
error = np.linalg.norm(pred - truth, axis=1)
baseline = np.linalg.norm(truth - train.obs[["x", "y"]].to_numpy().mean(axis=0), axis=1)
print("median error %.1f grid units, against %.1f if every spot were placed at the center"
      % (np.median(error), np.median(baseline)))
```

The error is the distance between the predicted and the true position of each held-out spot. The baseline places every spot at the center of the training spots. In a run on Windows in October 2026, the median error was 3.8 grid units, against 30.3 for the baseline. The values change slightly with the platform and the PyTorch version.

## Place the single cells

The same model now predicts the location of each query cell. The table also keeps the cell subclass annotated in the scRNA-seq data, which shows whether cells of one subclass land in the same region.

```python
cells = to_grid(cel.Predict_cord(data_test=query, path="output/quickstart", filename="quickstart_cord"),
                train.obs)
table = pd.DataFrame(cells, index=query.obs_names, columns=["x_pred", "y_pred"])
table["subclass"] = query.obs["subclass_label"].to_numpy()
table.to_csv(os.path.join("output", "quickstart", "cell_coordinates.csv"))
print(table.head())
```

## Next steps

- The [cell coordinates tutorial](tutorials/coordinates.md) trains on the full section and adds a confidence region to each prediction.
- The [layer](tutorials/layers.md) and [domain](tutorials/domains.md) tutorials assign cells to tissue layers and spatial domains instead of coordinates.
- [Reading the output](reading-output.md) explains each output and how to run CeLEry in a script on Windows and macOS.
