"""CeLEry quickstart: place single cells on a spatial transcriptomics map.

Trains the coordinate model (Fit_cord) on a mouse brain spatial
transcriptomics reference, checks it on reference spots held out from
training, and predicts the location of single cells profiled by scRNA-seq.
The data in examples/data are subsets of the tutorial toy data (see
make_quickstart_data.py). Run from the repository root:

    python examples/quickstart.py [output_folder]
"""
import os
import sys

import numpy as np
import pandas as pd
import scanpy as sc

import CeLEry as cel

DATA = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
EPOCHS = 50


def to_grid(pred, obs):
    # Fit_cord learns coordinates rescaled to (0, 1) with a margin of one grid
    # unit on each side (wrap_gene_location); undo that scaling for reporting.
    x, y = obs.iloc[:, 0].to_numpy(), obs.iloc[:, 1].to_numpy()
    xmin, xmax, ymin, ymax = x.min() - 1, x.max() + 1, y.min() - 1, y.max() + 1
    return np.column_stack([xmin + pred[:, 0] * (xmax - xmin), ymin + pred[:, 1] * (ymax - ymin)])


def main(out):
    ref = sc.read_h5ad(os.path.join(DATA, "MousePosteriorQuick.h5ad"))
    query = sc.read_h5ad(os.path.join(DATA, "MouseSCQuick.h5ad"))
    cel.get_zscore(ref)
    cel.get_zscore(query)

    held_out = np.random.RandomState(0).rand(ref.n_obs) < 0.2
    train, test = ref[~held_out].copy(), ref[held_out].copy()
    cel.Fit_cord(data_train=train, hidden_dims=[30, 25, 15], num_epochs_max=EPOCHS,
                 num_workers=0, path=out, filename="quickstart_cord")

    pred = to_grid(cel.Predict_cord(data_test=test, path=out, filename="quickstart_cord"), train.obs)
    truth = test.obs[["x", "y"]].to_numpy()
    error = np.linalg.norm(pred - truth, axis=1)
    centre = np.linalg.norm(truth - train.obs[["x", "y"]].to_numpy().mean(axis=0), axis=1)
    print("Held-out reference spots: median error %.1f grid units "
          "(%.1f if every spot were placed at the centre of the section)"
          % (np.median(error), np.median(centre)))

    cells = to_grid(cel.Predict_cord(data_test=query, path=out, filename="quickstart_cord"), train.obs)
    table = pd.DataFrame(cells, index=query.obs_names, columns=["x_pred", "y_pred"])
    table.to_csv(os.path.join(out, "quickstart_cell_coordinates.csv"))
    print("Predicted coordinates of %d cells written to %s" % (len(table), out))
    print(table.head())


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else os.path.join("examples", "output"))
