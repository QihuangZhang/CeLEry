"""Smoke tests: the package imports, the quickstart runs, Predict_cord returns
the documented output, and Fit_cord is reproducible under a fixed seed."""
import os
import subprocess
import sys

import numpy as np
import pandas as pd
import scanpy as sc
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(ROOT, "examples", "data")


def fit_small(path, seed=2021):
    import CeLEry as cel
    ref = sc.read_h5ad(os.path.join(DATA, "MousePosteriorQuick.h5ad"))
    cel.get_zscore(ref)
    return cel.Fit_cord(data_train=ref, hidden_dims=[30, 25, 15], num_epochs_max=2, num_workers=0,
                        path=str(path), filename="smoke", seednum=seed)


def test_import():
    import CeLEry
    assert callable(CeLEry.Fit_cord) and callable(CeLEry.Predict_cord)


def test_quickstart_runs(tmp_path):
    run = subprocess.run([sys.executable, os.path.join("examples", "quickstart.py"), str(tmp_path)],
                         cwd=ROOT, capture_output=True, text=True)
    assert run.returncode == 0, run.stderr[-3000:]
    table = pd.read_csv(tmp_path / "quickstart_cell_coordinates.csv", index_col=0)
    assert table.shape == (300, 2)


def test_predict_cord_shape_and_dtype(tmp_path):
    import CeLEry as cel
    fit_small(tmp_path)
    query = sc.read_h5ad(os.path.join(DATA, "MouseSCQuick.h5ad"))
    cel.get_zscore(query)
    pred = cel.Predict_cord(data_test=query, path=str(tmp_path), filename="smoke")
    assert pred.shape == (query.n_obs, 2)
    assert pred.dtype == np.float64
    assert np.isfinite(pred).all()


def test_fit_cord_same_seed_is_bitwise_identical(tmp_path):
    first = fit_small(tmp_path / "first").state_dict()
    second = fit_small(tmp_path / "second").state_dict()
    assert first.keys() == second.keys()
    for name in first:
        assert torch.equal(first[name], second[name]), name
