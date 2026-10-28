"""Build the quickstart data by subsampling the tutorial toy data.

The reference is a random subset of the spots in
tutorial/data/MousePosteriorToy.h5ad and the query a random subset of the
cells in tutorial/data/MouseSCToy.h5ad. Genes that do not vary within either
subset are dropped, because get_zscore cannot standardise a constant gene.
No expression value or annotation is altered.

Run once from the repository root:

    python examples/make_quickstart_data.py
"""
import os

import anndata as ad
import numpy as np

N_SPOTS, N_CELLS, SEED = 1000, 300, 2026
OUT = os.path.join("examples", "data")

if __name__ == "__main__":
    rng = np.random.RandomState(SEED)
    ref = ad.read_h5ad("tutorial/data/MousePosteriorToy.h5ad")
    query = ad.read_h5ad("tutorial/data/MouseSCToy.h5ad")
    assert list(ref.var_names) == list(query.var_names)
    ref = ref[np.sort(rng.choice(ref.n_obs, N_SPOTS, replace=False))]
    query = query[np.sort(rng.choice(query.n_obs, N_CELLS, replace=False))]
    varies = (ref.X.std(axis=0) > 0) & (query.X.std(axis=0) > 0)
    ref = ref[:, varies].copy()
    query = query[:, varies].copy()
    os.makedirs(OUT, exist_ok=True)
    ref.write_h5ad(os.path.join(OUT, "MousePosteriorQuick.h5ad"), compression="gzip")
    query.write_h5ad(os.path.join(OUT, "MouseSCQuick.h5ad"), compression="gzip")
    print("reference", ref.shape, "query", query.shape)
