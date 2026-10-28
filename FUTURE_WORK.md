# Future work

These changes were identified during the 2026 renovation and deliberately not made, because the renovation allowed no refactoring, no change in numerical behaviour and no new features (see `RENOVATION_LOG.md`). Each item states whether it would change results.

## Changes that would not alter results

- **pandas 3 in `get_Layer_LIBD`** (`util.py` lines 292-293). Assigning the rescaled coordinates as whole columns, for example `referlocation[referlocation.columns[0]] = ...`, gives the values that pandas 2 produced and would allow the pandas pin to be lifted.
- **Invalid escape sequences** in two docstrings of `ClusterVAE.py` (lines 130 and 175). Python 3.12 warns about `\m`, and a future Python release will reject it, so these docstrings should become raw strings.
- **Pickle files left open** in `fit_functions.py`. Opening them with `with open(...)` removes the `ResourceWarning` raised at every save and load.
- **Package metadata** in `CeLEry_package/pyproject.toml`. CeLEry imports matplotlib without declaring it, and `requires-python = ">=3.8"` admits Python versions that the tested dependencies no longer support.
- **Output of `Fit_layer` and `Fit_domain`.** Both fail when the folder in `path` does not exist, whereas `Fit_cord` creates it. Both also return `None` instead of the trained model.
- **`truth_label` in `Predict_layer` and `Predict_domain`** (`fit_functions.py` lines 213-221 and 305-314). Passing a label raises `NameError`, because `location_data` is defined only when `truth_label` is `None`.
- **Worker processes.** `Predict_*` and `DataAugmentation` hard-code `num_workers`, and the `Fit_*` functions default to 1. On Windows and macOS every data loader therefore starts a new Python process, which stretches the tutorial to hours. With `num_workers = 0`, `Fit_cord` produced bitwise identical weights.
- **Legacy data files.** The four h5ad files in `tutorial/data/` were written before anndata 0.8 and raise `OldFormatWarning`. Rewriting them with a current anndata keeps every value.

## Changes that would alter results, each needing a decision

- **Evaluation mode at prediction.** Calling `model.eval()` in `Predict_layer` and `Predict_domain`, and for the pretrained Biogen model, which loads in training mode, would switch off dropout and make every prediction deterministic.
- **Sorted gene lists.** Replacing `list(set(...))` with `sorted(set(...))` in `get_common_genes` (`util.py` line 444) and in the tutorial removes the dependence on `PYTHONHASHSEED`.
- **`KMeans(n_init=10)`** in `DataAugmentation` (`data_augmentation.py` line 39) restores the clustering of scikit-learn releases before 1.4.
- **`data_range` in `get_SSIM`** (`util_Mouse.py` line 63). The function fails without it, and the chosen range determines the value of the index.
- **Coordinates on the original scale.** `Predict_cord` returns coordinates rescaled to (0, 1) separately for each axis. `examples/quickstart.py` maps them back for its report, and the package could do the same.
- **Learning rate.** `TrainerExe.train` silently caps the learning rate at 1e-2, whereas the tutorial suggests 0.1. The default `initial_learning_rate` also differs between release 1.2.1 (1e-4) and `main` (1e-3).
- **Network and label handling.** A `hidden_dims` list with four widths silently builds three hidden layers, and `AugFit_domain` assigns labels correctly only when the input is sorted by domain.

## Documentation and packaging

- **`tutorial/BiogenPretrain.md` line 76** applies `.A` to a dense matrix, and removing `.A` would let the tutorial continue.
- **`code_paper/`** calls an older interface (`cel.SpaCluster`, `cel.DNNordinal_v1`), and `1_LIBD/CeLEry_train_Scenario3and4.py` does not parse (line 320).
- **A new PyPI release**, since release 1.2.1 lacks the repairs for sparse input.
- **A `CITATION.cff` file**, following DenMark.
- **Lock files for each platform**, for example with conda-lock, which would also pin indirect dependencies.
- **The original README sections** on system requirements and installation, which the new header supersedes and which could be merged into it.
