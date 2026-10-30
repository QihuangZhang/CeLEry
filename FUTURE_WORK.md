# Future work

These changes were identified during the 2026 renovation and deliberately not made, because the renovation allowed no refactoring, no change in numerical behaviour and no new features (see `RENOVATION_LOG.md`). Each item states whether it would change results.

## Changes that would not alter results

- **pandas 3 in `get_Layer_LIBD`** (`util.py` lines 292-293). Assigning the rescaled coordinates as whole columns, for example `referlocation[referlocation.columns[0]] = ...`, gives the values that pandas 2 produced and would allow the pandas pins in `environment.yml` and in the package metadata to be lifted.
- **Invalid escape sequences** in two docstrings of `ClusterVAE.py` (lines 130 and 175). Python 3.12 warns about `\m`, and a future Python release will reject it, so these docstrings should become raw strings.
- **Pickle files left open** in `fit_functions.py`. Opening them with `with open(...)` removes the `ResourceWarning` raised at every save and load.
- **Package metadata** in `CeLEry_package/pyproject.toml`. CeLEry imports matplotlib without declaring it, and `requires-python = ">=3.8"` admits Python versions that the tested dependencies no longer support. In addition, setuptools 84.2.1 warns that the table form `license = { file = "LICENSE" }` and the licence classifier are deprecated and will no longer be supported after 2027-02-18. Since `pip install ./CeLEry_package` builds with the newest setuptools, the installation route in the README could then fail. The SPDX form `license = "MIT"` with `license-files = ["LICENSE"]`, which needs setuptools 77 or later, replaces both.
- **Output of `Fit_layer` and `Fit_domain`.** Both fail when the folder in `path` does not exist, whereas `Fit_cord` creates it. Both also return `None` instead of the trained model.
- **`truth_label` in `Predict_layer` and `Predict_domain`** (`fit_functions.py` lines 213-221 and 305-314). Passing a label raises `UnboundLocalError`, a kind of `NameError`, because `location_data` is defined only when `truth_label` is `None`.
- **Worker processes.** `Predict_*` and `DataAugmentation` hard-code `num_workers`, and the `Fit_*` functions default to 1. On Windows and macOS every data loader therefore starts a new Python process, which stretches the tutorial to hours. With `num_workers = 0`, `Fit_cord` produced bitwise identical weights.
- **Legacy data files.** The four h5ad files in `tutorial/data/` were written before anndata 0.8 and raise `OldFormatWarning`. Rewriting them with a current anndata keeps every value.
- **Default output path and file names.** With the default `path = ""`, `Fit_cord` and `Fit_region` raise `FileNotFoundError` after training, so the trained model is lost, and `Fit_layer` and `Fit_domain` write the model to the root folder of the drive. The default file names do not match between `Fit_layer` and `Predict_layer` (`PreOrg_layersc` and `PreOrg_layernsc`) or between `Fit_region` and `Predict_region` (`PreOrg_Mousesc` and `PreOrg_ConfScore`).
- **Input checks.** Without `location_data`, `Fit_cord` and `Fit_region` convert the whole of `data_train.obs` to numbers and fail on any text column. `AugFit_domain` needs the class weights as a torch tensor, whereas `Fit_domain` needs a pandas Series. With NaN in the query, `Predict_domain` assigns every cell to the last domain without an error, and `Predict_layer` stops with an unrelated `ValueError`. After `make_annData_spatial`, the first column of `obs` is `select`, so `Fit_cord` and `Fit_region` read `select` and `x_cord` as the coordinates unless `location_data` is given. `AugFit_domain` and `DataAugmentation` have different default paths, so with both defaults `AugFit_domain` cannot find the replicates. Checks that name these causes would leave valid results unchanged.
- **Overlay in `Predict_region`** (`util_Mouse.py` line 306). `if hist != "":` raises `ValueError` when `hist` is an image array, and `UncertaintyPlot` multiplies the coordinates by a fixed 50. The function also discards the center and the semi-axes of each ellipse, which users could use.

## Changes that would alter results, each needing a decision

- **Evaluation mode at prediction.** Calling `model.eval()` in `Predict_layer` and `Predict_domain`, and for the pretrained Biogen model, which loads in training mode, would switch off dropout and make every prediction deterministic.
- **Sorted gene lists.** Replacing `list(set(...))` with `sorted(set(...))` in `get_common_genes` (`util.py` line 444) and in the tutorial removes the dependence on `PYTHONHASHSEED`.
- **`KMeans(n_init=10)`** in `DataAugmentation` (`data_augmentation.py` line 39) restores the clustering of scikit-learn releases before 1.4.
- **`data_range` in `get_SSIM`** (`util_Mouse.py` line 63). The function fails without it, and the chosen range determines the value of the index.
- **Coordinates on the original scale.** `Predict_cord` returns coordinates rescaled to (0, 1) separately for each axis. `examples/quickstart.py` maps them back for its report, and the package could do the same.
- **Learning rate.** `TrainerExe.train` silently caps the learning rate at 1e-2, whereas the tutorial suggests 0.1.
- **Network and label handling.** A `hidden_dims` list with four widths silently builds three hidden layers, and `AugFit_domain` assigns labels correctly only when the input is sorted by its grid coordinates.
- **Scales of `get_Layer_LIBD` and `Predict_region`.** `get_Layer_LIBD` maps the reference to `(x - x_min) / (x_max - x_min + 1)`, whereas `Fit_cord` trains on `(x - x_min + 1) / (x_max - x_min + 2)`, so a prediction can be matched to the neighbor of the closest spot (`util.py` lines 289-292). `Predict_region` rescales `location_data` with its own range instead of that of the training spots, so its coverage is exact only when both ranges agree.

## Documentation and packaging

- **`tutorial/BiogenPretrain.md` line 76** applies `.A` to a dense matrix, and removing `.A` would let the tutorial continue.
- **`code_paper/`** calls an older interface (`cel.SpaCluster`, `cel.DNNordinal_v1`), and `1_LIBD/CeLEry_train_Scenario3and4.py` does not parse (line 320).
- **Task 3 of `tutorial/tutorial.md`** skips `drop_NaN`. Fourteen reference genes do not vary in the query, so the network returns NaN for every cell, `Predict_domain` turns that into a probability of 1.0 for every domain, and every cell is assigned to the last domain, as the printed output of the tutorial shows. The spatial domains page of the documentation site adds the step.
- **Data of `code_paper/2_Alzheimer`.** The Box folder that held the AD snRNA-seq query data returns 404 (checked on 2026-10-29).
- **Lock files for each platform**, for example with conda-lock, which would also pin indirect dependencies.
