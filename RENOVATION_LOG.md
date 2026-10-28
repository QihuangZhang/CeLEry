# Renovation log

This log records the 2026 renovation of CeLEry. The goal was a package that a new user can install and run in 2026, with results that still follow the published method (Zhang et al., *Nature Communications* 14, 4050, 2023). The work followed four constraints:

- no refactoring, renaming or reorganisation of files;
- no change in numerical behaviour, so that a dependency upgrade which changes results is reported rather than absorbed;
- no new features;
- no deletion except files shown to be unreferenced, each approved before removal.

All changes are local commits on the branch `renovation-2026`, which starts from `main` at `b0b7376` (2026-10-28). Ideas that would have needed a refactor, a numerical change or a new feature are recorded in `FUTURE_WORK.md`.

## Diagnosis

The README route (`pip install CeLEryPy`, then `tutorial/tutorial.md`) was tested on 2026-10-27 in fresh virtual environments with Python 3.11.9 and 3.12.15 on Windows 11. `pip install CeLEryPy` took 341 s and 420 s, respectively, and `import CeLEry` succeeded on both without any change to the code. The tutorial then stopped at the breaks below, listed in the order a new user meets them.

| # | Symptom | Location | Cause | Outcome |
|---|---|---|---|---|
| 1 | `FileNotFoundError: 'data/tutorial/MouseSCToy.h5ad'` | `tutorial/tutorial.md`, lines 139-140 and 942-943 | The tutorial points to a folder that does not exist (the files are in `tutorial/data/`) | Fixed in `6c915ca` |
| 2 | `AttributeError: 'csr_matrix' object has no attribute 'A'` in `get_zscore` | `util.py` line 200 of the PyPI release 1.2.1 | SciPy 1.14 removed the `.A` attribute of sparse matrices | Fixed on `main` by `4ca8f71` (February 2025) but never released |
| 3 | The same error in every `Fit_*` and `Predict_*` function and in `getGeneImg` whenever `adata.X` is sparse | `fit_functions.py` lines 41, 77, 110, 138, 155, 195, 218 and 311, and `util.py` line 119 | SciPy 1.14, as in break 2 | Fixed in `d013dcb` |
| 4 | `TypeError: Invalid value ... for dtype 'int64'` in `get_Layer_LIBD` | `util.py` line 292 | pandas 3.0 raises an error where pandas 2 converted the integer column to floating point | Avoided by pinning pandas 2.3.3, with the code unchanged |
| 5 | `ValueError: ... you must specify the data_range parameter` in `get_SSIM` | `util_Mouse.py` line 63 | scikit-image requires `data_range` for floating-point images, and the call already failed with scikit-image 0.23.1 in April 2024 | Unchanged, see `FUTURE_WORK.md` |
| 6 | `AttributeError: 'ArrayView' object has no attribute 'A'` | `tutorial/BiogenPretrain.md` line 76 | The tutorial applies `.A` to a dense matrix, which failed with the April 2024 dependencies as well | Unchanged, see `FUTURE_WORK.md` |

To attribute each break, the same checks were repeated with the dependency versions that `pip` would have installed on 2024-04-11, the day after CeLEry 1.2.1 was published. Breaks 2 and 3 do not occur there, because SciPy 1.13.0 still provides `.A`, and the unmodified 1.2.1 tutorial runs to the end once the paths of break 1 are corrected. Breaks 5 and 6 occur there as well, so they predate the release.

Three further findings change results without raising an error.

- Since scikit-learn 1.4, `KMeans` runs a single initialisation by default (`n_init='auto'`) instead of ten. `DataAugmentation` relies on that default (`data_augmentation.py` line 39), and on the 356 toy gene images the default clustering agreed with a ten-initialisation clustering at an adjusted Rand index of 0.44. The 2024 baseline already had the new default.
- `Predict_layer` and `Predict_domain` keep dropout active at prediction (`DNN.py` lines 210 and 406), since no model is switched to evaluation mode. Two identical calls returned layer probabilities that differed by up to 0.245.
- Gene lists built as `list(set(...))`, in the tutorial and in `get_common_genes` (`util.py` line 444), change order between Python sessions because of hash randomisation, and the trained model changes with the gene order.

Training is slow on Windows, and macOS should behave alike because Python also spawns worker processes there instead of forking them (not tested). Every `Fit_*` function defaults to `num_workers = 1`, which starts a new worker process in every epoch on such systems, so the 500 epochs of the tutorial would take 1.3 to 2.6 hours per task on the test machine. With `num_workers = 0` the trained weights are bitwise identical (tested), and the quickstart, which trains on 800 spots, needs about one second per epoch.

The PyPI release 1.2.1 (2024-04-10) also differs from `main` in one default. Commit `3befc6d` raised the default `initial_learning_rate` of all four `Fit_*` functions from 1e-4 to 1e-3 after the release. After review, the maintainer chose to restore 1e-4 (see Changes).

## Changes

| Commit | Change | Reason | Effect on results |
|---|---|---|---|
| `f0cd80f` | Removed `CeLEry/layers.py` | The file does not compile (`def __init__(self, , )`, line 18), and no file imports it | None |
| `2c628c7` | Removed the tracked `__pycache__` files (21), `CeLEryPy.egg-info/` and `dist/` | These are build output that no file references, and the 1.2.1 files remain on PyPI and in the history | None |
| `28bd465` | Corrected the copyright holder in `CeLEry_package/LICENSE` | The packaged licence carried the template line "The Python Packaging Authority" instead of Qihuang Zhang, the holder named in the root `LICENSE` | None |
| `6c915ca` | Corrected four data paths in `tutorial/tutorial.md` | Break 1 | None |
| `d013dcb` | Replaced `.A` with `.toarray()` at nine sites | Break 3. SciPy 1.13 implements `.A` as `return self.toarray()`, so the values are identical | None (tested, see below) |
| `7308987` | Added `.gitignore` | Building and testing the package recreate the removed build output | None |
| `476d73a` | Added `environment.yml` | Exact, tested versions of every direct dependency | See the comparison below |
| `81851bc` | Added `examples/quickstart.py` and its data | A complete example that runs in about a minute on a CPU | None |
| `b6862ce` | Added `tests/test_smoke.py` | Import, quickstart, output shape and type, and seed control | None |
| `1a03fc3` | Added `.github/workflows/tests.yml` | Runs the smoke tests on every push and pull request | None |

This log and `FUTURE_WORK.md` form one further commit, and the new README header states the tested versions and the installation route.

After the maintainer's review on 2026-10-28, the default `initial_learning_rate` of `Fit_cord`, `Fit_region`, `Fit_layer` and `Fit_domain` (`fit_functions.py` lines 30, 66, 98 and 126) returned to 1e-4. This is the value of release 1.2.1 and of `AugFit_domain`, and it reverses the change of default in `3befc6d`. Results therefore change relative to `main`. Since the loss decreases more slowly at the smaller rate (0.118 against 0.075 after 50 epochs), the median held-out error of the quickstart rose from 3.1 to 3.8 grid units. The version number and the package dependencies were then prepared for release 1.2.2 (see Release 1.2.2).

The quickstart data are random subsets of the tutorial files, namely 1,000 of the 5,824 spots of `MousePosteriorToy.h5ad` and 300 of the 3,000 cells of `MouseSCToy.h5ad`, written by `examples/make_quickstart_data.py` with seed 2026. Four genes that do not vary within the 300 cells were dropped from both files, because `get_zscore` cannot standardise a constant gene, which leaves 352 genes. Every value in the subsets equals the corresponding value in the tutorial files (checked with anndata 0.10.7).

## Version pins

`environment.yml` pins every direct dependency to the version that was tested. CeLEry itself is installed from the repository with `pip install --no-deps ./CeLEry_package`, so that `pip` cannot replace a pinned package.

| Package | Pin | Reason |
|---|---|---|
| python | 3.12.15 | Python 3.12 receives security fixes until October 2028, and 3.12.15 is the interpreter used in all Python 3.12 tests |
| pytorch | 2.14.1, CPU build | The newest release. CeLEry contains no GPU code, so the CPU build is used on every platform and no separate GPU environment is needed |
| numpy | 2.5.4 | Newest release |
| scipy | 1.18.1 | Newest release |
| pandas | 2.3.3 | The last 2.x release, because pandas 3 breaks `get_Layer_LIBD` (break 4) |
| anndata | 0.13.4 | Newest release. It reads the legacy tutorial files with an `OldFormatWarning` |
| scanpy | 1.12.4 | Newest release. `normalize_per_cell`, used by `make_annData_query`, still works |
| scikit-learn | 1.9.1 | Newest release |
| scikit-image | 0.26.0 | Newest release |
| matplotlib-base | 3.11.2 | Newest release. CeLEry imports matplotlib although `pyproject.toml` does not declare it |
| tqdm | 4.70.1 | Newest release |
| pytest | 9.1.1 | Runs the smoke tests |

The pins solve on linux-64, osx-64, osx-arm64 and win-64. Only direct dependencies are pinned, so the solver may select newer builds of indirect dependencies in the future.

On Windows, micromamba 2.10 failed to build the environment from a root folder with a long path ("The directory is not empty"). Since micromamba 2 stores each package under the channel URL, the cached files of PyTorch and of its dependency scikit-build-core exceeded the 260-character Windows limit. A short root folder (`C:\Users\<user>\mm`) avoided the error. Enabling long paths in Windows should avoid it as well, but this was not tested.

## Verification

| Environment | Python | torch | numpy | scipy | pandas | anndata | scanpy | scikit-learn | scikit-image |
|---|---|---|---|---|---|---|---|---|---|
| 2024 baseline: PyPI as of 2024-04-11 (uv `--exclude-newer`), CeLEryPy 1.2.1 | 3.11.9 | 2.2.2 | 1.26.4 | 1.13.0 | 2.2.2 | 0.10.7 | 1.10.1 | 1.4.2 | 0.23.1 |
| 2026 `pip install CeLEryPy`, Python 3.11 | 3.11.9 | 2.14.1 | 2.4.6 | 1.17.1 | 2.3.3 | 0.12.19 | 1.11.5 | 1.9.1 | 0.26.0 |
| 2026 `pip install CeLEryPy`, Python 3.12 | 3.12.15 | 2.14.1 | 2.5.4 | 1.18.1 | 3.1.0 | 0.13.4 | 1.12.4 | 1.9.1 | 0.26.0 |
| Pinned (`environment.yml`, micromamba 2.10.0, conda-forge) | 3.12.15 | 2.14.1 | 2.5.4 | 1.18.1 | 2.3.3 | 0.13.4 | 1.12.4 | 1.9.1 | 0.26.0 |
| Control: 2024 baseline with torch 2.14.1 | 3.11.9 | 2.14.1 | 1.26.4 | 1.13.0 | 2.2.2 | 0.10.7 | 1.10.1 | 1.4.2 | 0.23.1 |

All runs used a Windows 11 PC (15.8 GB RAM, 12 PyTorch threads) unless marked Linux, which means Ubuntu under WSL2 (kernel 6.6.87). The checks below used the pinned environment and the repaired code. The quickstart, the smoke tests and the tutorial were repeated on Windows after the default learning rate returned to 1e-4, and the table shows those results. The other checks, including those on Linux, ran with the earlier default of 1e-3 and were not repeated.

| Check | Result |
|---|---|
| Environment build | Built on Windows in 190 s from downloaded packages, and on Linux |
| `import CeLEry` | Succeeds, 26 s for the first import on Windows |
| `python examples/quickstart.py` | Windows 57 s, Linux under 15 s. The median error on the 200 held-out spots is 3.8 grid units on Windows, against 30.3 when every spot is placed at the centre of the section. With the earlier default it was 3.1 on Windows and 3.2 on Linux |
| `python -m pytest -q tests` | 4 passed, in 70 s on Windows and 15 s on Linux. Linux shows one Python 3.12 warning that `fork()` is called in a multi-threaded process when a `DataLoader` starts its worker |
| `tutorial/tutorial.md`, all code blocks of tasks 1 to 3 with 3 epochs per model | All 24 steps succeed in 246 s. The only warnings are `OldFormatWarning` for the legacy data files, `ResourceWarning` for pickle files left open, and `ImplicitModificationWarning` where `Predict_layer`, `Predict_domain` and one tutorial line write to the `.obs` of an AnnData view |
| Targeted checks | Sparse input to `Fit_cord` and `Predict_cord`, loading and running the pretrained Biogen model (`pretrainmodel/Biogen`), `make_annData_query`, `prefilter_cells`, `get_Layer_LIBD` and `plot_confusion_matrix` all succeed. `get_SSIM` and line 76 of `BiogenPretrain.md` still fail (breaks 5 and 6) |
| `DataAugmentation` | Clustering finishes and the variational autoencoder trains (epoch 32 of 249 after 240 s, stopped by design). A complete run was not verified |
| GitHub Actions | Not verified, because the workflow first runs after the branch is pushed |
| macOS | Not verified. The environment solves for osx-64 and osx-arm64 but was not built |

## Comparison with the 2024 baseline

To test whether the renovation changes results, one fixed-seed script trained all four model types and saved every output in three settings. The baseline ran the unmodified code of `main` in the 2024 environment. The candidate ran the repaired code in the pinned environment. The control ran the unmodified code in the 2024 environment with torch 2.14.1 as the only change. All three read the same input files, which were written once with anndata 0.10.7. `Fit_cord` and `Fit_region` were trained for 20 and 10 epochs on 800 of the quickstart spots. `Fit_layer` and `Fit_domain` were trained for 10 epochs on 600 spots of `DataLayerToy.h5ad` and predicted 200 cells of `AlzheimerToy.h5ad`, whose expression matrix is sparse. Since dropout is active at prediction, `torch.manual_seed(0)` was called before each `Predict_layer` and `Predict_domain` call. The comparison ran before the default learning rate returned to 1e-4, so all three settings trained at 1e-3.

| Output | Baseline vs candidate | Control vs candidate |
|---|---|---|
| `get_zscore`, dense and sparse input | Bitwise identical | Bitwise identical |
| `Fit_region` | Losses identical for all 10 epochs. Weights differ by at most 1.8e-6 and confidence areas by 7.5e-7 | Bitwise identical |
| `Fit_cord` | Losses identical to four decimals for 9 epochs, then drifting. After 20 epochs, weights differ by up to 0.050 and predicted coordinates by up to 8.1 grid units. The median held-out error is 3.82 and 3.77 grid units, respectively | Bitwise identical |
| `Fit_layer` | First-epoch losses 452.28 and 453.48. Weights differ by up to 0.18, and the predicted layers agree for 93.5% of cells | Bitwise identical |
| `Fit_domain` | First-epoch losses 253.14 and 255.61. Weights differ by up to 0.26, and the predicted domains agree for 64.0% of cells | Probabilities differ by at most 2.4e-7, and all predicted domains agree |

The control shows that the upgrade from torch 2.2.2 to 2.14.1 explains every difference between the baseline and the candidate. The `.toarray()` repair and the upgrades of Python, NumPy, SciPy, pandas, anndata and scanpy change nothing measurable in these outputs. A second control confirmed this, since the pip-installed torch 2.14.1 on Python 3.11 with numpy 2.4.6, anndata 0.12.19 and scanpy 1.11.5 reproduced the candidate bitwise. The comparison does not exercise scikit-learn or scikit-image, whose `KMeans` default is discussed under Diagnosis.

Within torch, weight initialisation, the shuffling order of the data loader, `torch.randn` and a test matrix product are bitwise identical between 2.2.2 and 2.14.1. The dropout masks drawn for the same seed differ, which explains why `Fit_layer` and `Fit_domain`, whose networks use dropout during training, differ from the first epoch. `Fit_cord` has no dropout, and its losses separate only after nine epochs, which suggests rounding-level differences in other torch operations that training amplifies. That operation was not isolated.

To judge whether these differences matter, the candidate was rerun with `seednum = 2022` instead of the default 2021. Changing the seed moved predicted coordinates by up to 15.6 grid units (against 8.1 for the torch upgrade) and raised the median held-out error to 3.94 grid units. The predicted layers agreed for 93.5% of cells and the predicted domains for 62.5%, against 93.5% and 64.0% for the torch upgrade. The loss of the domain model had not settled by epoch 10 (it moved between 4.4 and 17.4 over the last four epochs), and the low agreement of its predictions in both comparisons reflects this unconverged model. The torch upgrade therefore changes individual predictions by about as much as a different random seed does, and it leaves the accuracy of the coordinate model unchanged.

`DataAugmentation` was not compared, because its 449 hard-coded epochs exceed the time available.

## Release 1.2.2

Release 1.2.2 packages the repaired code for PyPI. A file-by-file comparison with release 1.2.1, as installed from PyPI, found three differences in the code.

- `.toarray()` replaces `.A` at ten sites, namely in `get_zscore` (`4ca8f71`) and at the nine sites of `d013dcb`. This repairs sparse input with SciPy 1.14 or later and leaves the values unchanged.
- `layers.py` is no longer included.
- `__version__` is 1.2.2.

The default learning rate is 1e-4 in both releases. The metadata of release 1.2.2 also names the correct copyright holder (`28bd465`) and requires pandas below 3, as `environment.yml` does (break 4). The wheel and the source distribution were built with build 1.6.1 and setuptools 84.2.1 from the tracked files with LF line endings, and `twine check` passed for both. setuptools warned that the form of the licence metadata is deprecated (see `FUTURE_WORK.md`). Installing the wheel with `pip` into a fresh environment, which is the route of a new user, was not verified. The upload to PyPI is left to the maintainer.

## Reproducibility

- Each `Fit_*` function seeds Python, NumPy and PyTorch and the shuffling of its data loader from `seednum` (default 2021). With the same seed, data, platform and PyTorch version, two runs give bitwise identical weights. `tests/test_smoke.py` tests this for `Fit_cord`, and the comparison above observed it for all four functions.
- CeLEry contains no CUDA code, so cuDNN settings do not apply and every result above was computed on a CPU.
- `Predict_layer` and `Predict_domain` keep dropout active, so repeated calls differ. Calling `torch.manual_seed()` before each call makes them repeatable.
- The order of genes built with `set()` depends on `PYTHONHASHSEED`. Sorting the list, or setting `PYTHONHASHSEED`, removes this dependence.
- `DataAugmentation` fixes its own seeds (2021, and `random_state=0` for `KMeans`), and the variational autoencoder reseeds PyTorch inside `reparameterize` (`ClusterVAE.py` line 115). Whether a complete run is reproducible was not verified.
- Results are not bitwise identical across PyTorch releases (see the comparison above) or across operating systems. With identical pins and the earlier default learning rate of 1e-3, the quickstart gave a median held-out error of 3.1 grid units on Windows and 3.2 on Linux.

## Paper and code (report only, nothing changed)

The published Methods were compared with the code on `main`, and each code reference below was checked by reading the cited lines.

**Agree.** The per-gene z-score (`util.py` lines 199-221), the ReLU layers with a sigmoid output (`DNN.py` lines 19-34), the squared-error loss for coordinates (`DNN.py` line 54), the ellipse loss of the confidence region, the ordinal loss for layers, the augmentation design with two replicates and β = 1e-5, and the prediction of the class with the highest probability.

**In the code but not in the paper.**

- Each coordinate axis is rescaled separately to (0, 1) with a margin of one grid unit (`datasetgenemap.py` lines 67-72). The aspect ratio is therefore not preserved, and `Predict_cord` returns coordinates on this rescaled scale.
- The optimiser is Adam with weight decay 1e-5, the learning rate is silently capped at 1e-2, and it is halved after repeated increases of the training loss (`TrainerExe.py` lines 17-46). The paper describes gradient descent, and the tutorial suggests a learning rate of 0.1, which the cap reduces to 0.01.
- Dropout of 0.25 stays active at prediction (`DNN.py` lines 210 and 406).
- The confidence score is 1 minus the product of the two radii of the ellipse (`DNN.py` line 378).
- The variational autoencoder resets the PyTorch seed on every call (`ClusterVAE.py` line 115).
- `AugFit_domain` assigns labels correctly only when the input is sorted by domain (`data_augmentation.py` lines 145-155).
- A `hidden_dims` list with four widths silently uses only the first three.

**Disagree.**

- The paper gives 50, 10 and 5 nodes as the default network, which only `AugFit_domain` uses. The defaults are 30, 25 and 15 for `Fit_cord` and `Fit_region`, 10, 5 and 2 for `Fit_layer` and `Fit_domain`, and 200, 100 and 50 for the `DNN` class.
- Equation 5 of the paper uses a sigmoid, whereas the code uses a weighted softmax cross-entropy. The equation is garbled in the PubMed Central text, so the typeset PDF should be checked.
- The paper describes 16, 8 and 4 channels in the autoencoder, whereas the code hard-codes 8, 4 and 2 (`data_augmentation.py` line 57).
- The paper sets the embedding dimension to 512, whereas the code uses 512 minus the number of clusters (`data_augmentation.py` line 64).
- Equation 1 sums the reconstruction loss over spots, whereas the code averages it over the whole grid.

Whether the published analyses used the code on `main` cannot be checked from the repository. The scripts in `code_paper/` call an older interface (`cel.SpaCluster`, `cel.DNNordinal_v1`), and `code_paper/1_LIBD/CeLEry_train_Scenario3and4.py` does not parse (line 320).

## Decisions

The maintainer decided three questions on 2026-10-28.

1. PyTorch stays at 2.14.1 rather than 2.2.2 with NumPy 1.26, since the upgrade changes results by about as much as a different seed.
2. The default `initial_learning_rate` returns to 1e-4, the value of release 1.2.1.
3. The repairs are released on PyPI as version 1.2.2 (see Release 1.2.2).

Two questions remain open.

1. The behaviour-changing items in `FUTURE_WORK.md`, namely evaluation mode at prediction, sorted gene lists, `KMeans(n_init=10)` and `data_range` in `get_SSIM`.
2. A `CITATION.cff` file, which the DenMark repository has and CeLEry does not.
