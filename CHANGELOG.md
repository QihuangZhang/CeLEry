# Changelog

The entries before 1.2.2 were reconstructed in October 2026 by comparing the releases on PyPI. They list changes to functions, arguments, defaults and dependencies, and they omit changes inside functions that leave the interface unchanged.

## Development version

No change to the package code since 1.2.2.

* Documentation website at <https://qihuangzhang.github.io/CeLEry/>, built with MkDocs from `docs/` and deployed by GitHub Actions.
* A CeLEry logo.
* The README follows the layout of [DenMark](https://github.com/StaGill/DenMark) and [PGAlign](https://github.com/StaGill/PGAlign), and the README of the package (`CeLEry_package/README.md`), which PyPI shows from the next release on, matches it.
* `CITATION.cff`, which GitHub shows as a "Cite this repository" button.
* This changelog.
* `code_paper/README.md` maps each folder of paper code to its section, figures and data, and the four folders that had no README now have one.
* Issue templates for bug reports and questions about data or results.
* A workflow that publishes each GitHub release to PyPI through trusted publishing.

## CeLEry 1.2.2 (2026-10-29)

Compatibility release. The functions compute what they computed in 1.2.1 and run with the scientific Python releases of 2026.

* Sparse expression matrices work again with SciPy 1.14 and later. Every use of the `.A` attribute, which SciPy 1.14 removed, was replaced by `.toarray()` (10 places in `fit_functions.py` and `util.py`). With SciPy 1.14 or later, release 1.2.1 stops with `AttributeError: 'csr_matrix' object has no attribute 'A'`.
* pandas is limited to versions below 3.0, because pandas 3 breaks `get_Layer_LIBD()`.
* `layers.py`, which did not compile and was never imported, is no longer shipped.
* The packaged `LICENSE` names Qihuang Zhang as the copyright holder.
* The default `initial_learning_rate` of the `Fit_*()` functions remains 1e-4, as in 1.2.1.
* New in the repository: `environment.yml` with the tested versions, a quick start on bundled data in `examples/`, smoke tests run by GitHub Actions, and `RENOVATION_LOG.md` and `FUTURE_WORK.md`, which record what was changed, what was tested and what remains open.

## CeLEry 1.2.1 (2024-04-10)

* New functions `get_common_genes()` and `drop_NaN()`.
* Depends on `scikit-learn` instead of `sklearn`. Releases up to 1.1.3 list `sklearn`, a deprecated package name that pip has refused to install since December 2023.

## CeLEry 1.1.3 (2023-09-26)

* `DataAugmentation()` and `Data_Generation()` take the names of the coordinate columns as `obs_location` (default `["x_cord", "y_cord"]`), and the default `path` of `DataAugmentation()` changes from `"../output/Biogene"` to `"output/Project"`.
* `FitGenModel()` gains `hidden` and `number_error_try`, `GeneratePlot()` gains `sigma`, and `FitGenModel_continue()` no longer takes `cdataexpand`.
* `coloruse` of `plot_layer()` becomes optional.

## CeLEry 1.1.1 (2022-11-10)

* `Predict_region()` names its area plots after the model file.

## CeLEry 1.1.0 (2022-11-08)

* New functions `Fit_region()` and `Predict_region()`, which estimate a confidence region for each predicted location.
* `num_workers` of `Fit_cord()`, `Fit_layer()` and `Fit_domain()` defaults to 1 instead of 4.
* `RelocationPlot()` gains `linewidth`, and `geneset` of `getGeneImg()` is renamed `obsset`.

## CeLEry 1.0.4 (2022-10-29)

* `Fit_cord()` accepts five hidden layers (`hidden_dims` of length 5).

## CeLEry 1.0.3 (2022-09-29)

* `predtype` of `Predict_layer()` and `Predict_domain()` takes `"probabilistic"` instead of `"probability"`.
* `make_annData_spatial()` gains `filtered`.
* `RelocationPlot()` no longer takes `referadata`, and its `filename` becomes optional.
* `getGeneImgSparse()` is removed.
* Depends on scikit-image.

## CeLEry 1.0.2 (2022-07-15)

* New function `Predict_layer()`.

## CeLEry 1.0.1 (2022-06-26)

* `Predict_domain()` no longer stops with a `TypeError`.

## CeLEry 1.0.0 (2022-06-20)

* First release with the fitting functions. It contains `Fit_cord()`, `Predict_cord()`, `Fit_layer()`, `Fit_domain()`, `Predict_domain()`, the data augmentation functions `DataAugmentation()` and `AugFit_domain()`, and the preprocessing and plotting functions.

## CeLEry 0.0.1 (2022-06-20)

* First upload. The package was imported as `CeLEryPy` and did not yet contain the fitting functions.
