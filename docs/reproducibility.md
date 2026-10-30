# Reproducibility

## Seeds

The `Fit_*()` functions seed Python, NumPy, PyTorch and the data loader with `seednum` (default 2021). Two fits with the same data, settings and seed therefore give bitwise identical weights when they run with the same package versions on the same platform. The test `test_fit_cord_same_seed_is_bitwise_identical` in `tests/test_smoke.py` checks this on every change to the repository.

## What still varies

Results change in four situations, even with a fixed seed.

- **Prediction of layers and domains.** `Predict_layer` and `Predict_domain` keep dropout active, so two calls give different probabilities. Calling `torch.manual_seed()` with a fixed value before each call makes a prediction repeatable.
- **Gene order.** Gene lists built with `set()`, including the list returned by `get_common_genes()`, change order between Python sessions, and the trained model changes with the order of the genes. Sorting the list, or fixing `PYTHONHASHSEED`, makes the order repeatable.
- **Versions and platforms.** PyTorch releases and operating systems differ in rounding and in their random number streams. In a 20-epoch test, moving from the dependencies of April 2024 to those of October 2026 changed predicted coordinates by up to 8.1 grid units, against 15.6 for a change of seed, and every difference was traced to the PyTorch version.
- **Data augmentation.** Since scikit-learn 1.4, k-means runs one initialization by default instead of ten. `DataAugmentation` relies on that default, so its gene clusters, and therefore the generated data, differ from those of the environment in which the paper was written.

## Records

`RENOVATION_LOG.md` describes the 2026 maintenance release: each change, each version pin with its reason, and the comparison of results between the dependencies of 2024 and 2026. `FUTURE_WORK.md` lists the changes that would alter results and were therefore left for a decision by the authors, such as switching dropout off at prediction.

The analyses of the paper ran in 2022 and 2023 with the package versions of that time. The scripts that produced them are listed in [Reproducing the paper](case-studies.md).
