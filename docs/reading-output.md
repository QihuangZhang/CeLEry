# Reading the output

## Coordinates

[Predict_cord](reference.md#predict_cord) returns coordinates on a rescaled scale, not in the units of the reference. [Fit_cord](reference.md#fit_cord) maps each axis of the training coordinates to the interval (0, 1), with a margin of one unit beyond the smallest and the largest value. A prediction `p` on the first axis therefore corresponds to

```text
x = (x_min - 1) + p * (x_max - x_min + 2)
```

in the units of the reference, where `x_min` and `x_max` are the smallest and largest training coordinates, and likewise for the second axis. The function `to_grid` of the [quick start](get-started.md#check-the-predictions-on-the-held-out-spots) applies this conversion. The columns `x_cord_pred` and `y_cord_pred` that `Predict_cord` adds to `obs`, and the file `<filename>_predmatrix.csv`, hold the rescaled values.

A prediction can fall between spots or outside the tissue, because the network predicts a continuous position. To assign a cell to the closest reference spot, convert its prediction with `to_grid` and take the nearest spot in the units of the reference. [get_Layer_LIBD](reference.md#get_layer_libd) matches predictions to spots on a slightly different scale, so it can return the neighbor of the closest spot.

## Confidence regions

[Fit_region](reference.md#fit_region) and [Predict_region](reference.md#predict_region) give each cell an ellipse that is meant to contain its true location with probability `alpha`. `Predict_region` adds two columns to `obs`:

- `area_record`, the product of the two semi-axes of the ellipse on the rescaled scale;
- `conf_score`, equal to 1 minus `area_record`.

A cell whose expression points to a narrow region gets a high confidence score. The areas are on the rescaled scale of one trained model, so compare confidence scores between cells predicted by the same model, not between models trained on different references. The ellipses themselves are not returned, and their centers come from the network of `Fit_region`, which can differ slightly from the predictions of `Predict_cord`.

With known coordinates in `location_data`, `Predict_region` returns the proportion of rows whose location falls inside their ellipse. It rescales `location_data` with its own range rather than with that of the training spots. This proportion therefore measures the coverage, which should be close to `alpha`, only when the held-out spots span the same range as the training spots on both axes.

## Layers and domains

[Predict_layer](reference.md#predict_layer) and [Predict_domain](reference.md#predict_domain) return, for each cell, one probability per class, and the probabilities of a cell sum to 1. The most probable class is added to `obs` and written in the first column of `<filename>_probmat.csv`, followed by the probabilities.

- Layers are numbered from 1, in the order in which they follow each other in the tissue. The layer model is ordinal: it learns where a cell lies along that order, which suits structures such as the cortical layers.
- Domains are numbered from 0 and have no order.

Both classifiers keep dropout active at prediction, so two calls on the same cells give different probabilities. In a check in October 2026, layer probabilities differed by up to 0.245 between two identical calls. Call `torch.manual_seed()` with a fixed value before each prediction to make it repeatable.

## Running CeLEry in a script

The `Fit_*()` functions load the training data with `num_workers` worker processes (1 by default), and every `Predict_*()` function uses one worker process. On Windows and macOS, Python starts these workers by importing the main script again, so a script must run CeLEry inside a main guard:

```text
import CeLEry as cel


def main():
    ...  # the CeLEry calls


if __name__ == "__main__":
    main()
```

`examples/quickstart.py` follows this pattern. With `num_workers=0`, the `Fit_*()` functions load the data in the main process.

## Settings that matter

The defaults are a starting point, and the tutorials use larger networks than the defaults of some functions.

- `num_epochs_max` (default 500) is the largest number of passes over the reference. Training stops earlier once the learning rate falls below 1e-7.
- `initial_learning_rate` (default 1e-4) is the starting learning rate. Values above 1e-2 are reduced to 1e-2.
- `number_error_try` (default 15) is the number of epochs whose loss may exceed the lowest loss so far before the learning rate is halved. After each halving, the lowest loss is raised by 10 and the count starts again. With many epochs, a larger value, such as 60, halves the rate less often.
- `batch_size` (default 4) is the number of spots per gradient step. If the loss becomes NaN, a different batch size can restore stable training.
- `hidden_dims` sets the number of nodes in each hidden layer. The tutorials on coordinates, layers and domains use `[30, 25, 15]`, the augmentation tutorial uses `[50, 10, 5]`, the default of `AugFit_domain`, and `Fit_layer` and `Fit_domain` default to `[10, 5, 2]`.
- `num_workers` (default 1) is the number of processes that load the training data. On Windows and macOS, a value of 1 or more starts new processes in every epoch, and on the Windows test machine of October 2026 this put a 500-epoch tutorial task at an estimated 1.3 to 2.6 hours. `num_workers=0` loads the data in the main process, and with it `Fit_cord` gave bitwise identical weights.

## Common problems

- **NaN after `get_zscore`.** A gene with the same value in every spot or cell cannot be standardized, and its z-scores are NaN. Remove such genes with [drop_NaN](reference.md#drop_nan) before matching the genes of the two datasets. A NaN in the input makes the output of the network NaN. `Predict_layer` then stops with a `ValueError`, but `Predict_domain` assigns every cell to the last domain without an error, so check with `np.isnan(adata.X).any()` before predicting.
- **The model predicts nonsense for the query.** The network reads genes by column position, so the reference and the query must contain the same genes in the same order. Subset both to their common genes, sorted, as the [layer tutorial](tutorials/layers.md) does.
- **`FileNotFoundError` in `Fit_layer` or `Fit_domain`.** These two functions do not create the output folder, and the error comes only after training, so the trained model is lost. Create the folder before training with `os.makedirs(path, exist_ok=True)`.
- **Coordinates after `make_annData_spatial`.** That function puts `select` in the first column of `obs`, so `Fit_cord` and `Fit_region` would read `select` and `x_cord` as the coordinates. Pass `location_data=adata.obs[["x_cord", "y_cord"]]`.
- **`FileNotFoundError` at the end of `Fit_cord` or `Fit_region`.** `path` was left at its default, the empty string, which names no folder. Set `path` to the folder for the model file.
- **`UnboundLocalError: cannot access local variable 'location_data'` in `Predict_layer` or `Predict_domain`.** Leave `truth_label` at `None`. Any other value raises this error (see `FUTURE_WORK.md`).
- **`ValueError: ... you must specify the data_range parameter` in `get_SSIM`.** This function currently fails with scikit-image (see `FUTURE_WORK.md`).
- **`AttributeError: 'csr_matrix' object has no attribute 'A'`.** This comes from CeLEry 1.2.1 or older together with SciPy 1.14 or later. Upgrade to CeLEry 1.2.2.
