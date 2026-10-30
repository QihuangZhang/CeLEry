# Reference

All functions are available as `cel.<name>` after `import CeLEry as cel`. The signatures and defaults below are those of release 1.2.2. A trained model is stored as a pickle file `<path>/<filename>.obj`, which the matching `Predict_*` function reads back.

## Data preparation

### get_zscore

```python
cel.get_zscore(adata, mean=None, sd=None)
```

Standardizes the expression of every gene in place: `adata.X` becomes a dense array of z-scores, and the means and standard deviations are kept as `adata.genemean` and `adata.genesd`. Each dataset is standardized with its own mean and standard deviation unless `mean` and `sd` are given. A second call on the same object prints "Data has already been z-scored" and changes nothing. Genes without variation get NaN, which [drop_NaN](#drop_nan) removes.

### make_annData_spatial

```python
cel.make_annData_spatial(adata, spatial, min_cells=3, filtered=False)
```

Prepares a 10x Visium section read with `scanpy.read_10x_h5`. `spatial` is the table of spot positions of Space Ranger, read with `pd.read_csv(..., header=None, index_col=0)` from `tissue_positions_list.csv`; for `tissue_positions.csv` of Space Ranger 2.0 and later, which starts with a header row, add `skiprows=1`. Its columns 1 to 5, the in-tissue flag, the array row, the array column and the two pixel coordinates, become `select`, `x_cord`, `y_cord`, `x_pixel` and `y_pixel` in `adata.obs`. With `filtered=False`, only spots with `select == 1` are kept; with `filtered=True`, `adata.obs` must have no columns yet, and the in-tissue rows of `spatial` are joined to it. Since `select` becomes the first column of `adata.obs`, pass `location_data=adata.obs[["x_cord", "y_cord"]]` to [Fit_cord](#fit_cord) and [Fit_region](#fit_region), which otherwise read the coordinates from the first two columns. Gene names are converted to upper case and copied to `adata.var["genename"]`. Genes detected in fewer than `min_cells` spots and genes whose names start with `ERCC` or `MT-` are removed, and the counts are normalized per spot and log-transformed. Returns the processed AnnData.

### make_annData_query

```python
cel.make_annData_query(adata)
```

Prepares single-cell data in the same way: upper-case gene names in `adata.var["genename"]`, genes detected in at least 3 cells, no `ERCC` or `MT-` genes, normalization per cell and log transformation. Returns the processed AnnData.

### get_common_genes

```python
cel.get_common_genes(Qdata, Rdata)
```

Returns the genes present in both datasets, matched on `var["genename"]`. The list is built from a set, so its order changes between Python sessions and the trained model changes with it. Sort the list (`sorted(cel.get_common_genes(Qdata, Rdata))`) for results that repeat across sessions.

### drop_NaN

```python
cel.drop_NaN(adata)
```

Returns a new AnnData without the genes whose expression contains NaN, as left by [get_zscore](#get_zscore) for genes that do not vary.

### prefilter_cells, prefilter_genes, prefilter_specialgenes

```python
cel.prefilter_cells(adata, min_counts=None, max_counts=None, min_genes=200, max_genes=None)
cel.prefilter_genes(adata, min_counts=None, max_counts=None, min_cells=10, max_cells=None)
cel.prefilter_specialgenes(adata, Gene1Pattern="ERCC", Gene2Pattern="MT-")
```

Filter `adata` in place. `prefilter_cells` keeps the cells within the given count and gene limits and stores log-transformed data in `adata.raw`; `prefilter_genes` keeps the genes within the given limits; `prefilter_specialgenes` removes the genes whose names start with either pattern.

## Model fitting

The four `Fit_*` functions share their training settings.

| Argument | Meaning |
|---|---|
| `hidden_dims` | Number of nodes in each hidden layer. `Fit_cord` builds five hidden layers from a list of five values and otherwise uses the first three values; the other functions use the first three. |
| `num_epochs_max` | Maximum number of passes over the training data. |
| `path`, `filename` | The model is saved as `<path>/<filename>.obj`. Always set `path`: with the default empty string, `Fit_cord` and `Fit_region` stop with a `FileNotFoundError` once training has finished, and `Fit_layer` and `Fit_domain` try to write the model to the root folder of the drive. Pass the same `filename` to the matching `Predict_*` function, because the defaults do not match: `Fit_layer` saves `PreOrg_layersc` where `Predict_layer` reads `PreOrg_layernsc`, and `Fit_region` saves `PreOrg_Mousesc`, the default of `Fit_cord`, where `Predict_region` reads `PreOrg_ConfScore`. |
| `batch_size` | Number of spots per gradient step (default 4). |
| `num_workers` | Number of processes that load the data (default 1). See [Reading the output](reading-output.md#running-celery-in-a-script) for scripts on Windows and macOS. |
| `number_error_try` | Number of epochs whose loss exceeds the lowest loss so far before the learning rate is halved (default 15). |
| `initial_learning_rate` | Starting learning rate of the Adam optimizer (default 1e-4, at most 1e-2). |
| `seednum` | Seed of Python, NumPy, PyTorch and the data loader (default 2021). |

Training uses Adam with a weight decay of 1e-5. Each time `number_error_try` epochs have ended above the lowest loss, the learning rate is halved and the lowest loss is raised by 10 before the count starts again. Training stops after `num_epochs_max` epochs or once the learning rate falls below 1e-7.

### Fit_cord

```python
cel.Fit_cord(data_train, location_data=None, hidden_dims=[30, 25, 15], num_epochs_max=500,
             path="", filename="PreOrg_Mousesc", batch_size=4, num_workers=1,
             number_error_try=15, initial_learning_rate=0.0001, seednum=2021)
```

Trains the network that predicts the 2D coordinates of a cell from its expression. `data_train` holds the z-scored expression of the reference spots in `X`. `location_data` is a table whose first two columns are the coordinates of each spot, in the row order of `data_train`; by default it is `data_train.obs`. Every column of this table is converted to numbers, so with the default, all columns of `data_train.obs` must be numeric; otherwise pass the coordinates as `location_data`. The coordinates are rescaled to the interval (0, 1), with a margin of one unit beyond the smallest and largest coordinate on each axis, and the network predicts on that scale. Creates `path` if it does not exist and returns the trained model.

### Fit_region

```python
cel.Fit_region(data_train, alpha=0.95, location_data=None, hidden_dims=[30, 25, 15],
               num_epochs_max=500, path="", filename="PreOrg_Mousesc", batch_size=4,
               num_workers=1, number_error_try=15, initial_learning_rate=0.0001, seednum=2021)
```

Trains a network that predicts, for each cell, an ellipse that contains its true location with probability `alpha`. The ellipse is centered on the predicted coordinates and its axes are parallel to the coordinate axes; both live on the rescaled scale of [Fit_cord](#fit_cord). The training loss weights a true location outside the ellipse by `alpha` and one inside by `1 - alpha`. `location_data` works as in `Fit_cord`. Creates `path` if it does not exist and returns the trained model.

### Fit_layer

```python
cel.Fit_layer(data_train, layer_weights, layer_data=None, layerkey="layer", hidden_dims=[10, 5, 2],
              num_epochs_max=500, path="", filename="PreOrg_layersc", batch_size=4, num_workers=1,
              number_error_try=15, initial_learning_rate=0.0001, seednum=2021)
```

Trains an ordinal classifier of tissue layers, for layers that follow each other in space such as the cortical layers. The column `layerkey` of `layer_data` (by default `data_train.obs`) holds the layer of each spot as an integer from 1 to the number of layers. `layer_weights` is a pandas Series with one weight per layer, in layer order; weights inversely proportional to the number of spots per layer balance the layers. The model is saved to `<path>/<filename>.obj`, and nothing is returned. `path` must exist.

### Fit_domain

```python
cel.Fit_domain(data_train, domain_weights, domain_data=None, domainkey="layer", hidden_dims=[10, 5, 2],
               num_epochs_max=500, path="", filename="PreOrg_domainsc", batch_size=4, num_workers=1,
               number_error_try=15, initial_learning_rate=0.0001, seednum=2021)
```

Trains a classifier of spatial domains without an order. The column `domainkey` of `domain_data` (by default `data_train.obs`) holds the domain of each spot as an integer from 0 to the number of domains minus 1. `domain_weights` is a pandas Series with one weight per domain, in domain order. The model is saved to `<path>/<filename>.obj`, and nothing is returned. `path` must exist.

## Prediction

### Predict_cord

```python
cel.Predict_cord(data_test, path="", filename="PreOrg_Mousesc", location_data=None)
```

Predicts the coordinates of each cell of `data_test` with the model `<path>/<filename>.obj` from [Fit_cord](#fit_cord). Returns an array with one row per cell and two columns, on the rescaled (0, 1) scale of the training coordinates. The same values are added to `data_test.obs` as `x_cord_pred` and `y_cord_pred` and written to `<path>/<filename>_predmatrix.csv`, which each call overwrites. `location_data` does not affect the prediction. [Reading the output](reading-output.md#coordinates) shows how to convert the predictions back to the units of the reference.

### Predict_region

```python
cel.Predict_region(data_test, path="", filename="PreOrg_ConfScore", location_data=None, hist="")
```

Predicts the confidence region of each cell with the model from [Fit_region](#fit_region). Adds two columns to `data_test.obs`: `area_record`, the product of the two semi-axes of the ellipse on the rescaled scale, and `conf_score`, equal to 1 minus `area_record`. A smaller region gives a higher confidence score. The center and the semi-axes of the ellipse are neither returned nor saved, and the center comes from the network of `Fit_region`, not from that of `Fit_cord`. The function returns the proportion of rows whose location in `location_data` falls inside their ellipse. `location_data` is rescaled to (0, 1) with its own smallest and largest values, not with those of the training spots, so this proportion is the coverage of the regions only when `location_data` spans the same range as the training coordinates on both axes. Without `location_data`, it has no meaning. When `data_test.obs` has `x_cord` and `y_cord` columns, the areas are drawn as an image in `<path>/<filename>_AREA.pdf`. Leave `hist` at its default: an image array raises `ValueError: The truth value of an array with more than one element is ambiguous` before any column is added (see `FUTURE_WORK.md`).

### Predict_layer

```python
cel.Predict_layer(data_test, class_num, path="", filename="PreOrg_layernsc", truth_label=None,
                  predtype="probabilistic")
```

Predicts the layer of each cell with the model from [Fit_layer](#fit_layer). `class_num` is the number of layers used in training. With `predtype="probabilistic"`, it returns an array of layer probabilities with one row per cell and one column per layer; with `predtype="deterministic"`, it returns the most probable layer of each cell, numbered from 1. The most probable layer is also added to `data_test.obs` as `pred_layer`, `pred_layer_str` and `layer_cel_pred`, and `<path>/<filename>_probmat.csv` holds the predicted layer in its first column followed by the probabilities. Leave `truth_label` at `None`: any other value raises `UnboundLocalError: cannot access local variable 'location_data'` (see `FUTURE_WORK.md`). The layer classifier keeps dropout active at prediction, so call `torch.manual_seed()` before each call to make it repeatable.

### Predict_domain

```python
cel.Predict_domain(data_test, class_num, path="", filename="PreOrg_domainsc", truth_label=None,
                   predtype="probabilistic")
```

Predicts the spatial domain of each cell with the model from [Fit_domain](#fit_domain). `class_num` is the number of domains used in training. Returns domain probabilities (`predtype="probabilistic"`) or the most probable domain, numbered from 0 (`predtype="deterministic"`). The most probable domain is also added to `data_test.obs` as `pred_domain`, `pred_domain_str` and `domain_cel_pred`, and `<path>/<filename>_probmat.csv` holds the predicted domain in its first column followed by the probabilities. As for [Predict_layer](#predict_layer), leave `truth_label` at `None` and set `torch.manual_seed()` before each call.

## Data augmentation

### DataAugmentation

```python
cel.DataAugmentation(RefDataOrigin, obs_location=["x_cord", "y_cord"], path="output/Project",
                     filename="SpatialTranscript", clusterready=False, n_clusters=100, beta=1e-5,
                     nrep=2, generateplot=True)
```

Generates replicates of a spatial reference with a variational autoencoder. Each gene is first turned into an image over the spot grid given by the integer coordinates in the columns `obs_location`. The gene images are grouped into `n_clusters` clusters with k-means, or read from `<path>/DataAugmentation/<filename>_cluster.npy` when `clusterready=True`. The autoencoder is then trained for 249 epochs, during which the weight of its Kullback-Leibler term rises from 0 in steps of `beta / 4` every 50 epochs, and for 200 further epochs. `nrep` replicates are generated and saved as `<path>/DataAugmentation/DataGen/<filename>_data_gen_<beta>_n<nrep>.npy`. With `generateplot=True`, the image of every gene and ten generated versions of it are also saved as PDF files under `<path>/DataAugmentation/<filename>_Generation/`. The tutorial [Data augmentation](tutorials/augmentation.md) shows the full workflow.

### AugFit_domain

```python
cel.AugFit_domain(RefDataOrigin, domain_weights, domain_data=None, domainkey="layer",
                  hidden_dims=[50, 10, 5], num_epochs_max=500, beta=1e-5, nrep=2,
                  path="../output/Biogene", filename="SpatialTranscript", batch_size=4, num_workers=4,
                  number_error_try=15, initial_learning_rate=0.0001, seednum=2021)
```

Trains the domain classifier of [Fit_domain](#fit_domain) on the reference together with the replicates that [DataAugmentation](#dataaugmentation) generated with the same `path`, `filename`, `beta` and `nrep`. Its default `path` differs from that of `DataAugmentation` (`"../output/Biogene"` against `"output/Project"`), so pass the same `path` to both. `RefDataOrigin.X` must be a dense array, as [get_zscore](#get_zscore) leaves it. The spots of `RefDataOrigin` must be sorted by their grid coordinates, as `DataAugmentation` sorts its own copy, because the generated spots are written in that order and receive the labels of `domain_data` by position; otherwise they receive the labels of other spots. Unlike `Fit_domain`, this function passes `domain_weights` to the model as given, so it must be a float32 torch tensor, for example `torch.tensor(weights.to_numpy(), dtype=torch.float32)`. The model is saved to `<path>/DataAugmentation/PredictionModel/<filename>_domain_<beta>_n<nrep>.obj`.

The steps of `DataAugmentation` are also available separately: `FitGenModel` and `FitGenModel_continue` train the autoencoder, `GeneratePlot` writes example images, and `Data_Generation` writes the replicates.

## Evaluation and plots

### get_Layer_LIBD

```python
cel.get_Layer_LIBD(adata, coords_predict, referann)
```

Assigns to each predicted location the layer of the closest reference spot. The first three columns of `referann.obs` must be the two coordinates and the layer. The reference coordinates are mapped to `(x - x_min) / (x_max - x_min + 1)`, which differs from the scale of [Predict_cord](#predict_cord), `(x - x_min + 1) / (x_max - x_min + 2)`, so a prediction that falls exactly on a spot can be matched to its neighbor. The layers are added to `adata.obs["pred_layer"]` and returned.

### plot_layer

```python
cel.plot_layer(adata, folder, name, coloruse=None)
```

Draws the predicted layers (`pred_layer_str`) and the reference layers (`Layer`) of the spots at the coordinates `x2` and `x3`, and saves them as `<folder>/<name>_Layer_pred.pdf` and `<folder>/<name>_Layer_ref.pdf`.

### plot_confusion_matrix

```python
cel.plot_confusion_matrix(referadata, filename, nlayer=7)
```

Compares `referadata.obs["Layer"]` with `referadata.obs["pred_layer"]`. Writes the confusion matrix as row proportions to `<filename>.csv` and the precision, recall and F1 score of each layer to `<filename>_Classification_Metric.json`.

### RelocationPlot

```python
cel.RelocationPlot(coords, referlocation, filename=None, linewidth=0.8)
```

Draws a line from the true location of each spot (`referlocation`, integer grid coordinates) to its predicted location (`coords`, on the (0, 1) scale). The plot is shown, or saved as `<filename>.pdf` when `filename` is given.

### Pred_Density

```python
cel.Pred_Density(coords, referadata, referlocation)
```

Returns the number of predicted locations that fall in each pixel of the grid spanned by `referlocation`.

### UncertaintyPlot

```python
cel.UncertaintyPlot(cords, filename, hist)
```

Draws the confidence region areas stored in `cords.X` at the locations `cords.obs["x"]` and `cords.obs["y"]`, multiplied by 50, over the image `hist`, and saves the result as `<filename>.pdf`. The factor 50 matches the histology pixels per grid unit of the mouse brain data of the paper.

### get_SSIM

```python
cel.get_SSIM(coords, referadata, referlocation, trainAnn, genelist)
```

Compares, gene by gene, the expression image at the predicted locations with the image at the true locations. It currently raises `ValueError: ... you must specify the data_range parameter` with every scikit-image release since at least 0.23, because the range of the images is not given (see `FUTURE_WORK.md`).

### getGeneImg, plotGeneImg

```python
cel.getGeneImg(datainput, emptypixel, obsset=None)
cel.plotGeneImg(img, filename=None, range=None, plotcolor="YlGnBu")
```

`getGeneImg` turns each gene into an image over the integer spot coordinates (`x_cord` and `y_cord`, or the two columns named in `obsset`) and stores the stack as `datainput.GeneImg`; pixels without a spot take the value `emptypixel`. `plotGeneImg` draws one such image, and saves it as `<filename>.pdf` when `filename` is given.
