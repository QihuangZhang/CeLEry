# Installation

CeLEry is a Python package. It is distributed on PyPI as `CeLEryPy` and imported as `CeLEry`.

## The tested environment

The repository pins the versions that were tested together in `environment.yml`. Clone the repository, create the environment with conda, mamba or micromamba, and install CeLEry from the clone:

```bash
git clone https://github.com/QihuangZhang/CeLEry.git
cd CeLEry
conda env create -f environment.yml     # or: micromamba create -f environment.yml
conda activate celery
pip install --no-deps ./CeLEry_package
```

Creating the environment took about five minutes on the test machine. The clone also contains the data of the [quick start](get-started.md) and of the tutorials.

On Windows, micromamba 2 can fail with "The directory is not empty" when its root folder has a long path, because the cached PyTorch files then exceed the Windows limit of 260 characters. A short root folder, such as `C:\Users\<user>\mm`, avoids the error. Enabling long paths in Windows should avoid it as well, but this was not tested.

## From PyPI

```bash
pip install CeLEryPy
```

This installs release 1.2.2, which contains the same code as the repository, and lets pip choose the version of each dependency. The data of the quick start and the tutorials are not part of the package, so clone the repository to run them.

## Tested versions

| Python | PyTorch | NumPy | SciPy | pandas | anndata | scanpy | scikit-learn | scikit-image | matplotlib |
|---|---|---|---|---|---|---|---|---|---|
| 3.12.15 | 2.14.1 (CPU) | 2.5.4 | 1.18.1 | 2.3.3 | 0.13.4 | 1.12.4 | 1.9.1 | 0.26.0 | 3.11.2 |

These versions were tested on 2026-10-28 on Windows 11, with the quick start, the smoke tests and every code block of tasks 1 to 3 of the original tutorial (`tutorial/tutorial.md`, 3 training epochs per model), and on Ubuntu, with the quick start and the smoke tests. pandas stays below 3.0, because pandas 3 breaks `get_Layer_LIBD()`. `RENOVATION_LOG.md` gives the reason for each pin.

## GPU

CeLEry trains and predicts on the CPU. It never moves data or models to a GPU, so a CUDA build of PyTorch brings no speed-up, and `environment.yml` installs the CPU build, which conda-forge provides for Linux, macOS and Windows.

## Check the installation

From the root of the clone:

```bash
python examples/quickstart.py
python -m pytest -q tests
```

The quick start prints the error on held-out spots and writes the predicted coordinates of 300 cells to `examples/output/`. The tests check that the package imports, that `Predict_cord` returns one pair of coordinates per cell, and that two fits with the same seed give identical weights. They also run the quick start, the Python example of the README and the code of the [Quick start](get-started.md) page as written, and the code of the coordinates, layers and domains tutorials with 3 training epochs instead of 500. Every `cel.` function called in the Python code of the documentation, and every keyword argument of these calls, is checked against the package. The tests write their output to `output/` and take a few minutes.
