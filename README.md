# CeLEry
## CeLEry: Cell Location Recovery in Single-cell RNA Sequencing 

### Qihuang Zhang, Jian Hu, David Dai, Edward B. Lee, Rui Xiao, Mingyao Li*

Single-cell RNA sequencing provides resourceful information to study the cells systematically. However, their locational information is usually unavailable. We present CeLEry, a supervised deep learning algorithm to recover the origin of tissues in assist of spatial transcriptomic data, integrating a data augmentation procedure via variational autoencoder to improve the robustness of methods in the overfitting and the data contamination. CeLEry provides a generic framework and can be implemented in multiple tasks depending on the research objectives, including the spatial coordinates discovery as well as the layer discovery. It can make use of the information of multiple tissues of spatial transcriptomics data. Thorough assessments exhibit that CeLEry achieves a leading performance compared to the state-of-art methods. We illustrated the usage of CeLEry in the discovery of neuron cell layers to study the development of Alzheimer's disease. The identified cell location information is valuable in many downstream analyses and can be indicative of the spatial organization of the tissues.

![CeLEry workflow](docs/asserts/images/workflow.png)

## Usage

The [**CeLEry**](https://github.com/QihuangZhang/CeLEry) package is an implementation of a deep neural network in discovering location information for single cell RNA data. With CeLEry, you can:

- Preprocess spatial transcriptomics data from various formats.
- Build a deep neural network to predict cell locations.
- Generate synthetic spatial transcriptomic data.



## Tutorial


A Jupyter Notebook of the tutorial (*UNDER CONSTRUCTION*) is accessible from : 
<br>
https://github.com/QihuangZhang/CeLEry/blob/main/tutorial/tutorial.ipynb
<br>


## System Requirements
Python support packages: torch>1.8, pandas>1.4, numpy>1.20, scipy, tqdm, scanpy>1.5, anndata, sklearn, pickle, random, math, os


# Install packages
In the command, input
```
pip install -i https://test.pypi.org/simple/ CeLEryPy
```