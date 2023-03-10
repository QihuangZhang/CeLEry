import os
import random
import numpy as np
import pandas as pd
import torch
import pickle

from .util import *

from sklearn.cluster import KMeans
from . datasetgenemap import datagenemapclust
from . ClusterVAE import ClusterVAEmask
from . TrainerExe import TrainerExe
from . datasetgenemap import wrap_gene_domain
from . DNN import DNN
from . DNN import DNNordinal
from . DNN import DNNdomain

def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)

def DataAugmentation (RefDataOrigin, obs_location = ['x_cord','y_cord'], path = "output/Project", filename = "SpatialTranscript", clusterready = False, n_clusters=100,  beta = 1e-5, nrep = 2, generateplot = True):
    #Prepare
    RefDataOriginsort = RefDataOrigin.obs.sort_values (by = obs_location)
    RefDataOrigin = RefDataOrigin[RefDataOriginsort.index]
    cdata = RefDataOrigin.copy()
    getGeneImg(cdata, emptypixel = 0, obsset = obs_location)
    cdataexpand =  np.expand_dims(cdata.GeneImg, axis=1) 
    #Clustering
    try:
        os.makedirs("{path}/DataAugmentation".format(path = path))
    except FileExistsError:
        print("Folder already exists")
    if clusterready:
        kmeansresults = np.load("{path}/DataAugmentation/{filename}_cluster.npy".format(path = path, filename = filename))
    else:
        kmeansmodel =  KMeans(n_clusters, random_state=0)
        cdatacentral = centralize(cdataexpand.copy())
        direclust = [cdatacentral[x,0,:,:] for x in range(cdatacentral.shape[0])]
        direflat = [x.flat for x in direclust]
        direflatnp = np.stack(direflat)
        kmeans = kmeansmodel.fit(direflatnp)
        kmeansresults = kmeans.labels_
        np.save("{path}/DataAugmentation/{filename}_cluster.npy".format(path = path, filename = filename), kmeansresults)
    # 
    full_RefData = datagenemapclust(cdataexpand, kmeansresults)
    CVAEmodel, clg = FitGenModel(path = path, filename = filename, traindata = full_RefData, cdataexpand = cdataexpand, Kmeans_cluster = kmeansresults, beta = beta)
    CVAEmodel, clg = FitGenModel_continue(path = path, filename = filename, model = CVAEmodel, clg = clg, traindata = full_RefData, beta = beta)
    if generateplot:
        print("Now generating the plots for the augmented data...")
        GeneratePlot(path, filename, beta = beta, traindata = full_RefData)
    Data_Generation(path, filename, obs_location = obs_location, beta= beta, dataSection1 = RefDataOrigin, traindata = full_RefData, nrep = nrep)


def FitGenModel (path, filename, traindata, cdataexpand, Kmeans_cluster, beta, hidden = [8,4,2,4,4], learning_rate = 1e-3,  number_error_try = 30):
    random.seed(2021)
    torch.manual_seed(2021)
    np.random.seed(2021)
    #
    trainloader= torch.utils.data.DataLoader(traindata, batch_size=1, num_workers = 1, shuffle = True, worker_init_fn=seed_worker)
    ## Set up Autoencoder
    CVAEmodel = ClusterVAEmask(latent_dim = 511-Kmeans_cluster.max(), total_cluster = Kmeans_cluster.max(), fgx = cdataexpand.shape[2], fgy = cdataexpand.shape[3], KLDw = 0, hidden = hidden)
    CVAEmodel = CVAEmodel.float()
    file = "{path}/DataAugmentation/{filename}_CVAE_{beta}.obj".format(path = path, filename = filename, beta = beta)
    #
    ## Run Autoencoder 
    clg = TrainerExe()
    clg.train(model = CVAEmodel, train_loader = trainloader, num_epochs= 249, annealing = True, KLDwinc = beta/4, n_incr =50, RCcountMax = number_error_try, learning_rate = learning_rate)
    # Save the model to a local folder
    filehandler = open(file, 'wb') 
    pickle.dump(CVAEmodel, filehandler)
    print('save model to: {filename}'.format(filename = file))
    CVAEmodel.filename = file
    return CVAEmodel, clg

## if still converging
def FitGenModel_continue (path, filename, model, clg, traindata, beta):
    trainloader= torch.utils.data.DataLoader(traindata, batch_size=1, num_workers = 1, shuffle = True, worker_init_fn=seed_worker)
    #
    file = "{path}/DataAugmentation/{filename}_CVAE_{beta}.obj".format(path = path, filename = filename, beta = beta)
    clg.train(model = model, train_loader = trainloader, num_epochs= 200, annealing = False, RCcountMax = 5, learning_rate = clg.learning_rate)
    # Save the model to a local folder
    filehandler = open(file, 'wb') 
    pickle.dump(model, filehandler)
    print('save model to: {filename}'.format(filename=file))
    model.filename = file
    return model, clg

def GeneratePlot(path, filename, beta, traindata, sigma = 0):
    trainloader= torch.utils.data.DataLoader(traindata, batch_size=1, num_workers = 4)
    file = "{path}/DataAugmentation/{filename}_CVAE_{beta}.obj".format(path = path, filename = filename, beta = beta)
    # 
    filehandler = open(file, 'rb') 
    CVAEmodel = pickle.load(filehandler)
    #
    clg=TrainerExe()
    clg.model = CVAEmodel
    try:
        os.makedirs("{path}/DataAugmentation/{file}_Generation/Glimps/Gen{beta}".format(path = path, file = filename, beta = beta))
    except FileExistsError:
        print("Folder {path}/DataAugmentation/{file}_Generation/Glimps/Gen{beta} already exists".format(path = path, file = filename, beta = beta))
    for j, img in enumerate(trainloader):
        # img = next(dataloader_iterator)
        plotGeneImg(img[0][0,0,:,:], filename = "{path}/DataAugmentation/{file}_Generation/Glimps/Gen{beta}/img{j}".format(path = path, file = filename, beta = beta, j = j))
        omin = img[0].min()
        omax = img[0].max()
        if sigma == 0:
            sigma = (omax-omin)/6
        for i in range(10):
            CVAEmodel.seed = i
            result = CVAEmodel(img) 
            outputraw = result[0][0,0,:,:].detach().numpy()
            outputimg = (outputraw + np.random.normal(0,sigma,outputraw.shape)) * result[4][0,0,:,:].detach().numpy()
            plotGeneImg( outputimg , filename = "{path}/DataAugmentation/{file}_Generation/Glimps/Gen{beta}/img{j}var{i}".format(path = path, file = filename, beta = beta, j = j, i = i), range = (-3, 3))


def Data_Generation(path, filename, beta, dataSection1, traindata, nrep, obs_location = ['x_cord','y_cord']):
    trainloader= torch.utils.data.DataLoader(traindata, batch_size=1, num_workers = 4)
    random.seed(2021)
    torch.manual_seed(2021)
    np.random.seed(2021)
    #
    fileto = "{path}/DataAugmentation/{filename}_CVAE_{beta}.obj".format(path = path, filename = filename, beta = beta)
    filehandler = open(fileto, 'rb') 
    CVAEmodel = pickle.load(filehandler)
    #
    clg= TrainerExe()
    clg.model = CVAEmodel
    data_gen=clg.fast_generation(trainloader, nrep)
    # data_gen=np.load("../output/{folder}/data_gen.npy".format(folder = folder))
    data_gen_rs = clg.deep_reshape (data = data_gen, refer = dataSection1.obs[obs_location])
    try:
        os.makedirs("{path}/DataAugmentation/DataGen".format(path = path))
    except FileExistsError:
        print("Folder already exists")
    np.save("{path}/DataAugmentation/DataGen/{filename}_data_gen_{beta}_n{nrep}.npy".format(path = path, filename = filename, beta = beta, nrep = nrep), data_gen_rs)


def AugFit_domain (RefDataOrigin, domain_weights, domain_data = None, domainkey = "layer", hidden_dims =  [50, 10, 5], num_epochs_max = 500, beta = 1e-5, nrep = 2,  path = "../output/Biogene", filename = "SpatialTranscript", batch_size = 4, num_workers = 4, number_error_try = 15, initial_learning_rate = 0.0001, seednum = 2021):
    random.seed(seednum)
    torch.manual_seed(seednum)
    np.random.seed(seednum)
    if domain_data is None:
        domain_data = RefDataOrigin.obs
    #
    # Original Version
    data_gen_rs = np.load("{path}/DataAugmentation/DataGen/{filename}_data_gen_{beta}_n{nrep}.npy".format(path = path, filename = filename, beta = beta, nrep = nrep))
    # Attach the original
    tdatax = np.expand_dims(RefDataOrigin.X, axis = 0)
    tdata_rs = np.swapaxes(tdatax, 1, 2)
    datacomp = np.concatenate((data_gen_rs, tdata_rs), axis=0)
    #
    dataDNN = wrap_gene_domain(datacomp, domain_data, domainkey)
    CoReg_loader = torch.utils.data.DataLoader(dataDNN, batch_size=batch_size, num_workers = num_workers, shuffle = True, worker_init_fn=seed_worker)
    # Create Deep Neural Network for Coordinate Regression
    DNNmodel = DNNdomain( in_channels = data_gen_rs.shape[1], num_classes = domain_weights.shape[0], hidden_dims = hidden_dims, importance_weights = domain_weights)
    DNNmodel = DNNmodel.float()
    #
    CoReg = TrainerExe()
    CoReg.train(model = DNNmodel, train_loader = CoReg_loader, num_epochs= num_epochs_max, RCcountMax = number_error_try, learning_rate = initial_learning_rate)
    #
    try:
        os.makedirs("{path}/DataAugmentation/PredictionModel".format(path = path))
    except FileExistsError:
        print("Note: Folder {path}/DataAugmentation/PredictionModel already exists".format(path = path))
    filename2 = "{path}/DataAugmentation/PredictionModel/{filename}_domain_{beta}_n{nrep}.obj".format(filename = filename, path = path, beta = beta, nrep = nrep)
    filehandler2 = open(filename2, 'wb') 
    pickle.dump(DNNmodel, filehandler2)

