import random
import numpy as np
import torch
import pickle

from . import util

from sklearn.cluster import KMeans


def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def FitGenModel (path, filename, traindata, cdataexpand, Kmeans_cluster, beta, learning_rate = 1e-3):
    random.seed(2021)
    torch.manual_seed(2021)
    np.random.seed(2021)
    #
    trainloader= torch.utils.data.DataLoader(traindata, batch_size=1, num_workers = 4, shuffle = True, worker_init_fn=seed_worker)
    ## Set up Autoencoder
    CVAEmodel = cel.ClusterVAEmask(latent_dim = 511-Kmeans_cluster.max(), total_cluster = Kmeans_cluster.max(), fgx = cdataexpand.shape[2], fgy = cdataexpand.shape[3], KLDw = 0, hidden = [8,4,2,4,4])
    CVAEmodel = CVAEmodel.float()
    filename = "{path}/DataAugmentation/{filename}_CVAE_{beta}.obj".format(path = path, filename = filename, beta = beta)
    #
    ## Run Autoencoder 
    clg=cel.TrainerExe()
    clg.train(model = CVAEmodel, train_loader = trainloader, num_epochs= 249, annealing = True, KLDwinc = beta/4, n_incr =50, RCcountMax = 30, learning_rate = 0.001)
    # Save the model to a local folder
    filehandler = open(filename, 'wb') 
    pickle.dump(CVAEmodel, filehandler)
    print('save model to: {filename}'.format(filename = filename))
    CVAEmodel.filename = filename
    return CVAEmodel, clg

## if still converging
def FitGenModel_continue (path, filename, model, clg, traindata, cdataexpand, beta):
    trainloader= torch.utils.data.DataLoader(traindata, batch_size=1, num_workers = 4, shuffle = True, worker_init_fn=seed_worker)
    #
    filename = "{path}/DataAugmentation/{filename}_CVAE_{beta}.obj".format(path = path, filename = filename, beta = beta)
    clg.train(model = model, train_loader = trainloader, num_epochs= 200, annealing = False, RCcountMax = 5, learning_rate = clg.learning_rate)
    # Save the model to a local folder
    filehandler = open(filename, 'wb') 
    pickle.dump(model, filehandler)
    print('save model to: {filename}'.format(filename=filename))
    model.filename = filename
    return model, clg

