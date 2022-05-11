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

def GeneratePlot(path, filename, beta, traindata):
    trainloader= torch.utils.data.DataLoader(traindata, batch_size=1, num_workers = 4)
    filename = "{path}/DataAugmentation/{filename}_CVAE_{beta}.obj".format(path = path, filename = filename, beta = beta)
    # 
    filehandler = open(filename, 'rb') 
    CVAEmodel = pickle.load(filehandler)
    #
    clg=cel.TrainerExe()
    clg.model = CVAEmodel
    try:
        os.makedirs("{path}/DataAugmentation/{filename}_Generation/Glimps/Gen{beta}".format(path = path, filename = filename, beta = beta))
    except FileExistsError:
        print("Folder already exists")
    for j, img in enumerate(trainloader):
        # img = next(dataloader_iterator)
        cel.plotGeneImg(img[0][0,0,:,:], filename = "../output/LIBD/Generation/Glimps/Gen{beta}/img{j}".format(beta = beta, j = j))
        omin = img[0].min()
        omax = img[0].max()
        for i in range(10):
            result = CVAEmodel(img) 
            outputimg = result[0][0,0,:,:].detach().numpy() * result[4][0,0,:,:].detach().numpy()
            cel.plotGeneImg( outputimg , filename = "../output/LIBD/Generation/Glimps/Gen{beta}/img{j}var{i}".format(beta = beta, j = j, i = i), range = (omin.item(), omax.item()))


def Data_Generation(path, filename, beta, dataSection1, traindata, nrep):
    trainloader= torch.utils.data.DataLoader(traindata, batch_size=1, num_workers = 4)
    random.seed(2021)
    torch.manual_seed(2021)
    np.random.seed(2021)
    #
    fileto = "{path}/DataAugmentation/{filename}_CVAE_{beta}.obj".format(path = path, filename = filename, beta = beta)
    filehandler = open(fileto, 'rb') 
    CVAEmodel = pickle.load(filehandler)
    #
    clg= cel.TrainerExe()
    clg.model = CVAEmodel
    data_gen=clg.fast_generation(trainloader, nrep)
    # data_gen=np.load("../output/{folder}/data_gen.npy".format(folder = folder))
    data_gen_rs = clg.deep_reshape (data = data_gen, refer = dataSection1.obs[["x_cord","y_cord"]])
    try:
        os.makedirs("{path}/DataAugmentation/DataGen".format(path = path))
    except FileExistsError:
        print("Folder already exists")
    np.save("{path}/DataAugmentation/DataGen/{filename}_data_gen_{beta}_n{nrep}.npy".format(path = path, filename = filename, beta = beta, nrep = nrep), data_gen_rs)

