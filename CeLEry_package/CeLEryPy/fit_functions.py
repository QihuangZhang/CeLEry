import random
import numpy as np
import pandas as pd
import torch
import os

from . datasetgenemap import wrap_gene_location
from . datasetgenemap import wrap_gene_layer
from . DNN import DNN
from . DNN import DNNordinal
from . TrainerExe import TrainerExe
import pickle


def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)

def Fit_cord (data_train, location_data = None, hidden_dims = [30, 25, 15], num_epochs_max = 500, path = "", filename = "PreOrg_Mousesc", batch_size = 4, num_workers = 4, number_error_try = 15, initial_learning_rate = 0.0001, seednum = 2021):
    #
    random.seed(seednum)
    torch.manual_seed(seednum)
    np.random.seed(seednum)
    g = torch.Generator()
    g.manual_seed(seednum)
    #
    if location_data is None:
        location_data = data_train.obs
    #
    tdatax = np.expand_dims(data_train.X, axis = 0)
    tdata_rs = np.swapaxes(tdatax, 1, 2)
    DataTra = wrap_gene_location(tdata_rs, location_data)
    t_loader= torch.utils.data.DataLoader(DataTra, batch_size=batch_size, num_workers = num_workers, shuffle = True, worker_init_fn=seed_worker, generator=g)
    # Create Deep Neural Network for Coordinate Regression
    DNNmodel = DNN( in_channels = DataTra[1][0].shape[0], hidden_dims = hidden_dims) # [100,50,25] )
    DNNmodel = DNNmodel.float()
    #
    CoOrg = TrainerExe()
    CoOrg.train(model = DNNmodel, train_loader = t_loader, num_epochs= num_epochs_max, RCcountMax = number_error_try, learning_rate = initial_learning_rate)
    #
    try:
        os.makedirs("{path}".format(path = path))
    except FileExistsError:
        print("Folder already exists")
    filename3 = "{path}/{filename}.obj".format(path = path, filename = filename) #"../output/CeLEry/Mousesc/PreOrg_Mousesc.obj"
    filehandler2 = open(filename3, 'wb') 
    pickle.dump(DNNmodel, filehandler2)
    return DNNmodel

def Fit_layer (data_train, layer_weights, layer_data = None, layerkey = "layer", hidden_dims = [10, 5, 2], num_epochs_max = 500, path = "", filename = "PreOrg_layersc", batch_size = 4, num_workers = 4, number_error_try = 15, initial_learning_rate = 0.0001, seednum = 2021):
    #
    random.seed(seednum)
    torch.manual_seed(seednum)
    np.random.seed(seednum)
    g = torch.Generator()
    g.manual_seed(seednum)
    #
    if layer_data is None:
        layer_data = data_train.obs
    #
    tdatax = np.expand_dims(data_train.X, axis = 0)
    tdata_rs = np.swapaxes(tdatax, 1, 2)
    DataTra = wrap_gene_layer(tdata_rs, data_train.obs, layerkey)
    t_loader= torch.utils.data.DataLoader(DataTra, batch_size = batch_size, num_workers = num_workers, shuffle = True, worker_init_fn=seed_worker, generator=g)
    # Create Deep Neural Network for Coordinate Regression
    DNNmodel = DNNordinal( in_channels = DataTra[1][0].shape[0], num_classes = layer_weights.shape[0], hidden_dims = hidden_dims, importance_weights = layer_weights) # [100,50,25] )
    DNNmodel = DNNmodel.float()
    #
    CoOrg= TrainerExe()
    CoOrg.train(model = DNNmodel, train_loader = t_loader, num_epochs= num_epochs_max, RCcountMax = number_error_try, learning_rate = initial_learning_rate)
    #
    filename3 = "{path}/{filename}.obj".format(path = path, filename = filename)
    filehandler2 = open(filename3, 'wb') 
    pickle.dump(DNNmodel, filehandler2)


def Predict_cord (data_test, path = "", filename = "PreOrg_Mousesc", location_data = None):
    if location_data is None:
        location_data = pd.DataFrame(np.ones((data_test.shape[0],2)), columns = ["psudo1", "psudo2"])
    ## Wrap up Validation data in to dataloader
    vdatax = np.expand_dims(data_test.X, axis = 0)
    vdata_rs = np.swapaxes(vdatax, 1, 2)
    DataVal = wrap_gene_location(vdata_rs, location_data)
    Val_loader= torch.utils.data.DataLoader(DataVal, batch_size=1, num_workers = 4)
    #
    cord = report_prop_method_sc(folder = path,
                        name = filename, data_test = data_test,
                        Val_loader = Val_loader)
    return cord



def report_prop_method_sc (folder, name, data_test, Val_loader, outname = ""):
    """
        Report the results of the proposed methods in comparison to the other method
        :folder: string: specified the folder that keep the proposed DNN method
        :name: string: specified the name of the DNN method, also will be used to name the output files
        :data_test: AnnData: the data of query data
        :Val_loader: Dataload: the validation data from dataloader
        :outname: string: specified the name of the output, default is the same as the name
    """
    filename2 = "{folder}/{name}.obj".format(folder = folder, name = name)
    filehandler = open(filename2, 'rb') 
    DNNmodel = pickle.load(filehandler)
    #
    coords_predict = np.zeros((data_test.obs.shape[0],2))
    #
    for i, img in enumerate(Val_loader):
        recon = DNNmodel(img)
        coords_predict[i,:] = recon[0].detach().numpy()
    np.savetxt("{folder}/{name}_predmatrix.csv".format(folder = folder, name = name), coords_predict, delimiter=",")
    return coords_predict

