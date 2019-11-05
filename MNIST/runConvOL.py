#!/usr/bin/env python

def RunExperiment(train_loader, val_loader, epochs, torchOptim, lossFn, net, results_directory, name, device = 'cpu', saveAccToo = False):
    losses = train(net, train_loader, val_loader, torchOptim, lossFn, 0, epochs, name, device = device, checkForDiv = True, saveAccToo = saveAccToo)
    np.savetxt(results_directory + name + '.txt', np.array(losses), fmt='%s')

if __name__ == '__main__':
    import numpy as np
    import pickle
    import torch
    from torchvision import datasets, transforms
    from torch.utils.data import Dataset, DataLoader
    import torch.nn as nn
    import torch.nn.functional as F
    import os
    import time
    import sys

    current_directory = os.getcwd()
    parent_directory = os.path.abspath(os.path.join(current_directory, os.pardir))
    import sys

    sys.path.append(parent_directory)
    sys.path.append(parent_directory+'/Models')

    from Utilities import ScaleParameters, FixLayerData
    from OptimizerCode import   MixOptimizer
    from RunCode import train
    from modelsMNIST import MNIST_Conv as netToUse

    read_path = os.path.join(current_directory, 'DataMNIST/')

    # Let's read data common to all experiments.
    trainDS = pickle.load(open(read_path+'trainDS.p',"rb"))
    valDS = pickle.load(open(read_path+'valDS.p',"rb"))
    testDS = pickle.load(open(read_path+'testDS.p',"rb"))

    # Weights associated to the convolutional network
    wConvGraphD = pickle.load(open(read_path+'wConvGraphD.p',"rb"))
    wConvPyt = pickle.load(open(read_path+'wConvPyt.p',"rb"))

    suffix = 'ResultsConvOL/'
    results_directory = os.path.join(current_directory, suffix)
    if not os.path.exists(results_directory):
        os.makedirs(results_directory)

    batch_size = 250
    lNabla = [1, 4, 16, 64, 256, 1024, 4096]
    delta_Layer1 = [1, 5, 10, 50, 100]
    delta_Layer2 = [1, 5, 10, 50, 100]
    # possibleSNR =['SNRInfty', 'SNR1', 'SNR5', 'SNR10']
    methods = ['SGD', 'SFW', 'OneTwo']
    lossFn = nn.NLLLoss()
    use_cuda = True
    device = torch.device("cuda" if use_cuda else "cpu")
    shuffle = True
    epochs = 25
    kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}

    train_loader = torch.utils.data.DataLoader(trainDS, batch_size = batch_size, shuffle=True, **kwargs)
    val_loader = torch.utils.data.DataLoader(valDS, batch_size = 1000, shuffle=False, **kwargs)
    test_loader = torch.utils.data.DataLoader(testDS, batch_size=1000, shuffle=False, **kwargs)

    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--id', help = "Instance and nabla to use")
    args = parser.parse_args()

    index = int(args.id)
    ins = 0
    lNab = 0
    count = 0
    delta1 = 0
    for aux_delta1 in delta_Layer1:
        for aux_lNab in lNabla:
            if count  == index:
                delta1 = aux_delta1
                lNab = aux_lNab
            count += 1

    weightsGraph = wConvGraphD['Ins'+str(ins)]
    weightsPyt = wConvPyt['Ins'+str(ins)]
  
    nameInsAndLnab = 'Ins'+str(ins)+'_lNab'+str(lNab)+'_D1lta'+str(delta1)
    indFWs = [4]
    nodesFirst = 500
    ## We do not have a delta2, so let us just use a default for the name
    delta2 = 5

    ## Fix seed for reproducibility
    torch.manual_seed(564736827)

    ## Run SGD First
    stepSGD = 1.0/(2*lNab)
    print('Running SGD')
    ## Run SGD
    ## SGD does not use the deltas, but to follow the same name format I use them
    name = nameInsAndLnab+'_D2lta'+str(delta2)+'_SGD'
    net = netToUse().to(device)
    FixLayerData(net, weightsPyt)
    optimPerLayer = [methods[0]]*8
    partParams = [[stepSGD]]*8
    start = time.time()
    torchOptim =  MixOptimizer(net.parameters(), optimPerLayer, partParams, device = device)
    RunExperiment(train_loader, val_loader, epochs, torchOptim, lossFn, net, results_directory, name, device = device, saveAccToo = True)
    print('Running SGD took '+str(time.time() -start)+' seconds')
    print()


    stepFW1 = 1.0/(8*lNab*delta1*delta1)
    print('Running SFW')
    ## Run SFW
    name = nameInsAndLnab+'_D2lta'+str(delta2)+'_SFW'
    net = netToUse().to(device)
    FixLayerData(net, weightsGraph)
    ScaleParameters(net, indFWs, [delta1])
    optimPerLayer = [methods[0], methods[0], methods[0], methods[0], methods[1], methods[0], \
        methods[0], methods[0]]
    partParams = [[stepSGD]]*8
    partParams[4] = [delta1, (torch.ones(nodesFirst)).to(device)*stepFW1, (torch.ones(nodesFirst)).to(device),\
            torch.zeros(nodesFirst).to(device), True]
    torchOptim =  MixOptimizer(net.parameters(), optimPerLayer, partParams, device = device)
    start = time.time()
    RunExperiment(train_loader, val_loader, epochs, torchOptim, lossFn, net, results_directory, name, device = device, saveAccToo = True)
    print('Running SFW took '+str(time.time() -start)+' seconds')
    print()
    print('Running ALT')

    ## Run ALT
    name = nameInsAndLnab+'_D2lta'+str(delta2)+'_ALT'
    net = netToUse().to(device)
    FixLayerData(net, weightsGraph)
    ScaleParameters(net, indFWs, [delta1])
    optimPerLayer = [methods[0], methods[0], methods[0], methods[0], methods[2], methods[0], \
        methods[0], methods[0]]
    partParams = [[stepSGD]]*8
    partParams[4] = [delta1, (torch.ones(nodesFirst)).to(device)*stepFW1, (torch.ones(nodesFirst)).to(device),\
            torch.zeros(nodesFirst).to(device), True]
    torchOptim =  MixOptimizer(net.parameters(), optimPerLayer, partParams, device = device)
    start = time.time()
    RunExperiment(train_loader, val_loader, epochs, torchOptim, lossFn, net, results_directory, name, device = device, saveAccToo = True)
    print('Running ALT took '+str(time.time() -start)+' seconds')
    print()
