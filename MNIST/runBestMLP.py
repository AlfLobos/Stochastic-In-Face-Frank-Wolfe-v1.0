#!/usr/bin/env python

def RunExperiment(trainVal_loader, test_loader, epochs, torchOptim, lossFn, net, results_directory, name, device = 'cpu', saveAccToo = False, saveNetDict = True):
    losses = train(net, trainVal_loader, test_loader, torchOptim, lossFn, 0, epochs, name, device = device, checkForDiv = True, saveAccToo = saveAccToo, pathToSave = results_directory,  saveNetDict = saveNetDict)
    np.savetxt(results_directory + name + '.txt', np.array(losses))

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
    from modelsMNIST import MNIST_MLP as netToUse

    pathForConf = os.path.join(current_directory, 'BestConf/')
    bestConf = pickle.load(open(pathForConf+'bestMLP.p', "rb"))

    read_path = os.path.join(current_directory, 'DataMNIST/')

    # Let's read data common to all experiments.
    trainValDS = pickle.load(open(read_path+'trAndValDS.p',"rb"))
    testDS = pickle.load(open(read_path+'testDS.p',"rb"))

    # Weights associated to the convolutional network
    wMLPGraphD = pickle.load(open(read_path+'wMLPGraphD.p',"rb"))
    wMLPPyt = pickle.load(open(read_path+'wMLPPyt.p',"rb"))

    suffix = 'ResultsBestMLP/'
    results_directory = os.path.join(current_directory, suffix)
    if not os.path.exists(results_directory):
        os.makedirs(results_directory)

    batch_size = 250
    methods = ['SGD', 'SFW', 'OneTwo']
    lossFn = nn.NLLLoss()
    use_cuda = True
    device = torch.device("cuda" if use_cuda else "cpu")
    epochs = 25
    kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}

    trainVal_loader = torch.utils.data.DataLoader(trainValDS, batch_size = batch_size, shuffle=True, **kwargs)
    test_loader = torch.utils.data.DataLoader(testDS, batch_size=1000, shuffle=False, **kwargs)

    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--id', help = "Instance and nabla to use")
    args = parser.parse_args()

    ## There are only 3 runs to make
    methodToUse = ''
    conf = ''
    index = int(args.id)
    ins  = int(index/3)
    indexMet = index % 3
    for i, key in enumerate(list(bestConf.keys())):
        if i == indexMet:
            methodToUse = key
            conf = bestConf[key]
            break
    partsConf = conf.split('-')
    lNab = float(partsConf[0])
    delta1 = float(partsConf[1])
    delta2 = float(partsConf[2])
    indFWs = [0, 2]
    nodesFirst = 512
    nodesSecond = 512

    weightsGraph = wMLPGraphD['Ins'+str(ins)]
    weightsPyt = wMLPPyt['Ins'+str(ins)]
    stepSGD = 1.0/(2*lNab)
    stepFW1 = 1.0/(8*lNab*delta1*delta1)
    stepFW2 = 1.0/(8*lNab*delta2*delta2)

    ## For reproducibility
    torch.manual_seed(564736827)

    if methodToUse == 'SGD':
        name = 'Ins'+str(ins)+'_SGD'
        net = netToUse().to(device)
        FixLayerData(net, weightsPyt)
        optimPerLayer = [methods[0]]*6
        partParams = [[stepSGD]]*6
        start = time.time()
        torchOptim =  MixOptimizer(net.parameters(), optimPerLayer, partParams, device = device)
        RunExperiment(trainVal_loader, test_loader, epochs, torchOptim, lossFn, net, results_directory, name, device = device, saveAccToo = True)
        print('Running SGD took '+str(time.time() -start)+' seconds')
        print()
    elif methodToUse == 'SFW':
        name = 'Ins'+str(ins)+'_SFW'
        net = netToUse().to(device)
        FixLayerData(net, weightsGraph)
        ScaleParameters(net, indFWs, [delta1, delta2])
        optimPerLayer = [methods[1], methods[0], methods[1], methods[0], methods[0], methods[0]]
        partParams = [[stepSGD]]*6
        partParams[0] = [delta1, (torch.ones(nodesFirst)).to(device)*stepFW1, torch.ones(nodesFirst).to(device), torch.zeros(nodesFirst).to(device), True]
        partParams[2] = [delta2, (torch.ones(nodesSecond)).to(device)*stepFW2, torch.ones(nodesSecond).to(device), torch.zeros(nodesSecond).to(device), True]
        torchOptim =  MixOptimizer(net.parameters(), optimPerLayer, partParams, device = device)
        start = time.time()
        RunExperiment(trainVal_loader, test_loader, epochs, torchOptim, lossFn, net, results_directory, name, device = device, saveAccToo = True)
        print('Running SFW took '+str(time.time() -start)+' seconds')
        print()
    
    elif methodToUse =='ALT':
        name = 'Ins'+str(ins)+'_ALT'
        net = netToUse().to(device)
        FixLayerData(net, weightsGraph)
        ScaleParameters(net, indFWs, [delta1, delta2])
        optimPerLayer = [methods[2], methods[0], methods[2], methods[0], methods[0], methods[0]]
        partParams = [[stepSGD]]*6
        partParams[0] = [delta1, (torch.ones(nodesFirst)).to(device)*stepFW1, torch.ones(nodesFirst).to(device), torch.zeros(nodesFirst).to(device), True]
        partParams[2] = [delta2, (torch.ones(nodesSecond)).to(device)*stepFW2, torch.ones(nodesSecond).to(device), torch.zeros(nodesSecond).to(device), True]
        torchOptim =  MixOptimizer(net.parameters(), optimPerLayer, partParams, device = device)
        start = time.time()
        torchOptim =  MixOptimizer(net.parameters(), optimPerLayer, partParams, device = device)
        RunExperiment(trainVal_loader, test_loader, epochs, torchOptim, lossFn, net, results_directory, name, device = device, saveAccToo = True)
        print('Running ALT took '+str(time.time() -start)+' seconds')
        print()
