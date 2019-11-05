#!/usr/bin/env python

def RunExpSavNetAndUsingTest(train_loader, test_loader, epochs, torchOptim, lossFn, net, results_directory, name, device = 'cpu', saveAccToo = False, saveNetDict = True):
    losses = train(net, train_loader, test_loader, torchOptim, lossFn, 0, epochs, name, device = device, checkForDiv = True, saveNetDict = saveNetDict, pathToSave = results_directory)
    np.savetxt(results_directory + name + '.txt', np.array(losses))

def RunExpSGDTSavNetAndUsingTest(train_loader, test_loader, epochs, torchOptim, lossFn, net, results_directory, name, paramIndToThres, nzEntries, device = 'cpu', saveAccToo = False, saveNetDict = True):
    losses = trainSGDT(net, train_loader, test_loader, torchOptim, lossFn, 0, epochs, name, paramIndToThres, nzEntries, device = device, checkForDiv = True, saveNetDict = saveNetDict, pathToSave = results_directory)
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
    import sys
    from UtilitiesSynthetic import CreateRealLabelsAndSNR, CreateStartingLayerData, CreateFeatures, CreateDataLoader

    current_directory = os.getcwd()
    parent_directory = os.path.abspath(os.path.join(current_directory, os.pardir))
    import sys

    sys.path.append(parent_directory)
    sys.path.append(parent_directory+'/Models')

    from Utilities import CreateWeightsGraphical, createWeightsPytorchWay, FixLayerData, PrintModelStructure, ScaleParameters
    from OptimizerCode import   MixOptimizer
    from RunCode import train, trainSGDT
    from modelsSyntheticExp import SynRELU as netToUse

    read_path =os.path.join(current_directory, 'DataSynthetic/')
    # Let's save data common to all experiments.
    X_mat = pickle.load(open(read_path+'X_mat.p', "rb"))
    input_size = pickle.load(open(read_path+'input_size.p', "rb"))
    first_layer = pickle.load(open(read_path+'first_layer.p', "rb"))
    second_layer = pickle.load(open(read_path+'second_layer.p', "rb"))
    third_layer = pickle.load(open(read_path+'third_layer.p', "rb"))
    # mapLayerData = pickle.load(open(read_path+'mapLayerData.p', "rb"))

    pathForConf = os.path.join(current_directory, 'BestConf/')
    bestConf = pickle.load(open(pathForConf+'bestRELU.p', "rb"))

    # RELU

    read_path =os.path.join(current_directory, 'DataSynthetic/')
    mapLabelsRELU = pickle.load(open(read_path+'mapLabelsRELU.p', "rb"))
    mapInitWsPyRELU = pickle.load(open(read_path+'mapInitWsPyRELU.p', "rb"))
    mapInitWsGraphRELU = pickle.load(open(read_path+'mapInitWsGraphRELU.p', "rb"))
    mapInitWsGraphDRELU = pickle.load(open(read_path+'mapInitWsGraphDRELU.p', "rb"))

    suffix = 'ResultsBestRELU/'
    results_directory = os.path.join(current_directory, suffix)
    if not os.path.exists(results_directory):
        os.makedirs(results_directory)

    batch_size = 250
    batch_sizeVal = 1000
    lNabla = [1, 4, 16, 64, 256, 1024, 4096]
    # possibleSNR =['SNRInfty', 'SNR1', 'SNR5', 'SNR10']
    methods = ['SGD', 'SFW', 'OneTwo']
    lossFn = nn.MSELoss()
    use_cuda = False
    device = 'cpu'
    shuffle = True
    epochs = 25
    instances = 30

    deltaVecMap = {}
    deltaVecMap['5'] = [5, 5, 50]
    deltaVecMap['10'] = [10, 10, 50]
    deltaVecMap['15'] = [15, 15, 50]

    X_train = X_mat[0:100000,:]
    X_val = X_mat[100000:120000,:]
    X_test = X_mat[120000:220000,:]

    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--id', help = "Name of label to use")
    args = parser.parse_args()

    ## Fix seed for Reproducibility
    torch.manual_seed(7654367)


    auxArgsId = int(args.id)
    count  = 0
    methodToRun = ''
    confToUse = ''
    keysInBestConf = list(bestConf.keys())
    for methodName in keysInBestConf:
        for i, conf in enumerate(bestConf[methodName]):
            if count ==auxArgsId:
                methodToRun = methodName
                confToUse = conf
            count += 1
    
    partsConf = confToUse.split('-')
    nzAsStr = partsConf[0]
    snr = partsConf[1]
    lNab = float(partsConf[2])
    delta_vector = deltaVecMap[nzAsStr]

    ## For all
    stepSGD = 1.0/(2 * lNab)
    stepFW1 = 1.0/(8 * lNab*delta_vector[0] * delta_vector[0])
    stepFW2 = 1.0/(8 * lNab*delta_vector[1] * delta_vector[1])
    
    ## For SGDT
    nzVector = [delta_vector[0], delta_vector[1]]
    paramIndToThres = [0, 1]

    ## For SFW and ALT
    indFW = [0,1]
    deltaVecToUse = [delta_vector[0], delta_vector[1]]

    for i in range(instances):
        keyForLabels = 'NZ'+nzAsStr+'_Ins'+str(i)+'_SNR'+snr
        keyForWeights = 'NZ'+nzAsStr+'_Ins'+str(i)
        labels_train = mapLabelsRELU[keyForLabels][0:100000]
        labels_val = mapLabelsRELU[keyForLabels][100000:120000]
        labels_test = mapLabelsRELU[keyForLabels][120000:220000]
        train_loader = CreateDataLoader(X_train, labels_train, batch_size, shuffle, use_cuda)
        test_loader = CreateDataLoader(X_test, labels_test, batch_sizeVal, shuffle, use_cuda)
        if methodToRun == 'SGD':
            weightsToUse = mapInitWsPyRELU[keyForWeights]
            name = keyForLabels +'_lNabla'+str(lNab)+'_SGD'
            partParams = [[stepSGD], [stepSGD], [stepSGD]]
            net = netToUse(input_size, first_layer, second_layer, third_layer)
            FixLayerData(net, weightsToUse)
            optimPerLayer = [methods[0], methods[0], methods[0]]
            torchOptim =  MixOptimizer(net.parameters(), optimPerLayer, partParams)
            RunExpSavNetAndUsingTest(train_loader, test_loader, epochs, torchOptim, lossFn, net, results_directory, name, device = device)
        elif methodToRun == 'SGDT':
            weightsToUse = mapInitWsPyRELU[keyForWeights]
            name = keyForLabels +'_lNabla'+str(lNab)+'_SGDT'
            partParams = [[stepSGD], [stepSGD], [stepSGD]]
            net = netToUse(input_size, first_layer, second_layer, third_layer)
            FixLayerData(net, weightsToUse)
            optimPerLayer = [methods[0], methods[0], methods[0]]
            torchOptim =  MixOptimizer(net.parameters(), optimPerLayer, partParams)
            RunExpSGDTSavNetAndUsingTest(train_loader, test_loader, epochs, torchOptim, lossFn, net, results_directory, name, paramIndToThres, nzVector, device = device)
        elif methodToRun == 'SFW':
            weightsToUse = mapInitWsGraphDRELU[keyForWeights]
            name = keyForLabels +'_lNabla'+str(lNab)+'_SFW'
            net = netToUse(input_size, first_layer, second_layer, third_layer)
            FixLayerData(net, weightsToUse)
            ScaleParameters(net, indFW, deltaVecToUse)
            partParams = [[delta_vector[0], torch.ones(first_layer)*stepFW1, torch.ones(first_layer), torch.zeros(first_layer), True], [delta_vector[1], torch.ones(second_layer)*stepFW2, torch.ones(second_layer), torch.zeros(second_layer), True], [stepSGD]]
            optimPerLayer = [methods[1], methods[1], methods[0]]
            torchOptim =  MixOptimizer(net.parameters(), optimPerLayer, partParams)
            RunExpSavNetAndUsingTest(train_loader, test_loader, epochs, torchOptim, lossFn, net, results_directory, name, device = device)
        elif methodToRun == 'ALT':
            weightsToUse = mapInitWsGraphDRELU[keyForWeights]
            name = keyForLabels +'_lNabla'+str(lNab)+'_ALT'
            net = netToUse(input_size, first_layer, second_layer, third_layer)
            FixLayerData(net, weightsToUse)
            ScaleParameters(net, indFW, deltaVecToUse)
            partParams = [[delta_vector[0], torch.ones(first_layer)*stepFW1, torch.ones(first_layer), torch.zeros(first_layer), True], [delta_vector[1], torch.ones(second_layer)*stepFW2, torch.ones(second_layer), torch.zeros(second_layer), True], [stepSGD]]
            optimPerLayer = [methods[2], methods[2], methods[0]]
            torchOptim =  MixOptimizer(net.parameters(), optimPerLayer, partParams)
            RunExpSavNetAndUsingTest(train_loader, test_loader, epochs, torchOptim, lossFn, net, results_directory, name, device = device)