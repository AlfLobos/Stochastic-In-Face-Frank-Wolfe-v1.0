import numpy as np
import pickle
import torch
from torchvision import datasets, transforms
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import normalize
import torch.nn as nn
import torch.nn.functional as F
import os
import sys
from UtilitiesSynthetic import CreateRealLabelsAndSNR, CreateStartingLayerData, CreateFeatures

current_directory = os.getcwd()
parent_directory = os.path.abspath(os.path.join(current_directory, os.pardir))
import sys

sys.path.append(parent_directory)
sys.path.append(parent_directory+'/Models')

from Utilities import CreateWeightsGraphical, createWeightsPytorchWay, FixLayerData, PrintModelStructure
from modelsSyntheticExp import SynRELU, SynSigmoid


if __name__ == '__main__':
    ## Common Data
    input_size = 50
    first_layer = 50
    second_layer = 50
    third_layer =1
    valDifFromZero = 1
    instancesOfNetwork = 30
    ## We will try 3 sizes of delta vector (therefore numOfSparsityLevels = 3)
    deltaVectorList = [[5,5,50], [10,10,50], [15,15,50]] 
    numOfSparsityLevels = len(deltaVectorList)
    namesNZ = ['NZ5', 'NZ10', 'NZ15']
    layerSizes = [input_size, first_layer, second_layer,\
                third_layer]

    if len(namesNZ) != len(deltaVectorList):
        print('Error: The cardinality of namesNZ must match the one of deltaVectorList,')
        print('it would be wise to fix the issue and re-start the code.')

    namesSNR = ['SNR1', 'SNR5', 'SNR10', 'SNRInfty']
    SNRLevels = [1, 5, 10]
    ## Seeds for reproducibility
    # (I fix the seeds here in general for this whole script and not for each function separately
    #  as it makes the code uglier, and the procedure should be fix in any case.)
    seedNumpy = 12345
    seedTorch = 92839
    seedDesignMat = 4789
    np.random.seed(seedNumpy)
    torch.manual_seed(seedTorch)


    ## Let's create a folder call DataSynthetic in case it does not exist already in the current Directory
    # We will use this folder to store the info for both experiments.

    
    data_directory = os.path.join(current_directory, 'DataSynthetic/')
    if not os.path.exists(data_directory):
        os.makedirs(data_directory)

    ## Common to both experiments will be the Design matrix 
    ## Our design matrix will have 100000 points, the validation one 20000, and the test 100000.
    allDataPoints = 100000 + 20000 + 100000
    mu = 0
    stdNormal = 1
    X_mat = CreateFeatures(allDataPoints, input_size,  mu, stdNormal, seedDesignMat)

    ## Let's create the real layer data

    mapLayerData = {}

    for i in range(instancesOfNetwork):
        for j, nameNZ  in enumerate(namesNZ):
            name = nameNZ+'_Ins'+str(i)
            mapLayerData[name] = CreateStartingLayerData(layerSizes = layerSizes, \
                entPerNodeDifFromZero =deltaVectorList[j], valDifFromZero = valDifFromZero)
    
    # Let's save data common to all experiments.
    pickle.dump(X_mat, open(data_directory+'X_mat.p',"wb"))
    pickle.dump(input_size, open(data_directory+'input_size.p',"wb"))
    pickle.dump(first_layer, open(data_directory+'first_layer.p',"wb"))
    pickle.dump(second_layer, open(data_directory+'second_layer.p',"wb"))
    pickle.dump(third_layer, open(data_directory+'third_layer.p',"wb"))
    pickle.dump(mapLayerData, open(data_directory+'mapLayerData.p',"wb"))
    
    ## Let's now create data particular to RELU and Sigmoid

    netRELU = SynRELU(*layerSizes)
    netSigmoid = SynSigmoid(*layerSizes)

    print('Relu Model Structure')
    PrintModelStructure(netRELU)

    print('Sigmoid Model Structure')
    PrintModelStructure(netSigmoid)

    ## Let's now create the labels under different SNR levels.
    mapLabelsRELU = {}
    mapLabelsSigmoid = {}

    for i in range(instancesOfNetwork):
        for j, nameNZ  in enumerate(namesNZ):
            nameLayer = nameNZ+'_Ins'+str(i)
            ## For Relu
            FixLayerData(netRELU, mapLayerData[nameLayer])
            labelsRELU = CreateRealLabelsAndSNR(X_mat, netRELU, SNRLevels)
            for z, snrName in enumerate(namesSNR):
                mapLabelsRELU[nameNZ+'_Ins'+str(i)+'_'+snrName] = labelsRELU[z]
            ## For Sigmoid
            FixLayerData(netSigmoid, mapLayerData[nameLayer])
            labelsSigmoid = CreateRealLabelsAndSNR(X_mat, netSigmoid, SNRLevels)
            for z, snrName in enumerate(namesSNR):
                mapLabelsSigmoid[nameNZ+'_Ins'+str(i)+'_'+snrName] = labelsSigmoid[z]
    
    mapInitWsPyRELU = {}
    mapInitWsGraphRELU = {}
    mapInitWsGraphDRELU = {}
    mapInitWsPySigmoid = {}
    mapInitWsGraphSigmoid = {}
    mapInitWsGraphDSigmoid = {}

    ## indToInit is a list that has in order the layers that I want to initialize using the graphical way when
    ## I run CreateWeightsGraphical.  In this case for the three layers that both models use.
    indToInit = np.arange(3)

    for i in range(instancesOfNetwork):
        for j, nameNZ  in enumerate(namesNZ):
            name = nameNZ+'_Ins'+str(i)
            mapInitWsPyRELU[name] = createWeightsPytorchWay(SynRELU, indToScale = indToInit, \
                delta_vector = deltaVectorList[j], paramsFnNet = layerSizes)
            mapInitWsGraphRELU[name] = CreateWeightsGraphical(SynRELU, indToInit, deltaVectorList[j],\
                paramsFnNet = layerSizes , double = False)
            mapInitWsGraphDRELU[name] = CreateWeightsGraphical(SynRELU, indToInit, deltaVectorList[j], \
                paramsFnNet = layerSizes , double = True)
            mapInitWsPySigmoid[name] = createWeightsPytorchWay(SynSigmoid, indToScale = indToInit, \
                delta_vector = deltaVectorList[j], paramsFnNet = layerSizes)
            mapInitWsGraphSigmoid[name] = CreateWeightsGraphical(SynSigmoid, indToInit, deltaVectorList[j], \
                paramsFnNet = layerSizes , double = False)
            mapInitWsGraphDSigmoid[name] = CreateWeightsGraphical(SynSigmoid, indToInit, deltaVectorList[j], \
                paramsFnNet = layerSizes , double = True)

    ## Let's now save the rest of the data
    # RELU

    pickle.dump(mapLabelsRELU, open(data_directory+'mapLabelsRELU.p',"wb"))
    pickle.dump(mapInitWsPyRELU, open(data_directory+'mapInitWsPyRELU.p',"wb"))
    pickle.dump(mapInitWsGraphRELU, open(data_directory+'mapInitWsGraphRELU.p',"wb"))
    pickle.dump(mapInitWsGraphDRELU, open(data_directory+'mapInitWsGraphDRELU.p',"wb"))

    # Sigmoid

    pickle.dump(mapLabelsSigmoid, open(data_directory+'mapLabelsSigmoid.p',"wb"))
    pickle.dump(mapInitWsPySigmoid, open(data_directory+'mapInitWsPySigmoid.p',"wb"))
    pickle.dump(mapInitWsGraphSigmoid, open(data_directory+'mapInitWsGraphSigmoid.p',"wb"))
    pickle.dump(mapInitWsGraphDSigmoid, open(data_directory+'mapInitWsGraphDSigmoid.p',"wb"))