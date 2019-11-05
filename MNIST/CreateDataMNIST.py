import numpy as np
import pickle
import os 

import torch
from torchvision import datasets, transforms
from torch.utils.data import Sampler
from torch.utils.data.dataset import Subset
import torch.nn as nn
import torch.nn.functional as F

current_directory = os.getcwd()
parent_directory = os.path.abspath(os.path.join(current_directory, os.pardir))
import sys

sys.path.append(parent_directory)
sys.path.append(parent_directory+'/Models')

from Utilities import CreateWeightsGraphical, createWeightsPytorchWay, FixLayerData, PrintModelStructure
from modelsMNIST import MNIST_Conv, MNIST_MLP

if __name__ == '__main__':

    ## Let's create the folder 'DataMNIST' in case it does not exist.
    data_directory = os.path.join(current_directory, 'DataMNIST/')
    if not os.path.exists(data_directory):
        os.makedirs(data_directory)

    print('We first download the dataset')

    ## Let's first create the dataset having splits for Train, Validation, and Test.
    print('Let us split the dataset in train, validation, and test')
    print()
    mnistDataTrVal = datasets.MNIST('../data', train=True, download=True,
                    transform=transforms.Compose([
                        transforms.ToTensor(),
                        transforms.Normalize((0.1307,), (0.3081,))
                    ]))
    testDS = datasets.MNIST('../data', train=False, transform=transforms.Compose([
                        transforms.ToTensor(),
                        transforms.Normalize((0.1307,), (0.3081,))
                    ]))
    np.random.seed(7654321)
    ## MNIST train has 60000 elements
    indShuf = np.arange(60000)
    np.random.shuffle(indShuf)
    trainDS = Subset(mnistDataTrVal, indShuf[0:50000])
    valDS = Subset(mnistDataTrVal, indShuf[50000:])
    pickle.dump(trainDS, open(data_directory+'trainDS.p',"wb"))
    pickle.dump(valDS, open(data_directory+'valDS.p',"wb"))
    pickle.dump(mnistDataTrVal, open(data_directory+'trAndValDS.p',"wb"))
    pickle.dump(testDS, open(data_directory+'testDS.p',"wb"))


    ## Let's create the weights.
    print('Let us create the weights for the experiment')
    print('First for the Conv Architecture')
    PrintModelStructure(MNIST_Conv())
    print()
    instances = 30
    torch.manual_seed(809376)
    indFWs = [4]
    ## Delta is actually non-important as I will later scale the values
    delta_vector = [1]

    ### Weights for Conv
    wConvGraphD = {}
    wConvPyt = {}
    for i in range(instances):
        wConvGraphD['Ins'+str(i)] = CreateWeightsGraphical(MNIST_Conv, indFWs, delta_vector, double = True)
        wConvPyt['Ins'+str(i)] = createWeightsPytorchWay(MNIST_Conv)

    ### Weights for MLP
    torch.manual_seed(5476839)
    print('Now for the MLP Architecture')
    PrintModelStructure(MNIST_MLP())
    indFWs = [0, 2]
    wMLPGraphD = {}
    wMLPPyt = {}
    for i in range(instances):
        wMLPGraphD['Ins'+str(i)] = CreateWeightsGraphical(MNIST_MLP, indFWs, delta_vector, double = True)
        wMLPPyt['Ins'+str(i)] = createWeightsPytorchWay(MNIST_MLP)

    pickle.dump(wConvGraphD, open(data_directory+'wConvGraphD.p',"wb"))
    pickle.dump(wConvPyt, open(data_directory+'wConvPyt.p',"wb"))
    pickle.dump(wMLPGraphD, open(data_directory+'wMLPGraphD.p',"wb"))
    pickle.dump(wMLPPyt, open(data_directory+'wMLPPyt.p',"wb"))