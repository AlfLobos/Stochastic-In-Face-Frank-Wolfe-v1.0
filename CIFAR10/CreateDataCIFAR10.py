import numpy as np
import pickle
import os 

import torch
import torchvision
import torchvision.transforms as transforms
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
from modelsCIFAR10 import CIFAR10_ConvSimple as netCIFAR10

if __name__ == '__main__':

    ## Let's create the folder 'DataMNIST' in case it does not exist.
    data_directory = os.path.join(current_directory, 'DataCIFAR10/')
    if not os.path.exists(data_directory):
        os.makedirs(data_directory)

    # Data downloaded in the same way as in 
    # https://github.com/pytorch/tutorials/blob/master/beginner_source/blitz/cifar10_tutorial.py

    print('We first download the dataset')

    ## Let's first create the dataset having splits for Train, Validation, and Test.
    print('Let us split the dataset in train, validation, and test')
    print()
    transform = transforms.Compose(
        [transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    trainAndValset = torchvision.datasets.CIFAR10(root='./data', train=True,
    download=True, transform=transform)

    testset = torchvision.datasets.CIFAR10(root='./data', train=False,
    download=True, transform=transform)

    ## CIFAR10 has 50000 examples, of those we will use 45000 for train and 5000 for validation
    indShuf = np.arange(50000)
    np.random.shuffle(indShuf)
    trainDS = Subset(trainAndValset, indShuf[0:45000])
    valDS = Subset(trainAndValset, indShuf[45000:])

    pickle.dump(trainDS, open(data_directory+'trainDS.p',"wb"))
    pickle.dump(valDS, open(data_directory+'valDS.p',"wb"))
    pickle.dump(testset, open(data_directory+'testDS.p',"wb"))
    pickle.dump(trainAndValset, open(data_directory+'trAndValDS.p',"wb"))


    ## Let's create the weights.
    print('Let us create the weights for the experiment')
    print('First for the Simple Convolutional Architecture')
    PrintModelStructure(netCIFAR10())
    print()
    instances = 30
    torch.manual_seed(809376)
    indFWs = [4, 6]
    ## Delta is actually non-important as I will later scale the values
    delta_vector = [1, 1]

    ### Weights for Conv
    wConvSimGraphD = {}
    wConvSimPyt = {}
    for i in range(instances):
        wConvSimGraphD['Ins'+str(i)] = CreateWeightsGraphical(netCIFAR10, indFWs, delta_vector, double = True)
        wConvSimPyt['Ins'+str(i)] = createWeightsPytorchWay(netCIFAR10)

    pickle.dump(wConvSimGraphD, open(data_directory+'wConvSimGraphD.p',"wb"))
    pickle.dump(wConvSimPyt, open(data_directory+'wConvSimPyt.p',"wb"))