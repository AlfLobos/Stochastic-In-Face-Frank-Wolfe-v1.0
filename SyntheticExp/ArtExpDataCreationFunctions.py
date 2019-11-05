# # Import Libraries

import numpy as np
import torch

from torchvision import datasets, transforms
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import normalize
from OptimizerCode import LinSeqRandomSampler
# import math

#
#
## Data Creation Artificial Experiment


def ScaleDictOfTensors(dictOfTensors, indToScale, delta_vector):
    i = 0
    for index, key in enumerate(dictOfTensors.keys()):
        if index in indToScale:
            l1Norm = torch.sum(torch.abs(dictOfTensors[key]), dim =1)
            rows, _ = dictOfTensors[key].size()
            with torch.no_grad():
                for j in range(int(rows)):
                    if l1Norm[j] > delta_vector[i]:
                        dictOfTensors[key][j,:] *= delta_vector[i]/l1Norm[j]
            i += 1

def ThresHoldParameters(net, layersToThres, thres):
    for i, param in enumerate(net.parameters()):
        if i in layersToThres:
            param.data[torch.abs(param.data) < thres]

def CreateFeatures(allDataPoints, input_layer,  mu, stdNormal, seed_numpy):
    np.random.seed(seed_numpy)
    X =  np.random.normal(loc = mu, scale = stdNormal,\
        size = (allDataPoints, input_layer))
    return torch.tensor(normalize(X), dtype = torch.float)

def CreateWeightsPytorchWay(layerSizes = [50, 50, 50, 1], delta_vector = [5, 5, 50],\
    numToCreate = 25, numpy_seed = 1234):
    toRet = []
    np.random.seed(numpy_seed)
    for j in range(numToCreate):
        auxWeights = []
        for i in range(1,len(layerSizes)):
            auxWeight = np.random.rand(layerSizes[i],layerSizes[i-1])
            absSum = np.sum(np.abs(auxWeight), axis =1)
            for z in range(layerSizes[i]):
                auxWeight[z] *= (delta_vector[i-1]/absSum[z])
            auxWeights.append(torch.tensor(auxWeight, dtype = torch.float))
        toRet.append(auxWeights)
    return toRet

def CreateWeightsGraphicalWay(layerSizes = [50, 50, 50, 1], delta_vector = [5, 5, 50],\
    outgoingEdges = 1, numToCreate = 25, numpy_seed = 1637, double = False):
    def edgesMat(rows, cols, outgoingEdges):
        matToRet = np.zeros((rows, cols))
        indexes = np.random.choice(rows, cols *outgoingEdges, replace = True)
        uniforms = np.random.uniform(0.0, 1.0, size = cols *outgoingEdges) - 0.5
        for j in range(cols):
            matToRet[indexes[j*outgoingEdges:(j+1)*outgoingEdges] , j] = \
                uniforms[j*outgoingEdges:(j+1)*outgoingEdges]
        if double:
            indexes2 = np.random.choice(cols, rows *outgoingEdges, replace = True)
            uniforms2 = np.random.uniform(0.0, 1.0, size = rows *outgoingEdges) - 0.5
            for i in range(rows):
                matToRet[i, indexes2[i*outgoingEdges:(i+1)*outgoingEdges]] = \
                    uniforms2[i*outgoingEdges:(i+1)*outgoingEdges]
        return matToRet
    toRet = []
    np.random.seed(numpy_seed)
    for j in range(numToCreate):
        auxWeights = []
        for i in range(1,len(layerSizes)):
            auxWeight = edgesMat(layerSizes[i],layerSizes[i-1], outgoingEdges)
            absSum = np.sum(np.abs(auxWeight), axis =1)
            for z in range(layerSizes[i]):
                if absSum[z] > 0.0:
                    auxWeight[z] *= (delta_vector[i-1]/absSum[z])
            auxWeights.append(torch.tensor(auxWeight, dtype = torch.float))
        toRet.append(auxWeights)
    return toRet
        

def CreateLayerData(layerSizes = [150, 100, 100, 1], entPerNodeDifFromZero = [50, 50, 50],\
    valDifFromZero = 1, seed_numpy = 54321, torch_seed = 12345):
    retWeights = []
    torch.manual_seed(torch_seed)
    ## Pytorch seems to work with matrices as of dimension (end_nodes, starting_nodes)
    ## For example in a first layer of a mtrix that is just linear the parameter matrix size
    ## would be (first_layer, input_size)
    totalNonZeros = 0
    for i in range(1,len(layerSizes)):
        totalNonZeros += entPerNodeDifFromZero[i-1] * layerSizes[i]
    unifs = np.random.uniform(low =-0.5, high =0.5, size  = totalNonZeros)
    count = 0
    for i in range(1,len(layerSizes)):
        matToWorkWith = torch.zeros(layerSizes[i], layerSizes[i-1])
        for j in range(layerSizes[i]):
            non_zeroIndexes = np.random.choice(a = np.arange(layerSizes[i-1]),\
                size= entPerNodeDifFromZero[i-1], replace = False)
            for index in non_zeroIndexes:
                if unifs[count] >= 0.0:
                    matToWorkWith[j,index] += valDifFromZero
                else:
                    matToWorkWith[j,index] -= valDifFromZero
                count += 1
        retWeights.append(matToWorkWith)
    return retWeights
        
def FixLayerData(net, weights):#, transpose = True)
    with torch.no_grad():
        for i,layer in enumerate(net.children()):
            layerSize = layer.weight.data.size()
            weightSize = weights[i].size()
            if layerSize == weightSize:
                layer.weight.data = weights[i].clone()
            else:
                print('In FixLayerData for Layer '+str(i)+' dimensions do not Match')
        
def CreateRealLabelsAndSNR(X_mat, net, snr_vector = [1, 5, 10], seed_numpy = 255):
    ## First Let's create The Real Labels
    real_labels = 0
    with torch.no_grad():
        real_labels = net(X_mat)
    ## Let's create Labels with Noise
    np.random.seed(seed_numpy)
    labelsWithNoise = []
    for snr in snr_vector:
        labelsWithNoise.append(SNR_labels(np.squeeze(np.array(real_labels)), snr))
    return torch.squeeze(real_labels), labelsWithNoise


def SNR_labels(y_woutn, snr):
    r"""
    y_woutn is the series without noise
    snr is the noise to be applied.. I assume that if we want snr=\infty, i.e., 
        no noise we just don't call this function.
    The equation I used is SNR = \frac{\sigma^2_{signal}}{\sigma^2_{noise}}
    And, then I return as the modified series 
    """
    # np.random.seed(seed_noise)
    std_signal = np.std(y_woutn)
    std_noise = np.sqrt((std_signal**2)/snr)
    noiseVec = np.random.normal(0.0, std_noise, len(y_woutn))
    return torch.tensor(y_woutn + noiseVec, dtype = torch.float)

class CreateArtificialDataset(Dataset):
    """Face Landmarks dataset."""

    def __init__(self, X_Mat, Y_Vec):
        self.X_Mat = X_Mat 
        self.Y_Vec = Y_Vec

    def __len__(self):
        return self.X_Mat.size()[0]

    def __getitem__(self, idx):
        return [self.X_Mat[idx], self.Y_Vec[idx]]

def CreateDataLoader(X_Mat, Y_Vec, batch_size, shuffle=True, use_cuda =  False,\
    useSeqSampler = False, num_samples = 1, increaseRate = 1):
    #device = torch.device("cuda" if use_cuda else "cpu")
    kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}
    dataset = CreateArtificialDataset(X_Mat, Y_Vec)
    if useSeqSampler:
        sampler = LinSeqRandomSampler(dataset, num_samples)
        return torch.utils.data.DataLoader(dataset,\
            shuffle = False, sampler = sampler,  **kwargs)
    else:
        return torch.utils.data.DataLoader(dataset, batch_size = batch_size ,\
            shuffle=True,  **kwargs)