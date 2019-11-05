import numpy as np
import pickle
import torch
from torchvision import datasets, transforms
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import normalize
import torch.nn as nn
import torch.nn.functional as F
import os

## The following functions are used in the CreateDataSynthetic.py

def SNR_labels(y_woutn, snr):
    r"""
    y_woutn is the series without noise
    snr is the noise to be applied.. I assume that if we want snr=\infty, i.e., 
        no noise we just don't call this function.
    The equation I used is SNR = \frac{\sigma^2_{signal}}{\sigma^2_{noise}}
    And, then I return as the modified series 
    """
    std_signal = np.std(y_woutn)
    std_noise = np.sqrt((std_signal**2)/snr)
    noiseVec = np.random.normal(0.0, std_noise, len(y_woutn))
    return torch.tensor(y_woutn + noiseVec, dtype = torch.float)

def CreateRealLabelsAndSNR(X_mat, net, snr_vector = [1, 5, 10], torchSeed = None):
    r"""
    Here I return a list with elements labels under the SNR levels given in snr_vector and as the last element
    the ``real labels'' (SNR = Infty), i.e., the labels obtained when I do 'real_labels = net(X_mat)'.
    """
    if torchSeed:
        torch.manual_seed(torchSeed)
    real_labels = 0
    with torch.no_grad():
        real_labels = net(X_mat)
    ## Let's create Labels with Noise
    labelsToRet = []
    for snr in snr_vector:
        labelsToRet.append(SNR_labels(np.squeeze(np.array(real_labels)), snr))
    labelsToRet.append(torch.squeeze(real_labels))
    return labelsToRet

def CreateStartingWeight(nzEntries, rows, cols, valDifFromZero):
    r"""
    This function returns a matrix that has nzEntries non-zero entries per row. Each of these
    non-zero entries take the value valDifFromZero or -valDifFromZero with equal probability.
    The position of the non-zero entries is obtained using np.random.choice without replacement.
    """
    weight = torch.zeros(rows, cols)
    unifs = torch.rand(rows * nzEntries)
    nzPos = np.zeros((rows, nzEntries))
    for j in range(rows):
        nzPos[j,:] = np.random.choice(a = np.arange(cols), size= nzEntries, replace = False)
    nzEntriesInOrder = (torch.from_numpy(nzPos)).view(rows*nzEntries,).long()
    rowsInOrder = (torch.arange(rows)).view(-1, 1).repeat(1, nzEntries).view(rows*nzEntries,)
    weight[rowsInOrder, nzEntriesInOrder] = ((unifs>=0.5).float() - (unifs<0.5).float())*valDifFromZero 
    return weight

def CreateStartingLayerData(layerSizes = [150, 100, 100, 1], entPerNodeDifFromZero = [50, 50, 1],\
    valDifFromZero = 1, torchSeed = None, npSeed = None):
    r"""
    Write Sthg.
    """
    if torchSeed:
        torch.manual_seed(torchSeed)
    if npSeed:
        np.random.seed(npSeed)
    retWeights = []
    ## Pytorch works with weights of dimension (end_nodes, starting_nodes)
    ## For example in a first layer of a matrix that is just linear the parameter matrix size
    ## would be (first_layer, input_size)
    for i in range(0,len(layerSizes)-1):
        rows = layerSizes[i+1]
        cols = layerSizes[i]
        nzEntries = entPerNodeDifFromZero[i]
        retWeights.append(CreateStartingWeight(nzEntries, rows, cols, valDifFromZero))
    return retWeights

def CreateFeatures(allDataPoints, input_layer,  mu, stdNormal, seed_numpy):
    np.random.seed(seed_numpy)
    X =  np.random.normal(loc = mu, scale = stdNormal,\
        size = (allDataPoints, input_layer))
    return torch.tensor(normalize(X), dtype = torch.float)

class CreateArtificialDataset(Dataset):
    """Face Landmarks dataset."""

    def __init__(self, X_Mat, Y_Vec):
        self.X_Mat = X_Mat 
        self.Y_Vec = Y_Vec

    def __len__(self):
        return self.X_Mat.size()[0]

    def __getitem__(self, idx):
        return [self.X_Mat[idx], self.Y_Vec[idx]]

def CreateDataLoader(X_Mat, labels, batch_size, shuffle=True, use_cuda =  False,\
    useSeqSampler = False, num_samples = 1, increaseRate = 1):
    #device = torch.device("cuda" if use_cuda else "cpu")
    kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}
    dataset = CreateArtificialDataset(X_Mat, labels)
    return torch.utils.data.DataLoader(dataset, batch_size = batch_size , shuffle=True,  **kwargs)