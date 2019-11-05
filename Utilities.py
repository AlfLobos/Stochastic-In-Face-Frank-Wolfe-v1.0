# # Import Libraries
import numpy as np
import torch
import torch.nn.functional as F
#
#

## General function for all experiments.

def CreateWeightsGraphical(fnNet, indToInit, delta_vector , paramsFnNet = None, double = True,\
    outgoingEdges =1, typeTen = torch.float, torchSeed = None, npSeed = None):
    r"""
    This function returns a  list of lists of weights which are initilized using our graphical way.
    An element of this list is a list of weights that matches the size and elements when we run
    net.parameters(). Not all parameters are obligated to be initialized using our graphical 
    initialization, and those who don't are just initialized to the std. Pytorch initialization
    (the default call by Pytorch when we call net = Net()).
    [0] fnNet (Net): Network architecture of interest.
    [1] indToInit (list): List with the position of the parameters we eventually want to initiliaze
        using our grahical way.
    [2] delta_vector (list): List of float numbers representing the value of delta for the layers that
        we want to initialize. The size of this vector must match the size of indToInit.
    [3] numToCreate (int): Number of initializations to create.
    [4] outgoingEdges (int): Minimum number of non-zero edges going from a node in the starting layer
        to the next layer. Notice that then it can occur that nodes in the next layer do not have any
        non-zer incoming edge.
    [5] double (boolean): If True we obligate that each node in the `next' layer has at least one 
        non-zero edge connected to it. 
    [6] pytorchSeed (int): Seed for reproducibility.
    [7] typeTen (torch.object): Type of torch tensor that each weight will be. Default is torch.float.
    """
    if torchSeed:
        torch.manual_seed(torchSeed)
    if npSeed:
        np.random.seed(npSeed)
    def edgesMat(rows, cols, outgoingEdges, double):
        weight = torch.zeros(rows, cols)
        unifs = torch.rand(rows * outgoingEdges)
        nzPos = np.zeros((rows, outgoingEdges))
        for j in range(rows):
            nzPos[j,:] = np.random.choice(a = np.arange(cols), size= outgoingEdges, replace = False)
        # if rows>1:
        #     for j in range(rows):
        #         nzPos[j,:] = np.random.choice(a = np.arange(cols), size= outgoingEdges, replace = False)
        # else:
        #     for j in range(rows):
        #         nzPos[j,:] = [0]
        nzEntriesInOrder = (torch.from_numpy(nzPos)).view(rows*outgoingEdges,).long()
        rowsInOrder = (torch.arange(rows)).view(-1, 1).repeat(1, outgoingEdges).view(rows*outgoingEdges,)
        weight[rowsInOrder, nzEntriesInOrder] = ((unifs>=0.5).float() - (unifs<0.5).float()) 
        if double and cols> 1:
            unifs = torch.rand(cols * outgoingEdges)
            nzPos = np.zeros((cols, outgoingEdges))
            for j in range(cols):
                nzPos[j,:] = np.random.choice(a = np.arange(rows), size= outgoingEdges, replace = False)
            nzEntriesInOrder = (torch.from_numpy(nzPos)).view(cols*outgoingEdges,).long()
            colsInOrder = (torch.arange(cols)).view(-1, 1).repeat(1, outgoingEdges).view(cols*outgoingEdges,)
            weight[nzEntriesInOrder, colsInOrder] = ((unifs>=0.5).float() - (unifs<0.5).float()) 
        ## else: Do nothing as the condition is already satisfied. weight.size() is [rows, 1]
        return weight
    toRet = []
    with torch.no_grad():
        net = 0
        if paramsFnNet is not None:
            net = fnNet(*paramsFnNet)
        else:
            net = fnNet()
        toRet = []
        for i, weight in enumerate(net.parameters()):
            if i in indToInit:
                rows, cols = weight.data.size()
                toRet.append(edgesMat(rows, cols, outgoingEdges, double))
            else:
                toRet.append(weight.clone())
    return toRet


def createWeightsPytorchWay(fnNet, delta_vector = None, paramsFnNet = None,\
    torchSeed = None, indToScale= None):
    if torchSeed:
        torch.manual_seed(torchSeed)
    net = 0
    if paramsFnNet:
        net = fnNet(*paramsFnNet)
    else:
        net = fnNet()
    if indToScale is not None:
        ScaleParameters(net, indToScale, delta_vector)
    listToRet = []
    for param in net.parameters():
        listToRet.append(param.data.clone())
    return listToRet


def FixLayerData(net, weights):
    r"""
    Function that copies into net.parameters() the tensors from the list of weights `weights'
    """
    with torch.no_grad():
        for i, param in enumerate(net.parameters()):
            paramSize = param.data.size()
            weightSize = weights[i].size()
            if paramSize == weightSize:
                param.data[:] = weights[i]
            else:
                print('In FixLayerData for parameter '+str(i)+' dimensions do not Match')

def ScaleParameters(net, indToScale, delta_vector):
    r"""
    Function that scale the parameters data of some of net.parameters(). The parameters to be scaled
    are the ones whose index appear in `indToScale' and they are scaled to the values that appear in
    the list `delta' (the size of delta should be bigger or equal than indToScale).
    """
    i = 0
    for index, param in enumerate(net.parameters()):
        if index in indToScale:
            l1Norm = torch.sum(torch.abs(param.data), dim =1)
            rows, _ = param.data.size()
            with torch.no_grad():
                for j in range(int(rows)):
                    param.data[j,:] *= delta_vector[i]/l1Norm[j]
            i += 1

            

def PrintModelStructure(net):
    print('When printing model.children() we see that the modules of the model are:')
    for layer in net.children():
        print(layer)
    print()
    print('In order the size of the parameters of our model are'+\
         ' (info obtained running net.parameters())')
    for i,p in enumerate(net.parameters()):
        print(str(i),end=':  ')
        print(p.data.size())