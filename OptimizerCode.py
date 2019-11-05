# # Import Libraries

import numpy as np
import torch
from torch.optim.optimizer import Optimizer
from torch.utils.data import Sampler
from FrankWolfeRelated import awayStep, retFwHat
import sys

# # Different Per Layer Optimizer Option
def OptimSimpleSGD(w, w_grad, listParam, device = 'cpu'):
    ## listParam only contains the lr rate
    #w -= lr*w_grad
    w.add_(-listParam[0], w_grad)

def ThresSGD(w, w_grad, listParam, device = 'cpu'):
    ## Does thresholded SGD
    ## listParam[0] is the linear rate
    ## listParam[1] is the number of elements to keep non-zero per node.

    ## We first find the indexes of the listParams[1] highest elements in 
    ## absolute value of w_grad.  Notice that we don't save the values of 
    ## them (which the topk function also returns) as they should not
    ## be used in absolute value in the SGD step.
    rows = w.size()[0]
    lr, kTop = listParam
    rowsInOrder = (torch.arange(rows)).view(-1, 1).repeat(1, kTop).view(rows*kTop,)
    maxIndexes = (torch.topk(torch.abs(w_grad), k = kTop)[1]).view(rows*kTop,)
    w[rowsInOrder, maxIndexes] -= lr*w_grad[rowsInOrder, maxIndexes]   

def SFW(w, w_grad, listParam, device = 'cpu'): #, perColumn = False
    ''' For Simple Frank Wolfe:
    listParam[0]: delta
    listParam[1]: vecDivC
    listParam[2]: onesVec
    listParams[3]: zeroesVec
    '''
    # First Let's obtain the Frank Wolfe Matrix
    # print('w.size(): '+str(w.size()))
    # print('torch.sum(torch.abs(w), dim =1).size(): '\
    #     +str(torch.sum(torch.abs(w), dim =1).size()))
    # print('Sums per Row w')
    # print(torch.sum(torch.abs(w), dim =1))

    xHat_FW = retFwHat(w, w_grad, listParam[0], device = device)

    # print('xHat_FW.size(): '+str(xHat_FW.size()))
    # print('Sums per Row xHat_FW')
    # print(torch.sum(torch.abs(xHat_FW), dim =1))
    auxMatrix = w - xHat_FW
    ## In gCapTilde there should be a minus, for that reason
    ## I multiply by (w - xHat_FW) instead of (xHat_FW - w)
    gCapTilde = torch.sum(w_grad * auxMatrix, dim = 1)
    # print('len(gCapTilde): '+str(len(gCapTilde)))
    # print('gCapTilde: '+str(gCapTilde))
    alphaFin = torch.max(torch.min(gCapTilde * listParam[1], listParam[2]), listParam[3])
    # print('len(alphaFin): '+str(len(alphaFin)))
    # print('alphaFin: '+str(alphaFin))
    w -= (auxMatrix.t() * alphaFin).t()
    rows = w.size()[0]
    indEqToOne = torch.arange(rows)[alphaFin == 1]
    w[indEqToOne,:] = xHat_FW[indEqToOne,:]

# def AwayPart(w, w_grad, delta, vecDivC, onesMat=0, turn=False):
def AwayPart(w, w_grad, listParam, device = 'cpu'):
    '''
    For the Away Method:
    listParam[0]: delta
    listParam[1]: vecDivC
    listParam[3]: zeroesVec
    '''
    hat_w, alphas = awayStep(w, w_grad, listParam[0], device = device)
    auxMatrix = hat_w - w
    # print('hat_what_w.size(): '+str(hat_w.size()))
    # print('torch.sum(hat_w, dim =1): '+str(torch.sum(hat_w, dim =1)))
    # print('alphas.size(): '+str(alphas.size()))
    ## In aCapTilde there should be a minus, for that reason
    ## I multiply by (hat_w - w) instead of w -hat_w
    aCapTilde = torch.sum(w_grad * auxMatrix, dim = 1) * listParam[1]
    # I need to use max with zeroes as when away decides to not move 
    # in a row says alpha = 0 and hat_w = 0.
    betas = torch.max(torch.min(aCapTilde, alphas), listParam[3])
    # print('betas.size(): '+str(betas.size()))
    w -= (auxMatrix.t() * betas).t()

# def OneSimpleOneAway(w, w_grad, vecDivC, onesMat, delta, turn):
def OneSimpleOneAway(w, w_grad, listParam, device = 'cpu'):
    ''' For Simple Frank Wolfe:
    listParam[0]: delta
    listParam[1]: vecDivC
    listParam[2]: onesVec
    listParams[3]: zeroesVec
    listParams[4]: turn
    '''
    if listParam[4]:
        SFW(w, w_grad, listParam, device = device) 
    else:
        AwayPart(w, w_grad, listParam, device = device)
    listParam[4] =  not listParam[4]
    
# def RunOptimizer(optimizerFN, w, w_grad, *args):
def RunOptimizer(optimizerFN, w, w_grad, listParam, device = 'cpu'):
    with torch.no_grad():
        optimizerFN(w, w_grad, listParam, device  = device)

# # Mix Optimizer Code

class MixOptimizer(Optimizer):

    def __init__(self, params, optimPerLayer=[], partParams=[], device = 'cpu'):
        ## Here I need to make a dynamic way of ensuring
        ## that the names and params are correct.
        ## For now I will just leave like this.

        ## print('Caution: Currently the solvers: SimpleFW, AwayFW, and OneTwoSimpleAway'+\
        ##    ' only work with matrix parameters. For tensors of more than'+\
        ##    ' two dimensions please select SGD for now.')
        self.__paramPerLayer = partParams
        self.__optimPerLayer = []
        self.device = device

        ## Now we set each 
        for name_optim in optimPerLayer:
            if name_optim == 'None':
                self.__optimPerLayer.append('None')
            elif name_optim == "SGD":
                self.__optimPerLayer.append(OptimSimpleSGD)
            elif name_optim == "SGDThres":
                self.__optimPerLayer.append(ThresSGD)
            elif name_optim == "SFW":
                self.__optimPerLayer.append(SFW)
            elif name_optim == "AwayFW":
                self.__optimPerLayer.append(AwayPart)
            elif name_optim == "OneTwo":
                self.__optimPerLayer.append(OneSimpleOneAway)
            else:
                print('Optimizer of name: '+name_optim+ 'is not recognized')
                sys.exit('')

        ## I never use the defaults because I have different configurations
        ## per layer. Still, I can set set the weights of the net 
        ## to fix values by using net.load_state_dict.. check if this is true.
        defaults = dict(optimPerLayer=self.__optimPerLayer, \
                        partParams=self.__paramPerLayer)
        super(MixOptimizer, self).__init__(params, defaults)

# def __setstate__(self, state):
#     super(MixOptimizer, self).__setstate__(state)
#         for group in self.param_groups:
#             group.setdefault('amsgrad', False)

    @staticmethod
    def checkDimensions(params, optimPerLayer=[], partParams=[]):
        ## First check that optimPerLayer and partParams len 
        ## matches the number of params.
        paramsLen = sum(1 for _ in params)
        if len(optimPerLayer) != paramsLen:
            print('Error Found: ')
            print('The number of parameters for which you need to specify a solver is ' \
                +str(paramsLen)+', while you only gave the name of '\
                +str(len(optimPerLayer)) )
            print('You can run the function PrintModelStructure to see the model parameters sizes')
        elif len(partParams) != paramsLen:
            print('Error Found: ')
            print('The number of parameters for which you need to specify a solver is '\
                +str(paramsLen)+', while you only specify the parameters of '\
                +str(len(partParams)) )
            print('You can run the function PrintModelStructure to see the model parameters sizes')
        else:
            print('Dimensions Match')
        print()

    def step(self, closure=None):
        """Performs a single optimization step.
        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            loss = closure()

        group = self.param_groups[0]
        for i, p  in enumerate(group['params']):
            if p.grad is None:
                continue
            # print(self.__optimPerLayer[i])
            ## I wanted to use *self.__paramPerLayer[i] and then work 
            ## everything with *args, but I may need to keep track in changes in the
            ## parameters (e.g. turn in OneSimpleOneAway) so I decided to 
            ## to pass the list itself.
            if self.__optimPerLayer[i] != 'None' and p.grad is not None:  
                RunOptimizer(self.__optimPerLayer[i], p.data, \
                        p.grad.data, self.__paramPerLayer[i], device = self.device)
            # if self.__optimPerLayer[i] != 'None' and p.grad is not None:  
            #     RunOptimizer(self.__optimPerLayer[i], p.data, \
            #             p.grad.data, *self.__paramPerLayer[i])
        return loss


class LinSeqRandomSampler(Sampler):
    r"""Samples randomly a number of samples at each iteration. It starts by sampling 'num_samples'
    given by the user and at each iteration it increases the number of samples to use in the next iteration
    in 1 (could be extended). This method is called Linear Sequential .. (LinSeq) instead of just 
    Seq.. as SequentialRandomSampler already exists in Pytorch. 
    Arguments:
        data_source (Dataset): dataset to sample from
        init_num_samples (int): initial number of samples to draw, default=1
        increaseRate (int): Rate at which we increase the number of samples taken at each iteration
    """

    def __init__(self, data_source, init_num_samples = 0):
        self.data_source = data_source
        self.num_samples = init_num_samples
        self.init_num_samples = init_num_samples

        if not isinstance(self.num_samples, int) or self.num_samples < 0:
            raise ValueError("num_samples should be a non-negative integer "
                             "value, but got init_num_samples={}".format(self.num_samples))

    # def __iter__(self):
    #     n = len(self.data_source)
    #     self.num_samples += 1
    #     return iter(torch.randint(high=n, size=(self.num_samples,), dtype=torch.int64).tolist()) 

    # def __len__(self):
    #     return self.num_samples+1

    def __iter__(self):
        n = len(self.data_source)
        numOfIte = self.__len__()
        listToRet = []
        # print('len(self.data_source): '+str(len(self.data_source)))
        for i in range(numOfIte):
            listToRet.append(torch.randint(high=n, size=(self.num_samples,), dtype=torch.int64).tolist()) 
            self.num_samples += 1
        return iter(listToRet)

    def __len__(self):
        auxTermInSqrt = 2*len(self.data_source) + (self.num_samples-1)*(self.num_samples)
        return int(np.sqrt(auxTermInSqrt) - 1) - self.num_samples
