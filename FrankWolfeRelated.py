# # Import Libraries
import torch

#
# 
## The following function is used inside in the function SimpleFW in OptimizerCode.py

# Given that I'm using the nn.module I need to use dim = 1 and w[i, indices_max[i]] and not w[indices_max[i], i]  
def retFwHat(w, w_grad, delta, perColumn = False, device = 'cpu'):
    '''
    This is the most simple FW step for one layer at a time
    assuming the one norm per row to be <= delta
    '''
    dim_0, dim_1 = w.size()
    hatFW = torch.zeros(dim_0, dim_1).to(device)
    rowsInOrder = torch.arange(dim_0)
    argMaxs = torch.argmax(torch.abs(w_grad),dim=1)
    signs = (torch.sign(w_grad)[rowsInOrder,argMaxs]).to(device)
    hatFW[rowsInOrder,argMaxs] = - signs * delta + (signs == 0).float()* w[rowsInOrder,argMaxs]
    return hatFW

## Away Step And Related Functions

def alphaNotInBound(norm1_xVec, x_kstarVec, abs_x_kstarVec, deltaVec, signW_gradVec):
    a = deltaVec - norm1_xVec + abs_x_kstarVec + x_kstarVec*signW_gradVec
    b = norm1_xVec + deltaVec - abs_x_kstarVec - x_kstarVec*signW_gradVec
    return a/b

def alphaInBoundary(abs_CoordVec, deltaVec):
    return (abs_CoordVec/(deltaVec - abs_CoordVec))

def awayStep(w, w_grad, delta, tol = 0.000001, device = 'cpu'):
     ## hat_w is the hat_x_k that appears in step 2 of Algorithm 2
    rows, cols = w.size()
    hat_w = torch.zeros(*w.size()).to(device)
    # w_AsNumpy = w.numpy()
    abs_w_grad = torch.abs(w_grad).to(device)
    ## We calculate the l1 norm for each node
    l1_norms = torch.sum(torch.abs(w), dim = 1)
    ## We will have alphas per node.
    alphas = torch.zeros(rows).to(device)
    ## Nodes that are not tight in l1 Norm
    noEq_indexes = (torch.arange(rows)[l1_norms < delta-tol]).to(device)
    ## Sign Matrices
    signMat_w = torch.sign(w)
    ## The following matrix is used only in the boundary cases.
    ## The following vector helps the nonEq case (is used only in
    ## "for index in noEq_indexes:")
    argMaxNoEqCase = torch.argmax(abs_w_grad, dim =1)
    
    ## No Equality First
    posMaxNoEq = argMaxNoEqCase[noEq_indexes]
    wNoEq = w[noEq_indexes, posMaxNoEq]
    signW_NoEq = torch.sign(wNoEq).float() 
    signW_gradNoEq = torch.sign(w_grad[noEq_indexes, posMaxNoEq]).float()
    deltaToUseNoEq = delta * signW_gradNoEq
    
    hat_w[noEq_indexes, posMaxNoEq] = deltaToUseNoEq
    alphas[noEq_indexes] = alphaNotInBound(l1_norms[noEq_indexes], wNoEq,\
            wNoEq * signW_NoEq, delta*torch.ones(len(noEq_indexes)).to(device),  signW_gradNoEq)
    
    ## For Boundary Cases 
    ## Nodes that are tight in the l1 norm
    helperMat = signMat_w * w_grad
    eq_indexes = torch.arange(rows)[l1_norms >= delta-tol]
    minW_Grad = -(torch.max(abs_w_grad) + 1) 
    helperMat[signMat_w == 0] = minW_Grad
    argMaxEqCase = torch.argmax(helperMat, dim =1)
    wEqFull = w[eq_indexes, argMaxEqCase[eq_indexes]]
    
    # The boundary case has the following peculiarity. If w is equal to zero then we should not 
    # move in that node. If the value of w in absolute value is equal to delta, then we should 
    # not move in a face step, and if it is greater than delta it could be a numerical error.
    relIndValid = (wEqFull != 0) & (torch.abs(wEqFull) < delta )
    eq_indexesValid = eq_indexes[relIndValid]
    posMaxEqValid = argMaxNoEqCase[eq_indexesValid]
    wEqValid = wEqFull[relIndValid]
    signWEqValid = torch.sign(wEqValid).float()
    alphas[eq_indexesValid] = alphaInBoundary(wEqValid*signWEqValid, l1_norms[eq_indexesValid])
    hat_w[eq_indexesValid, posMaxEqValid] = delta *  signWEqValid
    return hat_w, alphas  