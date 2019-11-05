# # Import Libraries

import numpy as np
import torch
from torch.optim.optimizer import Optimizer
import torch.nn.functional as F
import pickle

## train, val, and test.

def trainSGDT(net, train_loader, val_loader, optimizer, lossFn, init_epoch, final_epoch, name,\
    paramIndToThres, nzEntriesVec, device = 'cpu', checkForDiv = False, tol = 1000000, \
    saveNetDict = False,  pathToSave = None,  saveAccToo = False):
    net.train()
    epoch = init_epoch
    toLossesAndAcc = []
    while epoch <final_epoch:
        for _, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = torch.squeeze(net(data)).to(device)
            loss = lossFn(output, target)
            loss.backward()
            optimizer.step()
            with torch.no_grad():
                auxCounter = 0
                for i, param in enumerate(net.parameters()):
                    if i in paramIndToThres:
                        rows, cols = param.data.size()
                        nzEntries = nzEntriesVec[auxCounter]
                        topK_1 = torch.topk(torch.abs(param.data), k = nzEntries, dim = 1)[1]
                        arangeIndexes = (torch.arange(rows)).view(-1, 1).repeat(1, nzEntries).view(rows*nzEntries,)
                        nonZeroIndexes = torch.squeeze(topK_1.view(-1,1)).long()
                        matToMultiply = torch.zeros(rows, cols)
                        matToMultiply[arangeIndexes, nonZeroIndexes] = 1
                        param.data *= matToMultiply
                        auxCounter +=1
        epoch += 1
        epochBeforeChecking = epoch
        if checkForDiv:
            with torch.no_grad():
                for param in net.parameters():
                    if torch.max(torch.abs(param.data)) > tol:
                        print(name+ ' diverged at epoch '+str(epochBeforeChecking))
                        epoch = final_epoch + 1
        with torch.no_grad():
            if saveNetDict:
                torch.save(net.state_dict(), pathToSave + name +'_Epoch_'+str(epochBeforeChecking)+'.pt')
            if epoch!= (final_epoch + 1):
                if saveAccToo:
                    lossTr, accTr = val(net, lossFn, train_loader, device = device, withAcc = saveAccToo)
                    lossTest, accTest = val(net, lossFn, val_loader, device = device, withAcc = saveAccToo)
                    toLossesAndAcc.append([epochBeforeChecking, str(lossTr), str(accTr), str(lossTest), str(accTest)])
                else:
                    toLossesAndAcc.append([str(epochBeforeChecking), str(val(net, lossFn, train_loader, device = device, withAcc = saveAccToo)),\
                        str(val(net, lossFn, val_loader, device = device, withAcc = saveAccToo))])
            else:
                if saveAccToo:
                    toLossesAndAcc.append([str(epochBeforeChecking), str(np.nan), str(np.nan), str(np.nan), str(np.nan)])
                else:
                    toLossesAndAcc.append([str(epochBeforeChecking), str(np.nan), str(np.nan)])
    return toLossesAndAcc
        
def train(net, train_loader, val_loader, optimizer, lossFn, init_epoch, final_epoch, \
    name, device = 'cpu', checkForDiv = False, tol = 1000000, saveNetDict = False, \
    pathToSave = None,  saveAccToo = False):
    net.train()
    epoch = init_epoch
    toLossesAndAcc = []
    while epoch <final_epoch:
        for _, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = torch.squeeze(net(data)).to(device)
            loss = lossFn(output, target)
            loss.backward()
            optimizer.step()
        epoch += 1
        epochBeforeChecking = epoch
        if checkForDiv:
            with torch.no_grad():
                for param in net.parameters():
                    if torch.max(torch.abs(param.data)) > tol:
                        print(name+ ' diverged at epoch '+str(epochBeforeChecking))
                        epoch = final_epoch + 1
        with torch.no_grad():
            if saveNetDict:
                torch.save(net.state_dict(), pathToSave + name +'_Epoch_'+str(epochBeforeChecking)+'.pt')
            if epoch!= (final_epoch + 1):
                if saveAccToo:
                    lossTr, accTr = val(net, lossFn, train_loader, device = device, withAcc = saveAccToo)
                    lossTest, accTest = val(net, lossFn, val_loader, device = device, withAcc = saveAccToo)
                    toLossesAndAcc.append([epochBeforeChecking, str(lossTr), str(accTr), str(lossTest), str(accTest)])
                else:
                    toLossesAndAcc.append([str(epochBeforeChecking), str(val(net, lossFn, train_loader, device = device, withAcc = saveAccToo)),\
                        str(val(net, lossFn, val_loader, device = device, withAcc = saveAccToo))])
            else:
                if saveAccToo:
                    toLossesAndAcc.append([str(epochBeforeChecking), str(np.nan), str(np.nan), str(np.nan), str(np.nan)])
                else:
                    toLossesAndAcc.append([str(epochBeforeChecking), str(np.nan), str(np.nan)])
    return toLossesAndAcc
    

def val(net, lossFn, val_loader, device = 'cpu', withAcc = True):
    net.eval()
    val_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in val_loader:
            data, target = data.to(device), target.to(device)
            output = torch.squeeze(net(data)).to(device)
            val_loss += lossFn(output, target).item() # sum up batch loss
            if withAcc:
                pred = output.argmax(dim=1, keepdim=True) # get the index of the max log-probability
                correct += pred.eq(target.view_as(pred)).sum().item()

    val_loss /= len(val_loader.dataset)
    if withAcc:
        return [val_loss, 100. * correct / len(val_loader.dataset)]
    else:
        return val_loss

def test(net, lossFn, test_loader, device = 'cpu'):
    net.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = torch.squeeze(net(data)).to(device)
            test_loss += lossFn(output, target).item() # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True) # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)
    return [test_loss, 100. * correct / len(test_loader.dataset)]