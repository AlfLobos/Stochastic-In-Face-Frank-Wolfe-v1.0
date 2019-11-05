import torch
import torch.nn as nn
import torch.nn.functional as F

class SynRELU(nn.Module):
    def __init__(self, input_size, first_layer, second_layer, third_layer):
        super(SynRELU, self).__init__()
        self.lin1 = nn.Linear(input_size, first_layer, bias = False)
        self.lin2 = nn.Linear(first_layer, second_layer, bias = False)
        self.lin3 = nn.Linear(second_layer, third_layer, bias = False)

    def forward(self, x):
        x = F.relu(self.lin1(x))
        x = F.relu(self.lin2(x))
        x = self.lin3(x)
        return x

class SynSigmoid(nn.Module):
    def __init__(self, input_size, first_layer, second_layer, third_layer):
        super(SynSigmoid, self).__init__()
        self.lin1 = nn.Linear(input_size, first_layer, bias = False)
        self.lin2 = nn.Linear(first_layer, second_layer, bias = False)
        self.lin3 = nn.Linear(second_layer, third_layer, bias = False)

    def forward(self, x):
        x = torch.sigmoid(self.lin1(x))
        x = torch.sigmoid(self.lin2(x))
        x = self.lin3(x)
        return x