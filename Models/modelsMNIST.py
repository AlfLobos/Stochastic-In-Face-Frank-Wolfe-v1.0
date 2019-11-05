import torch
import torch.nn as nn
import torch.nn.functional as F

class MNIST_Conv(nn.Module):
    def __init__(self):
        super(MNIST_Conv, self).__init__()
        self.conv1 = nn.Conv2d(1, 20, 5, 1)
        self.conv2 = nn.Conv2d(20, 50, 5, 1)
        self.fc1 = nn.Linear(4*4*50, 500)
        self.fc2 = nn.Linear(500, 10)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2, 2)
        x = x.view(-1, 4*4*50)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)

class MNIST_MLP(nn.Module):
    def __init__(self):
        super(MNIST_MLP, self).__init__()
        self.ln1 = nn.Linear(784, 512)
        self.ln2 = nn.Linear(512, 512)
        self.ln3 = nn.Linear(512, 10)

    def forward(self, x):
        x = F.relu(self.ln1(x.view(-1, 784)))
        x = F.relu(self.ln2(x))
        return F.log_softmax(self.ln3(x), dim=1)