import torch
import torch.nn as nn
import torch.nn.functional as F


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        # 3 input image channel, 6 output channels, 5x5 square convolution
        # kernel
        self.conv1_3c = nn.Conv2d(3, 20, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(20, 50, 5)
        # an affine operation: y = Wx + b
        self.fc1_ = nn.Linear(50 * 5 * 5, 120)  # from image dimension
        self.fc2 = nn.Linear(120, 120)
        self.fc3 = nn.Linear(120, 10)

    def forward(self, x):
        # Max pooling over a (2, 2) window
        x = self.pool(F.relu(self.conv1_3c(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 50 * 5 * 5)
        x = F.relu(self.fc1_(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x
