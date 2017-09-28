from __future__ import print_function
import torch
import torch.nn as nn

class LeNet_5(nn.Module):
    def __init__(self):
        super(LeNet_5, self).__init__()
        self.conv1 = nn.Conv2d(1, 20, kernel_size=5, stride=1)
        self.relu_conv1 = nn.ReLU(inplace=True)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(20, 50, kernel_size=5, stride=1)
        self.relu_conv2 = nn.ReLU(inplace=True)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.ip1 = nn.Linear(50*4*4, 500)
        self.relu_ip1 = nn.ReLU(inplace=True)
        self.ip2 = nn.Linear(500, 10)
        return

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu_conv1(x)
        x = self.pool1(x)
        x = self.conv2(x)
        x = self.relu_conv2(x)
        x = self.pool2(x)

        x = x.view(x.size(0), 50*4*4)

        x = self.ip1(x)
        x = self.relu_ip1(x)
        x = self.ip2(x)
        return x
