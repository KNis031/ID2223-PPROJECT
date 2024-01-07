import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
import matplotlib.pyplot as plt

class ConvNet(nn.Module):
    def __init__(self, num_class=1000):
        super(ConvNet, self).__init__()

        # init bn
        self.bn_init = nn.BatchNorm2d(1)

        # layer 1
        self.conv_1 = nn.Conv2d(1, 64, 3, padding=1)
        self.bn_1 = nn.BatchNorm2d(64)
        self.mp_1 = nn.MaxPool2d(2) # 48

        # layer 2
        self.conv_2 = nn.Conv2d(64, 128, 3, padding=1)
        self.bn_2 = nn.BatchNorm2d(128)
        self.mp_2 = nn.MaxPool2d(2) # 24

        # layer 3
        self.conv_3 = nn.Conv2d(128, 256, 3, padding=1)
        self.bn_3 = nn.BatchNorm2d(256)
        self.mp_3 = nn.MaxPool2d(2) # 12

        # layer 4
        self.conv_4 = nn.Conv2d(256, 256, 3, padding=1)
        self.bn_4 = nn.BatchNorm2d(256)
        self.mp_4 = nn.MaxPool2d(3) # 4
        # flatten (256x4x4)

        # # classifier
        self.dense_1 = nn.Linear(4096, 4096)
        self.dense_2 = nn.Linear(4096, num_class)
        self.dropout = nn.Dropout(0.25)

    def forward(self, x):
        x = x.unsqueeze(1)
        # print(x.size())

        # init bn
        x = self.bn_init(x)

        # layer 1
        x = self.mp_1(nn.ReLU()(self.bn_1(self.conv_1(x))))
        # print(x.size())

        # layer 2
        x = self.mp_2(nn.ReLU()(self.bn_2(self.conv_2(x))))
        # print(x.size())

        # layer 3
        x = self.mp_3(nn.ReLU()(self.bn_3(self.conv_3(x))))
        # print(x.size())

        # layer 4
        x = self.mp_4(nn.ReLU()(self.bn_4(self.conv_4(x))))
        # print(x.size())

        # classifier
        x = x.view(x.size(0), -1)
        x = self.dropout(x)

        x = nn.ReLU()(self.dense_1(x))
        logit = nn.Sigmoid()(self.dense_2(x))

        return logit
