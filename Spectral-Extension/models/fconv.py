import torch
from torch import nn
from torch.nn import functional as F
from torch.autograd import Variable

import numpy as  np


class FConv(nn.Module):
    def __init__(self, input_size=(6, 128, 128)):
        super(FConv, self).__init__()
        output_size=(10,128,128)

        self.input_size = input_size
        self.output_size = output_size
        #convolutions
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=self.input_size[0],
                     out_channels=int(self.input_size[0]/2),
                     kernel_size=7,
                     stride=1,
                     padding='same',
                     padding_mode='reflect'),
            nn.BatchNorm2d(int(self.input_size[0]/2)),
            nn.LeakyReLU(0.2, inplace=True))
        self.conv2 = nn.Sequential(    
            nn.Conv2d(in_channels=int(self.input_size[0]/2),
                     out_channels=int(self.output_size[0]/2),
                     kernel_size=5,
                     stride=1,
                     padding='same',
                     padding_mode='reflect'),
            nn.BatchNorm2d(int(self.output_size[0]/2)),
            nn.LeakyReLU(0.2, inplace=True))
        self.conv3 = nn.Sequential(
            nn.Conv2d(in_channels=int(self.output_size[0]/2),
                     out_channels=self.output_size[0],
                     kernel_size=7,
                     stride=1,
                     padding='same',
                     padding_mode='reflect'),
            nn.BatchNorm2d(self.output_size[0]),
            nn.Sigmoid()
            
        )

    def forward(self, x):
       
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        return x


