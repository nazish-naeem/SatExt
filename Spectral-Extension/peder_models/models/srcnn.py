import numpy as np
import torch
import torch.optim as optim

from torch import nn
from math import log10, sqrt


def initialize_weight(weight_list):    
    for wgt in weight_list:
        nn.init.xavier_normal_(wgt.weight, gain=0.2)
        #nn.init.xavier_uniform_(wgt.weight, gain=0.2)

        
        
class Color_Transform_SRCNN(nn.Module):
        
        
    def __init__(self, num_in_band, num_out_band):
        super(Color_Transform_SRCNN, self).__init__() 
        self.conv1 = nn.Conv2d(num_in_band, 64, kernel_size = 9, padding = 9 // 2, bias = True)
        self.conv2 = nn.Conv2d(64, 32, kernel_size = 5, padding = 5 // 2, bias = True)
        self.conv3 = nn.Conv2d(32, num_out_band, kernel_size = 5, padding=5 // 2, bias = True)
        self.relu = nn.ReLU(inplace=True)
        self.lrelu = nn.LeakyReLU(0.2, inplace=True)
        initialize_weight([self.conv1, self.conv2, self.conv3])


    def forward(self, x):
        x = self.lrelu(self.conv1(x))
        x = self.lrelu(self.conv2(x))
        x = self.conv3(x)
        return x

    
    
class Color_Transform_SRCNN2(nn.Module):
    
    
    def __init__(self, num_in_band, num_out_band):
        super(Color_Transform_SRCNN2, self).__init__() 
        self.conv1 = nn.Conv2d(num_in_band, 64, kernel_size = 9, padding = 9 // 2, bias = True)
        self.conv2 = nn.Conv2d(64, 32, kernel_size = 5, padding = 5 // 2, bias = True)
        self.conv3 = nn.Conv2d(32, 32, kernel_size = 5, padding = 5 // 2, bias = True)
        self.conv4 = nn.Conv2d(32, num_out_band, kernel_size = 5, padding=5 // 2, bias = True)
        self.relu = nn.ReLU(inplace=True)
        self.lrelu = nn.LeakyReLU(0.2, inplace=True)
        initialize_weight([self.conv1, self.conv2, self.conv3, self.conv4])

        
    def forward(self, x):
        x = self.lrelu(self.conv1(x))
        x = self.lrelu(self.conv2(x))
        x = self.lrelu(self.conv3(x))
        x = self.conv4(x)
        return x
    
    
class Color_s2_abus_SRCNN_v1(nn.Module):
    
    
    def __init__(self, num_in_band, num_out_band):
        super(Color_s2_abus_SRCNN_v1, self).__init__() 
        self.conv1 = nn.Conv2d(num_in_band, 80, kernel_size = 9, padding = 9 // 2, bias = True)
        self.conv2 = nn.Conv2d(80, 40, kernel_size = 5, padding = 5 // 2, bias = True)
        self.conv3 = nn.Conv2d(40, num_out_band, kernel_size = 5, padding=5 // 2, bias = True)
        self.relu = nn.ReLU(inplace=True)
        self.lrelu = nn.LeakyReLU(0.2, inplace=True)
        initialize_weight([self.conv1, self.conv2, self.conv3])

        
    def forward(self, x):
        x = self.lrelu(self.conv1(x))
        x = self.lrelu(self.conv2(x))
        x = self.conv3(x)
        return x
##########################################################################################################

class Coreg_SRCNN(nn.Module):
    
    
    def __init__(self, num_in_band, num_out_band): 
        super(Coreg_SRCNN, self).__init__() 
        self.conv1 = nn.Conv2d(num_in_band, 80, kernel_size = 9, padding = 9//2, bias = True)
        self.conv2 = nn.Conv2d(80, 60, kernel_size = 9, padding = 9//2, bias = True)
        self.conv3 = nn.Conv2d(60, 40,kernel_size = 5, padding= 5//2, bias = True)
        self.conv4= nn.Conv2d(40, num_out_band, kernel_size = 5, padding= 5//2, bias = True)
        self.relu = nn.ReLU(inplace=True)
        self.lrelu = nn.LeakyReLU(0.2, inplace=True)
        initialize_weight([self.conv1, self.conv2, self.conv3, self.conv4])

        
    def forward(self, x):
        x = self.lrelu(self.conv1(x))
        x = self.lrelu(self.conv2(x))
        x = self.lrelu(self.conv3(x))
        x = self.conv4(x)
        return x
    
    
    
class Coreg_SRCNN_v1(nn.Module):
    
    
    def __init__(self, num_in_band, num_out_band):
        super(Coreg_SRCNN_v1, self).__init__() 
        self.conv1 = nn.Conv2d(num_in_band, 80, kernel_size = 9, padding = 9 // 2, bias = True)
        self.conv2 = nn.Conv2d(80, 40, kernel_size = 5, padding = 5 // 2, bias = True)
        self.conv3 = nn.Conv2d(40, 20, kernel_size = 5, padding=5 // 2, bias = True)
        self.conv4 = nn.Conv2d(20, num_out_band, kernel_size = 5, padding=5 // 2, bias = True)
        self.relu = nn.ReLU(inplace=True)
        self.lrelu = nn.LeakyReLU(0.2, inplace=True)
        initialize_weight([self.conv1, self.conv2, self.conv3, self.conv4])

        
    def forward(self, x):
        x = self.lrelu(self.conv1(x))
        x = self.lrelu(self.conv2(x))
        x = self.lrelu(self.conv3(x))
        x = self.conv4(x)
        return x
