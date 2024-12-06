from torch import nn
import numpy as np
import torch
import math

class ChannelAttention(nn.Module):
    def __init__(self, num_features, reduction):
        super(ChannelAttention, self).__init__()
        self.module = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(num_features, num_features // reduction, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(num_features // reduction, num_features, kernel_size=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return x * self.module(x)


class RCAB(nn.Module):
    def __init__(self, num_features, reduction, survival_rate):
        super(RCAB, self).__init__()
        self.survival_rate = survival_rate
        assert(survival_rate>0.0)
        self.module = nn.Sequential(
            nn.Conv2d(num_features, num_features, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(num_features, num_features, kernel_size=3, padding=1),
            ChannelAttention(num_features, reduction)
        )

    def forward(self, x):
        if self.training:
            survived = np.random.choice([1, 0], p=[self.survival_rate, 1-self.survival_rate])
            if survived:
                return(x + self.module(x).div_(self.survival_rate))
            else:
                return(x)
        else:
            return x + self.module(x)

        
class RCAB_with_downsample(nn.Module):
    def __init__(self, num_features, reduction, survival_rate, num_features_out):
        super(RCAB_with_downsample, self).__init__()
        self.survival_rate = survival_rate
        
        assert(survival_rate>0.0)
        self.module = nn.Sequential(
            nn.Conv2d(num_features, num_features, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(num_features, num_features_out, kernel_size=3, padding=1),
            ChannelAttention(num_features_out, reduction)
        )

    def forward(self, x):
        '''
        if self.training:
            survived = np.random.choice([1, 0], p=[self.survival_rate, 1-self.survival_rate])
            if survived:
                return(x + self.module(x).div_(self.survival_rate))
            else:
                return(x)
        else:
            return x + self.module(x)
        ''' 
        return self.module(x)


class RG(nn.Module):
    def __init__(self, num_features, num_rcab, reduction, survival_rate, rcan_prob):
        super(RG, self).__init__()
        self.survival_rate = survival_rate
        
        step = rcan_prob / num_rcab
        p = 1.0
        rcab_survival_rates = [p - step*i for i in range(num_rcab)]

        self.module = [RCAB(num_features, reduction, rcab_survival_rates[i]) for i in range(num_rcab)]
        self.module.append(nn.Conv2d(num_features, num_features, kernel_size=3, padding=1))
        self.module = nn.Sequential(*self.module)

    def forward(self, x):
        if self.training:
            survived = np.random.choice([1, 0], p=[self.survival_rate, 1-self.survival_rate])
            if survived:
                return(x + self.module(x).div_(self.survival_rate))
            else:
                return(x)
        else:
            return x + self.module(x)


class RCAN(nn.Module):
    def __init__(self, scale=5, num_features=64, num_rg=10, num_rcab=20, reduction=16, in_channels=10, out_channels=4, rg_prob=0.0, rcan_prob=0.5):
        super(RCAN, self).__init__()
        scale = scale
        num_features = num_features
        num_rg = num_rg
        num_rcab = num_rcab
        reduction = reduction
        in_nc = in_channels
        out_nc = out_channels
        
        
        step = rg_prob / num_rg
        p = 1.0
        survival_rates = [p - step*i for i in range(num_rg)]

        self.sf = nn.Conv2d(in_nc, num_features, kernel_size=3, padding=1)
        self.rgs = nn.Sequential(*[RG(num_features, num_rcab, reduction, survival_rates[i], rcan_prob) for i in range(num_rg)])
        self.conv1 = nn.Conv2d(num_features, num_features, kernel_size=3, padding=1)
        self.upscale = nn.Sequential(
            nn.Conv2d(num_features, num_features * (scale ** 2), kernel_size=3, padding=1),
            nn.PixelShuffle(scale)
        )
        self.conv2 = nn.Conv2d(num_features, out_nc, kernel_size=3, padding=1)

    def forward(self, x):
        x = self.sf(x)
        residual = x
        x = self.rgs(x)
        x = self.conv1(x)
        x += residual
        x = self.upscale(x)
        x = self.conv2(x)
        return x


    
class RCAN_dual(nn.Module):
    def __init__(self, device, scale=5, 
                 num_features_lr=64, 
                 num_rg_lr=4, 
                 num_rcab_lr=10, 
                 reduction_lr=16,
                 in_channel_lr=10,
                 rg_prob_lr=0.5,
                 rcan_prob_lr=0.5,
                 num_features_hr=32, 
                 num_rg_hr=3, 
                 num_rcab_hr=4, 
                 reduction_hr=8,
                 in_channel_hr=4,
                 rg_prob_hr=0.5,
                 rcan_prob_hr=0.5,
                 out_channels=4):
        
        super(RCAN_dual, self).__init__()
        
        num_features_stack = num_features_lr + num_features_hr
        p = 1.0
        step_lr = rg_prob_lr / num_rg_lr
        survival_rate_lr = [p - step_lr * i for i in range(num_rg_lr)]
        
        step_hr = rg_prob_hr / num_rg_hr
        survival_rate_hr = [p - step_hr * i for i in range(num_rg_hr)]
        
        torch.cuda.empty_cache()
        self.num_rg_lr = num_rg_lr
        self.num_rg_hr = num_rg_hr
        self.sf_lr = nn.Conv2d(in_channel_lr, num_features_lr, kernel_size=3, padding=1)
        self.sf_hr = nn.Conv2d(in_channel_hr, num_features_hr, kernel_size=3, padding=1)
        
        self.rgs_lr = nn.ModuleList([RG(num_features_lr, num_rcab_lr, reduction_lr, survival_rate_lr[i], rcan_prob_lr).to(device) for i in range(num_rg_lr)])
        self.rgs_hr = nn.ModuleList([RG(num_features_hr, num_rcab_hr, reduction_hr, survival_rate_hr[i], rcan_prob_hr).to(device) for i in range(num_rg_hr)])
        
        self.rcab_1 = nn.Sequential(
            RCAB_with_downsample(num_features_stack, reduction_hr, survival_rate_hr[0], num_features_hr),
            RCAB(num_features_hr, reduction_hr, survival_rate_hr[0]))
        self.rcab_2 = nn.Sequential(
            RCAB_with_downsample(num_features_stack, reduction_hr, survival_rate_hr[0], num_features_hr),
            RCAB(num_features_hr, reduction_hr, survival_rate_hr[0]))
         
        self.upscale_hr_1 = nn.Sequential(
            nn.Conv2d(num_features_lr, num_features_lr * (scale ** 2), kernel_size=3, padding=1),
            nn.PixelShuffle(scale))
        self.upscale_hr_2 = nn.Sequential(
            nn.Conv2d(num_features_lr, num_features_lr * (scale ** 2), kernel_size=3, padding=1),
            nn.PixelShuffle(scale))
        
        self.conv1 = nn.Conv2d(num_features_lr, num_features_lr, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(num_features_hr, num_features_hr, kernel_size=3, padding=1)
        self.upscale = nn.Sequential(
            nn.Conv2d(num_features_lr, num_features_lr * (scale ** 2), kernel_size=3, padding=1),
            nn.PixelShuffle(scale))
        self.conv3 = nn.Conv2d(num_features_stack, out_channels, kernel_size=3, padding=1)


    def forward(self, x):
        
        num_lr = math.ceil(self.num_rg_lr / 3)
        num_hr = math.ceil(self.num_rg_hr / 3)
        
        # Feed the LR input
        x_lr = self.sf_lr(x[0])
        residual_lr = x_lr
        # Feed the HR input
        x_hr = self.sf_hr(x[1])
        residual_hr = x_hr
        
        # Block 1 of Residual Groups         
        for i in range(num_lr):
            x_lr = self.rgs_lr[i](x_lr)

        for i in range(num_hr):
            x_hr = self.rgs_hr[i](x_hr)
        
        # Upscale
        x_hr = torch.cat((x_hr, self.upscale_hr_1(x_lr)), dim=1)
        x_hr = self.rcab_1(x_hr)
        
        
        delta_lr = math.ceil((self.num_rg_lr - num_lr)/2)
        delta_hr = math.ceil((self.num_rg_hr - num_hr)/2)
        
        
        # Block 2 of Residual Groups
        for i in range(delta_lr):
            x_lr = self.rgs_lr[i + num_lr](x_lr)
            
        for i in range(delta_hr):
            x_hr = self.rgs_hr[i + num_hr](x_hr)
        
        # Upscale
        x_hr = torch.cat((x_hr, self.upscale_hr_2(x_lr)), dim=1)
        x_hr = self.rcab_2(x_hr)
        
        # Block 3 of Residual Groups
        for i in range(delta_lr + num_lr, self.num_rg_lr, 1):
            x_lr = self.rgs_lr[i](x_lr)
            
        for i in range(delta_hr + num_hr, self.num_rg_hr, 1):
            x_hr = self.rgs_hr[i](x_hr)
        
        x_lr = self.conv1(x_lr)
        x_lr += residual_lr
        x_lr = self.upscale(x_lr)
        
        x_hr = self.conv2(x_hr)
        x_hr += residual_hr
        
        # Merge LR and HR
        x_hr = torch.cat((x_hr, x_lr), dim=1)
        x_hr = self.conv3(x_hr)
        
        del(x_lr)
        torch.cuda.empty_cache()
        return x_hr
