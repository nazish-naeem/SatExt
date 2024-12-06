import functools
import torch
import torch.nn as nn
import torch.nn.functional as F



def make_layer(block, n_layers):
    layers = []
    for _ in range(n_layers):
        layers.append(block())
    return nn.Sequential(*layers)


def initialize_weight(weight_list):    
    for wgt in weight_list:
        nn.init.xavier_normal_(wgt.weight, gain=0.2)
        #nn.init.xavier_uniform_(wgt.weight, gain=0.2)

        
class ResidualDenseBlock(nn.Module):
    def __init__(self, nf = 64, gc = 32, bias = True):
        super(ResidualDenseBlock, self).__init__()
        
        #gc: growth channel, i.e. intermediate channels
        self.conv1 = nn.Conv2d(nf, gc, 3, 1, 1, bias=bias)
        self.conv2 = nn.Conv2d(nf + gc, gc, 3, 1, 1, bias=bias)
        self.conv3 = nn.Conv2d(nf + 2 * gc, gc, 3, 1, 1, bias=bias)
        self.conv4 = nn.Conv2d(nf + 3 * gc, gc, 3, 1, 1, bias=bias)
        self.conv5 = nn.Conv2d(nf + 4 * gc, nf, 3, 1, 1, bias=bias)
        
        self.lrelu = nn.LeakyReLU(negative_slope = 0.2, inplace=True)

        #Initialization
        # mutil.initialize_weights([self.conv1, self.conv2, self.conv3, self.conv4, self.conv5], 0.1)
        initialize_weight([self.conv1, self.conv2, self.conv3, self.conv4, self.conv5])

    def forward(self, x):
        
        x1 = self.lrelu(self.conv1(x))
        x2 = self.lrelu(self.conv2(torch.cat((x, x1), 1)))
        x3 = self.lrelu(self.conv3(torch.cat((x, x1, x2), 1)))
        x4 = self.lrelu(self.conv4(torch.cat((x, x1, x2, x3), 1)))
        x5 = self.conv5(torch.cat((x, x1, x2, x3, x4), 1))
        return x5 * 0.2 + x


class RRDB(nn.Module):
    '''Residual in Residual Dense Block'''
    
    def __init__(self, nf, gc = 32):
        
        super(RRDB, self).__init__()
        self.RDB1 = ResidualDenseBlock(nf, gc)
        self.RDB2 = ResidualDenseBlock(nf, gc)
        self.RDB3 = ResidualDenseBlock(nf, gc)

    def forward(self, x):
        
        out = self.RDB1(x)
        out = self.RDB2(out)
        out = self.RDB3(out)
        
        return out * 0.2 + x


class RRDBModel(nn.Module):
    def __init__(self, in_nc, out_nc, nf, nb, gc=32, scale_factor=5):
        
        super(RRDBModel, self).__init__()
        self.scale_factor = scale_factor
        RRDB_block_f = functools.partial(RRDB, nf = nf, gc = gc)
        
        
        self.conv1 = nn.Conv2d(in_nc, nf, 3, 1, 1, bias=True) # First convolution
        self.RRDB_unit = make_layer(RRDB_block_f, nb)  # Residual Blocks
        self.conv2 = nn.Conv2d(nf, nf, 3, 1, 1, bias=True) # Second Convolution
        
        self.upconv1 = nn.Conv2d(nf, nf, 3, 1, 1, bias=True) # Pre-upsampling
        self.upconv2 = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)
        
        # Post-upsampling (ADD PAST AIRBUS IMAGE)
        self.conv3 = nn.Conv2d(nf, nf, 3, 1, 1, bias = True)
        self.conv4 = nn.Conv2d(nf, nf, 3, 1, 1, bias = True)
        
        # Fine-tune 
        self.HRconv = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)        
        self.conv_last = nn.Conv2d(nf, out_nc, 3, 1, 1, bias=True)
        self.lrelu = nn.LeakyReLU(negative_slope = 0.2, inplace=True)
        
        initialize_weight([self.conv1, self.conv2, self.conv3, self.conv4, self.upconv1, self.HRconv, self.conv_last])

    
    def forward(self, x):
        
        fea = self.conv1(x)
        fea = fea + self.conv2(self.RRDB_unit(fea))

        # Pre-upsampling
        fea = self.lrelu(self.upconv1(F.interpolate(fea, scale_factor = self.scale_factor, mode='nearest')))
        
        # Post-sampling
        fea = self.lrelu(self.conv3(fea))
        fea = self.lrelu(self.conv4(fea))
        out = self.conv_last(self.lrelu(self.HRconv(fea)))

        return out
    
    
class RRDBModel_pan(nn.Module):
    def __init__(self, in_nc, out_nc, nf, nb, gc=32):
        
        super(RRDBModel_pan, self).__init__()
        
        RRDB_block_f = functools.partial(RRDB, nf = nf, gc = gc)
        
        # First convolution
        self.conv1 = nn.Conv2d(in_nc, nf, 3, 1, 1, bias=True)
        
        # Residual Blocks
        self.RRDB_unit = make_layer(RRDB_block_f, nb)
       
        # Second Convolution
        self.conv2 = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)
        
        # Pre-upsampling
        self.upconv1 = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)
        #self.upconv2 = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)
        
        # Post-upsampling (ADD PAST AIRBUS IMAGE)
        self.conv3 = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)
        self.conv4 = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)
        
        # Fine-tune 
        self.HRconv = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)        
        self.conv_last = nn.Conv2d(nf, out_nc, 3, 1, 1, bias=True)
        self.lrelu = nn.LeakyReLU(negative_slope = 0.2, inplace=True)
        
        initialize_weight([self.conv1, self.conv2, self.conv3, self.conv4, self.upconv1, self.HRconv, self.conv_last])

    def forward(self, x):
        
        fea = self.conv1(x)
        fea = fea + self.conv2(self.RRDB_unit(fea))

        # Pre-upsampling
        fea = self.lrelu(self.upconv1(F.interpolate(fea, scale_factor = 2, mode='nearest')))
        fea = self.lrelu(self.upconv1(F.interpolate(fea, scale_factor = 2, mode='nearest')))
        #fea = torch.cat((fea, x0), 1)
        
        # Post-sampling
        fea = self.lrelu(self.conv3(fea))
        #fea = torch.cat((fea, x0), 1)
        
        #fea = self.lrelu(self.conv4(fea))
        #fea = torch.cat((fea, x0), 1)
        
        fea = self.lrelu(self.conv4(fea))
        out = self.conv_last(self.lrelu(self.HRconv(fea)))

        return out
