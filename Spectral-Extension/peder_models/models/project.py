import numpy as np
import torch
import torch.optim as optim

from torch import nn
from math import log10, sqrt


class Project(nn.Module):
    """
    This module projects a super-resolution prediction onto a high resolution 
    image history.  The projection is done in forward which expects the super-resolution
    image followed by the history.  The batch size has to be 1 for both if the history sizes
    varies.
    """
    def __init__(self, grid_size=25, damping_coefficient=0.01):
        super(Project, self).__init__()
        self.damping_coefficient = damping_coefficient
        self.grid_size = grid_size

    def forward(self, x):
        local_x, ms_history = x
        new_hist = ms_history[0]
        batch_size = ms_history.shape[0]
        
        nt, nb, nx, ny = new_hist.shape
        _,nb2, _, _ = local_x.shape
        ngs = self.grid_size
        assert(batch_size==1 and nx%ngs==0 and ny%ngs==0)
        id_mat = torch.eye(nb*nt).float().to(local_x.get_device())
        ngx = nx // self.grid_size
        ngy = ny // self.grid_size
        V = new_hist.view(nb*nt, ngx, ngs, ngy, ngs).permute(1, 3, 0, 2, 4).reshape(ngx,ngy,nb*nt,ngs**2)
        X = local_x.view(nb2, ngx, ngs, ngy, ngs).permute(1, 3, 0, 2, 4).reshape(ngx,ngy,nb2,ngs**2)
        Y = X * 1.0
        for ix in range(ngx):
            for iy in range(ngy):
                M = (V[ix,iy].transpose(0,1) @ torch.inverse(V[ix,iy] @ V[ix,iy].transpose(0,1)
                                                             +self.damping_coefficient*id_mat))
                Y[ix,iy] = ((X[ix,iy] @ M) @ V[ix,iy])
                W = Y.view(ngx, ngy, nb2, ngs, ngs).permute(2, 0, 3, 1, 4).reshape(1, nb2, nx, ny)
        return W



class AdaptiveProjection(nn.Module):
    """
    We expand the number of bands using a 2-layer convolutional network.
    """
    def __init__(self, num_in_band, num_out_band, grid_size=25, damping_coefficient=0.01):
        super(AdaptiveProjection, self).__init__()
        projection = Project(grid_size=grid_size, damping_coefficient=damping_coefficient)
        nbm = (num_in_band+num_out_band)//2
        self.sigmoid = nn.Sigmoid()
        self.conv1 = nn.Conv2d(num_in_band, nbm, kernel_size=1, padding=0, bias=True)
        self.conv2 = nn.Conv2d(nbm, num_out_band, kernel_size=1, padding=0, bias=True)
        
    def forward(self, x):
        local_x, ms_history = x
        new_hist = self.sigmoid(self.conv2(self.sigmoid(self.conv1(ms_history[0]))))
        x = projection([local_x, new_hist[None,...]])
        return(x)



