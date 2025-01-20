from io import BytesIO
import lmdb
from PIL import Image
from torch.utils.data import Dataset
import random
import dataloader.util as Util
import numpy as np
import rasterio
import helper.metrics as Metrics
import sys

import numpy as np
import matplotlib.pyplot as plt
from skimage import data  # for loading example image
from torchvision.transforms import functional as trans_fn
import skimage
import torch
import os
import glob
import random

def clip_data(data,type="Sentinel-2"):
    if type=="Sentinel-2":
        data = (data)/10000.0 #imperical normalization
    else:
        data = (data)/25000.0 #imperical normalization
    return data.astype(np.float32)


def resize_and_convert(img, size, resample):
    #resize and convert the images
    if(img.shape[1] != size):
        img = trans_fn.resize(img, size, resample)
        img = trans_fn.center_crop(img, size)
    return img

def HR_to_LR_SR(img_HR,lr = 128, hr = 384):
    #given a high resolution image this function returns the low resolution and interpolated super resolution image
    SR = np.zeros(img_HR.shape)
    LR = np.zeros((img_HR.shape[0],lr,lr))
    for i in range(img_HR.shape[0]):
        LR[i,:,:] = img_HR[i,::int(hr/lr),::int(hr/lr)] #resize_and_convert(img_HR[i,:,:], lr, Image.BICUBIC)
        SR[i,:,:] = skimage.transform.resize(LR[i,:,:], img_HR[i,:,:].shape, order=3,preserve_range=True) #resize_and_convert(img_LR[i,:,:], hr, Image.BICUBIC)
    
    return LR.astype(np.float32),SR.astype(np.float32)

from os import listdir
from os.path import isfile, join

class LSS2Dataset(Dataset):
    def __init__(self, dataroot, split='train', data_len=-1):
        self.data_len = data_len
        self.split = split
        self.dataroot = dataroot
        self.ls_path = Util.get_paths_from_images(dataroot+'/Landsat/')
        random.Random(4).shuffle(self.ls_path)
        
        if self.split=='train':
            self.ls_path=self.ls_path[:int(0.8*len(self.ls_path))]
        else:
            self.ls_path=self.ls_path[int(0.8*len(self.ls_path)):]

        self.dataset_len = len(self.ls_path)
        if self.data_len <= 0:
            self.data_len = self.dataset_len
        else:
            self.data_len = min(self.data_len, self.dataset_len)

    def __len__(self):
        return self.data_len

    def __getitem__(self, index):
        img_LS = None
        img_S2 = None
        
        #read tif file into a numpy array
        with rasterio.open(self.ls_path[index]) as src:
            img_LS = clip_data(src.read(),"Landsat")
            img_LS = img_LS[:,:128,:128]
        base =  os.path.basename(self.ls_path[index])
        l_fname = os.path.splitext(base)[0]
        num  = l_fname.split('_')[-1]
        f_name = glob.glob(self.dataroot+'/S2_aligned/*_dw_'+str(num)+'.tif')
        s2_path = f_name[0]
        with rasterio.open(s2_path) as src:
            img_S2 = clip_data(src.read(),"Sentinel-2")
            img_S2 = img_S2[:,:128*3,:128*3]
           
        return {'LS': img_LS, 'S2': img_S2, 'Index': index}
