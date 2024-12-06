from io import BytesIO
import lmdb
from PIL import Image
from torch.utils.data import Dataset
import random
import data.util as Util
import numpy as np
import rasterio
import core.metrics as Metrics
def clip_data(data):
    data = data/10000.0
    data = np.clip(data, a_min=0.0, a_max=1.0)
    return data.astype(np.float32)

import numpy as np
import matplotlib.pyplot as plt
from skimage import data  # for loading example image
from torchvision.transforms import functional as trans_fn
import skimage
import torch

def resize_and_convert(img, size, resample):
    print("inside resize and convert function")
    print(img.shape)
    if(img.shape[1] != size):
        img = trans_fn.resize(img, size, resample)
        img = trans_fn.center_crop(img, size)
    return img

def HR_to_LR_SR(img_HR,lr = 128, hr = 384):
    SR = np.zeros(img_HR.shape)
    LR = np.zeros((img_HR.shape[0],lr,lr))
    for i in range(img_HR.shape[0]):
        LR[i,:,:] = img_HR[i,::int(hr/lr),::int(hr/lr)] #resize_and_convert(img_HR[i,:,:], lr, Image.BICUBIC)
        SR[i,:,:] = skimage.transform.resize(LR[i,:,:], img_HR[i,:,:].shape, order=3,preserve_range=True) #resize_and_convert(img_LR[i,:,:], hr, Image.BICUBIC)
    
    return LR.astype(np.float32),SR.astype(np.float32)



class LRHRDataset(Dataset):
    def __init__(self, dataroot, datatype, l_resolution=128, r_resolution=384, split='train', data_len=-1, need_LR=False):
        self.datatype = datatype
        self.l_res = l_resolution
        self.r_res = r_resolution
        self.data_len = data_len
        self.need_LR = need_LR
        self.split = split
        self.min_max = (-1,1)

        if datatype == 'img':
            self.hr_path = Util.get_paths_from_images('{}'.format(dataroot))
            self.dataset_len = len(self.hr_path)
            if self.data_len <= 0:
                self.data_len = self.dataset_len
            else:
                self.data_len = min(self.data_len, self.dataset_len)
        else:
            raise NotImplementedError(
                'data_type [{:s}] is not recognized.'.format(datatype))

    def __len__(self):
        return self.data_len

    def __getitem__(self, index):
        img_HR = None
        img_LR = None
        if self.split!='train':
            print(self.hr_path[index])
        #read tif file into a numpy array
        with rasterio.open(self.hr_path[index]) as src:
            img_HR = clip_data(src.read())
            img_HR=img_HR[:,:self.r_res,:self.r_res]
            # img_HR = src.read().astype(np.float32)
        # print("clipped data type")
        # print(img_HR.dtype)    
        img_LR, img_SR = HR_to_LR_SR(img_HR,lr = self.l_res, hr = self.r_res)
        
        # print("cropped data type")
        # print(img_HR.dtype)
        # print(img_LR.dtype)
        # print(img_SR.dtype)
        # img_HR = np.float32(img_HR)
        # img_LR = np.float32(img_LR)
        # img_SR = np.float32(img_SR)
        Metrics.save_img(Util.s2_to_img(img_HR),'test_hr.png')
        if self.need_LR:
            [img_LR, img_SR, img_HR] = Util.transform_augment_tif(
                [img_LR, img_SR, img_HR], split=self.split, min_max=self.min_max)
            # return {'LR': img_LR.to(torch.float32), 'HR': img_HR.to(torch.float32), 'SR': img_SR.to(torch.float32), 'Index': index}
            return {'LR': img_LR, 'HR': img_HR, 'SR': img_SR, 'Index': index}
        else:
            [img_SR, img_HR] = Util.transform_augment_tif(
                [img_SR, img_HR], split=self.split, min_max=self.min_max)
            # return {'HR': img_HR.to(torch.float32), 'SR': img_SR.to(torch.float32), 'Index': index}
            return {'HR': img_HR, 'SR': img_SR, 'Index': index}
