######################################################################
##### AUTHORS: Olaoluwa Adigun and Peder Olsen                   #####  
##### EMAIL: t-oladig@microsoft.com, Peder.Olsen@microsoft.com   ##### 
##### DATE: 06-19-2020.                                          #####   
######################################################################

import sys
import time
#import h5py

#import argparse
#import shutil
#import datetime
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors 

sys.path.append("/home/adigun/source_code/Cloned-Repo/final_version")

from utils.utils import *
from utils.image import *
from utils.projection import *

from skimage.measure import label
from skimage.measure import regionprops
from skimage.segmentation import find_boundaries
from skimage.morphology import disk, binary_dilation



def kill_labels_(clabel, min_area):
    
    """
     USAGE: kill_list = kill_labels(clabel, min_area)
     Make a list of regions with area below min_area and return the list of regions.
    """
    props = regionprops(clabel)
    kill_list = []
    
    for p in props:
        
        if p.area<min_area:
            kill_list.append(p.label)

    return(kill_list)

def get_mask_airbus(arr, spectral_type):
    """
    Get the mask that denotes the bad pixels in the image array
    
    Args: 
        arr : Image array
        image_type : {'MS': Multispectral, 'P': Panchromatic}
    Returns:
        Mask of bad pixel points
    """
    #print(arr.shape)
    if spectral_type == "MS":
        #arr_T = np.transpose(arr, [1,2,0])
        mask = np.sum(arr[:,:,0:3] != 0.0, axis = 2) != 0.0
    elif spectral_type == "P":
        mask = (arr != 0.0)[:,:,0]
    else:
        raise ValueError("Pick one of the allowed spectral-type!!!")
    
    return mask.astype(float)

                                              
def get_mask(arr, threshold, spectral_type):
    
    cloud = (np.min(arr, axis = 2) > threshold).astype(float)
    bad_pixel = get_mask_airbus(arr, spectral_type)
    
    #print( np.sum((cloud == 1.0) == (bad_pixel == 0.0)))
    
    return cloud, bad_pixel
                          
                         
                          
def remove_small_components(cmask, min_area = 400):

    """
    USAGE: new_mask = remove_small_components(cmask, min_area=400)
    First removes small connected cloud component:her cloud mask.

    """
    assert(cmask.ndim == 2)

    cm2_comp = label(cmask) # remove small clouds
    
    tmp = cmask.copy()
    kill_list = kill_labels_(cm2_comp,min_area)
    
    small_clouds = np.isin(cm2_comp,kill_list)
    tmp[small_clouds] = False
    
    tmp = tmp.astype(bool)
    
    cm2_inv = label(~tmp) # fill small holes in clouds
    kill_list = kill_labels_(cm2_inv, min_area)
    
    small_cloud_holes = np.isin(cm2_inv,kill_list)
    tmp[small_cloud_holes] = True
    return(tmp)
                          
    
def cloud_mask_from_image(img, threshold=150, min_area=1000, dilation=10, spectral_type='MS'):
    cloud_mask, missing_mask = get_mask(img, threshold, spectral_type)
    cloud_mask = remove_small_components(cloud_mask, min_area)
    selem = disk(dilation)
    cmask = binary_dilation(cloud_mask, selem)
    mask = 2 * (1 -  missing_mask) + cmask
    return(mask)
                         
        
def get_cloud_mask(loc_info, loc_map, data_list, threshold=150, min_area=1000, spectral_type='MS'):
     
    clean_list = loc_info['clean_list']
    cloud_past = []
    for i in clean_list:
        arr, _ =  get_mask(
            get_single_abus_image(loc_info, loc_map, i, spectral_type), threshold, spectral_type)
        cloud_past.append(arr)
    cloud_past = np.sum(np.asarray(cloud_past), axis = 0)
    
    abus = []
    img_mask = []
    
    for idx in data_list: 
        
        # Get the image
        img = get_single_abus_image(loc_info, loc_map, idx, spectral_type)
        cloud, bad_pixel = get_mask(img, threshold, spectral_type)
                          
        #clean_list = loc_info['clean_list']
        #cloud_past = np.zeros(cloud.shape)

        #for i in clean_list:
        #    arr, _ =  get_mask(
        #        get_single_abus_image(loc_info, loc_map, i, spectral_type), threshold, spectral_type)
        #    cloud_past += arr
    
        update_cloud = np.maximum(cloud - (cloud_past > 2).astype(float) , 0.0)
        update_cloud = remove_small_components(update_cloud, min_area)
        selem = disk(10)
        cmask = binary_dilation(update_cloud, selem)
        #print(np.unique(update_cloud))
        percentage_cloud = np.mean(update_cloud)*100
        print("Percentage of Cloud is : {:.4f} %".format(percentage_cloud))
        
        #print(bad_pixel.shape, update_cloud.shape)
        mask = 2 * (1.0 -  bad_pixel) + update_cloud
        mask = 2 * (1.0 -  bad_pixel) + cmask
        
        abus.append(img)
        img_mask.append(mask)

    return abus, img_mask




def create_cloud_mask(loc_info, loc_map, data_list , threshold, min_area, spectral_type):
    
    #abus = []
    #mask = []
    
    assert (set(data_list).issubset(set(range(len(loc_map))))), "Check the possible image indices for" + loc_info['name']  + "!!!" 
    
    clean_list = loc_info['clean_list']
    
    for idx in data_list:
        tmp_list = clean_list.copy()
        try:
            tmp_list.remove(idx)
            print("The image is on clean list")
        except ValueError:
            print("The image is on cloudy list")
    
    abus, mask = get_cloud_mask(loc_info, loc_map, data_list, threshold, min_area, spectral_type)
    #mask_, img_ = get_cloud_mask(loc_info, loc_map, idx, threshold, min_area, spectral_type)
    #abus.append(img_)
    #mask.append(mask_)
                          
    #mask = np.expand_dims(np.asarray(mask).astype(int), axis = 3)
    abus = np.asarray(abus).astype(int)
    mask = np.asarray(mask).astype(int)
    mask = np.expand_dims(mask, axis = 3)
    return abus, mask