from google.cloud import storage
import rasterio
from rasterio.plot import show
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from tqdm import tqdm
import os
from skimage import exposure
# Initialize a Google Cloud Storage client
client = storage.Client('FoundationModel')

# Specify the bucket name and blob name
bucket_name = "lem-earthengine4"
blob_name = "dw_144_shard_0/dw_-51.0098080550_-23.3562130402-20190811/S1_S2_ERA5_SRTM_2020_2021/"

# Get the bucket and blob
bucket = client.bucket(bucket_name)
# blob = bucket.blob(blob_name)

# Download the blob to a local file
# local_path = "0000000512-0000000256.tif"
# blob.download_to_filename(local_path)

def combine_chunks_from_cloud(blob_name,band):
    tif_path1 = "0000000000-0000000000.tif"
    blob_name1=blob_name+tif_path1
    blob = bucket.blob(blob_name1)
    blob.download_to_filename(tif_path1)

    tif_path2 = "0000000000-0000000256.tif"
    blob_name2=blob_name+tif_path2
    blob = bucket.blob(blob_name2)
    blob.download_to_filename(tif_path2)

    tif_path3 = "0000000256-0000000000.tif"
    blob_name3=blob_name+tif_path3
    blob = bucket.blob(blob_name3)
    blob.download_to_filename(tif_path3)

    tif_path4 = "0000000256-0000000256.tif"
    blob_name4=blob_name+tif_path4
    blob = bucket.blob(blob_name4)
    blob.download_to_filename(tif_path4)
    
    with rasterio.open(tif_path1) as dataset:
        d_1=dataset.read()[band,:,:]
    with rasterio.open(tif_path2) as dataset:
        d_2=dataset.read()[band,:,:]
    with rasterio.open(tif_path3) as dataset:
        d_3=dataset.read()[band,:,:]
    with rasterio.open(tif_path4) as dataset:
        d_4=dataset.read()[band,:,:]
    d = np.concatenate((np.concatenate((d_1,d_3), axis=0),np.concatenate((d_2,d_4), axis=0)), axis=1)

    return d

def combine_chunks(path,band):
    tif_path1 = "0000000000-0000000000.tif"
    tif_path2 = "0000000000-0000000256.tif"
    tif_path3 = "0000000256-0000000000.tif"
    tif_path4 = "0000000256-0000000256.tif"
    
    with rasterio.open(path+"/"+tif_path1) as dataset:
        d_1=dataset.read()[band,:,:]
    with rasterio.open(path+"/"+tif_path2) as dataset:
        d_2=dataset.read()[band,:,:]
    with rasterio.open(path+"/"+tif_path3) as dataset:
        d_3=dataset.read()[band,:,:]
    with rasterio.open(path+"/"+tif_path4) as dataset:
        d_4=dataset.read()[band,:,:]
    d = np.concatenate((np.concatenate((d_1,d_3), axis=0),np.concatenate((d_2,d_4), axis=0)), axis=1)

    return d

def save_png(np_array,f_name):
    # print(np.max(np_array))
    # im = Image.fromarray((np_array/np.max(np_array)*255).astype(int))
    # img = (np_array/10000)
    # print(img)
    # print(np.max(img))
    # print(img)
    im = Image.fromarray(np_array)
    im.save(f_name)

def s2_to_img(img, percentile=(0.5, 99.5), gamma=0.9, bands = [3,2,1]):
    """
    Function to create RGB representation of a Sentinel-2 image.
    """
    # img = s2[bands].transpose([1, 2, 0])
    img = (img/10000)
    nb = img.shape[2]
    # image_numpy = np.clip((image_numpy + 1.0) / 2.0 * 1.8, a_min=0.0, a_max=1.0) * 255.0
    #img = np.clip((img + 1.0) / 2.0, a_min=0.0, a_max=1.0) * 255.0
    img = np.clip(img, a_min=0.0, a_max=1.0) * 255.0
    for b in range(nb):
        plow, phigh = np.percentile(img[...,b], percentile)
        if phigh>plow:
            x_ = exposure.rescale_intensity(img[...,b], in_range=(plow, phigh))
            y_ = (x_ - x_.min()) / (x_.max()-x_.min())
            if gamma!=1.0:
                y_ = y_ ** gamma
            img[...,b] = (y_*255)
    return img.astype(np.uint8)

def make_rgb(r,g,b):
    return np.concatenate((np.expand_dims(r, axis=2),np.expand_dims(g, axis=2),np.expand_dims(b, axis=2)), axis=2)
    
def s2_to_bands(img):
    """
    Function to create RGB representation of a Sentinel-2 image.
    """
    img = img/10000
    # nb = img.shape[2]
    # image_numpy = np.clip((image_numpy + 1.0) / 2.0 * 1.8, a_min=0.0, a_max=1.0) * 255.0
    #img = np.clip((img + 1.0) / 2.0, a_min=0.0, a_max=1.0) * 255.0
    img = np.clip(img, a_min=0.0, a_max=1.0) * 255.0
    # for b in range(nb):
    x_ = img
    y_ = (x_ - x_.min()) / (x_.max()-x_.min())
    img = (y_*255)
    return img.astype(np.uint8)

def main():
    save_path_root = '/home/t-nnaeem/workspace/RemoteSensingFoundationModels/Image-Super-Resolution-via-Iterative-Refinement/dataset/sentinal_ee_20/'
    os.makedirs(save_path_root, exist_ok = True)
    
    
    # for shard_num in tqdm(range(26)):
    #     blob_name = "dw_144_shard_"+str(shard_num)+"/dw_-51.0098080550_-23.3562130402-20190811/S1_S2_ERA5_SRTM_2020_2021/"

    # Get the bucket and blob
    # data_path = "/workspace/RemoteSensingFoundationModels/Image-Super-Resolution-via-Iterative-Refinement/dataset/GC_dataset/"
    data_path = "/home/t-nnaeem/workspace/RemoteSensingFoundationModels/EEdata/"
    folders = os.listdir(data_path)
    for folder in tqdm(folders):
        save_path = save_path_root + "RGB"
        os.makedirs(save_path, exist_ok = True)
        path_to_files = data_path + folder + "/S1_S2_ERA5_SRTM_2020_2021"
        # path_to_files = data_path + folder +"/DynamicWorldMonthly2020_2021"
        if os.path.exists(path_to_files):
            r = combine_chunks(path_to_files,5)
            g = combine_chunks(path_to_files,4)
            b = combine_chunks(path_to_files,3)
            # print(np.max(r))
            # print(type(r))
            f_name = save_path+"/shard_20"+folder+"_time_0.png"
            # print(f_name)
            rgb_img = s2_to_img(make_rgb(r,g,b), percentile=(0.5, 99.5), gamma=0.9, bands = [3,2,1])
            save_png(rgb_img,f_name)
            for b in range(6,15,1):
                if b!=11 and b!=12:
                    save_path = save_path_root+"B"+str(b-2)+"/"
                    os.makedirs(save_path, exist_ok = True)
                    band_img = combine_chunks(path_to_files,b)
                    f_name = save_path+"/shard_0"+folder+"_time_0.png"
                    img = s2_to_bands(band_img)
                    save_png(img,f_name)
                    



main()




