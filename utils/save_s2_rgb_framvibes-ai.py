from vibe_core.client import get_default_vibe_client
import rasterio
from rasterio.plot import show
import matplotlib.pyplot as plt
from PIL import Image
from tqdm import tqdm
import os
client = get_default_vibe_client()
run_ids = client.list_runs()
# id = 2
# run_ids = run_ids[925:]


import numpy as np
from skimage import exposure
def s2_to_img(s2, percentile=(0.5, 99.5), gamma=0.9, bands = [3,2,1]):
    """
    Function to create RGB representation of a Sentinel-2 image.
    """
    img = s2[bands].transpose([1, 2, 0])
    img = img/10000
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

def save_png(np_array,f_name):
    im = Image.fromarray(np_array)
    im.save(f_name)


def main():
    save_path = "/workspace/RemoteSensingFoundationModels/Image-Super-Resolution-via-Iterative-Refinement/dataset/farmvibes_S2_RGB/"
    for i in tqdm(range(925,len(run_ids))):
        run_id = run_ids[i] # will list the most recent run_id
        run = client.get_run_by_id(run_id)
        #save rgb image
        f_name = save_path + run.name+".png"
        with rasterio.open(run.output['raster'][0].assets[0].path_or_url) as dataset:
            img = s2_to_img(dataset.read(), percentile=(0.5, 99.5), gamma=0.9, bands = [3,2,1])
            save_png(img[:512,:512,:],f_name)
    for b in range(4,12,1):
            if b!=9:
                save_path = "/workspace/RemoteSensingFoundationModels/Image-Super-Resolution-via-Iterative-Refinement/dataset/farmvibes_S2_B"+str(b)+"/"
                os.makedirs(save_path, exist_ok = True)
                for i in tqdm(range(925,len(run_ids))):
                    run_id = run_ids[i] # will list the most recent run_id
                    run = client.get_run_by_id(run_id)
                    #save rgb image
                    f_name = save_path + run.name+".png"
                    with rasterio.open(run.output['raster'][0].assets[0].path_or_url) as dataset:
                        img = s2_to_bands(dataset.read()[b])
                        save_png(img[:512,:512],f_name)

main()