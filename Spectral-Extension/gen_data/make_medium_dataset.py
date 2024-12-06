import os
import argparse
import random
import json
import time
import numpy as np
import matplotlib.pyplot as plt
#from preprocessing.scihub.utils import load_tiff_to_numpy
from datetime import datetime, timedelta
import rasterio
from rasterio.warp import reproject, Resampling
from rasterio.warp import calculate_default_transform
from rasterio.transform import from_bounds
from rasterio.windows import Window
from rasterio.errors import RasterioIOError

from osgeo import ogr
import geopandas as gpd

import pystac_client
import planetary_computer
import odc.stac
from pystac.extensions.eo import EOExtension as eo
from pystac_client.exceptions import APIError

from glob import glob
from tqdm import tqdm
from collections import Counter


from superres import data_alignment

# parameters
parser = argparse.ArgumentParser(description='Script to download Matched Landsat and Sentinel-2 images.')
parser.add_argument('--first', default=0, type=int, help='start on <first> sample.  Default=0')
parser.add_argument('--last', default=-1, type=int, help='end on <last> sample.  Default=-1')
parser.add_argument('--data_folder', type=str, help='directory to put data in', default='data_medium')
parser.set_defaults(augment=True)
args = parser.parse_args()

first = args.first
last = args.last
data_folder = args.data_folder
if not os.path.exists(data_folder):
    os.mkdir(data_folder)

status_file = os.path.join(data_folder, f"status_{first}_{last}.txt")
dw_samples = gpd.read_file('dynamic_world_samples.geojson')
if last==-1:
    last = len(dw_samples)
if os.path.exists(status_file):
    with open(status_file) as f:
        sample = int(f.readline())
    print(f"Resuming from sample {sample}.")
    first = sample

print(f"Processing samples {first}:{last} out of 0:{len(dw_samples)}")
    
years = [2016, 2017, 2018, 2019, 2020, 2021, 2022, 2023]
max_tries = 5
n_successes = 0
n_failed = 0

for sample in tqdm(range(first, last)):
    files = (None, None, None)
    ntries = 0
    bbox_of_interest, time_of_interest = data_alignment.geopandas_planetary_computer_query(dw_samples, sample=sample, days_before=0, days_after=0)
    random.shuffle(years)
    for year in years:
        success = False
        try:
            matches = data_alignment.get_s2_lsat_matches(year, bbox_of_interest)
        except APIError:
            print(f"{sample=}, {year=}: API Error from Planetary Computer.")
            time.sleep(20)
            ntries += 1
            continue

        match_list = list(matches)
        random.shuffle(match_list)
        for lsat_dt in match_list:
            #print(f"{year}, {lsat_dt}, {ntries}")
            lsat_item, s2_item = matches[lsat_dt]
            try:
                success, files = data_alignment.write_aligned_data(sample, lsat_item, s2_item, data_folder, bbox_of_interest, check_clouds=True, verbose=False)
            except RasterioIOError:
                print(f"{sample=}, {year=}, {lsat_dt=}: RasterioIOError")
                time.sleep(20)
                ntries += 1
            if success:
                break
            else:
                ntries += 1
            if ntries>max_tries:
                break
        if success or ntries>max_tries:
            break
    if success:
        n_successes += 1
        # update status file
        with open(status_file, 'w') as f:
            f.write(str(sample))
    else:
        n_failed += 1
    

print(f"Successful matches {n_successes}, failed {n_failed}.")                                                                                                                                                                                                                                    
