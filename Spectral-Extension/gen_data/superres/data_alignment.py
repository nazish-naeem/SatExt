import os
import numpy as np
import skimage
import matplotlib.pyplot as plt
#from preprocessing.scihub.utils import load_tiff_to_numpy
from datetime import datetime, timedelta
import rasterio
from rasterio.warp import reproject, Resampling
from rasterio.warp import calculate_default_transform
from rasterio.transform import from_bounds
from rasterio.windows import Window

from weed import weed_data
from osgeo import ogr
import geopandas as gpd

import pystac_client
import planetary_computer
import odc.stac
import matplotlib.pyplot as plt
from pystac.extensions.eo import EOExtension as eo
from skimage import exposure
from rasterio.errors import RasterioIOError


def s2_to_img(s2, percentile=(0.5, 99.5), gamma=0.9, bands = [3,2,1]):
    """
    Function to create RGB representation of a Sentinel-2 image.
    """
    img = s2[bands].transpose([1, 2, 0])
    nb = img.shape[2]
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


def geopandas_planetary_computer_query(df, sample, days_before=0, days_after=0):
    """
    Give a search string +- num_days around the data entry.
    USAGE: bbox_of_interest, time_of_interest = geopandas_planetary_computer_query(dw_samples, sample=33, num_days=2)
    """
    day = df.loc[sample].date
    # got both a string and a geopandas time type, so need to catch both cases when converting to datetime
    if hasattr(day, 'strftime'):
        day = day.strftime('%Y-%m-%d')
    dd = datetime.strptime(day, '%Y-%m-%d')
        
    time_of_interest = f"{(dd-timedelta(days=days_before)).strftime('%Y-%m-%d')}/{(dd+timedelta(days=days_after)).strftime('%Y-%m-%d')}"
    poly = df.loc[sample].geometry
    [lon1, lon2, lat1, lat2] = ogr.CreateGeometryFromWkt(poly.wkt).GetEnvelope()
    bbox_of_interest = [lon1, lat1, lon2, lat2]
    return(bbox_of_interest, time_of_interest)


def nesting_box(bbox, ref_fname, src_crs="EPSG:4326", cut_at_edge=False, multiplier=1):
    """
    bbox is assumed to be in src_crs.  We transform bbox to crs of ref_fname and align to it's coordinate grid with
    the constraint that grid points have to be a multiple of multiplier.
    
    USAGE: lsat_bounds, coord_box, at_edge = nesting_box(bbox_of_interest, selected_item.assets['blue'].href)
    
    lsat_bound == [left, bottom, right, up]
    coord_box = ((first_row, last_row), (first_col, last_col))
    """
    with rasterio.open(ref_fname) as ref_src:
        target_crs = ref_src.crs
        res = ref_src.res
        new_box = rasterio.warp.transform_bounds(src_crs, target_crs, *bbox)
        big_box = ref_src.bounds
        x0, dx, _, y0, _, dy = ref_src.transform.to_gdal()
        assert(dy<0) # otherwise expand_box_to_grid will fail
    return(expand_box_to_grid(new_box, big_box, res, cut_at_edge=False, multiplier=multiplier, flips=(dx<0, dy<0)))


def expand_box_to_grid(bbox, big_box, resolution, cut_at_edge=False, multiplier=1, flips=(False, True)):
    """
    Assumes the usual scenario where the y-axis is pointing down (i.e. dy is negative in gdal.transform)
    """
    l, b, r, u = bbox
    L, B, R, U = big_box
    (l, r), i, at_edge1 = expand_interval(l, r, L, R, resolution[0], cut_at_edge=cut_at_edge, multiplier=multiplier, flip=flips[0])
    (b, u), j, at_edge2 = expand_interval(b, u, B, U, resolution[1], cut_at_edge=cut_at_edge, multiplier=multiplier, flip=flips[1])
    at_edge = at_edge1 | at_edge2
    bounds = (l, b, r, u)
    coord_box = (i,j)
    return(bounds, coord_box, at_edge)


def expand_interval(x1, x2, X1, X2, dx, cut_at_edge=False, multiplier=1, flip=True):
    """
    Adjust x1, x2 so that x1 = X1 + a*dx, x2 = X1 + b*dx, where a, b are integers and multiples of multiplier
    if flip then x1 = X2 - a*dx, x2 = X2 - b*dx.  The new x1, x2 values are subsequently returned along with 
    a flag to indicate whether the interval hit the edge
    """
    #print(f"{flip=}, {x1=}, {x2=}, {X1=}, {X2=}, {dx=}")
    sign = 1
    if flip:
        X1, X2 = X2, X1
        sign *= -1
    i1 = multiplier*int((x1-X1)/(multiplier*dx*sign))
    i2 = multiplier*int((x2-X1)/(multiplier*dx*sign))
    N = multiplier*int((X2-X1)/(multiplier*dx*sign))
    if sign>0:
        #print(sign, i1, i2, N)
        i1, i2, at_edge = boundary_conditions(i1, i2, N, cut=cut_at_edge)
    else:
        #print(sign, i1, i2, N)
        i2, i1, at_edge = boundary_conditions(i2, i1, N, cut=cut_at_edge)
    x1 = X1 + i1 * sign * dx
    x2 = X1 + i2 * sign * dx
    return((x1, x2), (min(i1,i2), max(i1,i2)), at_edge)


def boundary_conditions(i1, i2, N, cut=False):
    """
    adjust the interval i1:i2 to be inside 0:N.  
    cut==false: shift interval to start/end at boundary
    cut==false: cut of interval at boundary.

    USAGE: (i1,i2,at_edge) = boundary_conditions(i1, i2, N)
    """
    assert(i1<=i2)
    w = i2-i1
    if i1<0:
        i1 = 0
        if not cut:
            i2 = w
        i2 = min(N, i2)
    if i2>=N:
        i2 = N
        if not cut:
            i1 = N-w
        i1 = max(0,i1)
    at_edge = (i1==0) | (i2==N)
    return(i1, i2, at_edge)


def create_lsat_tif(pc_item, out_fname, bounds, coord_box, target_bands = ['blue', 'green', 'red', 'nir08', 'swir16', 'swir22', 'qa_pixel'], verbose=False):
    """
    returns False if there is no data in the input.  
    """
    ((i1,i2), (j1,j2)) = coord_box
    width = i2-i1
    height = j2-j1

    [left, bottom, right, up] = bounds
    nb = len(target_bands)
    transform = from_bounds(left, bottom, right, up, width, height)
    with rasterio.open(pc_item.assets[target_bands[0]].href) as src:
        kwargs = src.meta.copy()
    kwargs.update({'transform': transform, 'width': width, 'height': height, 'compress': 'zstd', 'count': nb-1})

    # read all bands from planetary computer
    win = Window.from_slices((j1,j2), (i1,i2))
    #print(win)
    lsat_data = np.zeros((nb, height, width), dtype=np.uint16)
    for b, band in enumerate(target_bands):
        #print(pc_item.assets[band].href)
        with rasterio.open(pc_item.assets[band].href) as src:
            try :
                tmp = src.read(1, window=win)
            except RasterioIOError as e:
                print(f"Failed read from {pc_item.assets[band].href}")
                print(f"window: {band}: {src.height}x{src.width}: {win}")
                raise e
            lsat_data[b,:, : ] = tmp[:,:]
            
    if lsat_data[:-1].mean()==0.0:
        if verbose:
            print("All the data is zero.  Not creating new file.")
        return(False, None)
    if np.any(lsat_data[:-1].mean(axis=0)==0):
        if verbose:
            print("Some of the data is missing.  Not creating new file.")
        return(False, None)
            
    # write stacked bands to tif file
    with rasterio.open(pc_item.assets['blue'].href) as src:
        with rasterio.open(out_fname, 'w', **kwargs) as dst:
           dst.write(lsat_data[:-1,:,:])
    return(True, lsat_data[-1,:,:])


def create_s2_tif(pc_item, out_fname, bounds, coord_box, verbose=False):
    """
    returns False if there is no data in the input.  
    """
    ((i1,i2), (j1,j2)) = coord_box
    width = i2-i1
    height = j2-j1
    assert( i1%2==0 and i2%2==0 and j1%2==0 and j2%2==0) # box must align with 20m grid.

    # extract 10 and 20m bands in order
    bands_10m = []
    bands_20m = []
    all_bands = []
    for band in pc_item.assets:
        if pc_item.assets[band].title.startswith('Band'):
            if pc_item.assets[band].extra_fields['gsd']==10.0:
                bands_10m.append(band)
                all_bands.append(band)
            if pc_item.assets[band].extra_fields['gsd']==20.0:
                bands_20m.append(band)
                all_bands.append(band)
    bands_20m.append('SCL') # cloud and cloud-shadow masks
    all_bands.append('SCL')
    
    # calculate geotransform corresponding to window
    [left, bottom, right, up] = bounds
    nb = len(all_bands)
    transform = from_bounds(left, bottom, right, up, width, height)
    with rasterio.open(pc_item.assets[all_bands[0]].href) as src:
        kwargs = src.meta.copy()
    kwargs.update({'transform': transform, 'width': width, 'height': height, 'compress': 'zstd', 'count': nb-1})

    # read all bands from planetary computer
    win = Window.from_slices((j1,j2), (i1,i2))
    win_20 = Window.from_slices((j1//2,j2//2), (i1//2,i2//2))
    #print(win, win_20)
    s2_data = np.zeros((nb, height, width), dtype=np.uint16)
    for b, band in enumerate(all_bands):
        #print(band)
        with rasterio.open(pc_item.assets[band].href) as src:
            if band in bands_10m:
                s2_data[b,:, : ] = src.read(1, window=win)
            else:
                s2_data[b,:, : ] = skimage.transform.resize(src.read(1, window=win_20), 
                                                              (height, width),
                                                              preserve_range = True, 
                                                              order=3) # order=3 is bicubic interpolation, use 0 for nearest-neighbor          
    if s2_data[:-1].mean()==0.0:
        if verbose:
            print("All the data is zero.  Not creating new file.")
        return(False, None)
    if np.any(s2_data[:-1].mean(axis=0)==0):
        if verbose:
            print("Some of the data is missing.  Not creating new file.")
        return(False, None)
        
    # write stacked bands to tif file
    with rasterio.open(out_fname, 'w', **kwargs) as dst:
        dst.write(s2_data[:-1,:,:])
    return(True, s2_data[-1])


def reproject_sentinel2(lsat_fname, s2_fname, s2_aligned_fname, scale=3, resampling_method = rasterio.warp.Resampling.cubic_spline):
    """
    Reproject Sentinel-2 image to match projection of Landsat 8/9 image and rescale 
    """
    assert(os.path.exists(s2_fname) and os.path.exists(lsat_fname))
    with rasterio.open(lsat_fname) as lsat_src:
        with rasterio.open(s2_fname) as s2_src:
            nb = s2_src.count
            l,b,r,u = lsat_src.bounds # origin is l, u
            transform = rasterio.transform.from_bounds(l, b, r, u, scale*lsat_src.width, scale*lsat_src.height)
            if s2_src.width<scale*lsat_src.width or s2_src.height<scale*lsat_src.height:
                s2_dims = f"Sentinel-2 {s2_src.height}:{s2_src.width}"
                lsat_dims = f"Landsat comparison {3*lsat_src.height}:{3*lsat_src.width}"
                print(f"Looks like Sentinel-2 image smaller than expected: {s2_dims} {lsat_dims}")
                return(False)
            if s2_src.width>scale*np.sqrt(2)*lsat_src.width or s2_src.height>scale*np.sqrt(2)*lsat_src.height:
                s2_dims = f"Sentinel-2 {s2_src.height}:{s2_src.width}"
                lsat_dims = f"Landsat comparison {3*lsat_src.height}:{3*lsat_src.width}"
                print(f"Looks like Sentinel-2 image much larger than expected: {s2_dims} {lsat_dims}")
                return(False)
            kwargs = lsat_src.meta.copy() # use the projection from landsat
            kwargs.update({'transform': transform, 'width': scale*lsat_src.width, 'height': scale*lsat_src.height, 'count': nb, 'compress': 'zstd'})
            with rasterio.open(s2_aligned_fname, 'w', **kwargs) as dst:
                for band in range(1, nb + 1):
                    reproject(
                        source=rasterio.band(s2_src, band),
                        destination=rasterio.band(dst, band),
                        src_transform=s2_src.transform,
                        src_crs=s2_src.crs,
                        dst_transform=transform,
                        dst_crs=lsat_src.crs,
                        resampling=resampling_method)
    return(True)
  

def check_data(lsat_fname, s2_aligned_fname):
    """
    Check that aligned Sentinel-2 file has the same origin as Landsat image and
    that the sizes and resolutions are as expected.
    """
    with rasterio.open(lsat_fname) as src:
        x0, dx, _, y0, _, dy = src.transform.to_gdal()
        w = src.width
        h = src.height
    with rasterio.open(s2_aligned_fname) as src:
        if src.width != 3*w or src.height != 3*h:
            print(f"Landsat: {h}x{w}, Sentinel-2: {src.height}x{src.width}")
            return(False)
        X0, dX, _, Y0, _, dY = src.transform.to_gdal()
        if (np.abs(X0-x0)>0.5 or np.abs(Y0-y0)):
            print(f"Landsat origin {x0},{y0}; Sentinel-2 origin {X0},{Y0}")
            return(False)
        if np.abs(dX-10.0)>0.001 or np.abs(dY+10.0)>0.001:
            print(f"Unexpected resolution for Sentinel-2: {dX},{dY}")
            return(False)
    return(True)


def show_data(lsat_fname, s2_aligned_fname, sample=-1):
    """
    Show Landsat and aligned Sentinel-2 image side by side
    """
    with rasterio.open(s2_aligned_fname) as src:
        tmp = src.read()
    s2_img = s2_to_img((tmp/10000), percentile=(0.5, 99.5), gamma=1.0, bands = [2,1,0])
    with rasterio.open(lsat_fname) as src:
        tmp = src.read()
    lsat_img = s2_to_img((tmp/32616).clip(0.0,1.0), percentile=(0.5, 99.5), gamma=1.0, bands = [2,1,0])
    #rgb_img = (tmp[[2,1,0],:,:]/25000).clip(0.0, 1.0).transpose((1,2,0))
    f, (ax1, ax2) = plt.subplots(1, 2)
    ax1.imshow(s2_img)
    ax1.set_title(f'Sentinel-2 {sample=}')
    ax2.imshow(lsat_img)
    ax2.set_title(f'Landsat {sample=}')
    plt.show()


def get_s2_lsat_matches(year, bbox_of_interest):
    """
    Look for any day within the year with cloud-free Sentinel-2 and Landsat data.
    """
    time_of_interest = f"{year}-01-01/{year}-12-30"
    catalog = pystac_client.Client.open("https://planetarycomputer.microsoft.com/api/stac/v1", modifier=planetary_computer.sign_inplace)
    search = catalog.search(collections=["landsat-c2-l2"], 
                            bbox=bbox_of_interest, 
                            datetime=time_of_interest, 
                           query={
                               "eo:cloud_cover": {"lt": 10},
                               "platform": {"in": ["landsat-8", "landsat-9"]},
                           })
    lsat_items = search.item_collection()
    search = catalog.search(
        collections=["sentinel-2-l2a"],
        bbox=bbox_of_interest,
        datetime=time_of_interest,
        query={"eo:cloud_cover": {"lt": 10}},
    )
    s2_items = search.item_collection()
    lsat_by_time = {}
    for lsat_item in lsat_items:
        #lsat_dt = datetime.strptime(lsat_item.properties['datetime'], '%Y-%m-%dT%H:%M:%S.%fZ')
        lsat_dt = datetime.strptime(lsat_item.properties['datetime'][:19], '%Y-%m-%dT%H:%M:%S')
        lsat_by_time[lsat_dt.strftime('%m%d')] = lsat_item
    s2_by_time = {}
    for s2_item in s2_items:
        s2_dt = datetime.strptime(s2_item.properties['datetime'], '%Y-%m-%dT%H:%M:%S.%fZ')
        s2_dt = datetime.strptime(s2_item.properties['datetime'][:19], '%Y-%m-%dT%H:%M:%S')
        s2_by_time[s2_dt.strftime('%m%d')] = s2_item
    matches = {}
    for lsat_dt in lsat_by_time:
        if lsat_dt in s2_by_time:
            matches[lsat_dt] = (lsat_by_time[lsat_dt], s2_by_time[lsat_dt])
    return(matches)


def write_aligned_data(sample, lsat_item, s2_item, data_folder, bbox_of_interest, check_clouds=False, verbose=False):
    bbox, coord_box, at_edge = nesting_box(bbox_of_interest, lsat_item.assets['blue'].href, src_crs="EPSG:4326", cut_at_edge=False, multiplier=2)
    lsat_fname = os.path.join(data_folder, lsat_item.id + f'_dw_{sample}.tif')
    
    success = False
    if not at_edge:
        success, qpix = create_lsat_tif(lsat_item, lsat_fname, bbox, coord_box, verbose=verbose)
    else:
        if verbose:
            print(f"{sample=} discarding as box goes outside Landsat image.")
    if success and check_clouds:
        cloud_conf = (qpix & (2**8+2**9))//2**8
        shadow_conf = (qpix  & (2**10+2**11))//2**10
        cirrus_conf = (qpix & (2**14+2**15))//2**14
        bad_pix = np.logical_or(np.logical_or(cloud_conf>1, shadow_conf>1), cirrus_conf>2)
        if np.sum(bad_pix)>50:
            success = False
            if verbose:
                print(f"{sample=} discarding as Landsat has cloudy/shadowy pixels.")
    else:
        if verbose:
            print(f"{sample=} discarding as Landsat has missing data.")
    if at_edge or not success:
        if os.path.exists(lsat_fname):
            os.remove(lsat_fname)
        return(False, (None, None, None))
        
    # Use a slightly larger extent for Sentinel-2
    with rasterio.open(lsat_fname) as src:
        lsatbox = src.bounds
        lsat_crs = src.crs
    expanded_interest = rasterio.warp.transform_bounds(lsat_crs, "EPSG:4326", *lsatbox)
    bbox, coord_box, at_edge = nesting_box(lsatbox, s2_item.assets['B02'].href, src_crs=lsat_crs, cut_at_edge=False, multiplier=2)
    s2_fname =  os.path.join(data_folder, s2_item.id + f'_stacked_dw_{sample}.tif')
    success = False
    if not at_edge:
        success, scl = create_s2_tif(s2_item, s2_fname, bbox, coord_box, verbose=verbose)
    else:
        if verbose:
            print(f"{sample=} discarding as box goes outside Sentinel-2 image.")
    if success and check_clouds:
        cloud = (scl&(2**8+2**9))>0
        cirrus = (scl&2**10)>0
        shadow = (scl&2**3)>0
        s2_bad_pixels = np.logical_or(np.logical_or(cloud, shadow), cirrus)
        if np.sum(s2_bad_pixels)>50:
            success = False
            if verbose:
                print(f"{sample=} discarding as Sentinel-2 has cloudy/shadowy pixels.")
    else:
        if verbose:
            print(f"{sample=} discarding as Sentinel-2 has missing data.")

    if at_edge or not success:
        if os.path.exists(lsat_fname):
            os.remove(lsat_fname)
        if os.path.exists(s2_fname):
            os.remove(s2_fname)
        return(False, (None, None, None))
    s2_aligned_fname =  os.path.join(data_folder, s2_item.id + f'_stacked_reprojected_dw_{sample}.tif')
    success = reproject_sentinel2(lsat_fname, s2_fname, s2_aligned_fname, scale=3, resampling_method = rasterio.warp.Resampling.cubic_spline)
    if not success:
        if os.path.exists(lsat_fname):
            os.remove(lsat_fname)
        if os.path.exists(s2_fname):
            os.remove(s2_fname)
        if os.path.exists(s2_aligned_fname):
            os.remove(s2_aligned_fname)
        return(False, (None, None, None))
    check = check_data(lsat_fname, s2_aligned_fname)
    if not check:
        print(f"{sample=} failed.")
        assert(False)
    return(True, (lsat_fname, s2_fname, s2_aligned_fname))
