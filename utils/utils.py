from PIL import Image
from vibe_core.client import get_default_vibe_client
import rasterio


def get_low_res(img,ratio):
    #this function returns low resolution images with 1. size/ratio and 2) same size
    orig_size = img.size
    img.thumbnail(int(orig_size/ratio))
    img_org_size = img.transform(orig_size, Image.EXTENT, (0,0, orig_size[0]/ratio, orig_size[1]/ratio))
    return img, img_org_size

def load_tifs(client,id):
    run_id = client.list_runs()[id] # will list the most recent run_id
    run = client.get_run_by_id(run_id)

    # with rasterio.open(run.output['raster'][0].assets[0].path_or_url) as src:
    #     print(src.meta)
    return rasterio.open(run.output['raster'][0].assets[0].path_or_url)

