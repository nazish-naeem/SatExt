import rasterio
import numpy as np
from tqdm import tqdm
import os

IMG_EXTENSIONS = ['.jpg', '.JPG', '.jpeg', '.JPEG',
                  '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP','.tif']


def is_image_file(filename):
    return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)

def get_paths_from_images(path):
    assert os.path.isdir(path), '{:s} is not a valid directory'.format(path)
    images = []
    for dirpath, _, fnames in sorted(os.walk(path)):
        for fname in sorted(fnames):
            if is_image_file(fname):
                img_path = os.path.join(dirpath, fname)
                images.append(img_path)
    assert images, '{:s} has no valid image file'.format(path)
    return sorted(images)



path = '/home/t-nnaeem/workspace/RemoteSensingFoundationModels/Image-Super-Resolution-Iterative-Refinement/dataset/PC_S2_tif_subset'

paths = get_paths_from_images('{}'.format(path))
meanv=[]
varv=[]
stdv=[]
for index in tqdm(range(len(paths))):
    with rasterio.open(paths[index]) as src:
        d = (src.read()/10000.0)
        meanv.append(np.mean(d))
        varv.append(np.var(d))
        stdv.append(np.std(d))

print('Mean of mean', np.mean(meanv))

print('Mean of var', np.mean(varv))
print('Mean of std', np.mean(stdv))