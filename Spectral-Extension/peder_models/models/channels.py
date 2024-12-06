import os
from utils import tiff
import numpy as np
import torch
from utils.utils import get_airbus_fname

def read_s2(loc_info, loc_map, test=True, image_no=0, idx=-1, upsample=False):
    """
    Use image_no if indexing relative to test_coreg or train_coreg, otherwise use idx directly.
    """
    if idx ==-1:
        if test:
            idx = loc_info['test_coreg'][image_no]
        else:
            idx = loc_info['train_coreg'][image_no]
    key = list(loc_map.keys())[idx]
    s2_fname = os.path.join(loc_info['s2_dir'],key+'.tif')
    if upsample:
        s2_fname = os.path.join(loc_info['s2_dir'],'orig_upsample_'+key+'.tif')
    s2 = tiff.read(s2_fname)
    return(s2)


def read_abus(loc_info, loc_map, test=True, image_no=0, idx=-1):
    if idx ==-1:
        if test:
            idx = loc_info['test_coreg'][image_no]
        else:
            idx = loc_info['train_coreg'][image_no]
    key = list(loc_map.keys())[idx]
    abus_dir = os.path.join(loc_info["airbus_dir"], 'reprojected')
    ms_fname = os.path.basename(get_airbus_fname(loc_info["airbus_dir"] + loc_map[key], 'img', 'MS'))
    abus_fname = os.path.join(abus_dir, ms_fname)
    abus = tiff.read(abus_fname)
    return(abus)


    

def read_svec(loc_info, loc_map):
    abus_dir = os.path.join(loc_info["airbus_dir"], 'reprojected')
    svec_fname = os.path.join(loc_info["airbus_dir"], 'landtypes', "singular_vectors.tif")
    svec = tiff.read(svec_fname)
    svec = svec.transpose((1,2,0))
    return(svec)
    
    
    
def read_landtype(loc_info, loc_map, test=True, image_no=0, idx=-1, downsample=False):
    if idx ==-1:
        if test:
            idx = loc_info['test_coreg'][image_no]
        else:
            idx = loc_info['train_coreg'][image_no]
    key = list(loc_map.keys())[idx]
    abus_dir = os.path.join(loc_info["airbus_dir"], 'reprojected')
    ms_fname = os.path.basename(get_airbus_fname(loc_info["airbus_dir"] + loc_map[key], 'img', 'MS'))
    landtypes_fname = os.path.join(loc_info["airbus_dir"], 'base_landtypes', f"predicted_{ms_fname}")
    landtype = tiff.read(landtypes_fname)
    return(landtype)


def read_model_pred_(loc_info, loc_map, model_name, landtypes=False, test=True, image_no=0, downsample=False):
    if test:
        idx = loc_info['test_coreg'][image_no]
    else:
        idx = loc_info['train_coreg'][image_no]
    key = list(loc_map.keys())[idx]
    abus_dir = os.path.join(loc_info["airbus_dir"], 'reprojected')
    ms_fname = os.path.basename(get_airbus_fname(loc_info["airbus_dir"] + loc_map[key], 'img', 'MS'))
    down = ''
    if downsample:
        down = '_down'
    tag = 'base_predictions'
    if landtypes:
        tag = 'base_landtypes_predictions'
    fname = os.path.join(loc_info["airbus_dir"], f'{model_name}_{tag}', f"predicted{down}_{ms_fname}")
    #print("reading <%s>" % fname)
    return(tiff.read(fname))    


def read_srcnn(loc_info, loc_map, landtypes=False, test=True, image_no=0, downsample=False):
    return(read_model_pred_(loc_info, loc_map, 'srcnn', landtypes=landtypes, test=test, image_no=image_no, downsample=downsample))

def read_rrdb(loc_info, loc_map, landtypes=False, test=True, image_no=0, downsample=False):
    return(read_model_pred_(loc_info, loc_map, 'rrdb', landtypes=landtypes, test=test, image_no=image_no, downsample=downsample))

def read_rrdb_plain(loc_info, loc_map, landtypes=False, test=True, image_no=0, downsample=False):
    return(read_model_pred_(loc_info, loc_map, 'rrdb_plain', landtypes=landtypes, test=test, image_no=image_no, downsample=downsample))

def read_rrdb_sd(loc_info, loc_map, landtypes=False, test=True, image_no=0, downsample=False):
    return(read_model_pred_(loc_info, loc_map, 'rrdb_sd', landtypes=landtypes, test=test, image_no=image_no, downsample=downsample))

def read_rcan(loc_info, loc_map, landtypes=False, test=True, image_no=0, downsample=False):
    return(read_model_pred_(loc_info, loc_map, 'rcan', landtypes=landtypes, test=test, image_no=image_no, downsample=downsample))

def read_rcan_sd(loc_info, loc_map, landtypes=False, test=True, image_no=0, downsample=False):
    return(read_model_pred_(loc_info, loc_map, 'rcan_sd', landtypes=landtypes, test=test, image_no=image_no, downsample=downsample))

def read_rcan_plain(loc_info, loc_map, landtypes=False, test=True, image_no=0, downsample=False):
    return(read_model_pred_(loc_info, loc_map, 'rcan_plain', landtypes=landtypes, test=test, image_no=image_no, downsample=downsample))

def read_rcan_dual_sd(loc_info, loc_map, test=True, image_no=0, downsample=False):
    return(read_model_pred_(loc_info, loc_map, 'rcan_dual_sd', landtypes=False, test=test, image_no=image_no, downsample=downsample))


def landtype_s2_torch(landtype, s2, cuda=''):
    x = np.concatenate((s2, landtype), axis = 0)
    if cuda:
        device = torch.device(cuda if torch.cuda.is_available() else "cpu")
        x = torch.from_numpy(x[np.newaxis,...]).float().to(device)
    return(x)
    
    
def landtype_both_torch(landtype, pred_srcnn, pred_rrdb, cuda=''):
    x = np.concatenate((pred_srcnn, pred_rrdb, landtype), axis = 0)
    if cuda:
        device = torch.device(cuda if torch.cuda.is_available() else "cpu")
        x = torch.from_numpy(x[np.newaxis,...]).float().to(device)
    return(x)


def landtype_all_torch(landtype, pred_srcnn, pred_rrdb, pred_rcan, cuda=''):
    x = np.concatenate((pred_srcnn, pred_rrdb, pred_rcan, landtype), axis = 0)
    if cuda:
        device = torch.device(cuda if torch.cuda.is_available() else "cpu")
        x = torch.from_numpy(x[np.newaxis,...]).float().to(device)
    return(x)


# 'Operator'-s used by train_ensemble.py and evaluate_base_model.py

def s2_raw(dset, data, box):
    x1, x2, y1, y2 = box
    location = dset['location']
    curr_idx = dset['curr_idx']
    x = data[location, curr_idx, "s2"][:,x1:x2,y1:y2]
    return(x)


def s2_10m(dset, data, box): # identical to s2_raw...
    x1, x2, y1, y2 = box
    location = dset['location']
    curr_idx = dset['curr_idx']
    x = data[location, curr_idx, "s2"][:,x1:x2,y1:y2]
    return(x)


def ensemble2(dset, data, box):
    x1, x2, y1, y2 = box
    location = dset['location']
    curr_idx = dset['curr_idx']
    #print("data keys:", list(data.keys()))
    srcnn = data[location, curr_idx, "srcnn"][:,x1:x2,y1:y2]
    rrdb = data[location, curr_idx, "rrdb"][:,x1:x2,y1:y2]
    rcan = data[location, curr_idx, "rcan"][:,x1:x2,y1:y2]
    x = np.concatenate((srcnn, rrdb, rcan), axis = 0)
    return(x)


def lt_ensemble2(dset, data, box):
    #print("dataset-keys:", list(data.keys()))
    x1, x2, y1, y2 = box
    location = dset['location']
    curr_idx = dset['curr_idx']
    #print("data keys:", list(data.keys()))
    srcnn = data[location, curr_idx, "srcnn_landtypes"][:,x1:x2,y1:y2]
    rrdb = data[location, curr_idx, "rrdb_landtypes"][:,x1:x2,y1:y2]
    rcan = data[location, curr_idx, "rcan_landtypes"][:,x1:x2,y1:y2]
    x = np.concatenate((srcnn, rrdb, rcan), axis = 0)
    return(x)


def svec(dset, data, box):
    x1, x2, y1, y2 = box
    location = dset['location']
    curr_idx = dset['curr_idx']
    s2 = data[location, curr_idx, "s2"][:,x1:x2,y1:y2]
    svec = data[location, curr_idx, "singular_vectors"][:,x1:x2,y1:y2]
    x = np.concatenate((s2, svec), axis = 0)
    return(x)


def landtypes(dset, data, box):
    x1, x2, y1, y2 = box
    location = dset['location']
    curr_idx = dset['curr_idx']
    s2 = data[location, curr_idx, "s2"][:,x1:x2,y1:y2]
    landtypes = data[location, curr_idx, "landtypes"][:,x1:x2,y1:y2]
    x = np.concatenate((s2, landtypes), axis = 0)
    return(x)


def landtypes_srcnn(dset, data, box):
    x1, x2, y1, y2 = box
    location = dset['location']
    curr_idx = dset['curr_idx']
    s2 = data[location, curr_idx, "srcnn"][:,x1:x2,y1:y2]
    landtypes = data[location, curr_idx, "landtypes"][:,x1:x2,y1:y2]
    x = np.concatenate((s2, landtypes), axis = 0)
    return(x)


def landtypes_rrdb(dset, data, box):
    x1, x2, y1, y2 = box
    location = dset['location']
    curr_idx = dset['curr_idx']
    s2 = data[location, curr_idx, "rrdb"][:,x1:x2,y1:y2]
    landtypes = data[location, curr_idx, "landtypes"][:,x1:x2,y1:y2]
    x = np.concatenate((s2, landtypes), axis = 0)
    return(x)


def landtypes_both(dset, data, box):
    x1, x2, y1, y2 = box
    location = dset['location']
    curr_idx = dset['curr_idx']
    srcnn = data[location, curr_idx, "srcnn"][:,x1:x2,y1:y2]
    rrdb = data[location, curr_idx, "rrdb"][:,x1:x2,y1:y2]
    landtypes = data[location, curr_idx, "landtypes"][:,x1:x2,y1:y2]
    x = np.concatenate((srcnn, rrdb, landtypes), axis = 0)
    return(x)


def s2_diff(dset, data, box):
    x1, x2, y1, y2 = box
    location = dset['location']
    curr_idx = dset['curr_idx']
    past_idx = dset['past_idx']
    s2_diff = data[location, curr_idx, "s2"][:,x1:x2,y1:y2] - data[location, past_idx, "s2"][:,x1:x2,y1:y2]
    past_ms = data[location, past_idx, "ms"][:,x1:x2,y1:y2]/256.
    x = np.concatenate((past_ms, s2_diff), axis = 0)
    return(x)


def srcnn_diff(dset, data, box):
    x1, x2, y1, y2 = box
    location = dset['location']
    curr_idx = dset['curr_idx']
    past_idx = dset['past_idx']
    s2_diff = data[location, curr_idx, "srcnn"][:,x1:x2,y1:y2] - data[location, past_idx, "srcnn"][:,x1:x2,y1:y2]
    past_ms = data[location, past_idx, "ms"][:,x1:x2,y1:y2]/256.
    x = np.concatenate((past_ms, s2_diff), axis = 0)
    return(x)


def srcnn_diff2(dset, data, box):
    x1, x2, y1, y2 = box
    location = dset['location']
    curr_idx = dset['curr_idx']
    past_idx = dset['past_idx']
    s2_diff = data[location, curr_idx, "srcnn"][:,x1:x2,y1:y2] - data[location, past_idx, "srcnn"][:,x1:x2,y1:y2]
    past_ms = data[location, past_idx, "ms"][:,x1:x2,y1:y2]/256.
    x = past_ms + s2_diff
    return(x)


def srcnn_stack(dset, data, box):
    x1, x2, y1, y2 = box
    location = dset['location']
    curr_idx = dset['curr_idx']
    past_idx = dset['past_idx']
    srcnn_curr = data[location, curr_idx, "srcnn"][:,x1:x2,y1:y2]
    srcnn_past = data[location, past_idx, "srcnn"][:,x1:x2,y1:y2]
    past_ms = data[location, past_idx, "ms"][:,x1:x2,y1:y2]/256.
    x = np.concatenate((past_ms, srcnn_past, srcnn_curr), axis = 0)
    return(x)


def srcnn_stack2(dset, data, box):
    x1, x2, y1, y2 = box
    location = dset['location']
    curr_idx = dset['curr_idx']
    past_idx = dset['past_idx']
    srcnn_curr = data[location, curr_idx, "srcnn"][:,x1:x2,y1:y2]
    srcnn_past = data[location, past_idx, "srcnn"][:,x1:x2,y1:y2]
    srcnn_diff = srcnn_curr-srcnn_past
    past_ms = data[location, past_idx, "ms"][:,x1:x2,y1:y2]/256.
    ms_curr_estimated = past_ms + srcnn_diff
    x = np.concatenate((ms_curr_estimated, srcnn_curr), axis = 0)
    return(x)


def rrdb_diff(dset, data, box):
    x1, x2, y1, y2 = box
    location = dset['location']
    curr_idx = dset['curr_idx']
    past_idx = dset['past_idx']
    s2_diff = data[location, curr_idx, "rrdb"][:,x1:x2,y1:y2] - data[location, past_idx, "rrdb"][:,x1:x2,y1:y2]
    past_ms = data[location, past_idx, "ms"][:,x1:x2,y1:y2]/256.
    x = np.concatenate((past_ms, s2_diff), axis = 0)
    return(x)


def rrdb_diff2(dset, data, box):
    x1, x2, y1, y2 = box
    location = dset['location']
    curr_idx = dset['curr_idx']
    past_idx = dset['past_idx']
    s2_diff = data[location, curr_idx, "rrdb"][:,x1:x2,y1:y2] - data[location, past_idx, "rrdb"][:,x1:x2,y1:y2]
    past_ms = data[location, past_idx, "ms"][:,x1:x2,y1:y2]/256.
    x = past_ms + s2_diff
    return(x)


def rrdb_stack(dset, data, box):
    x1, x2, y1, y2 = box
    location = dset['location']
    curr_idx = dset['curr_idx']
    past_idx = dset['past_idx']
    rrdb_curr = data[location, curr_idx, "rrdb"][:,x1:x2,y1:y2] 
    rrdb_past = data[location, past_idx, "rrdb"][:,x1:x2,y1:y2]
    past_ms = data[location, past_idx, "ms"][:,x1:x2,y1:y2]/256.
    x = np.concatenate((past_ms, rrdb_past, rrdb_curr), axis = 0)
    return(x)


def rrdb_stack2(dset, data, box):
    x1, x2, y1, y2 = box
    location = dset['location']
    curr_idx = dset['curr_idx']
    past_idx = dset['past_idx']
    rrdb_curr = data[location, curr_idx, "rrdb"][:,x1:x2,y1:y2] 
    rrdb_past = data[location, past_idx, "rrdb"][:,x1:x2,y1:y2]
    rrdb_diff = rrdb_curr-rrdb_past
    past_ms = data[location, past_idx, "ms"][:,x1:x2,y1:y2]/256.
    curr_ms_estimated = rrdb_diff + past_ms
    x = np.concatenate((curr_ms_estimated, rrdb_curr), axis = 0)
    return(x)
