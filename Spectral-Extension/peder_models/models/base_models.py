import torch
from models import srcnn, rrdb, rcan, rcan_sd, rrdb_sd


def predict_gpu(model, x):
    """
    USAGE: pred_y = predict_gpy(model, x)
    Given input x, run the model, clean the output and return data without removing from CPU.
    """
    with torch.no_grad():
        pred = model(x).clamp(0.0, 1.0)
    return(pred)


def predict(model, x):
    """
    USAGE: pred_y = predict(model, x)
    Given input x, run the model, clean the output and return data as numpy array. 
    """
    with torch.no_grad():
        pred = model(x).clamp(0.0, 1.0)
    pred_y = pred.detach().cpu().numpy()
    return(pred_y)


def predict_img(model, x):
    """
    USAGE: img = predict_img(model, x)
    Given input data create a viewable image from the model prediction.
    """
    pred_y = predict(model, x)
    pred_img = pred_y[0,:3,:,:].transpose([1,2,0])
    return(pred_img)
    

def srcnn_load_(cuda, num_in_band, checkpoint_fname):
    device = torch.device(cuda if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(checkpoint_fname, map_location='cpu')
    model = srcnn.Coreg_SRCNN_v1(num_in_band=num_in_band, num_out_band=4).to(device)
    model.load_state_dict(checkpoint)
    return(model)


def rrdb_load_(cuda, num_in_band, checkpoint_fname, nf=32, nb=8, gc=16, sf=5):
    device = torch.device(cuda if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(checkpoint_fname, map_location='cpu')
    model = rrdb.RRDBModel(in_nc=num_in_band, out_nc=4, nf=nf, nb=nb, gc=gc, scale_factor=sf).to(device)
    model.load_state_dict(checkpoint)
    return(model)


def rcan_load_(cuda, num_in_band, checkpoint_fname):
    device = torch.device(cuda if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(checkpoint_fname, map_location='cpu')
    model = rcan.RCAN(num_features=64, num_rg=10, num_rcab=20, scale=5, reduction=16, 
                      in_channels=num_in_band, out_channels=4).to(device)
    model.load_state_dict(checkpoint)
    return(model)



def srcnn_s2(cuda="cuda:0"):
    checkpoint_fname = "/mnt/sar-vision-data/image-sr/trained_models/base_s2_srcnn_s2_abus/loss-L1/psnr_25.3539.pth"
    model = srcnn_load_(cuda, 10, checkpoint_fname)
    return(model)


def rrdb_s2(cuda="cuda:0"):
    #checkpoint_fname = "/mnt/sar-vision-data/image-sr/trained_models/base_s2_rrdb_s2_abus/loss-MSE/ssim_0.8225.pth"
    checkpoint_fname = "/mnt/sar-vision-data/image-sr/trained_models/base_s2_rrdb_s2_abus/loss-L1/ssim_0.8264.pth"
    model = rrdb_load_(cuda, 10, checkpoint_fname, nb=8) # large model
    return(model)


def rrdb_plain_s2(cuda="cuda:0"):
    checkpoint_fname = "/mnt/sar-vision-data/image-sr/trained_models/base_s2_rrdb_s2_abus/loss-L1/ssim_0.8159.pth"
    model = rrdb_load_(cuda, 10, checkpoint_fname, nb=8) # large model
    return(model)


def rrdb_sd_s2(cuda="cuda:0"):
    checkpoint_fname = "/mnt/sar-vision-data/image-sr/trained_models/base_s2_rrdb_sd_s2_abus/loss-L1/ssim_0.8260.pth"
    #checkpoint_fname = "/mnt/sar-vision-data/image-sr/trained_models/base_s2_rrdb_sd_s2_abus/loss-L1/ssim_0.8302.pth"
    device = torch.device(cuda if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(checkpoint_fname, map_location='cpu')
    # L1, nb=8, lr=0.006, batch=40, lamb2=1e-3, nrdb=6
    model = rrdb_sd.RRDBModel(in_nc=10, out_nc=4, nf=32, nb=8, gc=16, scale_factor=5,
                              rdb_prob=0.0, rrdb_prob=0.5, num_rdb=6).to(device)
    #model = rrdb_sd.RRDBModel(in_nc=10, out_nc=4, nf=32, nb=12, gc=16, scale_factor=5,
    #                          rdb_prob=0.0, rrdb_prob=0.5, num_rdb=6).to(device)
    model.load_state_dict(checkpoint)
    return(model)


def rcan_plain_s2(cuda="cuda:0"):
    checkpoint_fname = "/mnt/sar-vision-data/image-sr/trained_models/base_s2_rcan_s2_abus/loss-L1/ssim_0.8099.pth"
    device = torch.device(cuda if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(checkpoint_fname, map_location='cpu')
    model = rcan.RCAN(num_features=64, num_rg=10, num_rcab=20, scale=5, reduction=16, 
                      in_channels=10, out_channels=4).to(device)
    model.load_state_dict(checkpoint)
    return(model)


def rcan_s2(cuda="cuda:0"):
    # checkpoint_fname = "/mnt/sar-vision-data/image-sr/trained_models/base_s2_rcan_s2_abus/loss-L1/ssim_0.8219.pth"
    checkpoint_fname = "/mnt/sar-vision-data/image-sr/trained_models/base_s2_rcan_s2_abus/loss-L1/ssim_0.8159.pth"
    device = torch.device(cuda if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(checkpoint_fname, map_location='cpu')
    model = rcan.RCAN(num_features=64, num_rg=10, num_rcab=20, scale=5, reduction=16, 
                      in_channels=10, out_channels=4).to(device)
    model.load_state_dict(checkpoint)
    return(model)


def rcan_dual_sd_s2(cuda="cuda:0"):
    checkpoint_fname = "/mnt/sar-vision-data/image-sr/trained_models/base_s2-landtypes_rcan_dual_sd_s2_abus/loss-L1/ssim_0.8431.pth"
    print("loading <%s>" % checkpoint_fname)
    device = torch.device(cuda if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(checkpoint_fname, map_location='cpu')
    params = 30, 10, 10, 20, 0.0, 0.5, 15, 4, 10, 10, 0.0, 0.5 
    nf_lr = params[0]
    num_rg_lr = params[1]
    num_rcab_lr = params[2]
    reduction_lr = params[3]
    rg_prob_lr = params[4]
    rcan_prob_lr = params[5]
    nf_hr = params[6]
    num_rg_hr = params[7]
    num_rcab_hr = params[8]
    reduction_hr = params[9]
    rg_prob_hr = params[10]
    rcan_prob_hr = params[11]
    model = rcan_sd.RCAN_dual(scale=5, num_features_lr=nf_lr, num_rg_lr=num_rg_lr, num_rcab_lr=num_rcab_lr, 
                              reduction_lr=reduction_lr,
                              in_channel_lr=10, rg_prob_lr=rg_prob_lr, rcan_prob_lr=rcan_prob_lr, num_features_hr=nf_hr,
                              num_rg_hr=num_rg_hr, num_rcab_hr=num_rcab_hr, reduction_hr=reduction_hr,in_channel_hr=4, 
                              rg_prob_hr=rg_prob_hr, rcan_prob_hr=rcan_prob_hr, out_channels=4, device = device).to(device)
    model.load_state_dict(checkpoint)
    return(model)


def rcan_sd_s2(cuda="cuda:0"):
    checkpoint_fname = "/mnt/sar-vision-data/image-sr/trained_models/base_s2_rcan_sd_s2_abus/loss-L1/ssim_0.8169.pth"
    device = torch.device(cuda if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(checkpoint_fname, map_location='cpu')
    model = rcan_sd.RCAN(num_features=64, num_rg=10, num_rcab=30, scale=5, reduction=16, 
                      in_channels=10, out_channels=4, rg_prob=0.0, rcan_prob=0.5).to(device)
    model.load_state_dict(checkpoint)
    return(model)


def srcnn_ensemble2(cuda='cuda:0'):
    checkpoint = "/mnt/sar-vision-data/image-sr/trained_models/base_ensemble2_srcnn_s2_abus/loss-MSE/psnr_26.2477.pth"
    #checkpoint = "/mnt/sar-vision-data/image-sr/trained_models/base_ensemble2_srcnn_s2_abus/loss-MSE/psnr_26.4012.pth"
    model = srcnn_load_(cuda, 12, checkpoint)
    return(model)


def srcnn_lt_ensemble2(cuda='cuda:0'):
    checkpoint = "/mnt/sar-vision-data/image-sr/trained_models/base_lt_ensemble2_srcnn_s2_abus/loss-MSE/psnr_26.6973.pth"
    model = srcnn_load_(cuda, 12, checkpoint)
    return(model)
    

def srcnn_landtypes(cuda='cuda:0'):
    checkpoint_fname = '/mnt/sar-vision-data/image-sr/trained_models/base_landtypes_srcnn_s2_abus/loss-L1/psnr_25.7502.pth'
    device = torch.device(cuda if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(checkpoint_fname, map_location='cpu')
    model = srcnn.Coreg_SRCNN_v1(num_in_band=14, num_out_band=4).to(device)
    model.load_state_dict(checkpoint)
    return(model)


def rrdb_landtypes(cuda="cuda:0"):
    checkpoint_fname = '/mnt/sar-vision-data/image-sr/trained_models/base_landtypes_rrdb_s2_abus/loss-L1/ssim_0.8351.pth'
    device = torch.device(cuda if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(checkpoint_fname, map_location='cpu')
    model = rrdb.RRDBModel(in_nc=14, out_nc=4, nf=32, nb=3, gc=16, scale_factor=1).to(device) # nb = 8 or 3?
    model.load_state_dict(checkpoint)
    return(model)


def srcnn_landtypes_all(cuda='cuda:0'):
    checkpoint_fname = '/mnt/sar-vision-data/image-sr/trained_models/base_lt_ensemble2_srcnn_s2_abus/loss-MSE/psnr_26.6973.pth'
    #checkpoint_fname = '/mnt/sar-vision-data/image-sr/trained_models/base_landtypes_all_srcnn_s2_abus/loss-MSE/psnr_26.1783.pth'
    device = torch.device(cuda if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(checkpoint_fname, map_location='cpu')
    model = srcnn.Coreg_SRCNN_v1(num_in_band=16, num_out_band=4).to(device)
    model.load_state_dict(checkpoint)
    return(model)



# do not use these as they evaluated the results on the training data... 
def srcnn_s2_(cuda="cuda:0"):
    checkpoint_fname = "/mnt/sar-vision-data/image-sr/trained_models/ensemble_s2_srcnn_s2_abus/loss-L1/psnr_27.1885.pth"
    model = srcnn_load_(cuda, 10, checkpoint_fname)
    return(model)


def srcnn_s210m_(cuda="cuda:0"):
    checkpoint_fname = "/mnt/sar-vision-data/image-sr/trained_models/ensemble_s210m_srcnn_s2_abus/loss-MSE/psnr_26.0839.pth"
    model = srcnn_load_(cuda, 4, checkpoint_fname)
    return(model)


def rrdb_s2_(cuda="cuda:0"):
    checkpoint_fname = "/mnt/sar-vision-data/image-sr/trained_models/ensemble_s2_rrdb_s2_abus/loss-L1/ssim_0.8934.pth"
    model = rrdb_load_(cuda, 10, checkpoint_fname)
    return(model)


def rrdb_s210m_(cuda="cuda:0"):
    checkpoint_fname = "/mnt/sar-vision-data/image-sr/trained_models/ensemble_s210m_rrdb_s2_abus/loss-MSE/ssim_0.8646.pth"
    model = rrdb_load_(cuda, 4, checkpoint_fname, nb=3)
    return(model)

    
def rcan_s210m_(cuda="cuda:0"):
    checkpoint_fname = "/mnt/sar-vision-data/image-sr/trained_models/ensemble_s210m_rcan_s2_abus/loss-L1/ssim_0.9101.pth"
    model = rcan_load_(cuda, 4, checkpoint_fname)
    return(model)


def rcan_s2_(cuda="cuda:0"):
    checkpoint_fname = "/mnt/sar-vision-data/image-sr/trained_models/ensemble_s2_rcan_s2_abus/loss-L1/ssim_0.9159.pth"
    device = torch.device(cuda if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(checkpoint_fname, map_location='cpu')
    model = rcan.RCAN(num_features=64, num_rg=10, num_rcab=20, scale=5, reduction=16, 
                      in_channels=10, out_channels=4).to(device)
    model.load_state_dict(checkpoint)
    return(model)


def rcan_landtypes(cuda="cuda:0"):
    # checkpoint_fname = '/mnt/sar-vision-data/image-sr/trained_models/ensemble_landtypes_rcan_s2_abus/loss-L1/ssim_0.9474_frozen.pth'
    checkpoint_fname = '/mnt/sar-vision-data/image-sr/trained_models/ensemble_landtypes_rcan_s2_abus/loss-L1/ssim_0.9510.pth'
    device = torch.device(cuda if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(checkpoint_fname, map_location='cpu')
    model = rcan.RCAN(scale=5, num_features=64, num_rg=10, num_rcab=20, reduction=16, in_channels=14, out_channels=4).to(device)
    model.load_state_dict(checkpoint)
    return(model)


def rrdb_landtypes_both(cuda="cuda:0"):
    checkpoint_fname = "/mnt/sar-vision-data/image-sr/trained_models/ensemble_landtypes_both_rrdb_1_s2_abus/loss-MSE/ssim_0.9643.pth"
    device = torch.device(cuda if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(checkpoint_fname, map_location='cpu')
    model = rrdb.RRDBModel(in_nc=12, out_nc=4, nf=32, nb=3, gc=16, scale_factor=1).to(device)
    model.load_state_dict(checkpoint)
    return(model)
