import torch

import argparse
import logging
import sys
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
sys.path.append('Image-Super-Resolution-Iterative-Refinement')
import core.logger as Logger
import core.metrics as Metrics
from core.wandb_logger import WandbLogger
import model as Model
sys.path.append('Spectral-Extension')
from dataloader.data_loader import LSS2Dataset
from models.AE import CNN_Encoder, CNN_Decoder, AE
from peder_models.models.rcan import RCAN
from peder_models.models import rrdb 
from models.fconv import FConv
from torch.autograd import Variable
import skimage
import tensorflow as tf
# from tensorboardX import SummaryWriter
import os
import sys
import torch.nn.functional as F
import numpy as np
import torchvision
import pickle
def visualize(visuals,t,result_path,current_step,idx):
    t_step = t
    
    if t==len(visuals['SR'][:,0,0,0])-1:
        # print('mean val sr ', t)
        ggh = Metrics.tensor2numpy(visuals['SR'][t_step,:,:,:],(-1,1))
        # print(np.mean(ggh))

        print('mean val hr ', t)
        ggh = Metrics.tensor2numpy(visuals['HR'][0,:,:,:],(-1,1))
        print(np.mean(ggh))

    sr_img = Metrics.s2_to_img(visuals['SR'][t_step,:,:,:])
    hr_img = Metrics.s2_to_img(visuals['HR'][0,:,:,:])
    lr_img = Metrics.s2_to_img(visuals['LR'][0,:,:,:])
    fake_img = Metrics.s2_to_img(visuals['INF'][0,:,:,:])


    Metrics.save_img(
        hr_img, '{}/{}/{}_{}_hr.png'.format(result_path, idx, current_step,t_step))
    Metrics.save_img(
        sr_img,'{}/{}/{}_{}_sr.png'.format(result_path, idx, current_step,t_step))
    Metrics.save_img(
        lr_img, '{}/{}/{}_{}_lr.png'.format(result_path, idx, current_step,t_step))
    Metrics.save_img(
        fake_img, '{}/{}/{}_{}_inf.png'.format(result_path, idx, current_step,t_step))


def loss_fun(pred,out):
    return F.smooth_l1_loss(pred,out)

def f_pass(inp,out,model):
    inp = inp.float()
    out = out.float()
    inp = Variable(inp)
    out = Variable(out)
    
    if args.cuda:
        inp, out = inp.cuda(), out.cuda()

    # optimizer.zero_grad()

    pred = model(inp)
    loss = loss_fun(pred,out)

    # loss.backward()
    # optimizer.step()

    return loss,pred


def HR_to_LR_SR(LR,lr = 128, hr = 384):
    # SR = np.zeros((LR.shape[0],LR.shape[1],hr,hr))
    # for j in range(SR.shape[1]):
    #     for i in range(SR.shape[0]):
    #         SR[i,j,:,:] = skimage.transform.resize(LR[i,j,:,:], SR[i,j,:,:].shape, order=3,preserve_range=True) #resize_and_convert(img_LR[i,j,:,:], hr, Image.BICUBIC)
    SR = F.interpolate(LR,size=(hr,hr),mode='bicubic')
    return LR,SR

def load_spect_ext_model(args):
    if args.model=='AE':
        model = AE((6, 128, 128),10)
    if args.model=='FCONV':
        model = FConv((6, 128, 128))
    if args.model=='RCAN':
        model = RCAN(num_features=64, num_rg=5, num_rcab=10, scale=1, reduction=16, in_channels=6, out_channels=10)

    if args.model=='RRDB':
        model = rrdb.RRDBModel(in_nc=6, out_nc=10, nf=32, nb=8, gc=16, scale_factor=1)

    if args.cuda:
        model = nn.DataParallel(model)
        model.cuda()
        print('GPU',torch.cuda.is_available())

    state_dict = torch.load(args.resume_path)
    model.load_state_dict(state_dict['state_dict'])
    return model


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', type=str, default='Image-Super-Resolution-Iterative-Refinement/config/sr_S2_RGB.json',
                        help='JSON file for configuration')
    parser.add_argument('-p', '--phase', type=str, choices=['val'], help='val(generation)', default='val')
    parser.add_argument('-gpu', '--gpu_ids', type=str, default=None)
    parser.add_argument('-debug', '-d', action='store_true')
    parser.add_argument('-enable_wandb', action='store_true')
    parser.add_argument('-log_infer', action='store_true')
    parser.add_argument('-b', '--batch_size', type=float, default=1)
    parser.add_argument('--model', type=str, default='RCAN', help='type of the model that is being used')
    parser.add_argument('--resume_path', type=str, default='/home/t-nnaeem/workspace/RemoteSensingFoundationModels/Spectral-Extension/experiments/spectral_extension_20240719-162755/checkpoint/checkpoint_E40.tar', help='pretrained model path')
    parser.add_argument('--data_path', type=str, default='/home/t-nnaeem/workspace/RemoteSensingFoundationModels/Spectral-Extension/data', help='training data path')
    parser.add_argument('--cuda', type=bool, default=True, help='use gpu')
    parser.add_argument('--no-cuda', action='store_true', default=False)
    parser.add_argument('-n', '--num_workers', type=float, default=15)
    

    
    # parse configs
    args = parser.parse_args()
    opt = Logger.parse(args)
    # Convert to NoneDict, which return None for missing key.
    opt = Logger.dict_to_nonedict(opt)

    # logging
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True


    args.cuda = not args.no_cuda and torch.cuda.is_available()
    
    model = load_spect_ext_model(args)

    dataset_test = LSS2Dataset(args.data_path, split='test', data_len=-1)

    test_loader = torch.utils.data.DataLoader(
                dataset_test,
                batch_size=args.batch_size,
                shuffle=True,
                num_workers=args.num_workers,
                pin_memory=True)


    # model
    diffusion = Model.create_model(opt)


    diffusion.set_new_noise_schedule(
        opt['model']['beta_schedule']['val'], schedule_phase='val')
    


    idx = 0

    result_path = '{}'.format(opt['path']['results'])
    os.makedirs(result_path, exist_ok=True)
    
    # totensor = torchvision.transforms.ToTensor()
    print(result_path)
    current_step=0
    for _,  test_data in enumerate(test_loader):
        idx += 1

        inp, out, index = test_data['LS'], test_data['S2'], test_data['Index']
        model.eval()
        with torch.no_grad():
            loss,pred = f_pass(inp,out,model)
        print(pred.type)
        img_LR, img_SR = HR_to_LR_SR(pred,lr = 128, hr = 384)
        print(pred.shape)
        val_data = {'HR': test_data['S2_HR'], 'SR': torch.tensor(img_SR), 'Index': index}


        ## add val data here
        diffusion.feed_data(val_data)
        diffusion.test(continous=True)
        visuals = diffusion.get_current_visuals(need_LR=False)

        hr_img = Metrics.tensor2img(visuals['HR'])  # uint8
        fake_img = Metrics.tensor2img(visuals['INF'])  # uint8
        os.makedirs(result_path+'/'+str(idx)+'/', exist_ok=True)
        # for t_step in range(len(visuals['SR'][:,0,0,0])):
        #     visualize(visuals,t_step,result_path,current_step,idx)
        t_step = -1
        visualize(visuals,t_step,result_path,current_step,idx)
        
        # sr_img = Metrics.tensor2img(visuals['SR'])  # uint8
        # Metrics.save_img(
        #     sr_img, '{}/{}/{}_sr_process.png'.format(result_path, idx, current_step))
        # Metrics.save_img(
        #     Metrics.tensor2img(visuals['SR'][-1]), '{}/{}/{}_sr.png'.format(result_path, idx, current_step))

        # Metrics.save_img(
        #     hr_img, '{}/{}/{}_hr.png'.format(result_path, idx, current_step))
        # Metrics.save_img(
        #     fake_img, '{}/{}/{}_inf.png'.format(result_path, idx, current_step))
        visuals['LS'] = test_data['LS']
        with open('{}/{}/{}.pkl'.format(result_path, idx, current_step), "wb") as f:
            pickle.dump(visuals, f)
