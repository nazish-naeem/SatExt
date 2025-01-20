import torch
from torchvision import models
from torchsummary import summary
import torchvision.transforms as transforms
from models.AE import CNN_Encoder, CNN_Decoder, AE
from peder_models.models.rcan import RCAN
from peder_models.models import rrdb 
from models.fconv import FConv
from dataloader.data_loader_baseline import LSS2Dataset
from torch.utils.tensorboard import SummaryWriter
import helper.metrics as Metrics
import argparse
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from tqdm import tqdm
from torch.autograd import Variable
from helper.metrics import calculate_ssim, calculate_psnr
import matplotlib.pyplot as plt
import numpy as np
import os
import sys

parser = argparse.ArgumentParser()
parser.add_argument('-p', '--phase', type=str, choices=['train', 'val'],
                    help='Run either train(training) or val(generation)', default='train')
parser.add_argument('-gpu', '--gpu_ids', type=str, default=None)
parser.add_argument('-b', '--batch_size', type=float, default=1)
parser.add_argument('-n', '--num_workers', type=float, default=15)
parser.add_argument('-sh', '--shuffle', type=bool, default=True)
# parser.add_argument('-e', '--epoch', type=int, default=1000)
parser.add_argument('--no-cuda', action='store_true', default=False)
parser.add_argument('--seed', type=int, default=1)
parser.add_argument('--resume_path', type=str, default='/home/t-nnaeem/workspace/RemoteSensingFoundationModels/Spectral-Extension/experiments_baseline/spectral_extension_20240724-031130/checkpoint/checkpoint_E80.tar', help='pretrained model path')
# parser.add_argument('--lr', type=float, default=0.0001, help='learning rate')
parser.add_argument('--data_path', type=str, default='/home/t-nnaeem/workspace/RemoteSensingFoundationModels/Spectral-Extension/data', help='training data path')
parser.add_argument('--cuda', type=bool, default=True, help='use gpu')
# parser.add_argument('--save_freq', type=float, default=10, help='frequency of saving the model and results')
parser.add_argument('--model', type=str, default='RRDB', help='type of the model that is being used')
args = parser.parse_args()


args.cuda = not args.no_cuda and torch.cuda.is_available()
torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)
    print('Cuda is true')


transform = transforms.Compose(
    [transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))])
if args.model=='AE':
    model = AE((6, 128, 128),10)
if args.model=='FCONV':
    model = FConv((6, 128, 128))
if args.model=='RCAN':
    model = RCAN(num_features=64, num_rg=5, num_rcab=10, scale=3, reduction=16, in_channels=6, out_channels=10)

if args.model=='RRDB':
    model = rrdb.RRDBModel(in_nc=6, out_nc=10, nf=32, nb=8, gc=16, scale_factor=3)

dataset_test = LSS2Dataset(args.data_path, split='test', data_len=-1)

test_loader = torch.utils.data.DataLoader(
            dataset_test,
            batch_size=args.batch_size,
            shuffle=args.shuffle,
            num_workers=args.num_workers,
            pin_memory=True)

if args.cuda:
    model = nn.DataParallel(model)
    model.cuda()
    print('GPU',torch.cuda.is_available())

state_dict = torch.load(args.resume_path)
model.load_state_dict(state_dict['state_dict'])

print('Number of parameters in the Spectral Extension model: {}'.format(sum([p.data.nelement() for p in model.parameters()])) )

def loss_fun(pred,out):
    return F.smooth_l1_loss(pred,out)

def visualize(data,f_name):
    hr_img = Metrics.s2_to_img(data)
    Metrics.save_img(
        hr_img, f_name)

def visualize_band(data,f_name,band):
    hr_img = Metrics.s2_to_img(data,percentile=(0.5, 99.5), gamma=0.9, bands = band,min_max=(0, 1))
    Metrics.save_img(
        hr_img, f_name)

def psnr_band(pred,output,band=[2,1,0]):
    '''
    Given the pred and output of the size B,C,H,W with B=1, the function outputs the psnr of the image of the given band. for color images band can have length 3 and for grayscale image band can be a list of one element
    '''
    if len(band)>3:
        raise ValueError('band length >3')
    if len(band)==2:
        raise ValueError('band length =2')
    if len(band)==3:
        img = Metrics.s2_to_img(pred,percentile=(0.5, 99.5), gamma=0.9, bands = band,min_max=(0, 1))
        gt = Metrics.s2_to_img(output,percentile=(0.5, 99.5), gamma=0.9, bands = band,min_max=(0, 1))
        return calculate_psnr(img,gt)
    if len(band)==1:
        img = Metrics.s2_to_img(pred,percentile=(0.5, 99.5), gamma=0.9, bands = band,min_max=(0, 1))
        gt = Metrics.s2_to_img(output,percentile=(0.5, 99.5), gamma=0.9, bands = band,min_max=(0, 1))
        return calculate_psnr(img,gt)
    
def ssim_band(pred,output,band=[2,1,0]):
    '''
    Given the pred and output of the size B,C,H,W with B=1, the function outputs the ssim of the image of the given band. for color images band can have length 3 and for grayscale image band can be a list of one element
    '''
    if len(band)>3:
        raise ValueError('band length >3')
    if len(band)==2:
        raise ValueError('band length =2')
    if len(band)==3:
        img = Metrics.s2_to_img(pred,percentile=(0.5, 99.5), gamma=0.9, bands = band,min_max=(0, 1))
        gt = Metrics.s2_to_img(output,percentile=(0.5, 99.5), gamma=0.9, bands = band,min_max=(0, 1))
        return calculate_ssim(img,gt)
    if len(band)==1:
        img = Metrics.s2_to_img(pred,percentile=(0.5, 99.5), gamma=0.9, bands = band,min_max=(0, 1))
        gt = Metrics.s2_to_img(output,percentile=(0.5, 99.5), gamma=0.9, bands = band,min_max=(0, 1))
        return calculate_ssim(img,gt)


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
import time






folder_name='eval_baseline/'
os.makedirs(os.path.dirname(folder_name), exist_ok=True)
timestr = time.strftime("%Y%m%d-%H%M%S")
folder_name = folder_name+timestr+'/'
os.makedirs(os.path.dirname(folder_name), exist_ok=True)

os.makedirs(os.path.dirname(folder_name+args.model+'/'), exist_ok=True)

total_test_loss_l1=[]
total_test_loss_psnr_rgb=[]
total_test_loss_ssim_rgb=[]
total_test_loss_psnr_3=[]
total_test_loss_ssim_3=[]
total_test_loss_psnr_4=[]
total_test_loss_ssim_4=[]
total_test_loss_psnr_5=[]
total_test_loss_ssim_5=[]
total_test_loss_psnr_6=[]
total_test_loss_ssim_6=[]
total_test_loss_psnr_7=[]
total_test_loss_ssim_7=[]
total_test_loss_psnr_8=[]
total_test_loss_ssim_8=[]
total_test_loss_psnr_9=[]
total_test_loss_ssim_9=[]
num=0
# pbar = tqdm(enumerate(test_loader))
for batch_id, data in tqdm(enumerate(test_loader)):
    inp, out, index = data['LS'], data['S2'], data['Index']
    
    inp = inp[:,:,:64,:64]
    out = out[:,:,:192,:192]
    # print(inp.shape)

    model.eval()
    with torch.no_grad():
        loss,pred = f_pass(inp,out,model)
    pred = pred.detach().cpu()
    out = out.detach().cpu()
    torch.cuda.empty_cache()
    # print(pred.shape)
    total_test_loss_l1.append(loss)
    total_test_loss_psnr_rgb.append(psnr_band(pred,out))
    total_test_loss_ssim_rgb.append(ssim_band(pred,out))
    total_test_loss_psnr_3.append(psnr_band(pred,out,[3]))
    total_test_loss_ssim_3.append(ssim_band(pred,out,[3]))
    total_test_loss_psnr_4.append(psnr_band(pred,out,[4]))
    total_test_loss_ssim_4.append(ssim_band(pred,out,[4]))
    total_test_loss_psnr_5.append(psnr_band(pred,out,[5]))
    total_test_loss_ssim_5.append(ssim_band(pred,out,[5]))
    total_test_loss_psnr_6.append(psnr_band(pred,out,[6]))
    total_test_loss_ssim_6.append(ssim_band(pred,out,[6]))
    total_test_loss_psnr_7.append(psnr_band(pred,out,[7]))
    total_test_loss_ssim_7.append(ssim_band(pred,out,[7]))
    total_test_loss_psnr_8.append(psnr_band(pred,out,[8]))
    total_test_loss_ssim_8.append(ssim_band(pred,out,[8]))
    total_test_loss_psnr_9.append(psnr_band(pred,out,[9]))
    total_test_loss_ssim_9.append(ssim_band(pred,out,[9]))
    num=num+1

    if num==1:
        out_fname = folder_name+args.model+'/i_pred_rgb.png'
        visualize_band(pred,out_fname,[2,1,0])
        out_fname = folder_name+args.model+'/i_pred_3.png'
        visualize_band(pred,out_fname,[3])
        out_fname = folder_name+args.model+'/i_pred_4.png'
        visualize_band(pred,out_fname,[4])
        out_fname = folder_name+args.model+'/i_pred_5.png'
        visualize_band(pred,out_fname,[5])
        out_fname = folder_name+args.model+'/i_pred_6.png'
        visualize_band(pred,out_fname,[6])
        out_fname = folder_name+args.model+'/i_pred_7.png'
        visualize_band(pred,out_fname,[7])
        out_fname = folder_name+args.model+'/i_pred_8.png'
        visualize_band(pred,out_fname,[8])
        out_fname = folder_name+args.model+'/i_pred_9.png'
        visualize_band(pred,out_fname,[9])

        out_fname = folder_name+args.model+'/i_out_rgb.png'
        visualize_band(out,out_fname,[2,1,0])
        out_fname = folder_name+args.model+'/i_out_3.png'
        visualize_band(out,out_fname,[3])
        out_fname = folder_name+args.model+'/i_out_4.png'
        visualize_band(out,out_fname,[4])
        out_fname = folder_name+args.model+'/i_out_5.png'
        visualize_band(out,out_fname,[5])
        out_fname = folder_name+args.model+'/i_out_6.png'
        visualize_band(out,out_fname,[6])
        out_fname = folder_name+args.model+'/i_out_7.png'
        visualize_band(out,out_fname,[7])
        out_fname = folder_name+args.model+'/i_out_8.png'
        visualize_band(out,out_fname,[8])
        out_fname = folder_name+args.model+'/i_out_9.png'
        visualize_band(out,out_fname,[9])


        out_fname = folder_name+args.model+'/i_inp_rgb.png'
        visualize_band(inp,out_fname,[2,1,0])
        out_fname = folder_name+args.model+'/i_inp_3.png'
        visualize_band(inp,out_fname,[3])
        out_fname = folder_name+args.model+'/i_inp_4.png'
        visualize_band(out,out_fname,[4])
        out_fname = folder_name+args.model+'/i_inp_5.png'
        visualize_band(inp,out_fname,[5])

        print('data saved')
    
    
f = open(folder_name+args.model+'/stats.txt', 'w')
sys.stdout = f

print('resume: ',args.resume_path)

print('Total Number = ', num)
print('============RGB====================')
print('PSNR mean rgb= ', np.mean(total_test_loss_psnr_rgb))
print('PSNR mean 3= ', np.mean(total_test_loss_psnr_3))
print('PSNR mean 4= ', np.mean(total_test_loss_psnr_4))
print('PSNR mean 5= ', np.mean(total_test_loss_psnr_5))
print('PSNR mean 6= ', np.mean(total_test_loss_psnr_6))
print('PSNR mean 7= ', np.mean(total_test_loss_psnr_7))
print('PSNR mean 8= ', np.mean(total_test_loss_psnr_8))
print('PSNR mean 9= ', np.mean(total_test_loss_psnr_9))

print('PSNR var rgb= ', np.var(total_test_loss_psnr_rgb))
print('PSNR var 3= ', np.var(total_test_loss_psnr_3))
print('PSNR var 4= ', np.var(total_test_loss_psnr_4))
print('PSNR var 5= ', np.var(total_test_loss_psnr_5))
print('PSNR var 6= ', np.var(total_test_loss_psnr_6))
print('PSNR var 7= ', np.var(total_test_loss_psnr_7))
print('PSNR var 8= ', np.var(total_test_loss_psnr_8))
print('PSNR var 9= ', np.var(total_test_loss_psnr_9))

print('============SSIM====================')
print('ssim mean rgb= ', np.mean(total_test_loss_ssim_rgb))
print('ssim mean 3= ', np.mean(total_test_loss_ssim_3))
print('ssim mean 4= ', np.mean(total_test_loss_ssim_4))
print('ssim mean 5= ', np.mean(total_test_loss_ssim_5))
print('ssim mean 6= ', np.mean(total_test_loss_ssim_6))
print('ssim mean 7= ', np.mean(total_test_loss_ssim_7))
print('ssim mean 8= ', np.mean(total_test_loss_ssim_8))
print('ssim mean 9= ', np.mean(total_test_loss_ssim_9))

print('ssim var rgb= ', np.var(total_test_loss_ssim_rgb))
print('ssim var 3= ', np.var(total_test_loss_ssim_3))
print('ssim var 4= ', np.var(total_test_loss_ssim_4))
print('ssim var 5= ', np.var(total_test_loss_ssim_5))
print('ssim var 6= ', np.var(total_test_loss_ssim_6))
print('ssim var 7= ', np.var(total_test_loss_ssim_7))
print('ssim var 8= ', np.var(total_test_loss_ssim_8))
print('ssim var 9= ', np.var(total_test_loss_ssim_9))

f.close()
sys.stdout = sys.__stdout__
# creating the dataset
data = {'RGB':np.mean(total_test_loss_psnr_rgb), 'rededge 0.704um':np.mean(total_test_loss_psnr_3), 'rededge 0.74um':np.mean(total_test_loss_psnr_4), 
        'rededge 0.783um':np.mean(total_test_loss_psnr_5), 'NIR':np.mean(total_test_loss_psnr_6),'rededge 0.865um':np.mean(total_test_loss_psnr_7),  
        'swir16':np.mean(total_test_loss_psnr_8),'swir22':np.mean(total_test_loss_psnr_9)}
std_psnr = [np.std(total_test_loss_psnr_rgb), np.std(total_test_loss_psnr_3), np.std(total_test_loss_psnr_4), 
            np.std(total_test_loss_psnr_5), np.std(total_test_loss_psnr_6),np.std(total_test_loss_psnr_7),  
            np.std(total_test_loss_psnr_8), np.std(total_test_loss_psnr_9)]
courses = list(data.keys())
values = list(data.values())
 
fig = plt.figure(figsize = (10, 5))

# creating the bar plot
plt.bar(courses, values, 
        width = 0.4, yerr = std_psnr)

plt.xlabel("Bands")
plt.ylabel("PSNR")
plt.title("PSNR "+args.model+" Spectral Extension")
plt.savefig(folder_name+args.model+'/i_psnr.png')


data = {'RGB':np.mean(total_test_loss_ssim_rgb), 'rededge 0.704um':np.mean(total_test_loss_ssim_3), 'rededge 0.74um':np.mean(total_test_loss_ssim_4), 
        'rededge 0.783um':np.mean(total_test_loss_ssim_5), 'NIR':np.mean(total_test_loss_ssim_6),'rededge 0.865um':np.mean(total_test_loss_ssim_7),  
        'swir16':np.mean(total_test_loss_ssim_8),'swir22':np.mean(total_test_loss_ssim_9)}
std_ssim = [np.std(total_test_loss_ssim_rgb), np.std(total_test_loss_ssim_3), np.std(total_test_loss_ssim_4), 
            np.std(total_test_loss_ssim_5), np.std(total_test_loss_ssim_6),np.std(total_test_loss_ssim_7),  
            np.std(total_test_loss_ssim_8), np.std(total_test_loss_ssim_9)]
courses = list(data.keys())
values = list(data.values())
 
fig = plt.figure(figsize = (10, 5))

# creating the bar plot
plt.bar(courses, values, 
        width = 0.4, yerr = std_ssim)

plt.xlabel("Bands")
plt.ylabel("SSIM")
plt.title("SSIM "+args.model+" Spectral Extension")
plt.savefig(folder_name+args.model+'/i_ssim.png')



