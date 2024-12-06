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



parser = argparse.ArgumentParser()
parser.add_argument('-p', '--phase', type=str, choices=['train', 'val'],
                    help='Run either train(training) or val(generation)', default='train')
parser.add_argument('-gpu', '--gpu_ids', type=str, default=None)
parser.add_argument('-b', '--batch_size', type=float, default=5)
parser.add_argument('-n', '--num_workers', type=float, default=15)
parser.add_argument('-sh', '--shuffle', type=bool, default=False)
parser.add_argument('-e', '--epoch', type=int, default=1000)
parser.add_argument('--no-cuda', action='store_true', default=False)
parser.add_argument('--seed', type=int, default=1)
parser.add_argument('--resume_path', type=str, default=None, help='pretrained model path')
parser.add_argument('--lr', type=float, default=0.00001, help='learning rate')
parser.add_argument('--data_path', type=str, default='/home/t-nnaeem/workspace/RemoteSensingFoundationModels/Spectral-Extension/data', help='training data path')
parser.add_argument('--cuda', type=bool, default=True, help='use gpu')
parser.add_argument('--save_freq', type=float, default=10, help='frequency of saving the model and results')
parser.add_argument('--model', type=str, default='RCAN', help='type of the model that is being used')
args = parser.parse_args()





import sys

#output folder preparation
import os
import time





timestr = time.strftime("%Y%m%d-%H%M%S")
out_path = "experiments_baseline/spectral_extension_"+timestr+'/'
os.makedirs(os.path.dirname(out_path))
logger_path = out_path+'/tb_logger/'
os.makedirs(os.path.dirname(logger_path))

checkpoint_path = out_path+'/checkpoint/'
os.makedirs(os.path.dirname(checkpoint_path))
result_path = out_path+'/result/'
os.makedirs(os.path.dirname(result_path))
outlogger_path = out_path+'/log/'
os.makedirs(os.path.dirname(outlogger_path))


import logging
logging.basicConfig(filename=outlogger_path+'output.log', level=logging.INFO)

def myprint(s):
    with open(outlogger_path+'model_summary.txt', 'a') as f:
        print(s, file=f)

logging.info(args)

# f = open(outlogger_path+"/terminal.out", 'w+')
# sys.stdout = f

writer = SummaryWriter(logger_path)

args.cuda = not args.no_cuda and torch.cuda.is_available()
torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)
    print('Cuda is true')

# print('datapath: ', args.data_path)

transform = transforms.Compose(
    [transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))])
# if args.model=='AE':
#     model = AE((6, 128, 128),10)
# if args.model=='FCONV':
#     model = FConv((6, 128, 128))
if args.model=='RCAN':
    # model = RCAN(num_features=64, num_rg=10, num_rcab=20, scale=1, reduction=16, in_channels=6, out_channels=10)
    model = RCAN(num_features=64, num_rg=5, num_rcab=10, scale=3, reduction=16, in_channels=6, out_channels=10)

if args.model=='RRDB':
    # model = RCAN(num_features=64, num_rg=10, num_rcab=20, scale=1, reduction=16, in_channels=6, out_channels=10)
    model = rrdb.RRDBModel(in_nc=6, out_nc=10, nf=32, nb=8, gc=16, scale_factor=3)
    # model = rrdb_sd.RRDBModel(in_nc=6, out_nc=10, nf=32, nb=8, gc=16, scale_factor=1, rdb_prob=0.0, rrdb_prob=0.5, num_rdb=6).to(device)



dataset_train = LSS2Dataset(args.data_path, split='train', data_len=-1)
dataset_test = LSS2Dataset(args.data_path, split='test', data_len=-1)
train_loader = torch.utils.data.DataLoader(
            dataset_train,
            batch_size=args.batch_size,
            shuffle=args.shuffle,
            num_workers=args.num_workers,
            pin_memory=True)

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
f = open(outlogger_path+'model_summary.txt', 'w')
sys.stdout = f
# logging.info(summary(model,(6, 128, 128)))
summary(model,(6, 128, 128))
f.close()
sys.stdout = sys.__stdout__
if args.resume_path is not None:
    state_dict = torch.load(args.resume_path)
    model.load_state_dict(state_dict['state_dict'])

logging.info('Number of parameters in the Spectral Extension model: {}'.format(sum([p.data.nelement() for p in model.parameters()])) )

optimizer = optim.Adam(model.parameters(), lr = args.lr, betas = (0.9, 0.999))
def loss_fun(pred,out):
    return F.smooth_l1_loss(pred,out)

def visualize(data,f_name):
    hr_img = Metrics.s2_to_img(data)
    Metrics.save_img(
        hr_img, f_name)




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

##training loop

for i in range(args.epoch):
    total_train_loss=0
    min_epoch = -1
    min_loss = 1e6
    # sys.stdout.flush()
    pbar = tqdm(enumerate(train_loader))
    for batch_id, data in pbar:
        inp, out, index = data['LS'], data['S2'], data['Index']
        start_time = time.time()
        optimizer.zero_grad()
        model.train()
        loss,pred = f_pass(inp,out,model)
        loss.backward()
        optimizer.step()
        pbar.set_description('Iter %d, training loss = %.3f, time = %.2f' %(batch_id,loss.item(),time.time()-start_time))
        total_train_loss +=loss.item()
    
    logging.info('epoch %d, total training loss = %.3f' %(i,total_train_loss/len(train_loader)))
    writer.add_scalar('training loss', total_train_loss/len(train_loader), i)

    total_test_loss=0
    pbar = tqdm(enumerate(test_loader))
    for batch_id, data in pbar:
        inp, out, index = data['LS'], data['S2'], data['Index']
        start_time = time.time()
        model.eval()
        with torch.no_grad():
            loss,pred = f_pass(inp,out,model)
        pbar.set_description('Iter %d, testing loss = %.3f, time = %.2f' %(batch_id,loss.item(),time.time()-start_time))
        total_test_loss +=loss.item()
        if i%args.save_freq==0:
            if batch_id==0:
            ##save sample image
                folder_name = result_path + 'epoch_'+str(i)+'/'
                os.makedirs(os.path.dirname(folder_name))
                visualize(pred[0,:,:,:],folder_name+'pred.png')
                visualize(out[0,:,:,:],folder_name+'gt.png')
                visualize(inp[0,:,:,:],folder_name+'inp.png')
    norm_test_loss = total_test_loss/len(test_loader)
    if norm_test_loss<min_loss:
        min_loss=norm_test_loss
        min_epoch=i
    logging.info('epoch %d, total testing loss = %.3f' %(i,total_test_loss/len(test_loader)))
    writer.add_scalar('testing loss', total_test_loss/len(test_loader), i)

    


    ##save checkpoint
    if i%args.save_freq==0:
        save_cp_fname = checkpoint_path + 'checkpoint_E' + str(i) + '.tar'
        torch.save({
            'epoch': i,
            'state_dict': model.state_dict(),
            'train_loss': total_train_loss/len(train_loader),
            'test_loss': total_test_loss/len(test_loader)
        },save_cp_fname)

writer.flush()
logging.info('Min Test Loss Epoch: ', min_epoch)
logging.info('Min Test Loss: ', min_loss)


# f.close()



