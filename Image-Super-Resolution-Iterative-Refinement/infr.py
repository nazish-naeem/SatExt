import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0, 1, 2, 3" 
os.environ["WORLD_SIZE"] = "1"
import torch
import data as Data
import model as Model
import argparse
import logging
import core.logger as Logger
import core.metrics as Metrics
from core.wandb_logger import WandbLogger
from tensorboardX import SummaryWriter
import os
import numpy as np


import time
timestr = time.strftime("%Y%m%d-%H%M%S")
result_path = '/home/t-nnaeem/workspace/RemoteSensingFoundationModels/Image-Super-Resolution-Iterative-Refinement/inference_tests/infer_'+timestr+'/'
os.makedirs(os.path.dirname(result_path))


def visualize(visuals,t,result_path,current_step,idx):
    t_step = t
    
    if t==len(visuals['SR'][:,0,0,0])-1:
        print('mean val sr ', t)
        ggh = Metrics.tensor2numpy(visuals['SR'][t_step,:,:,:],(-1,1))
        print(np.mean(ggh))

        print('mean val hr ', t)
        ggh = Metrics.tensor2numpy(visuals['HR'][0,:,:,:],(-1,1))
        print(np.mean(ggh))

    sr_img = Metrics.s2_to_img(visuals['SR'][t_step,:,:,:])
    hr_img = Metrics.s2_to_img(visuals['HR'][0,:,:,:])
    lr_img = Metrics.s2_to_img(visuals['LR'][0,:,:,:])
    fake_img = Metrics.s2_to_img(visuals['INF'][0,:,:,:])
    # generation
    os.makedirs('{}/{}'.format(result_path, current_step), exist_ok=True)

    Metrics.save_img(
        hr_img, '{}/{}/{}_{}_hr.png'.format(result_path, current_step, idx,t_step))
    Metrics.save_img(
        sr_img, '{}/{}/{}_{}_sr.png'.format(result_path, current_step, idx,t_step))
    Metrics.save_img(
        lr_img, '{}/{}/{}_{}_lr.png'.format(result_path, current_step, idx,t_step))
    Metrics.save_img(
        fake_img, '{}/{}/{}_{}_inf.png'.format(result_path, current_step, idx,t_step))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', type=str, default='config/sr_S2_RGB_infr.json',
                        help='JSON file for configuration')
    parser.add_argument('-p', '--phase', type=str, choices=['train', 'val'],
                        help='Run either train(training) or val(generation)', default='train')
    parser.add_argument('-gpu', '--gpu_ids', type=str, default=None)
    parser.add_argument('-debug', '-d', action='store_true')
    parser.add_argument('-enable_wandb', action='store_true')
    parser.add_argument('-log_wandb_ckpt', action='store_true')
    parser.add_argument('-log_eval', action='store_true')

    # parse configs
    args = parser.parse_args()
    opt = Logger.parse(args)
    import os
    import shutil
    shutil.copyfile(args.config, result_path+os.path.basename(args.config))
    # import os
    # os.system('cp /path/to/source/file /path/to/destination/file')
    # Convert to NoneDict, which return None for missing key.
    opt = Logger.dict_to_nonedict(opt)
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True
    
    

    # dataset
    for phase, dataset_opt in opt['datasets'].items():
        if phase == 'train' and args.phase != 'val':
            train_set = Data.create_dataset(dataset_opt, phase)
            train_loader = Data.create_dataloader(
                train_set, dataset_opt, phase)
        elif phase == 'val':
            val_set = Data.create_dataset(dataset_opt, phase)
            val_loader = Data.create_dataloader(
                val_set, dataset_opt, phase)
            print(dataset_opt)
    
    # logger.info('Initial Dataset Finished')
    # print(xx)
    # model
    diffusion = Model.create_model(opt)
    # logger.info('Initial Model Finished')
    print("val data len: ",val_loader.__len__())

    # Train
    current_step = diffusion.begin_step
    current_epoch = diffusion.begin_epoch
    n_iter = opt['train']['n_iter']

    # if opt['path']['resume_state']:
        # logger.info('Resuming training from epoch: {}, iter: {}.'.format(
            # current_epoch, current_step))

    diffusion.set_new_noise_schedule(
        opt['model']['beta_schedule'][opt['phase']], schedule_phase=opt['phase'])
    idx=0
    avg_psnr=0
    avg_ssim=0
    for _,  val_data in enumerate(val_loader):
        # print(len(val_loader))
        idx += 1
        diffusion.feed_data(val_data)
        diffusion.test(continous=True)
        visuals = diffusion.get_current_visuals()
        # print('visuals shape')
        # print(visuals['HR'].shape)
        ## images for tme step 1
        for t_step in range(len(visuals['SR'][:,0,0,0])):
            # t_step = 2
            visualize(visuals,t_step,result_path,current_step,idx)
            # t_step = 4
            # visualize(visuals,t_step,result_path,current_step,idx)
            # t_step = 9
            # visualize(visuals,t_step,result_path,current_step,idx)
            # t_step = 11
            # visualize(visuals,t_step,result_path,current_step,idx)

        # sr_img = Metrics.s2_to_img(visuals['SR'][-1,:,:,:])
        # hr_img = Metrics.s2_to_img(visuals['HR'][-1,:,:,:])

        # avg_psnr += Metrics.calculate_psnr(
        #     sr_img, hr_img)
        # avg_ssim += Metrics.calculate_ssim(
        #     sr_img, hr_img)

