# official modules
import argparse
import json
import os
import time
import numpy as np
from math import floor
from shutil import copy2
import torch
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau, StepLR
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.multiprocessing as mp
import torch.distributed as dist
# self-defined module
from utils.helper import init_DDP, Logger, print_log, load_labels, build_model
from utils.data import dataloader
from utils.train_model import train_model
from utils.test_model import test_model


def learn_localization(rank,world_size,opt):
    opt.rank = rank
    opt.world_size = world_size
    init_DDP(opt)

    if opt.train_or_test == 'train':

        # calculate zoom ratio of z-axis
        opt.pixel_size_axial = (opt.zmax - opt.zmin + 1 + 2*opt.clear_dist) / opt.D

       # split dataset to train, validation 9:1
        train_IDs = np.arange(1,floor(opt.num_im*0.9)+1,1).tolist()
        val_IDs = np.arange(floor(opt.num_im*0.9)+1,opt.num_im+1).tolist()

        opt.partition = {'train': train_IDs, 'valid': val_IDs}
        opt.ntrain, opt.nval = len(train_IDs), len(val_IDs)

        # output folder name
        name_time = opt.name_time if opt.name_time else time.strftime('%Y-%m-%d-%H-%M-%S')
        save_name = name_time + '-lr'+str(opt.initial_learning_rate) + \
            '-bs'+str(opt.batch_size) + \
            '-D'+str(opt.D) + \
            '-Ep'+str(opt.max_epoch) + \
            '-nT'+str(opt.ntrain)
        if opt.extra_loss:
            save_name = save_name + '-w' + str(opt.weight) + '-' + str(opt.extra_loss)
        if opt.cel0_mu:
            save_name = save_name + '-' + str(opt.cel0_mu)
        if opt.klnc_a:
            save_name = save_name + '-' + str(opt.klnc_a)
        save_name = save_name + '-' + str(opt.model_use)

        if opt.resume:
            save_name = save_name + '-resume'
        if opt.postpro:
            save_name = save_name + '-postpro'
        opt.save_path = os.path.join(opt.save_path,save_name)
        os.makedirs(opt.save_path, exist_ok=True)

        # log files
        if rank == 0:
            log = open(os.path.join(opt.save_path, '{}_log.txt'.format(time.strftime('%H-%M-%S'))), 'w')
            logger = Logger(os.path.join(opt.save_path, '{}_tensorboard'.format(time.strftime('%H-%M-%S'))))

            print_log('[INFO==>] setup_params:',log)
            for key,value in opt._get_kwargs():
                if not key == 'partition':
                    print_log('{}: {}'.format(key,value),log)
            print_log(f'[INFO==>] Dataset: Train {len(train_IDs)} Val {len(val_IDs)}',log)

        device = torch.device('cuda')
        torch.backends.cudnn.benchmark = True

        if opt.rank==0:
            # save setup parameters in result folder as well
            with open(os.path.join(opt.save_path,'setup_params.json'),'w') as handle:
                json.dump(opt.__dict__, handle, indent=2)

        # Load labels and generate dataset
        labels = load_labels(os.path.join(opt.data_path,'label.txt'))

        # Parameters for dataloaders
        params_train = {'batch_size': opt.batch_size, 'shuffle': True,  'partition': opt.partition['train']}
        params_val = {'batch_size': opt.batch_size, 'shuffle': False, 'partition': opt.partition['valid']}

        training_generator = dataloader(opt.data_path, labels, params_train, opt, num_workers=0)
        validation_generator = dataloader(opt.data_path, labels, params_val, opt, num_workers=0)

        # model
        model = build_model(opt)
        model.to(device)
        model = DDP(model,find_unused_parameters=True,broadcast_buffers=False)

        optimizer = Adam(list(model.parameters()), lr=opt.initial_learning_rate)

        # opt.scheduler_type = 'ReduceLROnPlateau'
        # if opt.scheduler_type == 'StepLR':
        # scheduler = lr_scheduler.StepLR(optimizer, step_size=opt.lr_decay_per_epoch, gamma=opt.lr_decay_factor)
        scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=opt.lr_decay_factor, patience=opt.lr_decay_per_epoch)

        # Print Model layers and number of parameters
        if opt.rank == 0:
            # print_log(model, log)
            print_log("[INFO==>] Number of parameters: {}".format(sum(param.numel() for param in model.parameters())),log)


        # if resume_training, continue from a checkpoint
        if opt.resume:
            print_log("[INFO==>] Resume...",log)
            checkpoint = torch.load(opt.checkpoint_path)
            model.load_state_dict(checkpoint['model'])
            # optimizer.load_state_dict(checkpoint['optimizer'])

        # learn a localization model
        if opt.rank==0:
            train_model(model,optimizer,scheduler,device,training_generator,validation_generator,log,logger,opt)
        else:
            train_model(model,optimizer,scheduler,device,training_generator,validation_generator,None,None,opt)


    elif opt.train_or_test == 'test':

        if opt.postpro:
            opt.postpro_params = {'thresh':20,'radius':1}

        opt.pixel_size_axial = (opt.zmax - opt.zmin + 1 + 2*opt.clear_dist) / opt.D
        opt.device = 'cuda'

        time_start = time.time()

        if opt.extra_loss:
            if not opt.weight: # got weight from checkpoint name
                weight = opt.checkpoint_path.split('w')[-1].split('-')[0]
                opt.weight = weight

        name_time = opt.checkpoint_path.split('/')[-2][:19]
        nSource = opt.data_path.split('/')[-1][4:]
        
        opt.save_path = os.path.join(opt.save_path, name_time+'-w'+opt.weight+'-'+opt.extra_loss+'-'+opt.log_comment,'test'+nSource)
        os.makedirs(opt.save_path,exist_ok=True)

        # copy setup params train into save path
        path_train_result = os.path.dirname(opt.checkpoint_path)
        try:
            copy2(os.path.join(path_train_result,'setup_params.json'),os.path.join(opt.save_path,'setup_params_train.json'))
        except:
            pass

        # model testing
        model = build_model(opt)
        model.to('cuda')
        model = DDP(model,find_unused_parameters=True)
        model.load_state_dict(torch.load(opt.checkpoint_path)['model'])
        # model.module.load_state_dict(torch.load(opt.model_load_dir)['model'])
        model.eval()

        if rank == 0:
            log = open(os.path.join(opt.save_path, '{}_log.txt'.format(time.strftime('%Y-%m-%d-%H-%M-%S'))), 'w')
            print_log('[INFO==>] setup params -- test:',log)
            for key,value in opt._get_kwargs():
                if not key == 'partition':
                    print_log('{}: {}'.format(key,value),log)

            # save setup parameters in results folder as well
            with open(os.path.join(opt.save_path,'setup_params_test.json'),'w') as handle:
                json.dump(opt.__dict__, handle, indent=2)

        dist.barrier()

        if opt.rank==0:
            test_model(opt,model,log)
        else:
            test_model(opt,model,None)

        time_end = time.time()
        print(f'Time cost: {time_end-time_start}\n\n\n')

        # compute precision and recall
        # precision,recall = match_result(test_imgs_path,path_save,opt.test_id_loc,criterion_xy=2,criterion_z=1.2)
        # print(test_imgs_path, 'thresh: {}, radius: {}, precision: {:.4f}, recall: {:.4f}'.format(postprocess_params['thresh'],postprocess_params['radius'], precision,recall))

    else: print('no such process!')



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='3d localization')
    # phase
    parser.add_argument('--train_or_test', type=str, default='other', help='train or test')
    parser.add_argument('--resume', action='store_true', default=False)
    parser.add_argument('--gpu_number', type=str, default=None, help='assign gpu')
    # data info
    parser.add_argument('--num_im', type=int, default=None, help='Number of samples used, train:val=9:1')
    parser.add_argument('--H', type=int, default=96, help='Height of image')
    parser.add_argument('--W', type=int, default=96, help='Width of image')
    parser.add_argument('--zmin', type=int, default=-20, help='min zeta')
    parser.add_argument('--zmax', type=int, default=20, help='max zeta')
    parser.add_argument('--clear_dist', type=int, default=1, help='safe margin for z axis')
    parser.add_argument('--D', type=int, default=250, help='num grid of zeta axis')
    parser.add_argument('--scaling_factor', type=int, default=800, help='entry value for existence of pts')
    parser.add_argument('--upsampling_factor', type=int, default=2, help='grid dim=[H,W]*upsampling_factor')
    # train info
    parser.add_argument('--model_use', type=str, default='LocNet')
    parser.add_argument('--postpro',  action='store_true', default=False, help='whether do post processing in dnn')
    parser.add_argument('--batch_size', type=int, default=1, help='when training on multi GPU, is the batch size on each GPU')
    parser.add_argument('--initial_learning_rate', type=float, default=None, help='initial learning rate for adam')
    parser.add_argument('--lr_decay_per_epoch', type=int, default=None, help='number of epochs learning rate decay')
    parser.add_argument('--lr_decay_factor', type=float, default=None, help='lr decay factor')
    parser.add_argument('--max_epoch', type=int,   default=None, help='number of training epoches')
    parser.add_argument('--save_epoch', type=int, default=None, help='save model per save_epoch')
    # test info
    parser.add_argument('--test_id_loc', type=str, default=None)
    # path
    parser.add_argument('--checkpoint_path', type=str,  default=None,  help='checkpoint to resume from')
    parser.add_argument('--data_path', type=str, default='/home/lingjia/Documents/microscope/Data/training_images_zrange20', help='path for train and val data')
    parser.add_argument('--save_path', type=str, default=None, help='path for save models and results')
    # output
    parser.add_argument('--name_time', type=str, default=None, help='string of running time')
    # for nonconvex loss
    parser.add_argument('--port', type=str, default=None, help='DDP master port')
    parser.add_argument('--weight', type=str, default=None, help='lambda CEL0')
    parser.add_argument('--extra_loss', type=str, default=None, help='indicate whether use cel0 for gaussian or nc for possion')
    # for extra losses
    parser.add_argument('--cel0_mu', type=float, default=None, help='mu in cel0 loss')
    parser.add_argument('--klnc_a', type=float, default=None, help='a for nonconvex loss in KLNC')
    parser.add_argument('--log_comment', type=str, default=None)
    
    opt = parser.parse_args()
    # args,_=parser.parse_known_args()

    if opt.gpu_number is None:
        opt.gpu_number = '0'
    gpu_number = len(opt.gpu_number.split(','))
    mp.spawn(learn_localization,args=(gpu_number,opt),nprocs=gpu_number,join=True)




