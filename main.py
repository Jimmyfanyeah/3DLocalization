import numpy as np
import os
import time
import pickle
import argparse
import fnmatch
import re
from collections import defaultdict
import torch
from torch.optim import Adam
import torch.optim.lr_scheduler as lr_scheduler
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.multiprocessing as mp
import torch.distributed as dist
# self-defined functions
from utils.helper import print_log, Logger, init_DDP, load_labels, buildModel
from utils.data import dataloader
from utils.test_model import test_model
from utils.train_model import train_model


def learn_localization(rank,world_size,opt,setup_params):
    opt.rank = rank
    opt.world_size = world_size
    init_DDP(opt)

    if opt.train_or_test == 'train':
        training_volume = setup_params['training_volume'] + 1
        valid_start = training_volume
        valid_end = valid_start + setup_params['validation_volume']
        # split data to train and validation set
        tmp_train = np.arange(1,training_volume,1).tolist() 
        tmp_valid = np.arange(valid_start,valid_end,1).tolist()

        # add hard samples in training set
        if setup_params['train_with_hard_sample'] == 1:
            path_train = os.path.join(setup_params['training_data_path'],'train')
            path_hs = '../data_train/train_HS_b10_1'
            hardsamples = fnmatch.filter(os.listdir(path_hs),'im20000*mat')
            hs_IDs=[]
            for hs in hardsamples:
                hs_IDs.append(int(re.findall('[0-9]+',hs)[0]))
            tmp_train = tmp_train + hs_IDs

        train_IDs = [str(i) for i in tmp_train]
        valid_IDs = [str(i) for i in tmp_valid]

        partition = {'train': train_IDs, 'valid': valid_IDs}
        setup_params['partition'] = partition
        setup_params['ntrain'], setup_params['nvalid'] = len(train_IDs), len(valid_IDs)

        # calculate zoom ratio of z-axis 
        setup_params['pixel_size_axial'] = (setup_params['zmax'] - setup_params['zmin'] +1 + 2*setup_params['clear_dist'])/setup_params['D']

        # output folder for results
        t = time.strftime('%m%d%H') + \
            '-nTrain'+str(setup_params['ntrain']) + \
            '-lr'+str(setup_params['initial_learning_rate']) + \
            '-Epoch'+str(setup_params['maxEpoch']) + \
            '-batchSize'+str(setup_params['batch_size']) + \
            '-D'+str(setup_params['D']) + \
            '-'+str(setup_params['model_use'])

        if setup_params['resume_training']:
            t = t + '-resume'

        save_folder = os.path.join(setup_params['result_path'],t)
        os.makedirs(save_folder, exist_ok=True)
        setup_params['result_save_path'] = save_folder

        if rank == 0:
            log = open(os.path.join(setup_params['result_save_path'], 'log_{}.txt'.format(time.strftime('%H%M'))), 'w')
            logger = Logger(os.path.join(setup_params['result_save_path'], 'log_{}'.format(time.strftime('%m%d'))))
            print_log('setup_params:',log)
            for key in setup_params.keys():
                if not key == 'partition':
                    print_log('{}: {}'.format(key,setup_params[key]),log)

        device = torch.device('cuda')
        torch.backends.cudnn.benchmark = True

        if opt.rank==0:
            # save setup parameters in results folder as well
            path_setup_params = os.path.join(setup_params['result_save_path'],'setup_params.pickle')
            with open(path_setup_params, 'wb') as handle:
                pickle.dump(setup_params, handle, protocol=pickle.HIGHEST_PROTOCOL)

        # Load labels and generate dataset
        path_train = setup_params['training_data_path']
        labels = load_labels(os.path.join(path_train,'train','label.txt'))

        # Parameters for dataloaders
        params_train = {'batch_size': setup_params['batch_size'], 'shuffle': True,  'partition':setup_params['partition']['train']}
        params_valid = {'batch_size': setup_params['batch_size'], 'shuffle': False, 'partition':setup_params['partition']['valid']}

        training_generator = dataloader(path_train, labels, params_train, setup_params, opt, num_workers=0)
        validation_generator = dataloader(path_train, labels, params_valid, setup_params, opt, num_workers=0)

        # Model
        cnn = buildModel(setup_params)
        cnn.to(device)
        cnn = DDP(cnn,find_unused_parameters=True,broadcast_buffers=False)

        initial_learning_rate = setup_params['initial_learning_rate']
        optimizer = Adam(list(cnn.parameters()), lr=initial_learning_rate)

        opt.scheduler_type = 'ReduceLROnPlateau'
        if opt.scheduler_type == 'StepLR':
            scheduler = lr_scheduler.StepLR(optimizer, step_size=opt.lr_decay_per_epoch, gamma=opt.lr_decay_factor)
        else:
            scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=opt.lr_decay_factor, patience=opt.lr_decay_per_epoch , min_lr=0)

        # Print Model layers and number of parameters
        if opt.rank == 0:
            print_log(cnn, log)
            print_log("number of parameters: {}\n".format(sum(param.numel() for param in cnn.parameters())),log)

        # if resume_training, continue from a checkpoint
        if setup_params['resume_training']:
            checkpoint = torch.load(setup_params['checkpoint_path'])
            cnn.load_state_dict(checkpoint['model'])
            # optimizer.load_state_dict(checkpoint['optimizer'])

        # learn a localization cnn
        if opt.rank==0:
            train_model(cnn,optimizer,scheduler,device,training_generator,validation_generator,log,logger,opt,setup_params)
        else:
            train_model(cnn,optimizer,scheduler,device,training_generator,validation_generator,None,None,opt,setup_params)

    elif opt.train_or_test == 'test':
        time_start = time.time()

        path_model = opt.checkpoint_path
        test_imgs_path = opt.training_data_path
        path_save = opt.result_path
        os.makedirs(path_save,exist_ok=True)

        # model testing
        cnn = buildModel(setup_params)
        cnn.to('cuda')
        cnn = DDP(cnn,find_unused_parameters=True)
        cnn.load_state_dict(torch.load(path_model)['model'])
        # model.module.load_state_dict(torch.load(opt.model_load_dir)['model'])
        cnn.eval()

        if rank == 0:
            log = open(os.path.join(path_save,'setup_params.txt'), 'w')
            print_log('test_imgs_path: {}'.format(test_imgs_path),log)
            print_log('setup_params - test:',log)
            for key in setup_params.keys():
                if not key == 'partition':
                    print_log('{}: {}'.format(key,setup_params[key]),log)

        postprocess_params = {'thresh': 20,'radius': 1}

        dist.barrier()
        test_model(opt,cnn,postprocess_params,path_model,path_save,test_imgs_path,log)
        time_end = time.time()
        print(f'time cose {time_end-time_start}')

    else: print('no such process!')


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='3d localization')
    parser.add_argument('--H',                        type=int,           default=96,          help='Height of image')
    parser.add_argument('--W',                        type=int,           default=96,          help='Width of image')
    parser.add_argument('--clear_dist',               type=int,           default=0,           help='safe margin for z axis')
    parser.add_argument('--scaling_factor',           type=int,           default=800,         help='scaling facot for the loss function')
    parser.add_argument('--upsampling_factor',        type=int,           default=4,           help='grid dim=H*upsampling_factor, W*upsampling_factor')
    parser.add_argument('--train_or_test',            type=str,           default='test',       help='train or test')
    parser.add_argument('--zmax',                     type=int,           default=20,           help='min zeta')
    parser.add_argument('--D',                        type=int,           default=400,          help='num grid of zeta axis')
    parser.add_argument('--gpu_number',               type=str,           default='0',          help='assign gpu')
    # training
    parser.add_argument('--batch_size',               type=int,           default=6,            help='when training on multi GPU, is the batch size on each GPU')
    parser.add_argument('--maxEpoch',                 type=int,           default=30,           help='number of training epoches')
    parser.add_argument('--saveEpoch',                type=int,           default=3,            help='save model per saveEpoch')
    parser.add_argument('--initial_learning_rate',    type=float,         default=0.0005,       help='initial learning rate for adam')
    parser.add_argument('--lr_decay_per_epoch',       type=int,           default=10,           help='number of epochs learning rate decay')
    parser.add_argument('--lr_decay_factor',          type=float,         default=0.5,          help='lr decay factor')
    # resume
    parser.add_argument('--resume_training',          type=int,           default=0,            help='whether to resume training from checkpoint')
    parser.add_argument('--checkpoint_path',          type=str,           default='',           help='checkpoint to resume from')
    # path
    parser.add_argument('--test_id_loc',              type=str,           default='')
    parser.add_argument('--training_data_path',       type=str,           default='',           help='path for training and validation data')
    parser.add_argument('--result_path',              type=str,           default='',           help='path for save models')
    parser.add_argument('--post_pro',                 type=int,           default=0,            help='whether do post processing in dnn')
    parser.add_argument('--model_use',                type=str,           choices=['cnn','cnn_residual'])
    parser.add_argument('--training_volume',          type=int,           default=9000,         help='training set volume, up to 9k')
    parser.add_argument('--validation_volume',        type=int,           default=1000,         help='validation set volume')
    parser.add_argument('--train_with_hard_sample',   type=int,           default=0,            help='whether to train with hard samples')
    opt = parser.parse_args()

    opt.zmin = -opt.zmax

    # add parameters into setup_params
    setup_params = defaultdict(str)
    for arg in vars(opt):
        setup_params[arg] = getattr(opt, arg)

    gpu_number = len(opt.gpu_number.split(','))
    mp.spawn(learn_localization,args=(gpu_number,opt,setup_params),nprocs=gpu_number,join=True)




