# official modules
import os
import time
import json
import argparse
import numpy as np
import torch
from torch.optim import Adam
import torch.optim.lr_scheduler as lr_scheduler
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.multiprocessing as mp
import torch.distributed as dist
# self-defined module
from utils.helper import init_DDP, Logger, print_log, load_labels, build_model
from utils.data import dataloader
from utils.train_model import train_model
from utils.test_model import test_model
from utils.calc_metrics import match_result


def learn_localization(rank,world_size,opt):
    opt.rank = rank
    opt.world_size = world_size
    init_DDP(opt)

    if opt.train_or_test == 'train':
        # split data to train and validation set
        tmp_train = np.arange(0,9000,1).tolist() 
        tmp_val = np.arange(9000,10000,1).tolist()
        train_IDs = [str(i) for i in tmp_train]
        val_IDs = [str(i) for i in tmp_val]

        opt.partition = {'train': train_IDs, 'valid': val_IDs}
        opt.ntrain, opt.nval = len(train_IDs), len(val_IDs)

        # calculate zoom ratio of z-axis 
        opt.pixel_size_axial = (opt.zeta[1] - opt.zeta[0] + 1 + 2*opt.clear_dist) / opt.D

        # output folder name for results
        t = time.strftime('%m%d') + \
            '-nTrain'+str(opt.ntrain) + \
            '-lr'+str(opt.initial_learning_rate) + \
            '-Epoch'+str(opt.max_epoch) + \
            '-batchSize'+str(opt.batch_size) + \
            '-D'+str(opt.D) + \
            '-'+str(opt.model_use)

        if opt.resume:
            t = t + '-resume'

        opt.save_path = os.path.join(opt.save_path,t)
        os.makedirs(opt.save_path, exist_ok=True)

        if rank == 0:
            log = open(os.path.join(opt.save_path, 'log_{}.txt'.format(time.strftime('%H%M'))), 'w')
            logger = Logger(os.path.join(opt.save_path, 'log_{}'.format(time.strftime('%m%d'))))
            print_log('setup_params:',log)
            for key,value in opt._get_kwargs():
                if not key == 'partition':
                    print_log('{}: {}'.format(key,value),log)

        device = torch.device('cuda')
        torch.backends.cudnn.benchmark = True

        if opt.rank==0:
            # save setup parameters in results folder as well
            with open(os.path.join(opt.save_path,'setup_params.json'),'w') as handle:
                json.dump(opt.__dict__, handle, indent=2)

        # Load labels and generate dataset
        labels = load_labels(os.path.join(opt.data_path,'observed','label.txt'))

        # Parameters for dataloaders
        params_train = {'batch_size': opt.batch_size, 'shuffle': True,  'partition': opt.partition['train']}
        params_val = {'batch_size': opt.batch_size, 'shuffle': False, 'partition': opt.partition['valid']}

        training_generator = dataloader(opt.data_path, labels, params_train, opt, num_workers=0)
        validation_generator = dataloader(opt.data_path, labels, params_val, opt, num_workers=0)

        # model
        cnn = build_model(opt)
        cnn.to(device)
        cnn = DDP(cnn,find_unused_parameters=True,broadcast_buffers=False)

        optimizer = Adam(list(cnn.parameters()), lr=opt.initial_learning_rate)

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
        if opt.resume:
            checkpoint = torch.load(opt.checkpoint_path)
            cnn.load_state_dict(checkpoint['model'])
            # optimizer.load_state_dict(checkpoint['optimizer'])

        # learn a localization cnn
        if opt.rank==0:
            train_model(cnn,optimizer,scheduler,device,training_generator,validation_generator,log,logger,opt)
        else:
            train_model(cnn,optimizer,scheduler,device,training_generator,validation_generator,None,None,opt)


    elif opt.train_or_test == 'test':
        if opt.postpro:
            opt.postpro_params = {'thresh':20,'radius':1}

        time_start = time.time()
        os.makedirs(opt.save_path,exist_ok=True)

        # model testing
        cnn = build_model(opt)
        cnn.to('cuda')
        cnn = DDP(cnn,find_unused_parameters=True)
        cnn.load_state_dict(torch.load(opt.checkpoint_path)['model'])
        # model.module.load_state_dict(torch.load(opt.model_load_dir)['model'])
        cnn.eval()

        if rank == 0:
            log = open(os.path.join(opt.save_path, 'log_{}.txt'.format(time.strftime('%H%M'))), 'w')
            print_log('setup_params -- test:',log)
            for key,value in opt._get_kwargs():
                if not key == 'partition':
                    print_log('{}: {}'.format(key,value),log)

            # save setup parameters in results folder as well
            with open(os.path.join(opt.save_path,'setup_params_test.json'),'w') as handle:
                json.dump(opt.__dict__, handle, indent=2)

        dist.barrier()
        test_model(opt,cnn,log)
        time_end = time.time()
        print(f'Time cost: {time_end-time_start}')

        # compute precision and recall
        # precision,recall = match_result(test_imgs_path,path_save,opt.test_id_loc,criterion_xy=2,criterion_z=1.2)
        # print(test_imgs_path, 'thresh: {}, radius: {}, precision: {:.4f}, recall: {:.4f}'.format(postprocess_params['thresh'],postprocess_params['radius'], precision,recall))

    else: print('no such process!')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='3d localization')
    # phase
    parser.add_argument('--train_or_test',            type=str,           default='test',        help='train or test')
    parser.add_argument('--resume',                   action='store_true', default=False)
    parser.add_argument('--gpu_number',               type=str,           default='0',              help='assign gpu')
    # data info
    parser.add_argument('--H',                        type=int,           default=96,          help='Height of image')
    parser.add_argument('--W',                        type=int,           default=96,          help='Width of image')
    parser.add_argument('--zeta',                     type=tuple,         default=(-20,20),    help='min and mx zeta')
    parser.add_argument('--clear_dist',               type=int,           default=0,           help='safe margin for z axis')
    parser.add_argument('--D',                        type=int,           default=400,         help='num grid of zeta axis')
    parser.add_argument('--scaling_factor',           type=int,           default=800,         help='entry value for existence of pts')
    parser.add_argument('--upsampling_factor',        type=int,           default=4,           help='grid dim=H*upsampling_factor, W*upsampling_factor')
    # train info
    parser.add_argument('--model_use',                type=str,)
    parser.add_argument('--postpro',                 action='store_true',    default=False,           help='whether do post processing in dnn')
    parser.add_argument('--batch_size',               type=int,           default=6,           help='when training on multi GPU, is the batch size on each GPU')
    parser.add_argument('--initial_learning_rate',    type=float,         default=0.0005,      help='initial learning rate for adam')
    parser.add_argument('--lr_decay_per_epoch',       type=int,           default=10,          help='number of epochs learning rate decay')
    parser.add_argument('--lr_decay_factor',          type=float,         default=0.5,         help='lr decay factor')
    parser.add_argument('--max_epoch',                type=int,           default=30,          help='number of training epoches')
    parser.add_argument('--save_epoch',               type=int,           default=3,           help='save model per save_epoch')
    # test info
    parser.add_argument('--test_id_loc',              type=str,           default='/home/lingjia/Documents/microscope/tmp/id_test.txt')
    # path
    parser.add_argument('--checkpoint_path',          type=str,           default='/home/lingjia/Documents/microscope/Results_demo1/1015_maxEp1_nTr7000_nVal3000',       help='checkpoint to resume from')
    parser.add_argument('--data_path',       type=str,           default='/home/lingjia/Documents/microscope/Data/training_images_zrange20',     help='path for train and val data')
    parser.add_argument('--save_path',              type=str,           default='/home/lingjia/Documents/microscope/Results',            help='path for save models and results')
    opt = parser.parse_args()

    gpu_number = len(opt.gpu_number.split(','))
    mp.spawn(learn_localization,args=(gpu_number,opt),nprocs=gpu_number,join=True)




