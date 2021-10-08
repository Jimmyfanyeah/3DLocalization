from numpy.core.fromnumeric import choose
import torch
import numpy as np
import os
import time
import argparse
from collections import defaultdict
from torch.optim import Adam
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.multiprocessing as mp
from utils.helper import init_DDP, load_labels, buildModel
from utils.data import dataloader
from utils.loss import calculate_loss_lrfind as calculate_loss
# from utils.resnet import ResNet
try:
    from torch_lr_finder import LRFinder
except ImportError:
    # Run from source
    import sys
    sys.path.insert(0, '..')
    from torch_lr_finder import LRFinder

def learn_localization(rank,world_size,opt,setup_params):
    opt.rank = rank
    opt.world_size = world_size
    init_DDP(opt)

    if opt.train_or_test == 'train':
        # split data to train and validation set
        tmp_train = np.arange(0,9000,1).tolist() #+ np.arange(10000,10900,1).tolist() + np.arange(11000,11900,1).tolist()
        tmp_valid = np.arange(9000,10000,1).tolist() #+ np.arange(10900,11000,1).tolist() + np.arange(11900,12000,1).tolist()
        train_IDs = [str(i) for i in tmp_train]
        valid_IDs = [str(i) for i in tmp_valid]

        partition = {'train': train_IDs, 'valid': valid_IDs}
        setup_params['partition'] = partition
        setup_params['ntrain'], setup_params['nvalid'] = len(train_IDs), len(valid_IDs)

        # calculate zoom ratio of z-axis 
        setup_params['pixel_size_axial'] = (setup_params['zmax'] - setup_params['zmin']+ 2*setup_params['clear_dist'])/setup_params['D']

        device = torch.device('cuda')
        torch.backends.cudnn.benchmark = True

        # Load labels and generate dataset
        path_train = setup_params['training_data_path']
        labels = load_labels(os.path.join(path_train,'train','label.txt'))

        # Parameters for dataloaders
        params_train = {'batch_size': setup_params['batch_size'], 'shuffle': True,  'partition':setup_params['partition']['train']}
        params_valid = {'batch_size': setup_params['batch_size'], 'shuffle': False, 'partition':setup_params['partition']['valid']}

        training_generator = dataloader(path_train, labels, params_train, setup_params, opt, num_workers=0)
        validation_generator = dataloader(path_train, labels, params_valid, setup_params, opt, num_workers=0)

        # model
        cnn = buildModel(setup_params)
        cnn.to(device)
        cnn = DDP(cnn,find_unused_parameters=True,broadcast_buffers=False)

        from torch_lr_finder import LRFinder
        model = cnn
        criterion = calculate_loss(setup_params['scaling_factor'])
        optimizer = Adam(cnn.parameters(), lr=1e-7, weight_decay=0)
        # lr_finder = LRFinder(cnn, optimizer, criterion, device="cuda")
        # lr_finder.range_test(training_generator, end_lr=100, num_iter=300)
        # lr_finder.plot() # to inspect the loss-learning rate graph
        # lr_finder.reset() # to reset the model and optimizer to their initial state

        lr_finder = LRFinder(model, optimizer, criterion, device="cuda")
        lr_finder.range_test(training_generator, start_lr=1e-7, end_lr=10, num_iter=100, step_mode="exp")
        lr_finder.plot(log_lr=True)
        lr_finder.reset()


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='3d localization')
    parser.add_argument('--H',                        type=int,           default=96,            help='Height of image, default in setup_params=96')
    parser.add_argument('--W',                        type=int,           default=96,            help='Width of image, default in setup_params=96')
    parser.add_argument('--clear_dist',               type=int,           default=0,             help='safe margin for z axis, default in setup_params=1')
    parser.add_argument('--scaling_factor',           type=int,           default=170,           help='scaling facot for the loss function')
    parser.add_argument('--upsampling_factor',        type=int,           default=1,             help='grid dim=H*upsampling_factor, W*upsampling_factor')
    parser.add_argument('--train_or_test',            type=str,           default='train',        help='train or test')
    parser.add_argument('--zmax',                     type=int,           default=20,            help='min zeta')
    parser.add_argument('--D',                        type=int,           default=21,           help='num grid of zeta axis')
    parser.add_argument('--gpu_number',               type=str,           default='0',          help='assign gpu')
    # training
    parser.add_argument('--batch_size',               type=int,           default=8, )
    parser.add_argument('--training_data_path',       type=str,           default='/home/lingjia/Documents/3dloc_data/train/0620_uniformFlux', )
    parser.add_argument('--model_use', type=str, default='cnn_residual')
    opt = parser.parse_args()
    # print(opt)

    opt.zmin = -opt.zmax

    # add parameters into setup_params
    setup_params = defaultdict(str)
    for arg in vars(opt):
        setup_params[arg] = getattr(opt, arg)

    gpu_number = len(opt.gpu_number.split(','))
    mp.spawn(learn_localization,args=(gpu_number,opt,setup_params),nprocs=gpu_number,join=True)