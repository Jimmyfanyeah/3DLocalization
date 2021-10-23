# official modules
import numpy as np
import os
import argparse
import torch
from torch.optim import Adam
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.multiprocessing as mp
# self-defined modules
from utils.helper import init_DDP, load_labels, build_model
from utils.data import dataloader
from utils.loss import calculate_loss
try:
    from torch_lr_finder import LRFinder
except ImportError:
    # Run from source
    import sys
    sys.path.insert(0, '..')
    from torch_lr_finder import LRFinder

def learn_localization(rank,world_size,opt):
    opt.rank = rank
    opt.world_size = world_size
    init_DDP(opt)

    if opt.train_or_test == 'train':
        # split data to train and validation set
        tmp_train = np.arange(0,9000,1).tolist() #+ np.arange(10000,10900,1).tolist() + np.arange(11000,11900,1).tolist()
        tmp_val = np.arange(9000,10000,1).tolist() #+ np.arange(10900,11000,1).tolist() + np.arange(11900,12000,1).tolist()
        train_IDs = [str(i) for i in tmp_train]
        val_IDs = [str(i) for i in tmp_val]

        opt.partition = {'train': train_IDs, 'valid': val_IDs}
        opt.ntrain, opt.nval = len(train_IDs), len(val_IDs)

        # calculate zoom ratio of z-axis 
        opt.pixel_size_axial = (opt.zeta[1] - opt.zeta[0] + 1 + 2*opt.clear_dist) / opt.D

        device = torch.device('cuda')
        torch.backends.cudnn.benchmark = True

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

        model = cnn
        criterion = calculate_loss(opt)
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
    # phase
    parser.add_argument('--train_or_test',            type=str,           default='test',        help='train or test')
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
    parser.add_argument('--batch_size',               type=int,           default=6,           help='when training on multi GPU, is the batch size on each GPU')
    parser.add_argument('--initial_learning_rate',    type=float,         default=0.0005,      help='initial learning rate for adam')
    # path
    parser.add_argument('--data_path',       type=str,           default='/home/lingjia/Documents/microscope/Data/training_images_zrange20',     help='path for train and val data')
    opt = parser.parse_args()

    gpu_number = len(opt.gpu_number.split(','))
    mp.spawn(learn_localization,args=(gpu_number,opt),nprocs=gpu_number,join=True)