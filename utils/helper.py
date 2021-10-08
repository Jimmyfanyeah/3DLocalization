# Import modules and libraries
import torch
import numpy as np
import os
from skimage.io import imread
from collections import OrderedDict
from torch.utils.tensorboard import SummaryWriter
import torch.distributed as dist
from .cnn import *
import time

def load_labels(label_txt_path):
    label_raw = np.loadtxt(label_txt_path)
    if len(label_raw.shape) == 1:
        label_raw = np.expand_dims(label_raw, axis=0)
    labels = {}
    for i in range(len(label_raw)):
        if not str(int(label_raw[i,0])) in labels.keys():
            labels[str(int(label_raw[i,0]))] = label_raw[i,1:5]
            continue
        labels[str(int(label_raw[i,0]))]=np.c_[labels[str(int(label_raw[i,0]))], label_raw[i,1:5]]
    for i in labels.keys():
        if len(labels[i].shape)==2:
            labels[i] = np.expand_dims(labels[i].T,0) # [1,n,3]
        elif len(labels[i].shape)==1: # labels[i] = 3*n, n is number of source points,
            labels[i] = np.expand_dims(labels[i].T,0) # [1,3]
            labels[i] = np.expand_dims(labels[i],0) # [1,1,3]

    return labels


class Logger(object):

  def __init__(self, log_dir):
    """Create a summary writer logging to log_dir."""
    #self.writer = tf.summary.FileWriter(log_dir)
    self.writer = SummaryWriter(log_dir)

  def scalar_summary(self, tag, value, step):
    """Log a scalar variable."""
    # tf.summary.scalar(tag, value, step = step)
    self.writer.add_scalar(tag, value, step)

  def scalars_summary(self, tag, tag_scalar_dict, step):
        self.writer.add_scalars(tag, tag_scalar_dict, step)


def print_metrics(metrics, epoch_samples, phase,log):
    outputs = []
    for k in metrics.keys():
        outputs.append("{}: {:.4f}".format(k, metrics[k] / epoch_samples))

    print_log("{}: {}".format(phase, ", ".join(outputs)),log)

 
def print_log(print_string, log, arrow=True):
    if arrow:
        print("---> {}".format(print_string))
        log.write('---> {}\n'.format(print_string))
        log.flush()
    else:
        print("{}".format(print_string))
        log.write('{}\n'.format(print_string))
        log.flush()


def init_DDP(opt):
    os.environ['MASTER_ADDR']='localhost'
    os.environ['MASTER_PORT']='123457'
    gpus = [g.strip() for g in opt.gpu_number.split(',')]
    os.environ['CUDA_VISIBLE_DEVICES']=gpus[opt.rank]
    dist.init_process_group('GLOO',rank=opt.rank,world_size=opt.world_size)


def buildModel(setup_params):
    if setup_params['model_use'] == 'cnn':
        model = LocalizationCNN(setup_params)
    elif setup_params['model_use'] == 'cnn_residual':
        model = ResLocalizationCNN(setup_params)

    return model


# ======================================================================================================================
# saving and resuming utils
# ======================================================================================================================


# checkpoint saver for model weights and optimization status
def save_checkpoint(state, filename='checkpoint.pth.tar'):
    torch.save(state, filename)

# resume training from a check point
def resume_from_checkpoint(model, optimizer, filepath,log):
    print_log("---> loading checkpoint to resume training", log)
    checkpoint = torch.load(filepath)
    start_epoch = checkpoint['epoch']
    model.load_state_dict(checkpoint['state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer'])
    print_log("---> loaded checkpoint (epoch {})".format(checkpoint['epoch']), log)
    return start_epoch

# load pretrained LR model to fine-tune for HR recovery
def prone_state_dict(saved_state_dict, load_last=True):
    
    # go over saved keys
    saved_dict_short = {}
    for k, v in saved_state_dict.items():
        
        # check the seventh layer number of input channels
        if (k == 'layer7.conv.weight' and v.size(1) == 65) or not(load_last):
            break
        else:
            saved_dict_short[k] = v
    
    return OrderedDict(saved_state_dict.items())

# transform xyz from microns to nms for saving and handling later in ThunderSTORM
def xyz_to_nm(xyz_um, ch, cw, psize_rec_xy, zmin):

    xnm = (xyz_um[:, 0] + cw * psize_rec_xy) * 1000
    ynm = (xyz_um[:, 1] + ch * psize_rec_xy) * 1000
    znm = (xyz_um[:, 2] - zmin) * 1000

    return np.column_stack((xnm, ynm, znm))

# =====================================================================================================================
# Normalization factors calculation and image projection to the range [0,1]
# ======================================================================================================================


def normalize_01(im):
    return (im - im.min())/(im.max() - im.min())


def CalcMeanStd_All(path_train, labels):
    """
    function calculates the mean and std (per-pixel!) for the training dataset,
    both these normalization factors are used for training and validation.
    """
    num_examples = len(labels)
    mean = 0.0
    for i in range(num_examples):
        im_name_tiff = path_train + 'im' + str(i) + '.tiff'
        im_tiff = imread(im_name_tiff)
        mean += im_tiff.mean()
    mean = mean / num_examples
    var = 0.0
    for i in range(num_examples):
        im_name_tiff = path_train + 'im' + str(i) + '.tiff'
        im_tiff = imread(im_name_tiff)
        var += ((im_tiff - mean)**2).sum()
    H, W = im_tiff.shape
    var = var/(num_examples*H*W)
    std = np.sqrt(var)
    return mean, std


def print_time(epoch_time_elapsed):
    str = '{:.0f}h {:.0f}m {:.0f}s'.format(
        epoch_time_elapsed // 3600, 
        np.floor((epoch_time_elapsed / 3600 - epoch_time_elapsed // 3600)*60), 
        epoch_time_elapsed % 60)

    return str

