# official modules
import numpy as np
import os
import torch
from torch.utils.tensorboard import SummaryWriter
import torch.distributed as dist
# self-defined modules
from .networks import *


def init_DDP(opt):
    os.environ['MASTER_ADDR']='localhost'
    os.environ['MASTER_PORT']='123457'
    gpus = [g.strip() for g in opt.gpu_number.split(',')]
    os.environ['CUDA_VISIBLE_DEVICES']=gpus[opt.rank]
    dist.init_process_group('GLOO',rank=opt.rank,world_size=opt.world_size)


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


def print_log(print_string, log, arrow=True):
    if arrow:
        print("---> {}".format(print_string))
        log.write('---> {}\n'.format(print_string))
        log.flush()
    else:
        print("{}".format(print_string))
        log.write('{}\n'.format(print_string))
        log.flush()


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


def build_model(opt):
    if opt.model_use == 'cnn':
        model = LocalizationCNN(opt)
    elif opt.model_use == 'cnn_residual':
        # 0808 difference with concatim is
        # (1) out = layer(out) + out -> residual conv layer
        # (2) deconv1 and deconv2 with +out or not
        model = ResLocalizationCNN(opt)

    return model


def print_metrics(metrics,epoch_samples,phase,log):
    outputs = []
    for k in metrics.keys():
        outputs.append("{}: {:.4f}".format(k, metrics[k] / epoch_samples))

    print_log("{}: {}".format(phase, ", ".join(outputs)),log)


# checkpoint saver for model weights and optimization status
def save_checkpoint(state, filename='checkpoint.pth.tar'):
    torch.save(state, filename)


def print_time(epoch_time_elapsed):
    str = '{:.0f}h {:.0f}m {:.0f}s'.format(
        epoch_time_elapsed // 3600, 
        np.floor((epoch_time_elapsed / 3600 - epoch_time_elapsed // 3600)*60), 
        epoch_time_elapsed % 60)

    return str




