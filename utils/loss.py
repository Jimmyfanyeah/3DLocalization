# official modules
from numpy.core.numeric import Inf
import scipy.io
import numpy as np
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import time
# self-defined modules
from utils.fft_conv import fft_conv

def prod(obj):
    p = 1
    for ii in range(len(obj)):
        p = p*obj[ii]
    return p


def print_time(epoch_time_elapsed):
    str = '{:.0f}h {:.0f}m {:.0f}s'.format(
        epoch_time_elapsed // 3600, 
        np.floor((epoch_time_elapsed / 3600 - epoch_time_elapsed // 3600)*60), 
        epoch_time_elapsed % 60)

    return str



############# definition of loss function
def GaussianKernel(shape=(7, 5, 5), sigma=1, normfactor=1):
    """
    TODO: create a 3D gaussian kernel
    3D gaussian mask - should give the same result as MATLAB's
    fspecial('gaussian',[shape],[sigma]) in 3D
    """
    m, n, p = [(ss - 1.) / 2. for ss in shape]
    y, x, z = np.ogrid[-m:m + 1, -n:n + 1, -p:p + 1]
    h = np.exp(-(x * x + y * y + z * z) / (2 * sigma ** 2))
    h[h < np.finfo(h.dtype).eps * h.max()] = 0
    """
    sumh = h.sum()
    if sumh != 0:
        h /= sumh
        h = h * normfactor
    """
    maxh = h.max()
    if maxh != 0:
        h /= maxh
        h = h * normfactor
    h = torch.from_numpy(h).type(torch.FloatTensor).cuda() # Variable()
    h = h.unsqueeze(0)
    h = h.unsqueeze(1)
    return h


class MSE3D(nn.Module):
    def __init__(self, opt):
        super(MSE3D, self).__init__()
        self.kernel = GaussianKernel()
        self.factor = opt.scaling_factor

    def forward(self, pred_bol, target_bol, gt_im=None):

        # extract kernel dimensions
        N,C,D,H,W = self.kernel.size()

        target_bol = target_bol.unsqueeze(1)
        pred_bol = pred_bol.unsqueeze(1)

        # KDE for both input and ground truth spikes
        Din = F.conv3d(pred_bol, self.kernel, padding=(int(np.round((D - 1) / 2)), 0, 0))
        Dtar = F.conv3d(target_bol, self.kernel, padding=(int(np.round((D - 1) / 2)), 0, 0))
        # Din = fft_conv(pred_bol, self.kernel, padding=(int(np.round((D - 1) / 2)), 0, 0))
        # Dtar = fft_conv(target_bol, self.kernel, padding=(int(np.round((D - 1) / 2)), 0, 0))

        kde_loss = nn.MSELoss()(Din, Dtar)

        return kde_loss


class CEL0Loss(nn.Module):
    def __init__(self,opt):
        super(CEL0Loss,self).__init__()
        # select for different D, D=41, 127, 250
        norm_ai_path = f'./matlab_codes/norm_ai_D{opt.D}.mat'
        self.norm_ai = scipy.io.loadmat(norm_ai_path)['norm_ai']
        self.norm_ai = torch.from_numpy(self.norm_ai).permute(2,0,1).to('cuda')
        self.weight = opt.cel0_mu

        # MaxPool3D: from 255,192,192 => 41,96,96
        # self.pool_info = (max(round(1/opt.pixel_size_axial),1),opt.upsampling_factor,opt.upsampling_factor) 
        # self.pool = nn.MaxPool3d(kernel_size=self.pool_info, stride=self.pool_info)
        # MaxPool2D: from 255,192,192 => 255,96,96
        self.pool_info = (opt.upsampling_factor,opt.upsampling_factor)
        self.pool = nn.MaxPool2d(kernel_size=self.pool_info, stride=self.pool_info)


    def forward(self,upgrid, target_bol=None, gt_im=None):

        # spikes_pred = upgrid

        # for maxpool3d
        # spikes_pred = self.pool(upgrid.unsqueeze(1))*prod(self.pool_info)
        # spikes_pred = spikes_pred.squeeze(1)

        # for maxpool2d
        spikes_pred = self.pool(upgrid)*prod(self.pool_info)

        # CEL0 on the predicted spikes
        norm_ai2 = torch.square(self.norm_ai)
        thresh = np.sqrt(2*self.weight)/self.norm_ai
        abs_heat = torch.abs(spikes_pred)
        bound = torch.square(abs_heat-thresh)
        ind = (abs_heat<=thresh).type(torch.float32)
        loss_spikes = torch.mean((self.weight-0.5*(norm_ai2*bound)*ind))
        # print(f'max pool: {print_time(after_maxpool-before_maxpool)} loss cost: {print_time(after_loss-after_maxpool)}')

        return loss_spikes



class klncLoss(nn.Module):
    def __init__(self,opt):
        super(klncLoss,self).__init__()
        self.a = opt.klnc_a
        self.pool_info = (max(round(1/opt.pixel_size_axial),1),opt.upsampling_factor,opt.upsampling_factor) 
        self.pool = nn.MaxPool3d(kernel_size=self.pool_info, stride=self.pool_info)

    def forward(self, upgrid, target_bol=None, gt_im=None):
        input = self.pool(upgrid.unsqueeze(1))*prod(self.pool_info)
        input = input.squeeze(1)
        klnc_loss = torch.sum(input/(self.a + input))

        return klnc_loss



def PSF_matrix():
    # PSF_path = './utils/data_natural_order_A.mat'
    PSF_path = './utils/A_41slices.mat'
    PSF_mat = scipy.io.loadmat(PSF_path)
    PSF_np = np.float32(PSF_mat['A'])
    PSF = torch.from_numpy(PSF_np).permute(2,0,1).unsqueeze(0).unsqueeze(0) # from [H,W,D] -> [1,1,D,H,W]
    # Reverse the PSF since different conv in python and matlab
    # https://zhuanlan.zhihu.com/p/103102579
    PSF = torch.flip(PSF,[2,3,4]).cuda()

    return PSF


class ForwardLoss(nn.Module):
    # 2021-08-17 Forward loss as fidelity term, ||A*y^-I0||^2
    # 2021-08-31 Implement conv3d using fft, details found in fft_conv.py
    # compare 2d observed image
    def __init__(self,opt):
        super(ForwardLoss,self).__init__()
        self.PSF = PSF_matrix()
        self.pool_info = (max(round(1/opt.pixel_size_axial),1),opt.upsampling_factor,opt.upsampling_factor) 
        # self.pool = nn.AvgPool3d(kernel_size=self.pool_info, stride=self.pool_info)
        self.pool = nn.MaxPool3d(kernel_size=self.pool_info, stride=self.pool_info)
        print(self.pool_info)

    def forward(self, upgrid, gt_upgrid=None, target=None):
        '''
        input = 3d predict grid [N,D,H,W], D is # of channel, considered as depth here
        target = 2d observed image
        pred = 2d image generated from input and PSF
        '''

        normgrid = self.pool(upgrid.unsqueeze(1))*prod(self.pool_info)
        # gt_normgrid = self.pool(gt_upgrid.unsqueeze(1))*prod(self.pool_info)

        # input = input.unsqueeze(1)  # [N,D,H,W] -> [N,1,D,H,W], # of channel =1
        _,_,D,H,W = self.PSF.shape
        (pd_d,pd_h,pd_w) = (math.floor((D-1)/2),math.floor((H-1)/2),math.floor((W-1)/2))

        pred = fft_conv(normgrid, self.PSF, padding=(pd_d,pd_h,pd_w),padding_mode='reflect')[:,0,20,:]
        mse_fft = nn.MSELoss()(pred,target)

        # pred_conv = F.conv3d(input, self.PSF, padding=(pd_d,pd_h,pd_w))[:,0,20,:]
        # mse_conv = nn.MSELoss()(pred_conv,target)
        # print(f'{mse_fft} {mse_conv}')

        return mse_fft



################ criterion
class calculate_loss(nn.Module):
    # 2021-07-17 add forward loss in to loss function
    def __init__(self, opt, loss_type, weight):
        super(calculate_loss, self).__init__()
        self.scaling_factor = opt.scaling_factor
        self.operators = {'mse3d':MSE3D, 'cel0':CEL0Loss, 'klnc':klncLoss, 'forward':ForwardLoss}
        self.ops = {}
        self.weights = weight

        assert len(loss_type)-1 <= len(weight), f'wrong length: {len(weight)} weight and {len(loss_type)-1} extra loss'

        for idx, lt in enumerate(loss_type):
            if not lt == 'loss':
                self.ops[lt] = self.operators[lt](opt)


    def forward(self, upgrid, gt_upgrid, gt_im=None, metric=None, metrics=None): # 

        losses = {}
        loss = 0
        for idx, key in enumerate(self.ops.keys()):
            # now = time.time()
            temp = self.ops[key](upgrid, gt_upgrid, gt_im)
            losses[key] = temp
            loss = loss + self.weights[idx] * temp
            # print(f'{key}: {print_time(time.time()-now)}')

        """ record loss this iter and total loss """
        if metric is not None:
            for key in losses:
                metric[key] = losses[key].detach().cpu().numpy()

            metric['loss'] = loss.data.cpu().numpy()

        if metrics is not None:
            for key in losses:
                metrics[key] += losses[key].detach().clone() * gt_upgrid.size(0)

            metrics['loss'] += loss.detach().clone() * gt_upgrid.size(0)

        return loss




def dice_loss(pred, target):
    """
    This definition generalize to real valued pred and target vector.
    pred: tensor with first dimension as batch
    target: tensor with first dimension as batch
    """

    smooth = 1.

    # have to use contiguous since they may from a torch.view op
    iflat = pred.contiguous().view(-1)
    tflat = target.contiguous().view(-1)
    intersection = (iflat * tflat).sum()

    A_sum = torch.sum(iflat * iflat)
    B_sum = torch.sum(tflat * tflat)

    return 1 - ((2. * intersection + smooth) / (A_sum + B_sum + smooth))
