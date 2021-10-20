# Import modules and libraries
import scipy.io
import numpy as np
from math import ceil,floor
import torch
import torch.nn as nn
import torch.nn.functional as F
from utils.fft_conv import fft_conv

def prod(obj):
    p = 1
    for ii in range(len(obj)):
        p = p*obj[ii]
    return p


# create a 3D gaussian kernel
def GaussianKernel(shape=(7, 5, 5), sigma=1, normfactor=1):
    """
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


def PSF_matrix():
    # PSF_path = '/home/lingjia/Documents/3dloc/utils/data_natural_order_A.mat'
    PSF_path = './utils/A_41slices.mat'
    PSF_mat = scipy.io.loadmat(PSF_path)
    PSF_np = np.float32(PSF_mat['A'])
    PSF = torch.from_numpy(PSF_np).permute(2,0,1).unsqueeze(0).unsqueeze(0) # from [H,W,D] -> [1,1,D,H,W]
    # Reverse the PSF since different conv in python and matlab
    # https://zhuanlan.zhihu.com/p/103102579
    PSF = torch.flip(PSF,[2,3,4]).cuda()

    return PSF


############# definition of loss function
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


def regularization_term(pred, a):

    reg_loss = torch.sum(pred)/(a + torch.sum(pred))
    return reg_loss


class MSE3D(nn.Module):
    def __init__(self, factor):
        super(MSE3D, self).__init__()
        self.kernel = GaussianKernel()
        self.factor = factor

    def forward(self, pred_bol, target_bol):

        # extract kernel dimensions
        N,C,D,H,W = self.kernel.size()

        target_bol = target_bol.unsqueeze(1)
        pred_bol = pred_bol.unsqueeze(1)

        # KDE for both input and ground truth spikes
        # Din = F.conv3d(pred_bol, self.kernel, padding=(int(np.round((D - 1) / 2)), 0, 0))
        # Dtar = F.conv3d(target_bol, self.kernel, padding=(int(np.round((D - 1) / 2)), 0, 0))
        Din = fft_conv(pred_bol, self.kernel, padding=(int(np.round((D - 1) / 2)), 0, 0))
        Dtar = fft_conv(target_bol, self.kernel, padding=(int(np.round((D - 1) / 2)), 0, 0))

        # kde_loss = nn.MSELoss(reduction='sum')(Din, Dtar)
        kde_loss = nn.MSELoss()(Din, Dtar)
        # kde_loss = nn.MSELoss()(pred_bol,target_bol)

        return kde_loss


class Forward_loss(nn.Module):
    # 2021-08-17 Forward loss as fidelity term, ||A*y^-I0||^2
    # 2021-08-31 Implement conv3d using fft, details found in fft_conv.py
    # compare 2d observed image
    def __init__(self):
        super(Forward_loss,self).__init__()
        self.PSF = PSF_matrix()

    def forward(self,input,target):
        '''
        input = 3d predict grid [N,D,H,W], D is # of channel, considered as depth here
        target = 2d observed image
        pred = 2d image generated from input and PSF
        '''

        # input = input.unsqueeze(1)  # [N,D,H,W] -> [N,1,D,H,W], # of channel =1
        _,_,D,H,W = self.PSF.shape
        (pd_d,pd_h,pd_w) = (floor((D-1)/2),floor((H-1)/2),floor((W-1)/2))

        pred = fft_conv(input, self.PSF, padding=(pd_d,pd_h,pd_w),padding_mode='reflect')[:,0,20,:]
        mse_fft = nn.MSELoss()(pred,target)

        # pred_conv = F.conv3d(input, self.PSF, padding=(pd_d,pd_h,pd_w))[:,0,20,:]
        # mse_conv = nn.MSELoss()(pred_conv,target)
        # print(f'{mse_fft} {mse_conv}')

        return mse_fft


################ criterion
class calculate_loss_v2(nn.Module):
    # 2021-07-17 add forward loss in to loss function
    def __init__(self, setup_params):
        super(calculate_loss_v2, self).__init__()
        self.scaling_factor = setup_params['scaling_factor']
        self.mse3d_loss = MSE3D(self.scaling_factor)

        self.pool_info = (round(1/setup_params['pixel_size_axial']),setup_params['upsampling_factor'],setup_params['upsampling_factor']) 
        self.pool = nn.MaxPool3d(kernel_size=self.pool_info, stride=self.pool_info)
        self.forward_loss = Forward_loss()

    def forward(self, upgrid, gt_upgrid, gt_im, metric, metrics):

        dice = dice_loss(upgrid/self.scaling_factor,gt_upgrid)
        reg = regularization_term(upgrid/self.scaling_factor, 1e6)
        mse3d = self.mse3d_loss(upgrid, gt_upgrid)

        normgrid = self.pool(upgrid.unsqueeze(1))
        mse2d = self.forward_loss(normgrid,gt_im)

        # final loss
        # loss = mse3d + mse2d/1e5 + reg
        # loss = mse3d + mse2d/1e5
        loss = mse3d

        # record loss this iter and total loss
        metric['dice'] = dice.data.cpu().numpy()
        metric['reg'] = reg.data.cpu().numpy()
        metric['mse3d'] = mse3d.detach().cpu().numpy()
        metric['mse2d'] = mse2d.data.cpu().numpy()

        metrics['Dice'] += dice.detach().clone() * gt_upgrid.size(0)
        metrics['Reg'] += reg.detach().clone() * gt_upgrid.size(0)
        metrics['MSE3D'] += mse3d.detach().clone() * gt_upgrid.size(0)
        metrics['MSE2D'] += mse2d.detach().clone() * gt_upgrid.size(0)

        metric['loss'] = loss.data.cpu().numpy()
        metrics['Loss'] += loss.detach().clone() * gt_upgrid.size(0)

        return loss



class calculate_loss(nn.Module):
    def __init__(self, scaling_factor):
        super(calculate_loss, self).__init__()
        self.scaling_factor = scaling_factor
        self.criterion = KDE_loss3D(self.scaling_factor)

    def forward(self, pred, target, metric, metrics):

        # only calc loss for 3d location, pred in [0,800] target in 0/1
        kde = self.criterion(pred, target)
        dice = dice_loss(pred/self.scaling_factor,target)
        reg_loss = reg(pred/self.scaling_factor, 100000)

        loss = kde

        metric['dice'] = dice.data.cpu().numpy()
        metric['kde'] = kde.detach().cpu().numpy()
        metric['reg_loss'] = reg_loss.detach().cpu().numpy()
        metric['loss'] = loss.data.cpu().numpy()

        metrics['Dice'] += dice.detach().clone() * target.size(0)
        metrics['KDE'] += kde.detach().clone() * target.size(0)
        metrics['Reg'] += reg_loss.detach().clone() * target.size(0)
        metrics['Loss'] += loss.detach().clone() * target.size(0)

        return loss


# used for lr_find.py
class calculate_loss_lrfind(nn.Module):
    def __init__(self, scaling_factor):
        super(calculate_loss_lrfind, self).__init__()
        self.scaling_factor = scaling_factor
        self.criterion = MSE3D(self.scaling_factor)

    def forward(self, pred, target):

        kde = self.criterion(pred, target)
        loss = kde

        return loss

