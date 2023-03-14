# official modules
from numpy.core.numeric import Inf
import scipy.io
import numpy as np
import math
import torch
import torch.nn as nn
# self-defined modules
from utils.fft_conv import fft_conv

def prod(obj):
    p = 1
    for ii in range(len(obj)):
        p = p*obj[ii]
    return p


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

        kde_loss = nn.MSELoss()(Din, Dtar)

        return kde_loss


################ criterion
class calculate_loss(nn.Module):
    # 2021-07-17 add forward loss in to loss function
    def __init__(self, opt):
        super(calculate_loss, self).__init__()
        self.scaling_factor = opt.scaling_factor
        self.mse3d_loss = MSE3D(self.scaling_factor)
        self.pool_info = (round(1/opt.pixel_size_axial),opt.upsampling_factor,opt.upsampling_factor) 
        # self.pool = nn.AvgPool3d(kernel_size=self.pool_info, stride=self.pool_info)
        self.pool = nn.MaxPool3d(kernel_size=self.pool_info, stride=self.pool_info)


    def forward(self, upgrid, gt_upgrid, gt_im=None, metric=None, metrics=None):

        dice = dice_loss(upgrid/self.scaling_factor,gt_upgrid)
        reg = regularization_term(upgrid/self.scaling_factor, 1e6)
        mse3d = self.mse3d_loss(upgrid, gt_upgrid)

        # final loss
        loss = mse3d

        # record loss this iter and total loss
        if metric is not None:
            metric['dice'] = dice.data.cpu().numpy()
            metric['mse3d'] = mse3d.detach().cpu().numpy()

            metric['loss'] = loss.data.cpu().numpy()

        if metrics is not None:
            metrics['Dice'] += dice.detach().clone() * gt_upgrid.size(0)
            metrics['MSE3D'] += mse3d.detach().clone() * gt_upgrid.size(0)

            metrics['Loss'] += loss.detach().clone() * gt_upgrid.size(0)

        return loss


