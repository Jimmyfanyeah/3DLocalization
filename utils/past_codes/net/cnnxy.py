# BRANCH - CNN-XY

# Import modules and libraries
import torch
from torch.utils.data import Dataset
import numpy as np
import scipy.io
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler as Disample

def GaussianKernel(shape=(7, 7), sigma=1, normfactor=1):
    m, n = [(ss - 1.) / 2. for ss in shape]
    y, x = np.ogrid[-m:m + 1, -n:n + 1]
    h = np.exp(-(x * x + y * y) / (2 * sigma ** 2))
    h[h < np.finfo(h.dtype).eps * h.max()] = 0

    maxh = h.max()
    if maxh != 0:
        h /= maxh
        h = h * normfactor
    h = torch.from_numpy(h).type(torch.FloatTensor)
    h = h.unsqueeze(0)
    h = h.unsqueeze(0)
    return h

def gen_label(xyz_np,setup_params):

    upsampling_factor = setup_params['upsampling_factor']
    gaussian_kernel = GaussianKernel()
    kernel = torch.ones([3,3]).double().unsqueeze(0).unsqueeze(0)
    kN,kC,kH,kW = kernel.size()

    zshift = xyz_np[:,:,2]
    batch_size, num_particles = zshift.shape
    vals = torch.ones(batch_size*num_particles)

    ######## channel 1 xy position
    H, W = setup_params['H'], setup_params['W']

    # project xyz locations on the grid and shift xy to the upper left corner
    xg = (np.floor((xyz_np[:, :, 0] + W/2)*upsampling_factor)).astype('int')
    yg = (np.floor((xyz_np[:, :, 1] + H/2)*upsampling_factor)).astype('int')

    H, W = int(H * upsampling_factor), int(W * upsampling_factor)

    indX, indY = (xg.flatten('F')).tolist(), (yg.flatten('F')).tolist()

    channel_xy_ind = torch.LongTensor([indY, indX])
    channel_xy = torch.sparse.FloatTensor(channel_xy_ind, vals, torch.Size([H, W])).to_dense().unsqueeze(0).unsqueeze(0)
    channel_xy = F.conv2d(channel_xy,gaussian_kernel,padding=(int(np.round((kH - 1) / 2)), int(np.round((kW - 1) / 2))))

    ####### channel 2 zeta position
    zshift = xyz_np[:,:,2] - setup_params['zmin']  # shift the z axis back to 0
    vals = torch.tensor(zshift).squeeze()

    channel_z = torch.sparse.FloatTensor(channel_xy_ind, vals, torch.Size([H, W])).to_dense().unsqueeze(0).unsqueeze(0)
    channel_z = F.conv2d(channel_z,kernel, padding=(int(np.round((kH - 1) / 2)), int(np.round((kW - 1) / 2))))

    ######## final label
    labels = torch.cat([channel_xy,channel_z],dim=1).squeeze(0).type(torch.FloatTensor)

    return labels

def gen_label_xy(xyz_np,setup_params):

    upsampling_factor = setup_params['upsampling_factor']
    scaling_factor = setup_params['scaling_factor']
    gaussian_kernel = GaussianKernel()
    kernel = torch.ones([3,3]).double().unsqueeze(0).unsqueeze(0)
    kN,kC,kH,kW = gaussian_kernel.size()

    zshift = xyz_np[:,:,2]
    batch_size, num_particles = zshift.shape
    vals = torch.ones(batch_size*num_particles)

    ######## channel 1 xy position
    H, W = setup_params['H'], setup_params['W']

    # project xyz locations on the grid and shift xy to the upper left corner
    xg = (np.floor((xyz_np[:, :, 0] + W/2)*upsampling_factor)).astype('int')
    yg = (np.floor((xyz_np[:, :, 1] + H/2)*upsampling_factor)).astype('int')

    H, W = int(H * upsampling_factor), int(W * upsampling_factor)

    indX, indY = (xg.flatten('F')).tolist(), (yg.flatten('F')).tolist()

    channel_xy_ind = torch.LongTensor([indY, indX])
    channel_xy = torch.sparse.FloatTensor(channel_xy_ind, vals, torch.Size([H, W])).to_dense().unsqueeze(0).unsqueeze(0)
    channel_xy = F.conv2d(channel_xy,gaussian_kernel*scaling_factor,padding=(int(np.round((kH - 1) / 2)), int(np.round((kW - 1) / 2))))

    ######## final label
    labels = channel_xy.squeeze(0).type(torch.FloatTensor)

    return labels

def dataloader(path_train, labels, params, setup_params, opt, num_workers=0):

    dataset = ImagesDataset(path_train, params['partition'], labels, setup_params)
    batch_size = params['batch_size']
    shuffle = params['shuffle']

    try:
        Sampler = Disample(dataset,num_replicas=opt.world_size,rank=opt.rank,shuffle=shuffle)
        dl = DataLoader(dataset,batch_size=batch_size,sampler=Sampler,num_workers=num_workers,pin_memory=True)
    except:
        dl = DataLoader(dataset,batch_size=batch_size,shuffle=shuffle,num_workers=num_workers,pin_memory=True)
    return dl

class ImagesDataset(Dataset):

    # initialization of the dataset
    def __init__(self, root_dir, list_IDs, labels, setup_params):
        self.root_dir = root_dir
        self.list_IDs = list_IDs
        self.labels = labels
        self.setup_params = setup_params
        # self.train_stats = setup_params['train_stats']

    def __len__(self):
        return len(self.list_IDs)

    def __getitem__(self, index):

        # select sample
        ID = self.list_IDs[index]

        im_name = self.root_dir + '/im' + ID + '.mat'
        im_mat = scipy.io.loadmat(im_name)
        im_np = np.float32(im_mat['g'])
        im_tensor = torch.from_numpy(im_np).unsqueeze(0)

        # corresponding xyz labels turned to a boolean tensor
        xyz_np = self.labels[ID]
        labels = gen_label_xy(xyz_np, self.setup_params)

        return im_tensor, labels, ID


def buildModel(setup_params):
    # from .cnn_utils import LocalizationCNN
    # model = LocalizationCNN(setup_params)
    # elif setup_params['model'] == 'resnet':
    #     from .resnet import ResNet
    #     model = ResNet([2, 2, 2, 2, 2],setup_params)

    # from .ResNetUNet import ResNetUNet
    # model = ResNetUNet(n_class=1)

    from .cnn import CNN
    model = CNN(setup_params)

    return model

def recall(pred, target):
    smooth = 1e-4

    # have to use contiguous since they may from a torch.view op
    iflat = pred.contiguous().view(-1)
    tflat = target.contiguous().view(-1)
    intersection = (iflat * tflat).sum()

    return 1 - (intersection + smooth) / ( torch.sum(tflat) + smooth)


class calculate_loss(nn.Module):
    def __init__(self, scaling_factor):
        super(calculate_loss, self).__init__()
        self.scaling_factor = scaling_factor

    def forward(self, pred_xy, target, metric, metrics):

        loss = nn.MSELoss()(pred_xy, target)

        metric['loss'] = loss.data.cpu().numpy()

        metrics['loss'] += loss.detach().clone() * target.size(0)

        return loss


import torch.nn as nn
import torch.nn.functional as F
from torch.nn.functional import interpolate

def convrelubn(in_channels, out_channels, kernel, padding=1, dilation=1):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel, padding=padding, dilation=dilation),
        nn.ReLU(inplace=True),
        nn.BatchNorm2d(out_channels)
    )

class CNN(nn.Module):
    def __init__(self, setup_params):
        super(CNN,self).__init__()

        self.norm = nn.BatchNorm2d(num_features=1, affine=True)
        self.layer1 = convrelubn(1, 16, 3)
        self.layer2 = convrelubn(16, 16, 3)
        self.layer3 = convrelubn(16, 32, 3)
        self.layer4 = convrelubn(32, 32, 3)
        self.layer5 = convrelubn(32, 64, 3)
        self.layer6 = convrelubn(64, 64, 3)
        self.layer6_1 = convrelubn(64, 64, 3)
        self.layer7 = convrelubn(64, 32, 3)
        self.layer8 = convrelubn(32, 32, 3)
        self.layer9 = convrelubn(32, 16, 3)
        self.layer10 = convrelubn(16, 1, 3)

        self.pred = nn.Hardtanh(min_val=0.0, max_val=setup_params['scaling_factor'])

    def forward(self, im):

        im = self.norm(im) # [4,1,96,96]
        out = self.layer1(im)
        out = self.layer2(out) 
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.layer5(out)
        out = self.layer6(out)
        out = interpolate(out, scale_factor=2)
        out = self.layer6_1(out)
        out = self.layer7(out)
        out = self.layer8(out)
        out = self.layer9(out)
        out = self.layer10(out)
        out = self.pred(out)

        return out

# Import modules and libraries
import numpy as np
import torch
from torch.nn import Module, MaxPool2d, ConstantPad3d
from torch.nn.functional import conv3d


# convert gpu tensors to numpy
def tensor_to_np(x):
    return np.squeeze(x.cpu().numpy())

# post-processing on GPU: thresholding and local maxima finding
class Postprocess(Module):
    def __init__(self, thresh, radius, setup_params):
        super().__init__()
        self.thresh = thresh
        self.r = radius
        self.device = setup_params['device']
        self.psize_xy = 1/setup_params['upsampling_factor']
        self.upsampling_shift = 0  # 2 due to floor(W/2) affected by upsampling factor of 4
        self.maxpool = MaxPool2d(kernel_size=2*self.r + 1, stride=1, padding=self.r)
        # self.pad = ConstantPad3d(self.r, 0.0)
        self.zero = torch.FloatTensor([0.0]).to(self.device)
        self.upsampling_factor = setup_params['upsampling_factor']
        self.H = setup_params['H']
        self.W = setup_params['W']

    def forward(self, pred_vol):
        # pred_vol = [N,1,192,192]
        num_dims = len(pred_vol.size())
        if np.not_equal(num_dims, 4):
            if num_dims == 3:
                pred_vol = pred_vol.unsqueeze(0)
            else:
                pred_vol = pred_vol.unsqueeze(0)
                pred_vol = pred_vol.unsqueeze(0)

        # apply the threshold
        pred_thresh = torch.where(pred_vol > self.thresh, pred_vol, self.zero)

        # apply the 3D maxpooling operation to find local maxima
        conf_vol = self.maxpool(pred_thresh)
        conf_vol = torch.where((conf_vol > self.zero) & (conf_vol == pred_thresh), conf_vol, self.zero)

        # find locations of confs (bigger than 0)
        conf_vol = torch.squeeze(conf_vol)
        batch_indices = torch.nonzero(conf_vol)
        ybool, xbool = batch_indices[:, 0], batch_indices[:, 1]

        # if the prediction is empty return None otherwise convert to list of locations
        if len(ybool) == 0:
            xyz_rec = None
            conf_rec = None
        else:
            # convert lists and tensors to numpy
            xbool, ybool = tensor_to_np(xbool), tensor_to_np(ybool)

            # dimensions of the prediction
            H, W = conf_vol.size()

            # calculate the recovered positions assuming mid-voxel
            xrec = (xbool - np.floor(W / 2) + self.upsampling_shift + 0.5) * self.psize_xy
            yrec = (ybool - np.floor(H / 2) + self.upsampling_shift + 0.5) * self.psize_xy

            # rearrange the result into a Nx3 array
            xyz_rec = np.column_stack((xrec, yrec))
            xyz_bool = np.column_stack((xbool,ybool))

            # confidence of these positions
            conf_rec = conf_vol[ybool, xbool]
            conf_rec = tensor_to_np(conf_rec)

        return xyz_rec, conf_rec, xyz_bool



import torch 
import torch.nn as nn

class UNet3D(nn.Module):
    def __init__(self, in_channel, n_classes):
        self.in_channel = in_channel
        self.n_classes = n_classes
        super(UNet3D, self).__init__()
        self.ec0 = self.encoder(self.in_channel, 32, bias=False, batchnorm=False)
        self.ec1 = self.encoder(32, 64, bias=False, batchnorm=False)
        self.ec2 = self.encoder(64, 64, bias=False, batchnorm=False)
        self.ec3 = self.encoder(64, 128, bias=False, batchnorm=False)
        self.ec4 = self.encoder(128, 128, bias=False, batchnorm=False)
        self.ec5 = self.encoder(128, 256, bias=False, batchnorm=False)
        self.ec6 = self.encoder(256, 256, bias=False, batchnorm=False)
        self.ec7 = self.encoder(256, 512, bias=False, batchnorm=False)

        self.pool0 = nn.MaxPool3d(2)
        self.pool1 = nn.MaxPool3d(2)
        self.pool2 = nn.MaxPool3d(2)

        self.dc9 = self.decoder(512, 512, kernel_size=2, stride=2, bias=False)
        self.dc8 = self.decoder(256 + 512, 256, kernel_size=3, stride=1, padding=1, bias=False)
        self.dc7 = self.decoder(256, 256, kernel_size=3, stride=1, padding=1, bias=False)
        self.dc6 = self.decoder(256, 256, kernel_size=2, stride=2, bias=False)
        self.dc5 = self.decoder(128 + 256, 128, kernel_size=3, stride=1, padding=1, bias=False)
        self.dc4 = self.decoder(128, 128, kernel_size=3, stride=1, padding=1, bias=False)
        self.dc3 = self.decoder(128, 128, kernel_size=2, stride=2, bias=False)
        self.dc2 = self.decoder(64 + 128, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.dc1 = self.decoder(64, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.dc0 = self.decoder(64, n_classes, kernel_size=1, stride=1, bias=False)

    def encoder(self, in_channels, out_channels, kernel_size=3, stride=1, padding=0,
                bias=True, batchnorm=False):
        if batchnorm:
            layer = nn.Sequential(
                nn.Conv3d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, bias=bias),
                nn.BatchNorm3d(out_channels),
                nn.ReLU())
        else:
            layer = nn.Sequential(
                nn.Conv3d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, bias=bias),
                nn.ReLU())
        return layer

    def decoder(self, in_channels, out_channels, kernel_size, stride=1, padding=0,
                output_padding=0, bias=True):
        layer = nn.Sequential(
            nn.ConvTranspose3d(in_channels, out_channels, kernel_size, stride=stride,
                               padding=padding, output_padding=output_padding, bias=bias),
            nn.ReLU())
        return layer

    def forward(self, x):
        e0 = self.ec0(x)   #32
        syn0 = self.ec1(e0)  #64

        e1 = self.pool0(syn0)
        e2 = self.ec2(e1)  #64
        syn1 = self.ec3(e2)  #128
        del e0, e1, e2

        e3 = self.pool1(syn1)
        e4 = self.ec4(e3)  #128
        syn2 = self.ec5(e4)  #256
        del e3, e4

        e5 = self.pool2(syn2)
        e6 = self.ec6(e5)  #256
        e7 = self.ec7(e6)  #512
        del e5, e6

        d9 = torch.cat((self.dc9(e7), syn2)) #512+256
        del e7, syn2

        d8 = self.dc8(d9) #256
        d7 = self.dc7(d8) #256
        del d9, d8

        d6 = torch.cat((self.dc6(d7), syn1)) #256+128
        del d7, syn1

        d5 = self.dc5(d6)
        d4 = self.dc4(d5)
        del d6, d5

        d3 = torch.cat((self.dc3(d4), syn0))
        del d4, syn0

        d2 = self.dc2(d3)
        d1 = self.dc1(d2)
        del d3, d2

        d0 = self.dc0(d1)
        return d0






















