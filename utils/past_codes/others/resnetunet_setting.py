# BRANCH ResNetUNet

def Postprocess_xy(pred_volume, setup_params):

    psize_xy = 1/setup_params['upsampling_factor']
    W, H = setup_params['W'], setup_params['H']
    zero = torch.FloatTensor([0.0]).to(pred_volume.device)

    pred_vol = torch.sigmoid(pred_volume)
    pred_thresh = torch.where(pred_vol > 0.5, pred_vol,zero)

    conf_vol = pred_thresh
    conf_vol = torch.where((conf_vol > 0) & (conf_vol == pred_thresh), conf_vol, zero)

    # find locations of confs (bigger than 0)
    conf_vol = torch.squeeze(conf_vol)
    batch_indices = torch.nonzero(conf_vol)

    ybool, xbool = batch_indices[:, 0], batch_indices[:, 1]

    H, W = conf_vol.size()
    xrec = (xbool  - np.floor(W / 2) ) * psize_xy
    yrec = (ybool  - np.floor(H / 2) ) * psize_xy

    yrec, xrec = tensor_to_np(yrec), tensor_to_np(xrec)

    xyz_bool = np.column_stack((yrec, xrec))
    conf_rec = conf_vol[ybool, xbool]
    conf_rec = tensor_to_np(conf_rec)

    return xyz_bool, conf_rec




# calculates the jaccard coefficient approximation using per-voxel probabilities
def jaccard_coeff(pred, target):
    """
    jaccard index = TP / (TP + FP + FN)
    pred: tensor with first dimension as batch
    target: tensor with first dimension as batch
    """

    # smoothing parameter
    smooth = 1e-6
    # number of examples in the batch
    N = pred.size(0)
    # have to use contiguous since they may from a torch.view op
    iflat = pred.contiguous().view(N,-1)
    tflat = target.contiguous().view(N,-1)
    intersection = (iflat * tflat).sum(1)
    jacc_index = (intersection / (iflat.sum(1) + tflat.sum(1) - intersection + smooth)).mean()

    return jacc_index


def GaussianKernel_2d(shape=(3, 3), sigma=1, normfactor=1):

    m, n = [(ss - 1.) / 2. for ss in shape]
    y, x = np.ogrid[-m:m + 1, -n:n + 1]
    h = np.exp(-(x * x + y * y) / (2 * sigma ** 2))
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


def calc_loss_xy(pred, target, metric, metrics):

    # gaussian blur
    # kernel = GaussianKernel_2d()
    kernel = torch.ones((1,1,3,3)).to(pred.device)
    N, C, H, W = kernel.size()  # [1,1,3,3]

    # kenel factor = 500
    target_blur = F.conv2d(target.unsqueeze(1), kernel, padding=(int(np.round((H - 1) / 2)), int(np.round((W - 1) / 2))))
    # pred_blur = F.conv3d(pred, kernel, padding=(int(np.round((H - 1) / 2)), int(np.round((W - 1) / 2))))

    # mse = nn.MSELoss()(target_blur,pred)
    # number of positive = nSource*7*7*7 (which is the kernel size), number of all = D*W*H
    bce = F.binary_cross_entropy_with_logits(pred,target_blur,pos_weight=torch.tensor(500,device=pred.device))
    loss = bce

    pred = torch.sigmoid(pred)
    jacc_ind = jaccard_coeff(pred, target_blur)
    dice = dice_loss(pred,target_blur)

    metric['dice'] = dice.data.cpu().numpy()
    metric['loss'] = loss.data.cpu().numpy()
    metric['jacc_ind'] = jacc_ind.detach().cpu().numpy()

    metrics['Dice'] += dice.detach().clone() * target.size(0)
    metrics['Loss'] += loss.detach().clone() * target.size(0)
    metrics['Jacc_ind'] += jacc_ind.detach().clone() * target.size(0)

    return loss

def calc_loss_xyz_flux(pred_xy, pred_zeta, pred_flux, target, metric, metrics):

    # gaussian blur
    # kernel = GaussianKernel_2d()
    # kenel factor = 500
    # mse = nn.MSELoss()(target_blur,pred)

    # loss for xy position
    target_xy = target[0,:,:]
    kernel = torch.ones((1,1,3,3)).to(pred_xy.device)
    N, C, H, W = kernel.size()  # [1,1,3,3]

    target_blur = F.conv2d(target_xy.unsqueeze(1), kernel, padding=(int(np.round((H - 1) / 2)), int(np.round((W - 1) / 2))))
    # pred_blur = F.conv3d(pred, kernel, padding=(int(np.round((H - 1) / 2)), int(np.round((W - 1) / 2))))

    # number of positive = nSource*3*3 (which is the kernel size), number of all = W*H, 96*96*4/(20*3*3) -> 200
    bce_xy = F.binary_cross_entropy_with_logits(pred_xy,target_blur,pos_weight=torch.tensor(500,device=pred_xy.device))

    weight_xy = torch.sigmoid(pred_xy)
    # mse for zeta position
    target_zeta = target[1,:,:]
    mse_zeta = nn.MSELoss()(target_zeta,pred_zeta).mul(weight_xy)

    target_flux = target[2,:,:]
    mse_flux = nn.MSELoss()(target_flux,pred_flux).mul(weight_xy)

    loss = bce_xy + mse_zeta + mse_flux

    pred = torch.sigmoid(pred_xy)
    jacc_ind = jaccard_coeff(pred, target_blur)
    dice = dice_loss(pred,target_blur)

    metric['dice'] = dice.data.cpu().numpy()
    metric['loss'] = loss.data.cpu().numpy()
    metric['jacc_ind'] = jacc_ind.detach().cpu().numpy()

    metrics['Dice'] += dice.detach().clone() * target.size(0)
    metrics['Loss'] += loss.detach().clone() * target.size(0)
    metrics['Jacc_ind'] += jacc_ind.detach().clone() * target.size(0)

    return loss



def buildModel(setup_params):
    if setup_params['model'] == 'deepstorm3d':
        from .cnn_utils import LocalizationCNN
        model = LocalizationCNN(setup_params)
    elif setup_params['model'] == 'resnet':
        from .resnet import ResNet
        model = ResNet([2, 2, 2, 2, 2],setup_params)
    elif setup_params['model'] == 'ResNetUNet':
        from .resnet_unet import ResNetUNet
        model = ResNetUNet()
    return model

# converts continuous xyz locations to a 2d image label with xy position in grid
def batch_xyz_to_2d_image_label(xyz_np,setup_params):

    upsampling_factor = setup_params['upsampling_factor']
    H, W, D = setup_params['H'], setup_params['W'], setup_params['D']

    batch_size, num_particles = xyz_np[:,:,1].shape

    # project xyz locations on the grid and shift xy to the upper left corner
    xg_floor = (np.floor((xyz_np[:, :, 0] + W/2)*upsampling_factor)).astype('int')
    yg_floor = (np.floor((xyz_np[:, :, 1] + H/2)*upsampling_factor)).astype('int')
    # xg_ceil = (np.ceil((xyz_np[:, :, 0] + W/2)*upsampling_factor)).astype('int')
    # yg_ceil = (np.ceil((xyz_np[:, :, 1] + H/2)*upsampling_factor)).astype('int')
    # indices for sparse tensor
    indX_floor, indY_floor = (xg_floor.flatten('F')).tolist(), (yg_floor.flatten('F')).tolist()
    # indX_ceil, indY_ceil = (xg_ceil.flatten('F')).tolist(), (yg_ceil.flatten('F')).tolist()
    # indX, indY = indX_floor + indX_ceil, indY_floor + indY_ceil
    indX, indY = indX_floor, indY_floor

    # if sampling a batch add a sample index
    if batch_size > 1:
        indS = (np.kron(np.ones(num_particles), np.arange(0, batch_size, 1)).astype('int')).tolist()
        ibool = torch.LongTensor([indS, indY, indX])
    else:
        ibool = torch.LongTensor([indY, indX])

    # spikes for sparse tensor
    vals = torch.ones(batch_size*num_particles)
    # vals = torch.ones(batch_size*num_particles)*xyz_np[:,:,3].squeeze()

    H, W = int(H * upsampling_factor), int(W * upsampling_factor)
    # resulting 3D boolean tensor
    # if batch_size > 1:
        # boolean_grid = torch.sparse.FloatTensor(ibool, vals, torch.Size([batch_size, H, W])).to_dense()
    # else:
    xy_grid = torch.sparse.FloatTensor(ibool, vals, torch.Size([H, W])).to_dense()
    xy_grid = xy_grid.type(torch.FloatTensor)

    return xy_grid

def batch_xyz_to_all_label(xyz_np,setup_params):
    
    upsampling_factor = setup_params['upsampling_factor']
    H, W, D = setup_params['H'], setup_params['W'], setup_params['D']

    batch_size, num_particles = xyz_np[:,:,1].shape

    # project xyz locations on the grid and shift xy to the upper left corner
    xg_floor = (np.floor((xyz_np[:, :, 0] + W/2)*upsampling_factor)).astype('int')
    yg_floor = (np.floor((xyz_np[:, :, 1] + H/2)*upsampling_factor)).astype('int')
    # xg_ceil = (np.ceil((xyz_np[:, :, 0] + W/2)*upsampling_factor)).astype('int')
    # yg_ceil = (np.ceil((xyz_np[:, :, 1] + H/2)*upsampling_factor)).astype('int')
    # indices for sparse tensor
    indX_floor, indY_floor = (xg_floor.flatten('F')).tolist(), (yg_floor.flatten('F')).tolist()
    # indX_ceil, indY_ceil = (xg_ceil.flatten('F')).tolist(), (yg_ceil.flatten('F')).tolist()
    # indX, indY = indX_floor + indX_ceil, indY_floor + indY_ceil
    indX, indY = indX_floor, indY_floor

    # if sampling a batch add a sample index
    if batch_size > 1:
        indS = (np.kron(np.ones(num_particles), np.arange(0, batch_size, 1)).astype('int')).tolist()
        ibool = torch.LongTensor([indS, indY, indX])
    else:
        ibool = torch.LongTensor([indY, indX])

    # spikes for sparse tensor
    vals = torch.ones(batch_size*num_particles)
    # vals = torch.ones(batch_size*num_particles)*xyz_np[:,:,3].squeeze()

    H, W = int(H * upsampling_factor), int(W * upsampling_factor)
    # resulting 3D boolean tensor
    # if batch_size > 1:
        # boolean_grid = torch.sparse.FloatTensor(ibool, vals, torch.Size([batch_size, H, W])).to_dense()
    # else:
    xy_grid = torch.sparse.FloatTensor(ibool, vals, torch.Size([H, W])).to_dense()
    xy_grid = xy_grid.type(torch.FloatTensor)

    vals_depth = torch.ones(batch_size*num_particles)*xyz_np[:,:,2].squeeze()
    vals_flux = torch.ones(batch_size*num_particles)*xyz_np[:,:,3].squeeze()

    zeta_grid = torch.sparse.FloatTensor(ibool, vals_depth, torch.Size([H, W])).to_dense()
    zeta_grid = xy_grid.type(torch.FloatTensor)

    flux_grid = torch.sparse.FloatTensor(ibool, vals_flux, torch.Size([H, W])).to_dense()
    flux_grid = xy_grid.type(torch.FloatTensor)

    xy_grid = xy_grid.unsqueeze(0)
    zeta_grid = zeta_grid.unsqueeze(0)
    flux_grid = flux_grid.unsqueeze(0)
    label_grid = torch.cat([xy_grid,zeta_grid,flux_grid],dim=0)

    return label_grid


def dataloader_xy(path_train, labels, params, setup_params, opt, num_workers=0):
    
    dataset = ImagesDataset_xy(path_train, params['partition'], labels, setup_params)
    batch_size = params['batch_size']
    shuffle = params['shuffle']

    try:
        Sampler = Disample(dataset,num_replicas=opt.world_size,rank=opt.rank,shuffle=shuffle)
        dl = DataLoader(dataset,batch_size=batch_size,sampler=Sampler,num_workers=num_workers,pin_memory=True)
    except:
        dl = DataLoader(dataset,batch_size=batch_size,shuffle=shuffle,num_workers=num_workers,pin_memory=True)
    return dl

# PSF images with corresponding xy labels
class ImagesDataset_xy(Dataset):

    # initialization of the dataset
    def __init__(self, root_dir, list_IDs, labels, setup_params):
        self.root_dir = root_dir
        self.list_IDs = list_IDs
        self.labels = labels
        self.setup_params = setup_params
        # self.train_stats = setup_params['train_stats']

    # total number of samples in the dataset
    def __len__(self):
        return len(self.list_IDs)

    # sampling one example from the data
    def __getitem__(self, index):

        # select sample
        ID = self.list_IDs[index]

        im_name = self.root_dir + '/im' + ID + '.mat'
        im_mat = scipy.io.loadmat(im_name)
        im_np = np.float32(im_mat['g'])
        im_np = np.repeat(np.expand_dims(im_np,0),3,axis=0)

        # turn image into torch tensor with 1 channel
        # im_np = np.expand_dims(im_np, 0)
        im_tensor = torch.from_numpy(im_np)

        # corresponding xyz labels turned to a boolean tensor
        xyz_np = self.labels[ID]
        label_image = batch_xyz_to_2d_image_label(xyz_np, self.setup_params)

        return im_tensor, label_image, ID

