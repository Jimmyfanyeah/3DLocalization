# Import modules and libraries
import torch
from torch.utils.data import Dataset
import numpy as np
import scipy.io
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler as Disample
import os


# converts continuous xyz locations to a boolean grid
def batch_xyz_to_boolean_grid(xyz_np, setup_params):

    upsampling_factor = setup_params['upsampling_factor']
    pixel_size_axial = setup_params['pixel_size_axial']

    # current dimensions
    H, W, D = setup_params['H'], setup_params['W'], setup_params['D']

    # shift the z axis back to 0
    zshift = xyz_np[:,:,2] - setup_params['zmin']
    batch_size, num_particles = zshift.shape

    # project xyz locations on the grid and shift xy to the upper left corner
    xg = (np.round((xyz_np[:, :, 0] + W/2)*upsampling_factor)).astype('int')
    yg = (np.round((xyz_np[:, :, 1] + H/2)*upsampling_factor)).astype('int')
    zg = (np.floor(zshift/pixel_size_axial)).astype('int')

    # indices for sparse tensor
    indX, indY, indZ = (xg.flatten('F')).tolist(), (yg.flatten('F')).tolist(), (zg.flatten('F')).tolist()

    # update dimensions
    H, W = int(H * upsampling_factor), int(W * upsampling_factor)

    # if sampling a batch add a sample index
    if batch_size > 1:
        indS = (np.kron(np.ones(num_particles), np.arange(0, batch_size, 1)).astype('int')).tolist()
        ibool = torch.LongTensor([indS, indZ, indY, indX])
    else:
        ibool = torch.LongTensor([indZ, indY, indX])

    # spikes for sparse tensor
    # vals = torch.ones(batch_size*num_particles)
    vals = torch.ones(batch_size*num_particles)*xyz_np[:,:,3].squeeze()

    # resulting 3D boolean tensor
    if batch_size > 1:
        boolean_grid = torch.sparse.FloatTensor(ibool, vals, torch.Size([batch_size, D, H, W])).to_dense()
    else:
        boolean_grid = torch.sparse.FloatTensor(ibool, vals, torch.Size([D, H, W])).to_dense()
        boolean_grid = boolean_grid.type(torch.FloatTensor)

    return boolean_grid


def batch_xyz_to_boolean_grid_v0(xyz_np, setup_params):
    # without upsample and D=21

    # current dimensions
    H, W, D = setup_params['H'], setup_params['W'], setup_params['D']

    # shift the z axis back to 0
    zshift = xyz_np[:,:,2] # - setup_params['zmin']
    batch_size, num_particles = zshift.shape

    # project xyz locations on the grid and shift xy to the upper left corner
    xg = (np.round((xyz_np[:, :, 0] + W/2))).astype('int')
    yg = (np.round((xyz_np[:, :, 1] + H/2))).astype('int')
    zg = (np.floor((zshift+21)/2.1)+1).astype('int')

    # indices for sparse tensor
    indX, indY, indZ = (xg.flatten('F')).tolist(), (yg.flatten('F')).tolist(), (zg.flatten('F')).tolist()

    # if sampling a batch add a sample index
    if batch_size > 1:
        indS = (np.kron(np.ones(num_particles), np.arange(0, batch_size, 1)).astype('int')).tolist()
        ibool = torch.LongTensor([indS, indZ, indY, indX])
    else:
        ibool = torch.LongTensor([indZ, indY, indX])

    # spikes for sparse tensor
    # vals = torch.ones(batch_size*num_particles)
    vals = torch.ones(batch_size*num_particles)*xyz_np[:,:,3].squeeze()

    # resulting 3D boolean tensor
    if batch_size > 1:
        boolean_grid = torch.sparse.FloatTensor(ibool, vals, torch.Size([batch_size, D, H, W])).to_dense()
    else:
        boolean_grid = torch.sparse.FloatTensor(ibool, vals, torch.Size([D, H, W])).to_dense()
        boolean_grid = boolean_grid.type(torch.FloatTensor)

    return boolean_grid


def dataloader(path_train, labels, params, setup_params, opt, num_workers=0):

    if opt.train_or_test == 'train':
        # dataset = ImagesDataset(path_train, params['partition'], labels, setup_params)
        dataset = ImagesDataset_v2(path_train, params['partition'], labels, setup_params)
    else:
        dataset = ImagesDataset_test(path_train, params['partition'], labels, setup_params)

    batch_size = params['batch_size']
    shuffle = params['shuffle']

    try:
        Sampler = Disample(dataset,num_replicas=opt.world_size,rank=opt.rank,shuffle=shuffle)
        dl = DataLoader(dataset,batch_size=batch_size,sampler=Sampler,num_workers=num_workers,pin_memory=True)
    except:
        dl = DataLoader(dataset,batch_size=batch_size,shuffle=shuffle,num_workers=num_workers,pin_memory=True)
    return dl


# PSF images with corresponding xyz labels dataset
class ImagesDataset(Dataset):

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

        im_name = os.path.join(self.root_dir,'train','/im' + ID + '.mat')
        im_mat = scipy.io.loadmat(im_name)
        im_np = np.float32(im_mat['g'])

        # turn image into torch tensor with 1 channel
        im_np = np.expand_dims(im_np, 0)
        im_tensor = torch.from_numpy(im_np)

        # corresponding xyz labels turned to a boolean tensor
        xyz_np = self.labels[ID]
        bool_grid = batch_xyz_to_boolean_grid(xyz_np, self.setup_params)

        return im_tensor, bool_grid, ID


class ImagesDataset_v2(Dataset):
    # v2 with noiseless 2d image
    # initialization of the dataset
    def __init__(self, root_dir, list_IDs, labels, setup_params):
        self.root_dir = root_dir
        self.list_IDs = list_IDs
        self.labels = labels
        self.setup_params = setup_params

    def __len__(self):
        return len(self.list_IDs)

    def __getitem__(self, index):
        ID = self.list_IDs[index]

        im_name = os.path.join(self.root_dir,'train','im' + ID)# + '.mat')
        im_mat = scipy.io.loadmat(im_name,appendmat=True)
        im_np = np.float32(im_mat['g'])

        # turn image into torch tensor with 1 channel
        im_tensor = torch.from_numpy(im_np).unsqueeze(0)

        # corresponding xyz labels turned to a boolean tensor
        xyz_np = self.labels[ID]
        bool_grid = batch_xyz_to_boolean_grid(xyz_np, self.setup_params)

        # target 2d image without noise
        gtimg_name = os.path.join(self.root_dir,'clean','I' + ID + '.mat')
        gtimg_mat = scipy.io.loadmat(gtimg_name)
        gtimg_np = np.float32(gtimg_mat['I0'])
        gtimg_tensor = torch.from_numpy(gtimg_np)

        return im_tensor, bool_grid, gtimg_tensor, ID


class ImagesDataset_test(Dataset):
    def __init__(self, root_dir, list_IDs, labels, setup_params):
        self.root_dir = root_dir
        self.list_IDs = list_IDs
        self.labels = labels
        self.setup_params = setup_params

    def __len__(self):
        return len(self.list_IDs)

    def __getitem__(self, index):

        ID = self.list_IDs[index]

        im_name = os.path.join(self.root_dir,'im'+ID+'.mat')
        im_mat = scipy.io.loadmat(im_name)
        im_np = np.float32(im_mat['g'])

        # turn image into torch tensor with 1 channel
        # im_np = np.expand_dims(im_np, 0)
        im_tensor = torch.from_numpy(im_np).unsqueeze(0)

        # corresponding xyz labels turned to a boolean tensor
        xyz_np = self.labels[ID]
        bool_grid = batch_xyz_to_boolean_grid(xyz_np, self.setup_params)

        # target 2d image without noise
        gtimg_name = os.path.join(self.root_dir, 'I'+ID+'.mat')
        gtimg_mat = scipy.io.loadmat(gtimg_name)
        gtimg_np = np.float32(gtimg_mat['I0'])
        gtimg_tensor = torch.from_numpy(gtimg_np)

        return im_tensor, bool_grid, gtimg_tensor, ID
