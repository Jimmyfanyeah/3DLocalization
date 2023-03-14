# Import modules and libraries
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



'''
When using UNet3D, output = 
--> For data
im_np = np.repeat(np.expand_dims(im_np,0),self.setup_params['D'] ,axis=0) #[D,96,96] -> turn image into torch tensor with 1 channel
im_np = np.expand_dims(im_np, 0)
im_tensor = torch.from_numpy(im_np).unsqueeze(0)  # [1,D,96,96] batch size

# corresponding xyz labels turned to a boolean tensor
xyz_np = self.labels[ID]
labels = gen_label(xyz_np, self.setup_params)

--> For output
output_xy, output_z, output_flux = cnn(inputs)
loss = calc_loss(output_xy, output_z, output_flux, targets, metric, metrics)

--> For loss
target_xy = target[:,0,:,:].unsqueeze(1) # location in xy plane
target_z = target[:,1,:,:].unsqueeze(1) # location in axial plane
loss_xy = F.binary_cross_entropy_with_logits(pred_xy, target_xy, pos_weight=torch.tensor(800,device=pred_flux.device))
loss_z = nn.MSELoss()(torch.sigmoid(pred_xy)*pred_z, target_z)
loss_flux = nn.MSELoss()(torch.sigmoid(pred_xy)*pred_flux, target_flux)
'''