# RNN different network architecture

def buildModel(setup_params):
    if setup_params['model_use'] == 'cnn':
        model = LocalizationCNN(setup_params)
    elif setup_params['model_use'] == 'rnn':
        model = RNN_CNN_v4(setup_params,3,3)   # setup_params,num_recurrent_layers
        # model = RNN_CNN_v5(setup_params)
    elif setup_params['model_use'] == 'cnn_no_dilate':
        # 0721 cnn without dilated conv
        model = LocalizationCNN_no_dilate(setup_params)

    return model


# 0706 add recurrent layer based on [1]
class RNN_CNN(nn.Module):
    def __init__(self, setup_params,num_recurrent_layers):
        super(RNN_CNN, self).__init__()
        self.num_recurrent_layers = num_recurrent_layers
        self.norm = nn.BatchNorm2d(num_features=1, affine=True)
        self.layer1 = Conv2DLeakyReLUBN(1, 64, 3, 1, 1, 0.2)
        self.layer2 = Conv2DLeakyReLUBN(64 + 1, 64, 3, 1, 1, 0.2)
        self.layer3 = Conv2DLeakyReLUBN(64, 64, 3, 1, 1, 0.2) # recurrent layer
        self.layer4 = Conv2DLeakyReLUBN(64, 64, 3, 1, 1, 0.2) 

        self.deconv1 = Conv2DLeakyReLUBN(64 + 1, 64, 3, 1, 1, 0.2)
        self.layer7 = Conv2DLeakyReLUBN(64, setup_params['D'], 3, 1, 1, 0.2)
        self.layer8 = Conv2DLeakyReLUBN(setup_params['D'], setup_params['D'], 3, 1, 1, 0.2)
        self.layer9 = Conv2DLeakyReLUBN(setup_params['D'], setup_params['D'], 3, 1, 1, 0.2)
        self.layer10 = nn.Conv2d(setup_params['D'], setup_params['D'], kernel_size=1, dilation=1)
        self.pred = nn.Hardtanh(min_val=0.0, max_val=setup_params['scaling_factor'])
        self.dropout = nn.Dropout(p=0.5)

    def forward(self, im):  # [4,1,96,96]

        # extract multi-scale features
        im = self.norm(im) # [4,1,96,96]
        out1 = self.layer1(im)  # [4,64,96,96]
        features = torch.cat((out1, im), 1) # [4,65,96,96]
        out = self.layer2(features) + out1  # [4,64,96,96] -> +out = [4,64,96,96]

        N,C,H,W = out.shape
        out_recurrent = torch.zeros(self.num_recurrent_layers,N,C,H,W,device=out.device)
        for idx in range(self.num_recurrent_layers):
            out = self.layer3(out)
            out_idx = out + out1
            out_idx = self.layer4(out_idx)
            out_recurrent[idx,:] = out_idx
        out = torch.mean(out_recurrent,0)  #[4,64,96,96]

        # upsample by 2 in xy
        features = torch.cat((out, im), 1)
        out = interpolate(features, scale_factor=2)
        out = self.deconv1(out)

        # refine z and exact xy
        out = self.layer7(out)
        out = self.layer8(out) + out
        out = self.layer9(out) + out

        # 1x1 conv and hardtanh for final result
        out = self.layer10(out)
        out = self.pred(out)
        return out

class RNN_CNN_v2(nn.Module):
    def __init__(self, setup_params,num_recurrent_layers):
        super(RNN_CNN_v2, self).__init__()
        self.num_recurrent_layers = num_recurrent_layers
        self.norm = nn.BatchNorm2d(num_features=1, affine=True)
        self.layer1 = Conv2DLeakyReLUBN(1, 64, 3, 1, 1, 0.2)
        self.deconv1 = Conv2DLeakyReLUBN(64 + 1, 64, 3, 1, 1, 0.2)
        self.layer7 = Conv2DLeakyReLUBN(64, setup_params['D'], 3, 1, 1, 0.2)

        self.layer3 = Conv2DLeakyReLUBN(setup_params['D'], setup_params['D'], 3, 1, 1, 0.2) # recurrent layer
        self.layer4 = Conv2DLeakyReLUBN(setup_params['D'], setup_params['D'], 3, 1, 1, 0.2) 

        self.layer10 = nn.Conv2d(setup_params['D'], setup_params['D'], kernel_size=1, dilation=1)
        self.pred = nn.Hardtanh(min_val=0.0, max_val=setup_params['scaling_factor'])

    def forward(self, im):  # [4,1,96,96]

        # extract multi-scale features
        im = self.norm(im) # [4,1,96,96]
        out = self.layer1(im)  # [4,64,96,96]
        features = torch.cat((out, im), 1) # [4,65,96,96]

        # upsample by 2 in xy
        out = interpolate(features, scale_factor=2)
        out = self.deconv1(out)
        out1 = self.layer7(out)
        out = out1

        N,C,H,W = out.shape
        out_recurrent = torch.zeros(self.num_recurrent_layers,N,C,H,W,device=out.device)
        for idx in range(self.num_recurrent_layers):
            out = self.layer3(out)
            out_idx = out + out1
            out_idx = self.layer4(out_idx)
            out = self.layer10(out)
            out_recurrent[idx,:] = out_idx

        out = torch.mean(out_recurrent,0)  #[4,D,96,96]
        out = self.pred(out)
        return out

class RNN_CNN_v3(nn.Module):
    def __init__(self, setup_params,num_recurrent_layers1,num_recurrent_layers2):
        super(RNN_CNN_v3, self).__init__()
        self.num_recurrent_layers1 = num_recurrent_layers1
        self.num_recurrent_layers2 = num_recurrent_layers2
        self.norm = nn.BatchNorm2d(num_features=1, affine=True)
        self.layer1 = Conv2DLeakyReLUBN(1, 64, 3, 1, 1, 0.2)
        self.layer2 = Conv2DLeakyReLUBN(64 + 1, 64, 3, 1, 1, 0.2)
        self.layer3 = Conv2DLeakyReLUBN(64, 64, 3, 1, 1, 0.2) # recurrent layer
        self.layer4 = Conv2DLeakyReLUBN(64+64, 64, 3, 1, 1, 0.2) 

        self.deconv1 = Conv2DLeakyReLUBN(64, 64, 3, 1, 1, 0.2)
        self.layer7 = Conv2DLeakyReLUBN(64, setup_params['D'], 3, 1, 1, 0.2)
        self.layer8 = Conv2DLeakyReLUBN(setup_params['D'], setup_params['D'], 3, 1, 1, 0.2)
        self.layer10 = nn.Conv2d(setup_params['D'], setup_params['D'], kernel_size=1, dilation=1)
        self.pred = nn.Hardtanh(min_val=0.0, max_val=setup_params['scaling_factor'])

    def forward(self, im):  # [4,1,96,96]

        # extract multi-scale features
        im = self.norm(im) # [4,1,96,96]
        out1 = self.layer1(im)  # [4,64,96,96]
        features = torch.cat((out1, im), 1) # [4,65,96,96]
        out = self.layer2(features) + out1  # [4,64,96,96] -> +out = [4,64,96,96]

        N,C,H,W = out.shape
        out_recurrent = torch.zeros(self.num_recurrent_layers1,N,C,H,W,device=out.device)
        for idx in range(self.num_recurrent_layers1):
            out = self.layer3(out) + out
            out_recurrent[idx,:] = self.layer4(torch.cat((out, out1),1))
        out = torch.mean(out_recurrent,0)  #[4,64,96,96]

        # upsample by 2 in xy
        out = interpolate(out, scale_factor=2)
        out = self.deconv1(out)  #[4,64,192,192]

        # refine z and exact xy
        out = self.layer7(out)
        N,C,H,W = out.shape
        out_recurrent = torch.zeros(self.num_recurrent_layers2,N,C,H,W,device=out.device)
        for idx in range(self.num_recurrent_layers2):
            out = self.layer8(out) + out
            out_recurrent[idx,:] = self.layer10(out)
        out = torch.mean(out_recurrent,0)  #[4,64,96,96]

        # 1x1 conv and hardtanh for final result
        out = self.pred(out)
        return out

class RNN_CNN_v4(nn.Module):
    def __init__(self, setup_params,num_recurrent_layers1,num_recurrent_layers2):
        super(RNN_CNN_v4, self).__init__()
        self.num_recurrent_layers1 = num_recurrent_layers1
        self.num_recurrent_layers2 = num_recurrent_layers2
        self.norm = nn.BatchNorm2d(num_features=1, affine=True)
        self.layer1 = Conv2DLeakyReLUBN(1, 64, 3, 1, 1, 0.2)
        self.layer2 = Conv2DLeakyReLUBN(64 + 1, 64, 3, 1, 1, 0.2)
        self.layer3 = Conv2DLeakyReLUBN(64, 64, 3, 1, 1, 0.2) # recurrent layer
        self.layer4 = Conv2DLeakyReLUBN(64+64, 64, 3, 1, 1, 0.2) 
        self.dropout = nn.Dropout(p=0.5)

        self.deconv1 = Conv2DLeakyReLUBN(64, 64, 3, 1, 1, 0.2)
        self.layer7 = Conv2DLeakyReLUBN(64, setup_params['D'], 3, 1, 1, 0.2)
        self.layer8 = Conv2DLeakyReLUBN(setup_params['D'], setup_params['D'], 3, 1, 1, 0.2)
        self.layer10 = nn.Conv2d(setup_params['D'], setup_params['D'], kernel_size=1, dilation=1)
        self.pred = nn.Hardtanh(min_val=0.0, max_val=setup_params['scaling_factor'])

    def forward(self, im):  # [4,1,96,96]

        # extract multi-scale features
        im = self.norm(im) # [4,1,96,96]
        out1 = self.layer1(im)  # [4,64,96,96]
        features = torch.cat((out1, im), 1) # [4,65,96,96]
        out = self.layer2(features) + out1  # [4,64,96,96] -> +out = [4,64,96,96]

        N,C,H,W = out.shape
        out_recurrent = torch.zeros(self.num_recurrent_layers1,N,C,H,W,device=out.device)
        for idx in range(self.num_recurrent_layers1):
            ### main difference with v3
            out = self.layer3(out)
            out = self.dropout(out)
            out_recurrent[idx,:] = self.layer4(torch.cat((out, out1),1))
        out = torch.mean(out_recurrent,0)  #[4,64,96,96]

        # upsample by 2 in xy
        out = interpolate(out, scale_factor=2)
        out = self.deconv1(out)  #[4,64,192,192]

        # refine z and exact xy
        out = self.layer7(out)
        N,C,H,W = out.shape
        out_recurrent = torch.zeros(self.num_recurrent_layers2,N,C,H,W,device=out.device)
        for idx in range(self.num_recurrent_layers2):
            out = self.layer8(out)
            out = self.dropout(out)
            out_recurrent[idx,:] = self.layer10(out)
        out = torch.mean(out_recurrent,0)  #[4,64,96,96]

        # 1x1 conv and hardtanh for final result
        out = self.pred(out)
        return out

class RNN_CNN_v5(nn.Module):
    # not use loop
    def __init__(self, setup_params):
        super(RNN_CNN_v5, self).__init__()
        self.num_recurrent_layers1 = 4
        self.num_recurrent_layers2 = 2
        self.norm = nn.BatchNorm2d(num_features=1, affine=True)
        self.layer1 = Conv2DLeakyReLUBN(1, 64, 3, 1, 1, 0.2)
        self.layer2 = Conv2DLeakyReLUBN(64 + 1, 64, 3, 1, 1, 0.2)
        self.layer3 = Conv2DLeakyReLUBN(64, 64, 3, 1, 1, 0.2) # recurrent layer
        self.layer4 = Conv2DLeakyReLUBN(64+64, 64, 3, 1, 1, 0.2) 
        # self.dropout = nn.Dropout(p=0.5)

        self.deconv1 = Conv2DLeakyReLUBN(64, 64, 3, 1, 1, 0.2)
        self.layer5 = Conv2DLeakyReLUBN(64, setup_params['D'], 3, 1, 1, 0.2)
        self.layer6 = Conv2DLeakyReLUBN(setup_params['D'], setup_params['D'], 3, 1, 1, 0.2)
        self.layer7 = nn.Conv2d(setup_params['D'], setup_params['D'], kernel_size=1, dilation=1)
        self.pred = nn.Hardtanh(min_val=0.0, max_val=setup_params['scaling_factor'])

    def forward(self, im):  # [4,1,96,96]

        # extract multi-scale features
        im = self.norm(im) # [4,1,96,96]
        out1 = self.layer1(im)  # [4,64,96,96]
        features = torch.cat((out1, im), 1) # [4,65,96,96]
        out = self.layer2(features) + out1  # [4,64,96,96] -> +out = [4,64,96,96]

        N,C,H,W = out.shape
        out_recurrent = torch.zeros(self.num_recurrent_layers1,N,C,H,W,device=out.device)

        out = self.layer3(out)
        out_recurrent[0,:] = self.layer4(torch.cat((out, out1),1))

        out = self.layer3(out)
        out_recurrent[1,:] = self.layer4(torch.cat((out, out1),1))

        out = self.layer3(out)
        out_recurrent[2,:] = self.layer4(torch.cat((out, out1),1))

        out = self.layer3(out)
        out_recurrent[3,:] = self.layer4(torch.cat((out, out1),1))

        out = torch.mean(out_recurrent,0)  #[4,64,96,96]

        # upsample by 2 in xy
        out = interpolate(out, scale_factor=2)
        out = self.deconv1(out)  #[4,64,192,192]

        # refine z and exact xy
        out = self.layer5(out)

        N,C,H,W = out.shape
        out_recurrent = torch.zeros(self.num_recurrent_layers2,N,C,H,W,device=out.device)

        out = self.layer6(out)
        out = self.pred(out)
        out_recurrent[0,:] = self.layer7(out)

        out = self.layer6(out)
        out = self.pred(out)
        out_recurrent[1,:] = self.layer7(out)

        out = torch.mean(out_recurrent,0)  #[4,64,96,96]

        return out


# [1] Li, J., et al. (2021). "Spatial and temporal super-resolution for fluorescence microscopy by a recurrent neural network." Optics Express 29(10): 15747-15763.