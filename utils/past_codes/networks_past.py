# Import modules and libraries
import torch
import torch.nn as nn
from torch.nn.functional import interpolate
import math


################################################################################################
# CNN_HDC: 2 versions
# 0810 new dilation strategy 1,[1,2,5,9,17]*2
class ResLocalizationCNN_HDC_v2(nn.Module):
    def __init__(self, setup_params):
        super(ResLocalizationCNN_HDC_v2, self).__init__()
        self.norm = nn.BatchNorm2d(num_features=1, affine=True)
        self.layer1 = Conv2DLeakyReLUBN(1, 64, 3, 1, 1, 0.2)
        self.layer2 = ResConv2DLeakyReLUBN(64, 64, 3, 1, 1, 0.2)

        self.layer3 = ResConv2DLeakyReLUBN(64, 64, 3, (2,2), (2,2), 0.2)
        self.layer4 = ResConv2DLeakyReLUBN(64, 64, 3, (5,5), (5,5), 0.2)
        self.layer5 = ResConv2DLeakyReLUBN(64, 64, 3, (9,9), (9,9), 0.2)
        self.layer6 = ResConv2DLeakyReLUBN(64, 64, 3, (17,17), (17,17), 0.2)

        self.deconv1 = ResConv2DLeakyReLUBN(64, 64, 3, 1, 1, 0.2)
        self.deconv2 = ResConv2DLeakyReLUBN(64, 64, 3, (2,2), (2,2), 0.2)
        self.layer7 = Conv2DLeakyReLUBN(64, setup_params['D'], 3, (5,5), (5,5), 0.2)
        self.layer8 = ResConv2DLeakyReLUBN(setup_params['D'], setup_params['D'], 3, (9,9), (9,9), 0.2)
        self.layer9 = ResConv2DLeakyReLUBN(setup_params['D'], setup_params['D'], 3, (17,17), (17,17), 0.2)
        self.layer10 = nn.Conv2d(setup_params['D'], setup_params['D'], kernel_size=1, dilation=1)
        self.pred = nn.Hardtanh(min_val=0.0, max_val=setup_params['scaling_factor'])
        self.dropout = nn.Dropout(p=0.5)

    def forward(self, im):  # [4,1,96,96]

        # extract multi-scale features
        im = self.norm(im) # [4,1,96,96]
        out = self.layer1(im)  # [4,64,96,96]
        out = self.layer2(out)  # [4,64,96,96] -> +out = [4,64,96,96]
        out = self.layer3(out)  # [4,64,96,96] -> +out = [4,64,96,96]
        out = self.dropout(out)
        out = self.layer4(out)
        out = self.layer5(out)
        out = self.dropout(out)
        out = self.layer6(out)

        # upsample by 2 in xy
        out = interpolate(out, scale_factor=2)
        out = self.deconv1(out)
        out = self.deconv2(out)

        # refine z and exact xy
        out = self.layer7(out)
        out = self.layer8(out)
        out = self.layer9(out)

        # 1x1 conv and hardtanh for final result
        out = self.layer10(out)
        out = self.pred(out)
        return out

# 0809 add hdc, try different dilation strategy [1,1,2,4,8,16] -> [1,2,5,1,2,5]
# 0810 new dilation strategy
class ResLocalizationCNN_HDC(nn.Module):
    def __init__(self, setup_params):
        super(ResLocalizationCNN_HDC, self).__init__()
        self.norm = nn.BatchNorm2d(num_features=1, affine=True)
        self.layer1 = Conv2DLeakyReLUBN(1, 64, 3, 1, 1, 0.2)
        self.layer2 = ResConvLeakyReLUBN(64, 64, 3, 1, 1, 0.2)

        self.layer3 = ResConvLeakyReLUBN(64, 64, 3, (2,2), (2,2), 0.2)
        self.layer4 = ResConvLeakyReLUBN(64, 64, 3, (5,5), (5,5), 0.2)
        self.layer5 = ResConvLeakyReLUBN(64, 64, 3, (9,9), (9,9), 0.2)
        self.layer6 = ResConvLeakyReLUBN(64, 64, 3, (17,17), (17,17), 0.2)

        self.deconv1 = ResConvLeakyReLUBN(64, 64, 3, 1, 1, 0.2)
        self.deconv2 = ResConvLeakyReLUBN(64, 64, 3, 1, 1, 0.2)
        self.layer7 = Conv2DLeakyReLUBN(64, setup_params['D'], 3, 1, 1, 0.2)
        self.layer8 = ResConvLeakyReLUBN(setup_params['D'], setup_params['D'], 3,1, 1, 0.2)
        self.layer9 = ResConvLeakyReLUBN(setup_params['D'], setup_params['D'], 3, 1, 1, 0.2)
        self.layer10 = nn.Conv2d(setup_params['D'], setup_params['D'], kernel_size=1, dilation=1)
        self.pred = nn.Hardtanh(min_val=0.0, max_val=setup_params['scaling_factor'])
        self.dropout = nn.Dropout(p=0.5)

    def forward(self, im):  # [4,1,96,96]

        # extract multi-scale features
        im = self.norm(im) # [4,1,96,96]
        out = self.layer1(im)  # [4,64,96,96]
        out = self.layer2(out)  # [4,64,96,96] -> +out = [4,64,96,96]
        out = self.layer3(out)  # [4,64,96,96] -> +out = [4,64,96,96]
        out = self.dropout(out)
        out = self.layer4(out)
        out = self.layer5(out)
        out = self.dropout(out)
        out = self.layer6(out)

        # upsample by 2 in xy
        out = interpolate(out, scale_factor=2)
        out = self.deconv1(out)
        out = self.deconv2(out)

        # refine z and exact xy
        out = self.layer7(out)
        out = self.layer8(out)
        out = self.layer9(out)

        # 1x1 conv and hardtanh for final result
        out = self.layer10(out)
        out = self.pred(out)
        return out


################################################################################################
# CNN_DUC - 3 versions
#### basic blocks
class ResConv2DLeakyReLUBN(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, padding, dilation, negative_slope):
        super(ResConv2DLeakyReLUBN, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, 1, padding, dilation)
        self.lrelu = nn.LeakyReLU(negative_slope, inplace=True)
        self.bn = nn.BatchNorm2d(out_channels)
        self.in_channels = in_channels
        self.out_channels = out_channels

    def forward(self, x):
        residual = x
        out = self.conv(x)
        out = self.lrelu(out)
        out = self.bn(out)

        # if self.in_channels == self.out_channels:
        out += residual

        return out

class DUC_plain(nn.Module):
    def __init__(self, in_channels, kernel_size, negative_slope, upsampling_factor=2):
        super(DUC_plain,self).__init__()
        self.upsampling_factor = upsampling_factor

        out_channels = in_channels*upsampling_factor*upsampling_factor
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride=1)
        # self.lrelu = nn.LeakyReLU(negative_slope, inplace=True)
        # self.bn = nn.BatchNorm2d(in_channels)

    def forward(self,x):

        out = self.conv(x)

        n,c,h,w = x.shape
        h,w = h*self.upsampling_factor, w*self.upsampling_factor
        out = torch.reshape(out,[n,-1,h,w])

        # out = self.lrelu(out)
        # out = self.bn(out)

        return out

class DUC_aspp(nn.Module):
    def __init__(self, in_channels, kernel_size, negative_slope, upsampling_factor=2, aspp_num=4, aspp_stride=6):
        super(DUC_aspp,self).__init__()
        self.upsampling_factor = upsampling_factor

        out_channels = in_channels*upsampling_factor*upsampling_factor
        pad = []
        for i in range(aspp_num):
            pad.append(((i + 1) * aspp_stride, (i + 1) * aspp_stride))

        self.aspp1 = nn.Conv2d(in_channels, out_channels, kernel_size, stride=1, padding=pad[0] , dilation=pad[0])
        self.aspp2 = nn.Conv2d(in_channels, out_channels, kernel_size, stride=1, padding=pad[1] , dilation=pad[1])
        self.aspp3 = nn.Conv2d(in_channels, out_channels, kernel_size, stride=1, padding=pad[2] , dilation=pad[2])
        self.aspp4 = nn.Conv2d(in_channels, out_channels, kernel_size, stride=1, padding=pad[3] , dilation=pad[3])
        self.lrelu = nn.LeakyReLU(negative_slope, inplace=True)
        self.bn = nn.BatchNorm2d(in_channels)

    def forward(self,x):

        out1 = self.aspp1(x)
        out2 = self.aspp2(x)
        out3 = self.aspp3(x)
        out4 = self.aspp4(x)

        out = (out1 + out2 + out3 + out4)/4

        n,c,h,w = x.shape
        h,w = h*self.upsampling_factor, w*self.upsampling_factor
        out = torch.reshape(out,[n,-1,h,w])

        out = self.lrelu(out)
        out = self.bn(out)

        return out

class DUCLeakyReLUBN(nn.Module):
    def __init__(self, in_channels, kernel_size, negative_slope, upsampling_factor=2):
        super(DUCLeakyReLUBN,self).__init__()
        self.upsampling_factor = upsampling_factor

        out_channels = in_channels*upsampling_factor*upsampling_factor
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride=1, padding=1)
        self.lrelu = nn.LeakyReLU(negative_slope, inplace=True)
        self.bn = nn.BatchNorm2d(in_channels)

    def forward(self,x):

        out = self.conv(x)

        n,c,h,w = x.shape
        h,w = h*self.upsampling_factor, w*self.upsampling_factor
        out = torch.reshape(out,[n,-1,h,w])

        out = self.lrelu(out)
        out = self.bn(out)

        return out

#### netowrks
# 0809 interpolate -> final layer duc
class ResLocalizationCNN_DUC_v2(nn.Module):
    def __init__(self, setup_params):
        super(ResLocalizationCNN_DUC_v2, self).__init__()
        self.norm = nn.BatchNorm2d(num_features=1, affine=True)
        self.layer1 = Conv2DLeakyReLUBN(1, 64, 3, 1, 1, 0.2)
        self.layer2 = ResConv2DLeakyReLUBN(64, 64, 3, 1, 1, 0.2)

        self.layer3 = ResConv2DLeakyReLUBN(64, 64, 3, (2, 2), (2, 2), 0.2)
        self.layer4 = ResConv2DLeakyReLUBN(64, 64, 3, (4, 4), (4, 4), 0.2)
        self.layer5 = ResConv2DLeakyReLUBN(64, 64, 3, (8, 8), (8, 8), 0.2)
        self.layer6 = ResConv2DLeakyReLUBN(64, 64, 3, (16, 16), (16, 16), 0.2)

        self.deconv1 = ResConv2DLeakyReLUBN(64, 64, 3, 1, 1, 0.2)
        self.deconv2 = ResConv2DLeakyReLUBN(64, 64, 3, 1, 1, 0.2)
        self.layer7 = Conv2DLeakyReLUBN(64, setup_params['D'], 3, 1, 1, 0.2)
        self.layer8 = ResConv2DLeakyReLUBN(setup_params['D'], setup_params['D'], 3, 1, 1, 0.2)
        self.layer9 = ResConv2DLeakyReLUBN(setup_params['D'], setup_params['D'], 3, 1, 1, 0.2)
        self.duc = DUCLeakyReLUBN(setup_params['D'], 3, 0.2, upsampling_factor=2)
        self.layer10 = nn.Conv2d(setup_params['D'], setup_params['D'], kernel_size=1, dilation=1)
        self.pred = nn.Hardtanh(min_val=0.0, max_val=setup_params['scaling_factor'])
        self.dropout = nn.Dropout(p=0.5)

    def forward(self, im):  # [4,1,96,96]

        # extract multi-scale features
        im = self.norm(im) # [4,1,96,96]
        out = self.layer1(im)  # [4,64,96,96]
        out = self.layer2(out)  # [4,64,96,96] -> +out = [4,64,96,96]
        out = self.layer3(out)  # [4,64,96,96] -> +out = [4,64,96,96]
        out = self.dropout(out)
        out = self.layer4(out)
        out = self.layer5(out)
        out = self.dropout(out)
        out = self.layer6(out)

        # upsample by 2 in xy
        # out = interpolate(out, scale_factor=2)
        out = self.deconv1(out)
        out = self.deconv2(out)

        # refine z and exact xy
        out = self.layer7(out)
        out = self.layer8(out)
        out = self.layer9(out)

        out = self.duc(out)

        # 1x1 conv and hardtanh for final result
        out = self.layer10(out)
        out = self.pred(out)
        return out

# 0809 add duc only 1 simple conv layer
class ResLocalizationCNN_DUC(nn.Module):
    def __init__(self, setup_params):
        super(ResLocalizationCNN_DUC, self).__init__()
        self.norm = nn.BatchNorm2d(num_features=1, affine=True)
        self.layer1 = Conv2DLeakyReLUBN(1, 64, 3, 1, 1, 0.2)
        self.layer2 = ResConv2DLeakyReLUBN(64, 64, 3, 1, 1, 0.2)

        self.layer3 = ResConv2DLeakyReLUBN(64, 64, 3, (2, 2), (2, 2), 0.2)
        self.layer4 = ResConv2DLeakyReLUBN(64, 64, 3, (4, 4), (4, 4), 0.2)
        self.layer5 = ResConv2DLeakyReLUBN(64, 64, 3, (8, 8), (8, 8), 0.2)
        self.layer6 = ResConv2DLeakyReLUBN(64, 64, 3, (16, 16), (16, 16), 0.2)

        self.duc = DUC_plain(64, 3, 0.2, upsampling_factor=2)
        self.deconv1 = ResConv2DLeakyReLUBN(64, 64, 3, 1, 1, 0.2)
        self.deconv2 = ResConv2DLeakyReLUBN(64, 64, 3, 1, 1, 0.2)
        self.layer7 = Conv2DLeakyReLUBN(64, setup_params['D'], 3, 1, 1, 0.2)
        self.layer8 = ResConv2DLeakyReLUBN(setup_params['D'], setup_params['D'], 3, 1, 1, 0.2)
        self.layer9 = ResConv2DLeakyReLUBN(setup_params['D'], setup_params['D'], 3, 1, 1, 0.2)
        self.layer10 = nn.Conv2d(setup_params['D'], setup_params['D'], kernel_size=1, dilation=1)
        self.pred = nn.Hardtanh(min_val=0.0, max_val=setup_params['scaling_factor'])
        self.dropout = nn.Dropout(p=0.5)

    def forward(self, im):  # [4,1,96,96]

        # extract multi-scale features
        im = self.norm(im) # [4,1,96,96]
        out = self.layer1(im)  # [4,64,96,96]
        out = self.layer2(out)  # [4,64,96,96] -> +out = [4,64,96,96]
        out = self.layer3(out)  # [4,64,96,96] -> +out = [4,64,96,96]
        out = self.dropout(out)
        out = self.layer4(out)
        out = self.layer5(out)
        out = self.dropout(out)
        out = self.layer6(out)

        # upsample by 2 in xy
        # out = interpolate(out, scale_factor=2)
        out = self.duc(out)
        out = self.deconv1(out)
        out = self.deconv2(out)

        # refine z and exact xy
        out = self.layer7(out)
        out = self.layer8(out)
        out = self.layer9(out)

        # 1x1 conv and hardtanh for final result
        out = self.layer10(out)
        out = self.pred(out)
        return out

# 0809 add duc with aspp
class Loc3dResCNN(nn.Module):
    def __init__(self, setup_params):
        super(Loc3dResCNN, self).__init__()
        self.norm = nn.BatchNorm2d(num_features=1, affine=True)
        self.layer1 = ResConvLeakyReLUBN(1, 64, 3, 1, 1, 0.2)
        self.layer2 = ResConvLeakyReLUBN(64, 64, 3, 1, 1, 0.2)

        self.layer3 = ResConvLeakyReLUBN(64, 64, 3, (2, 2), (2, 2), 0.2)
        self.layer4 = ResConvLeakyReLUBN(64, 64, 3, (4, 4), (4, 4), 0.2)
        self.layer5 = ResConvLeakyReLUBN(64, 64, 3, (8, 8), (8, 8), 0.2)
        self.layer6 = ResConvLeakyReLUBN(64, 64, 3, (16, 16), (16, 16), 0.2)

        self.duc = DUC_aspp(64, 3, 0.2, upsampling_factor=setup_params['upsampling_factor'])
        self.deconv1 = ResConvLeakyReLUBN(64, 64, 3, 1, 1, 0.2)
        self.deconv2 = ResConvLeakyReLUBN(64, 64, 3, 1, 1, 0.2)
        self.layer7 = ResConvLeakyReLUBN(64, setup_params['D'], 3, 1, 1, 0.2)
        self.layer8 = ResConvLeakyReLUBN(setup_params['D'], setup_params['D'], 3, 1, 1, 0.2)
        self.layer9 = ResConvLeakyReLUBN(setup_params['D'], setup_params['D'], 3, 1, 1, 0.2)
        self.layer10 = nn.Conv2d(setup_params['D'], setup_params['D'], kernel_size=1, dilation=1)
        self.pred = nn.Hardtanh(min_val=0.0, max_val=setup_params['scaling_factor'])
        self.dropout = nn.Dropout(p=0.5)

    def forward(self, im):  # [N,1,96,96]

        # extract multi-scale features
        im = self.norm(im) # [N,1,96,96]
        out = self.layer1(im)  # [N,64,96,96]
        out = self.layer2(out)  # [N,64,96,96]
        out = self.layer3(out)  # [N,64,96,96]
        out = self.dropout(out)

        out = self.layer4(out)
        out = self.layer5(out)
        out = self.dropout(out)

        out = self.layer6(out)

        # upsample by 2 in xy
        # out = interpolate(out, scale_factor=2)
        out = self.duc(out)
        out = self.deconv1(out)
        out = self.deconv2(out)

        # refine z and exact xy
        out = self.layer7(out)
        out = self.layer8(out)
        out = self.layer9(out)

        # 1x1 conv and hardtanh for final result
        out = self.layer10(out)
        out = self.pred(out)
        return out


################################################################################################
# 0808 remove torch.cat(out, im)
class LocalizationCNN_concatim(nn.Module):
    def __init__(self, setup_params):
        super(LocalizationCNN_concatim, self).__init__()
        self.norm = nn.BatchNorm2d(num_features=1, affine=True)
        self.layer1 = Conv2DLeakyReLUBN(1, 64, 3, 1, 1, 0.2)
        self.layer2 = Conv2DLeakyReLUBN(64, 64, 3, 1, 1, 0.2)

        self.layer3 = Conv2DLeakyReLUBN(64, 64, 3, (2, 2), (2, 2), 0.2)
        self.layer4 = Conv2DLeakyReLUBN(64, 64, 3, (4, 4), (4, 4), 0.2)
        self.layer5 = Conv2DLeakyReLUBN(64, 64, 3, (8, 8), (8, 8), 0.2)
        self.layer6 = Conv2DLeakyReLUBN(64, 64, 3, (16, 16), (16, 16), 0.2)

        self.deconv1 = Conv2DLeakyReLUBN(64, 64, 3, 1, 1, 0.2)
        self.deconv2 = Conv2DLeakyReLUBN(64, 64, 3, 1, 1, 0.2)
        self.layer7 = Conv2DLeakyReLUBN(64, setup_params['D'], 3, 1, 1, 0.2)
        self.layer8 = Conv2DLeakyReLUBN(setup_params['D'], setup_params['D'], 3, 1, 1, 0.2)
        self.layer9 = Conv2DLeakyReLUBN(setup_params['D'], setup_params['D'], 3, 1, 1, 0.2)
        self.layer10 = nn.Conv2d(setup_params['D'], setup_params['D'], kernel_size=1, dilation=1)
        self.pred = nn.Hardtanh(min_val=0.0, max_val=setup_params['scaling_factor'])
        self.dropout = nn.Dropout(p=0.5)

    def forward(self, im):  # [4,1,96,96]

        # extract multi-scale features
        im = self.norm(im) # [4,1,96,96]
        out = self.layer1(im)  # [4,64,96,96]
        out = self.layer2(out) + out  # [4,64,96,96] -> +out = [4,64,96,96]
        out = self.layer3(out) + out  # [4,64,96,96] -> +out = [4,64,96,96]
        out = self.dropout(out)
        out = self.layer4(out) + out
        out = self.layer5(out) + out
        out = self.dropout(out)
        out = self.layer6(out) + out

        out = interpolate(out, scale_factor=2)
        out = self.deconv1(out)
        out = self.deconv2(out)

        # refine z and exact xy
        out = self.layer7(out)
        out = self.layer8(out) + out
        out = self.layer9(out) + out

        # 1x1 conv and hardtanh for final result
        out = self.layer10(out)
        out = self.pred(out)
        return out


################################################################################################
# 0808 leaky ReLU -> ReLU
class Conv2DReLUBN(nn.Module):
    def __init__(self, input_channels, layer_width, kernel_size, padding, dilation, negative_slope):
        super(Conv2DReLUBN, self).__init__()
        self.conv = nn.Conv2d(input_channels, layer_width, kernel_size, 1, padding, dilation)
        self.relu = nn.ReLU(inplace=True)
        self.bn = nn.BatchNorm2d(layer_width)

    def forward(self, x):
        out = self.conv(x)
        out = self.relu(out)
        out = self.bn(out)
        return out

class LocalizationCNN_ReLU(nn.Module):
    def __init__(self, setup_params):
        super(LocalizationCNN_ReLU, self).__init__()
        self.norm = nn.BatchNorm2d(num_features=1, affine=True)
        self.layer1 = Conv2DReLUBN(1, 64, 3, 1, 1, 0.2)
        self.layer2 = Conv2DReLUBN(64 + 1, 64, 3, 1, 1, 0.2)

        self.layer3 = Conv2DReLUBN(64 + 1, 64, 3, (2, 2), (2, 2), 0.2)
        self.layer4 = Conv2DReLUBN(64 + 1, 64, 3, (4, 4), (4, 4), 0.2)
        self.layer5 = Conv2DReLUBN(64 + 1, 64, 3, (8, 8), (8, 8), 0.2)
        self.layer6 = Conv2DReLUBN(64 + 1, 64, 3, (16, 16), (16, 16), 0.2)

        self.deconv1 = Conv2DReLUBN(64 + 1, 64, 3, 1, 1, 0.2)
        self.deconv2 = Conv2DReLUBN(64, 64, 3, 1, 1, 0.2)
        self.layer7 = Conv2DReLUBN(64, setup_params['D'], 3, 1, 1, 0.2)
        self.layer8 = Conv2DReLUBN(setup_params['D'], setup_params['D'], 3, 1, 1, 0.2)
        self.layer9 = Conv2DReLUBN(setup_params['D'], setup_params['D'], 3, 1, 1, 0.2)
        self.layer10 = nn.Conv2d(setup_params['D'], setup_params['D'], kernel_size=1, dilation=1)
        self.pred = nn.Hardtanh(min_val=0.0, max_val=setup_params['scaling_factor'])
        self.dropout = nn.Dropout(p=0.5)

    def forward(self, im):  # [4,1,96,96]

        # extract multi-scale features
        im = self.norm(im) # [4,1,96,96]
        out = self.layer1(im)  # [4,64,96,96]
        features = torch.cat((out, im), 1) # [4,65,96,96]
        out = self.layer2(features) + out  # [4,64,96,96] -> +out = [4,64,96,96]
        features = torch.cat((out, im), 1) # [4,65,96,96]
        out = self.layer3(features) + out  # [4,64,96,96] -> +out = [4,64,96,96]
        out = self.dropout(out)
        features = torch.cat((out, im), 1)
        out = self.layer4(features) + out
        features = torch.cat((out, im), 1)
        out = self.layer5(features) + out
        out = self.dropout(out)
        features = torch.cat((out, im), 1)
        out = self.layer6(features) + out

        # upsample by 4 in xy
        features = torch.cat((out, im), 1)
        out = interpolate(features, scale_factor=2)
        # out = features
        out = self.deconv1(out)
        # out = interpolate(out, scale_factor=2)
        out = self.deconv2(out)

        # refine z and exact xy
        out = self.layer7(out)
        out = self.layer8(out) + out
        out = self.layer9(out) + out

        # 1x1 conv and hardtanh for final result
        out = self.layer10(out)
        out = self.pred(out)
        return out


################################################################################################
# 0721 remove dilated conv layers - change dilation from [1,2,4,8,16] to 1
class LocalizationCNN_no_dilate(nn.Module):
    def __init__(self, setup_params):
        super(LocalizationCNN_no_dilate, self).__init__()
        self.norm = nn.BatchNorm2d(num_features=1, affine=True)
        self.layer1 = Conv2DLeakyReLUBN(1, 64, 3, 1, 1, 0.2)
        self.layer2 = Conv2DLeakyReLUBN(64 + 1, 64, 3, 1, 1, 0.2)

        self.layer3 = Conv2DLeakyReLUBN(64 + 1, 64, 3, 1, 1, 0.2)
        self.layer4 = Conv2DLeakyReLUBN(64 + 1, 64, 3, 1, 1, 0.2)
        self.layer5 = Conv2DLeakyReLUBN(64 + 1, 64, 3, 1, 1, 0.2)
        self.layer6 = Conv2DLeakyReLUBN(64 + 1, 64, 3, 1, 1, 0.2)

        self.deconv1 = Conv2DLeakyReLUBN(64 + 1, 64, 3, 1, 1, 0.2)
        self.deconv2 = Conv2DLeakyReLUBN(64, 64, 3, 1, 1, 0.2)
        self.layer7 = Conv2DLeakyReLUBN(64, setup_params['D'], 3, 1, 1, 0.2)
        self.layer8 = Conv2DLeakyReLUBN(setup_params['D'], setup_params['D'], 3, 1, 1, 0.2)
        self.layer9 = Conv2DLeakyReLUBN(setup_params['D'], setup_params['D'], 3, 1, 1, 0.2)
        self.layer10 = nn.Conv2d(setup_params['D'], setup_params['D'], kernel_size=1, dilation=1)
        self.pred = nn.Hardtanh(min_val=0.0, max_val=setup_params['scaling_factor'])
        self.dropout = nn.Dropout(p=0.5)

    def forward(self, im):  # [4,1,96,96]

        # extract multi-scale features
        im = self.norm(im) # [4,1,96,96]
        out = self.layer1(im)  # [4,64,96,96]
        features = torch.cat((out, im), 1) # [4,65,96,96]
        out = self.layer2(features) + out  # [4,64,96,96] -> +out = [4,64,96,96]
        features = torch.cat((out, im), 1) # [4,65,96,96]
        out = self.layer3(features) + out  # [4,64,96,96] -> +out = [4,64,96,96]
        out = self.dropout(out)
        features = torch.cat((out, im), 1)
        out = self.layer4(features) + out
        features = torch.cat((out, im), 1)
        out = self.layer5(features) + out
        out = self.dropout(out)
        features = torch.cat((out, im), 1)
        out = self.layer6(features) + out

        # upsample by 4 in xy
        features = torch.cat((out, im), 1)
        out = interpolate(features, scale_factor=2)
        # out = features
        out = self.deconv1(out)
        # out = interpolate(out, scale_factor=2)
        out = self.deconv2(out)

        # refine z and exact xy
        out = self.layer7(out)
        out = self.layer8(out) + out
        out = self.layer9(out) + out

        # 1x1 conv and hardtanh for final result
        out = self.layer10(out)
        out = self.pred(out)
        return out


################################################################################################
# Initial deepstorm3d + dropout
class Conv2DLeakyReLUBN(nn.Module):
    def __init__(self, input_channels, layer_width, kernel_size, padding, dilation, negative_slope):
        super(Conv2DLeakyReLUBN, self).__init__()
        self.conv = nn.Conv2d(input_channels, layer_width, kernel_size, 1, padding, dilation)
        self.lrelu = nn.LeakyReLU(negative_slope, inplace=True)
        self.bn = nn.BatchNorm2d(layer_width)

    def forward(self, x):
        out = self.conv(x)
        out = self.lrelu(out)
        out = self.bn(out)
        return out

class LocalizationCNN(nn.Module):
    def __init__(self, setup_params):
        super(LocalizationCNN, self).__init__()
        self.norm = nn.BatchNorm2d(num_features=1, affine=True)
        self.layer1 = Conv2DLeakyReLUBN(1, 64, 3, 1, 1, 0.2)
        self.layer2 = Conv2DLeakyReLUBN(64 + 1, 64, 3, 1, 1, 0.2)
        if setup_params['dilation_flag']:
            self.layer3 = Conv2DLeakyReLUBN(64 + 1, 64, 3, (2, 2), (2, 2), 0.2)
            self.layer4 = Conv2DLeakyReLUBN(64 + 1, 64, 3, (4, 4), (4, 4), 0.2)
            self.layer5 = Conv2DLeakyReLUBN(64 + 1, 64, 3, (8, 8), (8, 8), 0.2)
            self.layer6 = Conv2DLeakyReLUBN(64 + 1, 64, 3, (16, 16), (16, 16), 0.2)
        else:
            self.layer3 = Conv2DLeakyReLUBN(64 + 1, 64, 3, (2, 2), (2, 2), 0.2)
            self.layer4 = Conv2DLeakyReLUBN(64 + 1, 64, 3, (4, 4), (4, 4), 0.2)
            self.layer5 = Conv2DLeakyReLUBN(64 + 1, 64, 3, 1, 1, 0.2)
            self.layer6 = Conv2DLeakyReLUBN(64 + 1, 64, 3, 1, 1, 0.2)
        self.deconv1 = Conv2DLeakyReLUBN(64 + 1, 64, 3, 1, 1, 0.2)
        self.deconv2 = Conv2DLeakyReLUBN(64, 64, 3, 1, 1, 0.2)
        self.layer7 = Conv2DLeakyReLUBN(64, setup_params['D'], 3, 1, 1, 0.2)
        self.layer8 = Conv2DLeakyReLUBN(setup_params['D'], setup_params['D'], 3, 1, 1, 0.2)
        self.layer9 = Conv2DLeakyReLUBN(setup_params['D'], setup_params['D'], 3, 1, 1, 0.2)
        self.layer10 = nn.Conv2d(setup_params['D'], setup_params['D'], kernel_size=1, dilation=1)
        self.pred = nn.Hardtanh(min_val=0.0, max_val=setup_params['scaling_factor'])
        self.dropout = nn.Dropout(p=0.5)


    def forward(self, im):  # [4,1,96,96]

        # extract multi-scale features
        im = self.norm(im) # [4,1,96,96]
        out = self.layer1(im)  # [4,64,96,96]
        features = torch.cat((out, im), 1) # [4,65,96,96]
        out = self.layer2(features) + out  # [4,64,96,96] -> +out = [4,64,96,96]
        features = torch.cat((out, im), 1) # [4,65,96,96]
        out = self.layer3(features) + out  # [4,64,96,96] -> +out = [4,64,96,96]
        out = self.dropout(out)
        features = torch.cat((out, im), 1)
        out = self.layer4(features) + out
        features = torch.cat((out, im), 1)
        out = self.layer5(features) + out
        out = self.dropout(out)
        features = torch.cat((out, im), 1)
        out = self.layer6(features) + out

        # upsample by 4 in xy
        features = torch.cat((out, im), 1)
        out = interpolate(features, scale_factor=2)
        # out = features
        out = self.deconv1(out)
        # out = interpolate(out, scale_factor=2)
        out = self.deconv2(out)

        # refine z and exact xy
        out = self.layer7(out)
        out = self.layer8(out) + out
        out = self.layer9(out) + out

        # 1x1 conv and hardtanh for final result
        out = self.layer10(out)
        out = self.pred(out)
        return out













def convrelubn(in_channels, out_channels, kernel, padding=1, dilation=1):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel, padding=padding, dilation=dilation),
        nn.ReLU(inplace=True),
        nn.BatchNorm2d(out_channels)
    )

class Conv2DLeakyReLUBN(nn.Module):
    def __init__(self, input_channels, layer_width, kernel_size, padding, dilation, negative_slope):
        super(Conv2DLeakyReLUBN, self).__init__()
        self.conv = nn.Conv2d(input_channels, layer_width, kernel_size, 1, padding, dilation)
        self.lrelu = nn.LeakyReLU(negative_slope, inplace=True)
        self.bn = nn.BatchNorm2d(layer_width)

    def forward(self, x):
        out = self.conv(x)
        out = self.lrelu(out)
        out = self.bn(out)
        return out

class ResConv2DReLUBN(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, padding, dilation,negative_slope):
        super(ResConv2DReLUBN, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, 1, padding, dilation)
        self.relu = nn.ReLU(inplace=True)
        self.bn = nn.BatchNorm2d(out_channels)
        self.in_channels = in_channels
        self.out_channels = out_channels

    def forward(self, x):
        residual = x
        out = self.conv(x)
        out = self.relu(out)
        out = self.bn(out)

        # if self.in_channels == self.out_channels:
        out += residual

        return out

class ResConvLeakyReLUBN(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, padding, dilation, negative_slope):
        super(ResConvLeakyReLUBN, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, 1, padding, dilation)
        self.lrelu = nn.LeakyReLU(negative_slope, inplace=True)
        self.bn = nn.BatchNorm2d(out_channels)
        self.in_channels = in_channels
        self.out_channels = out_channels

    def forward(self, x):
        residual = x
        out = self.conv(x)
        out = self.lrelu(out)
        out = self.bn(out)

        if self.in_channels == self.out_channels:
            out += residual

        return out

# 0901 2 output = 3d grid with upsampled size + 3d grid with original size
class ResLocalizationCNN(nn.Module):
    def __init__(self, setup_params):
        super(ResLocalizationCNN, self).__init__()
        self.norm = nn.BatchNorm2d(num_features=1, affine=True)
        self.layer1 = Conv2DLeakyReLUBN(1, 64, 3, 1, 1, 0.2)
        # self.layer2 = ResConv2DLeakyReLUBN(64, 64, 3, 1, 1, 0.2)

        self.layer3 = ResConv2DLeakyReLUBN(64, 64, 3, (2, 2), (2, 2), 0.2)
        self.layer4 = ResConv2DLeakyReLUBN(64, 64, 3, (4, 4), (4, 4), 0.2)
        self.layer5 = ResConv2DLeakyReLUBN(64, 64, 3, (8, 8), (8, 8), 0.2)
        self.layer6 = ResConv2DLeakyReLUBN(64, 64, 3, (16, 16), (16, 16), 0.2)

        self.deconv1 = ResConv2DLeakyReLUBN(64, 64, 3, 1, 1, 0.2)
        # self.deconv2 = ResConv2DLeakyReLUBN(64, 64, 3, 1, 1, 0.2)
        self.layer7 = Conv2DLeakyReLUBN(64, setup_params['D'], 3, 1, 1, 0.2)
        self.layer8 = ResConv2DLeakyReLUBN(setup_params['D'], setup_params['D'], 3, 1, 1, 0.2)
        self.layer9 = ResConv2DLeakyReLUBN(setup_params['D'], setup_params['D'], 3, 1, 1, 0.2)
        self.layer10 = nn.Conv2d(setup_params['D'], setup_params['D'], kernel_size=1, dilation=1)
        self.pred = nn.Hardtanh(min_val=0.0, max_val=setup_params['scaling_factor'])
        # self.dropout = nn.Dropout(p=0.5)

        # maxpool_info = (round(1/int(float(setup_params['pixel_size_axial']))),setup_params['upsampling_factor'],setup_params['upsampling_factor']) 
        maxpool_info = (6,2,2)
        self.maxpool = nn.MaxPool3d(kernel_size=maxpool_info, stride=maxpool_info)

    def forward(self, im):  # [4,1,96,96]

        im = self.norm(im) # [4,1,96,96]
        out = self.layer1(im)  # [4,64,96,96]
        # out = self.layer2(out)  # [4,64,96,96] -> +out = [4,64,96,96]
        out = self.layer3(out)  # [4,64,96,96] -> +out = [4,64,96,96]
        out = self.layer4(out)
        out = self.layer5(out)
        out = self.layer6(out)

        # upsample by 2 in xy
        out = interpolate(out, scale_factor=2)
        out = self.deconv1(out)
        # out = self.deconv2(out)

        # refine z and exact xy
        out = self.layer7(out)
        out = self.layer8(out)
        # out = self.layer9(out)

        # 1x1 conv and hardtanh for final result
        upgrid = self.layer10(out)
        upgrid = self.pred(upgrid)

        # normgrid = self.maxpool(upgrid.unsqueeze(1))

        return upgrid

# 0721 replace plain conv layer by residual layer
class ResLocalizationCNN(nn.Module):
    def __init__(self, setup_params):
        super(ResLocalizationCNN, self).__init__()
        self.norm = nn.BatchNorm2d(num_features=1, affine=True)
        self.layer1 = Conv2DLeakyReLUBN(1, 64, 3, 1, 1, 0.2)
        self.layer2 = ResConv2DLeakyReLUBN(64, 64, 3, 1, 1, 0.2)

        self.layer3 = ResConv2DLeakyReLUBN(64, 64, 3, (2, 2), (2, 2), 0.2)
        self.layer4 = ResConv2DLeakyReLUBN(64, 64, 3, (4, 4), (4, 4), 0.2)
        self.layer5 = ResConv2DLeakyReLUBN(64, 64, 3, (8, 8), (8, 8), 0.2)
        self.layer6 = ResConv2DLeakyReLUBN(64, 64, 3, (16, 16), (16, 16), 0.2)

        self.deconv1 = ResConv2DLeakyReLUBN(64, 64, 3, 1, 1, 0.2)
        self.deconv2 = ResConv2DLeakyReLUBN(64, 64, 3, 1, 1, 0.2)
        self.layer7 = Conv2DLeakyReLUBN(64, setup_params['D'], 3, 1, 1, 0.2)
        self.layer8 = ResConv2DLeakyReLUBN(setup_params['D'], setup_params['D'], 3, 1, 1, 0.2)
        self.layer9 = ResConv2DLeakyReLUBN(setup_params['D'], setup_params['D'], 3, 1, 1, 0.2)
        self.layer10 = nn.Conv2d(setup_params['D'], setup_params['D'], kernel_size=1, dilation=1)
        self.pred = nn.Hardtanh(min_val=0.0, max_val=setup_params['scaling_factor'])
        self.dropout = nn.Dropout(p=0.5)

    def forward(self, im):  # [4,1,96,96]

        # extract multi-scale features
        im = self.norm(im) # [4,1,96,96]
        out = self.layer1(im)  # [4,64,96,96]
        out = self.layer2(out)  # [4,64,96,96] -> +out = [4,64,96,96]
        out = self.layer3(out)  # [4,64,96,96] -> +out = [4,64,96,96]
        out = self.dropout(out)
        out = self.layer4(out)
        out = self.layer5(out)
        out = self.dropout(out)
        out = self.layer6(out)

        # upsample by 2 in xy
        out = interpolate(out, scale_factor=2)
        out = self.deconv1(out)
        out = self.deconv2(out)

        # refine z and exact xy
        out = self.layer7(out)
        out = self.layer8(out)
        out = self.layer9(out)

        # 1x1 conv and hardtanh for final result
        out = self.layer10(out)
        out = self.pred(out)
        return out

# original Localization architecture
class LocalizationCNN(nn.Module):
    def __init__(self, setup_params):
        super(LocalizationCNN, self).__init__()
        self.norm = nn.BatchNorm2d(num_features=1, affine=True)
        self.layer1 = Conv2DLeakyReLUBN(1, 64, 3, 1, 1, 0.2)
        self.layer2 = Conv2DLeakyReLUBN(64 + 1, 64, 3, 1, 1, 0.2)

        self.layer3 = Conv2DLeakyReLUBN(64 + 1, 64, 3, (2, 2), (2, 2), 0.2)
        self.layer4 = Conv2DLeakyReLUBN(64 + 1, 64, 3, (4, 4), (4, 4), 0.2)
        self.layer5 = Conv2DLeakyReLUBN(64 + 1, 64, 3, (8, 8), (8, 8), 0.2)
        self.layer6 = Conv2DLeakyReLUBN(64 + 1, 64, 3, (16, 16), (16, 16), 0.2)

        self.deconv1 = Conv2DLeakyReLUBN(64 + 1, 64, 3, 1, 1, 0.2)
        self.deconv2 = Conv2DLeakyReLUBN(64, 64, 3, 1, 1, 0.2)
        self.layer7 = Conv2DLeakyReLUBN(64, setup_params['D'], 3, 1, 1, 0.2)
        self.layer8 = Conv2DLeakyReLUBN(setup_params['D'], setup_params['D'], 3, 1, 1, 0.2)
        self.layer9 = Conv2DLeakyReLUBN(setup_params['D'], setup_params['D'], 3, 1, 1, 0.2)
        self.layer10 = nn.Conv2d(setup_params['D'], setup_params['D'], kernel_size=1, dilation=1)
        self.pred = nn.Hardtanh(min_val=0.0, max_val=setup_params['scaling_factor'])
        self.dropout = nn.Dropout(p=0.5)

    def forward(self, im):  # [4,1,96,96]

        # extract multi-scale features
        im = self.norm(im) # [4,1,96,96]
        out = self.layer1(im)  # [4,64,96,96]
        features = torch.cat((out, im), 1) # [4,65,96,96]
        out = self.layer2(features) + out  # [4,64,96,96] -> +out = [4,64,96,96]
        features = torch.cat((out, im), 1) # [4,65,96,96]
        out = self.layer3(features) + out  # [4,64,96,96] -> +out = [4,64,96,96]
        out = self.dropout(out)
        features = torch.cat((out, im), 1)
        out = self.layer4(features) + out
        features = torch.cat((out, im), 1)
        out = self.layer5(features) + out
        out = self.dropout(out)
        features = torch.cat((out, im), 1)
        out = self.layer6(features) + out

        # upsample by 4 in xy
        features = torch.cat((out, im), 1)
        out = interpolate(features, scale_factor=2)
        # out = features
        out = self.deconv1(out)
        # out = interpolate(out, scale_factor=2)
        out = self.deconv2(out)

        # refine z and exact xy
        out = self.layer7(out)
        out = self.layer8(out) + out
        out = self.layer9(out) + out

        # 1x1 conv and hardtanh for final result
        out = self.layer10(out)
        out = self.pred(out)
        return out

# plain conv layer
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

        self.layer7 = convrelubn(64, 64, 3)
        self.layer8 = convrelubn(64, setup_params['D'], 3)
        self.layer9 = convrelubn(setup_params['D'], setup_params['D'], 3)
        self.layer10 = convrelubn(setup_params['D'], setup_params['D'], 3)

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

        out = self.layer7(out)
        out = self.layer8(out)
        out = self.layer9(out)
        out = self.layer10(out)
        out = self.pred(out)

        return out

# my own network update 0809
class FinalLoc3dResCNN(nn.Module):
    def __init__(self, setup_params):
        super(FinalLoc3dResCNN, self).__init__()
        self.norm = nn.BatchNorm2d(num_features=1, affine=True)
        self.layer1 = ResConvLeakyReLUBN(1, 64, 3, 1, 1, 0.2)
        self.layer2 = ResConvLeakyReLUBN(64, 64, 3, 1, 1, 0.2)

        self.layer3 = ResConvLeakyReLUBN(64, 64, 3, (2, 2), (2, 2), 0.2)
        self.layer4 = ResConvLeakyReLUBN(64, 64, 3, (4, 4), (4, 4), 0.2)
        self.layer5 = ResConvLeakyReLUBN(64, 64, 3, (8, 8), (8, 8), 0.2)
        self.layer6 = ResConvLeakyReLUBN(64, 64, 3, (16, 16), (16, 16), 0.2)

        self.deconv1 = ResConvLeakyReLUBN(64, 64, 3, 1, 1, 0.2)
        self.deconv2 = ResConvLeakyReLUBN(64, 64, 3, 1, 1, 0.2)
        self.layer7 = ResConvLeakyReLUBN(64, setup_params['D'], 3, 1, 1, 0.2)
        self.layer8 = ResConvLeakyReLUBN(setup_params['D'], setup_params['D'], 3, 1, 1, 0.2)
        self.layer9 = ResConvLeakyReLUBN(setup_params['D'], setup_params['D'], 3, 1, 1, 0.2)
        self.layer10 = nn.Conv2d(setup_params['D'], setup_params['D'], kernel_size=1, dilation=1)
        self.pred = nn.Hardtanh(min_val=0.0, max_val=setup_params['scaling_factor'])
        self.dropout = nn.Dropout(p=0.5)

    def forward(self, im):  # [N,1,96,96]

        # extract multi-scale features
        im = self.norm(im) # [N,1,96,96]
        out = self.layer1(im)  # [N,64,96,96]
        out = self.layer2(out)  # [N,64,96,96]
        out = self.layer3(out)  # [N,64,96,96]
        out = self.dropout(out)

        out = self.layer4(out)
        out = self.layer5(out)
        out = self.dropout(out)

        out = self.layer6(out)

        # upsample by 2 in xy
        out = interpolate(out, scale_factor=2)
        out = self.deconv1(out)
        out = self.deconv2(out)

        # refine z and exact xy
        out = self.layer7(out)
        out = self.layer8(out)
        out = self.layer9(out)

        # 1x1 conv and hardtanh for final result
        out = self.layer10(out)
        out = self.pred(out)
        return out

# 2021/04/24 ResNetUNet is not useful here
def convrelu(in_channels, out_channels, kernel, padding):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel, padding=padding),
        nn.ReLU(inplace=True),
    )

def convreluBN(in_channels, out_channels, kernel, padding):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel, padding=padding),
        nn.ReLU(inplace=True),
        nn.BatchNorm2d(out_channels)
    )

class ResNetUNet(nn.Module):
    def __init__(self, setup_params):
        super().__init__()

        self.base_model = models.resnet18(pretrained=True)
        self.base_layers = list(self.base_model.children())

        # self.layer0 = nn.Sequential(*self.base_layers[:3]) # size=(N, 64, x.H/2, x.W/2)
        self.layer0 =nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False),
            nn.BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            nn.ReLU(inplace=True)
        ) 
        self.layer0_1x1 = convrelu(64, 64, 1, 0)
        self.layer1 = nn.Sequential(*self.base_layers[3:5]) # size=(N, 64, x.H/4, x.W/4)
        self.layer1_1x1 = convrelu(64, 64, 1, 0)
        self.layer2 = self.base_layers[5]  # size=(N, 128, x.H/8, x.W/8)
        self.layer2_1x1 = convrelu(128, 128, 1, 0)
        self.layer3 = self.base_layers[6]  # size=(N, 256, x.H/16, x.W/16)
        self.layer3_1x1 = convrelu(256, 256, 1, 0)
        self.layer4 = self.base_layers[7]  # size=(N, 512, x.H/32, x.W/32)
        self.layer4_1x1 = convrelu(512, 512, 1, 0)

        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

        self.conv_up3 = convrelu(256 + 512, 512, 3, 1)
        self.conv_up2 = convrelu(128 + 512, 256, 3, 1)
        self.conv_up1 = convrelu(64 + 256, 256, 3, 1)
        self.conv_up0 = convrelu(64 + 256, 128, 3, 1)

        self.conv_original_size0 = convrelu(1, 64, 3, 1)
        self.conv_original_size1 = convrelu(64, 64, 3, 1)
        self.conv_original_size2 = convrelu(64 + 128, 64, 3, 1)

        self.conv_upsample_size0 = convrelu(1, 64, 3, 1)
        self.conv_upsample_size1 = convrelu(64, 64, 3, 1)
        self.conv_upsample_size2 = convrelu(64 + 64, 64, 3, 1)

        self.layer7 = convreluBN(64, setup_params['D'], 3, 1)
        self.layer8 = convreluBN(setup_params['D'], setup_params['D'], 3, 1)
        self.layer9 = convreluBN(setup_params['D'], setup_params['D'], 3, 1)
        self.layer10 = nn.Conv2d(setup_params['D'], setup_params['D'], kernel_size=1)
        self.pred = nn.Hardtanh(min_val=0.0, max_val=setup_params['scaling_factor'])

    def forward(self, input):
        x_upsample = interpolate(input, scale_factor=2)   # (N, 1, H*2, W*2)
        x_upsample = self.conv_upsample_size0(x_upsample)   # (N, 64, H*2, W*2)
        x_upsample = self.conv_upsample_size1(x_upsample)   # (N, 64, H*2, W*2)

        x_original = self.conv_original_size0(input)  # (N, 64, H, W)
        x_original = self.conv_original_size1(x_original) # (N, 64, H, W)

        layer0 = self.layer0(input)    # (N, 64, H/2, W/2)
        layer1 = self.layer1(layer0)   # (N, 64, H/4, W/4)
        layer2 = self.layer2(layer1)   # (N, 128, H/8, W/8)
        layer3 = self.layer3(layer2)   # (N, 256, H/8, W/8)
        layer4 = self.layer4(layer3)   # (N, 512, H/32, W/32)

        layer4 = self.layer4_1x1(layer4)  
        x = self.upsample(layer4)    # (N, 512, H/16, W/16)
        layer3 = self.layer3_1x1(layer3)
        x = torch.cat([x, layer3], dim=1)  # (N, 512+256, H/16, W/16)
        x = self.conv_up3(x)               # (N, 512, H/16, W/16)

        x = self.upsample(x)               # (N, 512, H/8, W/8)
        layer2 = self.layer2_1x1(layer2)
        x = torch.cat([x, layer2], dim=1)  # (N, 512+128, H/8, W/8)
        x = self.conv_up2(x)               # (N, 256, H/8, W/8)

        x = self.upsample(x)               # (N, 256, H/4, W/4)
        layer1 = self.layer1_1x1(layer1)
        x = torch.cat([x, layer1], dim=1)  # (N, 256+64, H/4, W/4)
        x = self.conv_up1(x)               # (N, 256, H/4, W/4)

        x = self.upsample(x)               # (N, 256, H/2, W/2)
        layer0 = self.layer0_1x1(layer0)
        x = torch.cat([x, layer0], dim=1)  # (N, 256+64, H/2, W/2)
        x = self.conv_up0(x)               # (N, 128, H/2, W/2)

        x = self.upsample(x)               # (N, 128, H, W)
        x = torch.cat([x, x_original], dim=1)  # (N, 128+64, H, W)
        x = self.conv_original_size2(x)   # (N, 64, H, W)

        x = self.upsample(x)               # (N, 64, H*2, W*2)
        x = torch.cat([x, x_upsample], dim=1)  # (N, 64+64, H*2, W*2)
        x = self.conv_upsample_size2(x)   # (N, 64, H*2, W*2)

        # refine z and exact xy
        out = self.layer7(x)          # (N, D, H*2, W*2)
        out = self.layer8(out) + out  # (N, D, H*2, W*2)
        out = self.layer9(out) + out  # (N, D, H*2, W*2)

        # 1x1 conv and hardtanh for final result
        out = self.layer10(out)
        out = self.pred(out)

        return out