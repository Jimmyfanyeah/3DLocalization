# Import modules and libraries
import torch
import torch.nn as nn
from torch.nn.functional import interpolate
import math

###############################################
## blocks for all network
################################################

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



########################################
# network structure below
########################################

# 0721 replace plain conv layer by residual layer
class ResLocalizationCNN(nn.Module):
    def __init__(self, opt):
        super(ResLocalizationCNN, self).__init__()
        self.opt = opt

        self.norm = nn.BatchNorm2d(num_features=1, affine=True)
        self.layer1 = Conv2DLeakyReLUBN(1, 64, 3, 1, 1, 0.2)
        self.layer2 = ResConv2DLeakyReLUBN(64, 64, 3, 1, 1, 0.2)

        self.layer3 = ResConv2DLeakyReLUBN(64, 64, 3, (2, 2), (2, 2), 0.2)
        self.layer4 = ResConv2DLeakyReLUBN(64, 64, 3, (4, 4), (4, 4), 0.2)
        self.layer5 = ResConv2DLeakyReLUBN(64, 64, 3, (8, 8), (8, 8), 0.2)
        self.layer6 = ResConv2DLeakyReLUBN(64, 64, 3, (16, 16), (16, 16), 0.2)

        self.deconv1 = ResConv2DLeakyReLUBN(64, 64, 3, 1, 1, 0.2)
        self.deconv2 = ResConv2DLeakyReLUBN(64, 64, 3, 1, 1, 0.2)
        self.layer7 = Conv2DLeakyReLUBN(64, opt.D, 3, 1, 1, 0.2)
        self.layer8 = ResConv2DLeakyReLUBN(opt.D, opt.D, 3, 1, 1, 0.2)
        self.layer9 = ResConv2DLeakyReLUBN(opt.D, opt.D, 3, 1, 1, 0.2)
        self.layer10 = nn.Conv2d(opt.D, opt.D, kernel_size=1, dilation=1)
        self.pred = nn.Hardtanh(min_val=0.0, max_val=opt.scaling_factor)
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

        if self.opt.upsampling_factor == 2:
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
    def __init__(self, opt):
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
        self.layer7 = Conv2DLeakyReLUBN(64, opt.D, 3, 1, 1, 0.2)
        self.layer8 = Conv2DLeakyReLUBN(opt.D, opt.D, 3, 1, 1, 0.2)
        self.layer9 = Conv2DLeakyReLUBN(opt.D, opt.D, 3, 1, 1, 0.2)
        self.layer10 = nn.Conv2d(opt.D, opt.D, kernel_size=1, dilation=1)
        self.pred = nn.Hardtanh(min_val=0.0, max_val=opt.scaling_factor)
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

