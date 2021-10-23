import torch
import torch.nn as nn
from torch.nn.functional import interpolate
from torchvision import models

def convrelu(in_channels, out_channels, kernel, padding):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel, padding=padding),
        nn.ReLU(inplace=True),
    )

class ResNetUNet(nn.Module):
    def __init__(self, n_class=1):
        super().__init__()

        self.base_model = models.resnet18(pretrained=True)
        self.base_layers = list(self.base_model.children())

        self.layer0 = nn.Sequential(*self.base_layers[:3]) # size=(N, 64, x.H/2, x.W/2)
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

        self.conv_original_size0 = convrelu(3, 64, 3, 1)
        self.conv_original_size1 = convrelu(64, 64, 3, 1)
        self.conv_original_size2 = convrelu(64 + 128, 64, 3, 1)

        self.conv_upsample_size0 = convrelu(3, 64, 3, 1)
        self.conv_upsample_size1 = convrelu(64, 64, 3, 1)
        self.conv_upsample_size2 = convrelu(64+64, 64, 3, 1)

        self.conv_last = nn.Conv2d(64, n_class, 1)

    def forward(self, input):

        x_upsample = self.upsample(input)
        x_upsample = self.conv_upsample_size0(x_upsample)
        x_upsample = self.conv_upsample_size1(x_upsample)

        x_original = self.conv_original_size0(input)
        x_original = self.conv_original_size1(x_original)

        layer0 = self.layer0(input)
        layer1 = self.layer1(layer0)
        layer2 = self.layer2(layer1)
        layer3 = self.layer3(layer2)
        layer4 = self.layer4(layer3)

        layer4 = self.layer4_1x1(layer4)
        x = self.upsample(layer4)
        layer3 = self.layer3_1x1(layer3)
        x = torch.cat([x, layer3], dim=1)
        x = self.conv_up3(x)

        x = self.upsample(x)
        layer2 = self.layer2_1x1(layer2)
        x = torch.cat([x, layer2], dim=1)
        x = self.conv_up2(x)

        x = self.upsample(x)
        layer1 = self.layer1_1x1(layer1)
        x = torch.cat([x, layer1], dim=1)
        x = self.conv_up1(x)

        x = self.upsample(x)
        layer0 = self.layer0_1x1(layer0)
        x = torch.cat([x, layer0], dim=1)
        x = self.conv_up0(x)

        x = self.upsample(x)
        x = torch.cat([x, x_original], dim=1)
        x = self.conv_original_size2(x)

        x = self.upsample(x)
        x = torch.cat([x, x_upsample], dim=1)
        x = self.conv_upsample_size2(x)

        pred_xy = self.conv_last(x)


        return pred_xy

class ResNetUNet_v1(nn.Module):
    def __init__(self, n_class=1):
        super().__init__()

        self.base_model = models.resnet18(pretrained=True)
        self.base_layers = list(self.base_model.children())

        self.layer0 = nn.Sequential(*self.base_layers[:3]) # size=(N, 64, x.H/2, x.W/2)
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

        self.conv_original_size0 = convrelu(3, 64, 3, 1)
        self.conv_original_size1 = convrelu(64, 64, 3, 1)
        self.conv_original_size2 = convrelu(64 + 128, 64, 3, 1)

        self.conv_upsample_size0 = convrelu(3, 64, 3, 1)
        self.conv_upsample_size1 = convrelu(64, 64, 3, 1)
        self.conv_upsample_size2 = convrelu(64+64, 64, 3, 1)

        self.conv_last = nn.Conv2d(64, n_class, 1)

        self.conv_zeta1 = convrelu(64 + 1, 64, 3, 1)
        self.conv_zeta2 = convrelu(64 + 64, 64, 3, 1)
        self.conv_zeta3 = convrelu(64 + 64, 64, 3, 1)

        self.conv_last_zeta = nn.Conv2d(64, n_class, 1)
        self.conv_last_flux = nn.Conv2d(64, n_class, 1)
        # self.pred_zeta = nn.Hardtanh(min_val=0.0, max_val=40)
        # self.pred_flux = nn.Hardtanh(min_val=0.0, max_val=200)

    def forward(self, input):

        x_upsample = self.upsample(input)
        x_upsample = self.conv_upsample_size0(x_upsample)
        x_upsample = self.conv_upsample_size1(x_upsample)

        x_original = self.conv_original_size0(input)
        x_original = self.conv_original_size1(x_original)

        layer0 = self.layer0(input)
        layer1 = self.layer1(layer0)
        layer2 = self.layer2(layer1)
        layer3 = self.layer3(layer2)
        layer4 = self.layer4(layer3)

        layer4 = self.layer4_1x1(layer4)
        x = self.upsample(layer4)
        layer3 = self.layer3_1x1(layer3)
        x = torch.cat([x, layer3], dim=1)
        x = self.conv_up3(x)

        x = self.upsample(x)
        layer2 = self.layer2_1x1(layer2)
        x = torch.cat([x, layer2], dim=1)
        x = self.conv_up2(x)

        x = self.upsample(x)
        layer1 = self.layer1_1x1(layer1)
        x = torch.cat([x, layer1], dim=1)
        x = self.conv_up1(x)

        x = self.upsample(x)
        layer0 = self.layer0_1x1(layer0)
        x = torch.cat([x, layer0], dim=1)
        x = self.conv_up0(x)

        x = self.upsample(x)
        x = torch.cat([x, x_original], dim=1)
        x = self.conv_original_size2(x)

        x = self.upsample(x)
        x = torch.cat([x, x_upsample], dim=1)
        x = self.conv_upsample_size2(x)

        pred_xy = self.conv_last(x)

        # based on pred_xy and orignal image to predict depth and flux
        x = torch.cat([x,x_upsample], dim=1)
        x = self.conv_zeta1(x)
        x = self.conv_zeta2(x)
        x = self.conv_zeta3(x)
        pred_zeta = self.conv_last_zeta(x)
        pred_flux = self.conv_last_flux(x)

        return pred_xy, pred_zeta, pred_flux



'''
# 2021/04/24 ResNetUNet
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

class ResNetUNet_v1(nn.Module):
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

'''