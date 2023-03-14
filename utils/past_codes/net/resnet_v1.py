from torch import Tensor
from typing import Type, Callable, Union, List, Optional
import torch.nn as nn
from torch.nn.functional import interpolate
import torch

def conv3x3(in_planes, out_planes, stride= 1, groups= 1, dilation= 1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=dilation, groups=groups, bias=False, dilation=dilation)

def conv1x1(in_planes: int, out_planes: int, stride: int = 1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)

class BasicBlock(nn.Module):
    def __init__(
        self,
        inplanes: int,
        planes: int,
        stride: int = 1,
        downsample: Optional[nn.Module] = None,
        norm_layer: Optional[Callable[..., nn.Module]] = None
    ):
        super(BasicBlock, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d

        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = norm_layer(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out

class ResNet(nn.Module):
    def __init__(self, layers: List[int], setup_params = None, zero_init_residual: bool = False ):
        super(ResNet, self).__init__()

        block=Union[BasicBlock]
        self.norm_layer = nn.BatchNorm2d
        self.bn = self.norm_layer(num_features=1, affine=True)
        self.relu = nn.ReLU(inplace=True)

        self.layer0 = nn.Sequential(
            conv3x3(1, 64),
            self.norm_layer(64),
            self.relu,)

        self.layer1 = self._make_layer(block, 64, 64, layers[0])
        self.layer2 = self._make_layer(block, 64, 64, layers[1])
        self.layer3 = self._make_layer(block, 64, 64, layers[2])

        self.layer4 = self._make_layer(block, 64, 64, layers[3])
        self.layer5 = self._make_layer(block, 64, setup_params['D'], layers[4])
        self.layer6 = self._make_layer(block, setup_params['D'], setup_params['D'], layers[5])
        self.layer7 = nn.Sequential(
                conv1x1(setup_params['D'], setup_params['D']),
                self.norm_layer(setup_params['D']),
                self.relu,)

        self.pred = nn.Hardtanh(min_val=0.0, max_val=setup_params['scaling_factor'])

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)  # type: ignore[arg-type]

    def _make_layer(self, block: Type[Union[BasicBlock]], inplanes, planes, blocks, stride=1):
        norm_layer = self.norm_layer
        downsample = None

        layers = []
        if stride != 1 or inplanes != planes:
            downsample = nn.Sequential(
                conv1x1(inplanes, planes , stride),
                norm_layer(planes),
            )
        layers.append(block(inplanes, planes, stride, downsample, norm_layer))
        for _ in range(1, blocks):
            layers.append(block(planes, planes, norm_layer=norm_layer))

        return nn.Sequential(*layers)

    def _forward_impl(self, im: Tensor):
        im = self.bn(im)     # [N,1,H,W]
        out = self.layer0(im)    # [N,64,H,W]

        features = out
        # features = torch.cat((out,im),1)
        out = self.layer1(features)     # [N,64,H,W]
        features = out
        # features = torch.cat((out,im),1)
        out = self.layer2(features)     # [N,64,H,H]
        features = out
        # features = torch.cat((out,im),1)
        out = self.layer3(features)     # [N,64,H,W]

        # features = torch.cat((out,im),1)
        features = out
        out = interpolate(features, scale_factor=2, mode='bilinear',align_corners=True)   # [N,65,H*2,W*2]
        out = self.layer4(out)     # [N,64,H*2,W*2]

        out = self.layer5(out)     # [N,D,H*2,W*2]
        out = self.layer6(out)     # [N,D,H*2,W*2]

        out = self.layer7(out)
        out = self.pred(out)

        return out

    def forward(self, x: Tensor):
        return self._forward_impl(x)
