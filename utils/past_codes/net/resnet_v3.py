from torch import Tensor
from typing import Type, Callable, Union, List, Optional
import torch.nn as nn
from torch.nn.functional import interpolate

def conv3x3(in_planes, out_planes, stride= 1, groups= 1, dilation= 1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=dilation, groups=groups, bias=False, dilation=dilation)

def conv1x1(in_planes: int, out_planes: int, stride: int = 1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)

class BasicBlock(nn.Module):
    expansion: int = 1

    def __init__(
        self,
        inplanes: int,
        planes: int,
        stride: int = 1,
        downsample: Optional[nn.Module] = None,
        groups: int = 1,
        base_width: int = 64,
        dilation: int = 1,
        norm_layer: Optional[Callable[..., nn.Module]] = None
    ):
        super(BasicBlock, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if groups != 1 or base_width != 64:
            raise ValueError('BasicBlock only supports groups=1 and base_width=64')
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = norm_layer(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x: Tensor):
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
    def __init__(
        self,
        layers: List[int],
        setup_params = None,
        zero_init_residual: bool = False,
    ):
        super(ResNet, self).__init__()
        self.norm_layer = nn.BatchNorm2d
        block=Union[BasicBlock]
        self.inplanes = 64
        self.dilation = 1
        self.groups = 1
        self.base_width = 64
        self.conv1 = nn.Conv2d(1, self.inplanes, kernel_size=7, stride=1, padding=3, bias=False)
        self.bn1 = self.norm_layer(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        # self.upsample = nn.functional.interpolate(scale_factor=2, mode='bilinear', align_corners=True)
        # for depth
        self.layer1 = self._make_layer(block, 64, layers[0])
        # self.layer2 = self._make_layer(block, 128, layers[1])
        self.layer3 = self._make_layer(block, 64, layers[2])
        self.layer4 = self._make_layer(block, setup_params['D'], layers[3])
        # for lateral position
        self.layer5 = self._make_layer(block, setup_params['D'], layers[4])
        self.Hardtanh = nn.Hardtanh(min_val=0.0, max_val=setup_params['scaling_factor'])
        self.dropout = nn.Dropout(p=0.5)

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

    def _make_layer(self, block: Type[Union[BasicBlock]], planes: int, blocks: int, stride: int = 1, dilate: bool = False):
        norm_layer = self.norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion),
            )
        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, self.groups,
                            self.base_width, previous_dilation, norm_layer))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, groups=self.groups,
                                base_width=self.base_width, dilation=self.dilation,
                                norm_layer=norm_layer))

        return nn.Sequential(*layers)

    def _forward_impl(self, x: Tensor):
        x = self.conv1(x)      # [N,C,H,W] -> [N,64,H,W]
        x = self.bn1(x)
        x = self.relu(x)

        x = self.layer1(x)     # [N,64,H,W]
        # x = self.dropout(x)
        # x = self.layer2(x)     # [N,128,H,H]
        # x = self.layer3(x)     # [N,256,H,W]
        x = self.layer4(x)     # [N,D,H,W]

        x = interpolate(x, scale_factor=2, mode='bilinear')   # [N,D,H*2,W*2]
        x = self.layer5(x)     # [N,D,H*2,W*2]
        # x = self.Hardtanh(x)

        return x

    def _forward_impl_v2(self, x: Tensor):
        x = interpolate(x, scale_factor=2, mode='bilinear')   # [N,1,H*2,W*2]
        x = self.conv1(x)      # [N,1,H*2,W*2] -> [N,64,H*2,W*2]
        x = self.bn1(x)
        x = self.relu(x)

        x = self.layer1(x)     # [N,64,H*2,W*2]
        # x = self.layer2(x)     # [N,128,H,H]
        x = self.dropout(x)
        x = self.layer3(x)     # [N,D,H*2,W*2]
        x = self.layer4(x)     # [N,D,H*2,W*2]
        x = self.dropout(x)
        x = self.layer5(x)     # [N,D,H*2,W*2]
        x = self.Hardtanh(x)
        return x

    def forward(self, x: Tensor):
        return self._forward_impl_v2(x)
