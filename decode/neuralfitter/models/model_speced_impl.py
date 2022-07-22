import torch
from torch import nn
from typing import Union

from . import model_param


class SigmaMUNet_variant(model_param.DoubleMUnet):
    """ num channels = 3, allow 3 overlap points
    ch_out = 28
    out_channels_heads = (3, 12, 12, 1)  # p head, phot, xyz_mu head, phot, xyz_sig head, bg head

    # channel indices with respective activation function
    sigmoid_ch_ix = [0,1,2, 3,4,5, 15,16,17, 18,19,20, 21,22,23, 24,25,26, 27] # p_head, phot, phot_sig, xyz_sig, bg_head
    tanh_ch_ix = [6,7,8, 9,10,11, 12,13,14] # xyz_mu

    p_ch_ix = slice(0,3)  # channel indices of the respective parameters
    pxyz_mu_ch_ix = slice(3, 15)
    pxyz_sig_ch_ix = slice(15, 27)
    bg_ch_ix = [27]
    """

    # num channels = 2
    ch_out = 19
    out_channels_heads = (2, 8, 8, 1)  # p head, phot, xyz_mu head, phot, xyz_sig head, bg head

    # channel indices with respective activation function
    sigmoid_ch_ix = [0,1, 2,3, 10,11, 12,13, 14,15, 16,17, 18] # p_head, phot, phot_sig, xyz_sig, bg_head
    tanh_ch_ix = [4,5, 6,7, 8,9] # xyz_mu

    p_ch_ix = slice(0,2)  # channel indices of the respective parameters
    pxyz_mu_ch_ix = slice(2, 10)
    pxyz_sig_ch_ix = slice(10, 18)
    bg_ch_ix = [18]

    sigma_eps_default = 0.001

    def __init__(self, ch_in: int, *, depth_shared: int, depth_union: int, initial_features: int, 
                 inter_features: int, norm=None, norm_groups=None, norm_head=None, norm_head_groups=None, pool_mode='StrideConv',
                 upsample_mode='bilinear', skip_gn_level: Union[None, bool] = None,
                 activation=nn.ReLU(), disabled_attributes=None, kaiming_normal=True):

        super().__init__(ch_in=ch_in, ch_out=self.ch_out, depth_shared=depth_shared, depth_union=depth_union,
                         initial_features=initial_features, inter_features=inter_features,
                         norm=norm, norm_groups=norm_groups, norm_head=norm_head,
                         norm_head_groups=norm_head_groups, pool_mode=pool_mode,
                         upsample_mode=upsample_mode,
                         skip_gn_level=skip_gn_level, activation=activation,
                         disabled_attributes=disabled_attributes,
                         use_last_nl=False)

        self.mt_heads = torch.nn.ModuleList(
            [model_param.MLTHeads(in_channels=inter_features, out_channels=ch_out,
                                  activation=activation, last_kernel=1, padding=True,
                                  norm=norm_head, norm_groups=norm_head_groups)
             for ch_out in self.out_channels_heads]
        )

        """Register sigma as parameter such that it is stored in the models state dict and loaded correctly."""
        self.register_parameter('sigma_eps',
                                torch.nn.Parameter(torch.tensor([self.sigma_eps_default]),
                                                   requires_grad=False))

        if kaiming_normal:
            self.apply(self.weight_init)

            # custom
            torch.nn.init.kaiming_normal_(self.mt_heads[0].core[0].weight, mode='fan_in',
                                          nonlinearity='relu')
            torch.nn.init.kaiming_normal_(self.mt_heads[0].out_conv.weight, mode='fan_in',
                                          nonlinearity='linear')
            torch.nn.init.constant_(self.mt_heads[0].out_conv.bias, -6.)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # model_param, line 274
        # if channel_in = 3, input=3 frames (use spatial info), each frame go trhough UNet and union
        x = self._forward_core(x)

        """Forward through the respective heads"""
        x_heads = [mt_head.forward(x) for mt_head in self.mt_heads]
        x = torch.cat(x_heads, dim=1)

        """Clamp prob before sigmoid"""
        x[:, [0,1,2]] = torch.clamp(x[:, [0,1,2]], min=-8., max=8.)

        """Apply non linearities"""
        x[:, self.sigmoid_ch_ix] = torch.sigmoid(x[:, self.sigmoid_ch_ix])
        x[:, self.tanh_ch_ix] = torch.tanh(x[:, self.tanh_ch_ix])

        """Add epsilon to sigmas and rescale"""
        x[:, self.pxyz_sig_ch_ix] = x[:, self.pxyz_sig_ch_ix] * 3 + self.sigma_eps

        """Disabled attributes get set to constants"""
        if self.disabled_attr_ix is not None:
            print('Attention: disabled_attr_ix')
            for ix in self.disabled_attr_ix:
                # Set means to 0
                x[:, 1 + ix] = x[:, 1 + ix] * 0
                # Set sigmas to 0.1
                x[:, 5 + ix] = x[:, 5 + ix] * 0 + 0.1

        return x

    def apply_detection_nonlin(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def apply_nonlin(self, o: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    @classmethod
    def parse(cls, param, **kwargs):

        activation = getattr(torch.nn, param.HyperParameter.arch_param.activation)
        activation = activation()
        return cls(
            ch_in=param.HyperParameter.channels_in,
            depth_shared=param.HyperParameter.arch_param.depth_shared,
            depth_union=param.HyperParameter.arch_param.depth_union,
            initial_features=param.HyperParameter.arch_param.initial_features,
            inter_features=param.HyperParameter.arch_param.inter_features,
            activation=activation,
            norm=param.HyperParameter.arch_param.norm,
            norm_groups=param.HyperParameter.arch_param.norm_groups,
            norm_head=param.HyperParameter.arch_param.norm_head,
            norm_head_groups=param.HyperParameter.arch_param.norm_head_groups,
            pool_mode=param.HyperParameter.arch_param.pool_mode,
            upsample_mode=param.HyperParameter.arch_param.upsample_mode,
            skip_gn_level=param.HyperParameter.arch_param.skip_gn_level,
            disabled_attributes=param.HyperParameter.disabled_attributes,
            kaiming_normal=param.HyperParameter.arch_param.init_custom
        )

    @staticmethod
    def weight_init(m):
        """
        Apply Kaiming normal init. Call this recursively by model.apply(model.weight_init)

        Args:
            m: model
        """
        if isinstance(m, torch.nn.Conv2d):
            torch.nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')



class SigmaMUNet(model_param.DoubleMUnet):
    ch_out = 10
    out_channels_heads = (1, 4, 4, 1)  # p head, phot,xyz_mu head, phot,xyz_sig head, bg head

    sigmoid_ch_ix = [0, 1, 5, 6, 7, 8, 9]  # channel indices with respective activation function
    tanh_ch_ix = [2, 3, 4]

    p_ch_ix = [0]  # channel indices of the respective parameters
    pxyz_mu_ch_ix = slice(1, 5)
    pxyz_sig_ch_ix = slice(5, 9)
    bg_ch_ix = [10]
    sigma_eps_default = 0.001

    def __init__(self, ch_in: int, *, depth_shared: int, depth_union: int, initial_features: int, 
                 inter_features: int,
                 norm=None, norm_groups=None, norm_head=None, norm_head_groups=None, pool_mode='StrideConv',
                 upsample_mode='bilinear', skip_gn_level: Union[None, bool] = None,
                 activation=nn.ReLU(), disabled_attributes=None, kaiming_normal=True):

        super().__init__(ch_in=ch_in, ch_out=self.ch_out, depth_shared=depth_shared, depth_union=depth_union,
                         initial_features=initial_features, inter_features=inter_features,
                         norm=norm, norm_groups=norm_groups, norm_head=norm_head,
                         norm_head_groups=norm_head_groups, pool_mode=pool_mode,
                         upsample_mode=upsample_mode,
                         skip_gn_level=skip_gn_level, activation=activation,
                         disabled_attributes=disabled_attributes,
                         use_last_nl=False)

        self.mt_heads = torch.nn.ModuleList(
            [model_param.MLTHeads(in_channels=inter_features, out_channels=ch_out,
                                  activation=activation, last_kernel=1, padding=True,
                                  norm=norm_head, norm_groups=norm_head_groups)
             for ch_out in self.out_channels_heads]
        )

        """Register sigma as parameter such that it is stored in the models state dict and loaded correctly."""
        self.register_parameter('sigma_eps',
                                torch.nn.Parameter(torch.tensor([self.sigma_eps_default]),
                                                   requires_grad=False))

        if kaiming_normal:
            self.apply(self.weight_init)

            # custom
            torch.nn.init.kaiming_normal_(self.mt_heads[0].core[0].weight, mode='fan_in',
                                          nonlinearity='relu')
            torch.nn.init.kaiming_normal_(self.mt_heads[0].out_conv.weight, mode='fan_in',
                                          nonlinearity='linear')
            torch.nn.init.constant_(self.mt_heads[0].out_conv.bias, -6.)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # model_param, line 274
        # if channel_in = 3, input=3 frames (use spatial info), each frame go trhough UNet and union
        x = self._forward_core(x)

        """Forward through the respective heads"""
        x_heads = [mt_head.forward(x) for mt_head in self.mt_heads]
        x = torch.cat(x_heads, dim=1)

        """Clamp prob before sigmoid"""
        x[:, [0]] = torch.clamp(x[:, [0]], min=-8., max=8.)

        """Apply non linearities"""
        x[:, self.sigmoid_ch_ix] = torch.sigmoid(x[:, self.sigmoid_ch_ix])
        x[:, self.tanh_ch_ix] = torch.tanh(x[:, self.tanh_ch_ix])

        """Add epsilon to sigmas and rescale"""
        x[:, self.pxyz_sig_ch_ix] = x[:, self.pxyz_sig_ch_ix] * 3 + self.sigma_eps

        """Disabled attributes get set to constants"""
        if self.disabled_attr_ix is not None:
            for ix in self.disabled_attr_ix:
                # Set means to 0
                x[:, 1 + ix] = x[:, 1 + ix] * 0
                # Set sigmas to 0.1
                x[:, 5 + ix] = x[:, 5 + ix] * 0 + 0.1

        return x

    def apply_detection_nonlin(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def apply_nonlin(self, o: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    @classmethod
    def parse(cls, param, **kwargs):

        activation = getattr(torch.nn, param.HyperParameter.arch_param.activation)
        activation = activation()
        return cls(
            ch_in=param.HyperParameter.channels_in,
            depth_shared=param.HyperParameter.arch_param.depth_shared,
            depth_union=param.HyperParameter.arch_param.depth_union,
            initial_features=param.HyperParameter.arch_param.initial_features,
            inter_features=param.HyperParameter.arch_param.inter_features,
            activation=activation,
            norm=param.HyperParameter.arch_param.norm,
            norm_groups=param.HyperParameter.arch_param.norm_groups,
            norm_head=param.HyperParameter.arch_param.norm_head,
            norm_head_groups=param.HyperParameter.arch_param.norm_head_groups,
            pool_mode=param.HyperParameter.arch_param.pool_mode,
            upsample_mode=param.HyperParameter.arch_param.upsample_mode,
            skip_gn_level=param.HyperParameter.arch_param.skip_gn_level,
            disabled_attributes=param.HyperParameter.disabled_attributes,
            kaiming_normal=param.HyperParameter.arch_param.init_custom
        )

    @staticmethod
    def weight_init(m):
        """
        Apply Kaiming normal init. Call this recursively by model.apply(model.weight_init)
        Args:
            m: model
        """
        if isinstance(m, torch.nn.Conv2d):
            torch.nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')