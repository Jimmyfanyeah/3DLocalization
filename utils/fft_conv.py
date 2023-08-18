# -*- coding: utf-8 -*-
"""
Created on Tue Sep  7 23:20:33 2021

@author: mastaffs
"""

from functools import partial
from typing import Tuple, Union, Iterable
from torch.autograd import Function, Variable
import torch
from torch import nn, Tensor
from torch.fft import rfftn, irfftn
import torch.nn.functional as f
from torch.nn.modules.utils import _ntuple


def complex_matmul(a: Tensor, b: Tensor, groups: int = 1) -> Tensor:
    """Multiplies two complex-valued tensors."""
    # Scalar matrix multiplication of two tensors, over only the first channel
    # dimensions. Dimensions 3 and higher will have the same shape after multiplication.
    # We also allow for "grouped" multiplications, where multiple sections of channels
    # are multiplied independently of one another (required for group convolutions).
    scalar_matmul = partial(torch.einsum, "agc..., gbc... -> agb...")
    a = a.view(a.size(0), groups, -1, *a.shape[2:])
    b = b.view(groups, -1, *b.shape[1:])

    # Compute the real and imaginary parts independently, then manually insert them
    # into the output Tensor.  This is fairly hacky but necessary for PyTorch 1.7.0,
    # because Autograd is not enabled for complex matrix operations yet.  Not exactly
    # idiomatic PyTorch code, but it should work for all future versions (>= 1.7.0).
    real = scalar_matmul(a.real, b.real) - scalar_matmul(a.imag, b.imag)
    imag = scalar_matmul(a.imag, b.real) + scalar_matmul(a.real, b.imag)
    c = torch.zeros(real.shape, dtype=torch.complex64, device=a.device)
    c.real, c.imag = real, imag

    return c.view(c.size(0), -1, *c.shape[3:])


def to_ntuple(val: Union[int, Iterable[int]], n: int) -> Tuple[int, ...]:
    """Casts to a tuple with length 'n'.  Useful for automatically computing the
    padding and stride for convolutions, where users may only provide an integer.
    Args:
        val: (Union[int, Iterable[int]]) Value to cast into a tuple.
        n: (int) Desired length of the tuple
    Returns:
        (Tuple[int, ...]) Tuple of length 'n'
    """
    if isinstance(val, Iterable):
        out = tuple(val)
        if len(out) == n:
            return out
        else:
            raise ValueError(f"Cannot cast tuple of length {len(out)} to length {n}.")
    else:
        return n * (val,)


def flip(x: Variable, dim: int) -> Variable:
    """Flip torch Variable along given dimension axis."""
    xsize = x.size()
    dim = x.dim() + dim if dim < 0 else dim
    x = x.contiguous().view(-1, *xsize[dim:])
    x = x.view(x.size(0), x.size(1), -1)[:,
            getattr(torch.arange(x.size(1)-1, -1, -1),
                    ('cpu', 'cuda')[x.is_cuda])().long(), :]
    return x.view(xsize)


class ReflectionPadNd(Function):
    """Padding for same convolutional layer."""

    @staticmethod
    def forward(ctx: Function, input: Variable, pad: Tuple[int]) -> Variable:
        ctx.pad = pad
        ctx.input_size = input.size()
        ctx.l_inp = len(input.size())
        ctx.pad_tup = tuple([(a, b) for a, b in zip(pad[:-1:2], pad[1::2])][::-1])
        # ctx.pad_tup = tuple([(a, b) for a, b in zip(pad[:-1:2], pad[1::2])])
        ctx.l_pad = len(ctx.pad_tup)
        ctx.l_diff = ctx.l_inp - ctx.l_pad
        assert ctx.l_inp >= ctx.l_pad

        new_dim = tuple([sum((d,) + ctx.pad_tup[i])
                         for i, d in enumerate(input.size()[-ctx.l_pad:])])
        assert all([d > 0 for d in new_dim]), 'input is too small'

        # Create output tensor by concatenating with reflected chunks.
        output = input.new(input.size()[:(ctx.l_diff)] + new_dim).zero_()
        c_input = input

        for i, p in zip(range(ctx.l_inp)[-ctx.l_pad:], ctx.pad_tup):
            if p[0] > 0:
                # chunk1 = flip(c_input.narrow(i, 0, pad[0]), i)
                chunk1 = flip(c_input.narrow(i, 0, p[0]), i)
                c_input = torch.cat((chunk1, c_input), i)
            if p[1] > 0:
                chunk2 = flip(c_input.narrow(i, c_input.shape[i]-p[1], p[1]), i)
                c_input = torch.cat((c_input, chunk2), i)
        output.copy_(c_input)
        return output

    @staticmethod
    def backward(ctx: Function, grad_output: Variable) -> Variable:
        grad_input = Variable(grad_output.data.new(ctx.input_size).zero_())
        grad_input_slices = [slice(0, x,) for x in ctx.input_size]

        cg_output = grad_output
        for i_s, p in zip(range(ctx.l_inp)[-ctx.l_pad:], ctx.pad_tup):
            if p[0] > 0:
                cg_output = cg_output.narrow(i_s, p[0],
                                             cg_output.size(i_s) - p[0])
            if p[1] > 0:
                cg_output = cg_output.narrow(i_s, 0,
                                             cg_output.size(i_s) - p[1])
        gis = tuple(grad_input_slices)
        grad_input[gis] = cg_output

        return grad_input, None, None


def ReflectionPad3d(input: Variable, padding: Union[int, Tuple[int]]) -> Variable:
    """Wrapper for ReflectionPadNd function in 3 dimensions."""
    padding = _ntuple(6)(padding)
    return ReflectionPadNd.apply(input, padding)



def fft_conv(
    signal: Tensor,
    kernel: Tensor,
    bias: Tensor = None,
    padding: Union[int, Iterable[int]] = 0,
    padding_mode: str = 'constant',
    stride: Union[int, Iterable[int]] = 1,
    groups: int = 1,
) -> Tensor:
    """Performs N-d convolution of Tensors using a fast fourier transform, which
    is very fast for large kernel sizes. Also, optionally adds a bias Tensor after
    the convolution (in order ot mimic the PyTorch direct convolution).
    Args:
        signal: (Tensor) Input tensor to be convolved with the kernel.
        kernel: (Tensor) Convolution kernel.
        bias: (Tensor) Bias tensor to add to the output.
        padding: (Union[int, Iterable[int]) Number of zero samples to pad the
            input on the last dimension.
        stride: (Union[int, Iterable[int]) Stride size for computing output values.
    Returns:
        (Tensor) Convolved tensor
    """
    # Cast padding & stride to tuples.
    padding_ = to_ntuple(padding, n=signal.ndim - 2)
    stride_ = to_ntuple(stride, n=signal.ndim - 2)

    # Pad the input signal & kernel tensors
    signal_padding = [p for p in padding_[::-1] for _ in range(2)]
    if padding_mode == 'reflect' and signal.ndim==5:
        signal = ReflectionPad3d(signal, signal_padding)
    else:
        signal = f.pad(signal, signal_padding, mode = padding_mode)

    # Because PyTorch computes a *one-sided* FFT, we need the final dimension to
    # have *even* length.  Just pad with one more zero if the final dimension is odd.
    if signal.size(-1) % 2 != 0:
        signal_ = f.pad(signal, [0, 1])
    else:
        signal_ = signal

    kernel_padding = [
        pad
        for i in reversed(range(2, signal_.ndim))
        for pad in [0, signal_.size(i) - kernel.size(i)]
    ]
    padded_kernel = f.pad(kernel, kernel_padding)

    # Perform fourier convolution -- FFT, matrix multiply, then IFFT
    # signal_ = signal_.reshape(signal_.size(0), groups, -1, *signal_.shape[2:])
    signal_fr = rfftn(signal_, dim=tuple(range(2, signal.ndim)))
    kernel_fr = rfftn(padded_kernel, dim=tuple(range(2, signal.ndim)))

    # multiply -1 for imag part = conjugate
    kernel_fr.imag *= -1
    output_fr = complex_matmul(signal_fr, kernel_fr, groups=groups)
    output = irfftn(output_fr, dim=tuple(range(2, signal.ndim)))

    # Remove extra padded values
    # crop_slices = [slice(0, output.size(0)), slice(0, output.size(1))] + [
    #     slice(0, (signal.size(i) - kernel.size(i) + 1), stride_[i - 2])
    #     for i in range(2, signal.ndim)]
    
    crop_slices = [slice(0, output.size(0)), slice(0, output.size(1))]
    for i in range(2, signal.ndim):
        if signal.size(i)%2 == 0:
            crop_slices.append(slice(0,(signal.size(i)-kernel.size(i)+2), stride_[i-2]))
        else:
            crop_slices.append(slice(0,(signal.size(i)-kernel.size(i)+1), stride_[i-2]))
    
    output = output[crop_slices].contiguous()

    # Optionally, add a bias term before returning.
    if bias is not None:
        bias_shape = tuple([1, -1] + (signal.ndim - 2) * [1])
        output += bias.view(bias_shape)

    return output


