import time
import warnings
from functools import partial
from typing import Union, Callable

import torch
from tqdm import tqdm
import numpy as np

from decode.neuralfitter import dataset
from decode.generic import emitter
from decode.utils import hardware, frames_io
import decode.neuralfitter.utils


def ship_device(x, device: Union[str, torch.device]):
    """
    Ships the input to a pytorch compatible device (e.g. CUDA)

    Args:
        x:
        device:

    Returns:
        x

    """
    if x is None:
        return x

    elif isinstance(x, torch.Tensor):
        return x.to(device)

    elif isinstance(x, (tuple, list)):
        x = [ship_device(x_el, device) for x_el in x]  # a nice little recursion that worked at the first try
        return x

    elif device != 'cpu':
        raise NotImplementedError(f"Unsupported data type for shipping from host to CUDA device.")



class Infer:

    def __init__(self, model, ch_in: int, frame_proc, post_proc, device: Union[str, torch.device],
                 batch_size: Union[int, str] = 'auto', num_workers: int = 0,
                 pin_memory: bool = False, param = None,
                 forward_cat: Union[str, Callable] = 'emitter'):
        """
        Convenience class for inference.

        Args:
            model: pytorch model
            ch_in: number of input channels
            frame_proc: frame pre-processing pipeline
            post_proc: post-processing pipeline
            device: device where to run inference
            batch_size: batch-size or 'auto' if the batch size should be determined automatically (only use in combination with cuda)
            num_workers: number of workers
            pin_memory: pin memory in dataloader
            forward_cat: method which concatenates the output batches. Can be string or Callable.
            Use 'em' when the post-processor outputs an EmitterSet, or 'frames' when you don't use post-processing or if
            the post-processor outputs frames.
        """

        self.model = model
        self.ch_in = ch_in
        self.batch_size = batch_size
        self.device = device
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.frame_proc = frame_proc
        self.post_proc = post_proc

        self.forward_cat = None
        self._forward_cat_mode = forward_cat

        if str(self.device) == 'cpu' and self.batch_size == 'auto':
            warnings.warn(
                "Automatically determining the batch size does not make sense on cpu device. "
                "Falling back to reasonable value.")
            self.batch_size = 64

        self.loss = decode.neuralfitter.loss.GaussianMMLoss(
                xextent=param.Simulation.psf_extent[0],
                yextent=param.Simulation.psf_extent[1],
                img_shape=param.Simulation.img_size,
                device=device,
                chweight_stat=param.HyperParameter.chweight_stat)



    def forward(self, ds) -> emitter.EmitterSet:
        """
        Forward frames through model, pre- and post-processing and output EmitterSet

        Args:
            frames:

        """

        """Move Model"""
        model = self.model.to(self.device)
        model.eval()

        """Form Dataset and Dataloader"""
        # ds = dataset.InferenceDataset(frames=frames, frame_proc=self.frame_proc,
        #                               frame_window=self.ch_in)

        if self.batch_size == 'auto':
            # include safety factor of 20%
            # print(type(ds))
            # print(ds[0])
            bs = int(0.8 * self.get_max_batch_size(model, len(ds), 1, 512))
            print(bs)
        else:
            bs = self.batch_size

        # generate concatenate function here because we need batch size for this
        self.forward_cat = self._setup_forward_cat(self._forward_cat_mode, bs)

        dl = torch.utils.data.DataLoader(dataset=ds, batch_size=bs, shuffle=False, drop_last=False,
                                         num_workers=self.num_workers, pin_memory=self.pin_memory,
                                         collate_fn=decode.neuralfitter.utils.dataloader_customs.smlm_collate)

        out = []
        result = np.array(['loss_gmm','loss_bg'])

        with torch.no_grad():
            # for batch_num, sample in enumerate(tqdm(dl)):
            for batch_num, (x, y_tar, weight) in enumerate(tqdm(dl)):
                # x_in = sample.to(self.device)
                x, y_tar, weight = ship_device([x, y_tar, weight], self.device)

                # compute output
                y_out = model(x)

                # loss computation
                loss_val = self.loss(y_out, y_tar, weight)

                loss_gmm = loss_val[:, 0].mean().item()
                loss_bg = loss_val[:, 1].mean().item()
                # print(f'{loss_gmm},{loss_bg}')
                result = np.vstack((result, np.column_stack((loss_gmm, loss_bg))))

                """In post processing we need to make sure that we get a single Emitterset for each batch,
                so that we can easily concatenate."""
                if self.post_proc is not None:
                    out.append(self.post_proc.forward(y_out))
                else:
                    out.append(y_out.detach().cpu())

                """Cat to single emitterset / frame tensor depending on the specification of the forward_cat attr."""
            out = self.forward_cat(out)

        return out, result

    def _setup_forward_cat(self, forward_cat, batch_size: int):

        if forward_cat is None:
            return lambda x: x

        elif isinstance(forward_cat, str):

            if forward_cat == 'emitter':
                return partial(emitter.EmitterSet.cat, step_frame_ix=batch_size)

            elif forward_cat == 'frames':
                return partial(torch.cat, dim=0)

        elif callable(forward_cat):
            return forward_cat

        else:
            raise TypeError(f"Specified forward cat method was wrong.")

        raise ValueError(f"Unsupported forward_cat value.")

    @staticmethod
    def get_max_batch_size(model: torch.nn.Module, frame_size: Union[tuple, torch.Size],
                           limit_low: int, limit_high: int):
        """
        Get maximum batch size for inference.

        Args:
            model: model on correct device
            frame_size: size of frames (without batch dimension)
            limit_low: lower batch size limit
            limit_high: upper batch size limit
        """

        def model_forward_no_grad(x: torch.Tensor):
            """
            Helper function because we need to account for torch.no_grad()
            """
            with torch.no_grad():
                o = model.forward(x)

            return o

        assert next(model.parameters()).is_cuda, \
            "Auto determining the max batch size makes only sense when running on CUDA device."

        return hardware.get_max_batch_size(model_forward_no_grad, frame_size,
                                           next(model.parameters()).device,
                                           limit_low, limit_high)


class Infer12:

    def __init__(self, model, ch_in: int, frame_proc, post_proc, device: Union[str, torch.device],
                 batch_size: Union[int, str] = 'auto', num_workers: int = 0,
                 pin_memory: bool = False, forward_cat: Union[str, Callable] = 'emitter'):
        """
        Convenience class for inference.
        Args:
            model: pytorch model
            ch_in: number of input channels
            frame_proc: frame pre-processing pipeline
            post_proc: post-processing pipeline
            device: device where to run inference
            batch_size: batch-size or 'auto' if the batch size should be determined automatically (only use in combination with cuda)
            num_workers: number of workers
            pin_memory: pin memory in dataloader
            forward_cat: method which concatenates the output batches. Can be string or Callable.
            Use 'em' when the post-processor outputs an EmitterSet, or 'frames' when you don't use post-processing or if
            the post-processor outputs frames.
        """

        self.model = model
        self.ch_in = ch_in
        self.batch_size = batch_size
        self.device = device
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.frame_proc = frame_proc
        self.post_proc = post_proc

        self.forward_cat = None
        self._forward_cat_mode = forward_cat

        if str(self.device) == 'cpu' and self.batch_size == 'auto':
            warnings.warn(
                "Automatically determining the batch size does not make sense on cpu device. "
                "Falling back to reasonable value.")
            self.batch_size = 64

    def forward(self, ds) -> emitter.EmitterSet:
        """
        Forward frames through model, pre- and post-processing and output EmitterSet
        Args:
            frames:
        """

        """Move Model"""
        model = self.model.to(self.device)
        model.eval()

        """Form Dataset and Dataloader"""
        # ds = dataset.InferenceDataset(frames=frames, frame_proc=self.frame_proc,
        #                               frame_window=self.ch_in)

        if self.batch_size == 'auto':
            # include safety factor of 20%
            # print(type(ds))
            # print(ds[0])
            bs = int(0.8 * self.get_max_batch_size(model, len(ds), 1, 512))
            print(bs)
        else:
            bs = self.batch_size

        # generate concatenate function here because we need batch size for this
        self.forward_cat = self._setup_forward_cat(self._forward_cat_mode, bs)

        dl = torch.utils.data.DataLoader(dataset=ds, batch_size=bs, shuffle=False, drop_last=False,
                                         num_workers=self.num_workers, pin_memory=self.pin_memory,
                                         collate_fn=decode.neuralfitter.utils.dataloader_customs.smlm_collate)

        out = []

        with torch.no_grad():
            # for batch_num, sample in enumerate(tqdm(dl)):
            for batch_num, (sample, y_tar, weight) in enumerate(tqdm(dl)):
                x_in = sample.to(self.device)

                # compute output
                y_out = model(x_in)

                """In post processing we need to make sure that we get a single Emitterset for each batch,
                so that we can easily concatenate."""
                if self.post_proc is not None:
                    out.append(self.post_proc.forward(y_out))
                else:
                    out.append(y_out.detach().cpu())

        """Cat to single emitterset / frame tensor depending on the specification of the forward_cat attr."""

        out = self.forward_cat(out)

        return out

    @staticmethod
    def get_max_batch_size(model: torch.nn.Module, frame_size: Union[tuple, torch.Size],
                           limit_low: int, limit_high: int):
        """
        Get maximum batch size for inference.

        Args:
            model: model on correct device
            frame_size: size of frames (without batch dimension)
            limit_low: lower batch size limit
            limit_high: upper batch size limit
        """

        def model_forward_no_grad(x: torch.Tensor):
            """
            Helper function because we need to account for torch.no_grad()
            """
            with torch.no_grad():
                o = model.forward(x)

            return o

        assert next(model.parameters()).is_cuda, \
            "Auto determining the max batch size makes only sense when running on CUDA device."

        return hardware.get_max_batch_size(model_forward_no_grad, frame_size,
                                           next(model.parameters()).device,
                                           limit_low, limit_high)

    def _setup_forward_cat(self, forward_cat, batch_size: int):

        if forward_cat is None:
            return lambda x: x

        elif isinstance(forward_cat, str):

            if forward_cat == 'emitter':
                return partial(emitter.EmitterSet.cat, step_frame_ix=batch_size)

            elif forward_cat == 'frames':
                return partial(torch.cat, dim=0)

        elif callable(forward_cat):
            return forward_cat

        else:
            raise TypeError(f"Specified forward cat method was wrong.")

        raise ValueError(f"Unsupported forward_cat value.")


class LiveInfer(Infer):
    def __init__(self,
                 model, ch_in: int, *,
                 stream, time_wait=5, safety_buffer: int = 20,
                 frame_proc=None, post_proc=None,
                 device: Union[
                     str, torch.device] = 'cuda:0' if torch.cuda.is_available() else 'cpu',
                 batch_size: Union[int, str] = 'auto', num_workers: int = 0,
                 pin_memory: bool = False,
                 forward_cat: Union[str, Callable] = 'emitter'):
        """
        Inference from memmory mapped tensor, where the mapped file is possibly live being written to.

        Args:
            model: pytorch model
            ch_in: number of input channels
            stream: output stream. Will typically get emitters (along with starting and stopping index)
            time_wait: wait if length of mapped tensor has not changed
            safety_buffer: buffer distance to end of tensor to avoid conflicts when the file is actively being
            written to
            frame_proc: frame pre-processing pipeline
            post_proc: post-processing pipeline
            device: device where to run inference
            batch_size: batch-size or 'auto' if the batch size should be determined automatically (only use in combination with cuda)
            num_workers: number of workers
            pin_memory: pin memory in dataloader
            forward_cat: method which concatenates the output batches. Can be string or Callable.
            Use 'em' when the post-processor outputs an EmitterSet, or 'frames' when you don't use post-processing or if
            the post-processor outputs frames.
        """

        super().__init__(
            model=model, ch_in=ch_in, frame_proc=frame_proc, post_proc=post_proc,
            device=device, batch_size=batch_size, num_workers=num_workers, pin_memory=pin_memory,
            forward_cat=forward_cat)

        self._stream = stream
        self._time_wait = time_wait
        self._buffer_length = safety_buffer

    def forward(self, frames: Union[torch.Tensor, frames_io.TiffTensor]):

        n_fitted = 0
        n_waited = 0
        while n_waited <= 2:
            n = len(frames)

            if n_fitted == n - self._buffer_length:
                n_waited += 1
                time.sleep(self._time_wait)  # wait
                continue

            n_2fit = n - self._buffer_length
            out = super().forward(frames[n_fitted:n_2fit])
            self._stream(out, n_fitted, n_2fit)

            n_fitted = n_2fit
            n_waited = 0

        # fit remaining frames
        out = super().forward(frames[n_fitted:n])
        self._stream(out, n_fitted, n)
