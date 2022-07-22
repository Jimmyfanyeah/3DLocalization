import torch

from decode.neuralfitter.target_generator import UnifiedEmbeddingTarget


class Offset2Coordinate:
    """
    Convert sub-pixel pointers to absolute coordinates.
    """

    def __init__(self, xextent: tuple, yextent: tuple, img_shape: tuple):
        """
        Args:
            xextent (tuple): extent in x
            yextent (tuple): extent in y
            img_shape (tuple): image shape
        """

        off_psf = UnifiedEmbeddingTarget(xextent=xextent, yextent=yextent,
                                         img_shape=img_shape, roi_size=1)

        xv, yv = torch.meshgrid([off_psf._bin_ctr_x, off_psf._bin_ctr_y])
        self._x_mesh = xv.unsqueeze(0)
        self._y_mesh = yv.unsqueeze(0)

    def _subpx_to_absolute(self, x_offset, y_offset):
        """
        Convert subpixel pointers to absolute coordinates. Actual implementation

        Args:
            x_offset: N x H x W
            y_offset: N x H x W

        Returns:
        """
        batch_size = x_offset.size(0)
        x_coord = self._x_mesh[:,None].repeat(batch_size,2, 1, 1).to(x_offset.device) + x_offset
        y_coord = self._y_mesh[:,None].repeat(batch_size,2, 1, 1).to(y_offset.device) + y_offset
        return x_coord, y_coord

    @classmethod
    def parse(cls, param):
        return cls(param.TestSet.frame_extent[0],
                   param.TestSet.frame_extent[1],
                   param.TestSet.img_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward frames through post-processor.

        Args:
            x (torch.Tensor): features to be converted. Expecting x/y coordinates in channel index 2, 3.
             expected shape :math:`(N, C, H, W)`
        """

        if x.dim() != 4:
            raise ValueError("Wrong dimensionality. Needs to be N x C x H x W.")

        # p=0,1,2 phot=3,4,5 x=6,7,8 y=9,10,11 z=12,13,14 phot_sig=15,16,17 x_sig=18,19,20, y_sig=21,22,23, z_sig=24,25,26, bg=27
        # p=0,1 phot=2,3 x=4,5 y=6,7 z=8,9 phot_sig=10,11 x_sig=12,13, y_sig=14,15, z_sig=16,17, bg=18
        """Convert the channel values to coordinates"""
        x_coord, y_coord = self._subpx_to_absolute(x[:, 4:6], x[:, 6:8])

        output_converted = x.clone()
        output_converted[:, 4:6] = x_coord
        output_converted[:, 6:8] = y_coord

        return output_converted