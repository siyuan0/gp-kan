from typing import Tuple

import numpy as np
import torch

from lib.gp_dist import GP_dist
from lib.layer import LayerFused


class GP_conv2D(torch.nn.Module):
    """
    implement a conv2D, where the usual linear neurons are replaced by GP neurons
    for simplicity, input image's H and W must be fixed
    """

    def __init__(
        self,
        in_height: int,
        in_width: int,
        in_channel: int,
        out_channel: int,
        kernel_size: int | Tuple[int, int] = 3,
        dilation: int | Tuple[int, int] = 1,
        stride: int | Tuple[int, int] = 1,
        padding: int | Tuple[int, int] = 0,
    ) -> None:
        super().__init__()

        self.kernel_size = (
            (kernel_size, kernel_size) if isinstance(kernel_size, int) else kernel_size
        )
        self.padding = (padding, padding) if isinstance(padding, int) else padding
        self.stride = (stride, stride) if isinstance(stride, int) else stride
        self.dilation = (dilation, dilation) if isinstance(dilation, int) else dilation

        self.IC = in_channel
        self.OC = out_channel
        self.KH = self.kernel_size[0]
        self.KW = self.kernel_size[1]

        self.kernel_input_size = self.IC * self.KH * self.KW
        self.kernel_output_size = self.OC

        self.IH = in_height
        self.IW = in_width

        self.OH = int(
            np.floor(
                (self.IH + 2 * self.padding[0] - self.dilation[0] * (self.KH - 1) - 1)
                / self.stride[0]
                + 1
            )
        )
        self.OW = int(
            np.floor(
                (self.IW + 2 * self.padding[1] - self.dilation[1] * (self.KW - 1) - 1)
                / self.stride[1]
                + 1
            )
        )

        self.unfold = torch.nn.Unfold(
            kernel_size=self.kernel_size,
            dilation=self.dilation,
            padding=self.padding,
            stride=self.stride,
        )
        self.add_module("unfold", self.unfold)

        self.layerFused = LayerFused(
            self.kernel_input_size, self.kernel_output_size, num_gp_pts=5
        )
        self.add_module("internal", self.layerFused)

    def im2col(self, img: torch.Tensor) -> torch.Tensor:
        """
        convert a [N, IH, IW, IC] tensor into [N, L, IC * KH * KW] tensor
        where L depends on padding, stride, dilation
        """
        N = img.shape[0]
        LH = int(
            np.floor(
                (self.IH + 2 * self.padding[0] - self.dilation[0] * (self.KH - 1) - 1)
                / self.stride[0]
                + 1
            )
        )
        LW = int(
            np.floor(
                (self.IW + 2 * self.padding[1] - self.dilation[1] * (self.KW - 1) - 1)
                / self.stride[1]
                + 1
            )
        )
        L = LH * LW

        img_transpose_tmp = torch.transpose(img, dim0=3, dim1=1)  # (N, IC, IW, IH)
        img_transpose = torch.transpose(
            img_transpose_tmp, dim0=2, dim1=3
        )  # (N, IC, IH, IW)

        img_unfolded = self.unfold(img_transpose)  # (N, IC * KH * KW, L)

        t1 = torch.transpose(img_unfolded, dim0=1, dim1=2)  # (N, L, IC * KH * KW)
        assert t1.shape[0] == N
        assert t1.shape[1] == L
        assert t1.shape[2] == self.kernel_input_size

        return t1

    def col2im(self, tens: torch.Tensor) -> torch.Tensor:
        """
        convert a [N * L, OC] tensor into [N, OH, OW, OC] tensor
        where L = OH * OW
        """
        return tens.reshape(-1, self.OH, self.OW, self.OC)

    def forward(self, x: GP_dist) -> GP_dist:
        """
        for x.mean.shape = x.var.shape = [N, IH, IW, IC]
        return y.mean.shape = y.var.shape = [N, OH, OW, OC]
        """
        MAX_BATCHSIZE = 1000

        N = x.mean.shape[0]

        x_mean_unfolded = self.im2col(x.mean)  # (N, L, IC * KH * KW)
        x_var_unfolded = self.im2col(x.var)  # (N, L, IC * KH * KW)

        L = x_mean_unfolded.shape[1]

        n_step = max(int(np.floor(MAX_BATCHSIZE / L)), 1)  # always at least take 1 step

        out_means = []
        out_vars = []

        for i in range(int(np.ceil(N / n_step))):
            start_idx = i * n_step
            end_idx = min(N, start_idx + n_step)

            batch_mean = x_mean_unfolded[start_idx:end_idx].reshape(
                n_step * L, self.kernel_input_size
            )
            batch_var = x_var_unfolded[start_idx:end_idx].reshape(
                n_step * L, self.kernel_input_size
            )

            out = self.layerFused.forward(GP_dist(batch_mean, batch_var))
            out_means.append(self.col2im(out.mean))  # (n_step, OH, OW, OC)
            out_vars.append(self.col2im(out.var))  # (n_step, OH, OW, OC)

        out_mean = torch.concatenate(out_means, dim=0)  # (N, OH, OW, OC)
        out_var = torch.concatenate(out_vars, dim=0)  # (N, OH, OW, OC)

        return GP_dist(out_mean, out_var)
