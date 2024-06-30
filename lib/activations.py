from typing import List
from typing import Sequence

import numpy as np
import torch
import torchvision

from lib.gp_dist import GP_dist

MIN_VAR = 0.2


class NormaliseGaussian(torch.nn.Module):
    def __init__(self, min_var=MIN_VAR):
        super().__init__()
        self.min_var = min_var

    @staticmethod
    def inverse_sigmoid(x: float):
        # Note: sigmoid(x) = 1/(1 + exp(-x))
        t1 = (1 / x) - 1
        t2 = -np.log(t1)
        return t2

    def forward(self, x: GP_dist) -> GP_dist:
        out_mean = torch.tanh(x.mean)
        sigmoid_offset = self.inverse_sigmoid(self.min_var)
        out_var = torch.sigmoid(x.var - x.mean**2 + sigmoid_offset)
        return GP_dist(out_mean, out_var)


class ReshapeGaussian(torch.nn.Module):
    def __init__(self, new_shape: List[int]):
        super().__init__()
        self.new_shape = new_shape

    def forward(self, x: GP_dist) -> GP_dist:
        out_mean = torch.reshape(x.mean, self.new_shape)
        out_var = torch.reshape(x.var, self.new_shape)
        return GP_dist(out_mean, out_var)


class ResizeGaussian(torch.nn.Module):
    """DON'T USE THIS, DOESN'T PRESERVE GAUSSIAN DIST"""

    def __init__(
        self,
        new_size: List[int],
        mode: torchvision.transforms.InterpolationMode = torchvision.transforms.InterpolationMode.BILINEAR,
    ):
        super().__init__()
        self.resize = torchvision.transforms.Resize(size=new_size, interpolation=mode)

    def forward(self, x: GP_dist) -> GP_dist:
        # resize on N, H, W, C
        in_mean = torch.transpose(torch.transpose(x.mean, -1, -3), -1, -2)  # N, C, H, W
        in_var = torch.transpose(torch.transpose(x.var, -1, -3), -1, -2)
        out_mean_t = self.resize(in_mean)
        out_var_t = self.resize(in_var)
        out_mean = torch.transpose(torch.transpose(out_mean_t, -1, -3), -2, -3)
        out_var = torch.transpose(torch.transpose(out_var_t, -1, -3), -2, -3)
        return GP_dist(out_mean, out_var)


class AvgPool2DGaussian(torch.nn.Module):
    """
    This implements an image average pool 2D while
    maintaining the Gaussian Distribution of the output.
    Implemented via a convolution kernel
    """

    def __init__(
        self, kernel_size: int | Sequence[int] = 3, stride: int | Sequence[int] = 3
    ):
        super().__init__()

        if isinstance(kernel_size, int):
            self.kernel_size = [kernel_size, kernel_size]
        else:
            self.kernel_size = kernel_size

        if isinstance(stride, int):
            self.stride = [stride, stride]
        else:
            self.stride = stride

        self.kernel_mean = torch.ones(self.kernel_size).reshape(
            1, 1, *self.kernel_size
        ) / (self.kernel_size[0] * self.kernel_size[1])
        self.kernel_var = (
            self.kernel_mean**2
        )  # variance is scaled by square of the variable scaling

    def forward(self, x: GP_dist) -> GP_dist:
        # resize on N, H, W, C
        in_mean = torch.transpose(torch.transpose(x.mean, -1, -3), -1, -2)  # N, C, H, W
        in_var = torch.transpose(torch.transpose(x.var, -1, -3), -1, -2)  # N, C, H, W
        in_batch_dims = in_mean.shape[:-3]
        in_C = in_mean.shape[-3]
        in_H = in_mean.shape[-2]
        in_W = in_mean.shape[-1]

        in_mean_flatten = in_mean.reshape(-1, 1, in_H, in_W)  # N * C, 1, H, W
        in_var_flatten = in_var.reshape(-1, 1, in_H, in_W)  # N * C, 1, H, W

        out_mean_flatten = torch.conv2d(
            in_mean_flatten, self.kernel_mean, stride=self.stride
        )
        out_var_flatten = torch.conv2d(
            in_var_flatten, self.kernel_var, stride=self.stride
        )

        out_H = out_mean_flatten.shape[-2]
        out_W = out_mean_flatten.shape[-1]

        out_mean_t = out_mean_flatten.reshape(*in_batch_dims, in_C, out_H, out_W)
        out_var_t = out_var_flatten.reshape(*in_batch_dims, in_C, out_H, out_W)

        out_mean = torch.transpose(torch.transpose(out_mean_t, -1, -3), -2, -3)
        out_var = torch.transpose(torch.transpose(out_var_t, -1, -3), -2, -3)
        return GP_dist(out_mean, out_var)


class DownSample2x2Gaussian(torch.nn.Module):
    """
    This implements an image scale down on the x and y axis by factor of 2, while
    maintaining the Gaussian Distribution of the output.
    Implemented via a 2x2 convolution kernel of
    |0.25 0.25|
    |0.25 0.25|
    and stride=2
    """

    def __init__(self):
        super().__init__()
        self.kernel_mean = torch.tensor([[0.25, 0.25], [0.25, 0.25]]).reshape(
            1, 1, 2, 2
        )
        self.kernel_var = (
            self.kernel_mean**2
        )  # variance is scaled by square of the variable scaling

    def forward(self, x: GP_dist) -> GP_dist:
        # resize on N, H, W, C
        in_mean = torch.transpose(torch.transpose(x.mean, -1, -3), -1, -2)  # N, C, H, W
        in_var = torch.transpose(torch.transpose(x.var, -1, -3), -1, -2)  # N, C, H, W
        in_batch_dims = in_mean.shape[:-3]
        in_C = in_mean.shape[-3]
        in_H = in_mean.shape[-2]
        in_W = in_mean.shape[-1]

        assert in_H % 2 == 0
        assert in_W % 2 == 0

        in_mean_flatten = in_mean.reshape(-1, 1, in_H, in_W)  # N * C, 1, H, W
        in_var_flatten = in_var.reshape(-1, 1, in_H, in_W)  # N * C, 1, H, W

        out_mean_flatten = torch.conv2d(in_mean_flatten, self.kernel_mean, stride=2)
        out_var_flatten = torch.conv2d(in_var_flatten, self.kernel_var, stride=2)

        out_mean_t = out_mean_flatten.reshape(
            *in_batch_dims, in_C, int(in_H / 2), int(in_W / 2)
        )
        out_var_t = out_var_flatten.reshape(
            *in_batch_dims, in_C, int(in_H / 2), int(in_W / 2)
        )

        out_mean = torch.transpose(torch.transpose(out_mean_t, -1, -3), -2, -3)
        out_var = torch.transpose(torch.transpose(out_var_t, -1, -3), -2, -3)
        return GP_dist(out_mean, out_var)


class ReduceSumGaussian(torch.nn.Module):
    def __init__(self, dim: int, keep_dim=False):
        super().__init__()
        self.dim = dim
        self.keep_dim = keep_dim

    def forward(self, x: GP_dist) -> GP_dist:
        out_mean = torch.sum(x.mean, dim=self.dim, keepdim=self.keep_dim)
        out_var = torch.sum(x.var, dim=self.dim, keepdim=self.keep_dim)
        return GP_dist(out_mean, out_var)


class SoftmaxGaussian(torch.nn.Module):
    def forward(self, x: GP_dist) -> GP_dist:
        """
        no good current analytical way, do this via MC sampling
        and second moment matching. Input of shape (*, K) where K
        is the number of categories
        """
        NUM_OF_SAMPLES = 10000
        extra_dims = x.mean.shape[:-1]
        K = x.mean.shape[-1]
        z = torch.randn([*extra_dims, K, NUM_OF_SAMPLES])
        sample = z * x.var.unsqueeze(-1) ** 0.5 + x.mean.unsqueeze(-1)

        softmax_out = torch.softmax(sample, dim=-2)

        out_mean = torch.mean(softmax_out, dim=-1)
        out_var = torch.var(softmax_out, dim=-1)

        return GP_dist(out_mean, out_var)
