"""
Class and definitions of a single neuron
"""
# from typing import Tuple
# import numpy as np
import torch

DEFAULT_NUM_OF_POINTS = 5
GLOBAL_LENGTH_SCALE = 1
GLOBAL_COVARIANCE_SCALE = 1


class GP_Neuron:
    def __init__(
        self,
        covar_lengthscale=GLOBAL_LENGTH_SCALE,
        covar_scale=GLOBAL_COVARIANCE_SCALE,
        num_of_points=DEFAULT_NUM_OF_POINTS,
        requires_grad=True,
    ) -> None:
        self.h = torch.rand(num_of_points, requires_grad=requires_grad)
        self.z = torch.rand(num_of_points, requires_grad=requires_grad)
        self.l = covar_lengthscale
        self.s = covar_scale

    # def forward(self, x_mean, x_var) -> Tuple[torch.Tensor, torch.Tensor]:

    #     out_mean = np.sqrt(2 * np.pi) * (self.s**2)
