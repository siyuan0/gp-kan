from typing import List

import numpy as np
import torch

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
