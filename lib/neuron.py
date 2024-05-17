"""
Class and definitions of a single neuron
"""
import numpy as np
import torch

from .utils import get_kmatrix
from .utils import GP_dist
from .utils import normal_pdf

DEFAULT_NUM_OF_POINTS = 5
GLOBAL_LENGTH_SCALE = 1
GLOBAL_COVARIANCE_SCALE = 1
GLOBAL_GITTER = 1e-6

H_INIT_LOW = -1
H_INIT_HIGH = 10
Z_INIT_LOW = -2
Z_INIT_HIGH = 2


class GP_Neuron(torch.nn.Module):
    def __init__(
        self,
        covar_lengthscale=GLOBAL_LENGTH_SCALE,
        covar_scale=GLOBAL_COVARIANCE_SCALE,
        num_of_points=DEFAULT_NUM_OF_POINTS,
        requires_grad=True,
    ) -> None:
        super().__init__()
        self.h = torch.nn.Parameter(
            torch.rand(num_of_points, requires_grad=requires_grad)
            * (H_INIT_HIGH - H_INIT_LOW)
            + H_INIT_LOW
        )
        self.z = torch.nn.Parameter(
            torch.rand(num_of_points, requires_grad=requires_grad)
            * (Z_INIT_HIGH - Z_INIT_LOW)
            + Z_INIT_LOW
        )
        self.l = torch.nn.Parameter(torch.rand(1) * covar_lengthscale)
        self.s = torch.nn.Parameter(torch.rand(1) * covar_scale)
        self.a = torch.nn.Parameter(torch.rand(1) * 2 - 1)  # -1 to 1
        self.b = torch.nn.Parameter(torch.rand(1) * 2 - 1)  # -1 to 1

    def forward(self, x: GP_dist) -> GP_dist:
        x_mean = x.mean
        x_var = x.var

        kernel_func1 = lambda x1, x2: normal_pdf(x1, x2, x_var + self.l**2)
        kernel_func2 = lambda x1, x2: normal_pdf(x1, x2, torch.tensor([self.l**2]))

        Q_hh = get_kmatrix(self.z, self.z, kernel_func2)
        q_xh = get_kmatrix(x_mean, self.z, kernel_func1)
        L = torch.cholesky(
            Q_hh + GLOBAL_GITTER * torch.eye(Q_hh.shape[0])
        )  # type:ignore
        L_inv = torch.inverse(L)  # type:ignore
        Q_hh_inv = L_inv.T @ L_inv

        out_mean = self.a * x_mean + self.b + q_xh @ Q_hh_inv @ self.h.reshape(-1, 1)

        A = q_xh @ L_inv.T
        out_var = (
            (self.s**2) * (self.l / torch.sqrt(self.l**2 + 2 * x_var))
            - np.sqrt(2 * np.pi) * (self.s**2) * self.l * A @ A.T
            + GLOBAL_GITTER
        )

        if out_var < 0:
            print("out mean", out_mean)
            print("out var", out_var)
            print("Q_hh", Q_hh)
            print("q_xh", q_xh)
            print("l", self.l)
            print("z", self.z)
            print("input x mean, var", x_mean, x_var)
            raise RuntimeError("Negative Variance Encountered")
        return GP_dist(out_mean.reshape(1), out_var.reshape(1))

    # def update(self, lr=0.001):
    #     with torch.no_grad():
    #         for param in [self.h, self.z, self.a, self.b]:
    #             if param.grad is not None:
    #                 param -= param.grad * lr
    #                 # param.copy_(new_param) # type:ignore
    #                 # param.grad.zero_()
