import os
from typing import List

import matplotlib.axes
import matplotlib.pyplot as plt
import numpy as np
import torch

from lib.activations import NormaliseGaussian
from lib.gp_dist import GP_dist
from lib.utils import get_kmatrix
from lib.utils import normal_pdf


DEFAULT_NUM_OF_PTS = 10
Z_INIT_LOW = -2
Z_INIT_HIGH = 2
H_INIT_LOW = -5
H_INIT_HIGH = 5
GLOBAL_LENGTH_SCALE = 0.4
GLOBAL_COVARIANCE_SCALE = 1
GLOBAL_GITTER = 1e-1
BASELINE_GITTER = 1e-3
SQRT_2PI: float = np.sqrt(2 * np.pi)


class LayerFused(torch.nn.Module):
    """
    implements all neurons combined as tensors instead of python Lists, eg.

    output:       y1
                  |
                 (+)----------
                  |          |
                 (N1)       (N2)
                  |          |
    Input:        x1         x2
    """

    def __init__(
        self, input_size: int, output_size: int, num_gp_pts=DEFAULT_NUM_OF_PTS
    ) -> None:
        super().__init__()
        self.I = input_size  # input_size as I
        self.O = output_size  # output_size as O
        self.P = num_gp_pts  # P: Num of gp points per neuron
        self.num_neurons = input_size * output_size

        # initialise z: the neuron gp locations
        _z_single_neuron = torch.linspace(Z_INIT_LOW, Z_INIT_HIGH, self.P)
        _z = torch.zeros((self.I, self.O, self.P)) + _z_single_neuron.repeat(
            self.I, self.O, 1
        )
        self.z = torch.nn.Parameter(_z.clone(), requires_grad=True)  # (I, O, P)

        # initialise h: the neuron gp values
        self.h = torch.nn.Parameter(
            torch.rand((self.I, self.O, self.P)) * (H_INIT_HIGH - H_INIT_LOW)
            + H_INIT_LOW,
            requires_grad=True,
        )  # (I, O, P)

        # initialise gp hyperparameters
        self.l = torch.nn.Parameter(
            torch.ones((self.I, self.O)) * GLOBAL_LENGTH_SCALE, requires_grad=True
        )  # (I, O)
        self.s = torch.nn.Parameter(
            torch.ones((self.I, self.O)) * GLOBAL_COVARIANCE_SCALE, requires_grad=True
        )  # (I, O)
        self.jitter = torch.nn.Parameter(
            torch.ones((self.I, self.O)) * GLOBAL_GITTER, requires_grad=True
        )  # (I, O)

    def get_z(self):
        """always use this to access the inducing locations, to ensure consistent transformation applied to z"""
        return torch.tanh(self.z)

    def reset_gp_hyp(self):
        max_z = torch.max(self.get_z(), dim=-1).values
        min_z = torch.min(self.get_z(), dim=-1).values
        new_lengthscale = (max_z - min_z) / self.P
        self.l.copy_(new_lengthscale)  # (I, O)
        self.s.copy_(torch.ones((self.I, self.O)) * GLOBAL_COVARIANCE_SCALE)  # (I, O)
        self.jitter.copy_(torch.ones((self.I, self.O)) * GLOBAL_GITTER)  # (I, O)

    def forward(self, x: GP_dist) -> GP_dist:
        """
        input  x.mean & x.var shape: (N, I)
        output y.mean & y.var shape: (N, O)
        where N is the batch size
              I is the input_size
              O is the output_size
        """
        assert x.mean.dim() == 2
        assert x.mean.shape[1] == self.I

        N = x.mean.shape[0]
        I = self.I
        O = self.O
        P = self.P

        x_mean = x.mean.reshape(N, I, 1)  # (I, 1, 1)
        x_var = x.var.reshape(N, I, 1, 1)  # (I, 1, 1)

        s = self.s.reshape(1, I, O, 1)
        l = self.l.reshape(1, I, O, 1)
        jitter = self.jitter.reshape(1, I, O, 1, 1)
        z = self.get_z().reshape(1, I, O, P)  # (1, I, O, P)

        kernel_func1 = lambda x1, x2: normal_pdf(x1, x2, x_var + l**2)
        kernel_func2 = lambda x1, x2: normal_pdf(x1, x2, l**2)

        Q_hh = get_kmatrix(z, z, kernel_func2)  # (1, I, O, P, P)
        q_xh = get_kmatrix(
            x_mean.repeat(1, 1, self.O).reshape(N, I, O, 1), z, kernel_func1
        )  # (N, I, O, 1, P)
        Q_hh_noise = Q_hh + (
            (jitter**2 + BASELINE_GITTER)
            / (
                s.reshape(1, I, O, 1, 1) ** 2
                * torch.abs(l.reshape(1, I, O, 1, 1))
                * SQRT_2PI
            )
            + BASELINE_GITTER
        ) * torch.diag_embed(torch.ones(1, I, O, P), dim1=-2, dim2=-1)

        # L: (1, I, O, P, P)
        L = torch.linalg.cholesky(Q_hh_noise)  # pylint: disable=not-callable
        L_inv = torch.linalg.inv(L)  # pylint: disable=not-callable
        L_inv_T = torch.transpose(L_inv, -1, -2)
        Q_hh_inv = torch.linalg.matmul(L_inv_T, L_inv)  # pylint: disable=not-callable

        # mean
        # t1: (N, I, O, 1, P)
        t1 = torch.linalg.matmul(q_xh, Q_hh_inv)  # pylint: disable=not-callable
        h = self.h.reshape(1, I, O, P, 1)
        # t2: (N, I, O, 1, 1)
        t2 = torch.linalg.matmul(t1, h)  # pylint: disable=not-callable
        # out_mean: (N, O)
        out_mean = torch.sum(t2, 1).reshape(N, O)

        # variance
        # A: (N, I, O, 1, P)
        A = torch.linalg.matmul(q_xh, L_inv_T)  # pylint: disable=not-callable
        A_T = torch.transpose(A, -1, -2)  # (N, I, O, P, 1)
        t3 = (s**2) * (torch.abs(l) / torch.sqrt(l**2 + 2 * x_var))  # (N, I, O, 1)
        t4 = SQRT_2PI * (s**2) * torch.abs(l)  # (1, I, O, 1)
        # t5: (N, I, O, 1, 1)
        t5 = torch.linalg.matmul(A, A_T)  # pylint: disable=not-callable
        t6 = t5.reshape(N, I, O, 1)  # (N, I, O, 1)
        t7 = t3 - t4 * t6 + GLOBAL_GITTER  # (N, I, O, 1)
        # out_var: (N, O)
        out_var = torch.sum(t7, 1).reshape(N, O)

        return GP_dist(out_mean, out_var)

    def loglikelihood(self) -> torch.Tensor:
        """return log likelihood of each neuron's GP internal data points,
        summed across all neurons"""
        s = self.s.reshape(self.I, self.O, 1)
        l = self.l.reshape(self.I, self.O, 1)
        jitter = self.jitter.reshape(self.I, self.O, 1, 1)
        z = self.get_z()  # (I, O, P)
        covar_func = lambda x1, x2: s**2 * torch.exp(-((x1 - x2) ** 2) / (2 * l**2))

        K_hh = get_kmatrix(z, z, covar_func)  # (I, O, P, P)
        K_hh_noise = K_hh + (jitter**2 + BASELINE_GITTER) * torch.diag_embed(
            torch.ones(self.I, self.O, self.P), dim1=-2, dim2=-1
        )

        # L: (I, O, P, P)
        L = torch.linalg.cholesky(K_hh_noise)  # pylint: disable=not-callable
        L_inv = torch.linalg.inv(L)  # pylint: disable=not-callable
        L_inv_T = torch.transpose(L_inv, -1, -2)
        # A: (I, O, 1, P)
        h = self.h.reshape(self.I, self.O, 1, self.P)
        A = torch.linalg.matmul(h, L_inv_T)  # pylint: disable=not-callable
        A_T = torch.transpose(A, -1, -2)
        # t1: (I, O, 1)
        t1 = torch.log(torch.linalg.det(L))  # pylint: disable=not-callable
        # t2: (I, O, 1, 1)
        t2 = torch.linalg.matmul(A, A_T)  # pylint: disable=not-callable
        loglik = -0.5 * t2 - t1 - self.P * np.log(SQRT_2PI)  # (I, O, 1, 1)
        loglik_sum = torch.sum(loglik)  # (1)
        return loglik_sum

    def __gp_dist(self, x: torch.Tensor, I_idx: int, O_idx: int) -> GP_dist:
        """
        x: shape (1)
        return: GP_dist of mean and var
        """
        assert x.dim() == 1 and x.shape[0] == 1
        l = self.l[I_idx, O_idx]  # (1)
        s = self.s[I_idx, O_idx]  # (1)
        jitter = self.jitter[I_idx, O_idx]  # (1)
        z = self.get_z()[I_idx, O_idx].reshape(1, self.P)  # (1, P)
        h = self.h[I_idx, O_idx].reshape(1, self.P)  # (1, P)
        covar_func = lambda x1, x2: s**2 * torch.exp(-((x1 - x2) ** 2) / (2 * l**2))

        K_hh = get_kmatrix(z, z, covar_func)  # (1, P, P)
        K_hh_noise = K_hh + (jitter**2 + BASELINE_GITTER) * torch.diag_embed(
            torch.ones(1, self.P), dim1=-2, dim2=-1
        )
        k_xh = get_kmatrix(x.reshape(1, 1), z, covar_func)  # (1, 1, P)

        L = torch.linalg.cholesky(K_hh_noise)  # pylint: disable=not-callable
        L_inv = torch.linalg.inv(L)  # pylint: disable=not-callable
        L_inv_T = torch.transpose(L_inv, -1, -2)
        K_hh_inv = torch.linalg.matmul(L_inv_T, L_inv)  # pylint: disable=not-callable

        # mean
        # t1: (1, 1, P)
        t1 = torch.linalg.matmul(k_xh, K_hh_inv)  # pylint: disable=not-callable
        # t2: (1, 1, 1)
        h_reshaped = h.reshape(1, self.P, 1)
        t2 = torch.linalg.matmul(t1, h_reshaped)  # pylint: disable=not-callable
        mean = t2.reshape(1)

        # variance
        # A: (1, 1, P)
        A = torch.linalg.matmul(k_xh, L_inv_T)  # pylint: disable=not-callable
        A_T = torch.transpose(A, -1, -2)
        Kxx = covar_func(x, x)  # (1)
        t3 = Kxx - torch.linalg.matmul(A, A_T)  # pylint: disable=not-callable
        var = t3.reshape(1)

        return GP_dist(mean, var)

    def plot_neuron(self, axes: matplotlib.axes.Axes, I_idx: int, O_idx: int):
        """
        save a figure showing the (I_idx, O_idx)-th neuron's current GP
        """
        z = self.get_z()[I_idx, O_idx]  # (P)
        h = self.h[I_idx, O_idx]  # (P)
        NUM_PLT_PTS = 100
        x_pts = torch.linspace(
            torch.min(z) - 1, torch.max(z) + 1, NUM_PLT_PTS
        )  # (NUM_PLT_PTS)
        gp_dist_pts = [self.__gp_dist(x.reshape(1), I_idx, O_idx) for x in x_pts]

        mean = np.array([p.mean.detach().numpy() for p in gp_dist_pts]).reshape(-1)
        std_dev = np.array(
            [torch.sqrt(p.var).detach().numpy() for p in gp_dist_pts]
        ).reshape(-1)

        axes.plot(x_pts.numpy(), mean, color="black")
        axes.fill_between(
            x_pts.numpy(), mean + 2 * std_dev, mean - 2 * std_dev, color="gray"
        )

        axes.scatter(
            z.detach().numpy(), h.detach().numpy(), color="red"
        )  # pylint: disable=not-callable

    def save_fig(self, path: str):
        """save a figure showing the GP dist of all neurons"""

        fig, axes = plt.subplots(1, self.num_neurons, squeeze=False)

        axes_idx = 0
        for i_idx in range(self.I):
            for o_idx in range(self.O):
                self.plot_neuron(axes[0, axes_idx], i_idx, o_idx)
                axes_idx += 1

        fig.set_figwidth(20)
        fig.set_figheight(5)
        fig.savefig(path)

        plt.close(fig)

    def set_all_trainable(self):
        for p in self.parameters():
            p.requires_grad = True

    def freeze_all_params(self):
        for p in self.parameters():
            p.requires_grad = False

    def set_gp_pts_trainable(self):
        for p_name, p in self.named_parameters():
            if p_name in ["h", "z"]:
                p.requires_grad = True

    def set_gp_hyp_trainable(self):
        for p_name, p in self.named_parameters():
            if p_name in ["l", "s", "jitter"]:
                p.requires_grad = True

    def __repr__(self) -> str:
        return f"LayerFused(in={self.I} out={self.O} gp_pts_per_neuron={self.P})"


class simple_model(torch.nn.Module):
    def __init__(self, layout: List[int]) -> None:
        """
        initialise based on a provided layout.
        layout[0] must equal input size
        layout[-1] must equal output size
        """
        super().__init__()
        self.layout = layout
        self.layers: List[torch.nn.Module] = []

        # intermediate layers
        in_size = layout[0]
        for i, out_size in enumerate(layout[1:-1]):
            self.layers.append(LayerFused(in_size, out_size))
            self.add_module(f"layer_{i}", self.layers[-1])
            self.layers.append(NormaliseGaussian())
            self.add_module(f"normalise_{i}", self.layers[-1])
            in_size = out_size

        # final layer
        out_size = layout[-1]
        self.layers.append(LayerFused(in_size, out_size))
        self.add_module(f"layer_{len(layout)-2}", self.layers[-1])

    def forward(self, x: GP_dist) -> GP_dist:
        """
        run a provided input through the layers
        input   x.mean and x.var shape be (N, layer[0].I)
        output  y.mean and y.var shape be (N, layer[-1].O)
        """
        assert x.mean.dim() == 2
        assert x.mean.shape[1] == self.layout[0]

        intermediate = x
        for layer in self.layers:
            out_intermediate = layer.forward(intermediate)
            intermediate = out_intermediate
        return intermediate

    def predict(self, x: torch.Tensor) -> GP_dist:
        """
        run a provided deterministic input through the layers to do a prediction
        input   x shape be (N, layer[0].I)
        output  y.mean and y.out shape be (N, layer[-1].O)
        """
        x_mean = x
        x_var = 1e-6 * torch.ones_like(x)
        return self.forward(GP_dist(x_mean, x_var))

    def internal_loglik(self) -> torch.Tensor:
        """
        return the internal loglik summed across all neurons
        """
        total_loglik = torch.zeros(1)
        for layer in self.layers:
            if isinstance(layer, LayerFused):
                total_loglik += layer.loglikelihood()
        return total_loglik

    def reset_gp_hyp(self):
        for layer in self.layers:
            if isinstance(layer, LayerFused):
                layer.reset_gp_hyp()

    def set_all_trainable(self):
        for layer in self.layers:
            if isinstance(layer, LayerFused):
                layer.set_all_trainable()

    def freeze_all_params(self):
        for layer in self.layers:
            if isinstance(layer, LayerFused):
                layer.freeze_all_params()

    def set_gp_pts_trainable(self):
        for layer in self.layers:
            if isinstance(layer, LayerFused):
                layer.set_gp_pts_trainable()

    def set_gp_hyp_trainable(self):
        for layer in self.layers:
            if isinstance(layer, LayerFused):
                layer.set_gp_hyp_trainable()

    def save_fig(self, dirname="model_fig/"):
        """save graphs of the layers in the provided dir"""
        if not os.path.isdir(dirname):
            os.mkdir(dirname)
        for idx, layer in enumerate(self.layers):
            if isinstance(layer, LayerFused):
                fig_path = os.path.join(dirname, f"layer-{idx}.png")
                layer.save_fig(fig_path)
