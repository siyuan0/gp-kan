import matplotlib.axes
import matplotlib.pyplot as plt
import numpy as np
import torch

from lib.gp_dist import GP_dist
from lib.utils import get_kmatrix
from lib.utils import normal_pdf

DEFAULT_NUM_OF_PTS = 10
Z_INIT_LOW = -2
Z_INIT_HIGH = 2
H_INIT_LOW = -1
H_INIT_HIGH = 1
GLOBAL_LENGTH_SCALE = 0.4
MIN_LENGTH_SCALE = 0.2
GLOBAL_COVARIANCE_SCALE = 1
MIN_COVARIANCE_SCALE = 0.1
SQRT_2PI: float = np.sqrt(2 * np.pi)


class HYP_CTX:
    """hyperparameters that can be set"""

    GLOBAL_JITTER = 1e-1
    BASELINE_JITTER = 1e-2


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
            (
                torch.rand((self.I, self.O, self.P)) * (H_INIT_HIGH - H_INIT_LOW)
                + H_INIT_LOW
            ),
            requires_grad=True,
        )  # (I, O, P)

        # initialise gp hyperparameters
        self.l = torch.nn.Parameter(
            torch.ones((self.I, self.O)), requires_grad=True
        )  # (I, O)
        self.s = torch.nn.Parameter(
            torch.ones((self.I, self.O)), requires_grad=True
        )  # (I, O)
        self.jitter = torch.nn.Parameter(
            torch.ones((self.I, self.O)), requires_grad=True
        )  # (I, O)

        self.reset_gp_hyp()

    def set_jitter(self, jitter):
        self.jitter.copy_(torch.log(jitter).detach())

    def get_jitter(self):
        """always use this to access jitter, to ensure consistent transformation applied"""
        return torch.exp(self.jitter) + HYP_CTX.BASELINE_JITTER

    def set_s(self, s):
        self.s.copy_(torch.log(s).detach())

    def get_s(self):
        """always use this to access s, to ensure consistent transformation applied"""
        return torch.exp(self.s) + MIN_COVARIANCE_SCALE

    def set_l(self, l):
        self.l.copy_(torch.log(l).detach())

    def get_l(self):
        """always use this to access l, to ensure consistent transformation applied"""
        return torch.exp(self.l) + MIN_LENGTH_SCALE

    def get_z(self):
        """always use this to access z, to ensure consistent transformation applied"""
        return torch.tanh(self.z)

    def reset_gp_hyp(self):
        max_z = torch.max(self.get_z(), dim=-1).values
        min_z = torch.min(self.get_z(), dim=-1).values
        new_lengthscale = (max_z - min_z) / self.P
        with torch.no_grad():
            self.set_l(new_lengthscale)  # (I, O)
            self.set_s(torch.ones((self.I, self.O)) * GLOBAL_COVARIANCE_SCALE)  # (I, O)
            self.set_jitter(
                torch.ones((self.I, self.O)) * HYP_CTX.GLOBAL_JITTER
            )  # (I, O)

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

        s = self.get_s().reshape(1, I, O, 1)
        l = self.get_l().reshape(1, I, O, 1)
        jitter = self.get_jitter().reshape(1, I, O, 1, 1)
        z = self.get_z().reshape(1, I, O, P)  # (1, I, O, P)

        kernel_func1 = lambda x1, x2: normal_pdf(x1, x2, x_var + l**2)
        kernel_func2 = lambda x1, x2: normal_pdf(x1, x2, l**2)

        Q_hh = get_kmatrix(z, z, kernel_func2)  # (1, I, O, P, P)
        if torch.any(torch.isnan(Q_hh)):
            print(l)
            print(s)
            print(jitter)
        q_xh = get_kmatrix(
            x_mean.repeat(1, 1, self.O).reshape(N, I, O, 1), z, kernel_func1
        )  # (N, I, O, 1, P)
        Q_hh_noise = Q_hh + (
            (jitter**2)
            / (
                s.reshape(1, I, O, 1, 1) ** 2
                * torch.abs(l.reshape(1, I, O, 1, 1))
                * SQRT_2PI
            )
        ) * torch.diag_embed(torch.ones(1, I, O, P), dim1=-2, dim2=-1).to(Q_hh.device)

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
        t7 = t3 - t4 * t6 + HYP_CTX.GLOBAL_JITTER  # (N, I, O, 1)
        # out_var: (N, O)
        out_var = torch.sum(t7, 1).reshape(N, O)

        return GP_dist(out_mean, out_var)

    def loglikelihood(self) -> torch.Tensor:
        """return avg log likelihood of each neuron's GP internal data points,
        across all neurons"""
        s = self.get_s().reshape(self.I, self.O, 1)
        l = self.get_l().reshape(self.I, self.O, 1)
        jitter = self.get_jitter().reshape(self.I, self.O, 1, 1)
        z = self.get_z()  # (I, O, P)
        covar_func = lambda x1, x2: s**2 * torch.exp(-((x1 - x2) ** 2) / (2 * l**2))

        K_hh = get_kmatrix(z, z, covar_func)  # (I, O, P, P)
        K_hh_noise = K_hh + (jitter**2) * torch.diag_embed(
            torch.ones(self.I, self.O, self.P), dim1=-2, dim2=-1
        ).to(K_hh.device)

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
        return loglik_sum / (self.I * self.P)

    def __gp_dist(self, x: torch.Tensor, I_idx: int, O_idx: int) -> GP_dist:
        """
        x: shape (1)
        return: GP_dist of mean and var
        """
        assert x.dim() == 1 and x.shape[0] == 1
        device = x.device
        l = self.get_l()[I_idx, O_idx]  # (1)
        s = self.get_s()[I_idx, O_idx]  # (1)
        jitter = self.get_jitter()[I_idx, O_idx]  # (1)
        z = self.get_z()[I_idx, O_idx].reshape(1, self.P)  # (1, P)
        h = self.h[I_idx, O_idx].reshape(1, self.P)  # (1, P)
        covar_func = lambda x1, x2: s**2 * torch.exp(-((x1 - x2) ** 2) / (2 * l**2))

        K_hh = get_kmatrix(z, z, covar_func)  # (1, P, P)
        K_hh_noise = K_hh + (jitter**2) * torch.diag_embed(
            torch.ones(1, self.P), dim1=-2, dim2=-1
        ).to(device)
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
        x_pts = torch.linspace(torch.min(z) - 0.1, torch.max(z) + 0.1, NUM_PLT_PTS).to(
            self.h.device
        )  # (NUM_PLT_PTS)
        gp_dist_pts = [self.__gp_dist(x.reshape(1), I_idx, O_idx) for x in x_pts]

        mean = np.array([p.mean.cpu().detach().numpy() for p in gp_dist_pts]).reshape(
            -1
        )
        std_dev = np.array(
            [torch.sqrt(p.var).cpu().detach().numpy() for p in gp_dist_pts]
        ).reshape(-1)

        axes.plot(x_pts.cpu().numpy(), mean, color="black")
        axes.fill_between(
            x_pts.cpu().numpy(), mean + 2 * std_dev, mean - 2 * std_dev, color="gray"
        )

        axes.scatter(
            z.cpu().detach().numpy(), h.cpu().detach().numpy(), color="red"
        )  # pylint: disable=not-callable

    def save_fig(self, path: str):
        """save a figure showing the GP dist of all neurons"""
        MAX_NEURONS_SHOWN = 5
        plot_num = min(self.num_neurons, MAX_NEURONS_SHOWN)

        fig, axes = plt.subplots(1, plot_num, squeeze=False)

        axes_idx = 0
        for i_idx in range(self.I):
            for o_idx in range(self.O):
                if axes_idx < plot_num:
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
