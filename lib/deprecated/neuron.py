# """
# Class and definitions of a single neuron
# """
# import matplotlib.axes
# import matplotlib.pyplot as plt
# import numpy as np
# import torch
# import matplotlib
# from ..utils import get_kmatrix
# from ..gp_dist import GP_dist
# from ..utils import normal_pdf
# DEFAULT_NUM_OF_POINTS = 10
# GLOBAL_LENGTH_SCALE = 0.4
# GLOBAL_COVARIANCE_SCALE = 1
# GLOBAL_GITTER = 1e-1
# BASELINE_GITTER = 1e-3
# SQRT_2PI = np.sqrt(2 * np.pi)
# H_INIT_LOW = -5
# H_INIT_HIGH = 5
# Z_INIT_LOW = -5
# Z_INIT_HIGH = 5
# class GP_Neuron(torch.nn.Module):
#     def __init__(
#         self,
#         covar_lengthscale=GLOBAL_LENGTH_SCALE,
#         covar_scale=GLOBAL_COVARIANCE_SCALE,
#         num_of_points=DEFAULT_NUM_OF_POINTS,
#         requires_grad=True,
#         h_init_func=None
#     ) -> None:
#         super().__init__()
#         self.num_of_pts = num_of_points
#         self.z = torch.nn.Parameter(
#             # torch.rand(num_of_points, requires_grad=requires_grad)
#             # * (Z_INIT_HIGH - Z_INIT_LOW)
#             # + Z_INIT_LOW
#             torch.linspace(Z_INIT_LOW, Z_INIT_HIGH, num_of_points)
#         , requires_grad=True)
#         if h_init_func is None:
#             self.h = torch.nn.Parameter(
#                 torch.rand(num_of_points, requires_grad=requires_grad)
#                 * (H_INIT_HIGH - H_INIT_LOW) + H_INIT_LOW, requires_grad=True
#             )
#         else:
#             self.h = torch.nn.Parameter(
#                 h_init_func(self.z), requires_grad=True
#             )
#         self.l = torch.nn.Parameter(torch.ones(1) * covar_lengthscale)
#         self.s = torch.nn.Parameter(torch.ones(1) * covar_scale)
#         self.jitter = torch.nn.Parameter(torch.ones(1) * GLOBAL_GITTER)
#         # self.jitter = torch.ones(1) * GLOBAL_GITTER
#         # self.a = torch.nn.Parameter(torch.rand(1) * 2 - 1)  # -1 to 1
#         # self.b = torch.nn.Parameter(torch.rand(1) * 2 - 1)  # -1 to 1
#         self.a = 0
#         self.b = 0
#     def forward(self, x: GP_dist) -> GP_dist:
#         x_mean = x.mean
#         x_var = x.var
#         kernel_func1 = lambda x1, x2: normal_pdf(x1, x2, x_var + self.l**2)
#         kernel_func2 = lambda x1, x2: normal_pdf(x1, x2, torch.tensor([self.l**2]))
#         Q_hh = get_kmatrix(self.z, self.z, kernel_func2)
#         q_xh = get_kmatrix(x_mean, self.z, kernel_func1)
#         try:
#             L = torch.cholesky(
#                 Q_hh + ((self.jitter**2 + BASELINE_GITTER) / (self.s**2 * torch.abs(self.l) * \
# SQRT_2PI) + BASELINE_GITTER) * torch.eye(Q_hh.shape[0]) \
#             )  # type:ignore
#         except Exception as e:
#             print(Q_hh)
#             print(torch.linalg.eigvalsh(Q_hh))
#             raise e
#         L_inv = torch.inverse(L)  # type:ignore
#         Q_hh_inv = L_inv.T @ L_inv
#         out_mean = self.a * x_mean + self.b + q_xh @ Q_hh_inv @ (self.h - (self.a * self.z + self.b)).reshape(-1, 1)
#         A = q_xh @ L_inv.T
#         out_var = (
#             (self.s**2) * (torch.abs(self.l) / torch.sqrt(self.l**2 + 2 * x_var))
#             - np.sqrt(2 * np.pi) * (self.s**2) * torch.abs(self.l) * A @ A.T
#             + GLOBAL_GITTER
#         )
#         if out_var < 0:
#             print("out mean", out_mean)
#             print("out var", out_var)
#             print("Q_hh", Q_hh)
#             print("q_xh", q_xh)
#             print("l", self.l)
#             print("z", self.z)
#             print("input x mean, var", x_mean, x_var)
#             raise RuntimeError("Negative Variance Encountered")
#         return GP_dist(out_mean.reshape(1), out_var.reshape(1))
#     def covar_func(self, x1, x2):
#         return self.s**2 * torch.exp(-(x1-x2)**2 / (2 * self.l**2))
#     def loglikelihood(self):
#         """return the log likelihood of the internal data points"""
#         K_hh = get_kmatrix(self.z, self.z, self.covar_func) + \
#                (self.jitter**2 + BASELINE_GITTER) * torch.eye(self.z.shape[0])
#         L = torch.cholesky(K_hh)
#         L_inv = torch.inverse(L)
#         A = self.h.reshape(1, -1) @ L_inv.T
#         return -0.5 * A @ A.T - torch.log(torch.det(L)) - self.num_of_pts * np.log(SQRT_2PI)
#     def __gp_dist(self, x: torch.Tensor) -> GP_dist:
#         K_hh = get_kmatrix(self.z, self.z, self.covar_func) + \
#                (self.jitter**2 + BASELINE_GITTER) * torch.eye(self.z.shape[0])
#         k_xh = get_kmatrix(x, self.z, self.covar_func)
#         L = torch.cholesky(K_hh)
#         L_inv = torch.inverse(L)
#         K_hh_inv = L_inv.T @ L_inv
#         A = k_xh @ L_inv.T
#         mean = self.a * x + self.b + k_xh @ K_hh_inv @ (self.h - (self.a * self.z + self.b)).reshape(-1, 1)
#         var = self.covar_func(x, x) - A @ A.T
#         return GP_dist(mean.reshape(1), var.reshape(1))
#     def get_plot(self, axes : matplotlib.axes.Axes):
#         """
#         save a figure showing the neuron's current GP
#         """
#         x_pts = torch.linspace(torch.min(self.z) - 1, torch.max(self.z) + 1, 100)
#         gp_dist_pts = [self.__gp_dist(x.reshape(1)) for x in x_pts]
#         mean = np.array([p.mean.detach().numpy() for p in gp_dist_pts]).reshape(-1)
#         std_dev = np.array([torch.sqrt(p.var).detach().numpy() for p in gp_dist_pts]).reshape(-1)
#         axes.plot(x_pts.numpy(), mean, color="black")
#         axes.fill_between(x_pts.numpy(), mean + 2*std_dev, mean - 2*std_dev, color="gray")
#         axes.scatter(self.z.detach().numpy(), self.h.detach().numpy(), color="red") # pylint: disable=not-callable
# class Layer(torch.nn.Module):
#     """
#     A Layer consist of two groups of GP_neurons which operate in sequence.
#     Example when neurons_per_element = 1
#     output:       y1         y2
#                   |          |
#     Group 2:     (N3)       (N4)
#                   |          |
#                   |-----------
#     Intermediate: x3
#                   |-----------
#                   |          |
#     Group 1:     (N1)      (N2)
#                   |          |
#     Input:       x1          x2
#     Where x3 = sum of outputs of (N1) and (N2)
#     """
#     def __init__(self, input_size=1, neurons_per_element=2, output_size=1) -> None:
#         super().__init__()
#         self.input_size = input_size
#         self.output_size = output_size
#         self.neurons_per_element = neurons_per_element
#         self.group1_num_neurons = input_size * neurons_per_element
#         self.group2_num_neurons = output_size * neurons_per_element
#         self.group1_neurons = [GP_Neuron() for _ in range(self.group1_num_neurons)]
#         # f1 = lambda x: torch.sin(torch.pi * x)
#         # f2 = lambda x: x**2
#         # self.group1_neurons = [GP_Neuron(h_init_func=None), GP_Neuron(h_init_func=None)]
#         for i, neuron in enumerate(self.group1_neurons):
#             self.add_module(f"group1_{i}", neuron)
#         self.group2_neurons = [GP_Neuron() for _ in range(self.group2_num_neurons)]
#         # f3 = lambda x: torch.exp(x)
#         # self.group2_neurons = [GP_Neuron(h_init_func=None)]
#         for i, neuron in enumerate(self.group2_neurons):
#             self.add_module(f"group2_{i}", neuron)
#     def forward(self, x: GP_dist) -> GP_dist:
#         assert x.mean.shape[0] == self.input_size
#         # group 1
#         intermediate = [
#             GP_dist(torch.zeros(1), torch.zeros(1))
#             for _ in range(self.neurons_per_element)
#         ]
#         for x_idx in range(self.input_size):
#             x_single_neuron = GP_dist(x.mean[x_idx].reshape(1), x.var[x_idx].reshape(1))
#             for int_idx in range(self.neurons_per_element):
#                 output = self.group1_neurons[
#                     x_idx * self.neurons_per_element + int_idx
#                 ].forward(x_single_neuron)
#                 # print(f"input: {x_single_neuron}, output: {output}")
#                 intermediate[int_idx] = intermediate[int_idx] + output
#         # print(f"intermediate: {intermediate}")
#         # group 2
#         y = [GP_dist(torch.zeros(1), torch.zeros(1)) for _ in range(self.output_size)]
#         for int_idx in range(self.neurons_per_element):
#             for y_idx in range(self.output_size):
#                 output = self.group2_neurons[
#                     y_idx * self.neurons_per_element + int_idx
#                 ].forward(intermediate[int_idx])
#                 # print(f"input: {intermediate[int_idx]}, output: {output}")
#                 y[y_idx] = y[y_idx] + output
#         # exit()
#         # concat into single output
#         y_mean = torch.concatenate([element.mean for element in y])
#         y_var = torch.concatenate([element.var for element in y])
#         return GP_dist(y_mean, y_var)
#     def neuron_loglik(self):
#         """return the log likelihood of all neuron's internal points"""
#         total_loglik = torch.zeros(1)
#         for n in self.group1_neurons + self.group2_neurons:
#             total_loglik += n.loglikelihood().reshape(1)
#         return total_loglik
#     def predict(self, x: torch.Tensor) -> GP_dist:
#         # predict given a deterministic input
#         # a small noise is added for computational stability
#         x_mean = x
#         x_var = 1e-6 * torch.ones(x.shape[-1])
#         return self.forward(GP_dist(x_mean, x_var))
#     def set_all_trainable(self):
#         for p in self.parameters():
#             p.requires_grad = True
#     def freeze_all_params(self):
#         for p in self.parameters():
#             p.requires_grad = False
#     def set_gp_pts_trainable(self):
#         for _, mod in self.named_modules():
#             for p_name, p in mod.named_parameters():
#                 if p_name in ["h", "z"]:
#                     p.requires_grad = True
#     def set_gp_hyp_trainable(self):
#         for _, mod in self.named_modules():
#             for p_name, p in mod.named_parameters():
#                 if p_name in ["l", "s", "jitter"]:
#                     p.requires_grad = True
#     def save_fig(self, path:str):
#         """save a figure showing the GP dist of all neurons"""
#         fig, axes = plt.subplots(2, max(len(self.group1_neurons), len(self.group2_neurons)))
#         for i, n in enumerate(self.group1_neurons):
#             n.get_plot(axes[0, i])
#         for i, n in enumerate(self.group2_neurons):
#             n.get_plot(axes[1, i])
#         # for ax in axes.flat:
#         #     ax.label_outer()
#         fig.set_figwidth(20)
#         fig.set_figheight(5)
#         fig.savefig(path)
#         plt.close(fig)
