from typing import Callable

import torch


class GP_dist:
    mean: torch.Tensor
    var: torch.Tensor

    def __init__(self, mean: torch.Tensor, var: torch.Tensor):
        assert mean.dim() == 1
        assert var.dim() == 1
        assert mean.shape[0] == var.shape[0]

        self.mean = mean
        self.var = var

    def __add__(self, other):
        mean = self.mean + other.mean
        var = self.var + other.var
        return GP_dist(mean, var)


def normal_pdf(x: torch.Tensor, mean: torch.Tensor, var: torch.Tensor) -> torch.Tensor:
    return (1 / torch.sqrt(2 * torch.pi * var)) * torch.exp(
        -0.5 * ((x - mean) ** 2) / var
    )


def log_likelihood(
    pred_mean: torch.Tensor, pred_var: torch.Tensor, true_val: torch.Tensor
) -> torch.Tensor:
    """return log likelihood of true_val given pred_mean and pred_var"""

    return (
        -0.5 * torch.log(2 * torch.pi * pred_var)
        - 0.5 * ((pred_mean - true_val) ** 2) / pred_var
    )


def get_kmatrix(
    x1: torch.Tensor,
    x2: torch.Tensor,
    func: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
) -> torch.Tensor:
    """
    func: callable that takes two similar shaped 1-D tensors and return a similar shaped 1-D tensor
          (assumed the func is applied elementwise across the two tensors)
    x1, x2: 1-D tensor
    return: 2-D tensor whose element[i,j] is func(x1[i], x2[j])
    """
    assert x1.dim() == 1
    assert x2.dim() == 1

    n1 = x1.shape[0]
    n2 = x2.shape[0]

    x1_repeat = x1.repeat(n2).reshape(n2, n1).transpose(0, 1).reshape(-1)
    x2_repeat = x2.repeat(n1)

    k_matrix = func(x1_repeat, x2_repeat)
    return k_matrix.reshape(n1, n2)


if __name__ == "__main__":
    func1 = lambda x1, x2: torch.exp(-((x1 - x2) ** 2))

    z1 = torch.arange(0, 1, 0.1)
    z2 = torch.arange(0, 1, 0.5)

    k = get_kmatrix(z1, z2, func1)

    print(k.shape)
    print(k)
