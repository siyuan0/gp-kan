from typing import Callable

import torch


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
    func: callable that takes two similar shaped tensors and return a similarly shaped tensor
          (assumed the func is only applied elementwise across the two tensors)
    x1, x2: tensor of shape (..., N1) and (..., N2)
    return: tensor of shape (..., N1, N2) where element[..., i, j] is func(x1[..., i], x2[..., j])
    """
    N1 = x1.shape[-1]
    N2 = x2.shape[-1]

    assert x2.shape[0:-1] == x1.shape[0:-1]

    extra_dim = x1.shape[0:-1]
    ones_extra_dim = [1 for _ in extra_dim]

    x1_repeat = (
        x1.repeat([*ones_extra_dim, N2])
        .reshape(*extra_dim, N2, N1)
        .transpose(-2, -1)
        .reshape(*extra_dim, N1 * N2)
    )
    x2_repeat = x2.repeat([*ones_extra_dim, N1])

    k_matrix = func(x1_repeat, x2_repeat)
    return k_matrix.reshape(*extra_dim, N1, N2)


if __name__ == "__main__":
    func1 = lambda x1, x2: torch.exp(-((x1 - x2) ** 2))

    z1 = torch.arange(0, 1, 0.1)
    z2 = torch.arange(0, 1, 0.5)

    k = get_kmatrix(z1, z2, func1)

    print(k.shape)
    print(k)
