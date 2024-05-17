from typing import Callable

import torch


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
    assert x1.shape[0] == x2.shape[0]

    n = x1.shape[0]

    x1_repeat = x1.repeat(n).reshape(n, n).transpose(0, 1).reshape(-1)
    x2_repeat = x2.repeat(n)

    k_matrix = func(x1_repeat, x2_repeat)
    return k_matrix.reshape(n, n)


if __name__ == "__main__":
    func1 = lambda x1, x2: torch.exp(-((x1 - x2) ** 2))

    x = torch.arange(0, 1, 0.1)

    k = get_kmatrix(x, x, func1)

    print(k.shape)
    print(k)
