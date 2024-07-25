import time
from typing import Callable
from typing import Iterable
from typing import Sequence

import torch
from tqdm import tqdm


def count_parameters(model: torch.nn.Module, only_require_grad: bool = False) -> int:
    """return total number of trainable parameters"""
    count = 0
    for p in model.parameters():
        if only_require_grad:
            if p.requires_grad:
                count += p.numel()
        else:
            count += p.numel()
    return count


def normal_pdf(x: torch.Tensor, mean: torch.Tensor, var: torch.Tensor) -> torch.Tensor:
    """compute the normal pdf, standard broadcasting rules apply"""
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


def same_shape(shape1: Sequence, shape2: Sequence) -> bool:
    if len(shape1) != len(shape2):
        return False
    for i, _ in enumerate(shape1):
        if shape1[i] != shape2[i]:
            return False
    return True


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


class progress_bar:
    def __init__(self, iterable: Iterable, update_every: float = 0.25):
        """
        iterable: Iterable
        update_every: float, unit in seconds
        """
        self.iterable = iter(iterable)
        self.total_count = len(iterable)  # type:ignore
        self.curr_count = 0
        self.timetrack = time.time()
        self.update_timetrack = self.timetrack
        self.time_per_iter = 0
        self.bar_width = 50
        self.total_printed_len = 0
        self.update_every = update_every
        print("", end="\r")

    def update_bar(self):
        print(" " * self.total_printed_len, end="\r")
        done_bar = "=" * int(self.bar_width * self.curr_count / self.total_count)
        left_bar = "." * (self.bar_width - len(done_bar))
        spd = f"{1/self.time_per_iter:.2f}"
        _bar = f"[{done_bar}{left_bar}] [{self.curr_count}/{self.total_count}, {spd} iter/s]"
        self.total_printed_len = len(_bar)
        print(_bar, end="\r")

    def clear_bar(self):
        print(" " * self.total_printed_len, end="\r")

    def __iter__(self):
        return self

    def __next__(self):
        self.time_per_iter = time.time() - self.timetrack
        self.timetrack = time.time()
        if (self.timetrack - self.update_timetrack) > self.update_every:
            self.update_bar()
            self.update_timetrack = self.timetrack
        self.curr_count += 1
        try:
            item = next(self.iterable)
            return item
        except StopIteration as exc:
            self.clear_bar()
            raise StopIteration from exc


def get_progress_bar(iterable: Iterable, use_tqdm: bool = False):
    if use_tqdm:
        return tqdm(iterable, leave=False)
    return progress_bar(iterable)


if __name__ == "__main__":
    func1 = lambda x1, x2: torch.exp(-((x1 - x2) ** 2))

    z1 = torch.arange(0, 1, 0.1)
    z2 = torch.arange(0, 1, 0.5)

    k = get_kmatrix(z1, z2, func1)

    print(k.shape)
    print(k)
