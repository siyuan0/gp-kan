import torch


class GP_dist:
    mean: torch.Tensor
    var: torch.Tensor

    def __init__(self, mean: torch.Tensor, var: torch.Tensor):
        assert mean.shape == var.shape

        self.mean = mean
        self.var = var

    def __add__(self, other):
        mean = self.mean + other.mean
        var = self.var + other.var
        return GP_dist(mean, var)

    def __repr__(self) -> str:
        return f"mean: {self.mean}, var: {self.var}"
