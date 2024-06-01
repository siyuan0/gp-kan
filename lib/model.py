import os
from typing import List

import torch

from lib.activations import NormaliseGaussian
from lib.conv import GP_conv2D
from lib.gp_dist import GP_dist
from lib.layer import LayerFused


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
        return self.forward(GP_dist.fromTensor(x))

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


class GP_Model(torch.nn.Module):
    def __init__(self, layers: List[torch.nn.Module]) -> None:
        """
        initialise based on a provided layers layout
        """
        super().__init__()
        self.layers = layers

        for i, layer in enumerate(self.layers):
            self.add_module(f"layer_{i}", layer)

    def forward(self, x: GP_dist) -> GP_dist:
        """
        run a provided input through the layers
        """
        intermediate = x
        for layer in self.layers:
            out_intermediate = layer.forward(intermediate)
            intermediate = out_intermediate
        return intermediate

    def predict(self, x: torch.Tensor) -> GP_dist:
        """
        run a provided deterministic input through the layers to do a prediction
        """
        return self.forward(GP_dist.fromTensor(x))

    def internal_loglik(self) -> torch.Tensor:
        """
        return the internal loglik summed across all neurons
        """
        total_loglik = torch.zeros(1)
        count = 0
        for layer in self.layers:
            if isinstance(layer, (LayerFused, GP_conv2D)):
                total_loglik += layer.loglikelihood()
                count += 1
        return total_loglik / count

    def reset_gp_hyp(self):
        for layer in self.layers:
            if isinstance(layer, (LayerFused, GP_conv2D)):
                layer.reset_gp_hyp()

    def set_all_trainable(self):
        for layer in self.layers:
            if isinstance(layer, (LayerFused, GP_conv2D)):
                layer.set_all_trainable()

    def freeze_all_params(self):
        for layer in self.layers:
            if isinstance(layer, (LayerFused, GP_conv2D)):
                layer.freeze_all_params()

    def set_gp_pts_trainable(self):
        for layer in self.layers:
            if isinstance(layer, (LayerFused, GP_conv2D)):
                layer.set_gp_pts_trainable()

    def set_gp_hyp_trainable(self):
        for layer in self.layers:
            if isinstance(layer, (LayerFused, GP_conv2D)):
                layer.set_gp_hyp_trainable()
