from typing import List

import torch

from lib.conv import GP_conv2D
from lib.gp_dist import GP_dist
from lib.layer import LayerFused


class GP_Model(torch.nn.Module):
    def __init__(self, layers: List[torch.nn.Module], input_noise_log_init=-2) -> None:
        """
        initialise based on a provided layers layout
        """
        super().__init__()
        self.layers = layers

        # include a trainable input noise parameter
        tmp = input_noise_log_init * torch.ones(1)
        self.input_noise_log = torch.nn.Parameter(tmp.clone(), requires_grad=True)

        for i, layer in enumerate(self.layers):
            self.add_module(f"layer_{i}", layer)

    def get_input_noise(self):
        return torch.exp(self.input_noise_log)

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
        noise = self.get_input_noise() ** 2
        return self.forward(GP_dist(x, noise * torch.ones_like(x)))

    def internal_loglik(self) -> torch.Tensor:
        """
        return the internal loglik summed across all neurons
        """
        for p in self.parameters():
            device = p.device
        total_loglik = torch.zeros(1).to(device)
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
