import torch

from lib.neuron import GP_Neuron
from lib.utils import GP_dist


class Layer(torch.nn.Module):
    """
    A Layer consist of two groups of GP_neurons which operate in sequence.
    Example when neurons_per_element = 1

    output:       y1         y2
                  |          |
    Group 2:     (N3)       (N4)
                  |          |
                  |-----------
    Intermediate: x3
                  |-----------
                  |          |
    Group 1:     (N1)      (N2)
                  |          |
    Input:       x1          x2

    Where x3 = sum of outputs of (N1) and (N2)

    """

    def __init__(self, input_size=1, neurons_per_element=2, output_size=1) -> None:
        super().__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.neurons_per_element = neurons_per_element
        self.group1_num_neurons = input_size * neurons_per_element
        self.group2_num_neurons = output_size * neurons_per_element

        self.group1_neurons = [GP_Neuron() for _ in range(self.group1_num_neurons)]
        for i, neuron in enumerate(self.group1_neurons):
            self.add_module(f"group1_{i}", neuron)

        self.group2_neurons = [GP_Neuron() for _ in range(self.group2_num_neurons)]
        for i, neuron in enumerate(self.group2_neurons):
            self.add_module(f"group2_{i}", neuron)

    def forward(self, x: GP_dist) -> GP_dist:
        assert x.mean.shape[0] == self.input_size

        # group 1
        intermediate = [
            GP_dist(torch.zeros(1), torch.zeros(1))
            for _ in range(self.neurons_per_element)
        ]
        for x_idx in range(self.input_size):
            x_single_neuron = GP_dist(x.mean[x_idx].reshape(1), x.var[x_idx].reshape(1))

            for int_idx in range(self.neurons_per_element):
                output = self.group1_neurons[
                    x_idx * self.neurons_per_element + int_idx
                ].forward(x_single_neuron)
                intermediate[int_idx] = intermediate[int_idx] + output

        # group 2
        y = [GP_dist(torch.zeros(1), torch.zeros(1)) for _ in range(self.output_size)]
        for int_idx in range(self.neurons_per_element):
            for y_idx in range(self.output_size):
                output = self.group2_neurons[
                    y_idx * self.neurons_per_element + int_idx
                ].forward(intermediate[int_idx])
                y[y_idx] = y[y_idx] + output

        # concat into single output
        y_mean = torch.concatenate([element.mean for element in y])
        y_var = torch.concatenate([element.var for element in y])

        return GP_dist(y_mean, y_var)

    def predict(self, x: torch.Tensor) -> GP_dist:
        # predict given a deterministic input
        # a small noise is added for computational stability
        x_mean = x
        x_var = 1e-6 * torch.ones(x.shape[-1])

        return self.forward(GP_dist(x_mean, x_var))

    def set_all_trainable(self):
        for p in self.parameters():
            p.requires_grad = True

    def freeze_all_params(self):
        for p in self.parameters():
            p.requires_grad = False

    def set_gp_pts_trainable(self):
        for _, mod in self.named_modules():
            for p_name, p in mod.named_parameters():
                if p_name in ["h", "z"]:
                    p.requires_grad = True
