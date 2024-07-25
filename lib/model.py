import pathlib
from typing import List

import torch

# noreorder
# pylint: disable=unused-import
from lib.activations import NormaliseGaussian
from lib.activations import ReshapeGaussian
from lib.conv import GP_conv2D
from lib.gp_dist import GP_dist
from lib.layer import LayerFused, HYP_CTX
from lib.utils import count_parameters


class GP_Model(torch.nn.Module):
    def __init__(
        self,
        layers: List[torch.nn.Module] | None = None,
        input_noise_log_init=-4,
        saveload_path="model.pt",
    ) -> None:
        """
        initialise based on a provided layers layout
        """
        super().__init__()
        self.input_noise_log_init = input_noise_log_init
        self.saveload_path = pathlib.PosixPath(saveload_path)
        if layers is not None:
            self.layers = layers
            self.__model_init()
        else:
            assert self.saveload_path.is_file()

        if self.saveload_path.is_file():
            print(f"found existing saved model, loading from {self.saveload_path}")
            model_and_params = torch.load(self.saveload_path)
            if "model" in model_and_params:
                model_layer_list = model_and_params.pop("model")
                if layers is None:
                    self.layers = self.__list_to_layers(model_layer_list)
                    self.__model_init()
                    print(">loaded model arch")
            if "hyp" in model_and_params:
                HYP_CTX.from_dict(model_and_params.pop("hyp"))
                print(">loaded hyp")

            if "input_noise_log" not in model_and_params:
                model_and_params["input_noise_log"] = self.input_noise_log
            self.load_state_dict(
                model_and_params
            )  # this will error if defined model differs from saved model

    def __model_init(self):
        # include a trainable input noise parameter
        tmp = self.input_noise_log_init * torch.ones(1)
        self.input_noise_log = torch.nn.Parameter(tmp.clone(), requires_grad=False)
        for i, layer in enumerate(self.layers):
            self.add_module(f"layer_{i}", layer)

    # pylint: disable-all
    def __list_to_layers(self, layer_infos: list) -> List[torch.nn.Module]:
        layer_list = []
        for layer_info in layer_infos:
            if "GP_conv2D" in layer_info:
                att = layer_info["GP_conv2D"]
                layer = GP_conv2D(
                    *att["IH_IW_IC_OC"],
                    kernel_size=att["kernel_size"],
                    stride=att["stride"],
                    num_gp_pts=att["num_gp_pts"],
                    use_double_layer=att["use_double_layer"],
                )
            elif "LayerFused" in layer_info:
                att = layer_info["LayerFused"]
                layer = LayerFused(att["in"], att["out"], att["num_gp_pts"])
            elif "NormaliseGaussian" in layer_info:
                layer = NormaliseGaussian()
            elif "ReshapeGaussian" in layer_info:
                layer = ReshapeGaussian(layer_info["ReshapeGaussian"])
            else:
                raise KeyError
            layer_list.append(layer)
        return layer_list

    def __layers_to_list(self, layers: List[torch.nn.Module]) -> List[dict]:
        layer_infos = []
        for layer in layers:
            if isinstance(layer, GP_conv2D):
                att = {}
                att["IH_IW_IC_OC"] = [layer.IH, layer.IW, layer.IC, layer.OC]
                att["kernel_size"] = layer.kernel_size
                att["num_gp_pts"] = layer.num_gp_pts
                att["stride"] = layer.stride
                att["use_double_layer"] = layer.use_double_layer
                layer_info = {"GP_conv2D": att}
                layer_infos.append(layer_info)
            elif isinstance(layer, LayerFused):
                att = {}
                att["in"] = layer.I
                att["out"] = layer.O
                att["num_gp_pts"] = layer.P
                layer_info = {"LayerFused": att}
                layer_infos.append(layer_info)
            elif isinstance(layer, NormaliseGaussian):
                layer_infos.append({"NormaliseGaussian": None})
            elif isinstance(layer, ReshapeGaussian):
                layer_infos.append({"ReshapeGaussian": layer.new_shape})
            else:
                raise TypeError
        return layer_infos

    # pylint: enable-all

    def print_model(self, only_require_grad: bool = False):
        print(self)
        print("parameter count: ", count_parameters(self, only_require_grad))

    def save_model(self):
        params = self.state_dict()
        model_and_params = {}
        model_and_params["model"] = self.__layers_to_list(self.layers)
        model_and_params["hyp"] = HYP_CTX.to_dict()
        model_and_params.update(params)
        torch.save(model_and_params, self.saveload_path)

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
