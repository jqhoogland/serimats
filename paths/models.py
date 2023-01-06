import math
import warnings
from abc import ABC, abstractmethod
from collections import OrderedDict
from pathlib import Path
from typing import Callable, Optional, Tuple, TypedDict, Union

import torch as t
from torch import nn

from serimats.paths.utils import get_parameters


class ExtendedModule(nn.Module):
    hyperparams: dict

    def __init__(
        self,
        hyperparams: Optional[dict] = None,
    ):  
        self.hyperparams = hyperparams or {}
        super().__init__()
        # self.init_weights()

    @property
    def parameters_vector(self) -> t.Tensor:
        return get_parameters(self)

    @parameters_vector.setter
    def parameters_vector(self, value: t.Tensor):
        i = 0

        for p in self.parameters():
            p.data = value[i : i + p.numel()].view(p.shape)
            i += p.numel()

    @property
    def parameters_norm(self) -> t.Tensor:
        return t.norm(self.parameters_vector)

    @parameters_norm.setter
    def parameters_norm(self, value: t.Tensor):
        self.parameters_vector *= value / self.parameters_norm

    @staticmethod
    def _init_weights(module):
        if isinstance(module, nn.Linear):
            # Kaiming normal initialization (written by hand so we can constrain the norm of the matrix)
            fan_in, fan_out = nn.init._calculate_fan_in_and_fan_out(module.weight)
            n_params = fan_in * fan_out
            gain = nn.init.calculate_gain("relu")
            std = gain / math.sqrt(fan_in)

            # For a matrix whose elements are sampled from a normal distribution with mean 0 and standard deviation std,
            # the norm of the matrix is sqrt(fan_in) * std

            # Make sure the weights are perfectly normalized
            t.nn.init.normal_(module.weight.data)
            module.weight.data *= math.sqrt(n_params) * std / t.norm(module.weight.data)

            if module.bias is not None:
                # TODO: Not sure exactly how the bias should be initialized
                t.nn.init.normal_(module.bias.data)
                module.bias.data *= math.sqrt(fan_out) * std / t.norm(module.bias.data)

    def init_weights(self):
        self.apply(self._init_weights)

    @property
    def device(self) -> t.device:
        return next(self.parameters()).device

    @property
    def shape(self) -> OrderedDict:
        return OrderedDict((name, p.shape) for name, p in self.state_dict().items())


class MNISTHyperparams(TypedDict):
    n_hidden: Union[int, Tuple[int, ...]]


class MNIST(ExtendedModule):
    """A simple MNIST classifier with a variable number of hidden layers."""

    n_hidden: Union[int, Tuple[int, ...]]

    def __init__(self, hyperparams: Optional[MNISTHyperparams] = None, **kwargs):
        super().__init__(hyperparams=hyperparams, **kwargs)  # type: ignore

        self.n_hidden = n_hidden = self.hyperparams.get("n_hidden", 100)

        if isinstance(n_hidden, int):
            n_hidden = (784, n_hidden, 10)

        fc_layers = [
            nn.Linear(n_hidden[i], n_hidden[i + 1]) for i in range(len(n_hidden) - 1)
        ]
        activations = [nn.ReLU() for _ in range(len(n_hidden) - 1)]
        hidden_layers = [item for pair in zip(fc_layers, activations) for item in pair]

        self.model = nn.Sequential(
            nn.Flatten(),
            *hidden_layers[:-1],  # Skip the last ReLU
            nn.LogSoftmax(dim=1),
        )

    def forward(self, x: t.Tensor) -> t.Tensor:
        return self.model(x)
