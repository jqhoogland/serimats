import math
import warnings
from abc import ABC, abstractmethod
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

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            stdv = 1.0 / math.sqrt((module.weight.size(1)))
            module.weight.data.uniform_(-stdv, stdv)

            if module.bias is not None:
                module.bias.data.uniform_(-stdv, stdv)

    def init_weights(self):
        self.apply(self._init_weights)

    @property
    def device(self) -> t.device:
        return next(self.parameters()).device


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
