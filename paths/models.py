import math
import warnings
from abc import ABC, abstractmethod
from collections import OrderedDict
from functools import wraps
from pathlib import Path
from typing import (
    TYPE_CHECKING,
    Callable,
    Collection,
    Container,
    Iterable,
    List,
    Literal,
    Optional,
    Tuple,
    Type,
    TypedDict,
    Union,
)

import torch as t
from torch import nn
from torchvision.models.resnet import resnet18, resnet34, resnet50, resnet101, resnet152

from serimats.paths.constants import DEVICE
from serimats.paths.utils import get_parameters

if TYPE_CHECKING:
    from serimats.paths.weights import WeightInitializer


ParameterOrTensor = Union[nn.parameter.Parameter, t.Tensor]


def dot_parameters(
    p0s: Iterable[ParameterOrTensor], p1s: Iterable[ParameterOrTensor]
) -> t.Tensor:
    """Compute the dot product of two sets of parameters."""
    value = t.zeros(1, device=DEVICE)

    for p0, p1 in zip(p0s, p1s):
        if p0.shape != p1.shape:
            raise ValueError("Parameters have different shapes")

        value += p0 @ p1

    return value


def distance_bw_parameters(
    p0s: Iterable[ParameterOrTensor], p1s: Iterable[ParameterOrTensor], p="fro"
) -> t.Tensor:
    """Compute the distance between two sets of parameters."""
    value = t.zeros(1, device=DEVICE)

    for p0, p1 in zip(p0s, p1s):
        if p0.shape != p1.shape:
            raise ValueError("Parameters have different shapes")

        value += t.norm(p0 - p1, p=p) ** 2

    return value.sqrt()


def parameters_norm(ps: Iterable[ParameterOrTensor], p="fro") -> t.Tensor:
    """Compute the norm of a set of parameters."""
    value = t.zeros(1, device=DEVICE)

    for param in ps:
        value += t.norm(param, p=p) ** 2

    return value.sqrt()


def cosine_similarity_bw_parameters(
    p0s: Iterable[ParameterOrTensor], p1s: Iterable[ParameterOrTensor]
) -> t.Tensor:
    """Compute the cosine similarity between two sets of parameters."""
    return dot_parameters(p0s, p1s) / (parameters_norm(p0s) * parameters_norm(p1s))


def extract_parameters(
    model: Union["ExtendedModule", Iterable[ParameterOrTensor], "WeightInitializer"]
) -> Iterable[ParameterOrTensor]:
    """Extract the parameters of a model, weight initalizier, etc."""
    if hasattr(model, "model"):  # Learner
        model = model.model  # type: ignore
    if isinstance(model, ExtendedModule):
        return model.parameters()
    elif isinstance(model, WeightInitializer):
        if model.initial_weights is None:
            raise ValueError("Model has not been initialized")
        return model.initial_weights

    return model


ParameterExtractable = Union[
    "ExtendedModule", Iterable[ParameterOrTensor], "WeightInitializer"
]


def wrap_extractor(fn: Callable[..., t.Tensor]):
    """Wrap a function that takes parameters into a function that takes models."""

    @wraps(fn)
    def wrapper(
        self,
        other: ParameterExtractable,
        *args,
        **kwargs,
    ) -> t.Tensor:
        return fn(self, extract_parameters(other), *args, **kwargs)

    return wrapper


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
    def n_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters())

    @wrap_extractor
    def dot(self, other: Iterable[ParameterOrTensor]) -> t.Tensor:
        if isinstance(other, ExtendedModule):
            other = other.parameters()

        return dot_parameters(self.parameters(), other)

    def __matmul__(self, other: ParameterExtractable) -> t.Tensor:
        return self.dot(other)

    @wrap_extractor
    def lp_distance(
        self,
        other: Iterable[ParameterOrTensor],
        p: str = "fro",
    ) -> t.Tensor:
        return distance_bw_parameters(self.parameters(), other, p=p)

    def norm(self) -> t.Tensor:
        return parameters_norm(self.parameters())

    def cosine_similarity(self, other: Iterable[ParameterOrTensor]) -> t.Tensor:
        return cosine_similarity_bw_parameters(self.parameters(), other)

    @staticmethod
    def _init_weights(module):
        if isinstance(module, (nn.Linear, nn.Conv2d)):
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


class FCN(ExtendedModule):
    """A simple FCN classifier with a variable number of hidden layers."""

    n_hidden: Tuple[int, ...]
    n_units: Tuple[int, ...]

    def __init__(self, n_hidden: Union[int, Tuple[int]], n_in=784, n_out=10, **kwargs):
        if isinstance(n_hidden, int):
            n_hidden = (n_hidden,)

        hyperparams = {"n_hidden": n_hidden}

        super().__init__(hyperparams=hyperparams, **kwargs)  # type: ignore

        self.n_hidden = n_hidden
        self.n_units = n_units = (n_in, *n_hidden, n_out)

        fc_layers = [
            nn.Linear(n_units[i], n_units[i + 1]) for i in range(len(n_units) - 1)
        ]
        activations = [nn.ReLU() for _ in range(len(n_units) - 1)]
        hidden_layers = [item for pair in zip(fc_layers, activations) for item in pair]

        self.model = nn.Sequential(
            nn.Flatten(),
            *hidden_layers[:-1],  # Skip the last ReLU
            nn.LogSoftmax(dim=1),
        )

    def forward(self, x: t.Tensor) -> t.Tensor:
        return self.model(x)


class Lenet5(ExtendedModule):
    def __init__(self, **kwargs):
        super().__init__(hyperparams={}, **kwargs)  # type: ignore

        self.model = nn.Sequential(
            nn.Conv2d(1, 6, 5, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(6, 16, 5),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Flatten(),
            nn.Linear(16 * 5 * 5, 120),
            nn.ReLU(),
            nn.Linear(120, 84),
            nn.ReLU(),
            nn.Linear(84, 10),
            nn.LogSoftmax(dim=1),
        )

    def forward(self, x: t.Tensor) -> t.Tensor:
        return self.model(x)


class ResNet(ExtendedModule):
    def __init__(
        self,
        n_layers: Literal[18, 34, 50, 101, 152],
        **kwargs,
    ):
        self.n_layers = n_layers
        super().__init__(hyperparams={"n_layers": n_layers}, **kwargs)

        if n_layers == 18:
            self.model = resnet18(pretrained=False)
        elif n_layers == 34:
            self.model = resnet34(pretained=False)
        elif n_layers == 50:
            self.model = resnet50(pretrained=False)
        elif n_layers == 101:
            self.model = resnet101(pretrained=False)
        elif n_layers == 152:
            self.model = resnet152(pretrained=False)

    def forward(self, x: t.Tensor) -> t.Tensor:
        return self.model(x)
