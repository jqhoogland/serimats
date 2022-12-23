from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import (
    Callable,
    Collection,
    Container,
    Iterable,
    List,
    Optional,
    Sequence,
    Type,
)

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy
import torch as t
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import wandb
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from serimats.paths.models import ExtendedModule


class WeightInitializer(ABC):  # TODO: Nake Generic
    """A class that, when called, jointly initializes a sequence of models"""

    @abstractmethod
    def __call__(self, *models: nn.Module):
        raise NotImplementedError

    @property
    @abstractmethod
    def hyperparams(self) -> dict:
        raise NotImplementedError


@dataclass
class AbsolutePerturbationInitializer(WeightInitializer):
    """Copies over the weights from the first model to all other models,
    then applies noise with norm epsilon to each of the other models."""

    epsilon: float = 0.01
    norm: Callable = t.norm
    baseline: Optional[List[ExtendedModule]] = None

    def __call__(self, *models: ExtendedModule):
        # The first time this is called, we need to store the baseline weights
        if self.baseline is None:
            self.baseline = models[0]
            models = models[1:]

        parameters = list(self.baseline.parameters())
        n_params = sum(p.numel() for p in parameters)

        for model in models[1:]:
            i = 0

            noise = t.randn(n_params)
            noise = noise / self.norm(noise) * self.epsilon

            for p, q in zip(parameters, model.parameters()):
                q.data = p.data.clone() + noise[i : i + p.numel()].view(p.shape)
                i += p.numel()

    @property
    def hyperparams(self) -> dict:
        return {
            "weight_initialization": "absolute_perturbation",
            "epsilon": self.epsilon,
        }


@dataclass
class RelativePerturbationInitializer(WeightInitializer):
    """Copies over the weights from the first model to all other models,
    then applies noise with norm epsilon times the norm of the baseline model weights
     to each of the other models."""

    epsilon: float = 0.01
    norm: Callable = t.norm
    baseline: Optional[ExtendedModule] = None

    def __call__(self, *models: ExtendedModule):
        # The first time this is called, we need to store the baseline weights
        if self.baseline is None:
            self.baseline = models[0]

        total_epsilon = (self.baseline.parameters_norm * self.epsilon).item()
        absolute_perturber = AbsolutePerturbationInitializer(total_epsilon)

        return absolute_perturber(*models)

    @property
    def hyperparams(self) -> dict:
        return {
            "weight_initialization": "relative_perturbation",
            "epsilon": self.epsilon,
        }
