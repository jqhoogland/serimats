from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Callable, Collection, Container, Iterable, Sequence, Type

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


class WeightInitializer(ABC):  # TODO: Nake Generic
    """A class that, when called, jointly initializes a sequence of models"""

    @abstractmethod
    def __call__(self, *models: nn.Module):
        raise NotImplementedError


@dataclass
class AbsolutePerturbationInitializer(WeightInitializer):
    """Copies over the weights from the first model to all other models,
    then applies noise with norm epsilon to each of the other models."""

    epsilon: float = 0.01
    norm: Callable = t.norm

    def __call__(self, *models: nn.Module):
        # Assumes baseline is passed as the first argument
        parameters = list(models[0].parameters())
        n_params = sum(p.numel() for p in parameters)

        for model in models[1:]:
            i = 0

            noise = t.randn(n_params)
            noise = noise / self.norm(noise) * self.epsilon

            for p, q in zip(parameters, model.parameters()):
                q.data = p.data.clone() + noise[i : i + p.numel()].view(p.shape)
                i += p.numel()


@dataclass
class RelativePerturbationInitializer(WeightInitializer):
    """Copies over the weights from the first model to all other models,
    then applies noise with norm epsilon times the norm of the baseline model weights
     to each of the other models."""

    epsilon: float = 0.01
    norm: Callable = t.norm

    def __call__(self, *models: nn.Module):
        # Assumes baseline is passed as the first argument
        total_epsilon = self.norm(models[0].parameters_vector) * self.epsilon
        absolute_perturber = AbsolutePerturbationInitializer(total_epsilon)

        return absolute_perturber(*models)
