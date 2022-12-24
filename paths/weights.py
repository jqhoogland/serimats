from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import (Callable, Collection, Container, Iterable, List, Optional,
                    Sequence, Type)

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
class PerturbationInitializer(WeightInitializer, ABC):
    seed_weights: int
    seed_perturbation: int
    epsilon: float = 0.0
    norm: Callable = t.norm

    def initialize_weights(self, model: ExtendedModule):
        t.manual_seed(self.seed_weights)
        model.init_weights()

    @abstractmethod
    def apply_perturbation(self, model: ExtendedModule):
        raise NotImplementedError

    def __call__(self, model: ExtendedModule):
        self.initialize_weights(model)
        self.apply_perturbation(model)

    @property
    @abstractmethod
    def perturbation(self) -> str:
        raise NotImplementedError

    @property
    def hyperparams(self) -> dict:
        return {
            "perturbation": self.perturbation,
            "epsilon": self.epsilon,
            "seed_weights": self.seed_weights,
            "seed_perturbation": self.seed_perturbation,
        }


@dataclass
class AbsolutePerturbationInitializer(PerturbationInitializer):
    """Copies over the weights from the first model to all other models,
    then applies noise with norm epsilon to each of the other models."""
    
    def apply_perturbation(self, model: ExtendedModule):
        t.manual_seed(self.seed_perturbation)
        model.parameters_vector += t.randn_like(model.parameters_vector) * self.epsilon

    @property
    def perturbation(self) -> str:
        return "absolute"


@dataclass
class RelativePerturbationInitializer(PerturbationInitializer):
    """Copies over the weights from the first model to all other models,
    then applies noise with norm epsilon times the norm of the baseline model weights
     to each of the other models."""

    baseline: Optional[ExtendedModule] = None

    def apply_perturbation(self, model: ExtendedModule):
        t.manual_seed(self.seed_perturbation)
        model.parameters_vector += t.randn_like(model.parameters_vector) * self.epsilon * model.parameters_norm

    @property
    def perturbation(self) -> str:
        return "relative"
