import warnings
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
    Tuple,
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
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from serimats.paths.models import ExtendedModule
from serimats.paths.utils import tqdm


class WeightInitializer(ABC):  # TODO: Nake Generic
    """A class that, when called, jointly initializes a sequence of models"""

    initial_weights: Optional[Tuple[t.Tensor]]

    @abstractmethod
    def __call__(self, *models: nn.Module):
        raise NotImplementedError

    @property
    @abstractmethod
    def hyperparams(self) -> dict:
        raise NotImplementedError


@dataclass
class PerturbationInitializer(WeightInitializer, ABC):
    """
    Initialize weights normally (with Kaiming initialization) and then
    add a small perturbation to the weights.

    The tricky part is that the weights after the perturbation should
    still follow the same distribution as the weights before the perturbation.

    """

    seed_weights: int
    seed_perturbation: int
    epsilon: float = 0.0
    norm: Callable = t.norm
    initial_weights: Optional[Tuple[t.Tensor]] = None

    def initialize_weights(self, model: ExtendedModule):
        t.manual_seed(self.seed_weights)
        model.init_weights()

    @abstractmethod
    def apply_perturbation(self, model: ExtendedModule):
        raise NotImplementedError

    def __call__(self, model: ExtendedModule):
        self.initialize_weights(model)
        self.apply_perturbation(model)
        self.initial_weights = tuple(p.detach().clone() for p in model.parameters())

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


def get_householder_matrix(x: t.Tensor, y: t.Tensor) -> t.Tensor:
    """https://math.stackexchange.com/a/4524336/914272
    NOTE: This is very memory inefficient for large x and y.
    """

    warnings.warn("This function is very memory inefficient for large x and y.")

    nx = x / t.norm(x)
    ny = y / t.norm(y)
    c = (nx + ny).T

    return (2 * t.outer(c, c) / (c.T @ c)) - t.eye(x.shape[1])


def apply_householder_matrix_from_vertical_(x: t.Tensor, vs: t.Tensor):
    """Applies the above for the case that y = (0, ..., 0, 1)
    without having to compute the matrix explicitly.

    The Householder matrix maps a vector |x> onto |y>

    First, you normalize both vectors.

    Then you define |c> = |x> + |y>

    Then you define the Householder matrix:

    H = 2 |c><c| / (<c|c>) - I

    That is:

    H|v> = 2 |c><c|v> / <c|c> - |v>

    Parameters
    ----------
    x : t.Tensor shape (d,)
        Vector which defines the rotation. This is where (0, ..., 0, 1) is mapped.
    vs : t.Tensor shape (n, d)
        n Vectors to rotate

    """
    x_norm = t.norm(x)
    x /= x_norm
    x[-1] += 1

    vs -= (2 * t.inner(x, vs).view(-1, 1) / t.dot(x, x)) * x.view(1, -1)
    x[-1] -= 1
    x *= x_norm


def sample_from_hypersphere_intersection(
    r: t.Tensor,
    epsilon: float,
    n_samples: int,
):
    """Sample points from the intersection of two hyperspheres.

    Parameters
    ----------
    r : np.ndarray
        Vector that determines the radius of the larger hypersphere
    epsilon : float
        Radius of the smaller hypersphere, centered at r
    n_samples : int
        Number of samples to take

    Returns
    -------
    np.ndarray
        Array of shape (n_samples, r.shape[0])
    """

    d = r.shape[0]
    r_norm = t.norm(r)

    # Get the angle of the cone which goes through the center of the sphere and the intersection
    cone_angle = t.arccos(1 - epsilon**2 / (2 * r_norm**2))

    # Get the perp distance from r to the intersection
    epsilon_inner = r_norm * t.sin(cone_angle)

    # Sample a perturbation from the d-1 dimensional hypersphere of intersection
    perturbations = t.empty(n_samples, d)
    t.nn.init.normal_(perturbations)
    perturbations *= epsilon_inner / t.norm(perturbations[:, :-1], dim=1, keepdim=True)
    perturbations[:, -1] = 0

    # Apply the rotation
    apply_householder_matrix_from_vertical_(r, perturbations)

    # Shift the perturbations
    perturbations += r * t.cos(cone_angle)

    return perturbations


@dataclass
class AbsolutePerturbationInitializer(PerturbationInitializer):
    """Applies a perturbation with norm epsilon to each layer independently,
    while making sure that the norm of the weights remains the same."""

    def apply_perturbation(self, model: ExtendedModule):
        with t.no_grad():
            t.manual_seed(self.seed_perturbation)

            progress = tqdm(
                model.parameters(),
                desc="Perturbing weights",
            )

            for p in progress:
                tqdm.set_description(progress, f"Perturing ({p.shape})")

                p.data = sample_from_hypersphere_intersection(
                    p.data.view(-1),
                    self.epsilon,
                    n_samples=1,
                ).view_as(p)

    @property
    def perturbation(self) -> str:
        return "absolute"


@dataclass
class RelativePerturbationInitializer(PerturbationInitializer):
    """Applies a perturbation with norm epsilon to each layer independently,
    while making sure that the norm of the weights remains the same."""

    init_norm: Optional[float] = None
    delta_norm: Optional[float] = 0.0

    def __post_init__(self):
        if self.epsilon > 2.0:
            raise ValueError(f"epsilon must be less than 2.0, but got {self.epsilon}")
        elif self.epsilon < 0.0:
            raise ValueError(
                f"epsilon must be greater than 0.0, but got {self.epsilon}"
            )

    def apply_perturbation(self, model: ExtendedModule):
        with t.no_grad():
            t.manual_seed(self.seed_perturbation)
            epsilon = self.epsilon

            self.init_norm = model.norm().item()
            self.delta_norm = self.epsilon * self.init_norm

            if self.epsilon == 0.0:
                return
            if self.epsilon == 2.0:
                for p in model.parameters():
                    p.data = -p.data
                return

            for p in model.parameters():
                p_vec = p.data.view(-1)

                p.data = sample_from_hypersphere_intersection(
                    p_vec,
                    epsilon * t.norm(p_vec).item(),
                    n_samples=1,
                ).view_as(p)

    @property
    def perturbation(self) -> str:
        return "relative"


def test_seed_weights():
    from serimats.paths.models import FCN

    m0 = FCN(dict(n_hidden=1000))
    m1 = FCN(dict(n_hidden=1000))

    perturbation = RelativePerturbationInitializer(
        seed_weights=0,
        seed_perturbation=0,
        epsilon=0.1,
    )

    perturbation.initialize_weights(m0)
    perturbation.initialize_weights(m1)

    for p0, p1 in zip(m0.parameters(), m1.parameters()):
        assert t.allclose(p0, p1)


def test_seed_perturbation():
    from serimats.paths.models import FCN

    m0 = FCN(dict(n_hidden=1000))
    m1 = FCN(dict(n_hidden=1000))

    perturbation = RelativePerturbationInitializer(
        seed_weights=0,
        seed_perturbation=0,
        epsilon=0.1,
    )

    perturbation(m0)
    perturbation(m1)

    for p0, p1 in zip(m0.parameters(), m1.parameters()):
        assert t.allclose(p0, p1)


def test_relative_perturbation():
    from serimats.paths.models import FCN

    for epsilon in [0.0, 0.001, 0.01, 0.1, 0.5, 0.9, 1.0, 1.5, 2.0]:
        m0 = FCN(dict(n_hidden=1000))
        ms = [FCN(dict(n_hidden=1000)) for _ in range(10)]

        norms = []
        deltas = []

        for j in range(3):
            for i, m in enumerate(ms):
                perturbation = RelativePerturbationInitializer(
                    seed_weights=j,
                    seed_perturbation=i,
                    epsilon=epsilon,
                )
                perturbation.initialize_weights(m0)
                perturbation(m)

                m_norm = m.norm()
                m0_norm = m0.norm().item()

                distance = m.lp_distance(m0).item()

                norms.append(m0_norm)
                deltas.append(distance / m_norm)

                assert np.allclose(m_norm, m0_norm, atol=1e-2, rtol=1e-2)
                assert np.allclose(
                    distance,
                    (perturbation.epsilon * m0_norm),
                    atol=1e-2,
                    rtol=1e-2,
                )

        # plt.hist(norms, bins=100, alpha=0.5, label="norms")
        # plt.title(f"$|\\vec v|$ for $\\epsilon={epsilon}$")
        # plt.show()

        # plt.hist(deltas, bins=100, alpha=0.5, label="deltas")
        # plt.title(f"$|\\delta\\vec v|$ for $\\epsilon={epsilon}$")
        # plt.show()
