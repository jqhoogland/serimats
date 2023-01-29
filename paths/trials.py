import itertools
import logging
import os
from dataclasses import dataclass, field
from os import PathLike
from pathlib import Path
from pprint import pp
from typing import (
    Any,
    Callable,
    Dict,
    Generator,
    Generic,
    Iterable,
    List,
    Literal,
    Optional,
    Tuple,
    Type,
    TypedDict,
    TypeVar,
    Union,
)

import numpy as np
import pandas as pd
import torch as t
import yaml
from matplotlib import pyplot as plt
from matplotlib.figure import Figure
from pyrsistent import T
from torch import nn, optim
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms

from serimats.paths.interventions import Intervention
from serimats.paths.models import FCN, ExtendedModule, Lenet5, ResNet
from serimats.paths.plots import plot_metric_scaling
from serimats.paths.utils import (
    CallableWithLatex,
    OptionalTuple,
    WithOptions,
    dict_to_latex,
    setup,
    stable_hash,
    to_tuple,
    tqdm,
    trange,
    var_to_latex,
)
from serimats.paths.weights import (
    AbsolutePerturbationInitializer,
    RelativePerturbationInitializer,
    WeightInitializer,
)


def get_opt_hyperparams(opt: optim.Optimizer) -> dict:
    """Assumes that the optimizer has only a single set of hyperparams
    (and not a separate set for each parameter group)"""
    param_group = opt.param_groups[0]

    hyperparams = {}

    for k, v in param_group.items():
        if k not in ["params", "foreach", "maximize", "capturable", "fused"]:
            hyperparams[k] = v

    return hyperparams


class Trial:
    control: Optional["Trial"] = None
    model: Optional[ExtendedModule] = None
    opt: Optional[optim.Optimizer] = None

    def __init__(
        self,
        model: WithOptions[Type[ExtendedModule]],
        opt: WithOptions[Type[optim.Optimizer]],
        train_loader: DataLoader,
        variations: Optional[List[List[Intervention]]] = None,
    ):
        self.model_cls, self.model_hyperparams = to_tuple(model, {})
        self.model = None

        self.opt_cls, self.opt_hyperparams = to_tuple(opt, {})
        self.opt = None

        self.train_loader = train_loader
        self.variations = variations or []

        self.logs = {}
        self.step = 0
        self.active = False

    def activate(self):
        self.active = True

        self.model = self.model or self.model_cls(**self.model_hyperparams)  # type: ignore
        self.opt = self.opt or self.opt_cls(self.model.parameters(), **self.opt_hyperparams)  # type: ignore

    def deactivate(self):
        self.active = False

        self.model = None
        self.opt = None

    def __enter__(self):
        self.activate()
        return self

    def __exit__(self, *args):
        self.deactivate()

    def loss(
        self, output: t.Tensor, target: t.Tensor, reduction: str = "mean"
    ) -> t.Tensor:
        return F.nll_loss(output, target, reduction=reduction)

    def run(self, n_epochs: int = 1, **kwargs):
        with self:
            for epoch_idx in range(n_epochs):
                yield epoch_idx, self.run_epoch(epoch_idx, **kwargs)

    def run_epoch(self, epoch_idx: int, reset: bool = False, **kwargs):
        for batch_idx, batch in enumerate(self.train_loader):
            step = epoch_idx * len(self.train_loader) + batch_idx

            if step < self.step and not reset:
                continue

            yield batch_idx, step, batch, self.run_batch(batch, **kwargs)

    def run_batch(self, batch: Tuple[t.Tensor, t.Tensor]):
        # assert self.active and self.model is not None and self.opt is not None, "Trial not active"

        x, y = batch

        self.opt.zero_grad()  # type: ignore
        output = self.model(x)  # type: ignore
        loss = self.loss(output, y)
        loss.backward()
        self.opt.step()  # type: ignore

        self.step += 1

        return loss.item()

    def log(self, step: Optional[int] = None, **kwargs):
        if step is None:
            step = self.step

        self.logs[step] = self.logs.get(step, {})
        self.logs[step].update(kwargs)

    @property
    def name(self):
        return f"{self.model.__class__.__name__}_{self.hash}"

    @property
    def hash(self):
        return stable_hash(self.extra_repr)

    @property
    def hyperparams(self):
        hyperparams = {
            **self.model.hyperparams,
            **get_opt_hyperparams(self.opt),
            **self.weight_initializer.hyperparams,
        }

        # Consistent ordering
        return {k: hyperparams[k] for k in sorted(hyperparams.keys())}

    @property
    def extra_repr(self):
        return ", ".join(f"{k}={v}" for k, v in self.hyperparams.items())

    def df(self, full=True):
        """DataFrame of logs (add on hyperparams as cols)"""
        df = pd.DataFrame.from_dict(self.logs, orient="index")
        # df["step"] = df.index
        df = df.rename_axis("step").reset_index()

        if full:
            hyperparams = self.hyperparams.copy()

            for k, v in hyperparams.items():
                if isinstance(v, (list, tuple)):
                    hyperparams[k] = [v] * len(df)

            df = df.assign(**hyperparams)

        return df

    @property
    def device(self):
        return self.model.device

    def __call__(self, *args, **kwargs) -> t.Tensor:
        return self.model(*args, **kwargs)

    def __getattribute__(self, __name: str) -> Any:
        """Forward attributes to model"""
        try:
            return super().__getattribute__(__name)
        except AttributeError:
            return getattr(self.model, __name)
