# %%
import itertools
import logging
import os
from dataclasses import dataclass, field
from os import PathLike
from pathlib import Path
from pprint import pp
from typing import (Any, Callable, Dict, Generator, Generic, Iterable, List,
                    Literal, Optional, Tuple, Type, TypedDict, TypeVar, Union)

import numpy as np
import pandas as pd
import torch as t
import yaml
from matplotlib import pyplot as plt
from matplotlib.figure import Figure
from torch import nn, optim
from torch.nn import functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from tqdm import tqdm

from serimats.paths.metrics import Metrics
from serimats.paths.models import MNIST, ExtendedModule
from serimats.paths.plots import (corr_scaling, cos_sim_scaling,
                                  epsilon_scaling, loss_scaling,
                                  plot_metric_scaling)
from serimats.paths.utils import (OptionalTuple, dict_to_latex, setup,
                                  stable_hash, to_tuple, var_to_latex)
from serimats.paths.weights import (AbsolutePerturbationInitializer,
                                    RelativePerturbationInitializer)

setup()

def get_opt_hyperparams(opt: optim.Optimizer) -> dict:
    """Assumes that the optimizer has only a single set of hyperparams
    (and not a separate set for each parameter group)"""
    param_group = opt.param_groups[0]

    hyperparams = {}

    for k, v in param_group.items():
        if k not in ["params", "foreach", "maximize", "capturable", "fused"]:
            hyperparams[k] = v

    return hyperparams


class Learner:
    #: The model we're comparing against (important for certain metrics)
    baseline: "Learner"  

    def __init__(
        self,
        model: ExtendedModule,
        opt: optim.SGD,
        weight_initializer: Callable[[ExtendedModule], None],
        dir: PathLike = Path("results"),
    ):
        self.model = model
        self.opt = opt
        self.weight_initializer = weight_initializer
        self.weight_initializer(self.model)

        self.baseline = self

        self.dir = Path(dir)
        self.logs = {}
        self.step = 0

    @classmethod
    def from_hyperparams(
        cls,
        model_cls: Type[MNIST],
        opt_cls: Type[optim.SGD],
        # Model
        n_hidden: Union[int, List[int]] = 100,
        # Weight initialization
        seed_weights: int = 42,
        seed_perturbation: int = 0,
        epsilon: float = 0.0,
        perturbation: Literal["relative", "absolute"] = "relative",
        # Optimizer
        lr: float = 0.01,
        momentum: float = 0.0,
        nesterov: bool = False,
        weight_decay: float = 0.0,
        **kwargs,
    ):
        model = model_cls(dict(n_hidden=n_hidden))
        opt = opt_cls(
            model.parameters(),
            lr=lr,
            momentum=momentum,
            nesterov=nesterov,
            weight_decay=weight_decay,
        )  # type: ignore

        weight_initializer_cls = (
            RelativePerturbationInitializer
            if perturbation == "relative"
            else AbsolutePerturbationInitializer
        )

        weight_initializer = weight_initializer_cls(
            seed_weights, seed_perturbation, epsilon
        )

        return cls(
            model=model, opt=opt, weight_initializer=weight_initializer, **kwargs
        )

    def loss(
        self, output: t.Tensor, target: t.Tensor, reduction: str = "mean"
    ) -> t.Tensor:
        return F.nll_loss(output, target, reduction=reduction)

    def train_batch(
        self, epoch: int, batch_idx: int, step: int, batch: Tuple[t.Tensor, t.Tensor]
    ):
        x, y = batch

        self.opt.zero_grad()
        output = self.model(x)
        loss = self.loss(output, y)
        loss.backward()
        self.opt.step()
        self.step = step

        return loss.item()

    def log(self, step: Optional[int] = None, **kwargs):
        if step is None:
            step = self.step

        self.logs[step] = self.logs.get(step, {})
        self.logs[step].update(kwargs)

    def save(self, overwrite: bool = False):
        self.path.mkdir(parents=True, exist_ok=True)
        self.path_to_step.mkdir(parents=True, exist_ok=True)

        if not overwrite and (self.path_to_step / "model.pt").exists():
            # raise FileExistsError(self.path_to_step)
            logging.info(f"Step {self.step} already exists, skipping")
            return False

        self.df(full=False).to_csv(self.path / "logs.csv")
        t.save(self.model.state_dict(), self.path_to_step / "model.pt")
        t.save(self.opt.state_dict(), self.path_to_step / "opt.pt")

        yaml.dump(self.hyperparams, open(self.path / "hyperparams.yaml", "w"))

        logging.info(
            f"Saved model to {self.path_to_step} (step={self.step}, {self.extra_repr})"
        )

        return True

    def load(self, step: Optional[int] = None):
        try:
            logs_df = pd.read_csv(self.path / "logs.csv")

            if step is None:
                self.step = logs_df["step"].max().item()
            else:
                self.step = step

            logs_df = logs_df.set_index("step")
            self.logs = logs_df.to_dict(orient="index")

        except FileNotFoundError:
            logging.info(f"Could not load logs from {self.path / 'logs.csv'}")
            logging.info(yaml.dump(self.hyperparams))
            self.logs = {}
            return False

        try:
            self.model.load_state_dict(t.load(self.path_to_step / "model.pt"))
            self.opt.load_state_dict(t.load(self.path_to_step / "opt.pt"))

            logging.info(
                f"Loaded model from {self.path_to_step} (step={self.step}, {self.extra_repr})"
            )
        except FileNotFoundError:
            logging.info(f"Could not load model or optimizer from {self.path_to_step}")
            return False

        return True

    @property
    def path_to_step(self):
        return self.path / ("step-" + str(self.step))

    @property
    def path(self):
        return self.dir / self.name

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
            df = df.assign(**self.hyperparams)

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

class Ensemble:
    def __init__(
        self,
        model_cls: Type[ExtendedModule],
        opt_cls: Type[optim.Optimizer],
        train_dl: DataLoader,
        test_dl: DataLoader,
        seed_dl: int,
        dir: str,
        logging_ivl: int = 100,
        # Optimizer
        momentum: OptionalTuple[float] = 0.0,
        lr: OptionalTuple[float] = 0.01,
        nesterov: OptionalTuple[bool] = False,
        weight_decay: OptionalTuple[float] = 0.0,
        # Model
        n_hidden: OptionalTuple[Union[int, List[int]]] = 100,
        # Weight initialization
        seed_weights: OptionalTuple[int] = 0,
        seed_perturbation: OptionalTuple[int] = 0,
        epsilon: OptionalTuple[float] = 0.01,
        perturbation: OptionalTuple[Literal["absolute", "relative"]] = "relative",
        plot_fns: Optional[OptionalTuple[Callable[..., Tuple[Figure, plt.Axes]]]] = None,
        plot_ivl: int = 5000,
        save_ivl: int = 2000,
    ):
        self.model_cls = model_cls
        self.opt_cls = opt_cls
        self.seed_dl = seed_dl
        self.dir = Path(dir)
        self.logging_ivl = logging_ivl

        self.train_dl = train_dl
        self.test_dl = test_dl

        # Turn hyperparams into tuples
        self.momentum = to_tuple(momentum)
        self.lr = to_tuple(lr)
        self.nesterov = to_tuple(nesterov)
        self.weight_decay = to_tuple(weight_decay)
        self.n_hidden = to_tuple(n_hidden)
        self.seed_weights = to_tuple(seed_weights)
        self.seed_perturbation = to_tuple(seed_perturbation)
        self.epsilon = to_tuple(epsilon)
        self.perturbation = to_tuple(perturbation)

        # Compute the cartesian product of all hyperparams
        # (Except for the product of epsilon = 0.0 and multiple seed_perturbation because these are redundant)
        epsilon_without_0 = [e for e in self.epsilon if e != 0.0]
        epsilon_with_0 = (0.0,) if 0.0 in self.epsilon else ()

        self.hyperparams = list(
            itertools.product(
                self.momentum,
                self.lr,
                self.nesterov,
                self.weight_decay,
                self.n_hidden,
                self.seed_weights,
                (0,),  # No seed_perturbation if epsilon=0.0
                epsilon_with_0,
                self.perturbation,
            )
        ) + list(
            itertools.product(
                self.momentum,
                self.lr,
                self.nesterov,
                self.weight_decay,
                self.n_hidden,
                self.seed_weights,
                self.seed_perturbation,
                epsilon_without_0,
                self.perturbation,
            )
        )
        # Create a learner for each hyperparameter combination
        self.learners = [
            Learner.from_hyperparams(
                model_cls=model_cls,
                opt_cls=opt_cls,
                momentum=momentum,
                lr=lr,
                nesterov=nesterov,
                weight_decay=weight_decay,
                n_hidden=n_hidden,
                seed_weights=seed_weights,
                seed_perturbation=seed_perturbation,
                epsilon=epsilon,
                perturbation=perturbation,
                dir=dir
            )
            for momentum, lr, nesterov, weight_decay, n_hidden, seed_weights, seed_perturbation, epsilon, perturbation in self.hyperparams
        ]

        # Add baselines

        for learner in self.learners:
            if learner.hyperparams["epsilon"] != 0.0:
                hp = learner.hyperparams.copy()
                del hp["epsilon"]
                learner.baseline = next(self.filter_learners(**hp))
            else:
                learner.baseline = self.learners[0]

        self.metrics = Metrics(self.train_dl, self.test_dl)
        self.plot_fns = to_tuple(plot_fns) if plot_fns else []
        self.plot_ivl = plot_ivl
        self.save_ivl = save_ivl

    def train(self, n_epochs: int, start_epoch: int = 0, reset: bool = False):
        step, batch_idx = 0, 0
        for learner in self.learners:
            learner.load()

        for epoch in tqdm(range(start_epoch, n_epochs), desc="Training..."):
            t.manual_seed(self.seed_dl + epoch)  # To shuffle the data

            for batch_idx, batch in tqdm(
                enumerate(self.train_dl), desc=f"Epoch {epoch}"
            ):
                step = epoch * len(self.train_dl) + batch_idx

                for learner in self.learners:
                    if learner.step >= step:
                        continue

                    learner.train_batch(epoch, batch_idx, step, batch)

                if step % self.logging_ivl == 0:
                    self.test(step=step, epoch=epoch, batch_idx=batch_idx)

                if step % self.plot_ivl == 0 and step > 0:
                    self.plot(step=step)

                if step % self.save_ivl == 0:
                    self.save(step=step)

    def test(self, step, **kwargs):
        learners = [learner for learner in self.learners if step not in learner.logs]

        for metric, learner in tqdm(
            zip(self.metrics.measure(learners), learners), desc=f"Testing {step}"
        ):
            # pp(metric)
            learner.log(**metric, **kwargs)

    def plot(self, step: Optional[int] = None):
        df = self.df()

        for plot_fn in self.plot_fns:
            fig, ax = plot_fn(self, df, step=step, **self.fixed_hyperparams)

            # Save the figure
            if fig is not None:
                fig.savefig(
                    str(self.dir / f"img/{plot_fn.__name__}_{step}.png"),
                )

    def save(self, step: int, overwrite: bool = False):
        for learner in self.learners:
            learner.save(overwrite=overwrite)

    def load(self, *args, **kwargs):
        for learner in self.learners:
            learner.load(*args, **kwargs)

    def df(self):
        """Returns a dataframe with the logs of all learners"""
        return pd.concat([learner.df() for learner in self.learners])

    @property
    def models(self):
        """Returns a list of all models"""
        return [learner.model for learner in self.learners]

    @property
    def fixed_hyperparams(self):
        """Returns a dictionary with the hyperparams that are fixed for all learners"""

        hyperparams = {}
        variable_hyperparams = set()

        for learner in self.learners:
            for name, value in learner.hyperparams.items():
                if name in hyperparams and value != hyperparams[name]:
                    variable_hyperparams.add(name)
                    del hyperparams[name]
                elif name not in hyperparams and name not in variable_hyperparams:
                    hyperparams[name] = value

        return hyperparams


    def filter_learners(self, **kwargs) -> Generator[Learner, None, None]:
        """Filter learners by hyperparameters"""
        return (
            learner
            for learner in self.learners
            if all(learner.hyperparams[k] == v for k, v in kwargs.items())
        )

    def __getitem__(self, i):
        return self.learners[i]

    def __len__(self):
        return len(self.learners)


def get_mnist_dataloaders(batch_size: int = 64):
    train_ds = datasets.MNIST(
        root="data", train=True, download=True, transform=transforms.ToTensor()
    )
    test_ds = datasets.MNIST(
        root="data", train=False, download=True, transform=transforms.ToTensor()
    )

    train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    test_dl = DataLoader(test_ds, batch_size=batch_size, shuffle=False)

    return train_dl, test_dl


train_dl, test_dl = get_mnist_dataloaders()



# %%

defaults = dict(
    model_cls=MNIST,
    opt_cls=t.optim.SGD,
    train_dl=train_dl,
    test_dl=test_dl,
    logging_ivl=100,
    plot_ivl=1000,
    save_ivl=2000,
    plot_fns=(epsilon_scaling, corr_scaling, cos_sim_scaling, loss_scaling),
    seed_dl=0,
    epsilon=0.01,
    n_hidden=100,
    momentum=0.0,
    weight_decay=0.0,
    seed_perturbation=tuple(i for i in range(10)),
)

experiments = [
    {
        "epsilon": (0.0, 0.001, 0.01, 0.1, 1.0), 
        "dir": "results/vanilla"
    },
    {
        # Depth
        "epsilon": (0.0, 0.01),
        "n_hidden": ([784, 50, 10], [784, 50, 50, 10], [784, 50, 50, 50, 10], [784, 50, 50, 50, 50,]),
        "dir": "results/depth"
    },
    {
        # Width
        "epsilon": (0.0, 0.01),
        "n_hidden": (400, 200, 100, 50, 25,),
        "dir": "results/width"
    },
    {
        # Momentum
        "epsilon": (0.0, 0.01),
        "momentum": (0.0, 0.1, 0.5, 0.9),
        "dir": "results/momentum"
    },
    {
        # Weight decay
        "epsilon": (0.0, 0.01),
        "weight_decay": (0.0, 0.001, 0.01, 0.1, 0.5, 0.9),
        "dir": "results/weight_decay"
    },
]

for experiment in experiments:
    kwargs = {**defaults}
    kwargs.update(experiment)

    ensemble = Ensemble(**kwargs) # type: ignore
    ensemble.train(50)


# %%
