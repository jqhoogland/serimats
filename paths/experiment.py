# %%
import hashlib
import itertools
import os
from dataclasses import dataclass, field
from os import PathLike
from pathlib import Path
from typing import (
    Any,
    Callable,
    Dict,
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
from torch import nn, optim
from torch.nn import functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from tqdm import tqdm

from serimats.paths.models import MNIST, ExtendedModule
from serimats.paths.utils import dict_to_latex, setup, var_to_latex
from serimats.paths.weights import (
    AbsolutePerturbationInitializer,
    RelativePerturbationInitializer,
)

setup()

T = TypeVar("T")
Hyperparameter = Union[T, Tuple[T, ...]]


def stable_hash(x: Any) -> str:
    return hashlib.sha256(str(x).encode("utf-8")).hexdigest()[:32]


def to_tuple(x: Hyperparameter[T]) -> Tuple[T, ...]:
    return x if isinstance(x, tuple) else (x,)


def get_opt_hyperparams(opt: optim.Optimizer) -> dict:
    """Assumes that the optimizer has only a single set of hyperparams
    (and not a separate set for each parameter group)"""
    param_group = opt.param_groups[0]

    hyperparams = {}

    for k, v in param_group.items():
        if k not in ["params", "foreach", "maximize", "capturable", "fused"]:
            hyperparams[k] = v

    return hyperparams


@dataclass
class Metrics:
    baseline: ExtendedModule
    train_dl: DataLoader  # Currently not in use
    test_dl: DataLoader

    class Report(TypedDict):
        d_w: Optional[float]
        d_w_rel_to_init: Optional[float]
        d_w_rel_to_norm: Optional[float]
        L_train: Optional[float]
        del_L_train: Optional[float]
        acc_train: Optional[float]
        del_acc_train: Optional[float]
        L_train_compare: Optional[float]
        acc_train_compare: Optional[float]
        L_test: Optional[float]
        del_L_test: Optional[float]
        acc_test: Optional[float]
        del_acc_test: Optional[float]
        L_test_compare: Optional[float]
        acc_test_compare: Optional[float]

    def d_w(self, model: ExtendedModule) -> t.Tensor:
        return t.norm(model.parameters_vector - self.baseline.parameters_vector)

    def d_ws(self, learners: List["Learner"]) -> t.Tensor:
        return t.tensor([self.d_w(learner.model) for learner in learners])

    def d_ws_with_rel(self, learners: List["Learner"]) -> Tuple[t.Tensor, ...]:
        d_ws = self.d_ws(learners)

        # TODO: This will fail with absolute weight initializers
        delta_norms = t.Tensor(
            [learner.weight_initializer.delta_norm for learner in learners]
        )
        epsilons = t.Tensor(
            [learner.weight_initializer.epsilon for learner in learners]
        )
        # init_norms = np.array(
        #     [learner.weight_initializer.init_norm for learner in learners]
        # )

        d_ws_rel = d_ws - delta_norms
        d_ws_rel_to_init = d_ws_rel / delta_norms
        d_ws_rel_to_norm = d_ws_rel / (epsilons * self.baseline.parameters_norm)

        return d_ws, d_ws_rel_to_init, d_ws_rel_to_norm

    def w_norm(self, model: ExtendedModule) -> t.Tensor:
        return model.parameters_norm

    def w_norms(self, models: List[ExtendedModule]) -> t.Tensor:
        return t.tensor([self.w_norm(model) for model in models])

    def w_norms_with_rel(
        self, models: List[ExtendedModule]
    ) -> Tuple[t.Tensor, t.Tensor]:
        w_norms = self.w_norms(models)
        rel_w_norms = w_norms / self.baseline.parameters_norm

        return w_norms, rel_w_norms

    def measure_model(
        self, model: ExtendedModule, data: t.Tensor, target: t.Tensor
    ) -> Tuple[float, float]:
        output = model(data)

        loss = self.loss(output, target, reduction="sum").item()
        pred = output.argmax(dim=1, keepdim=True)
        acc = pred.eq(target.view_as(pred)).sum().item()

        return loss, acc

    def test_batch(
        self, models: List[ExtendedModule], data: t.Tensor, target: t.Tensor
    ) -> Tuple[t.Tensor, ...]:
        baseline_output = self.baseline(data)
        baseline_pred = baseline_output.argmax(dim=1, keepdim=False)

        L_test = t.zeros((len(models),))
        acc_test = t.zeros((len(models),))

        L_compare = t.zeros((len(models),))
        acc_compare = t.zeros((len(models),))

        for i, model in enumerate(models):
            L_test[i], acc_test[i] = self.measure_model(model, data, target)
            L_compare[i], acc_compare[i] = self.measure_model(
                model, data, baseline_pred
            )

        return L_test, acc_test, L_compare, acc_compare

    def data_set(
        self, models: List[ExtendedModule], dl: DataLoader
    ) -> Tuple[t.Tensor, ...]:
        """Returns the loss and accuracy averaged over the entire test set."""
        n_models = len(models)

        with t.no_grad():
            L_test = t.zeros((n_models,))
            acc_test = t.zeros((n_models,))
            L_compare = t.zeros((n_models,))
            acc_compare = t.zeros(
                (n_models),
            )

            if n_models == 0:
                return (
                    L_test,
                    acc_test,
                    L_compare,
                    acc_compare,
                    t.zeros((0,)),
                    t.zeros((0,)),
                )

            for data, target in dl:
                data, target = data.to(self.baseline.device), target.to(
                    self.baseline.device
                )

                (
                    L_test_batch,
                    acc_test_batch,
                    L_compare_batch,
                    acc_compare_batch,
                ) = self.test_batch(models, data, target)

                L_test += L_test_batch
                acc_test += acc_test_batch
                L_compare += L_compare_batch
                acc_compare += acc_compare_batch

            n_samples = len(dl.dataset)  # type: ignore
            L_test /= n_samples
            acc_test /= n_samples
            L_compare /= n_samples
            acc_compare /= n_samples

            del_L_test = L_test - L_test[0]
            del_acc_test = acc_test - acc_test[0]

            return L_test, acc_test, L_compare, acc_compare, del_L_test, del_acc_test

    def test_set(self, models: List[ExtendedModule]) -> Tuple[t.Tensor, ...]:
        """Returns the loss and accuracy averaged over the entire test set."""
        return self.data_set(models, self.test_dl)

    def train_set(self, models: List[ExtendedModule]) -> Tuple[t.Tensor, ...]:
        """Returns the loss and accuracy averaged over the entire test set."""
        return self.data_set(models, self.train_dl)

    def measure(self, learners: List["Learner"]) -> Iterable[Report]:
        n_models = len(learners)
        d_ws, d_ws_rel_to_init, d_ws_rel_to_norm = self.d_ws_with_rel(learners)

        models = [learner.model for learner in learners]

        (
            L_train,
            acc_train,
            L_train_compare,
            acc_train_compare,
            del_L_train,
            del_acc_train,
        ) = self.train_set(models)
        (
            L_test,
            acc_test,
            L_test_compare,
            acc_test_compare,
            del_L_test,
            del_acc_test,
        ) = self.test_set(models)

        return [
            self.Report(
                d_w=d_ws[i].item(),
                d_w_rel_to_init=d_ws_rel_to_init[i].item(),
                d_w_rel_to_norm=d_ws_rel_to_norm[i].item(),
                L_train=L_train[i].item(),
                del_L_train=del_L_train[i].item(),
                acc_train=acc_train[i].item(),
                del_acc_train=del_acc_train[i].item(),
                L_train_compare=L_train_compare[i].item(),
                acc_train_compare=acc_train_compare[i].item(),
                L_test=L_test[i].item(),
                del_L_test=del_L_test[i].item(),
                acc_test=acc_test[i].item(),
                del_acc_test=del_acc_test[i].item(),
                L_test_compare=L_test_compare[i].item(),
                acc_test_compare=acc_test_compare[i].item(),
            )
            for i in range(n_models)
        ]

    def loss(
        self, output: t.Tensor, target: t.Tensor, reduction: str = "mean"
    ) -> t.Tensor:
        return F.nll_loss(output, target, reduction=reduction)

    def keys(self) -> List[str]:
        return [
            "d_w",
            "d_w_rel_to_init",
            "d_w_rel_to_norm",
            "L_train",
            "del_L_train",
            "acc_train",
            "del_acc_train",
            "L_train_compare",
            "acc_train_compare",
            "L_test",
            "del_L_test",
            "acc_test",
            "del_acc_test",
            "L_test_compare",
            "acc_test_compare",
        ]


class Learner:
    def __init__(
        self,
        model: ExtendedModule,
        opt: optim.SGD,
        weight_initializer: Callable[[ExtendedModule], None],
        dir: PathLike,
    ):
        self.model = model
        self.opt = opt
        self.weight_initializer = weight_initializer
        self.weight_initializer(self.model)

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
            print(f"Step {self.step} already exists, skipping")
            return False

        self.df(full=False).to_csv(self.path / "logs.csv")
        t.save(self.model.state_dict(), self.path_to_step / "model.pt")
        t.save(self.opt.state_dict(), self.path_to_step / "opt.pt")

        yaml.dump(self.hyperparams, open(self.path / "hyperparams.yaml", "w"))

        print(
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

            print(logs_df)
        except FileNotFoundError:
            print(f"Could not load logs from {self.path / 'logs.csv'}")
            print(yaml.dump(self.hyperparams))
            self.logs = {}
            return False

        try:
            self.model.load_state_dict(t.load(self.path_to_step / "model.pt"))
            self.opt.load_state_dict(t.load(self.path_to_step / "opt.pt"))

            print(
                f"Loaded model from {self.path_to_step} (step={self.step}, {self.extra_repr})"
            )
        except FileNotFoundError:
            print(f"Could not load model or optimizer from {self.path_to_step}")
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

        print(df)

        return df

    @property
    def device(self):
        return self.model.device

    def __call__(self, *args, **kwargs) -> t.Tensor:
        return self.model(*args, **kwargs)


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
        momentum: Hyperparameter[float] = 0.0,
        lr: Hyperparameter[float] = 0.01,
        nesterov: Hyperparameter[bool] = False,
        weight_decay: Hyperparameter[float] = 0.0,
        # Model
        n_hidden: Hyperparameter[Union[int, List[int]]] = 100,
        # Weight initialization
        seed_weights: Hyperparameter[int] = 0,
        seed_perturbation: Hyperparameter[int] = 0,
        epsilon: Hyperparameter[float] = 0.01,
        perturbation: Hyperparameter[Literal["absolute", "relative"]] = "relative",
        plot_fns: Optional[Hyperparameter[Callable]] = None,
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
                dir=dir,
            )
            for momentum, lr, nesterov, weight_decay, n_hidden, seed_weights, seed_perturbation, epsilon, perturbation in self.hyperparams
        ]

        self.metrics = Metrics(self.baseline.model, self.train_dl, self.test_dl)
        self.plot_fns = plot_fns or []
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

                if step % self.plot_ivl == 0:
                    self.plot(step=step, epoch=epoch, batch_idx=batch_idx)

                if step % self.save_ivl == 0:
                    self.save(step=step)

    def test(self, step, **kwargs):
        learners = [learner for learner in self.learners if step not in learner.logs]

        for metric, learner in tqdm(
            zip(self.metrics.measure(learners), learners), desc=f"Testing {step}"
        ):
            learner.log(**metric, **kwargs)

    def plot(
        self,
        step: Optional[int] = None,
        epoch: Optional[int] = None,
        batch_idx: Optional[int] = None,
    ):
        df = self.df()

        for plot_fn in self.plot_fns:
            fig, ax = plot_fn(self, df, step=step, **self.fixed_hyperparams)

            # Save the figure
            if fig is not None:
                fig.savefig(
                    self.dir / f"img/{plot_fn.__name__}_{step}.png",
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
    def baseline(self):
        """Returns the baseline learner"""
        return self.learners[0]

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


def plot_metric_scaling(
    ensemble: Ensemble,
    df: pd.DataFrame,
    step: Optional[int] = None,
    metric: Tuple[str, ...] = ("d_w_by_eps", "d_w_rel_to_norm"),
    comparison: str = "epsilon",
    sample_axis: str = "seed_perturbation",
    include_baseline: bool = True,  # Whether to plot the first value of comparison
    **kwargs,
) -> Tuple[Figure, List[List[plt.Axes]]]:
    df = df.loc[df["step"] <= step]
    metric_labels = tuple(var_to_latex(m) for m in metric)

    comparison_label = var_to_latex(comparison)
    comparison_values = getattr(ensemble, comparison)

    if not include_baseline:
        comparison_values = comparison_values[1:]

    sample_values = df[sample_axis].unique()

    if "perturbation" in kwargs:
        del kwargs["perturbation"]

    details = dict_to_latex(kwargs)

    averages = df.groupby([comparison, "step"]).mean()
    averages.reset_index(inplace=True)

    steps = df["step"].unique()

    # Figure is as wide as there are metrics and
    # as tall as there are choices of `comparison` (+ 1 to compare averages)
    fig, axes = plt.subplots(
        len(comparison_values) + 1,
        len(metric),
        figsize=(len(metric) * 5, len(getattr(ensemble, comparison)) * 5),
    )

    # This works better with escaped LaTeX than f-strings. I think.
    title = (
        "$"
        + " ? "
        # str(metric_labels) +
        "$ for $"
        +
        # str(comparison_label) +
        "("
        + ", ".join(map(lambda s: str(s), comparison_values))
        + " )$\n($"
        + details
        + "$)"
    )

    fig.suptitle(title)
    fig.tight_layout(pad=4.0)

    for c, (m, m_label) in enumerate(zip(metric, metric_labels)):
        for r, v in enumerate(comparison_values):
            # Get the data for this comparison value
            data = df.loc[df[comparison] == v]

            # Plot the data
            for sample in sample_values:
                axes[r][c].plot(
                    steps,
                    data.loc[data[sample_axis] == sample][m].values,
                    alpha=0.75,
                    linewidth=0.5,
                )

            # Plot the average across samples
            axes[r][c].plot(
                steps,
                averages.loc[averages[comparison] == v][m].values,
                color="black",
                linestyle="--",
            )

            axes[r][c].set_title(f"${m_label}$ for ${comparison_label} = {v}$")
            axes[r][c].set_xlabel("Step $t$")
            axes[r][c].set_ylabel(f"${m_label}$")

    # Plot the comparison of averages
    for c, (m, m_label) in enumerate(zip(metric, metric_labels)):

        # Plot the averages across comparison values
        for v in comparison_values:
            axes[-1][c].plot(
                steps,
                averages.loc[averages[comparison] == v][m].values,
                label=f"${v}$",
            )

        axes[-1][c].set_title(f"${m_label}$ across ${comparison_label}$")
        axes[-1][c].set_xlabel("Step $t$")
        axes[-1][c].set_ylabel(f"$\\overline{{{m_label}}}$")
        axes[-1][c].legend()

    return fig, axes


# %%


# Create the ensemble
# ensemble1 = Ensemble(
#     model_cls=MNIST,
#     opt_cls=t.optim.SGD,
#     train_dl=train_dl,
#     test_dl=test_dl,
#     dir="results",
#     logging_ivl=100,
#     seed_dl=0,
#     # n_hidden=(10, 50, 100, 1000),
#     # n_hidden=(10, 50, 100),
#     n_hidden=50,
#     # lr=(0.001, 0.01, 0.1),
#     # weight_decay=(0.0, 0.1, 0.5, 0.9),
#     weight_decay=(0.0, 0.1),
#     # momentum=(0.0, 0.1, 0.5, 0.9),
#     momentum=(0.0, 0.1),
#     epsilon=(0.0, 0.01, 0.1),
#     seed_perturbation=tuple(i for i in range(25)),
# )

# ensemble1.train(50)

# ensemble2 = Ensemble(
#     model_cls=MNIST,
#     opt_cls=t.optim.SGD,
#     train_dl=train_dl,
#     test_dl=test_dl,
#     dir="results",
#     logging_ivl=100,
#     seed_dl=0,
#     epsilon=0.01,
#     momentum=(0.0, 0.1, 0.5, 0.9),
#     seed_perturbation=tuple(i for i in range(50)),
# )

# ensemble2.train(50)


def epsilon_scaling(
    ensemble: Ensemble, df: pd.DataFrame, step: Optional[int] = None, **kwargs
):
    return plot_metric_scaling(
        ensemble,
        df,
        step=step,
        metric=("d_w_rel_to_init", "d_w_rel_to_norm"),
        comparison="epsilon",
        sample_axis="seed_perturbation",
        include_baseline=False,
        **kwargs,
    )


ensemble = Ensemble(
    model_cls=MNIST,
    opt_cls=t.optim.SGD,
    train_dl=train_dl,
    test_dl=test_dl,
    dir="results",
    logging_ivl=100,
    seed_dl=1,
    epsilon=(0.0, 0.001, 0.01, 0.1, 1.0),
    # n_hidden=([784, 100, 50, 30, 10],),
    # momentum=(0.0, 0.1, 0.5, 0.9),
    # weight_decay=(0.0, 0.001, 0.01, 0.1, 0.5, 0.9),
    seed_perturbation=tuple(i for i in range(10)),
    plot_fns=(epsilon_scaling,),
    plot_ivl=1000,
)

ensemble.train(10)


# %%


def plot_metric_panel() -> Tuple[Figure, List[List[plt.Axes]]]:
    def plot_panel(
        self,
        df: pd.DataFrame,
        comparison: str,
        metrics: List[str],
        step: Optional[int] = None,
        epoch: Optional[int] = None,
        batch_idx: Optional[int] = None,
        overwrite: bool = False,
        **details,
    ):
        """Plots a panel of the given metrics for the given hyperparameter"""

        series = df

        baseline = series[series["epsilon"] == 0.0]

        # Filter the series
        for k, v in details.items():
            series = series[series[k] == v]
            details[k] = v

            if k != "epsilon":
                baseline = baseline[baseline[k] == v]

        # Stable order
        details = {k: v for k, v in sorted(details.items())}
        details_rep = dict_to_latex(details)

        if step is not None:
            series = series[series["step"] <= step]
            baseline = baseline[baseline["step"] <= step]
        else:
            step = series["step"].max().item()

        comparison_values = series[comparison].unique()

        # Check if the figure already exists
        img_dir = self.dir / "img"
        img_dir.mkdir(parents=True, exist_ok=True)
        filepath = img_dir / f"{step}_{comparison}_{stable_hash(details_rep)}.png"

        if filepath.exists() and not overwrite:
            return

        # Create a new figure
        fig, axes = plt.subplots(
            len(metrics) // 2, 2, figsize=(5 * 2, len(metrics) * 5 // 3)
        )
        fig.tight_layout(pad=5.0)

        fig.suptitle(
            f"Comparison over ${var_to_latex(comparison)}$\\ ($t\leq{step}, {details_rep}$)"
        )

        average = series.groupby("step").mean().reset_index()

        # Plot each metric
        for i, metric in enumerate(metrics):
            ax = axes[i // 2][i % 2]

            plot_baseline = metric in [
                "L_train",
                "acc_train",
                "L_test",
                "acc_test",
            ]

            # Plot the baseline
            if plot_baseline:
                ax.plot(
                    baseline["step"], baseline[metric], label="baseline", color="black"
                )

            # Plot the average
            ax.plot(average["step"], average[metric], label="average", color="red")

            # Plot the other models
            for comparison_value in comparison_values:
                df_for_value = series[series[comparison] == comparison_value]

                comparison_label = f"${var_to_latex(comparison)}={comparison_value}$"

                if comparison == "seed_perturbation":
                    comparison_label = f"_{comparison_value}"  # Hidden from legend

                ax.plot(
                    df_for_value["step"],
                    df_for_value[metric],
                    label=comparison_label,
                    alpha=0.25,
                )

            metric_rep = f"${var_to_latex(metric)}$"
            ax.set_title(metric_rep)
            ax.set_xlabel("step")
            ax.set_ylabel(metric_rep)
            ax.legend()

        # Save the figure
        # splt.show()

        print(f"Saving figure to {filepath}")
        fig.savefig(filepath)  # type: ignore

    def plot(
        self,
        step: Optional[int] = None,
        epoch: Optional[int] = None,
        batch_idx: Optional[int] = None,
    ):
        df = self.df()
        comparisons = self.comparisons

        for comparison in comparisons:
            # Plot a panel over a single comparison hyperparameter
            # for all possible other hyperparams held fixed

            other_comparisons: List[str] = [c for c in comparisons if c != comparison]
            other_hp_combos: Iterable[Tuple[Any, ...]] = itertools.product(
                *[getattr(self, hp) for hp in other_comparisons]
            )

            for hp_combo in other_hp_combos:
                details = dict(zip(other_comparisons, hp_combo))

                self.plot_panel(
                    df,
                    comparison,
                    metrics=self.metrics.keys(),
                    step=step,
                    epoch=epoch,
                    batch_idx=batch_idx,
                    **details,
                )

                if comparison in ("epsilon", "momentum", "lr", "weight_decay"):
                    # We only want to plot the panel over epsilon for one choice of seed perturbation
                    break


# %%
