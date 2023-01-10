import hashlib
import warnings
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import (Any, Callable, Collection, Container, Dict, Generic,
                    Iterable, List, Optional, Protocol, Sequence, Tuple, Type,
                    TypeVar, Union)

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


def setup() -> str:
    device = "cuda" if t.cuda.is_available() else "cpu"

    mpl.rcParams["text.usetex"] = True
    # mpl.rcParams[
    #     "text.latex.preamble"
    # ] = [r"\usepackage{amsmath}"]  # for \text command
    mpl.rcParams["figure.dpi"] = 300
    plt.style.use("ggplot")
    t.device(device)

    # wandb.login()

    return device


T = TypeVar("T")
OptionalTuple = Union[T, Tuple[T, ...]]


def stable_hash(x: Any) -> str:
    return hashlib.sha256(str(x).encode("utf-8")).hexdigest()[:32]


def to_tuple(x: OptionalTuple[T]) -> Tuple[T, ...]:
    return x if isinstance(x, tuple) else (x,)


def get_parameters(model: nn.Module) -> t.Tensor:
    """Get a flattened tensor of all parameters in a model."""
    return t.cat([p.view(-1) for p in model.parameters()])


def tensor_map(f: Callable[..., float], *args):
    """Map a function that returns a float over a (collection of) iterables, 
    then wrap the result in a tensor."""
    return t.tensor([f(*arg) for arg in zip(*args)])
 

# Latex & typesetting

class WithLatex(Protocol):
    __latex__: Tuple[str, str]


class CallableWithLatex(WithLatex, Protocol):
    __name__: str
    
    def __call__(self, *args, **kwargs) -> Any:
        ...


def add_latex(name: str, body: str):
    """Add a latex representation to a function (of the function name & body)."""
    
    def decorator(f) -> CallableWithLatex:
        f.__latex__ = (name, body)

        return f

    return decorator


VARS = {
    "epsilon": r"\epsilon",
    "weight_decay": r"\lambda",
    "lr": r"\eta",
    "momentum": r"\beta",
    "n_models": "N",
    "step": "t",
    "epoch": r"\mathrm{epoch}",
    "epochs": "T_i",
    "logging_interval": r"\Delta t",
    "L_test": r"L_\mathrm{test}",
    "L_train": r"L_\mathrm{train}",
    "L_compare": r"L_\mathrm{compare}",
    "L_compare_train": r"L_\mathrm{cf. train}",
    "L_compare_test": r"L_\mathrm{cf. test}",
    "acc_train": r"\mathrm{acc}_\mathrm{train}",
    "acc_test": r"\mathrm{acc}_\mathrm{test}",
    "acc_compare": r"\mathrm{acc}_\mathrm{compare}",
    "acc_compare_train": r"\mathrm{acc}_\mathrm{cf. train}",
    "acc_compare_test": r"\mathrm{acc}_\mathrm{cf. test}",
    "w": r"|w|",
    "w_normed": r"|w|/|w^{(0)}|",
    "d_w": r"d_W",
    "delta_L_train": r"\delta L_\mathrm{train}",
    "delta_L_test": r"\delta L_\mathrm{test}",
    "delta_acc_test": r"\delta \mathrm{acc}_\mathrm{test}",
    "delta_acc_train": r"\delta \mathrm{acc}_\mathrm{train}",
    "d_w_from_baseline": r"d_W(w^{(t)}, w_\mathrm{ref}^{(t)})",
    "d_w_from_init": r"d_W(w^{(t)}, w^{(0)})",
    # TODO angle between w and w_ref & angle between w and w_init
    "d_w_from_baseline_normed": r"\hat{d}_W(w^{(t)}, w_\mathrm{ref}^{(t)})",
    "d_w_from_init_normed": r"\hat{d}_W(w^{(t)}, w^{(0)})",
    "w_corr_with_baseline": r"w^{(t)} \cdot w_\mathrm{ref}^{(t)})",
    "w_autocorr": r"w^{(t)} \cdot w^{(0)})",
    "seed_weights": r"s_{\mathbf w_0}",
    "seed_perturbation": r"s_{\delta}",
    "perturbation": r"\mathrm{perturbation type}",
}


def var_to_latex(s: str) -> Optional[str]:
    return VARS.get(s)


def dict_to_latex(d: dict):
    d = d.copy()

    for k in list(d.keys()):
        k_prime = var_to_latex(k)

        if k_prime is not None:
            d[k_prime] = d.pop(k)
        else:
            d.pop(k)

    return ", ".join([f"{k} = {v}" for k, v in d.items()])


def calculate_n_params(widths: Tuple[int, ...]) -> int:
    return (
        sum((widths[i] + 1) * widths[i + 1] for i in range(len(widths) - 1))
        + widths[-1]
    )


def divide_params(
    n_params: int, n_layers: int, h_initial: int, h_final: int
) -> Tuple[int, ...]:
    """
    Divide parameters into n_layers layers and returns the number of units
    in each layer. Assumes each layer has c times as many parameters as the
    next layer, where c is a constant.
    """

    def _calculate_widths(c: float) -> Tuple[int, ...]:
        return (h_initial, *(int((c**i) * h_final) for i in range(n_layers, -1, -1)))

    def _calculate_n_params(c: float):
        return calculate_n_params(_calculate_widths(c))

    # Find the value of c that gives the closest number of parameters
    c = scipy.optimize.brentq(lambda c: _calculate_n_params(c) - n_params, 0.1, 10)

    return _calculate_widths(c)


class Snapshotter:
    def __init__(self, dir: Union[str, Path], prefix: str = "snapshot-"):
        self.dir = Path(dir)
        self.prefix = prefix

    def save(self, model: nn.Module, **kwargs):
        state_dict = model.state_dict()
        hash = self.get_hash(**kwargs)

        t.save(state_dict, self.dir / (hash + ".pt"))

    def load(self, model: nn.Module, **kwargs):
        hash = self.get_hash(**kwargs)
        path = self.dir / (hash + ".pt")

        if not path.exists():
            warnings.warn(f"Snapshot {path} does not exist ({kwargs}).")
            return False

        state_dict = t.load(path)
        model.load_state_dict(state_dict)

        return True

    def get_hash(self, **kwargs) -> str:
        kwargs = {k: kwargs[k] for k in sorted(kwargs.keys())}
        # print(f"Hashing {kwargs} -> {hash(str(kwargs))}")
        return f"{self.prefix}{hash(str(kwargs))}"


class Logger:
    def __init__(
        self, dir: Union[str, Path], prefix: str = "log-", columns: List[str] = []
    ):
        self.dir = Path(dir)
        self.prefix = prefix
        self.columns = columns
        self.logs = []

    def log(self, **data):
        self.logs.append(data)

    def df(self) -> pd.DataFrame:
        return pd.DataFrame(self.logs)

    def save(self, **kwargs):
        df = self.df()
        df.to_csv(self.dir / (self.get_hash(**kwargs) + ".csv"), index=False)

        return df

    def load(self, **kwargs):
        df = pd.read_csv(self.dir / (self.get_hash(**kwargs) + ".csv"))
        self.logs = df.to_dict(orient="records")

    def get_hash(self, **kwargs) -> str:
        kwargs = {k: kwargs[k] for k in sorted(kwargs.keys())}
        # print(f"Hashing {kwargs} -> {hash(str(kwargs))}")
        return f"{self.prefix}{hash(str(kwargs))}"

    def init(self):
        self.logs = []
        # self.save()


def avg_over_training(
    df: pd.DataFrame,
    key: str,
    include_baseline=True,
):
    return (
        (df[df["model_idx"] != 0] if not include_baseline else df)
        .groupby("step")[key]
        .mean()
    )


class Plotter:
    __metrics__ = [
        "L_train",
        "del_L_train",
        "L_test",
        "del_L_test",
        # "acc_train",
        # "del_acc_train",
        "acc_test",
        "del_acc_test",
        "L_compare",
        "acc_compare",
        "d_w",
        "d_w_rel_to_norm",
    ]

    def plot(self, df: pd.DataFrame, **kwargs):
        series = df

        # Filter for the right series
        for k, v in kwargs.items():
            series = series[series[k] == v]
            print(k, v, series.shape)

        details_rep = dict_to_latex(kwargs) or ""

        if details_rep:
            details_rep = f"\n(${details_rep}$)"

        fig, axs = plt.subplots(5, 2, figsize=(10, 15))
        fig.tight_layout(pad=5.0)

        i = 0

        for metric in self.__metrics__:
            metric_latex = var_to_latex(metric)

            include_baseline = metric not in (
                "d_w",
                "d_w_rel_to_norm",
                "del_L_train",
                "del_L_test",
                "del_acc_test",
                "acc_compare",
            )

            self.plot_over_training(
                series,
                metric,
                f"${metric_latex}$ over training" + details_rep,
                f"${metric_latex}$",
                include_baseline=include_baseline,
                ax=axs[i // 2][i % 2],
            )

            i += 1

            # TODO: compare wrt details

        plt.show()

    def plot_over_training(
        self,
        df: Union[pd.Series, pd.DataFrame],
        key: str,
        title: str,
        ylabel: str,
        include_baseline=True,
        n_models: int = 10,
        ax: Optional[plt.Axes] = None,
    ):
        start_idx = 0 if include_baseline else 1

        if ax is None:
            fig, ax = plt.subplots(1, 1)

        for i in range(start_idx, int(n_models)):
            data = df[df["model_idx"] == i]

            ax.plot(
                data["step"],
                data[key],
                "--",
                label=f"Model {i}",
                alpha=0.25,
            )

        steps = df[df["model_idx"] == 0]["step"]
        avg = avg_over_training(df, key, include_baseline)

        ax.plot(steps, avg, label="Average")

        ax.set_title(title)
        ax.set_ylabel(ylabel)
        ax.set_xlabel("t")
        # ax.legend()
