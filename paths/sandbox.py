"""
Train a few MNITS classifiers and compare their trajectories.
"""

# %%

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import (
    Any,
    Collection,
    Container,
    Generator,
    Iterable,
    Iterator,
    Optional,
    Sequence,
    Tuple,
    Type,
    Union,
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

from serimats.paths.learner import MultiLearner
from serimats.paths.models import MNIST
from serimats.paths.utils import dict_to_latex, divide_params, setup, var_to_latex
from serimats.paths.weights import (
    AbsolutePerturbationInitializer,
    RelativePerturbationInitializer,
    WeightInitializer,
)

device = setup()

# %%

train_loader = DataLoader(
    datasets.MNIST(
        "../data",
        train=True,
        download=True,
        transform=transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
        ),
    ),
    batch_size=64,
    shuffle=True,
)
test_loader = DataLoader(
    datasets.MNIST(
        "../data",
        train=False,
        transform=transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
        ),
    ),
    shuffle=True,
)
# %%


def avg_over_training(
    logs: pd.DataFrame,
    key: str,
    include_baseline=True,
):
    return (
        (logs[logs["model_idx"] != 0] if not include_baseline else logs)
        .groupby("step")[key]
        .mean()
    )


def plot_over_training(
    logs: Union[MultiLearner, pd.DataFrame, pd.Series],
    key: str,
    title: str,
    ylabel: str,
    include_baseline=True,
    n_models: int = 10,
):
    logs = logs.logs if isinstance(logs, MultiLearner) else logs

    start_idx = 0 if include_baseline else 1

    for i in range(start_idx, int(n_models)):
        data = logs[logs["model_idx"] == i]

        plt.plot(
            data["step"],
            data[key],
            "--",
            label=f"Model {i}",
            alpha=0.25,
        )

    steps = logs[logs["model_idx"] == 0]["step"]
    train_loss_avg = avg_over_training(logs, key, include_baseline)
    plt.plot(steps, train_loss_avg, label="Average")

    plt.title(title)
    plt.ylabel(ylabel)
    plt.legend()

    plt.show()


# %%


def plot_avgs_over_training(
    multilearners: list[MultiLearner],
    epsilons: list[float],
    key: str,
    title: str,
    ylabel: str,
):
    steps = multilearners[0].logs[multilearners[0].logs["model_idx"] == 0]["step"]

    for multilearner, epsilon in zip(multilearners, epsilons):
        train_loss_avg = avg_over_training(
            multilearner.logs, key, include_baseline=False
        )
        plt.plot(steps, train_loss_avg, label=f"$\epsilon={epsilon}$")

    plt.title(title)
    plt.xlabel("Training step")
    plt.ylabel(ylabel)
    plt.legend()

    plt.show()


# %%


def run(
    train_loader, test_loader, weight_initializer=None, model_cls=MNIST, **kwargs
) -> Iterator[Tuple[pd.DataFrame, dict]]:
    df = pd.DataFrame()

    # Recursive call for any parameters which are provided as sequences
    for k, v in kwargs.items():
        if isinstance(v, (tuple, list)):
            print(f"Iterating over {k}={v}")
            for subvalue in v:
                for subdf, details in run(
                    train_loader, test_loader, **{**kwargs, k: subvalue}
                ):
                    df = pd.concat([df, subdf])

                    if k == "config":
                        details.update(subvalue)
                    else:
                        details[k] = subvalue

                    yield subdf, details

    # Include seed in kwargs so we can iterate over multiple seeds
    assert "seed" in kwargs, "Must provide `seed` as keyword argument"
    t.manual_seed(kwargs.pop("seed"))

    # Include epsilon in kwargs so we can iterate over multiple epsilon
    epsilon = kwargs.pop("epsilon", 0.01)

    if weight_initializer is None:
        weight_initializer = RelativePerturbationInitializer(epsilon)

    multilearner = MultiLearner(
        model_cls,
        train_loader,
        test_loader,
        weight_initializer=weight_initializer,
        **kwargs,
    )

    # Set weights
    multilearner.train()

    yield multilearner.logs, {}


def run_and_plot(
    train_loader, test_loader, weight_initializer=None, **kwargs
) -> Optional[pd.DataFrame]:
    result = None

    for result, details in run(
        train_loader, test_loader, weight_initializer=weight_initializer, **kwargs
    ):
        details_rep = dict_to_latex(details)

        subresult = result

        for k, v in details.items():
            subresult = subresult[subresult[k] == v]

        for metric in MultiLearner.__metrics__:
            metric_latex = var_to_latex(metric)

            if metric_latex is None:
                continue

            include_baseline = metric not in (
                "d_w",
                "rel_d_w",
                "del_L_train",
                "del_L_test",
                "del_acc_test",
            )

            plot_over_training(
                subresult,
                metric,
                f"${metric_latex}$ over training (${details_rep}$)",
                f"${metric_latex}$",
                include_baseline=include_baseline,
            )

            # TODO: compare wrt details

    return result


# %%

epsilon_results = run_and_plot(
    train_loader,
    test_loader,
    epsilon=[0.001, 0.01, 0.1, 1.0, 10.0],
    seed=0,
    logging_interval=1000,
    epochs=10,
)
# %%

lr_experiments = run_and_plot(
    train_loader,
    test_loader,
    epsilon=0.01,
    lr=[0.001, 0.01, 0.1],
    seed=0,
    logging_interval=1000,
)

# %%

momentum_experiments = run_and_plot(
    train_loader,
    test_loader,
    epsilon=0.01,
    momentum=[0.001, 0.01, 0.1],
    seed=0,
    logging_interval=1000,
)

# %%

weight_decay_experiments = run_and_plot(
    train_loader,
    test_loader,
    epsilon=0.01,
    lr=0.01,
    momentum=0.01,
    weight_decay=[0.001, 0.01, 0.1],
    seed=0,
    logging_interval=1000,
)

# %%

seed_experiments = run_and_plot(
    train_loader,
    test_loader,
    epsilon=0.01,
    lr=0.01,
    momentum=0.01,
    weight_decay=0.01,
    seed=[0, 1, 2, 3, 4],
    logging_interval=1000,
)

# run(
#     train_loader,
#     test_loader,
#     epsilon=0.01,
#     lr=0.01,
#     momentum=0.01,
#     weight_decay=0.01,
#     seed=0,
#     batch_size=[1, 10, 100],
# )

# %%

width_experiments = run_and_plot(
    train_loader,
    test_loader,
    epsilon=0.01,
    lr=0.01,
    momentum=0.01,
    weight_decay=0.01,
    seed=0,
    logging_interval=1000,
    config=[{"n_hidden": n_h} for n_h in [10, 50, 100, 500, 1000]],
)


depth_experiments = run_and_plot(
    train_loader,
    test_loader,
    epsilon=0.01,
    lr=0.01,
    momentum=0.01,
    weight_decay=0.01,
    seed=0,
    logging_interval=1000,
    config=[{"n_hidden": divide_params(50000, i, 784, 10)} for i in range(1, 6)],
)
