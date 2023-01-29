# %%
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
from torch import nn, optim
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms

from serimats.paths.metrics import (
    Metrics,
    cos_sim_from_baseline,
    cos_sim_from_init,
    d_w_from_baseline,
    d_w_from_baseline_normed,
    d_w_from_init,
    d_w_from_init_normed,
    w_autocorr,
    w_corr_with_baseline,
    w_normed,
)
from serimats.paths.models import FCN, ExtendedModule, Lenet5, ResNet
from serimats.paths.plots import plot_metric_scaling
from serimats.paths.utils import (
    CallableWithLatex,
    OptionalTuple,
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

device = setup()


def get_mnist_data():
    train_ds = datasets.MNIST(
        root="data", train=True, download=True, transform=transforms.ToTensor()
    )
    test_ds = datasets.MNIST(
        root="data", train=False, download=True, transform=transforms.ToTensor()
    )

    return train_ds, test_ds


def get_cifar10_data():
    train_ds = datasets.CIFAR10(
        root="data", train=True, download=True, transform=transforms.ToTensor()
    )
    test_ds = datasets.CIFAR10(
        root="data", train=False, download=True, transform=transforms.ToTensor()
    )

    return train_ds, test_ds


def get_imagenet_data():
    train_ds = datasets.ImageFolder(
        root="data/imagenet/train", transform=transforms.ToTensor()
    )
    test_ds = datasets.ImageFolder(
        root="data/imagenet/val", transform=transforms.ToTensor()
    )

    return train_ds, test_ds


def get_data(dataset: str):
    if dataset == "mnist":
        return get_mnist_data()
    elif dataset == "cifar10":
        return get_cifar10_data()
    elif dataset == "imagenet":
        return get_imagenet_data()
    else:
        raise ValueError(f"Unknown dataset {dataset}")


DEFAULT_MODEL_HYPERPARAMS = dict(
    cls=FCN,
    n_hidden=100,
)

DEFAULT_SGD_HYPERPARAMS = dict(
    cls=t.optim.SGD,
    lr=0.01,
    momentum=0.0,
    weight_decay=0.0,
)

DEFAULT_WEIGHT_INITIALIZER_HYPERPARAMS = dict(
    cls=RelativePerturbationInitializer,
    epsilon=0.01,
    seed_weights=0,
    seed_perturbation=0,
)


def make_ensembles(
    experiments: List[Dict[str, Any]],
    dir: str = "results",
    batch_size=64,
    logging_ivl=100,
    plot_ivl=2000,
    save_ivl=1000,
    seed_dl=0,
    # dl_hyperparams=DEFAULT_DL_HYPERPARAMS,
    model_hyperparams=DEFAULT_MODEL_HYPERPARAMS,
    opt_hyperparams=DEFAULT_SGD_HYPERPARAMS,
    weight_initializer_hyperparams=DEFAULT_WEIGHT_INITIALIZER_HYPERPARAMS,
    baseline={"epsilon": 0.0, "seed_perturbation": 0},
):
    for experiment in experiments:
        dataset = experiment.pop("dataset", "mnist")
        train_data, test_data = get_data(dataset)

        comparison = experiment.pop("comparison", "epsilon")
        experiment["dir"] = os.path.join(
            dir, dataset, experiment.pop("dir", comparison)
        )

        def epsilon_scaling(
            experiment: "Experiment",
            df: pd.DataFrame,
            step: Optional[int] = None,
            **kwargs,
        ):
            return plot_metric_scaling(
                experiment,
                # df,
                df.loc[df["epsilon"] > 0.0],
                step=step,
                metric=(w_normed, d_w_from_baseline_normed, d_w_from_init_normed),
                comparison=comparison,
                sample_axis="seed_perturbation",
                include_baseline=False,
                **kwargs,
            )

        def corr_scaling(
            experiment: "Experiment",
            df: pd.DataFrame,
            step: Optional[int] = None,
            **kwargs,
        ):
            return plot_metric_scaling(
                experiment,
                # df,
                df.loc[df["epsilon"] > 0.0],
                step=step,
                metric=(w_corr_with_baseline, w_autocorr),
                comparison=comparison,
                sample_axis="seed_perturbation",
                include_baseline=True,
                **kwargs,
            )

        def cos_sim_scaling(
            experiment: "Experiment",
            df: pd.DataFrame,
            step: Optional[int] = None,
            **kwargs,
        ):
            return plot_metric_scaling(
                experiment,
                # df,
                df.loc[df["epsilon"] > 0.0],
                step=step,
                metric=(cos_sim_from_baseline, cos_sim_from_init),
                comparison=comparison,
                sample_axis="seed_perturbation",
                include_baseline=True,
                **kwargs,
            )

        def _loss_scaling(
            experiment: "Experiment",
            df: pd.DataFrame,
            metrics: List[Tuple[str, str]],
            step: Optional[int] = None,
            **kwargs,
        ):
            def mock_metric(name, latex_name, latex_body=None) -> CallableWithLatex:
                def metric(*args, **kwargs):
                    pass

                metric.__name__ = name
                metric.__latex__ = (latex_name, latex_body or latex_name)

                return metric  # type: ignore

            metric = tuple(mock_metric(*m) for m in metrics)

            return plot_metric_scaling(
                experiment,
                # df,
                df.loc[df["epsilon"] > 0.0],
                step=step,
                metric=metric,
                comparison=comparison,
                sample_axis="seed_perturbation",
                include_baseline=True,
                **kwargs,
            )

        def loss_cf_scaling(
            experiment: "Experiment",
            df: pd.DataFrame,
            step: Optional[int] = None,
            **kwargs,
        ):
            return _loss_scaling(
                experiment,
                df,
                step=step,
                metrics=[
                    ("L_compare_train", r"L_\mathrm{cf. train}"),
                    ("L_compare_test", r"L_\mathrm{cf. test}"),
                    ("acc_compare_train", r"\mathrm{acc}_\mathrm{cf. train}"),
                    ("acc_compare_test", r"\mathrm{acc}_\mathrm{cf. test}"),
                ],
                **kwargs,
            )

        def loss_true_scaling(
            experiment: "Experiment",
            df: pd.DataFrame,
            step: Optional[int] = None,
            **kwargs,
        ):
            return _loss_scaling(
                experiment,
                df,
                step=step,
                metrics=[
                    ("L_train", r"L_\mathrm{train}"),
                    ("L_test", r"L_\mathrm{test}"),
                    ("acc_train", r"\mathrm{acc}_\mathrm{train}"),
                    ("acc_test", r"\mathrm{acc}_\mathrm{test}"),
                ],
                **kwargs,
            )

        kwargs = {
            "batch_size": batch_size,
            "logging_ivl": logging_ivl,
            "plot_ivl": plot_ivl,
            "save_ivl": save_ivl,
            "seed_dl": seed_dl,
            # "dl_hyperparams": dl_hyperparams,
            "model_hyperparams": model_hyperparams,
            "opt_hyperparams": opt_hyperparams,
            "weight_initializer_hyperparams": weight_initializer_hyperparams,
            "baseline": baseline,
            **experiment,
        }

        experiment = Experiment(
            train_data=train_data,
            test_data=test_data,
            **kwargs,
            plot_fns=(
                epsilon_scaling,
                corr_scaling,
                cos_sim_scaling,
                loss_cf_scaling,
                loss_true_scaling,
            ),
        )  # type: ignore

        yield experiment


def gen_default_weight_initializer_hyperparams(n_perturbed=10, epsilon=0.01):
    return [{**DEFAULT_WEIGHT_INITIALIZER_HYPERPARAMS, "epsilon": 0.0}] + [
        {
            **DEFAULT_WEIGHT_INITIALIZER_HYPERPARAMS,
            "epsilon": epsilon,
            "seed_perturbation": seed,
        }
        for seed in range(n_perturbed)
    ]


def gen_epsilon_range(n_samples: int = 10, epsilons=(0.001, 0.01, 0.1, 1.0)):
    return [
        {
            **DEFAULT_WEIGHT_INITIALIZER_HYPERPARAMS,
            "epsilon": 0,
            "seed_perturbation": 0,
            "seed_weights": 1,
        }
    ] + [
        {
            **DEFAULT_WEIGHT_INITIALIZER_HYPERPARAMS,
            "epsilon": epsilon,
            "seed_perturbation": seed,
            "seed_weights": 1,
        }
        for epsilon in epsilons
        for seed in range(n_samples)
    ]


experiments = [
    {
        # 0 Vanilla
        "weight_initializer_hyperparams": gen_epsilon_range(10),
        "dir": f"vanilla",
    },
    {
        # 1 Depth
        "weight_initializer_hyperparams": gen_default_weight_initializer_hyperparams(),
        "model_hyperparams": [
            {"n_hidden": n_hidden} for n_hidden in tuple((50,) * i for i in range(1, 6))
        ],
        "comparison": f"n_hidden",
        "dir": f"depth",
    },
    {
        # 2 Width
        "weight_initializer_hyperparams": gen_default_weight_initializer_hyperparams(),
        "model_hyperparams": [
            {"n_hidden": n_hidden} for n_hidden in (400, 200, 100, 50, 25)
        ],
        "comparison": f"n_hidden",
        "dir": f"width",
    },
    {
        # 3 Momentum
        "weight_initializer_hyperparams": gen_default_weight_initializer_hyperparams(),
        "opt_hyperparams": [
            {**DEFAULT_SGD_HYPERPARAMS, "momentum": momentum}
            for momentum in (0.0, 0.1, 0.5, 0.9)
        ],
        "comparison": f"momentum",
    },
    {
        # 4 Weight decay
        "epsilon": (0.0, 0.01),
        "weight_decay": (0.0, 0.001, 0.01, 0.1, 0.5, 0.9),
        "comparison": "weight_decay",
    },
    {
        # 5 Learning rate (TODO: compare over normalized time)
        "weight_initializer_hyperparams": gen_default_weight_initializer_hyperparams(),
        "opt_hyperparams": [
            {**DEFAULT_SGD_HYPERPARAMS, "lr": lr} for lr in (0.1, 0.01, 0.001, 0.0001)
        ],
        "comparison": "lr",
    },
    {
        # 6 Optimizer
        "model_hyperparams": {
            "cls": FCN,
            "n_hidden": 100,
        },
        "opt_hyperparams": {
            "cls": t.optim.Adam,
            "lr": 0.001,
            "betas": (0.9, 0.999),
            "eps": 1e-08,
            "weight_decay": 0,
        },
        "weight_initializer_hyperparams": gen_epsilon_range(),
        "dir": "adam/2",
    },
    {
        # 7 Convnets
        "model_hyperparams": {
            "cls": Lenet5,
        },
        "weight_initializer_hyperparams": gen_epsilon_range(),
        "dir": f"lenet5/1",
    },
    {
        # 8 Other datasets
        "dataset": "cifar10",
        "opt_hyperparams": {
            "cls": t.optim.Adam,
            "lr": 0.001,
            "betas": (0.9, 0.999),
            "eps": 1e-08,
            "weight_decay": 0,
        },
        "model_hyperparams": {"cls": ResNet, "n_layers": 18},
        "weight_initializer_hyperparams": gen_epsilon_range(5),
        "dir": f"resnet/1",
    },
    {
        # 0 More Vanilla
        "weight_initializer_hyperparams": gen_epsilon_range(5, epsilons=(1.0,)),
        "opt_hyperparams": {
            **DEFAULT_SGD_HYPERPARAMS,
            "lr": 0.1,
        },
        "dir": f"vanilla-long",
    },
]

if __name__ == "__main__":
    experiments = [experiments[-1]]

    for experiment in tqdm(
        make_ensembles(
            experiments,
            logging_ivl=1000,
            plot_ivl=2000,
            save_ivl=1000,
        ),
        desc="Running experiments...",
        total=len(experiments),
    ):
        experiment.train(n_epochs=200)
