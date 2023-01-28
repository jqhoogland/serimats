# %%

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
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
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from serimats.paths.models import FCN
from serimats.paths.trainer import EnsembleLearner, Experiment
from serimats.paths.utils import (
    Logger,
    Plotter,
    Snapshotter,
    dict_to_latex,
    divide_params,
    setup,
    tqdm,
    var_to_latex,
)
from serimats.paths.weights import (
    AbsolutePerturbationInitializer,
    RelativePerturbationInitializer,
    WeightInitializer,
)

device = setup()

train_loader = DataLoader(
    datasets.FCN(
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
    datasets.FCN(
        "../data",
        train=False,
        transform=transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
        ),
    ),
    shuffle=True,
)


def run(
    train_loader,
    test_loader,
    snapshotter,
    plotter,
    logger,
    logging_ivl=100,
    model_cls=FCN,
    opt_cls=optim.SGD,
    **kwargs,
):
    # Recursive call for any parameters which are provided as sequences
    for k, v in kwargs.items():
        if isinstance(v, (tuple, list)):
            for subvalue in tqdm(v, desc=f"Iterating over {k}={v}"):
                run(
                    train_loader,
                    test_loader,
                    snapshotter,
                    plotter,
                    logger,
                    logging_ivl=logging_ivl,
                    model_cls=model_cls,
                    opt_cls=opt_cls,
                    **{**kwargs, k: subvalue},
                )

    ensemble = EnsembleLearner(
        model_cls,
        opt_cls,
        train_loader,
        test_loader,
        **kwargs,
    )

    experiment = Experiment(
        ensemble=ensemble,
        snapshotter=snapshotter,
        plotter=plotter,
        logger=logger,
        logging_ivl=logging_ivl,
    )

    experiment.train(epoch_start=2)


# DIR_PATH = Path("/content/drive/My Drive/AI/SERIMATS/Path Dependence")
DIR_PATH = Path("..")
SNAPSHOTS_PATH = DIR_PATH / "snapshots"
LOGS_PATH = DIR_PATH / "logs"

N_EPOCHS = 25
N_MODELS = 10
LOGGING_IVL = 100

# Variable
SEED = 0
EPSILON = 0.01
MOMENTUM = 0.0
LR = 0.01
WEIGHT_DECAY = 0.0


def get_opt_hyperparams(lr=LR, momentum=MOMENTUM, weight_decay=WEIGHT_DECAY):
    if isinstance(lr, (tuple, list)):
        return [get_opt_hyperparams(lr=lr_) for lr_ in lr]
    if isinstance(momentum, (tuple, list)):
        return [get_opt_hyperparams(momentum=m_) for m_ in momentum]
    if isinstance(weight_decay, (tuple, list)):
        return [get_opt_hyperparams(weight_decay=wd_) for wd_ in weight_decay]

    return {"lr": lr, "momentum": momentum, "weight_decay": weight_decay}


MODEL_HYPERPARAMS = {"n_hidden": 100}

snapshotter = Snapshotter(SNAPSHOTS_PATH)
logger = Logger(LOGS_PATH)
plotter = Plotter()  # TODO Save images as well

# %%
epsilon_results = run(
    train_loader,
    test_loader,
    snapshotter=snapshotter,
    plotter=plotter,
    logger=logger,
    logging_ivl=LOGGING_IVL,
    seed=SEED,
    weight_initializer=[
        RelativePerturbationInitializer(epsilon=epsilon)
        for epsilon in [0.001, 0.01, 0.1, 1.0, 10.0]
    ],
    n_epochs=N_EPOCHS,
    n_models=N_MODELS,
    opt_hyperparams=get_opt_hyperparams(),
    model_hyperparams=MODEL_HYPERPARAMS,
)

# %%

lr_experiments = run(
    train_loader,
    test_loader,
    snapshotter=snapshotter,
    plotter=plotter,
    logger=logger,
    epsilon=0.01,
    lr=[0.001, 0.01, 0.1],
    seed=0,
    logging_interval=1000,
)

# %%

momentum_experiments = run(
    train_loader,
    test_loader,
    snapshotter=snapshotter,
    plotter=plotter,
    logger=logger,
    epsilon=0.01,
    momentum=[0.001, 0.01, 0.1],
    seed=0,
    logging_interval=1000,
)

# %%

weight_decay_experiments = run(
    train_loader,
    test_loader,
    snapshotter=snapshotter,
    plotter=plotter,
    logger=logger,
    epsilon=0.01,
    lr=0.01,
    momentum=0.01,
    weight_decay=[0.001, 0.01, 0.1],
    seed=0,
    logging_interval=1000,
)

# %%

seed_experiments = run(
    train_loader,
    test_loader,
    snapshotter=snapshotter,
    plotter=plotter,
    logger=logger,
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

width_experiments = run(
    train_loader,
    test_loader,
    snapshotter=snapshotter,
    plotter=plotter,
    logger=logger,
    epsilon=0.01,
    lr=0.01,
    momentum=0.01,
    weight_decay=0.01,
    seed=0,
    logging_interval=1000,
    config=[{"n_hidden": n_h} for n_h in [10, 50, 100, 500, 1000]],
)


depth_experiments = run(
    train_loader,
    test_loader,
    snapshotter=snapshotter,
    plotter=plotter,
    logger=logger,
    epsilon=0.01,
    lr=0.01,
    momentum=0.01,
    weight_decay=0.01,
    seed=0,
    logging_interval=1000,
    config=[{"n_hidden": divide_params(50000, i, 784, 10)} for i in range(1, 6)],
)
