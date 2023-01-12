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
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms
from tqdm import tqdm

from serimats.paths.metrics import (Metrics, cos_sim_from_baseline,
                                    cos_sim_from_init, d_w_from_baseline,
                                    d_w_from_baseline_normed, d_w_from_init,
                                    d_w_from_init_normed, w_autocorr,
                                    w_corr_with_baseline, w_normed)
from serimats.paths.models import MNIST, ExtendedModule, Lenet5
from serimats.paths.plots import plot_metric_scaling
from serimats.paths.utils import (CallableWithLatex, OptionalTuple,
                                  dict_to_latex, setup, stable_hash, to_tuple,
                                  var_to_latex)
from serimats.paths.weights import (AbsolutePerturbationInitializer,
                                    RelativePerturbationInitializer,
                                    WeightInitializer)

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
        opt: optim.Optimizer,
        weight_initializer: WeightInitializer,
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
        model_hyperparams: dict = {},
        weight_initializer_hyperparams: dict = {},
        opt_hyperparams: dict = {},
        **kwargs,
    ):
        model_cls = model_hyperparams.pop("cls")
        opt_cls = opt_hyperparams.pop("cls")
        weight_initializer_cls = weight_initializer_hyperparams.pop("cls")

        model = model_cls(**model_hyperparams)
        opt = opt_cls(model.parameters(), **opt_hyperparams) 
        weight_initializer = weight_initializer_cls(**weight_initializer_hyperparams)

        return cls(
            model=model, 
            opt=opt, 
            weight_initializer=weight_initializer, 
            **kwargs
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
            hyperparams = self.hyperparams
            hyperparams["n_hidden"] = [hyperparams["n_hidden"]] * len(df)

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



class Ensemble:
    def __init__(
        self,
        train_data: Dataset,
        test_data: Dataset,
        seed_dl: int = 0,
        dir: str = "results",
        batch_size: int = 64,
        logging_ivl: int = 100,
        plot_fns: Optional[OptionalTuple[Callable[..., Tuple[Figure, plt.Axes]]]] = None,
        plot_ivl: int = 5000,
        save_ivl: int = 2000,
        opt_hyperparams: OptionalTuple[Dict[str, Any]] = {},
        model_hyperparams: OptionalTuple[Dict[str, Any]] = {},
        weight_initializer_hyperparams: OptionalTuple[Dict[str, Any]] = {},
        hyperparams: Optional[List[Tuple[Dict[str, Any], ...]]] = None,
        baseline: dict = {"epsilon": 0., "seed_perturbation": 0}
    ):
        self.dir = Path(dir)
        self.save_ivl = save_ivl
        self.logging_ivl = logging_ivl
        self.plot_ivl = plot_ivl
        self.plot_fns = to_tuple(plot_fns) if plot_fns else []

        self.seed_dl = seed_dl
        self.train_dl = DataLoader(train_data, batch_size=batch_size, shuffle=True)
        self.test_dl = DataLoader(test_data, batch_size=batch_size, shuffle=False)
        self.metrics = Metrics(self.train_dl, self.test_dl)
        
        # There are two ways to specify hyperparams:
        # 1. Fully specify a list of hyperparams (tuples of opt, model, weight_initializer hyperparams). 
        #    Then the kwargs (opt_hyperparams, model_hyperparams, weight_initializer_hyperparams) are used as defaults.
        # 2. Specify a list of hyperparams for each of opt, model, weight_initializer, then take the product
 
        if hyperparams is not None:
            assert isinstance(opt_hyperparams, (dict))
            assert isinstance(model_hyperparams, (dict))
            assert isinstance(weight_initializer_hyperparams, (dict))

            self.hyperparams = [
                (
                    {**opt_hyperparams, **o}, 
                    {**model_hyperparams, **m}, 
                    {**weight_initializer_hyperparams, **w}
                ) for o, m, w in hyperparams
            ]
        else:
            opt_hyperparams = to_tuple(opt_hyperparams)
            model_hyperparams = to_tuple(model_hyperparams)
            weight_initializer_hyperparams = to_tuple(weight_initializer_hyperparams)

            self.hyperparams = [
                (o.copy(), m.copy(), w.copy())
                for o, m, w in itertools.product(
                    model_hyperparams,
                    opt_hyperparams,
                    weight_initializer_hyperparams,
                )
            ]

        self.learners = [
            Learner.from_hyperparams(
                dir=dir,
                model_hyperparams=model_hyperparams,  # type: ignore
                opt_hyperparams=opt_hyperparams,  # type: ignore
                weight_initializer_hyperparams=weight_initializer_hyperparams,  # type: ignore
            )
            for model_hyperparams, opt_hyperparams, weight_initializer_hyperparams in self.hyperparams
        ] 

        # Add baselines
        self.baseline = baseline

        for learner in self.learners:
            hp = learner.hyperparams.copy()
            hp.update(self.baseline)
            learner.baseline = next(self.filter_learners(**hp))
 
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

                if step % self.plot_ivl == 0: # and step > 0:
                    self.plot(step=step)

                if step % self.save_ivl == 0:
                    self.save(step=step)

    def test(self, step, **kwargs):
        learners = [learner for learner in self.learners if step not in learner.logs]

        for metric, learner in tqdm(
            zip(self.metrics.measure(learners), learners), desc=f"Testing {step}"
        ):
            learner.log(**metric, **kwargs)

    def plot(self, step: Optional[int] = None, overwrite: bool = False):
        df = self.df()
        fig_dir = self.dir / "img"
        fig_dir.mkdir(parents=True, exist_ok=True)

        for plot_fn in self.plot_fns:
            fig, ax = plot_fn(self, df, step=step, baseline=self.baseline, **self.fixed_hyperparams)

            # Save the figure
            if fig is not None or overwrite:
                fig.savefig(
                    str(fig_dir / f"{plot_fn.__name__}_{step}.png"),
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


def get_mnist_data():
    train_ds = datasets.MNIST(
        root="data", train=True, download=True, transform=transforms.ToTensor()
    )
    test_ds = datasets.MNIST(
        root="data", train=False, download=True, transform=transforms.ToTensor()
    )

    return train_ds, test_ds


train_data, test_data = get_mnist_data()


DEFAULT_MODEL_HYPERPARAMS = dict(
    cls=MNIST,
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


DEFAULT_ENSEMBLE_KWARGS = dict(
    train_data=train_data,
    test_data=test_data,
    batch_size=64,
    logging_ivl=100,
    plot_ivl=2000,
    save_ivl=2000,
    seed_dl=0,
    # dl_hyperparams=DEFAULT_DL_HYPERPARAMS,
    model_hyperparams=DEFAULT_MODEL_HYPERPARAMS,
    opt_hyperparams=DEFAULT_SGD_HYPERPARAMS,
    weight_initializer_hyperparams=DEFAULT_WEIGHT_INITIALIZER_HYPERPARAMS,
    baseline={"epsilon": 0., "seed_perturbation": 0}
)

def gen_default_weight_initializer_hyperparams(n_perturbed=10, epsilon=0.01):
    return [
        { **DEFAULT_WEIGHT_INITIALIZER_HYPERPARAMS, "epsilon": 0. }
    ] + [
        { **DEFAULT_WEIGHT_INITIALIZER_HYPERPARAMS, "epsilon": epsilon, "seed_perturbation": seed }
        for seed in range(n_perturbed)
    ]

experiment_group = "mnist"

experiments = [
    # {
    #     "weight_initializer_hyperparams": [
    #         {
    #             **DEFAULT_WEIGHT_INITIALIZER_HYPERPARAMS,
    #             "seed_perturbation": seed,
    #         } for seed in range(10)
    #     ],
    #     "dir": f"results/{experiment_group}/vanilla"
    # },
    # {
    #     # Depth
    #     "weight_initializer_hyperparams": gen_default_weight_initializer_hyperparams(),
    #     "model_hyperparams": [
    #         {"n_hidden": n_hidden}
    #         for n_hidden in (50, (50,) * 2, (50,) * 3, (50,) * 4, (50,) * 5,)
    #     ]
    #     "dir": f"results/{experiment_group}/depth"
    # },
    # {
    #     # Width
    #     "weight_initializer_hyperparams": gen_default_weight_initializer_hyperparams(),
    #     "model_hyperparams": [
    #         {"n_hidden": n_hidden}
    #         for n_hidden in (400, 200, 100, 50, 25)
    #     ]
    #     "dir": f"results/{experiment_group}/width"
    # },
    # {
    #     # Momentum
    #     "epsilon": (0.0, 0.01),
    #     "momentum": (0.0, 0.1, 0.5, 0.9),
    #     "dir": "results/mnist/momentum"
    # },
    # {
    #     # Weight decay
    #     "epsilon": (0.0, 0.01),
    #     "weight_decay": (0.0, 0.001, 0.01, 0.1, 0.5, 0.9),
    #     "dir": "results/mnist/weight_decay",
    #     "comparison": "lr"
    # },
    {
        # Learning rate (TODO: compare over normalized time)
        "weight_initializer_hyperparams": gen_default_weight_initializer_hyperparams(),
        "opt_hyperparams": [
            { **DEFAULT_SGD_HYPERPARAMS, "lr": lr}
            for lr in (0.1, 0.01, 0.001, 0.0001)
        ],
        "dir": f"results/{experiment_group}/lr",
        "comparison": "lr"
    },
    # {
    #     # Optimizer
    #     "opt_cls": t.optim.Adam,
    # },
    {
        # Convnets
        "model_hyperparams": {
            **DEFAULT_MODEL_HYPERPARAMS,
            "cls": Lenet5,
        },
        "weight_initializer_hyperparams": [
            {
                **DEFAULT_WEIGHT_INITIALIZER_HYPERPARAMS,
                "seed_perturbation": seed,
            } for seed in range(10)
        ],
        "dir": f"results/{experiment_group}/lenet5"
    },
]

for experiment in experiments:
    comparison = experiment.pop("comparison", "epsilon")

    def epsilon_scaling(
        ensemble: "Ensemble", df: pd.DataFrame, step: Optional[int] = None, **kwargs
    ):
        return plot_metric_scaling(
            ensemble,
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
        ensemble: "Ensemble", df: pd.DataFrame, step: Optional[int] = None, **kwargs
    ):
        return plot_metric_scaling(
            ensemble,
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
        ensemble: "Ensemble", df: pd.DataFrame, step: Optional[int] = None, **kwargs
    ):
        return plot_metric_scaling(
            ensemble,
            # df,
            df.loc[df["epsilon"] > 0.0],
            step=step,
            metric=(cos_sim_from_baseline, cos_sim_from_init),
            comparison=comparison,
            sample_axis="seed_perturbation",
            include_baseline=True,
            **kwargs,
        )


    def loss_scaling(
        ensemble: "Ensemble", df: pd.DataFrame, step: Optional[int] = None, **kwargs
    ):
        def mock_metric(name, latex_name, latex_body="") -> CallableWithLatex:
            def metric(*args, **kwargs):
                pass

            metric.__name__ = name
            metric.__latex__ = (latex_name, latex_body)

            return metric  # type: ignore

        L_compare_train = mock_metric("L_compare_train", r"L_\mathrm{cf. train}")
        L_compare_test = mock_metric("L_compare_test", r"L_\mathrm{cf. test}")

        acc_compare_train = mock_metric("acc_compare_train", r"acc_\mathrm{cf. train}")
        acc_compare_test = mock_metric("acc_compare_test", r"acc_\mathrm{cf. test}")

        return plot_metric_scaling(
            ensemble,
            # df,
            df.loc[df["epsilon"] > 0.0],
            step=step,
            metric=(L_compare_train, L_compare_test, acc_compare_train, acc_compare_test),
            comparison=comparison,
            sample_axis="seed_perturbation",
            include_baseline=True,
            **kwargs,
        )

    kwargs = {
        **DEFAULT_ENSEMBLE_KWARGS,
        **experiment
    }

    ensemble = Ensemble(
        **kwargs,
        plot_fns=(epsilon_scaling, corr_scaling, cos_sim_scaling, loss_scaling)
    ) # type: ignore
    ensemble.train(20)


# %%
