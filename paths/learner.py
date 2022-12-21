from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import (Callable, ClassVar, Collection, Container, Dict, Iterable,
                    List, Optional, Sequence, Tuple, Type, Union)

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
from tqdm import tqdm

from serimats.paths.models import ExtendedModule
from serimats.paths.utils import (Logger, Plotter, Snapshotter, dict_to_latex,
                                  var_to_latex)
from serimats.paths.weights import (AbsolutePerturbationInitializer,
                                    RelativePerturbationInitializer,
                                    WeightInitializer)

# from tqdm.notebook import tqdm


@dataclass
class MultiLearner:
    """
    Train several models in parallel, and record a collection of metrics on those models
    throughout the training process.
    """

    __model__: Type[ExtendedModule]
    train_loader: DataLoader
    test_loader: DataLoader
    n_models: int = 10
    config: Optional[dict] = None
    lr: float = 0.01
    momentum: float = 0.0
    weight_decay: float = 0.0
    weight_initializer: Optional[WeightInitializer] = None
    loss: Callable = F.nll_loss
    epochs: int = 10

    __metrics__: ClassVar[List[str]] = [
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
        "rel_d_w",
    ]

    def __post_init__(self):
        self.models = [self.__model__(self.config) for _ in range(self.n_models)]
        self.optimizers = [
            optim.SGD(
                m.parameters(),
                lr=self.lr,
                momentum=self.momentum,
                weight_decay=self.weight_decay,
            )
            for m in self.models
        ]

        self.logs = pd.DataFrame(
            columns=["epoch", "batch", "step", "model_idx", *self.__metrics__]
        )

        # Initialize weights
        if self.weight_initializer:
            self.weight_initializer(*self.models)

    def train(
        self,
        epoch_start: int = 0,
        **kwargs,
    ):
        self.update(**kwargs)
        losses = [0.0] * self.n_models

        for epoch in range(epoch_start, self.epochs):
            for batch_idx, (data, target) in tqdm(
                enumerate(self.train_loader), desc=f"Epoch: {epoch}"
            ):
                for i, (model, optimizer) in enumerate(self):
                    optimizer.zero_grad()
                    output = model(data)
                    loss = self.loss(output, target)
                    loss.backward()
                    optimizer.step()
                    losses[i] = float(loss.item())

                yield epoch, batch_idx, losses

    def measure(self) -> List[Dict[str, Union[t.Tensor, int, float]]]:
        d_ws, rel_dw = self.d_ws_with_rel()
        test_results = [*self.test(), d_ws.detach().numpy(), rel_dw.detach().numpy()]

        return [
            dict(zip(self.__metrics__[2:], test_result))
            for test_result in zip(test_results)
        ]

    def test(
        self,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Returns the loss and accuracy averaged over the entire test set."""
        L_compare = np.zeros((self.n_models,))
        acc_compare = np.zeros((self.n_models,))

        L_test = np.zeros((self.n_models,))
        acc_test = np.zeros((self.n_models,))

        with t.no_grad():
            for data, target in self.test_loader:
                baseline_output = self.models[0](data)
                baseline_pred = baseline_output.argmax(dim=1, keepdim=False)

                for i, model in enumerate(self.models):
                    output = model(data)
                    L_test[i] += self.loss(output, target, reduction="sum").item()

                    pred = output.argmax(dim=1, keepdim=True)
                    acc_test[i] += pred.eq(target.view_as(pred)).sum().item()

                    L_compare[i] += self.loss(
                        output, baseline_pred, reduction="sum"
                    ).item()
                    acc_compare[i] += pred.eq(baseline_pred.view_as(pred)).sum().item()

        L_test /= self.n_test_samples
        acc_test = acc_test / self.n_test_samples
        L_compare /= self.n_test_samples
        acc_compare = acc_compare / self.n_test_samples

        del_L_test = L_test - L_test[0]
        del_acc_test = acc_test - acc_test[0]

        return L_test, del_L_test, acc_test, del_acc_test, L_compare, acc_compare

    def d_w(self, model: nn.Module) -> t.Tensor:
        return t.norm(model.parameters_vector - self.reference.parameters_vector)

    def d_ws(self) -> t.Tensor:
        return t.tensor([self.d_w(model) for model, _ in self])

    def d_ws_with_rel(self) -> Tuple[t.Tensor, t.Tensor]:
        d_ws = self.d_ws()
        w_ref_norm = self.reference.parameters_norm
        rel_d_ws = d_ws / w_ref_norm

        return d_ws, rel_d_ws

    def update(self, **kwargs):
        for k, v in kwargs.items():
            if hasattr(self, k):
                setattr(self, k, v)

    @property
    def params(self) -> dict:
        params = {
            k: getattr(self, k)
            for k in [
                "n_models",
                "lr",
                "momentum",
                "weight_decay",
                "epochs",
            ]
        }

        if self.config:
            params.update(self.config)

        params["weight_initializer"] = repr(self.weight_initializer)

        if isinstance(
            self.weight_initializer,
            (RelativePerturbationInitializer, AbsolutePerturbationInitializer),
        ):
            params["epsilon"] = self.weight_initializer.epsilon

        return params

    def get_model_params(self, model_idx: int) -> dict:
        return dict(
            model_idx=model_idx,
            lr=self.lr,
            momentum=self.momentum,
            weight_decay=self.weight_decay,
            weight_initializer=repr(self.weight_initializer),
            **self.models[model_idx].config,
        )

    @property
    def reference(self):
        return self.models[0]

    @property
    def n_test_samples(self):
        return len(self.test_loader.dataset)  # type: ignore

    @property
    def n_train_samples(self):
        return len(self.train_loader.dataset)  # type: ignore

    def __iter__(self):
        return zip(self.models, self.optimizers)


class MultiLearnerWithLoggingAndSaves:
    def __init__(
        self,
        learner: MultiLearner,
        snapshotter: Snapshotter,
        plotter: Plotter,
        logger: Logger,
        logging_ivl: int = 100,
    ):
        self.learner = learner
        self.snapshotter = snapshotter
        self.plotter = plotter
        self.logger = logger
        self.logging_ivl = logging_ivl

    def train(self, epoch_start: int = 0, **kwargs):
        self.logger.init(**kwargs)
        self.load(epoch_start, **kwargs)

        for (epoch, batch_idx, losses) in self.learner.train(
            epoch_start=epoch_start, **kwargs
        ):
            if batch_idx % self.logging_ivl == 0:
                measurements = self.learner.measure()

                for i, _ in enumerate(self.models):
                    self.logger.log(
                        epoch=epoch,
                        batch=batch_idx,
                        step=epoch * len(self.learner.train_loader) + batch_idx,
                        model_idx=i,
                        L_train=losses[i],
                        del_L_train=losses[i] - losses[0],
                        **measurements[i],
                    )

            if batch_idx == 0:
                df = self.save(epoch=epoch, **kwargs)
                self.plot(df, epoch=epoch, **self.params, **kwargs)

        df = self.save(epoch=self.learner.epochs, **kwargs)
        self.plot(df, **self.params, **kwargs)

    def save(self, epoch: int, **kwargs):
        df = self.logger.save()

        for i, model in enumerate(self.models):
            self.snapshotter.save(
                model,
                epoch=epoch,
                **self.learner.get_model_params(i),
                **kwargs,
            )

            print(f"Saved model {i} at epoch {epoch}.")

        return df

    def load(self, epoch: int, **kwargs):
        for i, model in enumerate(self.models):
            self.snapshotter.load(
                model,
                epoch=epoch,
                **self.learner.get_model_params(i),
                **kwargs,
            )

            print(f"Loaded model {i} at epoch {epoch}.")

    def plot(self, df: pd.DataFrame, **kwargs):
        self.plotter.plot(df, **kwargs)

    @property
    def models(self):
        return self.learner.models

    @property
    def params(self):
        params = self.learner.params

        del params["n_models"]
        del params["epochs"]

        return params
