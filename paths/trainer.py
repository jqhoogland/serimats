from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import (
    Any,
    Callable,
    ClassVar,
    Collection,
    Container,
    Dict,
    Iterable,
    Iterator,
    List,
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
from tqdm import tqdm

from serimats.paths.models import MNIST, ExtendedModule
from serimats.paths.utils import (
    Logger,
    Plotter,
    Snapshotter,
    dict_to_latex,
    var_to_latex,
)
from serimats.paths.weights import (
    AbsolutePerturbationInitializer,
    RelativePerturbationInitializer,
    WeightInitializer,
)

# from tqdm.notebook import tqdm


def get_opt_hyperparams(opt: optim.Optimizer) -> dict:
    """Assumes that the optimizer has only a single set of hyperparameters
    (and not a separate set for each parameter group)"""
    param_group = opt.param_groups[0]

    hyperparams = {}

    for k, v in param_group.items():
        if k not in ["params", "foreach", "maximize", "capturable", "fused"]:
            hyperparams[k] = v

    return hyperparams


@dataclass
class Learner:
    """
    Wraps a model and optimizer and provides a method for taking an update step.

    """

    model: ExtendedModule
    optimizer: optim.Optimizer
    loss: Callable = F.nll_loss
    extra_hyperparams: dict = field(default_factory=dict)

    def measure_one_batch(
        self, data: t.Tensor, target: t.Tensor
    ) -> Tuple[t.Tensor, t.Tensor, Tuple[t.Tensor, ...]]:
        output = self.model(data)
        loss = self.loss(output, target)
        pred = output.argmax(dim=1)
        acc = (pred == target).float().mean()
        return output, pred, (loss, acc)

    def train_one_batch(
        self, data: t.Tensor, target: t.Tensor
    ) -> Tuple[t.Tensor, t.Tensor, Tuple[t.Tensor, ...]]:
        self.optimizer.zero_grad()
        output, pred, (loss, acc) = self.measure_one_batch(data, target)
        loss.backward()
        self.optimizer.step()

        return output, pred, (loss, acc)

    def test_one_batch(
        self, data: t.Tensor, target: t.Tensor
    ) -> Tuple[t.Tensor, t.Tensor, Tuple[t.Tensor, ...]]:
        with t.no_grad():
            results = self.measure_one_batch(data, target)

        return results

    @property
    def hyperparams(self) -> dict:
        return {
            **get_opt_hyperparams(self.optimizer),
            **self.model.hyperparams,
            **self.extra_hyperparams,
            "loss_fn": self.loss.__name__,
        }

    def __getattribute__(self, __name: str) -> Any:
        if __name in self.hyperparams:
            return self.hyperparams[__name]


class Metrics:
    pass


class EnsembleLearner:
    """
    Trains several models in parallel and records a collection of metrics on those models
    throughout the training process.
    """

    model_cls: Type[ExtendedModule]
    opt_cls: Type[optim.Optimizer]

    train_loader: DataLoader
    test_loader: DataLoader

    n_models: int = 10
    n_epochs: int = 10

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

    def __init__(
        self,
        model_cls: Type[ExtendedModule],
        opt_cls: Type[optim.Optimizer],
        train_loader: DataLoader,
        test_loader: DataLoader,
        weight_initializer: Optional[WeightInitializer] = None,
        n_models: int = 10,
        n_epochs: int = 10,
        seed: int = 0,
        model_hyperparams: Optional[Union[dict, List[dict]]] = None,
        opt_hyperparams: Optional[Union[dict, List[dict]]] = None,
    ):

        self.model_cls = model_cls
        self.opt_cls = opt_cls

        self.train_loader = train_loader
        self.test_loader = test_loader
        self.n_epochs = n_epochs
        self.n_models = n_models

        self.seed = seed
        t.manual_seed(seed)

        if isinstance(model_hyperparams, list):
            assert len(model_hyperparams) == n_models
            models = [self.model_cls(hyperparams) for hyperparams in model_hyperparams]
        else:
            models = [self.model_cls(model_hyperparams) for _ in range(self.n_models)]

        if weight_initializer is not None:
            for model in models:
                weight_initializer(model)
                model.hyperparams.update(weight_initializer.hyperparams)

        if isinstance(opt_hyperparams, list):
            assert len(opt_hyperparams) == n_models
            optimizers = [
                self.opt_cls(model.parameters(), **hyperparams)
                for model, hyperparams in zip(models, opt_hyperparams)
            ]
        else:
            optimizers = [
                self.opt_cls(model.parameters(), **opt_hyperparams)  # type: ignore
                for model in models
            ]

        self.learners = [
            Learner(
                model=model,
                optimizer=optimizer,
                extra_hyperparams={"model_idx": i},
            )
            for i, (model, optimizer) in enumerate(zip(models, optimizers))
        ]

    def train_one_batch(self, data: t.Tensor, target: t.Tensor):
        for model_idx, learner in enumerate(self.learners):
            output, loss, acc = learner.train_one_batch(data, target)

            yield model_idx, output, loss, acc

    def train_batches(self, epoch: int = 0):
        for batch_idx, (data, target) in tqdm(
            enumerate(self.train_loader), desc=f"Epoch: {epoch}"
        ):
            yield batch_idx, data, target, self.train_one_batch(data, target)

    def train_epochs(self, epoch_start: int = 0):
        for epoch in range(epoch_start, self.n_epochs):
            yield epoch, self.train_batches(epoch)

    def train(
        self,
        epoch_start: int = 0,
    ):
        """
        I hate control flow that is based on exceptions and callbacks (like FastAI's).
        Instead, I recommend implementing a custom training loop (using train_epochs, train_batches, train_one_batch)
        """
        for epoch, batches in self.train_epochs(epoch_start):
            for batch_idx, data, target, steps in batches:
                for i, output, pred, metrics in steps:
                    pass

    def test_one_batch(self, data: t.Tensor, target: t.Tensor):
        baseline_output = self.baseline.model(data)
        baseline_pred = baseline_output.argmax(dim=1, keepdim=False)

        for model_idx, learner in enumerate(self.learners):
            output, pred, L_test, acc_test = learner.test_one_batch(data, target)

            L_compare = learner.loss(output, baseline_pred).item()
            acc_compare = pred.eq(baseline_pred.view_as(pred)).mean().item()

            yield model_idx, output, pred, L_test, acc_test, L_compare, acc_compare

    def test_batches(self):
        for batch_idx, (data, target) in enumerate(self.test_loader):
            yield batch_idx, data, target, self.test_one_batch(data, target)

    def test(
        self,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Returns the loss and accuracy averaged over the entire test set."""
        for batch_idx, (data, target) in enumerate(self.test_loader):
            for model_idx, learner in enumerate(self.learners):
                loss, acc = learner.test_one_batch(data, target)

                yield model_idx, loss, acc

    def inspect(self) -> List[Dict[str, Union[t.Tensor, int, float]]]:
        d_ws, rel_dw = self.d_ws_with_rel()
        test_results = (*self.test(), d_ws.detach().numpy(), rel_dw.detach().numpy())

        return [
            dict(zip(self.__metrics__[2:], test_result))
            for test_result in zip(*test_results)
        ]

    def test(
        self,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Returns the loss and accuracy averaged over the entire test set."""

        # TODO: Something a little more elegant than this
        L_compare = np.zeros((self.n_models,))
        acc_compare = np.zeros((self.n_models,))
        L_test = np.zeros((self.n_models,))
        acc_test = np.zeros((self.n_models,))

        with t.no_grad():
            for data, target in self.test_loader:
                baseline_output = self.baseline.model(data)
                baseline_pred = baseline_output.argmax(dim=1, keepdim=False)

                for i, learner in enumerate(self.learners):
                    output = learner.model(data)
                    L_test[i] += learner.loss(output, target, reduction="sum").item()

                    pred = output.argmax(dim=1, keepdim=True)
                    acc_test[i] += pred.eq(target.view_as(pred)).sum().item()

                    L_compare[i] += learner.loss(
                        output, baseline_pred, reduction="sum"
                    ).item()
                    acc_compare[i] += pred.eq(baseline_pred.view_as(pred)).sum().item()

        # Averages
        L_test /= self.n_test_samples
        acc_test /= self.n_test_samples
        L_compare /= self.n_test_samples
        acc_compare /= self.n_test_samples

        # Fill in delta metrics
        del_L_test = L_test - L_test[0]
        del_acc_test = acc_test - acc_test[0]

        return L_test, del_L_test, acc_test, del_acc_test, L_compare, acc_compare

    def d_w(self, model: nn.Module) -> t.Tensor:
        return t.norm(model.parameters_vector - self.baseline.model.parameters_vector)

    def d_ws(self) -> t.Tensor:
        return t.tensor([self.d_w(model) for model in self.models])

    def d_ws_with_rel(self) -> Tuple[t.Tensor, t.Tensor]:
        d_ws = self.d_ws()
        rel_d_ws = d_ws / self.baseline.model.parameters_norm

        return d_ws, rel_d_ws

    @property
    def n_test_samples(self) -> int:
        return len(self.test_loader.dataset)  # type: ignore

    @property
    def n_train_samples(self) -> int:
        return len(self.train_loader.dataset)  # type: ignore

    @property
    def baseline(self) -> Learner:
        return self.learners[0]

    @property
    def models(self) -> List[nn.Module]:
        return [learner.model for learner in self.learners]

    @property
    def optimizers(self) -> List[optim.Optimizer]:
        return [learner.optimizer for learner in self.learners]

    @property
    def hyperparams(self) -> Dict[str, Union[int, float, str]]:
        hyperparams = dict(
            # n_models=self.n_models,
            # n_epochs=self.n_epochs,
            # batch_size=self.batch_size,
        )

        variable_hyperparams = []

        # Add any hyperparams that are common to all sublearners
        for learner in self.learners:
            learner_hyperparams = learner.hyperparams

            for key, value in learner_hyperparams.items():
                if key not in hyperparams:
                    if key not in variable_hyperparams:
                        hyperparams[key] = value
                elif hyperparams[key] != value:
                    variable_hyperparams.append(key)

                    # Remove
                    del hyperparams[key]

        return hyperparams


class Experiment:
    def __init__(
        self,
        ensemble: EnsembleLearner,
        snapshotter: Snapshotter,
        plotter: Plotter,
        logger: Logger,
        logging_ivl: int = 100,
    ):
        self.ensemble = ensemble
        self.snapshotter = snapshotter
        self.plotter = plotter
        self.logger = logger
        self.logging_ivl = logging_ivl

    def train(self, epoch_start: int = 0, **kwargs):
        self.logger.init(**kwargs)
        self.load(epoch_start, **kwargs)

        losses = np.zeros((self.ensemble.n_models,))

        for epoch, batches in self.ensemble.train_epochs(epoch_start):
            learners = self.load(epoch, **kwargs)

            for batch_idx, data, target, _ in batches:
                for learner in learners:
                    loss, acc = learner.train_one_batch(data, target)
                    losses[learner.idx] = loss

                yield epoch, batch_idx, losses

                if batch_idx % self.logging_ivl == 0:
                    measurements = self.ensemble.inspect()

                    for i, learner in enumerate(self.learners):
                        self.logger.log(
                            epoch=epoch,
                            batch=batch_idx,
                            step=epoch * len(self.ensemble.train_loader) + batch_idx,
                            L_train=losses[i],
                            del_L_train=losses[i] - losses[0],
                            **measurements[i],
                            **learner.hyperparams,
                            **kwargs,
                        )

                if batch_idx == 0 and epoch > 0:
                    df = self.save(epoch=epoch - 1, **kwargs)
                    self.plot(df, epoch=epoch - 1, **self.hyperparams, **kwargs)

        df = self.save(epoch=self.ensemble.n_epochs - 1, **kwargs)
        self.plot(df, **self.hyperparams, **kwargs)

    def save(self, epoch: int, **kwargs):
        """Save models to snapshot. Returns logs as a dataframe."""
        df = self.logger.save(**kwargs)

        for i, learner in enumerate(self.learners):
            self.snapshotter.save(
                learner.model,
                epoch=epoch,
                **learner.hyperparams,
                **kwargs,
            )

            # print(f"Saved model {i} at epoch {epoch}.")

        return df

    def load(self, epoch: int, **kwargs):
        """Load models from snapshot. Any models that are not loaded are yielded."""
        for i, learner in enumerate(self.learners):
            if not self.snapshotter.load(
                learner.model,
                epoch=epoch,
                **learner.hyperparams,
                **kwargs,
            ):
                yield learner
            else:
                print(f"Loaded model {i} at epoch {epoch}.")

    def plot(self, df: pd.DataFrame, **kwargs):
        self.plotter.plot(df, **kwargs)

    @property
    def learners(self) -> List[Learner]:
        return self.ensemble.learners

    @property
    def models(self) -> List[ExtendedModule]:
        return self.ensemble.models

    @property
    def hyperparams(self) -> Dict[str, Any]:
        return self.ensemble.hyperparams
