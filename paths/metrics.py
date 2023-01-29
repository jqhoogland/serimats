# Weight Space measurements

import functools
from abc import ABC
from dataclasses import dataclass
from functools import wraps
from typing import TYPE_CHECKING, Callable, Dict, List, Optional, Tuple, Union

import torch as t
import torch.nn.functional as F
from torch.utils.data.dataloader import DataLoader

from serimats.paths.models import parameters_norm

if TYPE_CHECKING:
    from serimats.paths.experiment import Trial


def metric(a: t.Tensor, b: t.Tensor, p="fro", **kwargs):
    return t.norm(a - b, p=p, **kwargs)


class AbstractMetrics(ABC):
    def __init__(self, dls: Optional[Dict[str, DataLoader]] = None, ivl=1000) -> None:
        self.dls = dls or {}
        self.ivl = ivl

        self.metrics: Dict[
            str, Tuple[Callable, str, bool]
        ] = {}  # TODO This would make more sense as a class attribute

    def add_dl(self, name: str, dl: DataLoader) -> None:
        self.dls[name] = dl

    def add_dls(self, dls: Dict[str, DataLoader]) -> None:
        self.dls.update(dls)

    def register_metric(
        self,
        name: Union[str, List[str]],
        fn: Callable,
        latex: Optional[Union[str, List, str]] = None,
        requires_data: bool = True,
    ):
        latex = latex or name

        if isinstance(name, list):
            return [
                self.register_metric(n, l, requires_data) for n, l in zip(name, latex)
            ]

        self.metrics[name] = (fn, latex, requires_data)  # type: ignore

    @property
    def data_metrics(self):
        return {
            name: metric
            for name, (metric, _, requires_data) in self.metrics.items()
            if requires_data
        }

    @property
    def data_free_metrics(self):
        return {
            name: metric
            for name, (metric, _, requires_data) in self.metrics.items()
            if not requires_data
        }

    def measure(
        self,
        epoch_idx: int,
        batch_idx: int,
        step: int,
        trial: Trial,
        **kwargs,
    ) -> Dict[str, float]:

        metrics = {
            name: metric(trial, **kwargs)
            for name, (metric, _, requires_data) in self.metrics.items()
        }

        if self.dls:
            for dl_name, dl in self.dls.items():
                for metric_name, metric in self.data_metrics.items():
                    metrics[f"{dl_name}_{metric_name}"] = t.zeros(1)

                    with t.no_grad():
                        for data, target in dl:
                            metrics[f"{dl_name}_{metric_name}"] += metric(
                                trial, data, target, **kwargs
                            )

                        metrics[f"{dl_name}_{metric_name}"] /= len(dl.dataset)

        return {
            name: metric.item() if isinstance(metric, t.Tensor) else metric
            for name, metric in metrics.items()
        }


class Metrics(AbstractMetrics):
    def __init__(self, dls: Optional[Dict[str, DataLoader]] = None, ivl=1000) -> None:
        super().__init__(dls, ivl)

        self.register_metric(["L", "acc"], self.loss_and_accuracy, requires_data=True)

    def loss_and_accuracy(
        self, trial: Trial, data: t.Tensor, target: t.Tensor, **kwargs
    ) -> Tuple:
        output = trial.model(data)  # type: ignore
        loss = F.nll_loss(output, target, reduction="sum")
        pred = output.argmax(dim=1, keepdim=True)
        acc = pred.eq(target.view_as(pred)).sum()

        return loss, acc


class FullPanelMetrics(AbstractMetrics):
    def __init__(self, dls: Optional[Dict[str, DataLoader]] = None, ivl=1000) -> None:
        super().__init__(dls, ivl)

        self.register_metric(
            ["L", "acc", "L_cf", "acc_cf"],
            self.loss_and_accuracy_with_cf,
            requires_data=True,
        )
        self.register_metric(
            ["w_norm", "w_norm_init", "w_norm_cf"],
            self.ws,
            requires_data=False,
        )

        self.register_metric(
            ["dw_init", "dw_cf", "dw_control_normed"],
            self.dws,
            requires_data=False,
        )

        self.register_metric(
            ["cos_sim_init", "cos_sim_control"],
            self.cos_sims,
            requires_data=False,
        )   

    def loss_and_accuracy_with_cf(
        self, trial: Trial, data: t.Tensor, target: t.Tensor, **kwargs
    ) -> Tuple:
        output = trial.model(data)  # type: ignore
        loss = F.nll_loss(output, target, reduction="sum")
        pred = output.argmax(dim=1, keepdim=True)
        acc = pred.eq(target.view_as(pred)).sum()

        control_output = trial.control.model(data)  # type: ignore
        loss_cf = F.cross_entropy(control_output, target, reduction="sum")
        pred_cf = control_output.argmax(dim=1, keepdim=True)
        acc_cf = pred_cf.eq(target.view_as(pred_cf)).sum()

        return loss, acc, loss_cf, acc_cf

    @staticmethod
    def ws(trial: "Trial"):
        w_norm = trial.norm()
        w_norm_init = trial.init.norm()
        w_norm_cf = trial.control.norm()

        return w_norm, w_norm / w_norm_init, w_norm / w_norm_cf

    def dws(self, trial: "Trial"):
        epsilon = self.epsilon(trial)
        dw_init = trial.lp_distance(trial.init)
        dw_control = trial.lp_distance(trial.control)

        dw_control_normed = self.normalize_wrt(dw_control, epsilon)

        return dw_init, dw_control, dw_control_normed

    def cos_sims(self, trial: "Trial"):
        cos_sim_init = trial.cosine_similarity(trial.init)
        cos_sim_control = trial.cosine_similarity(trial.control)

        return cos_sim_init, cos_sim_control

    def normalize_wrt(self, n: t.Tensor, d) -> t.Tensor:
        if d == 0:
            return t.tensor(0)

        return n / d

    @staticmethod
    def epsilon(trial: "Trial") -> t.Tensor:
        # TODO: These two are not the same for small epsilon?
        # return trial.weight_initializer.epsilon
        norm = t.zeros(1, device=trial.device)

        if (
            trial.weight_initializer.initial_weights is None
            or trial.baseline.weight_initializer.initial_weights is None
        ):
            raise ValueError("Initial weights not set")

        for p1, p2 in zip(
            trial.weight_initializer.initial_weights,
            trial.baseline.weight_initializer.initial_weights,
        ):
            norm += t.norm(p1 - p2, p="fro") ** 2

        return norm.sqrt()
