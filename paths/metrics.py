# Weight Space measurements

from dataclasses import dataclass
from functools import wraps
from typing import TYPE_CHECKING, Dict, List, Tuple, Union

import torch as t
import torch.nn.functional as F
from torch.utils.data.dataloader import DataLoader

from serimats.paths.utils import add_latex

if TYPE_CHECKING:
    from serimats.paths.experiment import Learner



@add_latex("l_p(a, b)", "|a-b|^p")
def metric(a: t.Tensor, b: t.Tensor, p="fro", **kwargs):
    return t.norm(a - b, p=p, **kwargs)


# Misc helpers


@add_latex(r"w", r"|\mathbf {w}|")
def w_norm(learner: "Learner") -> t.Tensor:
    return learner.parameters_norm


@add_latex(r"\widehat{w}", r"w / w^{(0)}")
def w_normed(learner: "Learner") -> t.Tensor:
    return normalize_wrt_init(w_norm(learner), learner)


# Current model vs baseline model


@add_latex(r"\epsilon", r"d_W(w^{(0)}, w_\mathrm{ref}^{(0)})")
def epsilon(learner: "Learner") -> t.Tensor:
    # TODO: These two are not the same for small epsilon?
    # return learner.weight_initializer.epsilon
    return metric(learner.weight_initializer.initial_weights, learner.baseline.weight_initializer.initial_weights)

@add_latex(r"\widehat{v}", r"v / \epsilon")
def normalize_wrt_epsilon(v: t.Tensor, learner: "Learner") -> t.Tensor:
    eps = epsilon(learner)

    if eps == 0:
        return t.tensor(0)

    return v / eps

@add_latex(r"d_W(w^{(t)}, w_\mathrm{ref}^{(t)})", r"|w^{(t)} - w_\mathrm{ref}^{(t)}|")
def d_w_from_baseline(learner: "Learner") -> t.Tensor:
    """
    Calculate $d_W(w', w)$, where $w$ is the baseline (unperturbed) model, and
    $w'$ is the current (perturbed) model.
    """
    # TODO: Cache parameters_vector after training step until next training step
    return metric(learner.parameters_vector, learner.baseline.parameters_vector)


@add_latex(r"\widehat{d_W}(w^{(t)}, w_\mathrm{ref}^{(t)})", r"d_W / w_\mathrm{ref}^{(t)}")
def d_w_from_baseline_normed(learner: "Learner") -> t.Tensor:
    return normalize_wrt_epsilon(d_w_from_baseline(learner), learner)


@add_latex(r"C_\mathrm{ref}(t)", r"\frac{1}{D}\mathbf{w}^{(t)} \cdot \mathbf{w}_\mathrm{ref}^{(t)}")
def w_corr_with_baseline(learner: "Learner") -> t.Tensor:
    """
    Calculate the correlation between the current model and the baseline model.
    """
    return learner.parameters_vector @ learner.baseline.parameters_vector


@add_latex(r"S_C(w^{(t)}, w_\mathrm{ref}^{(t)})", r"\frac{|\mathbf{w}^{(t)} \cdot \mathbf{w}_\mathrm{ref}^{(t)}|}{|w^{(t)}| |w_\mathrm{ref}^{(t)}|}")
def cos_sim_from_baseline(learner: "Learner") -> t.Tensor:
    """
    Calculate the cosine similarity between the current model and the baseline model.
    """
    return F.cosine_similarity(learner.parameters_vector, learner.baseline.parameters_vector, dim=0)


# Current model vs initial model

@add_latex(r"w^{(0)}", r"|\mathbf{w}^{(0)}|")
def w_init(learner: "Learner") -> t.Tensor:
    return t.norm(learner.weight_initializer.initial_weights)


@add_latex(r"\widetilde{v}", r"v / w^{(0)}")
def normalize_wrt_init(d: t.Tensor, learner: "Learner") -> t.Tensor:
    return d / w_init(learner)


@add_latex(r"d_W(w^{(t)}, w^{(0)})", r"|w^{(t)} - w_\mathrm{ref}^{(t)}|")
def d_w_from_init(learner: "Learner") -> t.Tensor:
    """
    Calculate $d_W(w^{(t)}, w^{(0)}))$, where $w^{(t)}$ is the current model at time $t$
    and $w^{(0)}$ is the (same) model upon initialization (after perturbation, if any).
    """
    # TODO: Cache parameters_vector after training step until next training step
    return metric(learner.parameters_vector, learner.weight_initializer.initial_weights)


@add_latex(r"\widetilde{d_W}(w^{(t)}, w^{(0)})", r"d_W / w^{(0)}")
def d_w_from_init_normed(learner: "Learner") -> t.Tensor:
    return normalize_wrt_init(d_w_from_init(learner), learner)


@add_latex(r"R(t)", r"\frac{1}{D}\mathbf{w}^{(t)} \cdot \mathbf{w}_\mathrm{ref}^{(t)}")
def w_autocorr(learner: "Learner") -> t.Tensor:
    """
    Calculate the correlation between the current model and itself (at time 0).
    TODO: This is averaged over the parameters, but it should probably be tracked individually for each parameter.
    """
    return learner.parameters_vector @ learner.weight_initializer.initial_weights / learner.n_parameters


@add_latex(r"S_C(w^{(t)}, w^{(0)})", r"\frac{|\mathbf{w}^{(t)} \cdot \mathbf{w}^{(0)}|}{|w^{(t)}| |w^{(0)}|}")
def cos_sim_from_init(learner: "Learner") -> t.Tensor:
    """
    Calculate the cosine similarity between the current model and itself (at time 0).
    """
    return F.cosine_similarity(learner.parameters_vector, learner.weight_initializer.initial_weights, dim=0)



def _weight_space_metrics(learner: "Learner", **_) -> Dict[str, t.Tensor]:
    metrics = {
        "w": w_norm(learner),
        "d_w_from_baseline": d_w_from_baseline(learner),
        "d_w_from_init": d_w_from_init(learner),
        "w_corr_with_baseline": w_corr_with_baseline(learner),
        "w_autocorr": w_autocorr(learner),
        "cos_sim_from_baseline": cos_sim_from_baseline(learner),
        "cos_sim_from_init": cos_sim_from_init(learner),
    }

    metrics["w_normed"] = normalize_wrt_init(metrics["w"], learner)
    metrics["d_w_from_baseline_normed"] = normalize_wrt_epsilon(metrics["d_w_from_baseline"], learner)
    metrics["d_w_from_init_normed"] = normalize_wrt_init(metrics["d_w_from_init"], learner)

    return metrics


def weight_space_metrics(learners: List["Learner"], **_) -> List[Dict[str, t.Tensor]]:
    return [_weight_space_metrics(learner, **_) for learner in learners]


# Function space


def _function_space_metrics(
    learners: List["Learner"], data: t.Tensor, target: t.Tensor, loss_fn=F.nll_loss, **_
) -> t.Tensor:
    metrics = t.zeros((4, len(learners))) # L_test, acc_test, L_compare, acc_compare

    # Precompute so we don't have repeat computations

    baseline_models = {learner.baseline.model for learner in learners}
    baseline_outputs = {baseline: baseline(data) for baseline in baseline_models}
    baseline_preds = {baseline: baseline_outputs[baseline].argmax(dim=1, keepdim=False) for baseline in baseline_models}

    for i, learner in enumerate(learners):
        output = learner.model(data)
        pred = output.argmax(dim=1, keepdim=True)

        baseline_output = baseline_outputs[learner.baseline.model]
        baseline_pred = baseline_preds[learner.baseline.model]

        # metrics[0, i] = loss_fn(output, target, reduction="sum").item()
        # metrics[1, i] = pred.eq(target.view_as(pred)).sum().item()
        metrics[2, i] = loss_fn(output, baseline_pred, reduction="sum").item()  # TODO: Custom loss to allow for multi-target
        metrics[3, i] = pred.eq(baseline_pred.view_as(pred)).sum().item()

    return metrics


def function_space_metrics(
    learners: List["Learner"], dl: DataLoader, loss_fn = F.nll_loss, **_
) -> List[Dict[str, t.Tensor]]:
    """Returns the loss and accuracy averaged over the entire test set."""
    n_learners = len(learners)
    n_samples = len(dl.dataset)  # type: ignore

    metrics = t.zeros((4, n_learners))  # L_test, acc_test, L_compare, acc_compare

    with t.no_grad():
        if n_learners == 0:
            return []

        for data, target in dl:
            metrics += _function_space_metrics(learners, data, target, loss_fn=loss_fn)

        metrics /= n_samples

    # baselines = t.stack([metrics[:, learners.index(learner.baseline)] for learner in learners]).T

    # delta_L_test = metrics[0] - baselines[0]
    # delta_acc_test = metrics[1] - baselines[1]

    return [
        {
            # "L": metrics[0, i],
            # "acc": metrics[1, i],
            "L_compare": metrics[2, i],
            "acc_compare": metrics[3, i],
            # "delta_L": delta_L_test[i],
            # "delta_acc": delta_acc_test[i],
        } for i in range(n_learners)
    ]


# Combined

def metrics(learners: List["Learner"], dls: List[Tuple[str, DataLoader]], loss_fn = F.nll_loss, **_) -> List[Dict[str, t.Tensor]]:
    metrics = weight_space_metrics(learners, **_)

    for name, dl in dls:
        for i, learner_metrics in enumerate(function_space_metrics(learners, dl, loss_fn=loss_fn, **_)):
            for k, v in learner_metrics.items():
                metrics[i][f"{k}_{name}"] = v

    return metrics


def detensorize_metrics(metrics: List[Dict[str, t.Tensor]]) -> List[Dict[str, float]]:
    return [{k: v.item() for k, v in learner_metrics.items()} for learner_metrics in metrics]


@dataclass
class Metrics:
    train_dl: DataLoader  # Currently not in use
    test_dl: DataLoader

    def measure(self, learners: List["Learner"], loss_fn = F.nll_loss, **kwargs) -> List[Dict[str, float]]:
        return detensorize_metrics(metrics(
            learners,
            dls=[("train", self.train_dl), ("test", self.test_dl)],
            loss_fn=loss_fn,
            **kwargs
        ))