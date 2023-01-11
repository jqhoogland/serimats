import hashlib
import itertools
import logging
import os
from dataclasses import dataclass, field
from os import PathLike
from pathlib import Path
from pprint import pp
from typing import (TYPE_CHECKING, Any, Callable, Dict, Generator, Iterable,
                    List, Literal, Optional, Protocol, Tuple, Type, TypedDict,
                    TypeVar, Union)

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

from serimats.paths.metrics import (Metrics, cos_sim_from_baseline,
                                    cos_sim_from_init, d_w_from_baseline,
                                    d_w_from_baseline_normed, d_w_from_init,
                                    d_w_from_init_normed, w_autocorr,
                                    w_corr_with_baseline, w_normed)
from serimats.paths.models import MNIST, ExtendedModule
from serimats.paths.utils import (CallableWithLatex, dict_to_latex, setup,
                                  var_to_latex)
from serimats.paths.weights import (AbsolutePerturbationInitializer,
                                    RelativePerturbationInitializer)

if TYPE_CHECKING:
    from serimats.paths.experiment import Ensemble




def plot_metric_scaling(
    ensemble: "Ensemble",
    df: pd.DataFrame,
    metric: Tuple[CallableWithLatex, ...], 
    step: Optional[int] = None,
    comparison: str = "epsilon",
    sample_axis: str = "seed_perturbation",
    include_baseline: bool = True,  # Whether to plot the first value of comparison
    **kwargs,
) -> Tuple[Figure, List[List[plt.Axes]]]:
    if step:
        df = df.loc[df["step"] <= step]

    metric_labels = tuple(m.__latex__[0] for m in metric)

    comparison_label = var_to_latex(comparison)
    comparison_values = getattr(ensemble, comparison)

    if not include_baseline:
        comparison_values = comparison_values[1:]

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

    for c, (m_fn, m_label) in enumerate(zip(metric, metric_labels)):
        m = m_fn.__name__

        for r, v in enumerate(comparison_values):
            # Get the data for this comparison value
            data = df.loc[df[comparison] == v]
            sample_values = data[sample_axis].unique()

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
    for c, (m_fn, m_label) in enumerate(zip(metric, metric_labels)):
        m = m_fn.__name__

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
