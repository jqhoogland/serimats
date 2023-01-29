import itertools
from typing import Dict, Iterable, List, Optional, Tuple, Type, TypeVar, Union

import pandas as pd
from torch import optim
from torch.utils.data import DataLoader, Dataset

from serimats.paths.checkpoints import Checkpointer
from serimats.paths.interventions import Intervention
from serimats.paths.metrics import Metrics
from serimats.paths.models import ExtendedModule
from serimats.paths.plots import Plotter
from serimats.paths.trials import Trial
from serimats.paths.utils import WithOptions, tqdm

T = TypeVar("T")

InterventionGroup = Union[
    Intervention, Iterable[Intervention], Iterable[Iterable[Intervention]]
]


def _fix_intervention_group(
    intervention_group: InterventionGroup,
) -> List[List[Intervention]]:
    if isinstance(intervention_group, Intervention):
        return [[intervention_group]]

    elif isinstance(intervention_group, Iterable):
        intervention_group = list(intervention_group)

        if isinstance(intervention_group[0], Intervention):
            return [intervention_group]

        return [list(i) for i in intervention_group]

    raise TypeError(
        "intervention_group must be an Intervention, Iterable[Intervention], or Iterable[Iterable[Intervention]]"
    )


class Experiment:
    def __init__(
        self,
        model: WithOptions[Type[ExtendedModule]],
        datasets: Union[Tuple[Dataset, ...], Dict[str, Dataset]],
        interventions: InterventionGroup,
        opt: WithOptions[Type[optim.Optimizer]] = optim.SGD,
        dl: WithOptions[Type[DataLoader]] = DataLoader,
        variations: Optional[InterventionGroup] = None,
        plotter: Optional[Plotter] = None,
        checkpointer: Optional[Checkpointer] = None,
        metrics: Optional[Metrics] = None,
    ):
        # Datasets

        if isinstance(datasets, tuple):
            if len(datasets) != 2:
                raise ValueError("datasets must be a tuple of length 2")

            datasets = {"train": datasets[0], "test": datasets[1]}

        # Data loaders

        dl_kwargs = {"batch_size": 64, "shuffle": True}

        if isinstance(dl, tuple):
            dl, dl_kwargs = dl

        dls: Dict[str, DataLoader] = {}

        for name, dataset in datasets.items():
            dls[name] = dl(dataset, **dl_kwargs)

        self.dls = dls
        train_loader = dls["train"]

        # Metrics

        self.metrics = metrics or Metrics()
        self.metrics.add_dls(dls)

        # Plotter

        self.plotter = plotter

        if self.plotter is not None:
            self.plotter.register(self)
            self.plotter.add_metrics(self.metrics)

        # Checkpointer

        self.checkpointer = checkpointer or Checkpointer()

        # Controls

        self.trials = []

        if variations is None:
            self.variations = []
            self.trials.append(Trial(model, opt, train_loader))
        else:
            self.variations = _fix_intervention_group(variations)

            for variation_combo in itertools.product(*self.variations):
                self.trials.append(Trial(model, opt, train_loader, variation_combo))  # type: ignore

        # Interventions

        self.interventions = _fix_intervention_group(interventions)

        for trial in self.trials:
            for intervention_combo in itertools.product(*self.interventions):
                self.trials.append(trial.with_interventions(intervention_combo))

    def run(self, n_epochs: int, reset: bool = False):
        epoch_idx, batch_idx, step = 0, 0, 0
        epoch_len = len(self.dls["train"]) / self.dls["train"].batch_size  # type: ignore

        for trial in tqdm(self.trials, desc="Trials"):
            trial = self.checkpointer.load(trial)

            for epoch_idx, epoch in tqdm(
                trial.run(n_epochs, reset=reset), desc="Epochs", total=n_epochs
            ):
                for batch_idx, step, _, loss in tqdm(
                    epoch, desc=f"Epoch {epoch_idx}", total=epoch_len
                ):
                    if step % self.metrics.ivl == 0:
                        self.metrics.measure(
                            epoch_idx=epoch_idx,
                            batch_idx=batch_idx,
                            step=step,
                            trial=trial,
                        )

                    if step % self.checkpointer.ivl == 0:
                        self.checkpointer.save(
                            epoch_idx=epoch_idx,
                            batch_idx=batch_idx,
                            step=step,
                            trial=trial,
                        )

        if self.plotter:
            self.plotter.plot(
                epoch_idx=epoch_idx,
                batch_idx=batch_idx,
                step=step,
            )

    def df(self):
        """Returns a dataframe with the logs of all trials"""
        return pd.concat([trial.df() for trial in self.trials])

    def __getitem__(self, i):
        return self.trials[i]

    def __len__(self):
        return len(self.trials)
