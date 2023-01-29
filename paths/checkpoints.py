import logging
from pathlib import Path
from typing import Optional, Union

import pandas as pd
import yaml
from torch import t

from serimats.paths.trials import Trial

Pathy = Union[Path, str]


class Checkpointer:
    def __init__(self, ivl=1000, dir: Pathy = Path("checkpoints")) -> None:
        self.ivl = ivl
        self.dir = Path(dir)

    def path(self, name: str):
        return self.dir / "runs" / name

    def path_to_step(self, name: str, step: int):
        return self.path(name) / ("step-" + str(step))

    def save(
        self,
        epoch_idx: int,
        batch_idx: int,
        step: int,
        trial: Trial,
        overwrite: bool = False,
    ):
        path = self.path_to_step(trial.unique_name(), step)

        path.mkdir(parents=True, exist_ok=True)

        if not overwrite and (path / "model.pt").exists():
            # raise FileExistsError(path_to_step)
            logging.info(f"Step {step} already exists, skipping")
            return False

        trial.df(full=False).to_csv(path / "../logs.csv")
        yaml.dump(trial.hyperparams, open(path / "../hyperparams.yaml", "w"))

        t.save(trial.model.state_dict(), path / "model.pt")
        t.save(trial.opt.state_dict(), path / "opt.pt")

        logging.info(f"Saved model to {path} (step={step}, {trial.extra_repr})")

        return True

    def load(self, trial: Trial, step: Optional[int] = None) -> Trial:
        path = self.path(trial.unique_name())

        try:
            logs_df = pd.read_csv(path / "../logs.csv")

            if step is None:
                self.step = logs_df["step"].max().item()
            else:
                self.step = step

            logs_df = logs_df.set_index("step")
            trial.logs = logs_df.to_dict(orient="index")

        except FileNotFoundError:
            logging.info(f"Could not load logs from {trial.path / 'logs.csv'}")
            logging.info(yaml.dump(trial.hyperparams))
            trial.logs = {}

            return trial

        if not isinstance(step, int):
            raise ValueError(f"step must be an int, got {step} ({type(step)})")

        path_to_step = self.path_to_step(trial.unique_name(), step)

        try:
            if not trial.active:    
                trial.activate()

            trial.model.load_state_dict(t.load(path_to_step / "model.pt"))
            trial.opt.load_state_dict(t.load(path_to_step / "opt.pt"))

            logging.info(
                f"Loaded model from {path_to_step} (step={step}, {trial.extra_repr})"
            )
        except FileNotFoundError:
            logging.info(f"Could not load model or optimizer from {path_to_step}")

        return trial
