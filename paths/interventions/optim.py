

from typing import Optional, Type
from serimats.paths.interventions.base import Intervention
from serimats.paths.trials import Trial
from serimats.paths.variables.optim import ExtendedOptimizer


class ChangeOptimizer(Intervention):
    hyperparams: dict
    prev_opt: Optional[ExtendedOptimizer] = None

    def __init__(self, opt_cls: Type[ExtendedOptimizer], when=None, **kwargs):
        self.opt_cls = opt_cls

        super().__init__(when=when)
        
        self.hyperparams = {
            "opt": opt_cls.__name__,
            **kwargs
        }

    def apply(self, trial: Trial):
        trial.opt = self.opt_cls(trial.model.parameters(), **self.hyperparams)

    def revert(self, trial: Trial):
        trial.opt = self.prev_opt