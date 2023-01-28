# Experiments on Training

Some personal tools for running ML experiments:

- Run variations of anything and everything (weight initializations, architecture, hyperpararmeters, optimizer choices, etc.).
- Fine-grained interventions (perturb the weights, gradients, activations, hyperparameters, etc.).
- Take checkpoints any time.
- Custom loggers and plotters (record any metric you can imagine — on the models themselves, the test sets, training sets, etc.).
- Train in parallel on a cluster (ok, not yet, but, you know, eventually) or in serial on your local machine. 
- Reuse past results when running a new intervention if you've already tested a particular condition/control.
- Consistent seed management. 

The library is organized as follows:
- `Experiment` is a collection of `Learners` differentiated by `Intervention`.
- `Learner` is a (`Model`, `Optimizer`, `DataLoader`, `Logger`, `Intervention`) tuple. It is the basic unit of training.
- `Intervention` is a class that perturbs the model, optimizer, or data loader. 
- `Logger` is a class that records metrics and plots them.
- `Plotter` is a class that plots metrics.

## Examples

Suppose I want to test the effect of a small weight perturbation at initialization. It's as simple as the following:

```python
from torchvision import datasets, transforms

from serimats.experiments import Experiment
from serimats.interventions import PerturbWeights
from serimats.loggers import Logger
from serimats.plotter import Plotter
from serimats.models import Lenet5

dataset = datasets.MNIST('data', train=True, download=True,transform=transforms.ToTensor())

exp = Experiment(
    model=Lenet5(),
    dataset=dataset,
    interventions=[
        PerturbWeights.make_variations(
            epsilon=0,  
            seed=0,
            control=True
        ),
        PerturbWeights.make_variations(
            epsilon=(0.001, 0.01, 0.1),  
            seed=range(10)
        )
    ]
)
exp.run()

# This will generate 3 x 10 = 30 learners, each with different perturbaton size or perturbation seed.
```

Or maybe I want to compare the behavior of different optimizers.

```python
from serimats.interventions import InitWeightsIntervention, OptimizerIntervention

dataset = datasets.MNIST('data', train=True, download=True,transform=transforms.ToTensor())

exp = Experiment(
    model=Lenet5(),
    dataset=dataset,
    interventions=[
        InitWeightsIntervention.make_variations(
            seed=range(10)
        ),
        OptimizerIntervention.make_variations(
            optimizers=(torch.optim.SGD, torch.optim.Adam),
            lr=(0.001, 0.01, 0.1),
        )
    ],
)

# This will generate 10 x 2 x 3 = 60 learners, with different initial weights, optimizer, or learning rate.
```

Or maybe I want to test the effect of a temporary perturbation to momentum, depending on when it is applied during training.

```python
exp = Experiment(
    model=Lenet5(),
    dataset=dataset,
    interventions=[
        InitWeightsIntervention.make_variations(
            seed=range(10)
        ),
        OptimizerIntervention.make_variations(
            momentum=(0.9, 0.99, 0.999),
        ),
        PerturbMomentumIntervention.make_variations(
            momentum=lambda m: (m * (1 + epsilon) for epsilon in (0.001, -0.001, 0.01, -0.01, 0.1, -0.1)),
            when=((0, 100), (1000, 1100), (5000, 5100)),
        )
    ],
)

# This will generate 10 x 3 x 6 x 3 = 540 learners, with different initial weights, momenta, momentum perturbations, or time (step) of intervention.
```

That's a *lot* of variations. My computer will take several days to run all of them.

So I can get rid of the `InitWeightsIntervention`, which leaves me with a more reasonable 54 trials. 
After I've validated for a fixed choice of weight initialization, I can add it back in, and run the experiment again. Best of all, it'll automatically skip the trials that have already been run.

This allows for a more iterative experimentation loop so you can explore more faster.

## Logging and Plotting

By default, the `Logger` will record the loss and accuracy on the training and test sets at the end of each epoch. `Plotting` is disabled by default, but you can enable it by passing `plot=True` to `Experiment.run`.

You can add your own metrics by subclassing `Logger`.

```python
class CustomLogger(Logger):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    @register_metric
    def weight_norm(self, learner):
        return torch.norm(learner.model.fc1.weight)
    
    @register_metric
    def weight_space_distance(self, learner):
        return torch.dist(learner.model.fc1.weight, learner.model.fc2.weight)

```