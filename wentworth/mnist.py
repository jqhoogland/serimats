# %%
import os
from pprint import pp

import numpy as np
import torch as t
from matplotlib import pyplot as plt
from pyhessian import hessian  # Hessian computation
from torch import nn
from torch.utils.data import DataLoader
from torchtyping import TensorType
from torchvision import datasets
from torchvision.transforms import ToTensor
from tqdm import tqdm

# %%

# Download training data from open datasets.
training_data = datasets.MNIST(
    root=os.path.dirname(__file__),
    train=True,
    download=True,
    transform=ToTensor(),
)

# Download test data from open datasets.
test_data = datasets.MNIST(
    root=os.path.dirname(__file__),
    train=False,
    download=True,
    transform=ToTensor(),
)
# %%

batch_size = 64

# Create data loaders.
train_dataloader = DataLoader(training_data, batch_size=batch_size)
test_dataloader = DataLoader(test_data, batch_size=batch_size)

for X, y in test_dataloader:
    print(f"Shape of X [N, C, H, W]: {X.shape}")
    print(f"Shape of y: {y.shape} {y.dtype}")
    break
# %%

# Get cpu or gpu device for training.
device = "cuda" if t.cuda.is_available() else "cpu"
print(f"Using {device} device")

# Define model
class NeuralNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(28 * 28, 10),
        )

    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits


model = NeuralNetwork().to(device)
num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(model, num_params)
# %%

loss_fn = nn.CrossEntropyLoss()
optimizer = t.optim.SGD(model.parameters(), lr=1e-3)


def train(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)
    model.train()
    for batch, (X, y) in enumerate(dataloader):
        X, y = X.to(device), y.to(device)

        # Compute prediction error
        pred = model(X)
        loss = loss_fn(pred, y)

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if batch % 100 == 0:
            loss, current = loss.item(), batch * len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")


def test(dataloader, model, loss_fn):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    model.eval()
    test_loss, correct = 0, 0
    with t.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(t.float).sum().item()
    test_loss /= num_batches
    correct /= size
    print(
        f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n"
    )


# %%

epochs = 5
for i in range(epochs):
    print(f"Epoch {i+1}\n-------------------------------")
    train(train_dataloader, model, loss_fn, optimizer)
    test(test_dataloader, model, loss_fn)
print("Done!")
# %%

t.save(model.state_dict(), os.path.join(os.path.dirname(__file__), "mnist.pth"))
print("Saved PyTorch Model State to mnist.pth")

# model = NeuralNetwork()
# model.load_state_dict(t.load("fashion-mnist.pth"))

# %%

model.eval()
x, y = test_data[0][0], test_data[0][1]
with t.no_grad():
    pred = model(x)
    predicted, actual = pred[0].argmax(0), y
    print(f'Predicted: "{predicted}", Actual: "{actual}"')

accuracy = 0

for i in tqdm(range(len(test_data))):
    x_i, y_i = test_data[i][0], test_data[i][1]
    with t.no_grad():
        pred = model(x_i)
        predicted, actual = pred[0].argmax(0), y_i
        if predicted == actual:
            accuracy += 1

print(f"Accuracy: {accuracy / len(test_data)}")

# %%

# Behavioral gradients (for a single component of the output)


def get_parameters(model: nn.Module) -> list[t.Tensor]:
    return [p.reshape((-1,)) for p in model.parameters() if p.requires_grad]


def behavioral_gradient_i(
    model: nn.Module, x_i: t.Tensor, j: int
) -> TensorType["N_params"]:
    # g_i = \nabla_\theta f(\theta, x_i)
    model.zero_grad()
    y_hat_j = model(x_i.reshape((1, 28, 28)))[0][j]
    y_hat_j.backward()

    grads = [p.grad.reshape((-1,)) for p in model.parameters() if p.requires_grad]

    if None in grads:
        raise ValueError("Gradient is None")

    return t.cat(grads)


def behavioral_gradient(
    model: nn.Module, x: t.Tensor, j=0
) -> TensorType["N_samples", "N_params"]:
    # G = matrix with ith column = g_i
    return t.stack(
        [
            behavioral_gradient_i(model, x_i, j)
            for x_i in tqdm(x, desc="Calculating G...")
        ]
    ).T


def hessian_mse(
    model: nn.Module, x: t.Tensor, j=0
) -> TensorType["N_params", "N_params"]:
    # Hessian
    # H = 2 G G^T
    G = behavioral_gradient(model, x, j)

    return 2 * G @ G.T

# %%

# Behavioral gradients (for all components of the output)


def full_behavioral_gradient_i(
    model: nn.Module, x_i: t.Tensor
) -> TensorType["N_classes * N_params"]:
    # g_i = \nabla_\theta f(\theta, x_i)
    gradients: list[None | t.Tensor] = [None] * 10

    for j in range(10):

        model.zero_grad()
        y_j = model(x_i.reshape((1, 28, 28)))[0][j]

        y_j.backward()
        grads = [p.grad.reshape((-1,)) for p in model.parameters() if p.requires_grad]

        if None in grads:
            raise ValueError("Gradient is None")

        gradients[j] = t.cat(grads)

    return gradients


def full_behavioral_gradient(
    model: nn.Module, x: t.Tensor
) -> TensorType["N_samples", "N_classes * N_params"]:
    """Big G"""
    return t.stack(
        [
            grad_i_y_j
            for x_i in tqdm(x, desc="Calculating G...")
            for grad_i_y_j in full_behavioral_gradient_i(model, x_i)
        ]
    )


def full_hessian_mse(
    model: nn.Module, x: t.Tensor, j=0
) -> TensorType["N_classes * N_params", "N_classes * N_params"]:
    # Hessian
    # H = 2 G G^T
    G = full_behavioral_gradient(model, x).T

    return 2 * G @ G.T


# %%

# They seem to be randomly sorted
x_train = train_dataloader.dataset.data.float()[:1000]  # type: ignore
# G_train = full_behavioral_gradient(model, x_train)

# %%

# print(G_train.shape, t.linalg.matrix_rank(G_train))
# evals, evecs = t.linalg.eigh(G_train @ G_train.T)

# print(G_train.shape)
# plt.plot(evals)

# H_mse = 2 * G_train.T @ G_train
# print(H_mse.shape)

# %%

# H_mse_evals = t.linalg.eigvalsh(H_mse)
# plt.plot(H_mse_evals)


# %%

# For 7960 parameters:

# G has shape (1000, 7960 * 10)

# When looking at a single component of the output, we get the following:
# f[0]: 10,000 -> rank(G) = 1630
# f[1]: 10,000 -> rank(G) = 1904

# When looking at all the components (i.e., we flatten output * n_params), we get the following:
# f: 10 * 1,000 -> rank(G) = 4744

# %%

loss_fn = nn.CrossEntropyLoss()


def hessian(model: nn.Module, x: t.Tensor, y: t.Tensor) -> tuple:  # Awful tuple thing
    def f(*params):
        names = list(n for n, _ in model.named_parameters())
        out_samples = t.nn.utils.stateless.functional_call(
            model, {n: p for n, p in zip(names, params)}, x
        )
        return out_samples

    def cross_entropy_loss(*params: tuple[t.Tensor]) -> t.Tensor:
        y_hat = f(*params)
        return loss_fn(y_hat, y)

    return t.autograd.functional.hessian(cross_entropy_loss, tuple(model.parameters()))


def flatten_hessian(
    H: tuple,
) -> TensorType["N_params * N_classes", "N_params * N_classes"]:
    shapes = [p.shape for p in model.parameters()]
    i_params = [0 for _ in range(len(shapes) + 1)]
    for i in range(len(shapes)):
        i_params[i + 1] = i_params[i] + np.product(shapes[i])

    n_params = sum(p.numel() for p in model.parameters())
    out = t.empty(n_params, n_params)
    for i in range(len(shapes)):
        for j in range(len(shapes)):
            nvi = i_params[i + 1] - i_params[i]
            nvj = i_params[j + 1] - i_params[j]
            out[i_params[i] : i_params[i + 1], i_params[j] : i_params[j + 1]] = H[i][
                j
            ].reshape(nvi, nvj)

    return out


x_train = train_dataloader.dataset.data.float()[:100]
y_train = train_dataloader.dataset.targets[:100]

H = flatten_hessian(hessian(model, x_train, y_train))
print(H.shape)

# %%
print(H.shape)
H_evals, H_evecs = t.linalg.eig(H)
plt.plot(H_evals)

# %%

plt.hist(H_evals, bins=100)

# %%

print(H_evals.mean()) # tensor(6.2640e-06-1.4143e-20j)
print(H_evals.std()) # tensor(0.0005) 

print(np.sum(np.where(np.abs(H_evals) > 0.0005, 1, 0)))

# %%


def V_ball(n: int | float, R=1.0):
    return 1.0  # We care about relative volumes
    # return t.pi ** (n / 2) / t.exp(t.special.gammaln(n / 2 + 1)) * R**n


def V_ellipsoid(H: t.Tensor, T=0.5):
    # Ellipsoid volume
    # V_basin = V_n \prod_i \sqrt{2T/\lambda_i} = V_n (2T)^{n/2} / \sqrt{\det H},
    # where V_n is the volume of the n-dimensional unit ball, T is the loss threshold, and \lambda_i are the eigenvalues of the Hessian
    n = H.shape[0]

    return V_ball(n) * (2 * T) ** (n / 2) / t.sqrt(t.det(H))


def V_fuzzy_ellipsoid(H: t.Tensor, T=0.5, lmbda=1.0, k=1.0, sigma=1.0):
    # Fuzzy Ellipsoid (Gaussian) volume
    # V_basin = V_n (2T)^{n/2} / \sqrt{\det[H + (\lambda + c)I_n] },
    # where \lambda is the weight decay, c = k/\sigma^2 (\sigma is stdev of initialization Gaussian, k is a constant)
    n = H.shape[0]
    c = k / sigma**2
    H_prime = H + (lmbda + c) * t.eye(n)
    return V_ball(n) * (2 * T) ** (n / 2) / t.sqrt(t.det(H_prime))

print(V_fuzzy_ellipsoid(H))


# %%
