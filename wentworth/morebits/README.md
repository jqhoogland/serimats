# More bits


A neural network is a repeated function we have managed to tame into submission:


$$\mathbf{f}^{(l)}(\mathbf {x}, \boldsymbol \Theta_t^{(l)}) = \boldsymbol f^{(l)}\left(\mathbf f^{(l-1)}(\mathbf x, \boldsymbol\Theta^{(l-1)}_t),\ \boldsymbol \theta^{(i)}_t\right),$$

where $\mathbf{f}^{(l)} \in \mathbb R^{n_l}$ is the output of layer $l$.

Note: $\mathbf{f}^{(0)} = \mathbf x  \in \mathbb R^{n_0}$ (the data) starts the inductive loop, and $\mathbf{f}^{(L)} = \hat{\mathbf y}$ ends it.

 
$\mathbf f$ is a function of the data and parameters, $\boldsymbol \Theta_t^{(l)} \in \mathbb R^{p_l}= \{\boldsymbol{\theta}^{(i)}\}_{i=1}^l$ (at training timestep $t$ and in layer $i$).

For convenience: 

$$P_l = \sum_{i=1}^l p_i,\qquad N_l = \sum_{i=0}^l n_l.$$

We also define the vector of all activations:

$$ \mathbf{F}^{(l)}(\mathbf x, \boldsymbol \Theta_t^{(l)}) = \begin{bmatrix} \mathbf f^{(0)}(\mathbf x, \boldsymbol \Theta_t^{(0)}) \\ \vdots \\ \mathbf f^{(l)}(\mathbf x, \boldsymbol \Theta_t^{(l)}) \end{bmatrix} \in \mathbb R^{N_l}.$$


Finally, we are given some loss function:

$$\mathcal{L}(\mathbf{f}^{(L)}(\mathbf{x}, \boldsymbol\Theta_t^{(L)}), \mathbf y),$$

for a network of $L$ total layers.


## What can we measure?

---

**The Loss**

$$\mathcal{L}$$

(Especially when we clamp either the data or parameters, so we can study the landscape in the other.)

**Activations (of any layer)**:

$$\mathbf{F}^{(l)}$$

(Ditto)

**The Parameters**:

$$\boldsymbol \Theta_t^{(l)}$$

---

**Gradients of the loss (e.g., backprop updates)**:

$$\nabla_{\mathbf \Theta^{(l)}_t} \mathbf L \in \mathbb R^{P_l}$$

$$\nabla_{\mathbf F^{(l)}} \mathbf L \in \mathbb R^{n_l}$$


**Gradients of the outputs (or any of the preceding layers)**:

$$\nabla_{\mathbf \Theta^{(l)}_t} \mathbf f^{(l)} \in \mathbb R^{n_l \times P_l}$$

$$\nabla_{\mathbf F^{(l')}} \mathbf f^{(l)} \in \mathbb R^{n_l \times N_{l'}}$$


---

**Hessians** (this is where it starts to get expensive):

$$H = \nabla_{\mathbf \Theta^{(l)}_t} \nabla_{\mathbf \Theta^{(l')}_t} \mathbf L \in \mathbb R^{P_l \times P_{l'}}$$

$$H = \nabla_{\mathbf F^{(l)}} \nabla_{\mathbf F^{(l')}} \mathbf L \in \mathbb R^{N_l \times N_{l'}}$$


**Hessians of the outputs**:

$$H = \nabla_{\mathbf \Theta^{(l')}_t} \nabla_{\mathbf \Theta^{(l'')}_t} \mathbf f^{(l)} \in \mathbb R^{n_l \times P_{l'} \times P_{l''}}$$

$$H = \nabla_{\mathbf F^{(l')}} \nabla_{\mathbf F^{(l'')}} \mathbf f^{(l)} \in \mathbb R^{n_l \times N_{l'} \times N_{l''}}$$


---

**Averages**:

For all of the above we can study averages over (batches/specific clusters/classes) of the data instead (indicated as $\bar{\cdot}$).

You cal also study averages over different training runs (indicated as $\langle \cdot \rangle$).

**Over time**:

We can also study the evolution of the above quantities over time (indicated as $\mathbf \Theta_t^{(l)}$).

**Perturbations**:

Instead of infinitessimal perturbations (i.e., derivatives, $\nabla$), we can also study the effect of larger perturbations, $\delta$, to the parameters/activations/etc.:

---

## What to do with these huge matrices?

Plot them: a simple `plt.imshow` will do the trick.

Plot the eigenvalues: `plt.plot(np.linalg.eigvals(H))` will do the trick.

Compute the rank, condition number, various norms, and other statistics.

Do a PCA or some other decomposition (like an NMF). This is especially useful when studying averages over data/training runs.

Study a (more computationally feasible) subset:
-  Wherever you see $\boldsymbol \Theta^{(l)}_t$, you can swap it out for a specific layer's parameters, $\boldsymbol \theta^{(l)}_t$, or a specific weight, $(\boldsymbol\theta^{(l)}_{t})_i$. You can also filter by channel or by neuron.
- The same goes for $\mathbf F^{(l)}$, $\mathbf f^{(l)}$, and $\mathbf{f}^{(l)}_i$. Note that $\mathbf{f}^{(0)}=\mathbf x$ and $\mathbf f^{(L)} = \hat {\mathbf y}$ are special cases and of particular interest.


## What to do with the super-high dimensional functions?

Plot them along only a few dimensions. E.g.: near a saddle point, plot the loss along specific eigenvectors of the Hessian. 