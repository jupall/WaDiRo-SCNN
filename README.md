# WaDiRo-SCNN
A Python implementation of the Wasserstein Distributionally Robust Shallow Convex Neural Networks from the work of Julien Pallage and Antoine Lesage-Landry.

🚧 **Page in construction** 🚧
---
---
## Introduction:
This work generalizes prior work done by [1, 2, 3] under the lens of Wasserstein Distributionally Robust Optimization. We train our model with an exact reformulation of the order-1 Wasserstein DRO problem [4,5].

Why is it interesting? :
- Non-linear predictor which is scalable;
- Low-stochasticity training compared to standard Shallow Neural Networks training.
- Provable out-of-sample performance perfect for critical applications, e.g., energy, finance, healthcare;
- Easily solvable with open-source solvers, e.g., Clarabel [6];
- Conservative training procedure which generalizes standard regularization;
- Possibility to enforce hard physical constraints in training (physics-constrained NN).

## How it is made:
We use 'cvxpy' to solve the models and convert the neural networks into PyTorch's framework.

## How to use it:

### Create a model:

### Train the model under different options:

#### Choose the $\ell_1$-norm or the $\ell_2$-norm in the definition of the Wasserstein distance


### Convert into a PyTorch neural network:

## Cite our work:

## References:


