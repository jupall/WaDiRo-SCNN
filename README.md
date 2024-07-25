# WaDiRo-SCNN
A Python implementation of the Wasserstein Distributionally Robust Shallow Convex Neural Networks from the work of Julien Pallage and Antoine Lesage-Landry.

---

## Introduction:
This work generalizes prior work done by [1,2] under the lens of Wasserstein Distributionally Robust Optimization. We train our model with an exact reformulation of the order-1 Wasserstein DRO problem [3,4,5].

Why is it interesting? :
- Non-linear scalable predictor;
- Low-stochasticity training compared to standard Shallow Neural Networks training programs;
- Provable out-of-sample performance: perfect for critical applications, e.g., energy, finance, healthcare;
- Easily solvable with open-source solvers, e.g., Clarabel [6];
- Conservative training procedure which generalizes standard regularization;
- Possibility to enforce hard physical constraints in training (physics-constrained NN).

## How it is made:
We use `cvxpy` to solve the models and convert the neural networks into `PyTorch`'s framework. 

In addition to the WaDiRo-SCNN, we also implement a WaDiRo linear regression and our own version of the SCNN.

## How to use it:

### Download:
```python
pip install wadiroscnn
```

### Create and train a model:

Choose between:
1. `wadiro_scnn()`
2. `wadiro_linreg()`
3. `scnn()`

```python
solver_name = "CLARABEL" # default option
verbose = True 

radius = 0.01 # the Wasserstein ball's radius
bias = True # construct the model with bias weights
wasserstein = "l2" # choose the norm used in the definition of the Wasserstein distance

model = wadiro_scnn()
model.train(X_train=data.X_train, Y_train=data.Y_train, radius = radius, bias = bias, max_neurons=max_neurons, verbose=verbose, solver=solver_name, wasserstein=wasserstein)
```


### Convert into a PyTorch neural network and predict:

```python

model_torch = model.get_torch_model(verbose = verbose)

output = model_torch(torch.tensor(data.X_test))

```

### Tutorial:
Notebook:
https://github.com/jupall/WaDiRo-SCNN/blob/main/experiments/tutorial.ipynb

## Cite our work and read our paper:

```
@misc{pallage2024wassersteindistributionallyrobustshallow,
      title={Wasserstein Distributionally Robust Shallow Convex Neural Networks}, 
      author={Julien Pallage and Antoine Lesage-Landry},
      year={2024},
      eprint={2407.16800},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/2407.16800}, 
}
```

## References:

[1] M. Pilanci and T. Ergen, “Neural networks are convex regularizers: Exact polynomial-time convex optimization formulations for two-layer networks,” in International Conference on Machine Learning, PMLR, 2020, pp. 7695–7705.

[2] A. Mishkin, A. Sahiner, and M. Pilanci, “Fast Convex Optimization for Two-Layer ReLU Networks: Equivalent Model Classes and Cone Decompositions,” in International Conference on Machine Learning, 2022.

[3] P. Mohajerin Esfahani and D. Kuhn, “Data-driven distributionally robust optimization using the Wasserstein metric: Performance guarantees and tractable reformulations,” Mathematical Programming, vol. 171, no. 1, pp. 115–166, 2018.

[4] D. Kuhn, P. M. Esfahani, V. A. Nguyen, and S. Shafieezadeh-Abadeh, “Wasserstein distributionally robust optimization: Theory and applications in machine learning,” in Operations research & management science in the age of analytics, Informs, 2019, pp. 130–166.

[5] R. Chen and I. Ch. Paschalidis, “Distributionally Robust Learning,” Foundations and Trends® in Optimization, vol. 4, no. 1-2, pp. 1–243, 2020s

[6] P. J. Goulart and Y. Chen, “Clarabel: An interior-point solver for conic programs with quadratic objectives,” Department of Engineering Science, University of Oxford, 2024.

## Acknowledgements:

This work was possible thanks to:

1. the incredible Python package `pyscnn` : https://github.com/pilancilab/scnn
2. the open-source convex solver `Clarabel` : https://github.com/oxfordcontrol/Clarabel.jl
3. the library `benchmark functions` : https://gitlab.com/luca.baronti/python_benchmark_functions/-/tree/master?ref_type=heads
4. `CVXPY` : https://www.cvxpy.org/version/1.1/index.html


