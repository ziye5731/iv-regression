# Instrumental Variable Regression

This repository contains implementations of several recent methods for instrumental variable (IV) regression.

## Problem Setting

### General Formulation

In IV regression, our goal is to use $\boldsymbol{x} \in \mathbb{R}^{d_x}$ to regress $y \in \mathbb{R}$.
The general formulation of IV regression is

```math
y = g(\boldsymbol{\theta}_{*}; \boldsymbol{x}) + \varepsilon_y, \\
\boldsymbol{x} = \boldsymbol{h}(\gamma_{*} ; \boldsymbol{z}) + \boldsymbol{\varepsilon}_{\boldsymbol{x}},
```

where $`g(\boldsymbol{\theta}_{*}; \cdot)`$ is the true model with parameter $`\boldsymbol{\theta}_{*} \in \mathbb{R}^{d_\theta}`$, $`\varepsilon_y \in \mathbb{R}`$ and $`\boldsymbol{\varepsilon}_{\boldsymbol{x}} \in \mathbb{R}^{d_x}`$ are noises, $`\boldsymbol{z} \in \mathbb{R}^{d_z}`$ is the instrumental variable that is uncorrelated with both $`\varepsilon_y`$ and $`\boldsymbol{\varepsilon}_{\boldsymbol{x}}`$, and $`\boldsymbol{h}(\gamma_{*}; \cdot)`$ is the true model between $`\boldsymbol{x}`$ and $`\boldsymbol{z}`$.
It should be noted that the explanatory variable $`\boldsymbol{x}`$ is **correlated** with $`\varepsilon_y`$, and consequently, conventional regression methods such as least squares generally fail.

For simulation, we explicitly represent the endogeneity (correlation between $`\boldsymbol{x}`$ and $`\varepsilon_{Y}`$) and formulate the problem as follows

```math
y = g(\boldsymbol{\theta}_{*}; \boldsymbol{x}) + u(\boldsymbol{c}) + \tilde{\varepsilon}_y, \\
\boldsymbol{x} = \boldsymbol{h}(\gamma_{*} ; \boldsymbol{z}) + v(\boldsymbol{c}) + \tilde{\boldsymbol{\varepsilon}}_{\boldsymbol{x}},
```

where $`\tilde{\varepsilon}_y`$ is the true noise and is uncorrelated with $`\boldsymbol{x}`$.

### Simulation Setting

| Setting | $`g(\boldsymbol{\theta}_{*}; \boldsymbol{x})`$ | $`\boldsymbol{h}(\gamma_{*}; \boldsymbol{z})`$ | $`u(\boldsymbol{c})`$ | $`v(\boldsymbol{c})`$ | $`\boldsymbol{z}`$ | $`\boldsymbol{c}`$ | $`\tilde{\varepsilon}_y`$ | $`\tilde{\boldsymbol{\varepsilon}}_{\boldsymbol{x}}`$ |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| Linear | $`\boldsymbol{\theta}_{*}^\top \boldsymbol{x}`$ | $`\gamma_{*}^\top \boldsymbol{z}`$ | $`c`$ | $`c`$ | $`N(0, \sigma_{z}^2)`$ | $`N(0, \sigma_{c}^2)`$ | $`N(0, \sigma_{y}^2)`$ | $`N(\boldsymbol{0}, \sigma_{x}^2 I)`$ |

## Algorithms

### TOSG-IVaR

TOSG-IVaR (Algorithm 1 in [Chen et al. 2024](https://arxiv.org/abs/2405.19463)) uses two-sample oracle to perform stochastic gradient descent.
The update is calculated by

```math
\boldsymbol{\theta}_{t+1} = \boldsymbol{\theta}_{t} - \alpha_{t+1} \bigl( g(\boldsymbol{\theta}_{t}; \boldsymbol{x}_{t}) - y_{t} \bigr) \nabla_{\theta} g(\boldsymbol{\theta}_{t} ; \boldsymbol{x}_{t}'),
```

where $`\boldsymbol{x}_{t}`$ and $`\boldsymbol{x}_{t}'`$ are independently observed from the same $`\boldsymbol{z}_{t}`$ (conditionally independent).


### OTSG-IVaR
OTSG-IVaR (Algorithm 2 in [Chen et al. 2024](https://arxiv.org/abs/2405.19463)) uses one-sample oracle to perform stochastic gradient descent.
The update is calculated by

```math
\boldsymbol{\theta}_{t+1} = \boldsymbol{\theta}_{t} - \alpha_{t+1} \left( g(\boldsymbol{\theta}_{t}; \boldsymbol{h}(\boldsymbol{\gamma}_{t} ; \boldsymbol{z}_{t})) - y_{t} \right) \nabla_{\theta} g(\boldsymbol{\theta}_{t} ; \boldsymbol{h}(\boldsymbol{\gamma}_{t} ; \boldsymbol{z}_{t})), \\
\boldsymbol{\gamma}_{t+1} = \boldsymbol{\gamma}_{t} - \beta_{t+1}  \nabla_{\gamma} \boldsymbol{h}(\boldsymbol{\gamma}_{t} ; \boldsymbol{z}_{t})^\top \left( \boldsymbol{h}(\boldsymbol{\gamma}_{t} ; \boldsymbol{z}_{t}) - \boldsymbol{x}_{t} \right),
```
In this algorithm, we only need one pair of data $`(\boldsymbol{z}_{t}, \boldsymbol{x}_{t}, y_{t})`$ at each step, and we use the current estimate of $`\boldsymbol{x}_{t}`$ with $`\boldsymbol{h}(\boldsymbol{\gamma}_{t}; \boldsymbol{z}_{t})`$, just like the two-stage least squares (2SLS) method.

### SLIM

#### First-Order SLIM
First-Order SLIM (Algorithm 1 in [Chen et al. 2025](https://arxiv.org/abs/2510.20996)) solves IV regression from the perspective of the Generalized Methods of Moments (GMM).
The update is calculated by

```math
\boldsymbol{\theta}_{t+1} = \boldsymbol{\theta}_{t} - \alpha_{t+1} \widetilde{M}_{B_M}(\boldsymbol{\theta}_t)^\top W \widetilde{m}_{B_m}(\boldsymbol{\theta}_t),
```

where

```math
\widetilde{M}_{B_M}(\boldsymbol{\theta}) = \frac{1}{B_M} \sum_{i=1}^{B_M} \boldsymbol{z}_i \nabla_{\theta} g(\boldsymbol{\theta} ; \boldsymbol{x}_{i})^\top, \\
\widetilde{m}_{B_m}(\boldsymbol{\theta}) = \frac{1}{B_m} \sum_{j=1}^{B_m} \boldsymbol{z}_i \left(g(\boldsymbol{\theta}; \boldsymbol{x}_i) - y_i\right). \\
```

And the average over all past updates is also recorded:

```math
\bar{\boldsymbol{\theta}}_{t+1} = \frac{t-1}{t} \bar{\boldsymbol{\theta}}_t + \frac{1}{t} \boldsymbol{\theta}_{t+1}.
```

The final estimator is given by the overall average $` \bar{\boldsymbol{\theta}}_{N} `$, where $` N `$ is the total number of iterations.


In the streaming setting ($`B_M = B_m = 1 `$), the update is calculated by

```math
\boldsymbol{\theta}_{t+1} = \boldsymbol{\theta}_{t} - \alpha_{t+1} \bigl( g(\boldsymbol{\theta}_{t}; \boldsymbol{x}_{t,1}) - y_{t,1} \bigr) \nabla_{\theta} g(\boldsymbol{\theta}_{t} ; \boldsymbol{x}_{t,2}) \; \boldsymbol{z}_{t,1}^\top W \boldsymbol{z}_{t,2},
```

where $`(\boldsymbol{z}_{t,1}, \boldsymbol{x}_{t,1}, y_{t,1})`$ and $`(\boldsymbol{z}_{t,2}, \boldsymbol{x}_{t,2}, y_{t,2})`$ are two independent pairs of data, and $`W`$ is a positive definite weighting matrix.


### Conditional Moment Restriction (CMR) 

SLIM uses a moment condition $` \mathbb{E}(\boldsymbol{z} \varepsilon_y) =  \boldsymbol{0}`$. 
However, it's weaker than $` \mathbb{E}( \varepsilon_y \vert \boldsymbol{z}) = 0 `$, which indicates $` \mathbb{E}(\phi(\boldsymbol{z}) \varepsilon_y) =  \boldsymbol{0}`$ for all $` \phi `$.
[Chamberlain 1987](https://doi.org/10.1016/0304-4076(87)90015-7) derived an efficient choice for $`\phi `$, which is by letting 

```math
\phi(\boldsymbol{z}) = \mathbb{E}(\nabla g(\boldsymbol{\theta}; \boldsymbol{x}) \vert \boldsymbol{z}),
```

and hence we have moment condition

```math
\mathbb{E}\left[ (Y-g(\boldsymbol{\theta}; \boldsymbol{x})) \cdot \mathbb{E}\left( \nabla g(\boldsymbol{\theta}; \boldsymbol{x}) \vert \boldsymbol{z} \right)  \right] = \boldsymbol{0},
```

where the LHS is exactly the gradient of the objective function of TOSG-IVaR.
However, we try estimating $`\boldsymbol{\theta}`$ via optimizing such GMM objective, which is equivalent to optimizing the quadratic form of the gradient of TOSG objective.
Let $` \boldsymbol{m}(\boldsymbol{\theta}, \gamma) = \left( Y - g(\boldsymbol{\theta}, \boldsymbol{x}) \right) \cdot \mathbb{E}\left( \nabla g(\boldsymbol{\theta}; \boldsymbol{x}) \vert \boldsymbol{z} \right)`$, the objective is

```math
F(\boldsymbol{\theta}, \gamma) = \mathbb{E}(\boldsymbol{m}(\boldsymbol{\theta}, \gamma))^\top W \  \mathbb{E}(\boldsymbol{m}(\boldsymbol{\theta}, \gamma)),
```

where $`W`$ is the positive-definite weighting matrix. 



### Distance Covariance Optimization
It is natural to think whether we can use a condition stronger than $` \mathbb{E}( \varepsilon_y \vert \boldsymbol{z}) = 0 `$, which is $` \mathbb{z} \perp \!\!\! \perp \varepsilon_Y `$. 
A commonly used statistic to measure "independence" of two random vectors is the distance covariance, which is defined as 

```math
\text{dCov}^2(X, Y) = \mathbb{E}[\|X - X'\|_p \|Y - Y'\|_q] + \mathbb{E}[\|X - X'\|_p] \mathbb{E}[\|Y - Y'\|_q] - 2\mathbb{E}[\|X - X'\|_p \|Y - Y''\|_q],
```

where $` X \in \mathbb{R}^p `$, $` Y \in \mathbb{R}^q `$, $`(X, Y), (X', Y'), (X'', Y'')`$ are i.i.d., and we have 

```math
\text{dCov}^2(X, Y) \iff X \text{ and } Y \text{ are independent.}
```

Hence, we can estimate $`\boldsymbol{\theta}`$ via minimizing the distance covariance between instrumental variable $`\boldsymbol{z}`$ and residual $` Y- g(\boldsymbol{\theta}; \boldsymbol{x}) `$.
The objective is

```math

F(\boldsymbol{\theta}) = \text{dCov}^2(\boldsymbol{z}, Y- g(\boldsymbol{\theta}; \boldsymbol{x}))

```


