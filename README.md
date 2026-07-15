# Instrumental Variable Regression

This repository contains implementations of several recent methods for instrumental variable (IV) regression.

## Problem Setting

In IV regression, our goal is to use $\boldsymbol{x} \in \mathbb{R}^{d_x}$ to regress $y \in \mathbb{R}$.
The general formulation of IV regression is

```math
y = g(\boldsymbol{\theta}_{*}; \boldsymbol{x}) + \varepsilon_y, \\
\boldsymbol{x} = \boldsymbol{h}(\gamma_{*} ; \boldsymbol{z}) + \boldsymbol{\varepsilon}_{\boldsymbol{x}},
```

where $`g(\boldsymbol{\theta}_{*}; \cdot)`$ is the true model with parameter $`\boldsymbol{\theta}_{*} \in \mathbb{R}^{d_\theta}`$, $`\varepsilon_y \in \mathbb{R}`$ and $`\boldsymbol{\varepsilon}_{\boldsymbol{x}} \in \mathbb{R}^{d_x}`$ are noises, $`\boldsymbol{z} \in \mathbb{R}^{d_z}`$ is the instrumental variable that is uncorrelated with both $`\varepsilon_y`$ and $`\boldsymbol{\varepsilon}_{\boldsymbol{x}}`$, and $`\boldsymbol{h}(\gamma_{*}; \cdot)`$ is the true model between $`\boldsymbol{x}`$ and $`\boldsymbol{z}`$.
It should be noted that the explanatory variable $`\boldsymbol{x}`$ is **correlated** with $`\varepsilon_y`$, and consequently, conventional regression methods such as least squares generally fail.


## Data Generating Process

### TOSG
Independently draw

```math
\boldsymbol{z} \sim N(\boldsymbol{0}_{d_z}, I_{d_z}), \ \boldsymbol{h} \sim N(\boldsymbol{1}_{d_x}, I_{d_x}), \ \boldsymbol{\epsilon}_x \sim N(\boldsymbol{0}_{d_x}, I_{d_x}), \  \epsilon_y \sim N(0, 1).
```

Calculate

```math
\boldsymbol{x} = \phi(\gamma_*^\top \boldsymbol{z}) + c \cdot (\boldsymbol{h} + \boldsymbol{\epsilon}_x), \\
\boldsymbol{y} = \boldsymbol{\theta}_*^\top \boldsymbol{x} + c \cdot (h_1, \epsilon_y),
```
where $`c > 0`$ is a scalar to control the variance of the noise vector, and $`h_1`$ is the first coordinate of $`h`$.

Hyperparameter settings in the paper:

``` math
\begin{align*}
(d_x, d_z):& \  (4, 8), (8, 16); \\
c:& \  0.1, 1.0; \\
\phi(s):& \ s, s^2.
\end{align*}
```


### OTSG

Draw:

``` math
\boldsymbol{\epsilon} \sim  N(\boldsymbol{0},\sigma_{\epsilon}^2 I_{d_x}), \quad  \nu  \sim N(\rho \epsilon_1, 0.25),
```

where $`\epsilon_1`$ is the first coordinate of $`\boldsymbol{\epsilon}`$.
Then calculate:

```math
\boldsymbol{x} = \gamma_*^\top \boldsymbol{z} + \boldsymbol{\epsilon}, \quad
y = \boldsymbol{\theta_*^\top} \boldsymbol{x} + \nu.
```

Hyperparameter settings in the paper:

``` math
\begin{align*}
(d_x, d_z):& \  (1, 1), (8, 16); \\
\rho:& \  1.0, 4.0; \\
\sigma_\epsilon:& \ 0.5, 1.0.
\end{align*}
```


### DeepGMM
Independently draw:

```math
\epsilon \sim N(0,1), \quad \gamma, \delta \sim N(0,0.1), \\
\boldsymbol{z} = (z_1, z_2) \sim \text{Unif}([-3, 3]^2), \\
```

then calculate

```math
x = z_1 + \epsilon + \gamma, \\
y = h^\star(x) + \epsilon + \delta.
```

Settings of $h^\star$:

```math
\begin{align*}
\text{step}:&  \quad h^\star (x) = I(x>0), \\
\text{abs}:& \quad h^\star (x) = |x|, \\
\text{linear}:& \quad h^\star (x) = x, \\
\text{sin}:& \quad h^\star (x) = \sin(x).
\end{align*}
```






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

#### DCOV3
DCOV3 is an order-3 method to optimize $`F(\boldsymbol{\theta})`$.
The gradient can be derived by

```math
\begin{align*}
\nabla F(\boldsymbol{\theta}) &= \mathbb{E}[\Vert \boldsymbol{z} - \boldsymbol{z}' \Vert \cdot \nabla \vert \varepsilon_y - \varepsilon_y' \vert] 
+ \mathbb{E}[\Vert \boldsymbol{z} - \boldsymbol{z}' \Vert] \cdot \mathbb{E}[\nabla \vert \varepsilon_y - \varepsilon_y' \vert] - 2 \mathbb{E}[\Vert \boldsymbol{z} - \boldsymbol{z}'' \Vert \cdot \nabla \vert \varepsilon_y -\varepsilon_y' \vert ] \\

&=\mathbb{E} \left\{ \big(\Vert \boldsymbol{z} - \boldsymbol{z}' \Vert - 2 \Vert \boldsymbol{z} - \boldsymbol{z}'' \Vert  + \mathbb{E}[\Vert \boldsymbol{z} - \boldsymbol{z}' \Vert] \big)  \cdot  \nabla \vert \varepsilon_y - \varepsilon_y' \vert  \right\}.

\end{align*}
```

Hence, we can construct an unbiased order-3 stochastic gradient estimator. At iteration $`t`$, three i.i.d. samples
$`\{(\boldsymbol z_i,\boldsymbol x_i,y_i)\}_{i=1}^3`$ are drawn and the kernel

```math
v_t(\boldsymbol\theta;i,j,k)
=
\left(
\|\boldsymbol z_i-\boldsymbol z_j\|
-
2\|\boldsymbol z_i-\boldsymbol z_k\|
+
\hat\delta_{t-1}
\right)
\operatorname{sgn}(\varepsilon_i-\varepsilon_j)
\left(
\nabla g(\boldsymbol\theta;\boldsymbol x_j)
-
\nabla g(\boldsymbol\theta;\boldsymbol x_i)
\right)
```

is symmetrized over all permutations to obtain the stochastic gradient

```math
\widehat{\nabla F}(\boldsymbol\theta)
=
\frac1{3!}
\sum_{(i,j,k)!}
v_t(\boldsymbol\theta;i,j,k).
```

The parameter is updated by

```math
\boldsymbol\theta_t
=
\boldsymbol\theta_{t-1}
-
\alpha_t
\widehat{\nabla F}(\boldsymbol\theta_{t-1}).
```

The quantity

```math
\delta=\mathbb E\|\boldsymbol z-\boldsymbol z'\|
```

is unknown and is estimated online by the running average of the order-2 U-statistic

```math
\hat\delta_t
=
\frac{t-1}{t}\hat\delta_{t-1}
+
\frac1t
\cdot
\frac1{\binom32}
\sum_{i<j}
\|\boldsymbol z_i-\boldsymbol z_j\|.
```


#### DCOV4

We can also construct unbiased gradient estimator via U-statistic.

```math
\begin{align*}
\nabla F(\boldsymbol{\theta}) =& \mathbb{E}[\Vert \boldsymbol{z} - \boldsymbol{z}' \Vert \cdot \nabla \vert \varepsilon_y - \varepsilon_y' \vert] 
+ \mathbb{E}[\Vert \boldsymbol{z} - \boldsymbol{z}' \Vert] \cdot \mathbb{E}[\nabla \vert \varepsilon_y - \varepsilon_y' \vert] \\
&- \mathbb{E}[\Vert \boldsymbol{z} - \boldsymbol{z}'' \Vert \cdot \nabla \vert \varepsilon_y -\varepsilon_y' \vert ] - \mathbb{E}[\Vert \boldsymbol{z} - \boldsymbol{z}'' \Vert \cdot \nabla \vert \varepsilon_y -\varepsilon_y' \vert ].

\end{align*}
```

The kernel can be defined as

```math
\begin{align*}

v(\boldsymbol{\theta};i,j,k,l) =& \Vert \boldsymbol{z}_i - \boldsymbol{z}_j \Vert \cdot \nabla \vert \varepsilon_{y,i} - \varepsilon_{y,j} \vert
+ \Vert \boldsymbol{z}_i - \boldsymbol{z}_j \Vert \cdot \nabla \vert \varepsilon_{y,k} - \varepsilon_{y,l} \vert \\
&- \Vert \boldsymbol{z}_i - \boldsymbol{z}_j \Vert \cdot \nabla \vert \varepsilon_{y,i} -\varepsilon_{y,k} \vert - \Vert \boldsymbol{z}_i - \boldsymbol{z}_k \Vert \cdot \nabla \vert \varepsilon_{y,i} -\varepsilon_{y,j} \vert ,

\end{align*}
```

and we permute it over $` i,j,k,l `$ to obtain the unbiased gradient estimator

```math
\widehat{\nabla F}(\boldsymbol\theta)
=
\frac1{4!}
\sum_{(i,j,k,l)!}
v_t(\boldsymbol\theta;i,j,k,l).
```

The parameter is updated by

```math
\boldsymbol\theta_t
=
\boldsymbol\theta_{t-1}
-
\alpha_t
\widehat{\nabla F}(\boldsymbol\theta_{t-1}).
```


