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


### Quadratic

Independently draw:

```math
\boldsymbol{\varepsilon}_x \sim N(\boldsymbol{0}_{d_x}, (1 - \rho) I_{d_x}), \quad \boldsymbol{z} \sim N(\boldsymbol{0}_{d_z}, I_{d_z}), \quad \boldsymbol{c} \sim N(\boldsymbol{0}_{d_x}, \rho I_{d_x})
```

Calculate

```math
\boldsymbol{x} = \gamma_*^\top \boldsymbol{z} + \boldsymbol{c} + \boldsymbol{\varepsilon}_x, \\
y = (\boldsymbol{x} + \boldsymbol{\theta}_*)^\top (\boldsymbol{x} + \boldsymbol{\theta}_*) + \frac{1}{\sqrt{d_x}} \boldsymbol{1}_{d_x}^\top \boldsymbol{c} +  \varepsilon_y.
```


### Logistic

Independently draw:

```math
\boldsymbol{\varepsilon}_x \sim N(\boldsymbol{0}_{d_x}, (1 - \rho) I_{d_x}), \quad \boldsymbol{z} \sim N(\boldsymbol{0}_{d_z}, I_{d_z}), \quad \boldsymbol{c} \sim N(\boldsymbol{0}_{d_x}, \rho I_{d_x})
```

Calculate

```math
\boldsymbol{x} = \gamma_*^\top \boldsymbol{z} + \boldsymbol{c} + \boldsymbol{\varepsilon}_x, \\
y = \left[ 1 + \exp \left( -  \boldsymbol{\theta}_*^\top \boldsymbol{x} \right) \right]^{-1} + \frac{1}{\sqrt{d_x}} \boldsymbol{1}_{d_x}^\top \boldsymbol{c} +  \varepsilon_y.
```


### ExpIV (exponential / log-link)

Independently draw:

```math
\boldsymbol{\varepsilon}_x \sim N(\boldsymbol{0}_{d_x}, (1 - \rho) I_{d_x}), \quad \boldsymbol{z} \sim N(\boldsymbol{0}_{d_z}, I_{d_z}), \quad \boldsymbol{c} \sim N(\boldsymbol{0}_{d_x}, \rho I_{d_x})
```

Calculate

```math
\boldsymbol{x} = \gamma_*^\top \boldsymbol{z} + \boldsymbol{c} + \boldsymbol{\varepsilon}_x, \\
y = \exp\left( \boldsymbol{\theta}_*^\top \boldsymbol{x} \right) + \frac{\tau}{\sqrt{d_x}} \boldsymbol{1}_{d_x}^\top \boldsymbol{c} + \sigma_\varepsilon \, \varepsilon_y.
```

The exponential mean is the canonical Poisson / log-link specification used by log-link IV and GMM estimators (Mullahy 1997; Windmeijer & Santos Silva 1997). Because $\exp$ amplifies any scale error, $\boldsymbol{\theta}_*$ is auto-normalised so that $\operatorname{Var}(\boldsymbol{\theta}_*^\top \boldsymbol{x})$ equals `DGP_EXPIV_INDEX_SCALE` (default $0.5$); larger values make the problem considerably harder and can make fixed-step SGD diverge.


### Probit (probit / normal-CDF link)

Calculate

```math
\boldsymbol{x} = \gamma_*^\top \boldsymbol{z} + \boldsymbol{c} + \boldsymbol{\varepsilon}_x, \\
y = \Phi\!\left( \boldsymbol{\theta}_*^\top \boldsymbol{x} \right) + \frac{\tau}{\sqrt{d_x}} \boldsymbol{1}_{d_x}^\top \boldsymbol{c} + \sigma_\varepsilon \, \varepsilon_y,
```

where $` \Phi `$ is the standard normal CDF and the linear index is normalised to `DGP_PROBIT_INDEX_SCALE` (default $1.0$). This is the systematic part of the classic probit model and the natural sibling of the logistic DGP. The outcome is kept **continuous** (regression form) so that $` \mathbb{E}[g(\boldsymbol{\theta}_*;\boldsymbol{x}) - y \mid \boldsymbol{z}] = 0 `$ holds and the IV algorithms target $` \boldsymbol{\theta}_* `$; a genuinely binary outcome would require control-function / MLE methods that lie outside this framework. Since $` \Phi' = \phi `$ has much thinner tails than the logistic derivative, the gradient essentially vanishes far from $` \boldsymbol{\theta}_* `$ — a flat-objective stress test.


### Sine (periodic, non-convex)

Calculate

```math
\boldsymbol{x} = \gamma_*^\top \boldsymbol{z} + \boldsymbol{c} + \boldsymbol{\varepsilon}_x, \\
y = \theta_{*,1} \sin\left( x_1 + \theta_{*,2} \right) + \sum_{j \ge 2} \theta_{*,j+1}\, x_j + \theta_{*,d_\theta} + \frac{\tau}{\sqrt{d_x}} \boldsymbol{1}_{d_x}^\top \boldsymbol{c} + \sigma_\varepsilon \, \varepsilon_y,
```

with $` \boldsymbol{\theta}_* = (\text{amplitude},\, \text{phase},\, \boldsymbol{b} \in \mathbb{R}^{d_x - 1},\, \text{intercept}) `$, so $` d_\theta = d_x + 2 `$. The phase enters through a sine, so the objective is **non-convex and multimodal** — the same family as DeepGMM's $` h^\star(x) = \sin(x) `$. It is a global-convergence stress test: different algorithms may settle in different local minima.

In the three DGPs above $` \tau `$ is the confounder coefficient (`DGP_*_C_COEF`, controls both endogeneity strength and the noise floor) and $` \sigma_\varepsilon `$ is the exogenous noise scale (`DGP_*_NOISE_EPS_Y`).



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


### Sieve GMM (Sieve-SGMM)

The CMR objective above rests on the conditional moment restriction $` \mathbb{E}(\varepsilon_y \vert \boldsymbol{z}) = 0 `$, which is equivalent to the infinite family of moment conditions $` \mathbb{E}[\psi(\boldsymbol{z})\,\varepsilon_y]=0 `$ for all suitable $` \psi `$. 
Since the optimal function $`\phi(\boldsymbol{z}) = \mathbb{E}(\nabla g(\boldsymbol{\theta}; \boldsymbol{x}) \vert \boldsymbol{z})`$ is related to nuisance parameters ($`\gamma`$), it is hard to estimate $`\theta`$.
Hence, we consider using a fixed, finite instrument basis $` \psi_K(\boldsymbol{z}) \in \mathbb{R}^{d_K} `$ (a *sieve*) and solves the resulting GMM with an online (stochastic-approximation) scheme. It consumes a single stream sample per step: **no two-sample oracle and no first-stage nuisance model are required**.

The key device is that the current sample is used only for the moment $` m_t `$, while the Jacobian and the weighting matrix are built from **past** samples only. Hence the preconditioner $` A_{t-1} `$ is $` \mathcal{F}_{t-1} `$-measurable and

```math
\mathbb{E}\big[A_{t-1}\, m_t(\theta_{t-1}) \mid \mathcal{F}_{t-1}\big]
=
A_{t-1}\, m_K(\theta_{t-1}),
```

which removes the bias that a single-sample estimate of $` M(\theta)^\top W m(\theta) `$ would otherwise have.

#### Sieve1 (implemented as `Sieve1` in `iv_sim/algorithms.py`)

- Instrument basis (no intercept): degree 1 is $` \psi_K(\boldsymbol{z}) = (z_1,\dots,z_{d_z}) `$; degree 2 adds symmetric quadratic monomials $` z_i z_j `$ ($` i\le j `$, polynomial basis) or the orthonormal Hermite basis $` H_2(z_i),\, z_i z_j `$ for $` \boldsymbol{z}\sim N(0,I) `$.
- Moment and Jacobian:

```math
m_t(\theta) = \psi_K(\boldsymbol{z}_t)\,\big(g_\theta(\boldsymbol{x}_t)-y_t\big),
\qquad
J_t(\theta) = \psi_K(\boldsymbol{z}_t)\,\nabla_\theta g_\theta(\boldsymbol{x}_t)^\top .
```

- Preconditioner from past statistics only:

```math
A_{t-1}
=
\big(\bar J_{t-1}^\top \hat W_{t-1} \bar J_{t-1} + \lambda I\big)^{-1}
\bar J_{t-1}^\top \hat W_{t-1},
```

with ridge $` \lambda `$ and a diagonal inverse-variance weight $` \hat W_{t-1}=\operatorname{diag}\big(1/(\bar S_{t-1}+\lambda)\big) `$.

- Parameter update (no projection, descent form):

```math
\theta_t = \theta_{t-1} - \alpha_t\, A_{t-1}\, m_t(\theta_{t-1}),
\qquad
\alpha_t = \alpha_0\, t^{-a},\quad a=0.5 .
```

- Running statistics (updated after the parameter step):

```math
\bar J_t = (1-\beta_t)\,\bar J_{t-1} + \beta_t\, J_t(\theta_{t-1}),
\qquad
\bar S_t = (1-\beta_t)\,\bar S_{t-1} + \beta_t\, \psi_K(\boldsymbol{z}_t)^2 \odot \varepsilon_t^2,
```

where $` \varepsilon_t = g_{\theta_{t-1}}(\boldsymbol{x}_t)-y_t `$, $` \odot `$ is elementwise product, and $` \beta_t = 1/t `$ (running average) or an exponential-moving-average rate if `sieve_ema > 0`.

#### Sieve2 (proposed refinement)

Sieve2 keeps the same "past preconditioner + current moment" skeleton but adds the standard stochastic-approximation machinery needed to make the theory hold.

- Basis with intercept:

```math
\psi_K(\boldsymbol{z}) = \big(1,\ \psi_1(\boldsymbol{z}),\dots,\psi_{K-1}(\boldsymbol{z})\big)^\top .
```

- Single-sample moment (sign flipped relative to Sieve1, $` q = -m `$):

```math
q_t(\theta) = \psi_K(\boldsymbol{z}_t)\,\big[y_t - g_\theta(\boldsymbol{x}_t)\big],
\qquad
J_t(\theta) = \psi_K(\boldsymbol{z}_t)\,\nabla_\theta g_\theta(\boldsymbol{x}_t)^\top .
```

- Update with projection onto a compact set $` \Theta `$ and step-size exponent $` a\in(1/2,1) `$ (e.g. $` t^{-0.75} `$):

```math
\theta_t
=
\Pi_\Theta\!\left[\theta_{t-1} - \gamma_t\, A_{t-1}\, q_t(\theta_{t-1})\right],
\qquad
\gamma_t = \gamma_0\, t^{-a},\quad a\in(\tfrac12,1).
```

(Sign convention: the proposal writes $` \theta_t=\Pi_\Theta[\theta_{t-1}+\gamma_t A_{t-1}q_t] `$ with $` q_t=\psi_K(\boldsymbol{z}_t)(y_t-g_\theta(\boldsymbol{x}_t)) `$ and $` J_t=\psi_K(\boldsymbol{z}_t)\nabla_\theta g_\theta(\boldsymbol{x}_t)^\top `$. Since $` q=-m `$ and $` \bar J=\mathbb{E}[\psi_K\nabla_\theta g^\top]=\nabla_\theta m `$, this "plus" step is a descent step and coincides with Sieve1's $` \theta-\alpha A m `$ with $` m=\psi_K(g-y) `$.)

- Preconditioner (as in Sieve1, but with $` \lambda_t\to 0 `$):

```math
A_{t-1}
=
\big(\bar J_{t-1}^\top \hat W_{t-1} \bar J_{t-1} + \lambda_t I\big)^{-1}
\bar J_{t-1}^\top \hat W_{t-1},
\qquad
\bar J_t = \bar J_{t-1} + \eta_t\big(J_t(\theta_{t-1})-\bar J_{t-1}\big).
```

- Weighting: instead of a diagonal weight, estimate the full moment covariance $` \widehat\Omega \approx \mathbb{E}[q_t(\theta_0) q_t(\theta_0)^\top] `$ from a small pilot/second stage and set $` \hat W = \widehat\Omega^{-1} `$.
- Averaging: output the Polyak–Ruppert average $` \bar\theta_T = T^{-1}\sum_{t=1}^T \theta_t `$.

**Theory target (finite-dimensional smooth nonlinear IV).** For fixed $` K `$, $` \mathbb{E}[u\mid z]=0 `$, smooth $` g_\theta `$, $` m_K(\theta)=0 `$ uniquely identifying $` \theta_0 `$, full-column-rank $` M_K=\mathbb{E}[J_t(\theta_0)] `$, projected updates with $` a\in(1/2,1) `$, and $` \lambda_t\to0 `$:

```math
\theta_t \to \theta_0\ \ \text{a.s.},
\qquad
\sqrt{T}\big(\bar\theta_T - \theta_0\big) \Rightarrow N(0, V_K),
\qquad
V_K = \big(M_K^\top \Omega_K^{-1} M_K\big)^{-1},
```

i.e. first-order equivalence with the offline optimal finite-moment GMM. For genuinely nonparametric $` g `$ or large MLPs one needs $` K=K_T\to\infty `$, regularisation and stronger identification; only local identification / functional convergence can be claimed, and weak IV / non-convexity are not circumvented.

#### Sieve3
The objective can be formulated as 
```math
F(\boldsymbol{\theta}) = \mathbb{E}\left[ (Y-g(\boldsymbol{\theta}; \boldsymbol{x})) \psi_K(\boldsymbol{z})^\top \right]
W \ 
\mathbb{E}\left[ \psi_K(\boldsymbol{z}) (Y-g(\boldsymbol{\theta}; \boldsymbol{x}))  \right],
```
the derivative is 
```math
\nabla F(\boldsymbol{\theta}) \propto \mathbb{E} \left[ \psi_K(\boldsymbol{z}_1)^\top W \psi_K(\boldsymbol{z}_2) \cdot  \left( g(\boldsymbol{\theta}; \boldsymbol{x}_2) - Y_2 \right) \nabla g(\boldsymbol{\theta}; \boldsymbol{x}_1)   \right].
```

Note that, if we take the orthonormal Hermite basis, the optimal weighting matrix is

```math
W = \mathbb{E}(\psi_K(\boldsymbol{z}) \psi_K(\boldsymbol{z})^\top \cdot \varepsilon_Y^2) = \mathbb{E}(\psi_K(\boldsymbol{z}) \psi_K(\boldsymbol{z})^\top) = I,
```

the second equality holds if $`\boldsymbol{z}`$ and $`\varepsilon_Y`$ is independent (or, more precisely, there is no heteroskedasticity, which means that $`\mathbb{E}(\varepsilon_Y^2 | \boldsymbol{z})`$ does not depend on $`\boldsymbol{z}`$).

Therefore, we can formulate the algorithms below:
```math
\boldsymbol{\theta}_{t+1} = \boldsymbol{\theta}_{t} - \alpha_{t+1} \widetilde{M}_{B_M}(\boldsymbol{\theta}_t)^\top  \widetilde{m}_{B_m}(\boldsymbol{\theta}_t),
```

where

```math
\widetilde{M}_{B_M}(\boldsymbol{\theta}) = \frac{1}{B_M} \sum_{i=1}^{B_M} \psi_K(\boldsymbol{z}_i) \nabla_{\theta} g(\boldsymbol{\theta} ; \boldsymbol{x}_{i})^\top, \\
\widetilde{m}_{B_m}(\boldsymbol{\theta}) = \frac{1}{B_m} \sum_{j=1}^{B_m} \psi_K(\boldsymbol{z}_i) \left(g(\boldsymbol{\theta}; \boldsymbol{x}_i) - y_i\right), \\
```
where we set $`\psi_K(\boldsymbol{z})`$ as the Hermite basis.
The average over all past updates is recorded:

```math
\bar{\boldsymbol{\theta}}_{t+1} = \frac{t-1}{t} \bar{\boldsymbol{\theta}}_t + \frac{1}{t} \boldsymbol{\theta}_{t+1}.
```

The final estimator is given by the overall average $` \bar{\boldsymbol{\theta}}_{N} `$, where $` N `$ is the total number of iterations.
