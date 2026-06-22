# Instrumental Variable Regression

This repository contains implementations of several recent methods for instrumental variable (IV) regression.

## Problem Setting

### General Formulation

In IV regression, our goal is to use $\boldsymbol{x} \in \mathbb{R}^{d_x}$ to regress $y \in \mathbb{R}$.
The general formulation of IV regression is

$$
\begin{aligned}
y &= g(\boldsymbol{\theta}_*; \boldsymbol{x}) + \varepsilon_y, \\
\boldsymbol{x} &= \boldsymbol{h}(\gamma_* ; \boldsymbol{z}) + \boldsymbol{\varepsilon}_{\boldsymbol{x}},
\end{aligned}
$$

where $g(\boldsymbol{\theta}_*; \cdot)$ is the true model with parameter $\boldsymbol{\theta}_* \in \mathbb{R}^{d_\theta}$, $\varepsilon_y \in \mathbb{R}$ and $\boldsymbol{\varepsilon_x} \in \mathbb{R}^{d_x}$ are noises, $\boldsymbol{z} \in \mathbb{R}^{d_z}$ is the instrumental variable that is uncorrelated with both $\varepsilon_y$ and $\boldsymbol{\varepsilon_x}$, and $\boldsymbol{h}(\gamma_*; \cdot)$ is the true model between $\boldsymbol{x}$ and $\boldsymbol{z}$.
It should be noted that the explanatory variable $\boldsymbol{x}$ is **correlated** with $\varepsilon_y$, and consequently, conventional regression methods such as least squares generally fail.

For simulation, we explicitly represent the endogeneity (correlation between $\boldsymbol{x}$ and $\varepsilon_Y$) and formulate the problem as follows

$$
\begin{aligned}
y &= g(\boldsymbol{\theta}_*; \boldsymbol{x}) + u(\boldsymbol{c}) + \tilde{\varepsilon}_y, \\
\boldsymbol{x} &= \boldsymbol{h}(\gamma_* ; \boldsymbol{z}) + v(\boldsymbol{c}) + \tilde{\boldsymbol{\varepsilon}}_{\boldsymbol{x}},
\end{aligned}
$$

where $\tilde{\varepsilon}_y$ is the true noise and is uncorrelated with $\boldsymbol{x}$.

### Simulation Setting

| Setting | $g(\boldsymbol{\theta}_*; \boldsymbol{x})$ | $\boldsymbol{h}(\gamma_*; \boldsymbol{z})$ | $u(\boldsymbol{c})$ | $v(\boldsymbol{c})$ | $\boldsymbol{z}$ | $\boldsymbol{c}$ | $\tilde{\varepsilon}_y$ | $\tilde{\boldsymbol{\varepsilon}}_{\boldsymbol{x}}$ |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| Linear | $\boldsymbol{\theta}_*^\top \boldsymbol{x}$ | $\gamma_*^\top \boldsymbol{z}$ | $c$ | $c$ | $N(0, \sigma_{z}^2)$ | $N(0, \sigma_{c}^2)$ | $N(0, \sigma_{y}^2)$ | $N(\boldsymbol{0}, \sigma_{x}^2 I)$ |

## Algorithms

### TOSG-IVaR

TOSG-IVaR (Algorithm 1 in [Chen et al. 2024](https://arxiv.org/abs/2405.19463)) uses two-sample oracle to perform stochastic gradient descent.
The update is calculated by

$$
\boldsymbol{\theta}_{t+1} = \boldsymbol{\theta}_t - \alpha_{t+1} \bigl( g(\boldsymbol{\theta}_{t}; \boldsymbol{x}_t) - y_t \bigr) \nabla_{\theta} g(\boldsymbol{\theta}_t ; \boldsymbol{x}_t'),
$$

where $\boldsymbol{x}_t$ and $\boldsymbol{x}_t'$ are independently observed from the same $\boldsymbol{z}_t$ (conditionally independent).

### First-Order SLIM

First-Order SLIM (Algorithm 1 in [Chen et al. 2025](https://arxiv.org/abs/2510.20996)) solves IV regression from the perspective of the Generalized Methods of Moments (GMM).
The update is calculated by

$$
\boldsymbol{\theta}_{t+1} = \boldsymbol{\theta}_t - \alpha_{t+1} \bigl( g(\boldsymbol{\theta}_{t}; \boldsymbol{x}_{t,1}) - y_{t,1} \bigr) \nabla_{\theta} g(\boldsymbol{\theta}_t ; \boldsymbol{x}_{t,2}) \; \boldsymbol{z}_{t,1}^\top W \boldsymbol{z}_{t,2},
$$

where $(\boldsymbol{z}_{t,1}, \boldsymbol{x}_{t,1}, y_{t,1})$ and $(\boldsymbol{z}_{t,2}, \boldsymbol{x}_{t,2}, y_{t,2})$ are two independent pairs of data, and $W$ is a positive definite weighting matrix.
