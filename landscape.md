# Visualizing Landscape

在研究非线性的 IV 回归时，我们会关心不同方法的目标函数的 Landscape。
虽然目标函数一般是期望的形式（即随机优化的函数形式），但模型已知时依然可以通过随机采样的方式直接把目标函数近似地绘制出来，并观察真值点附近目标函数的形态。



## Objectives

### CSO
```math
F(\boldsymbol{\theta}) = \mathbb{E}_Z \mathbb{E}_{Y|Z} \left[ \left( Y - \mathbb{E}_{X|Z}[g(\boldsymbol{\theta}; \boldsymbol{x})] \right)^2 \right]
```

### GMM
```math
F(\boldsymbol{\theta})  = \mathbb{E}\left[ \big(Y_i - g(\boldsymbol{\theta}; \boldsymbol{x}_i)\big) \left(\boldsymbol{z}_i^\top W \boldsymbol{z}_j\right) \big(Y_j - g(\boldsymbol{\theta}; \boldsymbol{x}_j)\big) \right]
```

### DCOV
```math
F(\boldsymbol{\theta})  = \text{dCov}^2 \left[ \boldsymbol{z}, Y-g(\boldsymbol{\theta}; \boldsymbol{x}) \right] 
```