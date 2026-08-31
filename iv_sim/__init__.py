"""
iv_sim -- Instrumental Variable Regression Simulation Package

Modules:
    - config:          simulation parameter configuration
    - data_generator:  online/batch data generation (model-agnostic DGP)
    - models:          structural models g(theta; x) with gradient
    - algorithms:      TOSG-IVaR and First-Order SLIM implementations
    - metrics:         evaluation metrics and aggregation
    - visualization:   convergence plots and comparison tables
"""

from . import data_generator
from . import models
from . import algorithms
from . import metrics
from . import visualization
from . import config
