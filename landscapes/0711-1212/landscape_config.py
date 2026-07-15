"""
landscape_config.py -- Configuration for objective landscape visualization.

All parameters for landscape plotting are defined here.
landscape.py reads from this file.
"""

# ============================================================================
# 1. Model
# ============================================================================
MODEL = "sin"                # linear, quadratic, poly2, poly3, nn, sin, sigmoid

# ============================================================================
# 2. Data dimensions
# ============================================================================
D_X = 4                         # dimension of explanatory variable x
D_Z = 8                         # dimension of instrument z

# ============================================================================
# 3. Data distribution parameters
# ============================================================================
MEAN_Z = 0.0
SIGMA_Z = 1.0
SIGMA_C = 0.5
SIGMA_Y = 0.3
SIGMA_X = 0.3

# ============================================================================
# 4. Random seed
# ============================================================================
SEED = 42

# ============================================================================
# 5. Evaluation dataset
# ============================================================================
N_EVAL = 10000                  # number of samples for objective estimation
N_COND = 2                      # conditional samples for CSO per z

# ============================================================================
# 6. Landscape grid
# ============================================================================
# Direction type: "random" (random unit vector), "param" (along specific param axis)
DIRECTION_TYPE = "random"
N_DIRECTIONS = 4                # number of random directions to plot
N_GRID_POINTS = 100             # points per direction
GRID_RADIUS = 10.0               # radius around theta* (in ||theta*|| units)

# ============================================================================
# 7. Objectives to plot
# ============================================================================
OBJECTIVES = ["dcov"]
DCOV_MAX_SAMPLES = 2000         # subsample for dCov (O(N²) complexity)

# ============================================================================
# 8. GMM weighting
# ============================================================================
GMM_W_TYPE = "identity"         # "identity" or "random"

# ============================================================================
# 9. Output
# ============================================================================
OUTDIR = None                   # None = "landscapes/<timestamp>"
SAVE_PLOT = None
