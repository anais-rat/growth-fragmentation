#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 10 10:51:37 2022

@author: arat
"""

import numpy as np
import scheme_direct as scheme

# Parameters of the model
# -----------------------
# NB: information on the parameters are in the docstring of the function
#     `scheme.compute_evo_approximation`.

IS_CONSERVATIVE = False  # Equation form.

ALPHA = 0


def beta(sizes, alpha=ALPHA):  # Division rate per unit of size.
    """Growth rate function. Return the values of the growth rate at `sizes`."""
    return sizes**alpha


# FEATURES = np.array([1, 2]) # Such that `tau_i(x) = FEATURES[i]x`.
# KAPPA = {'uniform': np.array([[0.5, 0.5], # Transition matrix.
#                               [0.5, 0.5]]),
#           'irr1': np.array([[0.2, 0.8],
#                             [0.5, 0.5]]),
#           'irr2': np.array([[0.8, 0.2],
#                             [0.5, 0.5]]),
#           'irr3': np.array([[0.5, 0.5],
#                             [0.8, 0.2]]),
#           'irr4': np.array([[0.5, 0.5],
#                             [0.2, 0.8]])}

FEATURES = np.array([1, 2, 4])  # Such that `tau_i(x) = FEATURES[i]x`.
KAPPA = {
    "uniform": np.array(
        [[1 / 3, 1 / 3, 1 / 3], [1 / 3, 1 / 3, 1 / 3], [1 / 3, 1 / 3, 1 / 3]]
    ),
    "irr1": np.array([[0.8, 0.1, 0.1], [1 / 3, 1 / 3, 1 / 3], [1 / 3, 1 / 3, 1 / 3]]),
    "irr2": np.array([[1 / 3, 1 / 3, 1 / 3], [0.1, 0.8, 0.1], [1 / 3, 1 / 3, 1 / 3]]),
    "irr3": np.array([[1 / 3, 1 / 3, 1 / 3], [1 / 3, 1 / 3, 1 / 3], [0.1, 0.1, 0.8]]),
}
KEY = "irr3"
KAPPA = KAPPA[KEY]


# Parameters of the grids
# -----------------------

# Time grid.
PERIOD_COUNT = 20

# Size grid.
K = 2000
X_COUNT = 25001
X_IDX_STEP = 200

STEP_BF_1ST_X_COUNT = 50
STEP_AF_LAST_X_COUNT = 25
sizes, dxs, delta_x = scheme.geometrical_grid(X_COUNT, K)

sizes_phi = sizes[
    X_IDX_STEP
    * np.arange(STEP_BF_1ST_X_COUNT, X_COUNT // X_IDX_STEP - STEP_AF_LAST_X_COUNT + 1)
]

# # Parameters of data memory
# # -------------------------
# TSAVED_PER_PERIOD_COUNT = 10
# X_TEST = 1

if __name__ == "__main__":
    print("sizes_min = ", sizes[0])
    print("sizes_max = ", sizes[-1])
    print("delta_x = ", delta_x)
    print("max dx = ", np.max(np.diff(sizes)))
    print("sizes_phi_min = ", sizes_phi[0])
    print("sizes_phi_max = ", sizes_phi[-1])
    print("max dx_phi = ", np.max(np.diff(sizes_phi)))
    print("delta_x_phi = ", sizes_phi[1] / sizes_phi[0] - 1)
    print("len sizes = ", len(sizes))
    print("len sizes_phi = ", len(sizes_phi))
    print(
        '"SLURM_ARRAY_TASK_ID" should run from 0 to', len(FEATURES) * len(sizes_phi) - 1
    )
