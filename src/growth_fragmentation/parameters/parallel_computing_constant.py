#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov  9 18:01:36 2022

@author: arat

    Copyright (C) 2025  Anaïs Rat

    This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with this program.  If not, see <https://www.gnu.org/licenses/>.

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


FEATURES = np.array([1, 2])  # Such that `tau_i(x) = FEATURES[i]`.

KAPPA = np.array([[0.5, 0.5], [0.5, 0.5]])  # Transition matrix.


# Parameters of the grids
# -----------------------

# Time grid.
PERIOD_COUNT = 10

# Size grid.
DX = 0.00025
X_COUNT = 35001
X_IDX_STEP = 500

STEP_BF_1ST_X_COUNT = 2
STEP_AF_LAST_X_COUNT = 12
sizes, dxs = scheme.regular_grid(DX, X_COUNT)
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
    print("sizes_phi_min = ", sizes_phi[0])
    print("sizes_phi_max = ", sizes_phi[-1])
    print("dx_phi = ", sizes_phi[1] - sizes_phi[0])
    print(
        '"SLURM_ARRAY_TASK_ID" should run from 0 to', len(FEATURES) * len(sizes_phi) - 1
    )
