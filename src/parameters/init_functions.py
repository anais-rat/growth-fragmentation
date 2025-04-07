#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec 20 22:25:52 2023

@author: arat
"""

import numpy as np


def generate_beta(alpha, constant=1):
    """ Return the function defining the growth rate per unit of size 
    associated to parameters `alpha` and `constant`.

    """
    def beta(sizes):
        """ Growth rate per unit of size function. Return the values of the
        growth rate at `sizes`.
    
        """
        return constant * sizes ** alpha
    return beta

def generate_initial_condition(parameters, format_choice,
                               dxs_to_normalize=None):
    """ Return the function that takes `sizes`and `features` in argument and
    return the initial distribution (at these points) which corresponds to
    the parameters specified through `parameters`, format_choice` and
    `dxs_to_normalize`.

    Parameters
    ----------
    parameters : ndaray
        1D array of parameters `[par_1, par_2]` for the initial distribution.
    format_choice : string
        Initial distribution constant along features, along sizes:
        'heaviside': heaviside along x from index `par_1` to index `par_2`.
        'exponential': for `e^(-par_2 * x^2) * x^par_1`.
        Or variable also in feature:
        'dirac': for "dirac" (i.e. value `1e3`) in `v=par_1`, `x=par_2`.
        'linear': for `n(0, v, x) = v * x`.
    dxs_to_normalize: None or ndarray, optional
        Either None, and the initial distribution is not normalized, or
        1D array (x_count,) of the size intervals to use for normalization
        (typically `np.append(sizes[1], sizes[1:] - sizes[:-1]`).

    Returns
    -------
    initial_condition : function
        Function of sizes and features.

    """
    par_1, par_2 = parameters

    def initial_condition(sizes, features):
        """
        Parameters
        ----------
        features, sizes : ndarray
            1D array (v_count,), (x_count,) of the features and the sizes (resp.).
    
        Returns
        -------
        n_init : ndarray
            2D array (v_vount, x_count) describing the initial distribution.
    
        """
        feature_count, size_count = len(features), len(sizes)
        ones = np.ones((feature_count, 1))
        if format_choice == 'heaviside':
            n_init = np.zeros((feature_count, size_count))
            n_init[:, par_1:par_2] = 1
        elif format_choice == 'exponential':
            n_init = np.dot(ones, [sizes**par_1 * np.exp(-par_2 * sizes**2)])
        elif format_choice == 'dirac':
            n_init = np.zeros((feature_count, size_count))
            n_init[par_1, par_2] = 1e3
        elif format_choice == 'dirac2':
            n_init = np.zeros((feature_count, size_count))
            n_init[par_1, par_2] = 1e2
        elif format_choice == 'dirac1':
            n_init = np.zeros((feature_count, size_count))
            n_init[par_1, par_2] = 1e1
        elif format_choice == 'linear':
            n_init = np.dot(np.transpose([features]), [sizes])
        else:
            return "Wrong value for 'format_choice'"
        # Normalization if asked.
        if not isinstance(dxs_to_normalize, type(None)):
            n_init_sum = np.sum(n_init * dxs_to_normalize)
            n_init = n_init / n_init_sum
        return n_init
    return initial_condition


def geometrical_grid(point_count, k):
    """ Return the geometrical grid `points` (shape (point_count,)) made of
    `point_count` points (or `point_count + 1` if `point_count` is even) spaced
    with variable steps `dxs` corresponding to a constant geometrical step
    depending on the precision `k`.

    """
    if point_count % 2 == 0:  # Verification of arguments format.
        print("Warning: even number of points given in argument to `geometrica"
              "l_grid`, an additional point has been added to the grid.")
        point_count += 1
    delta_x = 2 ** (1/k) - 1  # Geometrical step.
    point_count_half = int(point_count / 2)
    points = 2 ** (np.arange(- point_count_half, point_count_half + 1) / k)
    dxs = np.diff(np.append(0, points))  # Grid steps.
    return points, dxs, delta_x


def regular_grid(step, point_count):
    """ Return the grid `points` (shape (point_count,)) made of `point_count`
    points regularly spaced of `step`.

    """
    points = np.linspace(0, step * (point_count - 1), point_count)
    dxs = np.diff(np.append(points, points[-1] + step))
    return points, dxs
