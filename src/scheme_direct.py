#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug 23 10:39:40 2021

From the script Direct_stat_osci_geom.m which follows the numerical
scheme presented in "Bernard, Doumic & Gabriel, Cyclic asymptotic... (2018)"

@author: anais rat
"""

import os
import numpy as np

import src.parameters.init_functions as init
import src.write_path as wp


# ---------
# Auxiliary
# ---------


def normalize_wrt_0_axis(array, weights=None):
    """Normalize a nD array along its rows (say of length `r`) using provided
    weights if some, or weights `np.ones(r)` otherwise (s.t. the sum along rows
    is one).

    """
    if not isinstance(weights, type(None)):
        sum_wrt_0_axis = np.dot(array, weights)
    else:
        sum_wrt_0_axis = np.sum(array, 1)
    return array / np.transpose([sum_wrt_0_axis])


def find_nearest(array, value):
    """Find the index and value of the element in the given array `array` that
    is closest to the specified value `value`.

    """
    array = np.asarray(array)
    idx = np.argmin(np.abs(array - value))
    return idx, array[idx]


def test_kappa_conform(kappa, v_count):
    """Raise an error if arguments are not conform."""
    if len(kappa) != v_count:
        raise ValueError(
            "The dimensions of the set of features and the "
            "variability kernel do not match."
        )


def test_sizes_conform(sizes, x_test):
    """Raise an error if arguments are not conform."""
    if sizes[-1] < x_test:
        print("sizes[-1]: ", sizes[-1])
        print("x_test: ", x_test)
        raise ValueError(
            "The size `x_test` at which saved `n_evo` is outside" "the x-grid."
        )


# -----------------
# Numerical schemes
# -----------------

#  Linear growth rates.
# ---------------------


def compute_evo_approximation(
    is_conservative,
    features,
    par_beta,
    par_kappa,
    par_ninit,
    par_grids,
    x_test,
    is_saved=False,
    normalization_wtr_feature=False,
    tmax=None,
):
    """Compute an approximation of the solution to the growth-fragmentation
    equation, in its conservative or non-conservative form, in the case of
    equal mitosis and variability in (linear) growth rate, with coefficients
    'tau(v,x) = vx' and 'gamma(v,x) = beta(x) tau(v,x)'.

    Uses a scheme chosen to reduce as much as possible dissipativity to capture
    the oscillations in the case mitose and tau linear:
        - geometrical grid in the x variable (size),
        - splitting in two steps handling growth through an upwind scheme with
          optimized CFL = 1 for the fastest subpopulation (with feature v_max).

    Parameters
    ----------
    is_conservative : bool
        True for the conservative equation, False for the non-conservative.
    features : ndarray
        1D array (v_count,) of the features.
    par_beta : list
        List of parameters `[alpha, constant]` defining the division rate per
        unit of size, function of the size only. See parameters.init_functions.
    par_kappa : list
        List made of:
        kappa : ndarray
            2D array (v_count, v_count) variability kernel (stochastic matrix
            st. `kappa[i, j]` is the proba for a cell of feature `features[i]`
            to give birth to a cell of feature `features[j]`).
        kappa_name : string or None
            If not None and simulation is saved, saving path uses the short
            name `kappa_name` rather than the composant of `kappa`.
    par_ninit : list
        List off parameters defining the initial condition.
        See parameters.init_functions.generate_initial_condition.
    par_grigs : list
        List of the following parameters, for space and time grigs:
        k : int
            Precision of the geometrical step `delta_x = 2^(1/k)+1` for x-axis.
        x_count : int
            Odd integer, the number of points of the x-grid.
        period_count : int
            Number of periods ln(2) before ending computations.
        tsaved_per_period_count : int
            Number of time steps per period ln(2) at which data is stored.
    x_test : int
        The size at which the value of the eigenvector is kept in memory
        (at all times and all features).
    is_saved : bool
        True for the output be saved in `data`, False otherwise.
    normalization_wtr_feature : bool, optional
        If true `n_evo_norm_wrt_v` computed otherwise set to None.
    tmax : float or None, optional
        Maximum time of computation, default is None which corresponds to
        `np.log(2) * period_count`. The actual maximum time can be less if
        `n_tdiff` becomes less than 1e-5, then computation stops.

    Returns
    -------
    A dictionary with the following keys and corresponding entries:
    times : ndarray
        1D array (t_count,) of the total time-grid.
    t_test : ndarray
        1D array (t_test_count,) of times at which 'n_evo' is stored.
    sizes : ndarray
        1D array (x_count,) of the (geometrical) size-grid.
    n_test : ndarray
        2D array (t_count, v_count) of the density evaluated at the size
        `x_test` for all features and at all times.
    n_evo: ndarray
        3D array (t_test_count, v_count, x_count) of the time evolution of the
        feature-size distribution.
    n_evo_normalized: ndarray
        Idem but normalized as: n(t, v, x) exp(-t * lambda_n[-1]).
    n_evo_norm_wrt_v:  ndarray
        Idem but normalized for each t, v as: n(t, v, x) / (int n(t, v, x) dx).
    n_sum_evo : ndarray
        1D array (t_count,) of the time evolution of the total number of
        individuals (iint n(t, v, x) dv dx).
    n_tdiff_evo : ndarray
        1D array (t_count,) of the difference of the normalized distribution
        between each time in l1 norm divided by dt.
    lambda_estimates : list
        List of time evolutions (starting from time `2dt`) of estimates of
        lambda: `lambda_estimates[i]` is a 1D array of length `t_count-2` s.t.
        - i=0: ('lambda_n') at each time 't[j]', j>0, the linear regression on
            '[int(t[j]/2), t[j]]' of the log of the total number of cell.
        In the case conservative:
            - i=1: ('lambda_error')
                   iint n_evo(t,x,v) [tau(x,x) - gamma(x,v)] dx dv.
        In the case non-conservative:
            - i=1: ('lambda_gamma') iint n_evo(t,v,x) gamma(v,x) dvdx.
            - i=2: ('lambda_tau') iint vx n_evo(t,v,x) dvdx / (iint x n_evo).

    """
    k, x_count, period_count, tsaved_per_period_count = par_grids
    kappa = par_kappa[0]
    beta = init.generate_beta(*par_beta)
    path = wp.write_output_path(
        is_conservative,
        features,
        par_beta,
        par_kappa,
        par_ninit,
        par_grids,
        normalization_wtr_feature=normalization_wtr_feature,
        tmax=tmax,
    )
    if os.path.exists(path):  # If the path exists we load it directly.
        print("Simulation loaded from: ", path, "\n")
        output = np.load(path, allow_pickle=True).item()
        return output  # list(output.values())

    # Definition of parameters and variables.
    i = 2 - is_conservative  # Factor in the source term of the equation.
    # > Features.
    v_count = len(features)
    test_kappa_conform(kappa, v_count)  # Verification of arguments format.
    v_max = np.max(features)
    features_col = np.transpose([features])
    v_ratios = features_col / v_max
    # > Size grid.
    sizes, dxs, delta_x = init.geometrical_grid(x_count, k)
    x_count = len(sizes)
    test_sizes_conform(sizes, x_test)  # Verification of arguments format.
    x_test_idx = find_nearest(sizes, x_test)[0]
    # > Time grid.
    dt = delta_x / (v_max * (delta_x + 1))  # st. CFL=1 for the fastest subpop.
    t_max = tmax or np.log(2) * period_count  # Maximum time of computation.
    # > Saving: distribution saved every 't_test_idx_step' times.
    t_test_idx_step = max(1, int((np.log(2) / dt) / tsaved_per_period_count))
    # > Division rate per unit of time (extended by `k` zeros at the right).
    gamma = np.zeros((v_count, x_count + k))
    gamma[:, :-k] = np.dot(features_col, [sizes * beta(sizes)])
    # > Initial condition (normalized).
    n_init = init.generate_initial_condition(*par_ninit, dxs_to_normalize=dxs)(
        sizes, features
    )

    # Initialization.
    times = [0]
    t_test = [0]
    # NB: current density extended by `k` zeros at the right one zero at the
    #    left (the boundary condition x=0) meant to be conserved for all times.
    n_temp = np.concatenate((n_init, np.zeros((v_count, k))), axis=1)
    n_temp = np.concatenate((np.zeros((v_count, 1)), n_temp), axis=1)
    n_evo = [n_init]
    n_evo_normalized = [n_init]
    n_evo_norm_wrt_v = None
    if normalization_wtr_feature:
        n_evo_norm_wrt_v = [normalize_wrt_0_axis(n_init, dxs)]
    n_test = [n_init[:, x_test_idx]]
    n_sum_evo = [1]  # Initial distribution normalized.
    n_tdiff = 1
    n_tdiff_evo = []
    lambda_n = []
    if is_conservative:
        lambda_error = []
    else:
        lambda_gamma = []
        lambda_tau = []

    # Iteration on times.
    time_count = 0
    while times[-1] <= t_max and n_tdiff >= 1e-5:
        # 1st step of the splitting: transport and loss of dividing mothers.
        n_1 = (
            (1 - v_ratios) * n_temp[:, 1:] + v_ratios * n_temp[:, :-1] / (delta_x + 1)
        ) / (
            1 + dt * gamma
        )  # Shape (v_count, x_count + k).
        # 2nd step of the splitting: new born daughters (v_count, x_count).
        n_2 = n_1[:, :-k] + 2 * i * dt * np.dot(
            np.transpose(kappa), gamma[:, k:] * n_1[:, k:]
        )
        # Udpdates and savings.
        n_sum_temp = np.sum(n_2 * dxs)
        n_sum_evo.append(n_sum_temp)
        n_tdiff = np.sum(abs(n_2 / n_sum_temp - n_temp[:, 1:-k] / n_sum_evo[-2]) / dt)
        n_tdiff_evo.append(n_tdiff)
        n_temp[:, 1:-k] = n_2  # Next computation will use the current
        #  distrib `n_temp` NOT normalized & that remains 0 outside of `sizes`.
        n_2 = n_2 / n_sum_temp  # But we save the renormalized distribution.
        n_test.append(n_2[:, x_test_idx])
        time_count += 1
        times.append(times[-1] + dt)
        if (time_count % t_test_idx_step) == 0:  # Multiple of t_test_idx_step.
            if isinstance(tmax, type(None)):
                print(
                    "Time progression: ",
                    times[-1],
                    time_count // t_test_idx_step / tsaved_per_period_count,
                    "/",
                    period_count,
                )
            else:
                print("Time progression: ", times[-1], "/", tmax)
            print("n_diff: ", n_tdiff, "\n")
            t_test.append(times[-1])
            n_evo.append(n_2)
            n_evo_normalized.append(n_temp[:, 1:-k])  # Normalized later.
            if normalization_wtr_feature:
                n_evo_norm_wrt_v.append(normalize_wrt_0_axis(n_temp[:, 1:-k], dxs))

        # Estimates of lambda (starting from time '2dt').
        if time_count > 1:
            time_count_half = int(time_count / 2)
            lambda_n.append(
                np.polyfit(
                    times[time_count_half:], np.log(n_sum_evo)[time_count_half:], 1
                )[0],
            )
            if is_conservative:
                diff_growth_frag = np.dot(features_col, [sizes]) - gamma[:, :-k]
                lambda_error.append(np.sum(diff_growth_frag * n_2 * dxs))
            else:
                lambda_gamma.append(np.sum(gamma[:, :x_count] * n_2 * dxs))
                xn_int = np.dot(sizes * n_2, dxs)
                lambda_tau.append(np.sum(features * xn_int) / np.sum(xn_int))

    # All estimates of lambda are gathered in 'lambda_estimates'.
    if is_conservative:
        lambda_estimates = [np.array(lambda_n), np.array(lambda_error)]
    else:
        lambda_estimates = [
            np.array(lambda_n),
            np.array(lambda_gamma),
            np.array(lambda_tau),
        ]

    n_evo_normalized = np.array(n_evo_normalized) * np.reshape(
        np.exp(-t_test * lambda_n[-1]), (len(t_test), 1, 1)
    )

    output = {
        "times": np.array(times),
        "t_test": np.array(t_test),
        "sizes": sizes,
        "n_test": np.array(n_test),
        "n_evo": np.array(n_evo),
        "n_evo_normalized": n_evo_normalized,
        "n_sum_evo": np.array(n_sum_evo),
        "n_tdiff_evo": np.array(n_tdiff_evo),
        "lambda_estimates": lambda_estimates,
        "n_evo_norm_wrt_v": np.array(n_evo_norm_wrt_v),
    }
    if is_saved:
        print("Simulation saved at: ", path, "\n")
        np.save(path, output)
    return output


def compute_longtime_approximation(
    is_conservative,
    features,
    par_beta,
    par_kappa,
    par_ninit,
    par_grids,
    is_saved=False,
    is_printed=True,
):
    """Compute an approximation of the solution to the direct eigenproblem
    associated to the growth-fragmentation equation, in its conservative or
    non-conservative form, in the case of equal mitosis and variability in
    (linear) growth rate, with coefficients 'tau(v,x) = vx' and
    'gamma(v,x) = beta(x) tau(v,x)'.

    Same function (and scheme) as `compute_evo_approximation` except that
    intermediate times of computation are not saved. `is_printed` can be set to
    False to mute any message printed during execution of the function.

    Returns
    -------
    A dictionary with the following keys and corresponding entries: 'times',
    'sizes', 'n_sum_evo', 'lambda_estimates' identical to
    `compute_evo_approximation` output, and:
    n_last : ndarray
        The density (v_count, x_count) at the last time of computation give by
        `period_count * ln(2)` or the first time for which `n_tdiff < 1e-5`.

    """
    # ........... Identical to `compute_evo_approximation` ..............
    k, x_count, period_count = par_grids
    kappa = par_kappa[0]
    beta = init.generate_beta(*par_beta)
    path = wp.write_output_path(
        is_conservative,
        features,
        par_beta,
        par_kappa,
        par_ninit,
        par_grids,
        is_longtime=True,
    )
    if os.path.exists(path):  # If the path exists we load it directly.
        if is_printed:
            print("loaded: ", path)
        output = np.load(path, allow_pickle=True).item()
        return output
    i = 2 - is_conservative
    # Feature-dependent variables.
    v_count = len(features)
    test_kappa_conform(kappa, v_count)  # Verification of arguments format.
    v_max = np.max(features)
    features_col = np.transpose([features])
    v_ratios = features_col / v_max
    # Size-grid variables.
    sizes, dxs, delta_x = init.geometrical_grid(x_count, k)
    x_count = len(sizes)
    # Time-grid variables.
    dt = delta_x / (v_max * (delta_x + 1))  # s.t. CFL=1 for the fastest subpop
    t_max = np.log(2) * period_count  # Maximum time of the computation.
    # Division rate per unit of time (extended by `k` zeros at the right).
    gamma = np.zeros((v_count, x_count + k))
    gamma[:, :-k] = np.dot(features_col, [sizes * beta(sizes)])
    # Initial condition.
    n_init = init.generate_initial_condition(*par_ninit, dxs_to_normalize=dxs)(
        sizes, features
    )
    # .........................................................................

    # Initialization.
    times = [0]
    # NB: current density extended by `k` zeros at the right one zero at the
    #    left (the boundary condition x=0) meant to be conserved for all times.
    n_temp = np.concatenate((n_init, np.zeros((v_count, k))), axis=1)
    n_temp = np.concatenate((np.zeros((v_count, 1)), n_temp), axis=1)
    n_sum_evo = [1]  # Initial distribution normalized.
    n_tdiff = 1
    lambda_n = []
    if is_conservative:
        lambda_error = []
    else:
        lambda_gamma = []
        lambda_tau = []

    # Iteration on times.
    time_count = 0
    while times[-1] <= t_max and n_tdiff >= 1e-5:
        # 1st step of the splitting: transport and loss of dividing mothers.
        n_1 = (
            (1 - v_ratios) * n_temp[:, 1:] + v_ratios * n_temp[:, :-1] / (delta_x + 1)
        ) / (
            1 + dt * gamma
        )  # Shape (v_count, x_count + k).
        # 2nd step of the splitting: new born daughters (v_count, x_count).
        n_2 = n_1[:, :-k] + 2 * i * dt * np.dot(
            np.transpose(kappa), gamma[:, k:] * n_1[:, k:]
        )
        # Updates and savings.
        n_sum_temp = np.sum(n_2 * dxs)
        n_sum_evo.append(n_sum_temp)
        n_tdiff = np.sum(abs(n_2 / n_sum_temp - n_temp[:, 1:-k] / n_sum_evo[-2]) / dt)
        n_temp[:, 1:-k] = n_2  # Remains zero outside of `sizes`.
        n_2 = n_2 / n_sum_temp
        time_count += 1
        times.append(times[-1] + dt)
        # Estimates of lambda (starting from time '2dt').
        if time_count > t_max / 2:
            time_count_half = int(time_count / 2)
            lambda_n.append(
                np.polyfit(
                    times[time_count_half:], np.log(n_sum_evo)[time_count_half:], 1
                )[0],
            )
            if is_conservative:
                diff_growth_frag = np.dot(features_col, [sizes]) - gamma[:, :-k]
                lambda_error.append(np.sum(diff_growth_frag * n_2 * dxs))
            else:
                lambda_gamma.append(np.sum(gamma[:, :x_count] * n_2 * dxs))
                xn_int = np.dot(sizes * n_2, dxs)
                lambda_tau.append(np.sum(features * xn_int) / np.sum(xn_int))

    # All estimates of lambda are gathered in 'lambda_estimates'.
    if is_conservative:
        lambda_estimates = [np.array(lambda_n), np.array(lambda_error)]
    else:
        lambda_estimates = [
            np.array(lambda_n),
            np.array(lambda_gamma),
            np.array(lambda_tau),
        ]

    output = {
        "times": np.array(times),
        "sizes": sizes,
        "n_last": n_temp[:, 1:x_count],
        "n_sum_evo": np.array(n_sum_evo),
        "lambda_estimates": lambda_estimates,
    }
    if path != "" and is_saved:
        print("Simulation saved at: ", path, "\n")
        np.save(path, output)
    return output


#  Constant growth rates.
# -----------------------


def compute_evo_approximation_constant(
    is_conservative,
    features,
    par_beta,
    par_kappa,
    par_ninit,
    par_grids,
    x_test,
    is_saved=False,
):
    """Computes an approximation of the solution to the growth-fragmentation
    equation, in its conservative or non-conservative form, in the case of
    equal mitosis and variability in (constant) growth rate, with coefficients
    'tau(v,x) = v' and 'gamma(v,x) = beta(x) tau(v,x)'.

    See `compute_evo_approximation_constant` docstring except for:

    Parameters
    ----------
    par_grids : list
        dx : float
            Step for the x-axis (regular grid taken).
        x_count : int
            Number of points of the x-grid.
        period_count : int
            Number of periods ln(2) before ending computations.
        tsaved_per_period_count : int
            Number of time steps per period ln(2) at which data is stored.
    x_test : int
        The size at which the value of the eigenvector is kept in memory (at
        all times and all features).

    """
    dx, x_count, period_count, tsaved_per_period_count = par_grids
    kappa = par_kappa[0]
    beta = init.generate_beta(*par_beta)
    path = wp.write_output_path(
        is_conservative,
        features,
        par_beta,
        par_kappa,
        par_ninit,
        par_grids,
        is_tau_constant=True,
    )
    if os.path.exists(path):  # If the path exists we load it directly.
        output = np.load(path, allow_pickle=True).item()
        return output

    # Definition of parameters and variables.
    i = 2 - is_conservative  # Factor in the source term of the equation.

    # > Features.
    v_count = len(features)
    test_kappa_conform(kappa, v_count)  # Verification of arguments format.
    v_max = np.max(features)
    features_col = np.transpose([features])
    v_ratios = features_col / v_max
    # > Size grid.
    sizes, dxs = init.regular_grid(*par_grids[:2])
    test_sizes_conform(sizes, x_test)  # Verification of arguments format.
    x_test_idx = np.argmin(sizes <= x_test) - 1
    # > Time grid.
    dt = dx / v_max  # s.t. CFL=1 for the fastest subpop.
    t_max = np.log(2) * period_count  # Maximum time of computation.
    # > Saving: distribution saved every 't_test_idx_step' times.
    t_test_idx_step = max(1, int((np.log(2) / dt) / tsaved_per_period_count))
    # > Division rate per unit of time (extended by `x_count-1` zeros at right)
    gamma = np.zeros((v_count, 2 * x_count - 1))
    gamma[:, :x_count] = np.dot(features_col, [beta(sizes)])
    # > Initial condition (normalized).
    n_init = init.generate_initial_condition(*par_ninit, dxs_to_normalize=dxs)(
        sizes, features
    )

    # Initialization.
    times = [0]
    t_test = [0]
    # NB: `n_temp` extended by zeros as `gamma`.
    n_temp = np.concatenate((n_init, np.zeros((v_count, x_count - 1))), 1)
    n_evo = [n_init]
    n_evo_normalized = [n_init]
    n_test = [n_init[:, x_test_idx]]
    n_sum_evo = [1]  # Initial distribution normalized.
    n_tdiff_evo = []
    lambda_n = []
    if is_conservative:
        lambda_error = []
    else:
        lambda_gamma = []
        lambda_tau = []

    # Iteration on times.
    time_count = 0
    while times[-1] <= t_max:
        # 1st step of the splitting: transport and loss of dividing mothers.
        n_1 = ((1 - v_ratios) * n_temp[:, 1:] + v_ratios * n_temp[:, :-1]) / (
            1 + dt * gamma[:, 1:]
        )  # Shape (v_count, 2 * x_count - 2).
        # 2nd step of the splitting: new born daughters (v_count, x_count - 1).
        mother_idxs = 2 * np.arange(1, x_count - 1)
        n_2 = n_1[:, : x_count - 1] + 2 * i * dt * np.dot(
            np.transpose(kappa), gamma[:, mother_idxs] * n_1[:, mother_idxs - 1]
        )
        # Null boundary condition (v_count, x_count).
        n_2 = np.concatenate((np.zeros((v_count, 1)), n_2), axis=1)

        # Updates and savings.
        n_sum_temp = np.sum(n_2 * dxs)
        n_sum_evo.append(n_sum_temp)
        n_tdiff_evo.append(
            np.sum(abs(n_2 / n_sum_temp - n_temp[:, :x_count] / n_sum_evo[-2]) / dt)
        )
        n_temp[:, :x_count] = n_2  # Next computation will use the
        # current distribution 'n_temp' NOT renormalized.
        # But we save the renormalized distribution.
        n_2 = n_2 / n_sum_temp
        n_test.append(n_2[:, x_test_idx])
        time_count += 1
        times.append(times[-1] + dt)
        if (time_count % t_test_idx_step) == 0:  # Multiple of t_test_idx_step.
            print(
                times[-1],
                time_count // t_test_idx_step / tsaved_per_period_count,
                "/",
                period_count,
            )
            t_test.append(times[-1])
            n_evo.append(n_2)
            n_evo_normalized.append(n_temp[:, :x_count])

        # Estimates of lambda (starting from time '2dt').
        if time_count > 1:
            time_count_half = int(time_count / 2)
            lambda_n.append(
                np.polyfit(
                    times[time_count_half:], np.log(n_sum_evo)[time_count_half:], 1
                )[0]
            )
            if is_conservative:
                diff_growth_frag = np.dot(features_col, [sizes]) - gamma[:, :x_count]
                lambda_error.append(np.sum(diff_growth_frag * n_2 * dxs))
            else:
                lambda_gamma.append(np.sum(gamma[:, :x_count] * n_2 * dxs))
                xn_int = np.dot(sizes * n_2, dxs)
                lambda_tau.append(np.sum(features * xn_int) / np.sum(xn_int))

    # All estimates of lambda are gathered in 'lambda_estimates'.
    if is_conservative:
        lambda_estimates = [np.array(lambda_n), np.array(lambda_error)]
    else:
        lambda_estimates = [
            np.array(lambda_n),
            np.array(lambda_gamma),
            np.array(lambda_tau),
        ]

    n_evo_normalized = np.array(n_evo_normalized) * np.reshape(
        np.exp(-t_test * lambda_n[-1]), (len(t_test), 1, 1)
    )

    output = {
        "times": np.array(times),
        "t_test": np.array(t_test),
        "sizes": sizes,
        "n_test": np.array(n_test),
        "n_evo": np.array(n_evo),
        "n_evo_normalized": n_evo_normalized,
        "n_sum_evo": np.array(n_sum_evo),
        "n_tdiff_evo": np.array(n_tdiff_evo),
        "lambda_estimates": lambda_estimates,
    }
    if is_saved:
        print("Simulation saved at: ", path, "\n")
        np.save(path, output)
    return output


def compute_longtime_approximation_constant(
    is_conservative,
    features,
    par_beta,
    par_kappa,
    par_ninit,
    par_grids,
    is_saved=False,
    is_printed=True,
):
    """Computes an approximation of the solution to the direct eigenproblem
    associated to the growth-fragmentation equation, in its conservative or
    non-conservative form, in the case of equal mitosis and variability in
    (constant) growth rate, with coefficients'tau(v,x) = v' and
    'gamma(v,x) = beta(x) tau(v,x)'.

    Same function (and scheme) as `compute_evo_approximation_constant` except
    that intermediate times of computation are not saved. `is_printed` can be
    set to False to mute any message printed during execution of the function.

    Returns
    -------
    A dictionary with the following keys and corresponding entries: 'times',
    'sizes', 'n_sum_evo', 'lambda_estimates' identical to
    `compute_evo_approximation_constant` output, and:
    n_last : ndarray
        The density (v_count, x_count) at the last time of computation give by
        `period_count * ln(2)` or the first time for which `n_tdiff < 1e-5`.

    """
    # .... See `compute_evo_approximation_constant` for mor comments ..........
    dx, x_count, period_count = par_grids
    kappa = par_kappa[0]
    beta = init.generate_beta(*par_beta)
    path = wp.write_output_path(
        is_conservative,
        features,
        par_beta,
        par_kappa,
        par_ninit,
        par_grids,
        is_tau_constant=True,
        is_longtime=True,
    )
    if os.path.exists(path):  # If the path exists we load it directly.
        if is_printed:
            print("loaded: ", path)
        output = np.load(path, allow_pickle=True).item()
        return output
    # Definition of parameters and variables.
    i = 2 - is_conservative
    v_count = len(features)
    test_kappa_conform(kappa, v_count)  # Verification of arguments format.
    v_max = np.max(features)
    features_col = np.transpose([features])
    v_ratios = features_col / v_max
    sizes, dxs = init.regular_grid(*par_grids[:2])
    dt = dx / v_max  # s.t. CFL=1 for the fastest subpop.
    t_max = np.log(2) * period_count  # Maximum time of the computation.
    gamma = np.zeros((v_count, 2 * x_count - 1))
    gamma[:, :x_count] = np.dot(features_col, [beta(sizes)])
    n_init = init.generate_initial_condition(*par_ninit, dxs_to_normalize=dxs)(
        sizes, features
    )
    # .........................................................................

    # Initialization.
    times = [0]
    n_temp = np.concatenate((n_init, np.zeros((v_count, x_count - 1))), 1)
    n_sum_evo = [1]
    n_tdiff = 1
    lambda_n = []
    if is_conservative:
        lambda_error = []
    else:
        lambda_gamma = []
        lambda_tau = []

    # Iteration on times.
    time_count = 0
    while times[-1] <= t_max and n_tdiff >= 1e-5:
        # .....................................................................
        n_1 = ((1 - v_ratios) * n_temp[:, 1:] + v_ratios * n_temp[:, :-1]) / (
            1 + dt * gamma[:, 1:]
        )
        mother_idxs = 2 * np.arange(1, x_count)
        n_2 = n_1[:, : x_count - 1] + 2 * i * dt * np.dot(
            np.transpose(kappa), gamma[:, mother_idxs] * n_1[:, mother_idxs - 1]
        )
        n_2 = np.concatenate((np.zeros((v_count, 1)), n_2), axis=1)
        # .....................................................................
        # Updates and savings.
        n_sum_temp = np.sum(n_2 * dxs)
        n_sum_evo.append(n_sum_temp)
        n_tdiff = np.sum(
            abs(n_2 / n_sum_temp - n_temp[:, :x_count] / n_sum_evo[-2]) / dt
        )
        n_temp[:, :x_count] = n_2
        time_count += 1
        times.append(times[-1] + dt)
        # Estimates of lambda (starting from time '2dt').
        n_2 = n_2[:, :x_count] / n_sum_temp
        if time_count > 1:
            time_count_half = int(time_count / 2)
            lambda_n.append(
                np.polyfit(
                    times[time_count_half:], np.log(n_sum_evo)[time_count_half:], 1
                )[0]
            )
            if is_conservative:
                diff_growth_frag = np.dot(features_col, [sizes]) - gamma[:, :x_count]
                lambda_error.append(np.sum(diff_growth_frag * n_2 * dxs))
            else:
                lambda_gamma.append(np.sum(gamma[:, :x_count] * n_2 * dxs))
                xn_int = np.dot(sizes * n_2, dxs)
                lambda_tau.append(np.sum(features * xn_int) / np.sum(xn_int))

    # All estimates of lambda are gathered in 'lambda_estimates'.
    if is_conservative:
        lambda_estimates = [np.array(lambda_n), np.array(lambda_error)]
    else:
        lambda_estimates = [
            np.array(lambda_n),
            np.array(lambda_gamma),
            np.array(lambda_tau),
        ]

    output = {
        "times": np.array(times),
        "sizes": sizes,
        "n_last": n_temp[:, :x_count],
        "n_sum_evo": np.array(n_sum_evo),
        "lambda_estimates": lambda_estimates,
    }
    if path != "" and is_saved:
        print("Simulation saved at: ", path, "\n")
        np.save(path, output)
    return output


# def compute_longtime_approximation_constant_2D_mix_vs_irr(
#         is_conservative, features, par_beta, kappa_irr, par_ninit, par_grids,
#         is_normalized_by_v=False):
#     """Computes the direct eigenvector associated to the growth-fragmentation
#     equation, in its conservative or non-conserva. form, in the case of equal
#     mitosis and 2 distinct growth rates, with coefficients ` tau(v,x) = v`
#     and `gamma(v,x) = beta(x) tau(v,x)`.

#     Return also
#     Parameters
#     ----------
#     is_conservative : bool
#         True for conservative form of the equation (lineage evolution). False
#         for population evolution (non-conservative)
#     features : ndarray
#         1D array of the  features associated with the populations.
#     par_beta : float
#         Parameter representing the growth rate or related dynamics.
#     kappa_irr : ndarray
#         (2, 2) (irreducible) matrix kappa: the heredity kernel.
#     par_ninit : list
#         List off parameters defining the initial condition.
#         See parameters.init_functions.generate_initial_condition.
#     par_grids : list
#         dx : float
#             Step for the x-axis (regular grid taken).
#         x_count : int
#             Number of points of the x-grid.
#         period_count : int
#             Number of periods ln(2) before ending computations.
#         tsaved_per_period_count : int
#             Number of time steps per period ln(2) at which data is stored.
#     is_normalized_by_v : bool, optional
#         If True, normalizes the "irr" distribution s.t.^. The default is False.

#     Returns
#     -------
#     out : dict
#         A dictionary containing:
#         - 'sizes': array of population sizes.
#         - 'irr': Normalized population distribution with irreducible mixing.
#         - 'red': Reduced population distribution (no mixing, independent traits).
#         - 'i1': Independent approximation for the first trait.
#         - 'i2': Independent approximation for the second trait.

#     Notes
#     -----
#     - The 'irr' scenario considers full mixing with coupling defined by `kappa_irr`.
#     - The 'red' scenario represents no mixing, modeled as independent approximations for each trait.
#     - Additional normalizations are applied to ensure proper comparison across scenarios.

#     """

#     # Full mixing.
#     par_kappa = {'irr': [kappa_irr, None]}

#     # No mixing computed as a "system" with kappa=Id (no coupling), rescaling
#     # each trait, rather than as 2 equations of homogeneous population.
#     par_kappa['red'] = ker.kappa_identity(2)

#     out = {}
#     for key, pkappa in par_kappa.items():
#         out_tmp = scheme.compute_longtime_approximation_constant(
#             is_conservative, features, par_beta, pkappa, par_ninit, par_grids,
#             is_printed=False)
#         if 'sizes' not in out:
#             out['sizes'] = out_tmp['sizes']
#         out[key] = out_tmp['n_last'] / np.sum(out_tmp['n_last'])
#     out['red'] = plot.normalize_by_v(out['red'])
#     if is_normalized_by_v:
#         out['irr'] = plot.normalize_by_v(out['irr'])

#     # Used to check that after rescaling, the 2 component of 'red' are the
#     # distributions obtained with 2 independent approximations.
#     out['i1'] = scheme.compute_longtime_approximation_constant(
#             is_conservative, [features[0]], par_beta, ker.kappa_identity(1),
#             par_ninit, par_grids, is_printed=False)['n_last']
#     out['i2'] = scheme.compute_longtime_approximation_constant(
#             is_conservative, [features[1]], par_beta, ker.kappa_identity(1),
#             par_ninit, par_grids, is_printed=False)['n_last']
#     out['i1'] = out['i1'] / np.sum(out['i1'])
#     out['i2'] = out['i2'] / np.sum(out['i2'])

#     return out
