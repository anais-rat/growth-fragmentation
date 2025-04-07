#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug 23 10:39:40 2021
From the script Direct_stat_osci_geom.m which follows the numerical
scheme presented in "Bernard, Doumic & Gabriel, Cyclic asymptotic... (2018)"

@author: anais rat
"""

import numpy as np
import src.scheme_direct as scheme
import src.parameters.init_functions as init
import src.plot as plot
import src.write_path as wp


def compute_evo_approximation_no_variability(is_conservative, beta,
                                             initial_condition, k, x_count,
                                             period_count,
                                             tsaved_per_period_count, x_test):
    """

    """
    # Size grid parameters.
    sizes, dxs, delta_x = scheme.geometrical_grid(x_count, k)
    x_count = len(sizes)
    r_x = (delta_x + 1) / delta_x
    if sizes[-1] > x_test:
        raise ValueError("The size `x_test` at which saved `n_evo` is outside"
                         "the x-grid.")
    x_test_idx = np.argmin(sizes <= x_test) - 1

    # Time grid parameters.
    dt_CFL = delta_x / (delta_x + 1) # s.t. CFL = 1 for the fastest subpop.
    t_max = np.log(2) * period_count # Maximum time of the computation.

    # Saving parameters: distribution saved every 't_test_idx_step' times.
    t_test_idx_step = max(1, int((np.log(2)/dt_CFL) / tsaved_per_period_count))

    # Division rate per unit of time.
    # NB: gamma & p_temp defined also at `k` sizes after `sizes[-1]`, set to 0.
    gamma = np.zeros(k + x_count)
    gamma[k:] = sizes * beta(sizes)
    print('gamma', np.shape(gamma))

    # Definition of the initial condition.
    p_init = initial_condition(sizes, [1], dxs_to_normalize=dxs)[0]

    # Initialization.
    times = np.array([0])
    t_test = np.array([0])
    print('p_init', np.shape(p_init))
    # p_temp = np.concatenate((p_init[:k] / 2, p_init))
    p_temp = np.concatenate((np.zeros(k), p_init))
    p_evo = np.array([p_init])
    p_test = np.array([p_init[x_test_idx]])
    p_sum_evo = np.array([1]) # Initial distribution normalized.
    p_tdiff_evo = np.array([])
    lambda_n = np.array([])
    if is_conservative:
        lambda_error = np.array([])
    else:
        lambda_gamma = np.array([])
        lambda_tau = np.array([])

    # Iteration on times.
    time_count = 0
    while times[-1] <= t_max:
        p_evo_norm = p_temp[k:] / p_sum_evo[-1]
        temp = p_temp[k-1:-1]  + dt_CFL*gamma[k:]*(p_temp[k:] - 2*p_temp[:-k])
        if np.min(temp) >= 0:
            dt = dt_CFL
            p_temp[k:] = temp
        else:
            dt_s = -p_temp[k:] / (r_x*(p_temp[k-1:-1] - p_temp[k:])
                                     + gamma[k:]*(p_temp[k:] - 2*p_temp[:-k]))
            dt = np.min(dt_s[dt_s > 0])
            print('dt: ', dt)
            p_temp[k:] = p_temp[k:] + dt*(r_x*(p_temp[k-1:-1] - p_temp[k:])
                                      + gamma[k:]*(p_temp[k:] - 2*p_temp[:-k]))
        # Boundary condition.
        # p_temp[:k] = p_temp[:k] + p_temp[:k] / 2

        # Udpdates and savings.
        p_sum_temp = np.sum(p_temp[k:] * dxs)
        p_sum_evo = np.append(p_sum_evo, p_sum_temp)

        # Distribution 'p_temp' NOT renormalized, we save the renormalized one.
        p_evo_norm_new = p_temp[k:] / p_sum_temp
        p_test = np.append(p_test, p_evo_norm_new[x_test_idx])

        time_count += 1
        times = np.append(times, times[-1] + dt)
        if (time_count % t_test_idx_step) == 0: # Multiple of 't_test_idx_step'
            t_test = np.append(t_test, times[-1])
            p_evo = np.append(p_evo, [p_evo_norm_new], axis=0)
        p_tdiff_evo = np.append(p_tdiff_evo,
                                np.sum(abs(p_evo_norm_new - p_evo_norm) / dt))

        # Estimates of lambda (starting from time '2dt').
        # if time_count > 1:
        #     time_count_half = int(time_count / 2)
            # lambda_n = np.append(lambda_n, np.polyfit(times[time_count_half:],
            #                       np.log(p_sum_evo)[time_count_half:], 1)[0])
            # if is_conservative:
            #     diff_growth_frag = np.dot(sizes, [features]) - gamma[:x_count]
            #     lambda_error = np.append(lambda_error, np.sum(
            #                              diff_growth_frag*p_2[:, k:] * dxs))
            # else:
            #     lambda_gamma = np.append(lambda_gamma, np.sum(
            #                              gamma[:, k:] * p_2[:, k:] * dxs))
            #     xp_int = np.dot(sizes * p_2[:, k:], dxs)
            #     lambda_tau = np.append(lambda_tau, np.sum(features * xp_int) /
            #                            np.sum(xp_int))

    # All estimates of lambda are gathered in 'lambda_estimates'.
    if is_conservative:
        lambda_estimates = [lambda_n, lambda_error]
    else:
        lambda_estimates = [lambda_n, lambda_gamma, lambda_tau]

    sizes = np.append([0], sizes)
    p_evo = np.append(np.zeros((len(p_evo), 1)), p_evo, axis=1)
    
    return (times, t_test, sizes, p_test, p_evo, p_sum_evo,
            p_tdiff_evo, lambda_estimates)


def compute_evo_approximation_1(is_conservative, features, beta, kappa,
                                initial_condition, k, x_count, period_count,
                                tsaved_per_period_count, x_test):
    """

    """
    # Verification of arguments format and definition of useful variables.
    v_count = len(features)
    if len(kappa) != v_count:
        raise ValueError("The dimensions of the set of features and the "
                         "variability kernel do not match.")
    v_max = np.max(features)
    features_col = np.transpose([features])
    v_ratios = features_col / v_max
    
    # Size grid parameters.
    sizes, dxs, delta_x = scheme.geometrical_grid(x_count, k)
    x_count = len(sizes)
    if sizes[-1] > x_test:
        raise ValueError("The size `x_test` at which saved `n_evo` is outside"
                         "the x-grid.")
    x_test_idx = np.argmin(sizes <= x_test) - 1
    r_x = (delta_x + 1) / (v_max * delta_x)

    # Time grid parameters.
    dt_CFL = delta_x / (delta_x + 1) # s.t. CFL = 1 for the fastest subpop.
    t_max = np.log(2) * period_count # Maximum time of the computation.

    # Saving parameters: distribution saved every 't_test_idx_step' times.
    t_test_idx_step = max(1, int((np.log(2)/dt_CFL) / tsaved_per_period_count))

    # Division rate per unit of time.
    # NB: gamma & p_temp defined also at `k` sizes after `sizes[-1]`, set to 0.
    gamma = np.zeros((v_count, k + x_count))
    gamma[:, k:] = np.dot(features_col, [sizes * beta(sizes)])
    print('gamma', np.shape(gamma))

    # Computation of N, the Perron eigenvector for renormalization.
    if len(features) > 1:
        n_eig = scheme.compute_evo_approximation(
            is_conservative, features, beta, kappa, initial_condition, k,
            x_count, period_count, tsaved_per_period_count, x_test)[4][-1]
    else:
        n_eig = np.ones((v_count, x_count))

    # Definition of the initial condition.
    p_init = initial_condition(sizes, features, dxs_to_normalize=n_eig * dxs)
    plot.plot_distribution_old(sizes, features, p_init)

    # Initialization.
    times = np.array([0])
    t_test = np.array([0])
    print('p_init', np.shape(p_init))
    p_temp = np.concatenate((np.zeros((v_count, k)), p_init), axis=1)
    # p_temp = np.concatenate((p_init[:, :k] / 2, p_init), axis=1)
    p_evo = np.array([p_init])
    p_test = np.array([p_init[:, x_test_idx]])
    p_tdiff_evo = np.array([])

    i = 2 - is_conservative
    # Iteration on times.
    time_count = 0
    while times[-1] <= t_max:
        temp = ((1-v_ratios)*p_temp[:, k:] + v_ratios*p_temp[:, k-1:-1]) \
            + dt_CFL*gamma[:, k:]*(p_temp[:, k:] - 2*p_temp[:, :-k])
        if np.min(temp) >= 0:
            dt = dt_CFL
            p_temp[:, k:] = temp
        else:
            dt_s = p_temp[:, k:] / (r_x*(p_temp[:, k-1:-1]-p_temp[:, k:])
             + gamma[:, k:]*(p_temp[:, k:] - i*np.dot(kappa, p_temp[:, :-k])))
            dt = np.nanmin(dt_s[dt_s > 0])
            print('dt: ', dt, dt_s, times[-1], t_max)
            p_temp[:, k:] = p_temp[:, k:] \
              + dt * (r_x * v_ratios * (p_temp[:, k-1:-1] - p_temp[:, k:]) \
              + gamma[:, k:]*(p_temp[:, k:] - i*np.dot(kappa, p_temp[:, :-k])))
        # Boundary condition.
        # p_temp[:, :k] = p_temp[:, :k] / 2

        # Distribution 'p_temp' renormalized.
        p_temp = p_temp / np.sum(p_temp[:, k:] * n_eig * dxs)

        # Udpdates and savings.
        p_test = np.append(p_test, [p_temp[:, x_test_idx]], axis=0)
        time_count += 1
        times = np.append(times, times[-1] + dt)
        if (time_count % t_test_idx_step) == 0: # Multiple of 't_test_idx_step'
            t_test = np.append(t_test, times[-1])
            p_evo = np.append(p_evo, [p_temp[:, k:]], axis=0)
        p_tdiff_evo = np.append(p_tdiff_evo,
                                np.sum(abs(p_temp[:, k:]-p_evo[-1]) / dt))

    sizes = np.append([0], sizes)
    p_evo = np.append(np.zeros((len(p_evo), v_count, 1)), p_evo, axis=2)

    return (times, t_test, sizes, p_test, p_evo, p_tdiff_evo)


def compute_evo_approximation(is_conservative, features, beta, kappa,
                              initial_condition, k, x_count, period_count,
                              tsaved_per_period_count, x_test):
    """ Computes an approximation of the solution to the dual growth-
    fragmentation equation, in its conservative or non-conservative form, in
    the case of equal mitosis and variability in (linear) growth rate, with
    coefficients 'tau(x,v) = vx' and 'gamma(x,v) = beta(x) tau(x,v)'.

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
    beta : function
        Division rate per unit of size, function of the size only
    kappa : ndarray
        2D array (v_count, v_count) variability kernel (stochastic matrix s.t.
        `kappa[i, j]` is the propbability for a cell of feature `features[i]`
        to give birth to a cell of feature `features[j]`).
    initial_condition : function
        Initial distribution, function of features and size.
    k : int
        Precision of the geometrical step `delta_x = 2^(1/k)-1` for the x-axis.
    x_count : int
        Odd integer, the number of points of the x-grid.
    period_count : int
        Number of periods ln(2) before endding computations.
    tsaved_per_period_count : int
        Number of time steps per period ln(2) at which data is kept in memory.
    x_test : int
        The size at which the value of the eigenvector is kept in memory (at
        all times and all features).

    Returns
    -------
    t : ndarray
        1D array (t_count,) of the total time-grid.
    t_test : ndarray
        1D array (t_test_count,) of times at which 'p_evo' is stored.
    x : ndarray
        1D array (x_count,) of the (geometrical) size-grid.
    p_test : ndarray
        2D array (t_count, v_count) of the density evaluated at the size
        `x_test` for all features and at all times.
    p_evo: ndarray
        3D array (t_test_count, v_count, x_count) of the time evolution of the
        feature-size distribution.
    p_sum_evo : ndarray
        1D array (t_count,) of the time evolution of the total number of
        individuals (iint n(t, v, x) dv dx).
    p_tdiff_evo : ndarray
        1D array (t_count,) of the difference of the normalized distribution
        between each time in l1 norm divided by dt.
    lambda_estimates : list
        List of time evolutions (starting from time `2dt`) of estimates of
        lambda: `lambda_estimates[i]` is a 1D array of length `t_count-2` s.t.
        - i=0: ('lambda_n') at each time 't[j]', j>0, the linear regression on
            '[int(t[j]/2), t[j]]' of the log of the total number of cell.
        In the case conservative:
            - i=1: ('lambda_error')
                   iint p_evo(t,v,x) [tau(v,x) - gamma(v,x)] dvdx.
        In the case non-conservative:
            - i=1: ('lambda_gamma') iint p_evo(t,v,x) gamma(v,x) dvdx.
            - i=2: ('lambda_tau') iint vx p_evo(t,v,x) dvdx / (iint x p_evo).

    """
    # Verification of arguments format and definition of useful variables.
    v_count = len(features)
    if len(kappa) != v_count:
        raise ValueError("The dimensions of the set of features and the "
                         "variability kernel do not match.")
    v_max = np.max(features)
    features_col = np.transpose([features])
    v_ratios = features / v_max

    # Size grid parameters.
    sizes, dxs, delta_x = scheme.geometrical_grid(x_count, k)
    x_count = len(sizes)
    if sizes[-1] > x_test:
        raise ValueError("The size `x_test` at which saved `n_evo` is outside"
                         "the x-grid.")
    x_test_idx = np.argmin(sizes <= x_test) - 1

    # Time grid parameters.
    dt = delta_x / v_max # s.t. CFL = 1 for the fastest subpop.
    t_max = np.log(2) * period_count # Maximum time of the computation.

    # Saving parameters: distribution saved every 't_test_idx_step' times.
    t_test_idx_step = max(1, int((np.log(2) / dt) / tsaved_per_period_count))

    # Division rate per unit of time.
    # NB: gamma & p_temp defined also at `k` sizes after `sizes[-1]`, set to 0.
    gamma = np.zeros((v_count, k + x_count))
    gamma[:, k:] = np.dot(features_col, [sizes * beta(sizes)])
    print('gamma', np.shape(gamma))

    # Definition of the initial condition.
    p_init = initial_condition(sizes, features, dxs_to_normalize=dxs)

    # Initialization.
    times = np.array([0])
    t_test = np.array([0])
    print('p_init', np.shape(p_init))
    p_temp = np.concatenate((np.zeros((v_count, k)), p_init), axis=1)
    p_evo = np.array([p_init])
    p_test = np.array([p_init[:, x_test_idx]])
    p_sum_evo = np.array([1]) # Initial distribution normalized.
    p_tdiff_evo = np.array([])
    lambda_n = np.array([])
    if is_conservative:
        lambda_error = np.array([])
    else:
        lambda_gamma = np.array([])
        lambda_tau = np.array([])

    i = 1 + ~is_conservative

    # Iteration on times.
    time_count = 0
    while times[-1] <= t_max:
        # 1st step of the splitting: transport and new born daughters.
        p_1 = np.zeros((v_count, k + x_count))
        for x_idx in range(k, x_count + k):
            p_1[:, x_idx] = ((1-v_ratios) * p_temp[:, x_idx]
                         + v_ratios * p_temp[:, x_idx-1]) \
                         - i*dt*gamma[:, x_idx]* np.dot(kappa, p_1[:, x_idx-k])
                    
        # 2nd step of the splitting: loss of dividing mothers.
        p_2 = (1 + dt * gamma) * p_1
        # Boundary conditions.
        # p_boundary = p_temp[:, -1] # np.dot(dxs[1:k+1], np.dot(gamma[1:k+1]*p_2[:k], kappa))
        # p_2 = np.concatenate((p_2, p_boundary), axis=1)

        # Udpdates and savings.
        p_sum_temp = np.sum(p_2[:, k:] * dxs)
        p_sum_evo = np.append(p_sum_evo, p_sum_temp)
        p_tdiff_evo = np.append(p_tdiff_evo, np.sum(abs(p_2/p_sum_temp -
                                         p_temp/p_sum_evo[-2]) / dt))
        p_temp = np.copy(p_2) # Next computation will use the current
            # distribution 'p_temp' NOT renormalized.
        p_2 = p_2 / p_sum_temp # But we save the renormalized distribution.
        p_test = np.append(p_test, [p_2[:, x_test_idx + k]], axis=0)
        time_count += 1
        times = np.append(times, times[-1] + dt)
        if (time_count % t_test_idx_step) == 0: # Multiple of 't_test_idx_step'
            t_test = np.append(t_test, times[-1])
            p_evo = np.append(p_evo, [p_2[:, k:]], axis=0)

        # Estimates of lambda (starting from time '2dt').
        # if time_count > 1:
        #     time_count_half = int(time_count / 2)
            # lambda_n = np.append(lambda_n, np.polyfit(times[time_count_half:],
            #                       np.log(p_sum_evo)[time_count_half:], 1)[0])
            # if is_conservative:
            #     diff_growth_frag = np.dot(sizes, [features]) - gamma[:x_count]
            #     lambda_error = np.append(lambda_error, np.sum(
            #                              diff_growth_frag*p_2[:, k:]*dxs))
            # else:
            #     lambda_gamma = np.append(lambda_gamma, np.sum(
            #                              gamma[:, k:] * p_2[:, k:] * dxs))
            #     xp_int = np.dot(sizes * p_2[:, k:], dxs)
            #     lambda_tau = np.append(lambda_tau, np.sum(features * xp_int) /
            #                            np.sum(xp_int))

    # All estimates of lambda are gathered in 'lambda_estimates'.
    if is_conservative:
        lambda_estimates = [lambda_n, lambda_error]
    else:
        lambda_estimates = [lambda_n, lambda_gamma, lambda_tau]

    return (times, t_test, sizes, p_test, p_evo, p_sum_evo,
            p_tdiff_evo, lambda_estimates)


def compute_approximation_from_dirac(is_conservative, features, alpha, kappa,
                                     sizes_phi, k, x_count, period_count,
                                     kappa_name=None, is_printed=False):
    # Verification of arguments format and definition of useful variables.
    v_count = len(features)
    if len(kappa) != v_count:
        raise ValueError("The dimensions of the set of features and the "
                         "variability kernel do not match.")
    # Size grid parameters.
    sizes, dxs, delta_x = init.geometrical_grid(x_count, k)
    x_phi_count = len(sizes_phi)

    # def beta(sizes): # Division rate per unit of size.
    #     return sizes ** alpha
    n_phi = np.zeros((v_count, x_phi_count))
    n_phi_normalized_pointwise = np.zeros((v_count, x_phi_count))
    exp_factor = np.array([])
    for v_idx in range(v_count):
        for x_idx in range(x_phi_count):
            init_par = [v_idx, np.where(sizes == sizes_phi[x_idx])[0][0]]
            def n_init(sizes_, features_, dxs_to_normalize=None):
                return scheme.initial_condition(sizes_, features_,
                                    c_init_par=init_par,
                                    c_init_choice='dirac',
                                    dxs_to_normalize=dxs)
            p = wp.write_output_path(is_conservative, features, alpha, kappa,
                                     [*init_par, 'dirac'], k, x_count,
                                     period_count, is_longtime=True,
                                     kappa_name=kappa_name)
            out = scheme.compute_longtime_approximation(is_conservative,
                                                        features, beta, kappa,
                                                        n_init, k, x_count,
                                                        period_count, path=p,
                                                        is_printed=is_printed)
            n_phi[v_idx, x_idx] = np.sum(out['n_last'] * dxs)
            exp_tmp = np.exp(- out['lambda_estimates'][0][-1]*out['times'][-1])
            n_phi_normalized_pointwise[v_idx, x_idx] = \
                np.sum(out['n_last'] * dxs) * exp_tmp
            if is_printed:
                print('(v_idx, x_idx) =', init_par)
                print('n_phi[v_idx, x_idx]: ', n_phi[v_idx, x_idx])
            exp_factor = np.append(exp_factor, exp_tmp)
            # lambdas = np.append(lambdas, out['lambda_estimates'][0][-1])
    return sizes, n_phi, n_phi_normalized_pointwise, exp_factor


def compute_longtime_approximation_from_dirac_constant(
        is_conservative, features, par_beta, par_kappa, sizes_phi, par_grids,
        is_printed=False):
    """
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
            st. `kappa[i, j]` is the propba for a cell of feature `features[i]`
            to give birth to a cell of feature `features[j]`).
        kappa_name : string or None
            If not None and simulation is saved, saving path uses the short
            name `kappa_name` rather than the composant of `kappa`.
    sizes_phi : ndarray
        1D array of the size at which compute phi.
    par_grids : list
        dx : float
            Step for the x-axis (regular grid taken).
        x_count : int
            Number of points of the x-grid.
        period_count : int
            Number of periods ln(2) before endding computations.
    x_test : int
        The size at which the value of the eigenvector is kept in memory
        (at all times and all features).

    Returns
    -------

    """
    # Verification of arguments format and definition of useful variables.
    v_count = len(features)
    if len(par_kappa[0]) != v_count:
        raise ValueError("The dimensions of the set of features and the "
                         "variability kernel do not match.")
    # Size grid parameters.
    print(par_grids[:2])
    sizes, dxs = init.regular_grid(*par_grids[:2])
    x_phi_count = len(sizes_phi)

    n_phi = np.zeros((v_count, x_phi_count))
    n_phi_normalized_pointwise = np.zeros((v_count, x_phi_count))
    exp_factor = np.array([])
    # count_avg = 0
    for v_idx in range(v_count):
        for x_idx in range(x_phi_count):
            par_init = [v_idx, np.where(sizes == sizes_phi[x_idx])[0][0]]
            # > Initial condition (normalized).
            par_ninit = [par_init, 'dirac', dxs]
            out = scheme.compute_longtime_approximation_constant(
                    is_conservative, features, par_beta, par_kappa, par_ninit,
                    par_grids, is_saved=True, is_printed=is_printed)
            n_phi[v_idx, x_idx] = np.sum(out['n_last'] * dxs)
            exp_tmp = np.exp(- out['lambda_estimates'][0][-1]*out['times'][-1])
            n_phi_normalized_pointwise[v_idx, x_idx] = \
                np.sum(out['n_last'] * dxs) * exp_tmp
            if is_printed:
                print('(v_idx, x_idx) =', par_init)
                print('n_phi[v_idx, x_idx]: ', n_phi[v_idx, x_idx])
            exp_factor = np.append(exp_factor, exp_tmp)
            # if count_avg == 0:
            #     count_avg = plot.count_to_average(len(out['times']))
            # lambdas = np.append(lambdas,
            #               np.mean(out['lambda_estimates'][0][-count_avg:]))
            # print('WARNING: you might need to redefine the estimate of lambda.')
    return sizes, n_phi, n_phi_normalized_pointwise, exp_factor


def compute_adjoint_contant(kappa, features):
    """ Copied from `functions_effective_fitness`. Compute the value of the
    Malthus parameter (lambda_), the spectral gap and phi in the case of
    `feature_count` features (`features`), and constant coefficients
    (with alpha = 0).

    Parameters
    ----------
    kappa : ndarray
        2D array `(feature_count, feature_count)`.
    features : ndarray
        1D array (feature_count,).
    beta : float

    """
    feature_count = len(features)
    matrix = np.array(kappa - 0.5 * np.identity(feature_count), dtype=float)
    eigvalues, eigvectors = np.linalg.eig(2* matrix * np.transpose([features]))
    idx =  np.argmax(eigvalues)
    eigvalues = np.sort(eigvalues)
    lambda_ = eigvalues[-1]
    gap = eigvalues[-1] - eigvalues[-2]
    phi = eigvectors[:, idx]
    # Computation of \int N_i to normalize phi.
    eigvalues, eigvectors = np.linalg.eig(2 * np.transpose(matrix) * features)
    idx = np.argmax(eigvalues)
    N = eigvectors[:, idx]
    # Normalization phi.
    phi = phi / np.sum(phi * N)
    return lambda_, phi, gap


# DX = 0.005
# X_COUNT = 2001
# X_IDX_STEP = 4

# import parameters_adjoint as par
# data = open('data/phi_2.csv')
# phi = np.loadtxt(data, delimiter=",")


# # TAU LINEAR.
# sizes, phi, lambdas = compute_approximation_from_dirac(par.IS_CONSERVATIVE,
#                                                   par.FEATURES,
#                                                   par.beta,
#                                                   par.KAPPA, par.K,
#                                                   par.X_COUNT,
#                                                   par.PERIOD_COUNT,
#                                                   par.TSAVED_PER_PERIOD_COUNT)
# plot.plot_distribution_old(sizes, par.FEATURES, phi)
# print('mean / std :', np.mean(lambdas), np.std(lambdas))