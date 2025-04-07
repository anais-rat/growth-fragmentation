#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 10 16:38:02 2022

@author: arat

Note: Depending on the publication trait may be used equivalently to feature.
      Here we use trait.

"""

from copy import deepcopy
from matplotlib import cm
# import matplotlib
import matplotlib.pylab as plt
import matplotlib.ticker as ticker
import numpy as np
import os
import pandas as pd
import seaborn as sns
import statistics as st
import warnings

import src.parameters.kappas as ker
import src.plot as plot
import src.scheme_direct as scheme
import src.write_path as wp


# ------------------
# Default parameters
# ------------------

# Transition matrix (only used by `generate_kappa`)
# -------------------------------------------------

# kappa_choice == "load".
KAPPA_LOAD = {2: np.array([[0.6, 0.4],
                           [0.5, 0.5]]),
              4: np.array([[0.7, 0.1, 0.1, 0.1],
                           [0.1, 0.7, 0.1, 0.1],
                           [0.1, 0.1, 0.7, 0.1],
                           [0.1, 0.1, 0.1, 0.7]])}
# kappa_choice == "band".
SIGMA = 2


# Longtime approximation
# ----------------------

# Initial condition.
N_INIT_CHOICE = 'exponential'
N_INIT_1 = 30
N_INIT_2 = 60
PAR_N_INIT_LONGTIME = [[N_INIT_1, N_INIT_2], N_INIT_CHOICE]

# Grids.
# > Geometrical size grid.
K_F = 200
X_COUNT_F = 2001
PERIOD_COUNT = 5
PAR_GRIDS_LINEAR_LONGTIME = [K_F, X_COUNT_F, PERIOD_COUNT]

# > Regular size grid.
DX = 0.005
X_COUNT = 2001

# > Time grid.
PERIOD_COUNT = 10
PAR_GRIDS_CONSTANT_LONGTIME = [DX, X_COUNT, PERIOD_COUNT]


# Plot
# ----
FOLDER = wp.FOLDER_FIG

LABELS = {'arithmetic': r"$m_A$",  # r"$\mathrm{arithmetic~mean}$",
          'geometric': r"$m_G$",  # r"$\mathrm{geometric~mean}$",
          'harmonic': r"$m_H$",  # r"$\mathrm{harmonic~mean}$",
          'v_eff': r"$\mathrm{effective~fitness}$"}

LABELS_MEANS = f"{LABELS['arithmetic']}, {LABELS['geometric']}, " + \
                f"{LABELS['harmonic']}"

if __name__ == "__main__":  # Display current palette.
    sns.palplot(sns.color_palette())
    plt.show()


def write_fig_folder(subfolder):
    folder = os.path.join(FOLDER, subfolder)
    if (not os.path.exists(folder)):
        os.makedirs(folder)
    return folder


def write_fig_path(subfolder, name):
    path = os.path.join(write_fig_folder(subfolder), name)
    print("Saved at: ", path, '\n')
    return path


def format_ticks_3D(format_x='%.1f', format_y='%.1f', format_z='%.1f'):
    ax = plt.gca()
    ax.tick_params(axis='both', which='major', pad=2)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        ax.set_xticklabels(ax.get_xticks(), va='center_baseline',
                           horizontalalignment='center')
        ax.xaxis.set_major_formatter(ticker.FormatStrFormatter(format_x))

        ax.set_yticklabels(ax.get_yticks(), verticalalignment='baseline',
                           horizontalalignment='left')
        ax.yaxis.set_major_formatter(ticker.FormatStrFormatter(format_y))


def format_ticks_2D(ax=None, format_x='%.1f', format_y='%.1f'):
    if ax is None:
        ax = plt.gca()  # Default to the current axis
    # x-axis.
    if isinstance(format_x, str):
        ax.xaxis.set_major_formatter(ticker.FormatStrFormatter(format_x))
    else:
        txts = [f'{float(txt.get_text()):{format_x}}' for txt in
                ax.get_xticklabels()]
        ax.set_xticklabels(txts)
    # y-axis.
    if isinstance(format_y, str):
        ax.yaxis.set_major_formatter(ticker.FormatStrFormatter(format_y))
    else:
        txts = [f'{float(txt.get_text()):{format_y}}' for txt in
                ax.get_yticklabels()]
        ax.set_yticklabels(txts)
    # ax.set_xticklabels(ax.get_xticks(),
    #                    verticalalignment='baseline',
    #                    horizontalalignment='left')
    # ax.xaxis.set_major_formatter(ticker.FormatStrFormatter(format_x))
    # ax.set_yticklabels(ax.get_yticks(),
    #                    verticalalignment='baseline',
    #                    horizontalalignment='left')
    # ax.yaxis.set_major_formatter(ticker.FormatStrFormatter(format_y))


# -------------------
# Computing functions
# -------------------

# Compute lambda for constant coefficients
# ----------------------------------------

def compute_lambda_2D(kappa, traits, beta):
    """Compute the value of the Malthus parameter (lambda) in the case of 2
    traits, and constant coefficients, from an exact explicit formula.

    Parameters
    ----------
    kappa : ndarray
        2D array (2, 2).
    traits : ndarray
        1D array (2,).
    beta : float
    """
    matrix = kappa - 0.5 * np.identity(2)
    sum_ = matrix[0, 0] * traits[0] + matrix[1, 1] * traits[1]
    diff = matrix[0, 0] * traits[0] - matrix[1, 1] * traits[1]
    product = matrix[0, 1] * matrix[1, 0] * traits[0] * traits[1]
    return beta * (sum_ + np.sqrt(diff ** 2 + 4 * product))


def compute_lambdas_2D_wrt_traits(kappa, fixed_trait, traits, beta):
    """ Compute the values of the Malthus parameter (lambda) in the case of
    constant coefficients and 2 traits: one fixed, and one running through
    `traits`. Compute from an exact explicit formula.

    Parameters
    ----------
    kappa : ndarray
        2D array (2, 2).
    fixed_trait : float
    traits : ndarray
        1D array (trait_count,).
    beta : float

    Returns
    -------
    lambdas : ndarray
        1D array (comput_count,).
    """
    matrix = kappa - 0.5 * np.identity(2)
    sums = matrix[0, 0] * fixed_trait + matrix[1, 1] * traits
    diffs = matrix[0, 0] * fixed_trait - matrix[1, 1] * traits
    products = matrix[0, 1] * matrix[1, 0] * fixed_trait * traits
    return beta * (sums + np.sqrt(diffs ** 2 + 4 * products))


def approx_lambda_constant(kappa, traits, beta):
    """ Compute the value of the Malthus parameter (lambda) in the case of
    `trait_count` traits (`traits`), and constant coefficients.

    Parameters
    ----------
    kappa : ndarray
        2D array `(trait_count, trait_count)`.
    fixed_trait : float
    traits : ndarray
        1D array (trait_count,).
    beta : float

    """
    trait_count = len(traits)
    if trait_count == 2:
        return compute_lambda_2D(kappa, traits, beta)
    matrix = kappa - 0.5 * np.identity(trait_count)
    return max(np.linalg.eig(2 * beta * matrix * np.transpose([traits]))[0])


def approx_adjoint_constant(kappa, traits, beta=1):
    """ Compute the value of the Malthus parameter (lambda), the spectral gap
    (lambda_1 - lambda_2) and phi in the case of `trait_count` traits
    (`traits`), and constant coefficients as well as the spectral gap.

    Parameters
    ----------
    kappa : ndarray
        2D array `(trait_count, trait_count)`.
    traits : ndarray
        1D array (trait_count,).
    beta : float

    """
    trait_count = len(traits)
    matrix = kappa - 0.5 * np.identity(trait_count)
    eigenvalues, eigenvectors = np.linalg.eig(2 * beta * matrix *
                                              np.transpose([traits]))
    imax = np.argmax(eigenvalues)
    print(eigenvalues, eigenvectors, imax)
    lambda_, phi = eigenvalues[imax], eigenvectors[:, imax]
    eigenvalues = np.sort(eigenvalues)
    gap = eigenvalues[-1] - eigenvalues[-2]
    return lambda_, phi, gap


def approx_int_direct_constant(kappa, traits, beta=1, is_gap_needed=False):
    """ Compute the value of the Malthus parameter (lambda), the spectral gap
    (lambda_1 - lambda_2) and int N in the case of `trait_count` traits
    (`traits`), and constant coefficients as well as the spectral gap.

    Parameters
    ----------
    kappa : ndarray
        2D array `(trait_count, trait_count)`.
    traits : ndarray
        1D array (trait_count,).
    beta : float

    """
    k1, k2 = kappa[0, 1], kappa[1, 0]
    v1, v2 = traits
    trait_count = len(traits)

    matrix = beta * (2 * np.transpose(kappa) - np.identity(trait_count)) \
        @ np.diag(traits)

    if len(traits) == 2 and not is_gap_needed:
        # Explicit formulae in the bimodal case.
        a1 = (.5 - k1) * v1
        a2 = (.5 - k2) * v2
        lambda_ = a1 + a2 + np.sqrt((a1 - a2) ** 2 + 4 * k1 * k2 * v1 * v2)
        if matrix[0, 0] - lambda_ == 0:
            eig_N = np.array([0, 1])
        else:
            eig_N = np.array([- matrix[0, 1] / (matrix[0, 0] - lambda_), 1])
        eig_N[eig_N == - np.inf] = 0
        eig_N = eig_N / np.sum(eig_N)
        gap = None
    else:
        eigenvalues, eigenvectors = np.linalg.eig(matrix)
        imax = np.argmax(eigenvalues)
        lambda_, eig_N = eigenvalues[imax], eigenvectors[:, imax]
        # Normalization of eig_N.
        eig_N[eig_N == - np.inf] = 0
        eig_N = eig_N / np.sum(eig_N)
        eigenvalues = np.sort(eigenvalues)
        gap = eigenvalues[-1] - eigenvalues[-2]
    return lambda_, eig_N, gap


# Auxiliary functions
# -------------------

# > Compute information.

def compute_means(values_s):
    """ Computes the arithmetic, geometrical and harmonic means for each set of
    values given in `values_s`.

    Parameters
    ----------
    values : List
        List of `values_count` 1D arrays.

    Returns
    -------
    means : dict
        Dictionary whose entries are a list of 1D array (values_count,).

    """
    means = {'arithmetic': [st.mean(values) for values in values_s],
             'geometric': [st.geometric_mean(values) for values in values_s],
             'harmonic': [st.harmonic_mean(values) for values in values_s]}
    return means


def compute_means_of_2(fixed_value, values):
    """ Computes the aritmetic, geometrical and harmonic means between
    `fixed_value` and the successive values of `values`.

    Parameters
    ----------
    fixed_trait : float
    values : ndarray
        1D array (value_count,).

    Returns
    -------
    means : dict
        Dictionary whose entries are 1D array (value_count,).

    """
    arithmetic = [st.mean([fixed_value, value]) for value in values]
    geometric = [st.geometric_mean([fixed_value, value]) for value in values]
    harmonic = [st.harmonic_mean([fixed_value, value]) for value in values]
    return {'arithmetic': arithmetic, 'geometric': geometric,
            'harmonic': harmonic}


def is_irreducible(non_negative_matrix):
    """
    Parameters
    ----------
    non_negative_matrix : ndarray
        2D array, a (m, m) matrix with non-negative components.

    Returns
    -------
    is_irr : bool
        True if the matrix given in argument is irreducible, False otherwise.

    """
    m = len(non_negative_matrix)
    power = 1
    power_matrix = non_negative_matrix
    while power <= m and not(np.all(power_matrix > 0)):
        power += 1
        power_matrix = np.dot(power_matrix, non_negative_matrix)
    return power <= m


# > Generate sets of traits.

def generate_numbers(mid_point, interval_length, number_count,
                     trait_choice='uniform'):
    """ Generate `number_count` floats distributed on an interval of length
    `interval_length` around `mid_point`.
    If `trait_choice` is 'uniform' (default option) the numbers are
    distributed regularly, if it is 'random', randomly according to a uniform
    law on the interval.

    """
    vmin = mid_point - interval_length / 2
    vmax = mid_point + interval_length / 2
    if trait_choice == 'uniform':
        return np.linspace(vmin, vmax, number_count)
    if trait_choice == 'random':
        return np.sort(np.random.uniform(vmin, vmax, number_count))


def generate_numbers_w_varying_std(mid_point, interval_lengths,
                                   number_count, trait_choice='uniform'):
    """ Generate `set_count` sets of `number_count` floats distributed
    > regularly if `trait_choice` is 'uniform' (default option)
    > randomly according to a uniform law if `trait_choice` is 'random'
    on the `set_count` intervals of length the lenghts given by
    `interval_lengths` around `mid_point`.

    Parameters
    ----------
    mid_point : float
        Positive number, the center of all the intervals generated.
    interval_lengths : ndarray
        1D array (set_count,) with all the lenghts of the intervals to
        generate.
    number_count : int
        Number of numbers in each interval.

    """
    numbers_s = [generate_numbers(mid_point, interval_length, number_count,
                                  trait_choice)
                 for interval_length in interval_lengths]
    return numbers_s


def generate_numbers_w_varying_count(mid_point, interval_length,
                                     number_counts, trait_choice='uniform'):
    """ Generate `set_count` sets of varying number of floats (given by
    `number_counts`) distributed
    > regularly if `trait_choice` is 'uniform' (default option)
    > randomly according to a uniform law if `trait_choice` is 'random'
    on the interval of length `interval_lengths` around `mid_point`.

    Parameters
    ----------
    mid_point : float
        Positive number, the center of the interval I where numbers are taken.
    interval_length : float
        The lenght of the interval I where numbers are taken.
    number_counts : ndarray
        Number of floats in the sets to generate.
    trait_choice : str
        Way to chose floats in the interval I.

    """
    numbers_s = [generate_numbers(mid_point, interval_length, number_count,
                                  trait_choice)
                 for number_count in number_counts]
    return numbers_s


# > Generate a transition matrix.

def kappa_limit(traits, idx_mother, sigma):
    """ Return the 1D array `row` (of length `trait_count`) such that
    `row[i]` is the probability that a cell with type `idx_mother` (i.e.
    trait `traits[idx_mother]`) give birth to a cell of type `i` given that
    the distribution of the trait of daughter cells is uniformly distributed
    among all the traits at a distance from `traits[idx_mother]` lower or
    equal than sigma.

    """
    vmin, vmax = min(traits), max(traits)
    v_mother = traits[idx_mother]
    row = np.array([])
    for v_daugther in traits:
        if abs(v_mother - v_daugther) <= sigma:
            if v_mother < vmin + sigma:
                coef = sigma + v_mother - vmin
            elif v_mother > vmax - sigma:
                coef = sigma + vmax - v_mother
            else:
                coef = 2 * sigma
        else:
            coef = 0
        row = np.append(row, coef)
    return row / np.sum(row)


def compute_specific_diag(trait_count):
    return 0.5 * (1 + 1 / trait_count)


def uniform_stochastic_matrix(n):
    M = np.random.exponential(scale=1.0, size=(n, n))  # Tirage exponentiel
    return M / M.sum(axis=1, keepdims=True)  # Normalisation ligne par ligne


def generate_kappa(traits, kappa_choice, diag=None):
    """ Generates a transition matrix (stochastic and a priori irreducible)
    according to the method specified by `kappa_choice`.

    Parameters
    ----------
    traits : ndarray
        1D array (trait_count,) of the individual traits (orderred).
    kappa_choice : str
        Way to generate kappa:
        > "uniform": uniform kernel i.e `kappa[i, j]` is `1 / trait_count`.
        > "random": `kappa[i, j]` draw randomly from a uniform law on [0, 1),
               and renormalized such that kappa is stochastic.
        > "load": kappa is `KAPPA_LOAD[trait_count]`.
        > "diag_n_uniform": `kappa` is `diag` on the diagonal and constant
              elsewhere (the constant is chosen s.t. kappa is stochastic).
        > "diag_n_uniform_specific": same as previous with `diag` chosen s.t.
              the effective fitness is (/approaches) arithmetic mean.
        > "band": the ith row of `kappa` is given by
            `kappa_limit(traits, i, SIGMA)` (see docstring of `kappa_limit`).
            In particular for SIGMA smaller than the length of the interval of
            traits `kappa` is a band matrix.
            NB: if traits are regularly spaced the values of kappa on the
            band are conserved at the limit `trait_count --> infty`.
        > "uniform_stochastic" `kappa` drawn uniformly amg stochastic matrices.

    Returns
    -------
    kappa : ndarray
         2D array (trait_count, trait_count) a stochatic matrix.

    """
    trait_count = len(traits)
    if kappa_choice == "uniform":
        return np.ones((trait_count, trait_count)) / trait_count
    if kappa_choice == "random":
        kappa = np.random.rand(trait_count, trait_count)
        if not is_irreducible(kappa):
            print(kappa)
            raise ValueError('Reducible kappa generated')
        return kappa / np.sum(kappa, axis=1)[:, None]
    if kappa_choice == "load":
        return KAPPA_LOAD[trait_count]
    if (kappa_choice == 'diag_n_uniform' or
       kappa_choice == 'diag_n_uniform_specific'):
        kappa = np.ones((trait_count, trait_count))
        if kappa_choice == 'diag_n_uniform_specific':
            diag = compute_specific_diag(trait_count)
        kappa = kappa * (1 - diag) / (trait_count - 1)
        for diag_idx in range(trait_count):
            kappa[diag_idx, diag_idx] = diag
        return kappa
    if kappa_choice == "band":
        kappa = np.zeros((trait_count, trait_count))
        for row_idx in range(trait_count):
            kappa[row_idx] = kappa_limit(traits, row_idx, SIGMA)
    if kappa_choice == "uniform_stochastic":
        kappa = np.random.exponential(scale=1.0,
                                      size=(trait_count, trait_count))
        return kappa / kappa.sum(axis=1, keepdims=True)  # Normalisation.
    return kappa


# > Writing functions.

def write_traits(traits):
    """ Convert the 1D array `traits` into a string to plot (LaTex format).

    NB: if traits is already a string, just return the string formatted into
        a legend.

    """
    if isinstance(traits, str):
        string = traits
    else:
        string = str(np.array(traits)).replace(' ', ', ')
        string = string.replace('[', r'\{{')
        string = string.replace(']', r'\}}')
    string = rf"$\mathcal{{V}} = {string}$"
    return string

def write_pmatrix(matrix):
    """ Returns a LaTeX pmatrix (to print proper Tex matrix on plots).

    Parameters
    ----------
    matrix : ndarray
        1D or 2D array of the matrix to plot.
    Return
    -------
    rv : str
        String corresponding to the matrix given in argument in latex format.

    """
    if len(matrix.shape) > 2:
        raise ValueError('pmatrix can at most display two dimensions')
    lines = str(matrix).replace('[', '').replace(']', '').splitlines()
    rv = [r' ' + r' & '.join(l.split()) for l in lines]
    rv = r' \\ '.join(rv)
    rv = r'$\kappa = \begin{pmatrix}' + rv + r' \end{pmatrix}$'
    return rv


def write_matrix(matrix, tmp=''):
    """ Convert a 2D array into a string used in a file name.

    """
    if not isinstance(matrix, list):
        return str(matrix.flatten()).replace(' ', '-')
    tmp = write_matrix(matrix[0])
    for mat in matrix[1:]:
        tmp = tmp + '_' + write_matrix(mat)
    return tmp


def write_kappa_choice(kappa_choice, diag=None, is_shorten=False):
    k_str = r"(\kappa_{{_M}})_"
    if kappa_choice == 'diag_n_uniform':
        if isinstance(diag, type(None)):
            end = rf'${k_str}{{ij}} = ' + r'\frac{1-\alpha}{M-1}, \; i \neq j$'
            if is_shorten:
                return end
            return rf'${k_str}{{ii}} = \alpha, \quad $' + end
        one_minus_diag = '0.1%f' % (1.-diag)
        one_minus_diag = one_minus_diag[3:6]
        end = rf'${k_str}{{ij}}=\frac{{{one_minus_diag}}}{{M-1}}, \; i \neq j$'
        if is_shorten:
            return end
        return rf'${k_str}{{ii}} = {diag}, \quad $' + end
    if kappa_choice == 'diag_n_uniform_specific':
        end = rf'${k_str}{{ij}} = \frac{{1}}{{2M}}, \; i \neq j$'
        if is_shorten:
            return end
        return rf'${k_str}{{ii}} = \frac{{M+1}}{{2M}}, \quad $' + end
    if kappa_choice == 'uniform':
        return rf'${k_str}{{ij}} = \frac{{1}}{{M}}$'
    if kappa_choice == 'uniform_stochastic':
        return r'$\kappa_{{_M}} \sim \mathcal{{U}}_{{\mathcal{{S}}_{{_M}}}}$'
    if kappa_choice == 'random':
        return r'$\kappa_{{_M}}~random$'
    # rf'${k_str}{{ij}} \sim \mathcal{{U}}_{{(0,1)}}$'
    return ''


# ------------------
# Plotting functions
# ------------------


def rearange_2D_kappa(kappa):
    return np.array([[kappa[1, 1], kappa[1, 0]], [kappa[0, 1], kappa[0, 0]]])


def plot_2D_v_eff_and_mean_wrt_traits_constant(
        kappas, fixed_trait, traits, beta, fig_directory,
        is_error_plotted=False, is_mirror=True, palette='rocket', labels=None,
        bbox_to_anchor=None):
    """Plot the effective trait `v_eff` and theoretical means w.r.t. traits.

    Constant coefficients case: tau(v,x)=v, beta(x)=beta, gamma(v,x)=v beta.

    Parameters
    ----------
    kappas : list or ndarray
        List of kappa matrices or a single kappa matrix.
    fixed_trait : float
        The fixed trait value.
    traits : ndarray
        Orderred individual traits (v_i = traits[i], with tau_i(x) = v_i).
    beta : float
        A scaling factor for the lambda computations.
    fig_directory : str, optional
        Directory to save the figure, if provided.
    is_error_plotted : bool, optional
        If True, plots the error between theoretical and approximated lambdas.
    is_mirror : bool, optional
        If True, adjusts kappa such that it corresponds to the set
        {min(v*, fixed_trait), max(v*, fixed_traits)}.
    """
    if isinstance(kappas, list):
        kappa_s = deepcopy(kappas)
        figsize = (6.2, 4.)  # Default (6.4, 4.8)
    else:
        kappa_s = [kappas]
        figsize = (6.8, 3.8)
    labels = labels or [LABELS['v_eff']] * len(kappa_s)
    linestyles = ['-', '-', '-', '-', ':', '-', '-.', '-', ':', '-', '-.']

    PALETTE = sns.color_palette(palette, len(kappa_s))

    # Compute.
    means_s = compute_means_of_2(fixed_trait, traits)
    if is_mirror:
        traits_l = traits[traits <= fixed_trait]
        traits_r = traits[traits > fixed_trait]
        kappa_1to2_s = [rearange_2D_kappa(kappa) for kappa in kappa_s]
        lambdas_l_s = [compute_lambdas_2D_wrt_traits(
            kappa, fixed_trait, traits_l, beta) for kappa in kappa_1to2_s]
        lambdas_r_s = [compute_lambdas_2D_wrt_traits(
            kappa_r, fixed_trait, traits_r, beta) for kappa_r in kappa_s]
        lambdas_s = [np.concatenate([lambdas_l_s[i], lambdas_r_s[i]]) for i in
                     range(len(lambdas_l_s))]
        matrices_s = [np.concatenate([
            [kappa_1to2_s[i] - 0.5 * np.identity(2) for v in traits_l],
            [kappa_s[i] - 0.5 * np.identity(2) for v in traits_r]]) for i
            in range(len(lambdas_l_s))]
    else:
        lambdas_s = [compute_lambdas_2D_wrt_traits(
            kappa, fixed_trait, traits, beta) for kappa in kappa_s]
        matrices_s = [[kappa - 0.5 * np.identity(2) for v in traits] for
                      kappa in kappa_s]

    # Plot the difference between theoretical & approximated lambdas.
    plt.figure(figsize=figsize)
    ax = plt.gca()
    for idx in range(len(kappa_s)):
        lambdas_approx = [max(np.linalg.eig(2 * beta * matrices_s[idx][i] *
                          np.array([[fixed_trait], [traits[i]]]))[0]) for i
                          in range(len(traits))]
        print('max(|lambdas - lambdas_approx|): ',
              max(abs(lambdas_s[idx] - lambdas_approx)))
        if is_error_plotted:
            plt.plot(lambdas_s[idx] - lambdas_approx,
                     label=r"$\lambda_{th}-\lambda_{app}$")
            plt.text(.7, 0.15, write_traits(r'\{{ v_1, v_2 \}}'),
                     transform=ax.transAxes)
        print(labels[idx], kappa_s[idx], "!!!")
        plt.plot(traits, lambdas_s[idx] / beta, label=labels[idx],
                 linestyle=linestyles[idx], color=PALETTE[idx])
        if len(kappa_s) == 1:
            plt.text(.05, 0.7, write_traits(rf'\{{ v^*, {fixed_trait} \}}')
                     + '\n' + write_pmatrix(kappa_s[0]),
                     transform=ax.transAxes)
    # > Classical means.
    color_means = "red"
    default_lwidth = sns.plotting_context()['lines.linewidth']
    for key, mean in means_s.items():
        if key == 'arithmetic':
            plt.plot(traits, mean, label=LABELS_MEANS, linestyle='--',
                     linewidth=.85 * default_lwidth, color=color_means)
        else:
            plt.plot(traits, mean, color=color_means, linestyle='--',
                     linewidth=.85 * default_lwidth)
    ax.xaxis.set_major_locator(ticker.MultipleLocator(base=1))
    plt.legend(  # loc="lower right", fontsize="small",
               bbox_to_anchor=bbox_to_anchor)
    plt.xlabel(r"Trait $v^*$", labelpad=6)  # Feature $v^*$", labelpad=6)
    plt.ylabel(r"Effective trait $v$", labelpad=8)  # fitness
    sns.despine()
    format_ticks_2D(format_x='%.0f', format_y='%.0f')
    if not isinstance(fig_directory, type(None)):
        path = write_fig_path(fig_directory, 'v_and_means_M2_wrt_1v_' +
                              f'kappa{write_matrix(kappa_s)}.pdf')
        # feature replaced by v in the figure name !!!!!!
        plt.savefig(path, bbox_inches='tight')
    plt.show()
    return lambdas_s


def plot_v_eff_and_means_w_varying_count_constant(
        fixed_trait, interval_length, trait_counts, kappa_choice, beta,
        fig_directory, trait_choice="uniform", diag=0.5, palette='rocket',
        labels=None):

    if isinstance(kappa_choice, list):
        kappa_choices = deepcopy(kappa_choice)
        figsize = (6.8, 4.2)
    else:
        kappa_choices = [kappa_choice]
        figsize = (6.8, 3.8)
    curve_count = len(kappa_choices)
    diags = [None] * curve_count
    if isinstance(diag, list):
        diags[kappa_choices == 'diag_n_uniform'] = diags
    else:
        diags[kappa_choices == 'diag_n_uniform'] = diag

    labels = labels or [LABELS['v_eff']] * curve_count
    linestyles = ['-', '-', '-', '-', ':', '-', '-.', '-', ':', '-', '-.']
    PALETTE = sns.color_palette(palette, curve_count)

    # Compute.
    traits_s = generate_numbers_w_varying_count(
        fixed_trait, interval_length, trait_counts, trait_choice)
    kappas_s = [[generate_kappa(v, kappa_choices[i], diag=diags[i]) for v in
                 traits_s] for i in range(curve_count)]
    lambdas_s = [np.array([approx_lambda_constant(
        kappas[i], traits_s[i], beta) for i in range(len(traits_s))]) for
        kappas in kappas_s]
    means = compute_means(traits_s)

    # Plot.
    plt.figure(figsize=figsize)
    for idx in range(curve_count):
        plt.plot(trait_counts, lambdas_s[idx] / beta,
                 label=labels[idx], linestyle=linestyles[idx],
                 color=PALETTE[idx])
        if len(kappa_choices) == 1:
            plt.title(write_kappa_choice(kappa_choice, diags[idx]), pad=8,
                      loc='right')
        if kappa_choices[idx] == 'diag_n_uniform':
            kappa_choices[idx] = kappa_choices[idx] + str(diags[idx])
    # > Classical means.
    color_means = "red"
    default_lwidth = sns.plotting_context()['lines.linewidth']
    for key, mean in means.items():
        if key == 'arithmetic':
            plt.plot(trait_counts, mean, label=LABELS_MEANS, linestyle='--',
                     linewidth=.85 * default_lwidth, color=color_means)
        else:
            plt.plot(trait_counts, mean, color=color_means, linestyle='--',
                     linewidth=.85 * default_lwidth)
    plt.legend()
    plt.ylim(bottom=min(means['harmonic']) * 0.8)
    plt.xlabel(r"Number of traits $M$", labelpad=6)  # traits
    format_ticks_2D(format_x='%.0f', format_y='%.1f')
    sns.despine()
    # Save.
    kappa_choices.sort()
    name_kappa = wp.list_to_string(kappa_choices)
    if not isinstance(fig_directory, type(None)):
        path = write_fig_path(fig_directory, 'v_and_means_wrt_M_kappa-' +
                              f'{name_kappa}_V-{trait_choice}.pdf')
        plt.savefig(path, bbox_inches='tight')
    plt.show()
    return lambdas_s


def plot_v_eff_and_means_w_varying_std_constant(
        fixed_trait, interval_lengths, trait_count, kappa_choice, beta,
        fig_directory, trait_choice, diag=0.5, is_legend=True,
        palette='rocket', labels=None, bbox_to_anchor=None):

    if isinstance(kappa_choice, list):
        kappa_choices = deepcopy(kappa_choice)
        figsize = (5.8, 4.2)
    else:
        kappa_choices = [kappa_choice]
        figsize = (6.8, 3.8)
    curve_count = len(kappa_choices)
    diags = [None] * curve_count
    idx_diag = [i for i, kc in enumerate(kappa_choices) if
                kc == 'diag_n_uniform']
    if isinstance(diag, list):
        for i, idx in enumerate(idx_diag):
            diags[idx] = diag[i]
    else:
        for idx in idx_diag:
            diags[idx] = diag

    labels = labels or [LABELS['v_eff']] * curve_count
    linestyles = ['-', '-', '-', '-', ':', '-', '-.', '-', ':', '-', '-.']
    PALETTE = sns.color_palette(palette, curve_count)

    # Compute.
    traits_s = generate_numbers_w_varying_std(fixed_trait, interval_lengths,
                                              trait_count, trait_choice)
    kappas_s = [[generate_kappa(traits, kappa_choices[i], diag=diags[i]) for
                 traits in traits_s] for i in range(curve_count)]
    lambdas_s = [np.array([approx_lambda_constant(
        kappas[i], traits_s[i], beta) for i in range(len(traits_s))]) for
        kappas in kappas_s]
    means = compute_means(traits_s)

    # Plot.
    plt.figure(figsize=figsize)
    idx_diag = 0
    for idx in range(curve_count):
        plt.plot(interval_lengths, lambdas_s[idx] / beta, label=labels[idx],
                 linestyle=linestyles[idx], color=PALETTE[idx])
        if len(kappa_choices) == 1:
            plt.title(write_kappa_choice(kappa_choice, diags[idx]), pad=8,
                      loc='right')
        if kappa_choices[idx] == 'diag_n_uniform':
            kappa_choices[idx] = kappa_choices[idx] + str(diags[idx])

    # > Classical means.
    color_means = "red"
    default_lwidth = sns.plotting_context()['lines.linewidth']
    for key, mean in means.items():
        if key == 'arithmetic':
            plt.plot(interval_lengths, mean, label=LABELS_MEANS,
                     linestyle='--', linewidth=.85 * default_lwidth,
                     color=color_means)
        else:
            plt.plot(interval_lengths, mean, color=color_means, linestyle='--',
                     linewidth=.85 * default_lwidth)
    plt.legend(  # fontsize="small",
               bbox_to_anchor=bbox_to_anchor)
    plt.xlabel(r"Length $\sigma$ of the interval of traits", labelpad=6)  # traits"
    format_ticks_2D(format_x='%.0f', format_y='%.1f')
    sns.despine()

    # Save.
    kappa_choices.sort()
    name_kappa = wp.list_to_string(kappa_choices)
    if not isinstance(fig_directory, type(None)):
        path = write_fig_path(
            fig_directory, f'v_and_means_M{trait_count}' +
            f'_wrt_std_kappa-{name_kappa}_V-{trait_choice}.pdf')
        plt.savefig(path, bbox_inches='tight')
    plt.show()
    plt.clf()

    # plt.figure(figsize=(6, 3.2))  # Default (6.4, 4.8)
    # for key in means:
    #     if is_legend:
    #         plt.plot(interval_lengths, means[key], label=LABELS[key])
    #     else:
    #         plt.plot(interval_lengths, means[key])
    # plt.plot(interval_lengths, lambdas / beta, label=LABELS['v_eff'],
    #          linestyle='--')
    # plt.title(write_kappa_choice(kappa_choice, diag), pad=8, loc='right')
    # plt.legend(fontsize="small")
    # plt.xlabel(r"Length of the interval of traits", labelpad=6)
    # if not isinstance(fig_directory, type(None)):
    #     path = write_fig_path(
    #         fig_directory, f'v_and_means_M{trait_count}_wrt_std_kappa-' +
    #         f'{kappa_choice1}_V-{trait_choice}_small.pdf')
    #     plt.savefig(path, bbox_inches='tight')
    # plt.show()
    return lambdas_s


def plot_v_eff_and_means_w_varying_std_n_correlation(
        diags, fixed_trait, interval_lengths, trait_count, par_beta,
        fig_directory, conserved_mean='arithmetic', is_classic_mean=False,
        is_title=True, is_small_only=False, case_coef="constant",
        formats=["small"]):
    to_add = f'_CV{conserved_mean[:3]}'
    if case_coef == 'constant':
        beta_constant = par_beta
        to_add = to_add + f"_{case_coef}-{par_beta}"
    elif case_coef == 'linear':
        beta_constant = par_beta[1]
        to_add = to_add + f"_{case_coef}-{par_beta[0]}-{par_beta[1]}"
    else:
        print(case_coef)
        raise ValueError('Wrong value for `case_coef`')
    curve_count = len(diags)
    colors = sns.color_palette('viridis', curve_count)
    # Compute.
    if conserved_mean == 'arithmetic':
        traits_s = generate_numbers_w_varying_std(fixed_trait,
                                                    interval_lengths,
                                                    trait_count, 'uniform')
    elif conserved_mean == 'geometric':
        csts = np.sqrt(1 + 4 * fixed_trait ** 2 / interval_lengths)
        interval_lengths_geo = np.log((csts + 1) / (csts - 1))
        traits_s = np.exp(generate_numbers_w_varying_std(
            np.log(fixed_trait), interval_lengths_geo,
            trait_count, 'uniform'))
    elif conserved_mean == 'harmonic':
        csts = np.sqrt(interval_lengths ** 2 + fixed_trait ** 2)
        v1s = (- interval_lengths + fixed_trait + csts) / 2
        interval_lengths_har = interval_lengths / (v1s * (interval_lengths +
                                                          v1s))
        traits_s = generate_numbers_w_varying_std(
            1 / fixed_trait, interval_lengths_har, trait_count, 'uniform')
        traits_s = np.sort([1 / traits for traits in traits_s])
    if is_classic_mean:
        means_s = {
            'arithmetic': [np.mean(fs) for fs in traits_s],
            'geometric': [st.geometric_mean(fs) for fs in traits_s],
            'harmonic': [st.harmonic_mean(fs) for fs in traits_s]}

    # Plot.
    if 'large' in formats:
        plt.figure(figsize=(6.8, 4))  # Default (6.4, 4.8)
        for i in range(curve_count):
            diag = diags[i]
            kappas = [generate_kappa(traits, 'diag_n_uniform', diag=diag) for
                      traits in traits_s]
            if case_coef == 'constant':
                lambdas = np.array([approx_lambda_constant(
                    kappas[i], traits_s[i], beta_constant) for i in
                    range(len(traits_s))])
            else:
                lambdas = np.array([approx_lambda_linear(
                    traits_s[i], par_beta, [kappas[i], f'diag{diag}'],
                    is_traits_regular=True) for i
                    in range(len(traits_s))])
            plt.plot(interval_lengths, lambdas / beta_constant,
                     label=rf'${diag}$', color=colors[i])
        if is_title:
            plt.title(write_kappa_choice('diag_n_uniform'), pad=8, loc='right')
        else:
            to_add = to_add + "_no-title"
        if is_classic_mean:
            to_add = to_add + "_w-means"
            for key, means in means_s.items():
                if key == 'arithmetic':
                    plt.plot(interval_lengths, means, color='red',
                             label=LABELS_MEANS, linestyle='--')
                else:
                    plt.plot(interval_lengths, means, color='red',
                             linestyle='--')
        plt.legend(title=r"$\alpha$", bbox_to_anchor=(1.02, 1.06),
                   loc='upper left', edgecolor='#F0F0F0')
        plt.xlabel(r"Length of the interval of traits", labelpad=6)
        plt.ylabel(r"Effective fitness", labelpad=8)
        sns.despine()
        # Save.
        if not is_small_only:
            if not isinstance(fig_directory, type(None)):
                path = write_fig_path(
                    fig_directory,
                    f'v_and_means_M{trait_count}_wrt_std_n_correlations' +
                    f'{diags[0]}-{diags[-1]}-{len(diags)}{to_add}.pdf')
                plt.savefig(path, bbox_inches='tight')
        plt.show()

    # Plot.
    if 'small' in formats:
        plt.figure(figsize=(5.4, 3.6))  # Default (6.4, 4.8)
        for i in range(curve_count):
            diag = diags[i]
            kappas = [generate_kappa(traits, 'diag_n_uniform', diag=diag) for
                      traits in traits_s]
            if case_coef == 'constant':
                lambdas = np.array([approx_lambda_constant(
                    kappas[i], traits_s[i], beta_constant) for i in
                    range(len(traits_s))])
            else:
                lambdas = np.array([approx_lambda_linear(
                    traits_s[i], par_beta, [kappas[i], f'diag{diag}'],
                    is_traits_regular=True) for i in
                    range(len(traits_s))])
            plt.plot(interval_lengths, lambdas / beta_constant,
                     label=rf'${diag}$', color=colors[i])
        if is_title:
            plt.title(write_kappa_choice('diag_n_uniform'), pad=10,
                      loc='right')
        if is_classic_mean:
            for key, means in means_s.items():
                if key == 'arithmetic':
                    plt.plot(interval_lengths, means, color='red',
                             label=LABELS_MEANS, linestyle='--')
                else:
                    plt.plot(interval_lengths, means, color='red',
                             linestyle='--')
        plt.legend(title=r"$\alpha$", bbox_to_anchor=(1, 1.15),
                   loc='upper left', edgecolor='#F0F0F0')
        plt.xlabel(r"Length $\sigma$ of the interval of traits", labelpad=6)
        plt.ylabel(r"Effective trait", labelpad=8)
        format_ticks_2D(format_x='%.0f', format_y='%.1f')
        sns.despine()
        # Save.
        if not isinstance(fig_directory, type(None)):
            path = write_fig_path(
                fig_directory,
                f'v_and_means_M{trait_count}_wrt_std_n_correlations' +
                f'{diags[0]}-{diags[-1]}-{len(diags)}_small{to_add}.pdf')
            plt.savefig(path, bbox_inches='tight')
        plt.show()
    return lambdas


def plot_my_surface(x, y, z, ticks=None, ticks_labels=None, label=None,
                    angle_rotation=0):
    xx, yy = np.meshgrid(x, y)
    # Rotate the samples by pi / 4 radians around y.
    # https://stackoverflow.com/questions/38326983/how-to-rotate-a-3d-surface
    # NB. Other method: change it interactively:
    # Click on Tools/Preferences/IPython console/Graphics.
    # Set “Backend” from “Inline” to “Automaticlly”, and reset spyder.
    if angle_rotation != 0:
        t = np.transpose(np.array([xx, yy, z]), (1, 2, 0))
        m = [[np.cos(angle_rotation), 0, np.sin(angle_rotation)], [0, 1, 0],
             [- np.sin(angle_rotation), 0, np.cos(angle_rotation)]]
        xx, yy, z = np.transpose(np.dot(t, m), (2, 0, 1))

    ticks_labels = ticks_labels or [None] * 3
    with sns.axes_style("ticks"):  # "ticks") # "darkgrid")  whitegrid
        fig = plt.figure(figsize=(8, 6))
        ax = fig.subplots(subplot_kw={"projection": "3d"})
        ax.plot_surface(xx, yy, z, cmap=cm.inferno,  # cmap="binary",
                        linewidth=0.01, rstride=5, cstride=5, alpha=.92,
                        label=label)
        if ticks is not None:
            if not isinstance(ticks, list):
                ticks = [ticks] * 3
            ax.set_xticks(ticks[0])
            ax.set_yticks(ticks[1])
            ax.set_zticks(ticks[2])
        ax.set_xlabel(ticks_labels[0])
        ax.set_ylabel(ticks_labels[1])
        ax.set_zlabel(ticks_labels[2])
        if label is not None:
            plt.legend()
        plt.show()


def compute_n_plot_heatmap_veff_wrt_kappa_constant(
        kappa_coefficients, traits, beta_constant, palette="inferno",
        is_mass=False, fig_dir=None):

    coef_count = len(kappa_coefficients)
    heatmap = {'lambda': np.zeros((coef_count, coef_count)),
               'mass': np.zeros((coef_count, coef_count))}
    for i in range(coef_count):
        for j in range(coef_count):
            kappa = ker.kappa_bimodal(kappa_11=1 - kappa_coefficients[j],
                                      kappa_22=1 - kappa_coefficients[i])
            out = approx_int_direct_constant(kappa[0], traits, beta_constant)
            heatmap['lambda'][i, j] = out[0]
            if is_mass:
                heatmap['mass'][i, j] = np.sum(out[1][0]) / np.sum(out[1])
    df_lambda = pd.DataFrame(data=heatmap['lambda'],
                             columns=kappa_coefficients,
                             index=kappa_coefficients)
    plt.figure()
    tick_positions = np.linspace(0, 1, 11)  # Desired ticks
    tick_idxs = [np.argmin(np.abs(kappa_coefficients - tick)) for tick in
                 tick_positions]
    s = sns.heatmap(df_lambda, cmap=palette, xticklabels=tick_idxs,
                    yticklabels=tick_idxs, cbar_kws=dict(format="{x:.2f}"),
                    square=True, linewidths=0.0, rasterized=True)
    # Manually set tick labels
    s.set_xticks(tick_idxs)
    s.set_xticklabels([f"{tick:.1f}" for tick in tick_positions])
    s.set_yticks(tick_idxs)
    s.set_yticklabels([f"{tick:.1f}" for tick in tick_positions])
    # Rotate for readability.
    plt.xticks(rotation=45)
    s.set(xlabel=r"$k_1$", ylabel=r"$k_2$")

    if not isinstance(fig_dir, type(None)):
        path = write_fig_path(
            fig_dir, f'heatmap_lambda_v{wp.list_to_string(traits)}.pdf')
        plt.savefig(path, bbox_inches='tight')
    plt.show()

    if is_mass:
        df_mass = pd.DataFrame(data=1 - heatmap['mass'][i, j],
                               columns=kappa_coefficients,
                               index=kappa_coefficients)
        plt.figure()
        sns.heatmap(df_mass, cmap=palette, xticklabels=tick_idxs, square=True,
                    yticklabels=tick_idxs, cbar_kws=dict(format="{x:.2f}"),
                    linewidths=0.0, rasterized=True)
        s.set_xticks(tick_idxs)
        s.set_xticklabels([f"{tick:.1f}" for tick in tick_positions])
        s.set_yticks(tick_idxs)
        s.set_yticklabels([f"{tick:.1f}" for tick in tick_positions])
        plt.xticks(rotation=45)
        s.set(xlabel=r"$k_1$", ylabel=r"$k_2$")
        if not isinstance(fig_dir, type(None)):
            path = write_fig_path(
                fig_dir, f'heatmap_mass_v{wp.list_to_string(traits)}.pdf')
            plt.savefig(path, bbox_inches='tight')
        plt.show()
    return heatmap


def compute_longtime_approximation_2D_mix_vs_irr(
        is_conservative, traits, par_beta, kappa_irr,
        par_ninit=PAR_N_INIT_LONGTIME, par_grids=PAR_GRIDS_CONSTANT_LONGTIME,
        is_normalized_by_v=False, fig_dir=None):
    """Approximation of the 1st eigenvector for mitosis, constant growth rates.

    Case of equal mitosis with coefficients:
        tau(v,x) = v and gamma(v,x) = v beta(x).

    Parameters
    ----------
    traits : ndarray
        Orderred individual traits (v_i = traits[i], with tau_i(x) = v_i).
    """
    # Full mixing.
    par_kappa = {'irr': [kappa_irr, None]}

    # No mixing computed as a "sytem" with kappa=Id (no couling), rescaling
    # each trait, rather than as 2 equations of homogenehous population.
    par_kappa['red'] = ker.kappa_identity(2)

    out = {}
    for key, pkappa in par_kappa.items():
        out_tmp = scheme.compute_longtime_approximation_constant(
            is_conservative, traits, par_beta, pkappa, par_ninit, par_grids,
            is_printed=False)
        if 'sizes' not in out:
            out['sizes'] = out_tmp['sizes']
        out[key] = out_tmp['n_last'] / np.sum(out_tmp['n_last'])
    out['red'] = plot.normalize_by_v(out['red'])
    if is_normalized_by_v:
        out['irr'] = plot.normalize_by_v(out['irr'])

    # Used to check that after rescaling, the 2 component of 'red' are the
    # distributions obtained with 2 independent aproximations.
    out['i1'] = scheme.compute_longtime_approximation_constant(
            is_conservative, [traits[0]], par_beta, ker.kappa_identity(1),
            par_ninit, par_grids, is_printed=False)['n_last']
    out['i2'] = scheme.compute_longtime_approximation_constant(
            is_conservative, [traits[1]], par_beta, ker.kappa_identity(1),
            par_ninit, par_grids, is_printed=False)['n_last']
    out['i1'] = out['i1'] / np.sum(out['i1'])
    out['i2'] = out['i2'] / np.sum(out['i2'])
    return out


def compute_n_plot_distribution_bimodal_wrt_kappa(
        is_conservative, traits, par_beta, kappa_coefficients,
        par_ninit=PAR_N_INIT_LONGTIME, par_grids=PAR_GRIDS_CONSTANT_LONGTIME,
        fig_dir=None, is_mirror=False, xmax=None):

    # Création de la figure et d'un axe 3D
    fig = plt.figure(figsize=((6.4, 6)))
    ax = fig.add_subplot(111, projection='3d')
    z = []
    for cc in kappa_coefficients[::None]:
        coef = 1 - cc
        if is_mirror:
            kappa = np.array([[1 - coef, coef], [1 - coef, coef]])
            ylabel = r"$k_1=1-k_2$"
            name_end = "k1_1-k2"
        else:
            kappa = np.array([[1 - coef, coef], [coef, 1 - coef]])
            ylabel = r"$k_1=k_2$"
            name_end = "k1_k2"

        out = compute_longtime_approximation_2D_mix_vs_irr(
            is_conservative, traits, par_beta, kappa, par_ninit, par_grids,
            is_normalized_by_v=False)

        if len(z) == 0:  # Legend and x-axis.
            if xmax is None:
                idx_max = len(out['sizes']) - 1
            else:
                idx_max = np.where(out['sizes'] <= xmax)[0][-1]
            sizes = out['sizes'][:idx_max]

            z.append([out['irr'][1]])
            ax.plot(sizes, ys=z[-1][0][:idx_max], zs=coef, zdir='y',
                    color='blue', label=r"$N_2$")
            z.append([out['irr'][0]])
            ax.plot(sizes, ys=z[-1][0][:idx_max], zs=coef, zdir='y',
                    color='orange', label=r"$N_1$")
        else:  # No legend.
            z.append([out['irr'][1]])
            ax.plot(sizes, ys=z[-1][0][:idx_max], zs=coef, zdir='y',
                    color='blue')
            z.append([out['irr'][0]])
            ax.plot(sizes, ys=z[-1][0][:idx_max], zs=coef, zdir='y',
                    color='orange')

        # # Test 'i{k}' and 'red{k}' coincide.
        # distributions = np.concatenate([out['irr'], [out['red'][0]]])
        # plot.plot_distribution(out['sizes'], traits, distributions,
        #                        labels=['irr1', 'irr2', 'red'], xmax=X_MAX)
        # plot.plot_distribution(
        #     out['sizes'], traits,
        #     np.concatenate(
        #         [np.concatenate([out['i1'], out['i2']]), out['red']]),
        #     labels=['i1', 'i2', 'red1', 'red2'], xmax=X_MAX)

    format_ticks_3D(format_x='%.0f')
    ax.set_xlabel("Size", labelpad=6)  # , rotation=-10)
    ax.set_ylabel(ylabel, labelpad=9)
    ax.set_zticks([])
    plt.legend()
    if not isinstance(fig_dir, type(None)):
        path = write_fig_path(
            fig_dir, f'N_v{wp.list_to_string(traits)}_wrt_{name_end}.pdf')
        plt.savefig(path, bbox_inches='tight')
    plt.show()

    # plot_my_surface(out['sizes'], coef, np.concatenate(z), ticks=None,
    #                    ticks_labels=None, label=None, angle_rotation=0)


def approx_lambda_linear(
        traits, par_beta, par_kappa, par_ninit=PAR_N_INIT_LONGTIME,
        par_grids=PAR_GRIDS_LINEAR_LONGTIME, is_traits_regular=False):
    """Approximation of the Malthus parameter for mitosis, linear growth rates.

    Case of equal mitosis with coefficients:
        tau(v,x) = vx and gamma(v,x) = vx beta(x).

    Parameters
    ----------
    traits : ndarray
        Orderred individual traits (v_i = traits[i], with tau_i(x) = v_i x).
    """
    d = scheme.compute_longtime_approximation(
        False, traits, par_beta, par_kappa, par_ninit, par_grids)
    print('\n', [lambda_[-1] for lambda_ in d['lambda_estimates']])
    return d['lambda_estimates'][0][-1]


# def plot_v_eff_and_means_w_varying_std_n_correlation_linear(
#         diags, fixed_trait, interval_lengths, trait_count, par_beta,
#         par_ninit, par_grids, fig_directory, linestyles):
#     curve_count = len(diags)
#     colors = sns.color_palette('viridis', curve_count)
#     # Compute.
#     traits_s = generate_numbers_w_varying_std(
#         fixed_trait, interval_lengths, trait_count, 'uniform')
#     # Plot.
#     plt.figure(figsize=(5.4, 3.6))  # Default (6.4, 4.8) # !!!
#     for i in range(curve_count):
#         diag = diags[i]
#         kappas = [generate_kappa(traits, 'diag_n_uniform', diag=diag) for
#                   traits in traits_s]
#         lambdas = np.array([approx_lambda_linear(
#             traits_s[i], par_beta, [kappas[i], f'diag{diag}'], par_ninit,
#             par_grids, is_traits_regular=True) for i in
#             range(len(traits_s))])

#         plt.plot(interval_lengths, lambdas / par_beta[1], linestyles[i],
#                  label=rf'${diag}$', color=colors[i])
#         plt.title(write_kappa_choice('diag_n_uniform'), pad=8,
#                   loc='right')
#     plt.legend(title=r"$\alpha$", bbox_to_anchor=(1, 1.1),
#                 loc='upper left', edgecolor='#F0F0F0')
#     plt.xlabel(r"Length of the interval of traits", labelpad=6)
#     plt.ylabel(r"Effective fitness", labelpad=8)
#     # Save.
#     if not isinstance(fig_directory, type(None)):
#         path = f'{fig_directory}/v_and_means_M{trait_count}_wrt_std_n_' + \
#                 f'correlations{diags[0]}-{diags[-1]}-{len(diags)}.pdf'
#         print("Saved at: ", path)
#         plt.savefig(path, bbox_inches='tight')
#     plt.show()
#     return lambdas
