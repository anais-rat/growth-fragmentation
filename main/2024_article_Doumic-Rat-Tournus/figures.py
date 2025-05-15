#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov  7 12:17:43 2022

@author: arat

Script to run to plot the figures of the article (and some more):
"Comparison Between Effective and Individual Growth Rates in a Heterogeneous
 Population; Marie Doumic, Anaïs Rat, Magali Tournus."

"""

import matplotlib.pyplot as plt
import matplotlib.font_manager
import numpy as np
import os
# import pandas as pd
import seaborn as sns

# import parameters as par
import project_path
import src.effective_fitness as fv
import src.parameters.figures_properties as fp
# import src.parameters.kappas as ker
import src.plot as plot
# import src.write_path as wp


# ----------
# Parameters
# ----------

IS_SAVED = True  # True to save plotted figures.

FORMAT = 'article'


# Parameter of the figures
# .............................................................................
FIG_DIR = None

# Global plotting parameters and figure directory.
sns.set_theme()  # Theme reset to default.
matplotlib.rcParams.update(matplotlib.rcParamsDefault)  # Reset to default.
if FORMAT == 'article':
    if IS_SAVED:
        FIG_DIR = os.path.join('2024_article_Doumic-Rat-Tournus')
    sns.set_style("ticks")  # "ticks") # "darkgrid")  whitegrid
    # sns.axes_style("ticks")
    sns.set_context("talk", font_scale=1)  # "poster", font_scale=1)
    # plt.rcParams.update(fp.PAR_RC_UPDATE_MANUSCRIPT)
    PAR_RC_UPDATE = {
        'figure.dpi': fp.DPI,
        'text.usetex': True,
        'text.latex.preamble':
            r'\usepackage{dsfont, amsmath}',  # amsmath: pmatrix
        'font.family': "sans-serif",
        'font.sans-serif': ["Helvetica"],
        'legend.frameon': False,
        'legend.framealpha': 1,
        'legend.facecolor': 'white',
        'legend.edgecolor': 'white'
        }
    plt.rcParams.update(PAR_RC_UPDATE)
else:
    print("Redine 'Format' correctly")
print(sns.plotting_context())

PALETTE = 'inferno'
# .............................................................................


# ------------------------------
# Global parameters of the model
# ------------------------------

# Equation form.
IS_CONSERVATIVE = False

# Growth and fragmentation rates: > tau(v, x) = v tau(x).
#                                 > gamma(v, x) = v tau(x) beta(x).
# Case A: tau(x) = 1, beta = 1.
# Case B: tau(x) = x, beta general (e.g. linear).

# Division rate per unit of size: beta(x) = CONSTANT x ** ALPHA.
# > Case A.
ALPHA = 0
CONSTANT = 1
PAR_BETA_CONSTANT = [ALPHA, CONSTANT]
# > Case B.
ALPHA = 1
CONSTANT = 1
PAR_BETA_LINEAR = [ALPHA, CONSTANT]

# Parameter for plot.
X_MAX = 5


# FIGURES 2, 5 (case  A/B, M = 2)
# -------------------------------

# Features: one fixed (`FIXED_TRAIT`) and one running through `TRAITS`.
FIXED_TRAIT = 4
TRAITS = np.linspace(1, 8, 200)

# Various transition matrices.
KAPPAS = [  # > 0. Uniform kernel (geometric mean).
          np.array([[0.5, 0.5], [0.5, 0.5]]),
          # > 1. "Diagonal and uniform" (arithmetic mean).
          np.array([[0.75, 0.25], [0.25, 0.75]]),
          # > 2. Mixing beneficial to v_2.
          np.array([[0.2, 0.8],  [0.2, 0.8]]),
          # > 3. Low mixing kernel.
          np.array([[0.8, 0.2], [0.2, 0.8]]),
          # > 4. High mixing kernel.
          np.array([[0.2, 0.8], [0.8, 0.2]]),
          # > 5. Mixing beneficial to v_1.
          np.array([[0.8, 0.2], [0.8, 0.2]])]

# # One per graphe.
# for kappa in KAPPAS:
#     fv.plot_2D_v_eff_and_mean_wrt_traits_constant(
#         kappa, FIXED_TRAIT, TRAITS, PAR_BETA_CONSTANT, FIG_DIR,
#         palette='inferno')

# Several per graph.
# > FIGURE 1 (effective fitness coincides with clasical means).
LABELS = [rf'$(k_{{1}}, k_{{2}}) = ({kappa[0, 1]}, {kappa[1, 0]})$' for kappa
          in KAPPAS[:2]]
fv.plot_2D_v_eff_and_mean_wrt_traits_constant(
    KAPPAS[:2], FIXED_TRAIT, TRAITS, PAR_BETA_CONSTANT[1], FIG_DIR,
    labels=LABELS, palette='inferno')

# > FIGURE 5 (effective fitness not "controled" by a clasical mean).
LABELS = [rf'$(k_{{1}}, k_{{2}}) = ({kappa[0, 1]}, {kappa[1, 0]})$'
          for kappa in KAPPAS[2:]]
fv.plot_2D_v_eff_and_mean_wrt_traits_constant(
    KAPPAS[2:], FIXED_TRAIT, TRAITS, PAR_BETA_CONSTANT[1], FIG_DIR,
    labels=LABELS, palette='inferno', bbox_to_anchor=(1, 1))


# FIGURE 3  (case  A/B, M = 2)
# ----------------------------

# Growth rate and traits: tau(i, x) = TRAITS[i].
TRAITS = np.array([0.5, 2.5])  # np.array([1.4, 1.6])

# Transition matrix.
KAPPA = np.array([[0.7, 0.4], [0.5, 0.5]])  # (irreducible, full mixing).

# > kappa = KAPPA
out = fv.compute_longtime_approximation_2D_mix_vs_irr(
    IS_CONSERVATIVE, TRAITS, PAR_BETA_CONSTANT, KAPPA,
    is_normalized_by_v=False)
out_n = fv.compute_longtime_approximation_2D_mix_vs_irr(
    IS_CONSERVATIVE, TRAITS, PAR_BETA_CONSTANT, KAPPA, is_normalized_by_v=True)

# Plot all.
N_1, N_2 = np.sum(out['irr'], axis=1)
davg = [out['irr'][0] * N_1 + out['irr'][1] * N_2]
davg_n = davg / np.sum(davg)

davg_arit = [(out['irr'][0] + out['irr'][1]) / 2]
davg_arit_n = davg_arit / np.sum(davg_arit)

LABELS = [r"$\frac{N_1}{\int N_1 }$", r"$\frac{N_2}{\int N_2 }$",
          r"$N_{v}$",
          r"$N_1 + N_2$",
          r"$\frac{N_1 \int N_1 + N_2 \int N_2}{(\int N_1)^2 + (\int N_2)^2}$"]

# # N = (N_1, N_2) normalized (in particular N_1, N_2 not normalized).
# plot.plot_distribution(
#     out['sizes'], TRAITS,
#     np.concatenate([out['irr'], [out['red'][0]], davg_arit_n, davg_n]),
#     fig_name=f'comparison_N_normalized_v{TRAITS[0]}-{TRAITS[1]}',
#     fig_dir=FIG_DIR, y_key_label=None, labels=LABELS,
#     xmax=X_MAX, palette=PALETTE)

# plot.plot_distribution(
#     out['sizes'], TRAITS,
#     np.concatenate([out['irr'], [out['red'][0]], davg_arit, davg]),
#     fig_name=f'comparison_N_normalized_v{TRAITS[0]}-{TRAITS[1]}',
#     fig_dir=FIG_DIR, y_key_label=None, labels=LABELS,
#     xmax=X_MAX, palette=PALETTE)

# All N_i normalized.
plot.plot_distribution(
    out['sizes'], TRAITS,
    np.concatenate([out_n['irr'], [out_n['red'][0]], davg, davg_arit]),
    fig_name=f'comparison_N_v{TRAITS[0]}-{TRAITS[1]}',
    is_normalized_by_v=True, fig_dir=FIG_DIR, y_key_label=None, labels=LABELS,
    xmax=X_MAX, palette=PALETTE)


# FIGURE 4 (case  A/B, M = 2)
# ---------------------------

# Growth rate and traits: tau(i, x)= TRAITS[i].
TRAITS_S = [np.array([0.5, 2.5]), np.array([1.4, 1.6])]

KAPPA_COEFS = np.linspace(0, 1, 101)

for traits in TRAITS_S:
    fv.compute_n_plot_heatmap_veff_wrt_kappa_constant(
        KAPPA_COEFS, traits, PAR_BETA_CONSTANT[1], fig_dir=FIG_DIR)

# Alternative visualization.
TRAITS_S = [np.array([0.5, 2.5]), np.array([2.5, 0.5]), np.array([1, 1.5])]

for traits in TRAITS_S:
    z = np.zeros((len(KAPPA_COEFS), len(KAPPA_COEFS)))
    for i in range(len(KAPPA_COEFS)):
        k1 = KAPPA_COEFS[i]
        kappas = [np.array([[1-k1, k1], [k2, 1-k2]]) for k2 in KAPPA_COEFS]
        z[i] = np.array([fv.compute_lambda_2D(
            kappa, traits, PAR_BETA_CONSTANT[1]) for kappa in kappas])
    fv.plot_my_surface(KAPPA_COEFS, KAPPA_COEFS, z,  # ticks=np.arange(1, 9),
                       ticks_labels=["$k_1$", "$k_2$", "Effective fitness"])


# FIGURE 6, 10 (case  A/B, M = 2)
# -------------------------------

# Growth rate and traits: tau(i, x) = TRAITS[i].
TRAITS = np.array([0.5, 2.5])  # np.array([1.4, 1.6])

# Transition matrix.

COEFFICIENTS = np.linspace(0, 1, 21)  # (see below).

# Plot
# Both row vaying.
# > kappa = [[1 - coef, coef], [coef, 1 - coef]], with varying coef.
fv.compute_n_plot_distribution_bimodal_wrt_kappa(
    IS_CONSERVATIVE, TRAITS, PAR_BETA_CONSTANT, COEFFICIENTS, fig_dir=FIG_DIR,
    is_no_heredity=False, xmax=X_MAX)

# > kappa = [[1 - coef, coef], [1 - coef, coef]], with varying coef.
fv.compute_n_plot_distribution_bimodal_wrt_kappa(
    IS_CONSERVATIVE, TRAITS, PAR_BETA_CONSTANT, COEFFICIENTS, fig_dir=FIG_DIR,
    is_no_heredity=True, xmax=X_MAX)

# One row fixed (FIGURE 10).
# > kappa = [[0.2, 0.8], [coef, 1 - coef]], with varying coef.
fv.compute_n_plot_distribution_bimodal_wrt_kappa(
    IS_CONSERVATIVE, TRAITS, PAR_BETA_CONSTANT, COEFFICIENTS,
    kappa_fixed_coef=(0, 0.2), fig_dir=FIG_DIR, xmax=X_MAX)

# > kappa = [[0.8, 0.2], [coef, 1 - coef]], with varying coef.
fv.compute_n_plot_distribution_bimodal_wrt_kappa(
    IS_CONSERVATIVE, TRAITS, PAR_BETA_CONSTANT, COEFFICIENTS,
    kappa_fixed_coef=(0, 0.8), fig_dir=FIG_DIR, xmax=X_MAX)

# > kappa = [[1 - coef, coef], [0.2, 0.8]], with varying coef.
fv.compute_n_plot_distribution_bimodal_wrt_kappa(
    IS_CONSERVATIVE, TRAITS, PAR_BETA_CONSTANT, COEFFICIENTS,
    kappa_fixed_coef=(1, 0.2), fig_dir=FIG_DIR, xmax=X_MAX)

# > kappa = [[1 - coef, coef], [0.8, 0.2]], with varying coef.
fv.compute_n_plot_distribution_bimodal_wrt_kappa(
    IS_CONSERVATIVE, TRAITS, PAR_BETA_CONSTANT, COEFFICIENTS,
    kappa_fixed_coef=(1, 0.8), fig_dir=FIG_DIR, xmax=X_MAX)


# FIGURE 7 (Case A/B, M > 2)
# --------------------------

# Random seed (for reproducible figures. Uncomment to generate new random).
np.random.seed(9)

# > Varying number of traits M at v_min, v_max fixed.
V_MIN = 1
V_MAX = 7
TRAIT_COUNTS = np.arange(2, 100)


interval_length = V_MAX - V_MIN
fixed_trait = interval_length / 2 + V_MIN


KAPPA_CHOICES = ["diag_n_uniform_specific", "random", "uniform"]
TRAIT_CHOICE = "uniform"

LABELS = [fv.write_kappa_choice(kchoice, is_shorten=True) for kchoice in
          KAPPA_CHOICES]
fv.plot_v_eff_and_means_w_varying_count_constant(
    fixed_trait, interval_length, TRAIT_COUNTS, KAPPA_CHOICES,
    PAR_BETA_CONSTANT[1], FIG_DIR, trait_choice=TRAIT_CHOICE, labels=LABELS)


# FIGURE 8 (case  A/B, M > 2)
# ---------------------------

V_MID = 4
INTERVAL_LENGTHS = np.linspace(0.01, 3, 21)  # np.linspace(0.01, 3)

TRAIT_COUNT = 10
DIAGS = [.01, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]

fv.plot_v_eff_and_means_w_varying_std_n_correlation(
    DIAGS, V_MID, INTERVAL_LENGTHS, TRAIT_COUNT, PAR_BETA_CONSTANT[1], FIG_DIR,
    is_classic_mean=True, is_title=False, is_small_only=True)

fv.plot_v_eff_and_means_w_varying_std_n_correlation(
    DIAGS, V_MID, INTERVAL_LENGTHS, TRAIT_COUNT, PAR_BETA_CONSTANT[1], FIG_DIR,
    conserved_mean="geometric", is_classic_mean=True, is_title=False,
    is_small_only=True)

fv.plot_v_eff_and_means_w_varying_std_n_correlation(
    DIAGS, V_MID, INTERVAL_LENGTHS, TRAIT_COUNT, PAR_BETA_CONSTANT[1], FIG_DIR,
    conserved_mean="harmonic", is_classic_mean=True, is_title=False,
    is_small_only=True)


# FIGURES 9 (Case A/B, M > 2)
# ---------------------------

# > Varying variance for V centered in v_mid, at v_mid and M fixed.
V_MID = 4
INTERVAL_LENGTHS = np.linspace(0.1, 3)


KAPPA_CHOICES = ["diag_n_uniform", "diag_n_uniform", "diag_n_uniform_specific",
                 "random", "uniform"]
TRAIT_CHOICE = "uniform"
DIAGS = [0.8, 0.2]

LABELS = [fv.write_kappa_choice(KAPPA_CHOICES[i], diag=DIAGS[i],
                                is_shorten=True) for i in range(2)]
LABELS.extend([fv.write_kappa_choice(kchoice, is_shorten=True) for kchoice in
               KAPPA_CHOICES[2:]])

TRAIT_COUNT = 10
fv.plot_v_eff_and_means_w_varying_std_constant(
    V_MID, INTERVAL_LENGTHS, TRAIT_COUNT, KAPPA_CHOICES, PAR_BETA_CONSTANT[1],
    FIG_DIR, TRAIT_CHOICE, diag=DIAGS, labels=LABELS, bbox_to_anchor=(1, 1))

TRAIT_COUNT = 100
fv.plot_v_eff_and_means_w_varying_std_constant(
    V_MID, INTERVAL_LENGTHS, TRAIT_COUNT, KAPPA_CHOICES, PAR_BETA_CONSTANT[1],
    FIG_DIR, TRAIT_CHOICE, diag=DIAGS, labels=LABELS, bbox_to_anchor=(1, 1))


# FIGURES 10 (M > 2)
# ------------------


# Equal mitosis, linear tau, constant beta.

V_MID = 4
INTERVAL_LENGTHS = np.linspace(0.01, 3, 21)
TRAIT_COUNT = 10
DIAGS = [.01, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]

fv.plot_v_eff_and_means_w_varying_std_n_correlation(
    DIAGS, V_MID, INTERVAL_LENGTHS, TRAIT_COUNT, PAR_BETA_CONSTANT, FIG_DIR,
    is_classic_mean=True, is_title=False, is_small_only=True,
    case_coef='linear')


fv.plot_v_eff_and_means_w_varying_std_n_correlation(
    DIAGS, V_MID, INTERVAL_LENGTHS, TRAIT_COUNT, PAR_BETA_CONSTANT, FIG_DIR,
    conserved_mean="geometric",  is_classic_mean=True, is_title=False,
    is_small_only=True, case_coef='linear')

fv.plot_v_eff_and_means_w_varying_std_n_correlation(
    DIAGS, V_MID, INTERVAL_LENGTHS, TRAIT_COUNT, PAR_BETA_CONSTANT, FIG_DIR,
    conserved_mean="harmonic",  is_classic_mean=True, is_title=False,
    is_small_only=True, case_coef='linear')


# Equal mitosis, linear tau, linear beta.

fv.plot_v_eff_and_means_w_varying_std_n_correlation(
    DIAGS, V_MID, INTERVAL_LENGTHS, TRAIT_COUNT, PAR_BETA_LINEAR, FIG_DIR,
    is_classic_mean=True, is_title=False, is_small_only=True,
    case_coef='linear')

fv.plot_v_eff_and_means_w_varying_std_n_correlation(
    DIAGS, V_MID, INTERVAL_LENGTHS, TRAIT_COUNT, PAR_BETA_LINEAR, FIG_DIR,
    conserved_mean="geometric",  is_classic_mean=True, is_title=False,
    is_small_only=True, case_coef='linear')

fv.plot_v_eff_and_means_w_varying_std_n_correlation(
    DIAGS, V_MID, INTERVAL_LENGTHS, TRAIT_COUNT, PAR_BETA_LINEAR, FIG_DIR,
    conserved_mean="harmonic",  is_classic_mean=True, is_title=False,
    is_small_only=True, case_coef='linear')
