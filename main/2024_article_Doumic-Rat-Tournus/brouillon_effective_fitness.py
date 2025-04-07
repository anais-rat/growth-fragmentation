#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 17 10:26:48 2023

@author: arat
"""

import matplotlib.pyplot as plt
import matplotlib.font_manager
import numpy as np
import os
import seaborn as sns

import src.effective_fitness as fv
import src.parameters.figures_properties as fp

# ----------
# Parameters
# ----------

IS_SAVED = True
IS_KAPPA_PRINTED = False

FORMAT = 'manuscript'  # 'manuscript' or 'article'.

# Random seed (for reproducible figures. Uncomment to generate new random).
np.random.seed(2)


# .............................................................................
# Global plotting parameters and figure directory
# ...............................................

# Figures directory if saved.
FIG_DIR = None
if IS_SAVED:
    FIG_DIR = FORMAT

# Global plotting parameters.
# matplotlib.rcParams.update(matplotlib.rcParamsDefault)  # Reset to default.
# if FORMAT == 'manuscript':
#     sns.set_style("darkgrid")
#     sns.set_context("talk", font_scale=1)
#     plt.rcParams.update(fp.PAR_RC_UPDATE_MANUSCRIPT)
# elif FORMAT == 'article':
#     sns.set_style("ticks")
#     sns.set_context("poster", font_scale=1)
#     plt.rcParams.update(fp.PAR_RC_UPDATE_ARTICLE)



# Global plotting parameters.
matplotlib.rcParams.update(matplotlib.rcParamsDefault)  # Reset to default.
matplotlib.rcParams['text.latex.preamble'] = r'\usepackage{{amsmath}}'  # pmatrix

if FORMAT == 'manuscript':
    # if IS_SAVED:
    #     fig_dir = 'figures_manuscript'
    #     if (not os.path.exists(fig_dir)):
    #         os.makedirs(fig_dir)
    sns.set_style("darkgrid")
    sns.set_context("talk", font_scale=1)
    plt.rcParams.update({'axes.facecolor': ".94",
                         'text.usetex': True,
                         'figure.dpi': fp.DPI,   # Resolution of plottings.
                         'font.family': "sans-serif",  # latex-like: 'serif',
                         'font.sans-serif': "Helvetica",  # ... 'cmr10'
                         'legend.frameon': True,
                         'legend.framealpha': 1,
                         'legend.facecolor': 'white',
                         'legend.edgecolor': '#EAEAF2',
                         'legend.fancybox': True,
                         'legend.title_fontsize': 16.5,
                         'legend.fontsize': 16})
elif FORMAT == 'article':
    # if IS_SAVED:
    #     fig_dir = 'figures_article'
    #     if (not os.path.exists(fig_dir)):
    #         os.makedirs(fig_dir)
    sns.set_style("darkgrid")
    sns.set_context("paper", font_scale=2)
    plt.rcParams.update({'axes.facecolor': ".94",
                         'text.usetex': True,
                         'figure.dpi': 600,
                         'font.family': "sans-serif",
                         'font.sans-serif': ["Helvetica"],
                         'legend.frameon': True,
                         'legend.framealpha': 1,
                         'legend.facecolor': 'white',
                         'legend.edgecolor': 'white',
                         'legend.fancybox': True,
                         'lines.linewidth': 1.8
                         })
    # 'legend.title_fontsize': 15.5,
    # 'legend.fontsize': 15})
else:
    print("Redefine 'Format' correctly")
print("Global plotting parameters: \n", sns.plotting_context(), '\n')
# .............................................................................

# Creation of a color map for the lines to plot.
MY_PALETTE = sns.color_palette("rocket", 4)
sns.set_palette(MY_PALETTE)


# ---------------------
# Constant coefficients
# ---------------------

# Growth and fragmentation rates: > tau(v, x) = v.
#                                 > gamma(v,x) = v BETA.
BETA = 1  # Fragmentation rate per unit of size (constant).


# Case 2 features (M = 2)
# -----------------------

# Features: one fixed (`FIXED_FEATURE`) and one running through `FEATURES`.
FIXED_FEATURE = 4
FEATURES = np.linspace(1, 8, 200)

# > Uniform kernel.
KAPPA = np.array([[0.5, 0.5],
                  [0.5, 0.5]])
fv.plot_2D_v_eff_and_mean_wrt_features_constant(
    KAPPA, FIXED_FEATURE, FEATURES, BETA, FIG_DIR)

# > High mixing kernel.
KAPPA = np.array([[0.2, 0.8],
                  [0.8, 0.2]])
fv.plot_2D_v_eff_and_mean_wrt_features_constant(
    KAPPA, FIXED_FEATURE, FEATURES, BETA, FIG_DIR)

# > Low mixing kernel.
KAPPA = np.array([[0.8, 0.2],
                  [0.2, 0.8]])
fv.plot_2D_v_eff_and_mean_wrt_features_constant(
    KAPPA, FIXED_FEATURE, FEATURES, BETA, FIG_DIR)

# > Mixing beneficial to v_1.
KAPPA = np.array([[0.8, 0.2],
                  [0.8, 0.2]])
fv.plot_2D_v_eff_and_mean_wrt_features_constant(
    KAPPA, FIXED_FEATURE, FEATURES, BETA, FIG_DIR)

# > Mixing beneficial to v_2.
KAPPA = np.array([[0.2, 0.8],
                  [0.2, 0.8]])
fv.plot_2D_v_eff_and_mean_wrt_features_constant(
    KAPPA, FIXED_FEATURE, FEATURES, BETA, FIG_DIR)

KAPPA = np.array([[0.75, 0.25],
                  [0.25, 0.75]])
fv.plot_2D_v_eff_and_mean_wrt_features_constant(
    KAPPA, FIXED_FEATURE, FEATURES, BETA, FIG_DIR)


# Case M > 2
# ----------

# > Varying number of features M at v_min, v_max fixed.
V_MIN = 1
V_MAX = 7
FEATURE_COUNTS = np.arange(2, 100)


interval_length = V_MAX - V_MIN
fixed_feature = interval_length / 2 + V_MIN

KAPPA_CHOICE = "uniform"
FEATURE_CHOICE = "uniform"
fv.plot_v_eff_and_means_w_varying_count_constant(fixed_feature, interval_length,
                                        FEATURE_COUNTS, KAPPA_CHOICE, BETA,
                                        FIG_DIR, FEATURE_CHOICE)

KAPPA_CHOICE = "uniform"
FEATURE_CHOICE = "random"
fv.plot_v_eff_and_means_w_varying_count_constant(fixed_feature, interval_length,
                                        FEATURE_COUNTS, KAPPA_CHOICE, BETA,
                                        FIG_DIR, FEATURE_CHOICE)

KAPPA_CHOICE = "random"
FEATURE_CHOICE = "uniform"
fv.plot_v_eff_and_means_w_varying_count_constant(fixed_feature, interval_length,
                                        FEATURE_COUNTS, KAPPA_CHOICE, BETA,
                                        FIG_DIR, FEATURE_CHOICE)

KAPPA_CHOICE = "random"
FEATURE_CHOICE = "random"
fv.plot_v_eff_and_means_w_varying_count_constant(fixed_feature, interval_length,
                                        FEATURE_COUNTS, KAPPA_CHOICE, BETA,
                                        FIG_DIR, FEATURE_CHOICE)

KAPPA_CHOICE = "random"
FEATURE_CHOICE = "random"
fv.plot_v_eff_and_means_w_varying_count_constant(fixed_feature, interval_length,
                                        FEATURE_COUNTS, KAPPA_CHOICE, BETA,
                                        FIG_DIR, FEATURE_CHOICE)

DIAG = 0.2
KAPPA_CHOICE = "diag_n_uniform"
FEATURE_CHOICE = "uniform"
fv.plot_v_eff_and_means_w_varying_count_constant(fixed_feature, interval_length,
                                        FEATURE_COUNTS, KAPPA_CHOICE, BETA,
                                        FIG_DIR, FEATURE_CHOICE, diag=DIAG)

DIAG = 0.8
KAPPA_CHOICE = "diag_n_uniform"
FEATURE_CHOICE = "uniform"
fv.plot_v_eff_and_means_w_varying_count_constant(fixed_feature, interval_length,
                                        FEATURE_COUNTS, KAPPA_CHOICE, BETA,
                                        FIG_DIR, FEATURE_CHOICE, diag=DIAG)

KAPPA_CHOICE = "diag_n_uniform_specific"
FEATURE_CHOICE = "uniform"
fv.plot_v_eff_and_means_w_varying_count_constant(fixed_feature, interval_length,
                                        FEATURE_COUNTS, KAPPA_CHOICE, BETA,
                                        FIG_DIR, FEATURE_CHOICE)


# > Varying variance for V centered in v_mid, at v_mid and M fixed.
V_MID = 4
INTERVAL_LENGTHS = np.linspace(0.1, 3)

FEATURE_COUNT = 10

KAPPA_CHOICE = "uniform"
FEATURE_CHOICE = "uniform"
fv.plot_v_eff_and_means_w_varying_std_constant(V_MID, INTERVAL_LENGTHS, FEATURE_COUNT,
                                      KAPPA_CHOICE, BETA, FIG_DIR,
                                      FEATURE_CHOICE)

KAPPA_CHOICE = "random"
FEATURE_CHOICE = "uniform"
fv.plot_v_eff_and_means_w_varying_std_constant(V_MID, INTERVAL_LENGTHS, FEATURE_COUNT,
                                      KAPPA_CHOICE, BETA, FIG_DIR,
                                      FEATURE_CHOICE)

DIAG = 0.8
KAPPA_CHOICE = "diag_n_uniform"
FEATURE_CHOICE = "uniform"
fv.plot_v_eff_and_means_w_varying_std_constant(V_MID, INTERVAL_LENGTHS, FEATURE_COUNT,
                                      KAPPA_CHOICE, BETA, FIG_DIR,
                                      FEATURE_CHOICE, diag=DIAG,
                                      is_legend=False)
DIAG = 0.2
KAPPA_CHOICE = "diag_n_uniform"
FEATURE_CHOICE = "uniform"
fv.plot_v_eff_and_means_w_varying_std_constant(V_MID, INTERVAL_LENGTHS, FEATURE_COUNT,
                                      KAPPA_CHOICE, BETA, FIG_DIR,
                                      FEATURE_CHOICE, diag=DIAG)

KAPPA_CHOICE = "diag_n_uniform_specific"
FEATURE_CHOICE = "uniform"
fv.plot_v_eff_and_means_w_varying_std_constant(V_MID, INTERVAL_LENGTHS, FEATURE_COUNT,
                                      KAPPA_CHOICE, BETA, FIG_DIR,
                                      FEATURE_CHOICE)

KAPPA_CHOICE = "diag_n_uniform_specific"
FEATURE_CHOICE = "random"
fv.plot_v_eff_and_means_w_varying_std_constant(V_MID, INTERVAL_LENGTHS, FEATURE_COUNT,
                                      KAPPA_CHOICE, BETA, FIG_DIR,
                                      FEATURE_CHOICE)


FEATURE_COUNT = 100

KAPPA_CHOICE = "uniform"
FEATURE_CHOICE = "uniform"
fv.plot_v_eff_and_means_w_varying_std_constant(V_MID, INTERVAL_LENGTHS, FEATURE_COUNT,
                                      KAPPA_CHOICE, BETA, FIG_DIR,
                                      FEATURE_CHOICE)

KAPPA_CHOICE = "random"
FEATURE_CHOICE = "uniform"
fv.plot_v_eff_and_means_w_varying_std_constant(V_MID, INTERVAL_LENGTHS, FEATURE_COUNT,
                                      KAPPA_CHOICE, BETA, FIG_DIR,
                                      FEATURE_CHOICE)

DIAG = 0.8
KAPPA_CHOICE = "diag_n_uniform"
FEATURE_CHOICE = "uniform"
fv.plot_v_eff_and_means_w_varying_std_constant(V_MID, INTERVAL_LENGTHS, FEATURE_COUNT,
                                      KAPPA_CHOICE, BETA, FIG_DIR,
                                      FEATURE_CHOICE, diag=DIAG,
                                      is_legend=False)

DIAG = 0.2
KAPPA_CHOICE = "diag_n_uniform"
FEATURE_CHOICE = "uniform"
fv.plot_v_eff_and_means_w_varying_std_constant(V_MID, INTERVAL_LENGTHS, FEATURE_COUNT,
                                      KAPPA_CHOICE, BETA, FIG_DIR,
                                      FEATURE_CHOICE, diag=DIAG)

KAPPA_CHOICE = "diag_n_uniform_specific"
FEATURE_CHOICE = "uniform"
fv.plot_v_eff_and_means_w_varying_std_constant(V_MID, INTERVAL_LENGTHS, FEATURE_COUNT,
                                      KAPPA_CHOICE, BETA, FIG_DIR,
                                      FEATURE_CHOICE)

KAPPA_CHOICE = "diag_n_uniform_specific"
FEATURE_CHOICE = "random"
fv.plot_v_eff_and_means_w_varying_std_constant(V_MID, INTERVAL_LENGTHS, FEATURE_COUNT,
                                      KAPPA_CHOICE, BETA, FIG_DIR,
                                      FEATURE_CHOICE)

DIAGS = [.01, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
fv.plot_v_eff_and_means_w_varying_std_n_correlation(DIAGS, V_MID,
                                                    INTERVAL_LENGTHS,
                                                    FEATURE_COUNT, BETA,
                                                    FIG_DIR)
