#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov  7 12:17:43 2022

@author: arat
"""

import matplotlib.pyplot as plt
import matplotlib.font_manager
import numpy as np
import os
import seaborn as sns

import src.scheme_direct as scheme
import src.parameters.figures_chap1 as par
import src.parameters.kappas as ker
import src.plot as plot
import src.write_path as wp


# ----------
# Parameters
# ----------

# IS_FAST_COMPUTING = {'s2': False, 's3_1': False, 's3_2': False}

IS_SAVED = False

FORMAT = 'manuscript' # 'manuscript' or 'article'.


# Parameter of the figures
# ------------------------

# # NB: comment one `LABELS` entry to prevent its compuation.
# tmp = r"$\kappa^{{_{{FtS}}}}$"
# LABELS = {# Subsection 3.2. (s2).
#           # 'uniform': r"$\kappa^{uniform}$",
#           # 'irr': r"$\kappa^{irr}$",
#           # 'red_no_mixing': r"$\kappa^{red}$",
#           # Subsection 3.3. first set of figures (s3_1).
#           # 'red_mixing_1': rf"$\kappa^{{_{{StF}}}} ( {par.PROB_StF} )$",
#           # 'red_mixing_2': tmp + rf"$( {par.PROB_FtS} )$",
#           # 'red_mixing_3': tmp + rf"$( 1 - {par.PROB_FtS} )$",
#           # Subsection 3.3. second set of figures (s3_2).
#             'red_no_mixing_test': r"$\kappa^{red}$",
#            'red_mixing_limit': tmp + r"$(p_0)$",
#            f'red_mixing_limit-{par.EPS}_l': tmp + r"$(p_0 - \varepsilon)$",
#            f'red_mixing_limit-{par.EPS}_r': tmp + r"$(p_0 + \varepsilon)$"
#           }

# # Location of legend.
# LEG_LOC = {'s2': None, 's3_1': None, 's3_2': None}

# # Title of legend.
# LEG_TIT = {key: None for key in LABELS.keys()}
# LEG_TIT[f'red_mixing_limit-{par.EPS}_l'] = r'$p = p_0 - \varepsilon$'
# LEG_TIT[f'red_mixing_limit-{par.EPS}_r'] = r'$p = p_0 + \varepsilon$'

# # Maximum time plotted (None for the end of computation).
# # > For `plot_evo_lambda_estimates`and `plot_evo_distribution_at_fixed_size`.
# T_MAX = {'s2': 12, 's3_1': 12,'s3_2': 20}
# # > For `plot_evo_distribution_discrete`.
# T_MAX_0 = {'s2': 19, 's3_1': 20, 's3_2': 20}

# IS_CONTINUOUS_TIME_COMPUTED = False
# if IS_CONTINUOUS_TIME_COMPUTED:
#     # .......................................................................
#     # Number of times saved per period (need to be changed for high time
#     # resolution video from images generated with
#     # `plot_evo_distribution_continuous`).
#     TSAVED_PER_PERIOD_COUNT = 50 # par.TSAVED_PER_PERIOD_COUNT
#     # > Maximal time to compute.
#     T_COMPUTE_MAX = {'s2': 6, 's3_1': 12, 's3_2': 12}
#     # > Maximal time to plot (lower than previous).
#     T_PLOT_MAX = {'s2': 4, 's3_1': 8, 's3_2': 8}
#     XTEST_C = par.X_TEST # If not None the images saved are contains 2 figures.
#     # .......................................................................
# else:
#     FIG_COUNT = 8
#     TSAVED_PER_PERIOD_COUNT = par.TSAVED_PER_PERIOD_COUNT
#     T_COMPUTE_MAX = {'s2': None, 's3_1': None, 's3_2': None}
#     # Size of the figure for discrete evolution.
#     if FORMAT == 'manuscript':
#         FIG_SIZE = {'s2': (5.8, 3), 's3_1': (4.8, 2.6), 's3_2': (5.2, 3)}
#     else:
#         FIG_SIZE = {'s2': None, 's3_1': (4.2, 2), 's3_2': None}



# .............................................................................
fig_dir = None

# Global plotting parameters and figure directory.
matplotlib.rcParams.update(matplotlib.rcParamsDefault) # Reset to default.
matplotlib.rcParams['text.latex.preamble'] = r'\usepackage{{amsmath}}'# pmatrix

if FORMAT == 'manuscript':
    if IS_SAVED:
        fig_dir = 'figures_manuscript/chapter_1'
        if (not os.path.exists(fig_dir)):
            os.makedirs(fig_dir)
    sns.set_style("darkgrid")
    sns.set_context("talk", font_scale = 1.3)
    plt.rcParams.update({'axes.facecolor': ".94",
                         'text.usetex': True,
                         'figure.dpi': 600, # Resolution of plottings.
                         'font.family': "sans-serif", # latex-like: 'serif',
                         'font.sans-serif': "Helvetica", # ... 'cmr10'
                         'legend.frameon': True,
                         'legend.framealpha': 1,
                         'legend.facecolor': 'white',
                         'legend.edgecolor': '#EAEAF2',
                         'legend.fancybox': True,
                          # 'legend.title_fontsize': 16.5,
                          # 'legend.fontsize': 16
                          })
    fig_size_discrete = {'s2': None, 's3_1': (4.8, 2), 's3_2': None}
    fig_size = (5, 3)
    bbox_to_anchor = (0, 1.4) # 0.67, -0.02)
elif FORMAT == 'article':
    if IS_SAVED:
        fig_dir = 'figures_article'
        if (not os.path.exists(fig_dir)):
            os.makedirs(fig_dir)
    sns.set_style("ticks") # "ticks") # "darkgrid")whitegrid
    sns.set_context("poster", font_scale = 1) #"paper", font_scale=2)
    plt.rcParams.update({'text.usetex': True,
                         'figure.dpi': 600,
                         'font.family': "sans-serif",
                         'font.sans-serif': ["Helvetica"],
                         'legend.frameon': True,
                         'legend.framealpha': 1,
                         'legend.facecolor': 'white',
                         'legend.edgecolor': 'white',
                         'legend.fancybox': True
                         })
    fig_size_discrete = {'s2': None, 's3_1': (4.2, 2), 's3_2': None}
    fig_size = (5, 3)
    bbox_to_anchor = None
else:
    print("Redine 'Format' correctly")
print(sns.plotting_context())
# .............................................................................


# ------------------------
# Computation and plotting
# ------------------------

# for section_key, kappas in par.KAPPAS.items():

# PARAMETERS
# ----------
FEATURES = np.array([1, 2])

IS_FAST_COMPUTING = True

ALPHA = 2
BETA_CONSTANT = 1

PERIOD_COUNT = 10


# leg_loc = LEG_LOC[section_key]
# t_max = T_MAX[section_key]
# t_max_0 = T_MAX_0[section_key]
# t_max_c = T_COMPUTE_MAX[section_key]
# t_compute_max = T_COMPUTE_MAX[section_key]
# last_key = list(kappas.keys())[-1]
IS_CONSERVATIVE = False
if IS_FAST_COMPUTING:
    PRECISION = par.K_F
    X_COUNT = par.X_COUNT_F
else:
    PRECISION = par.K
    X_COUNT = par.X_COUNT

# # Initialization of the list of outputs associated with `section_key`.
# times = None
# n_sum_evos = []
# labels = []

# Iteration on the kernels defined for `section_key` that appear in LABELS.
# for key, kappa in kappas.items():

# if key in list(LABELS.keys()):

# COMPUTATION
# -----------
# is_normalize_by_feature = False
# if 'red_no_mixing' in key:
#     is_normalize_by_feature = True


PAR_KAPPA = ker.kappa_identity(len(FEATURES))

N_INIT_CHOICE = 'exponential'
N_INIT_1 = 30
N_INIT_2 = 60


PAR_BETA = [ALPHA, BETA_CONSTANT]
PAR_GRIDS = [PRECISION, X_COUNT, PERIOD_COUNT]
PAR_N_INIT = [[N_INIT_1, N_INIT_2], N_INIT_CHOICE]


d = scheme.compute_longtime_approximation(IS_CONSERVATIVE, FEATURES, PAR_BETA,
                                          PAR_KAPPA, PAR_N_INIT, PAR_GRIDS,
                                          is_saved=IS_SAVED)
plot.plot_distribution_old(d[], features, distribution)
    
    # # UPDATE OF SAVED DATA
    # # --------------------
    # if isinstance(times, type(None)):
    #     times = d['times']
    # n_sum_evos.append(d['n_sum_evo'])
    # labels.append(LABELS[key])


    # PLOTTING OF THE CURRENT OUTPUTS
    # -------------------------------
    # name = 'kappa-' + key + f'_V{wp.list_to_string(features)}_' + \
    #     f'alpha{par.ALPHA}' # General name of figures.

    # # Time evolution of the difference between normalized distribution
    # # ................................................................
    # plot.plot_evo_n_tdiff(d['times'], d['n_tdiff_evo'], name=name,
    #                       fig_dir=fig_dir)

    # # Time evolution of lambda estimates
    # # ..................................
    # plot.plot_evo_lambda_estimates(d['times'], features,
    #                                d['lambda_estimates'], name=name,
    #                                t_max=t_max, fig_dir=fig_dir)

    # Time evolution of densities by features
    # .......................................
    # if IS_CONTINUOUS_TIME_COMPUTED:
    #     # Continuous version
    #     X_MAX = 2.5
    #     t_max_p = T_PLOT_MAX[section_key]
    #     # plot.plot_evo_distribution_continuous(d['t_test'], d['sizes'],
    #     #     features, d['n_evo'], x_max=X_MAX, name=name,
    #     #     is_saved=IS_SAVED, t_max=t_max_p, xtest=XTEST_C)
    # else:
    #     print(key, last_key)
    #     # Discrete version
    #     X_MAX = 2.5
    #     plot.plot_evo_distribution_discrete(FIG_COUNT, d['t_test'],
    #         d['sizes'], features, d['n_evo'], x_max=X_MAX, name=name,
    #         t_max=t_max_0, fig_dir=fig_dir, is_legend=key==last_key,
    #         figsize=fig_size_discrete[section_key])

    #     # At size = 'x_test' fixed (n(t, v, x_text) for every v).
    #     leg_loc_new = leg_loc
    #     if  key == f'red_mixing_limit-{par.EPS}_l':
    #         leg_loc_new = "best"
        # # All times plotted.
        # plot.plot_evo_distribution_at_fixed_size(d['times'], features,
        #     d['n_test'], fixed_size=par.X_TEST, name=name,
        #     fig_dir=fig_dir, leg_loc=leg_loc_new, title=LEG_TIT[key],
        #     figsize=FIG_SIZE[section_key])
        # # Only up to t_max.
        # plot.plot_evo_distribution_at_fixed_size(d['times'], features,
        #     d['n_test'], fixed_size=par.X_TEST, name=name,
        #     t_max=t_max, fig_dir=fig_dir, leg_loc=leg_loc_new,
        #     title=LEG_TIT[key], figsize=FIG_SIZE[section_key])

        # # Renormalized densities: m(t,v,x_test) = C(t,v) n(t,v,x_test)
        # #  s.t. \int m(t,v,x) dx = 1 for every (t,v).
        # if is_normalize_by_feature:
        #     plot.plot_evo_distribution_at_fixed_size(d['t_test'],
        #         features, d['n_evo_norm_wrt_v'][:, :, par.X_TEST],
        #         fixed_size=par.X_TEST, name='norm_wrt_v_' + name,
        #         fig_dir=fig_dir, leg_loc="upper right",
        #         is_wo_vmax_plotted=True, figsize=FIG_SIZE[section_key])
        #         # With & wo vmax for clarity.


# PLOTTING OF OUTPUTS COMMON TO ALL COMPUTATIONS
# ----------------------------------------------

# # Time evolution of the log of the total number
# # ..............................................
# if len(n_sum_evos) > 1:
#     name = f'kappa-x{len(n_sum_evos)}' + \
#             f'_V{wp.list_to_string(features)}_alpha{par.ALPHA}'
#     idx_tmax = min([len(n_sum) for n_sum in n_sum_evos])
#     plot.plot_evo_log_n_sum(times[:idx_tmax],
#                             [n_sum[:idx_tmax] for n_sum in n_sum_evos],
#                             labels, name=name, fig_dir=fig_dir,
#                             fig_size=fig_size,
#                             bbox_to_anchor=bbox_to_anchor)
