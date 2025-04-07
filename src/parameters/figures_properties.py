#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct  2 11:16:53 2024

@author: anais

Global parameters of the figures (which are common to all plots or should not
be changed locally) are defined below.

Other global parameters are also defined at the beginning of
`src/.../plot.py` scripts ; more versatile ones are defined at the
beginning of `main/.../plot.py` scripts.

"""


# Classical parameters for plots
# ------------------------------

PERCENT = 95  # Percentage of data to include in the percentile interval.
# ------------------------------
P_DOWN = (100 - PERCENT) / 2  # First percentile to compute.
P_UP = 100 - (100 - PERCENT) / 2  # Second percentile.
# ------------------------------

DPI = 300  # Resolution of plots.
DPI_VIDEO = 300  # Resolution of plots for video.
ALPHA = 0.25  # Transparency to fill gaps btw extremum values, std percentiles.


# Global plotting parameters
# --------------------------

PAR_RC_UPDATE_MANUSCRIPT = {
    'axes.facecolor': ".94",
    'text.latex.preamble': r'\usepackage{amsfonts, dsfont, amsmath}',
    'figure.dpi': DPI,
    'legend.frameon': True,
    'legend.framealpha': 1,
    'legend.facecolor': 'white',
    'legend.edgecolor': 'white',  # '#EAEAF2'.
    'legend.fancybox': True,
    'text.usetex': True,  # Removed to make `plt.ylabel(wrap=True)` work.
    # Font changed consequently.
    'font.family': "sans-serif",  # latex-like: 'serif'
    'font.sans-serif': "Helvetica",  # ... 'cmr10'
    # 'font.family': "sans-serif",
    # 'font.sans-serif': 'cmss10',
    # 'axes.unicode_minus': False
    }

PAR_RC_UPDATE_ARTICLE = {
    'text.usetex': True,
    'figure.dpi': DPI,
    'text.latex.preamble': r'\usepackage{dsfont, amsmath}',  # amsmath: pmatrix
    'font.family': "sans-serif",
    'font.sans-serif': ["Helvetica"],
    'legend.frameon': True,
    'legend.framealpha': 1,
    'legend.facecolor': 'white',
    'legend.edgecolor': 'white',
    'legend.fancybox': True
    }
