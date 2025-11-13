#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec 20 14:20:41 2023

This file lists  all the kernels kappa used in the simulations saved in memory
(`data` folder) and the *unique* name used to describe kappa (more compactly
than by listing its components) in the path to simulation outputs.

Thus, if you need to define a new kappa, the best practice to avoid complicated
but unique naming for simulation outputs is to define your kappa here, and
chose a name that was not already attributed.

@author: arat
"""

import numpy as np


# Generic kernels
# ---------------

# TMP_FTS = r"$\kappa^{{_{{FtS}}}}$"
# TMP_STF = r"$\kappa^{{_{{StF}}}}$"


def kappa_uniform(dim):
    """Return a uniform transition matrix with shape `(dim, dim)` and the name
    under which it should be saved.

    """
    frac = 1 / dim  # (import sympy as sy sy.Rational('1/3'))
    return frac * np.ones((dim, dim)), "uniform"  # , r"$\kappa^{uniform}$"


def kappa_identity(dim):
    """Return the identity matrix with shape `(dim, dim)` and the name under
    which it should be saved.

    """
    return np.identity(dim), "identity"  # , r"$\kappa^{red}$"


def kappa_partial_mixing(probability, is_slowest_mutating, probability_name=None):
    """Return the (2, 2) transition matrix corresponding to a population of
    two species, one stable, the other one able to mutate into the other with
    probability `1 - probability`,  and the name under which the matrix
    should be saved.. If `is_slowest_mutating` is True, the specie that can
    mutating is the slowest one (i.e. with smallest feature), otherwise it is
    the fastest one.

    """
    name = "partial-mixing"
    proba_name = probability_name or str(probability)
    if is_slowest_mutating:
        return (
            np.array([[probability, 1 - probability], [0, 1]]),
            f"{name}-stf{proba_name}",
        )  # , TMP_STF + rf"$( {probability} )$"
    else:
        return (
            np.array([[1, 0], [1 - probability, probability]]),
            f"{name}-fts{proba_name}",
        )  # , TMP_FTS + rf"$( {probability} )$"


def kappa_bimodal(kappa_11, kappa_22):
    """Return the (2, 2) transition matrix w. diagonal (kappa_11, kappa_22)."""
    name = f"irr-bimodal{kappa_11:1.4f}-{kappa_22:1.4f}"
    return np.array([[kappa_11, 1 - kappa_11], [1 - kappa_22, kappa_22]]), name


def compute_threshold_p0(features):
    """Compute the (conjectured) threshold probability p_0 s.t. when kappa is
    kappa_partial_mixing(p, False):
    - p < p_0 : the slowest specie overtakes the fastest in long time.
    - p > p_0 : both species coexist in long time.

    """
    return 2 ** (min(features) / max(features) - 1)


# Specific kernels
# ----------------
# NB: DICT_OF_KERNELS['kernel_unique_name'] = kernel

# Irreducible kernels.
KAPPAS_IRR = dict()
KAPPAS_IRR["3x3"] = (
    np.array([[0.7, 0.2, 0.1], [0.5, 0.4, 0.1], [0.3, 0.3, 0.4]]),
    "irr",
    r"$\kappa^{irr}$",
)
KAPPAS_IRR["2x2"] = (np.array([[0.7, 0.4], [0.5, 0.5]]), "irr", r"$\kappa^{irr}$")


# > Subsection 3.2.
# KAPPAS['s2'] = dict()
# >> Irreducible matrices (3x3).
# frac = 1/3 # (import sympy as sy sy.Rational('1/3'))
# KAPPAS['s2']['uniform'] = np.array([[frac, frac, frac],
#                                     [frac, frac, frac],
#                                     [frac, frac, frac]])
# KAPPAS['s2']['irr'] = np.array([[0.7, 0.2, 0.1],
#                                 [0.5, 0.4, 0.1],
#                                 [0.3, 0.3, 0.4]])
# >> Reducible matrix with no coupling (3x3).
# KAPPAS['s2']['red_no_mixing'] = np.identity(3)

# > Subsection 3.3
# >> Typical cases.
# KAPPAS['s3_1'] = dict()
# # >>> Reducible with partial coupling (2x2).
# PROB_StF = 0.5
# KAPPAS['s3_1']['red_mixing_1'] = np.array([[PROB_StF, 1-PROB_StF],
#                                              [0, 1]])
# PROB_FtS = .2
# KAPPAS['s3_1']['red_mixing_2'] = np.array([[1, 0],
#                                            [1-PROB_FtS, PROB_FtS]])
# KAPPAS['s3_1']['red_mixing_3'] = np.array([[1, 0],
#                                            [PROB_FtS, 1-PROB_FtS]])

# >>> Limit cases to test the critical threshold p_0  (2x2).
# KAPPAS['s3_2'] = dict()
# name = 'red_mixing_limit'
# P_0 = 2 ** (FEATURES['s3_1'][0] / FEATURES['s3_1'][1] - 1)
# KAPPAS['s3_2'][name]  = np.array([[1, 0],
#                                   [1-P_0, P_0]])
# EPS = .05
# KAPPAS['s3_2'][name + f'-{EPS}_l'] = np.array([[1, 0],
#                                                [1-(P_0-EPS), P_0-EPS]])
# KAPPAS['s3_2'][name + f'-{EPS}_r'] = np.array([[1, 0],
#                                                [1-(P_0+EPS), P_0+EPS]])
# # >>> Irreducible to asses numerical diffusivity.
# KAPPAS['s3_2']['red_no_mixing_test'] = np.identity(2)
