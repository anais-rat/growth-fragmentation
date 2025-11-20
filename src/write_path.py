#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov  9 15:29:04 2022

@author: arat

Note: For Arxiv compatible figure names, IS_ARXIV_FRIENDLY_FIG_NAMES must be set
      to True (used in `effective_fitness.py` and `plot.py`).

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
import os
from os.path import join

absolute_path = os.path.abspath(__file__)
current_dir = os.path.dirname(absolute_path)  # Path to `src` directory.
projet_dir = os.path.dirname(current_dir)

FOLDER_DAT = join(projet_dir, "data")
FOLDER_FIG = join(projet_dir, "figures")
FOLDER_VID = join(projet_dir, "videos")

IS_ARXIV_FRIENDLY_FIG_NAMES = True


def separate_directory_from_file(file_path):
    """Return the file name and its directory path from the file path."""
    idx = len(file_path) - 1 - file_path[::-1].find(os.path.sep)
    file_dir = file_path[:idx]
    file_name = file_path[idx + 1 :]
    return file_dir, file_name


if IS_ARXIV_FRIENDLY_FIG_NAMES:

    def remove_special_characters(string, extension="pdf", is_path=True):
        """Return the string without special characters: ., [, ] (other to add).

        Needed for the name of saved figure to submit to arXiv or HAL!
        If the string comes with an extension `extension` the dot in
        '.extension' is conserved
        """
        if is_path:
            file_dir, file_name = separate_directory_from_file(string)
            return join(
                file_dir, remove_special_characters(file_name, extension, is_path=False)
            )
        ext_len = len(extension)
        if string[-ext_len:] != extension:
            extension = ""
        string_ok = string.replace("0.", "p")
        string_ok = string_ok.replace(".", "p")
        string_ok = string_ok.replace("]", "")
        string_ok = string_ok.replace("[", "")
        if extension != "":
            string_ok = string_ok[: -ext_len - 1] + "." + extension
        return string_ok

else:

    def remove_special_characters(string, extension="pdf"):
        """Return identical object."""
        return string


def list_to_strings(list_to_write, decimal_count=None):
    """
    Same as `list_to_string` except that a list of well formatted float
    is returned rather than one string (with float turned to str and separated
    by '-') .

    """
    list_cp = list_to_write.copy()
    if not isinstance(decimal_count, type(None)):
        for i in range(len(list_cp)):
            if decimal_count == 2:
                list_cp[i] = f"{list_cp[i]:3.2f}"
            elif decimal_count == 3:
                list_cp[i] = f"{list_cp[i]:3.3f}"
            elif decimal_count == 4:
                list_cp[i] = f"{list_cp[i]:5.4f}"
            else:
                raise Exception(
                    "Please update `list_to_string' function to "
                    f"allow `decimal_count` to be {decimal_count}"
                )
    return list_cp


def list_to_string(list_to_write, decimal_count=None):
    """
    Parameters
    ----------
    list_to_write : list
        List of strings float or int to convert to a unique string with
        separator `-`.
    decimal_count : int or NoneType
        If None (default value) no change of format, otherwise the elements
        (except the last one if `is_last_int` is True), assumed to be floats,
        are returned in a decimal format, with `decimal_count` decimals after
        point.

    """
    list_cp = list_to_strings(list_to_write, decimal_count)
    # Concatenation of elements of the list in one string.
    string = ""
    for element in list_cp[:-1]:
        string += str(element) + "-"
    string += str(list_cp[-1])
    return string


def is_regular_spacing(arr):
    return np.all(np.diff(arr) - (arr[1] - arr[0]) < 1e-10)


def write_output_path(
    is_conservative,
    features,
    par_beta,
    par_kappa,
    par_ninit,
    par_grids,
    is_tau_constant=False,
    is_longtime=False,
    is_adjoint=False,
    normalization_wtr_feature=False,
    tmax=None,
):
    path = FOLDER_DAT
    if is_adjoint:
        temp = "adjoint_"
    else:
        temp = "direct_"
    if is_longtime:
        tsaved_per_period_count = None
        temp = temp + "longtime_"
    else:
        tsaved_per_period_count = par_grids[3]
    if normalization_wtr_feature:
        temp = temp + "w_normV_"
    if is_tau_constant:
        temp = temp + "tau-constant"
        k_or_dx = "dx"
    else:
        temp = temp + "tau-linear"
        k_or_dx = "k"
    if isinstance(par_kappa[1], type(None)):
        path = join(
            path,
            temp,
            f"kappa{list_to_string(par_kappa[0].flatten(), decimal_count=3)}",
        )
    else:
        path = join(path, temp, "kappa-" + par_kappa[1])
    if is_conservative:
        path = path + "_conservative"
    if len(features) > 4:
        if is_regular_spacing(features):
            path = path + f"_v-reg{features[0]:.2f}-{features[-1]:.2f}-{len(features)}_"
        elif is_regular_spacing(np.log(features)):
            path = path + f"_v-log{features[0]:.2f}-{features[-1]:.2f}-{len(features)}_"
        elif is_regular_spacing(1 / features):
            path = path + f"_v-har{features[0]:.2f}-{features[-1]:.2f}-{len(features)}_"
    else:
        path = path + f"_v{list_to_string(features)}_"
    path = (
        path + k_or_dx + f"{par_grids[0]}_xcount{par_grids[1]}_" + f"alpha{par_beta[0]}"
    )
    if par_beta[1] != 1:
        path = path + f"-{par_beta[1]}"
    path = path + f"_p{par_grids[2]}"
    if not isinstance(tsaved_per_period_count, type(None)):
        path = path + f"_tpp{tsaved_per_period_count}"
    if not isinstance(tmax, type(None)):
        path = path + f"_tmax{tmax}"
    if not os.path.exists(path):  # and make_dir:
        os.makedirs(path)
    path = join(
        path, f"output_init{par_ninit[0][0]}-{par_ninit[0][1]}" + f"-{par_ninit[1]}.npy"
    )
    #'_tsaved_count{tsaved_per_period_count}.npy'
    return path
