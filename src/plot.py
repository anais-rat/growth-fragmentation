#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct  6 13:49:34 2022

@author: arat

Note: For arXiv compatible figure names, IS_ARXIV_FRIENDLY_FIG_NAMES must be set
      to True in `write_path.py`.

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

import matplotlib
import matplotlib.pylab as plt
import numpy as np
import os

# import time
from textwrap import wrap
import seaborn as sns

# import sympy as sy


from src.write_path import FOLDER_FIG, FOLDER_VID, remove_special_characters
import src.parameters.figures_properties as fp
import src.effective_fitness as fv


# Parameters of the plots.
# ------------------------

# Global plotting parameters (font for plots and figure auto-sizing).
# matplotlib.rcParams.update(matplotlib.rcParamsDefault)  # Reset to default.
# plt.rcParams.update({# "text.usetex": True,
#                      # "text.latex.unicode": True,
#                      'font.family': "sans-serif",
#                      'font.sans-serif': 'cmss10',
#                      'axes.unicode_minus': False,
#                      'legend.framealpha': 1,
#                      'legend.facecolor': 'white',
#                      'legend.edgecolor': 'white',  # 'F0F0F0', '#EAEAF2'.
#                      # 'legend.fancybox': True,
#                      'legend.frameon': True,
#                      #
#                      "text.latex.preamble": r'\usepackage{{amsmath}}',
#                      # "font.family": "sans-serif",
#                      # "font.sans-serif": ["Helvetica"],
#                      "figure.autolayout": True})

# A few settings for seaborn theme.
# FONT_SCALE = 2
# CONTEXT = "paper" # "notebook"
# sns.set_context(CONTEXT, font_scale=FONT_SCALE)
# sns.set_style("darkgrid", {"axes.facecolor": ".94"})

# Creation of a color map for the lines to plot.
color_map = matplotlib.cm.get_cmap("rocket")
my_palette = [color_map(0.18), color_map(0.48), color_map(0.69)]
sns.set_palette(my_palette)
if __name__ == "__main__":  # Display current palette.
    sns.palplot(sns.color_palette())


def count_to_average(length):
    return int(length / 25)


# Definition of plot functions.
# -----------------------------
# NB: parameters of plotting (labels, line colors...) are defined within the
#    functions but should be rather accessible (see PARAMETERS to locate them).

AXES_FONTSIZE = 15
LABELS = {
    "y_beta": r"Fragmentation rate per unit of size $\beta$",
    "y_lambda": r"Population growth rate",
    "y_n_tdiff": r"Approximation of $\frac{\textrm{d}}{\textrm{d} t} "
    r"\left( \| \bar{n}_t \|_{_{L_1 (\mathcal{S})}} \right)$",
    # Difference of the normalized distribution" + "\n" + \
    # r"between each time",
    "y_density": r"Density",
    "y_log_n": r"Log of the total number",
    "y_phi": r"$\phi(v,\cdot)$",
    "x_time": r"Time",
    "x_size": r"Size",
    "leg_lbd_n": r"$\lambda_{n}$",
    "leg_lbd_g": r"$\lambda_{\gamma}$",
    "leg_lbd_t": r"$\lambda_{\tau}$",
    "leg_lbd_e": r"$\lambda_{error}$",
    "leg_V": r"Features",
}


def normalize_by_v(distribution):
    return distribution / np.transpose([np.sum(distribution, axis=1)])


def write_feature_legend(feature, index=""):
    return rf"$v_{{ {index} }} = {feature}$"


def write_time_legend(time):
    return rf"$t = {time:.2f}$"


def write_pmatrix(matrix):
    """Returns a LaTeX pmatrix (to print proper Tex matrix on plots).

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
        raise ValueError("pmatrix can at most display two dimensions")
    if np.all(matrix == matrix[0, 0]):
        return rf"$\kappa_{{ij}} = {matrix[0, 0]}$"
    lines = str(matrix).replace("[", "").replace("]", "").splitlines()
    rv = [r" " + r" & ".join(ll.split()) for ll in lines]
    rv = r" \\ ".join(rv)
    rv = r"$\kappa = \begin{pmatrix}" + rv + r" \end{pmatrix}$"
    print(rv)
    rv = rv.replace(". ", " ")
    return rv


def write_fig_folder(subfolder, is_video):
    if is_video:
        folder = os.path.join(FOLDER_VID, subfolder)
    else:
        folder = os.path.join(FOLDER_FIG, subfolder)
    if not os.path.exists(folder):
        os.makedirs(folder)
    return folder


def write_fig_path(subfolder, name, is_printed=True, is_video=False):
    path = os.path.join(write_fig_folder(subfolder, is_video), name)
    if is_printed:
        print("Saved at: ", path, "\n")
    return path


def plot_evo_n_tdiff(times, n_tdiff_evo, name="", fig_dir=None):
    plt.figure()
    plt.plot(times[2:], n_tdiff_evo[1:])
    plt.ticklabel_format(style="sci", axis="y", scilimits=(0, 0))
    plt.ylabel(LABELS["y_n_tdiff"], labelpad=8)
    plt.xlabel(LABELS["x_time"], labelpad=6)
    sns.despine()
    # Saving.
    if not isinstance(fig_dir, type(None)):
        path = write_fig_path(fig_dir, f"evo_n_tdiff_{name}.pdf")
        plt.savefig(path, bbox_inches="tight")
    plt.show()


def plot_division_rate(sizes, beta, fig_dir=None):
    plt.figure()
    sizes = np.append([0], sizes)
    plt.plot(sizes, beta(sizes))
    plt.xlabel(LABELS["x_size"], labelpad=6, wrap=True)
    plt.ylabel(LABELS["y_beta"], labelpad=8, wrap=True)
    plt.tight_layout()
    sns.despine()
    # Saving.
    if not isinstance(fig_dir, type(None)):
        path = write_fig_path(fig_dir, "division_rate.pdf")
        plt.savefig(path, bbox_inches="tight")
    plt.show()


def plot_evo_lambda_estimates(
    times, features, lambda_estimates, name, t_max=None, fig_dir=None
):  # , context=1):
    """Print the estimates of lambda at large time and plot the time evolution
    of lambda estimates.

    Parameters
    ----------
    times : ndarray
    lambda_estimates : list
        Outputs of `scheme.compute_evo_approximation`.
    name : string
        Name of the kappa used to compute the data. Used for plot saving path.
    fig_dir : string or None
         Name of the folder to save the figures plotted. If None, the figures
         plotted are not saved.
    """
    # Print the latest estimations and define main plotting PARAMETERS.
    print("\n Kappa " + name + "\n " + "-" * (6 + len(name)))
    print("mean feature = ", np.mean(features))
    count_avg = count_to_average(len(times))
    if len(lambda_estimates) == 2:  # Conservative case.
        print("lambda_n = ", np.mean(lambda_estimates[0][-count_avg:]))
        print("lambda_error = ", np.mean(lambda_estimates[1][-count_avg:]))
        LEGEND = [LABELS["leg_lbd_n"], LABELS["leg_lbd_e"]]
        LINESTYLES = ["-", "-."]
    else:  # Non-conservative case.
        print("lambda_n = ", np.mean(lambda_estimates[0][-count_avg:]))
        print("lambda_gamma = ", np.mean(lambda_estimates[1][-count_avg:]))
        print("lambda_tau = ", np.mean(lambda_estimates[2][-count_avg:]))
        LEGEND = [LABELS["leg_lbd_n"], LABELS["leg_lbd_g"], LABELS["leg_lbd_t"]]
        LINESTYLES = ["-", "--", "-."]
    # Plotting of time evolutions.
    plt.figure(figsize=(5, 3))  # Default (6.4, 4.8)
    estimate_count = len(lambda_estimates)
    for i in range(estimate_count):
        if isinstance(t_max, type(None)):
            tmax_idx = -1
            name_cp = name
        else:
            tmax_idx = np.argmin(times <= t_max) - 1
            name_cp = name + f"_tmax{t_max}"
        plt.plot(
            times[2:tmax_idx],
            lambda_estimates[i][: len(times[2:tmax_idx])],
            LINESTYLES[i],
            label=LEGEND[i],
            color=my_palette[i],
        )
    plt.legend(borderpad=0.2)
    plt.ylabel(LABELS["y_lambda"], labelpad=10)
    plt.xlabel(LABELS["x_time"], labelpad=8)
    sns.despine()
    # Saving.
    if not isinstance(fig_dir, type(None)):
        path = write_fig_path(fig_dir, f"evo_lambda_estimates_{name_cp}.pdf")
        plt.savefig(path, bbox_inches="tight")
    plt.show()
    # sns.set_context(CONTEXT, font_scale=FONT_SCALE)


def plot_evo_log_n_sum(
    times,
    n_sum_evos,
    labels,
    fig_dir=None,
    name="",
    fig_size=(6.4, 4.8),
    bbox_to_anchor=None,
):
    """Plot the time evolution of the log of the number of individuals.

    Parameters
    ----------
    times : ndarray
        1D array (length 't_count') of times at which estimates were computed.
    n_evos : ndarray
        Array of the corresponding number of individual at times 'times'.
    fig_dir : string or None
         Name of the folder to save the figures plotted. If None, the figures
         plotted are not saved.
    """
    colors = sns.color_palette("Greys", len(labels))  # rocket
    LINESTYLES = ["-", "-."] + ["--"] * (len(n_sum_evos) - 2)

    plt.figure(figsize=fig_size)  # Default (6.4, 4.8)
    plt.tight_layout()
    for i in range(len(n_sum_evos)):
        plt.plot(
            times[1:],
            np.log(n_sum_evos[i][1:]),
            LINESTYLES[i],
            label=labels[i],
            color=colors[i],
        )
    if isinstance(bbox_to_anchor, type(None)) or len(labels) == 2:
        plt.legend()
    else:
        plt.legend(bbox_to_anchor=bbox_to_anchor, loc="upper left")
    plt.ylabel(LABELS["y_log_n"], labelpad=10, wrap=True)
    plt.xlabel(LABELS["x_time"], labelpad=8)
    sns.despine()
    if not isinstance(fig_dir, type(None)):
        path = write_fig_path(fig_dir, f"evo_log_n_{name}.pdf")
        plt.savefig(path, bbox_inches="tight")
    plt.show()
    # sns.set_context(CONTEXT, font_scale=FONT_SCALE)


def plot_distribution_old(
    sizes,
    features,
    distribution,
    fig_name="",
    fig_dir=None,
    kappa=None,
    y_key_label="y_density",
    is_normalized=False,
    is_normalized_by_v=False,
):
    """Plot the initial distribution and save it if `fig_dir` is not None.

    Distrib. given by `distribution` at sizes `sizes` and traits `features`.
    """
    feature_count = len(features)
    COLORS = sns.color_palette("rocket", feature_count)
    LINESTYLES = ["-", "--", "-."] + ["-"] * (feature_count - 2)
    plt.figure(figsize=(5.8, 2.8))  # (6.4, 3.9)) # Default (6.4, 4.8)
    dist = np.copy(distribution)
    if is_normalized:
        dist = distribution / np.sum(distribution)
    if is_normalized_by_v:
        dist = distribution / np.transpose([np.sum(distribution, axis=1)])
    for feature_idx in range(feature_count):
        plt.plot(
            sizes,
            dist[feature_idx],
            LINESTYLES[feature_idx],
            label=write_feature_legend(features[feature_idx]),
            color=COLORS[feature_idx],
        )
    if not isinstance(kappa, type(None)):
        ax = plt.gca()
        if isinstance(kappa, str):
            kappa_str = kappa
        else:
            kappa_str = write_pmatrix(kappa)
        anchored_text = matplotlib.offsetbox.AnchoredText(
            # prop={'fontsize':'small'})
            kappa_str,
            loc="upper left",
            frameon=False,
            pad=0.2,
        )
        ax.add_artist(anchored_text)
        # plt.text(.05, 0.8, loc="upper left", write_pmatrix(kappa),
        #          transform=ax.transAxes)
    plt.ylabel(LABELS[y_key_label], labelpad=8)
    plt.xlabel(LABELS["x_size"], labelpad=6)
    plt.legend(loc="lower right")
    sns.despine()
    if not isinstance(fig_dir, type(None)):
        path = write_fig_path(fig_dir, f"{fig_name}.pdf")
        plt.savefig(path, bbox_inches="tight")
    plt.show()


def plot_distribution(
    sizes,
    features,
    distribution,
    fig_name="",
    fig_dir=None,
    kappa=None,
    y_key_label="y_density",
    is_normalized=False,
    is_normalized_by_v=False,
    labels=None,
    xmax=None,
    palette="viridis",
    fig_format="pdf",
):
    """Plot the initial distribution and save it if `fig_dir` is not None.

    Distrib. given by `distribution` at sizes `sizes` and traits `features`.
    """
    feature_count = len(features)
    plot_count = len(distribution)
    COLORS = sns.color_palette(palette, plot_count)
    LINESTYLES = ["-", "--", "-.", ":"] + ["-"] * (plot_count - 3)

    if labels is None:
        labels = [
            write_feature_legend(features[i], index=i + 1) for i in range(feature_count)
        ]
        if feature_count != plot_count:
            raise Exception("Define labels")
    y_key_label_cp = y_key_label

    if xmax is None:
        idx_max = len(sizes) - 1
    else:
        idx_max = np.where(sizes <= xmax)[0][-1]

    plt.figure()
    ax = plt.gca()
    dist = np.copy(distribution)
    sns.despine()
    if is_normalized:
        dist = distribution / np.sum(distribution)
        y_key_label_cp = None
    if is_normalized_by_v:
        y_key_label_cp = None
        dist = normalize_by_v(distribution)

    for i in np.arange(plot_count)[::None]:
        plt.plot(
            sizes[: idx_max + 1],
            dist[i][: idx_max + 1],
            LINESTYLES[i],
            color=COLORS[i],
            label=labels[i],
        )
        # Fill the area under each line
        plt.fill_between(
            sizes[: idx_max + 1], dist[i][: idx_max + 1], color=COLORS[i], alpha=0.2
        )

    if kappa is not None:
        ax = plt.gca()
        kappa_str = kappa if isinstance(kappa, str) else write_pmatrix(kappa)
        anchored_text = matplotlib.offsetbox.AnchoredText(
            kappa_str, loc="upper left", frameon=False, pad=0.2
        )
        ax.add_artist(anchored_text)

    if y_key_label_cp is None:
        ax.get_yaxis().set_visible(False)
        sns.despine(left=True)
    else:
        plt.ylabel(LABELS[y_key_label_cp], labelpad=8)
    plt.xlabel(LABELS["x_size"], labelpad=6)
    fv.format_ticks_2D(format_x="%.0f", format_y="%.3f")
    plt.legend(fontsize="large")  # (borderaxespad=2)

    if fig_dir is not None:
        path = write_fig_path(fig_dir, fig_name)
        plt.savefig(
            remove_special_characters(path) + "." + fig_format,
            bbox_inches="tight",
            format=fig_format,
        )
    plt.show()


def plot_evo_distribution_continuous(
    times,
    sizes,
    features,
    n_evo,
    x_max=None,
    name="",
    is_saved=False,
    fig_dir=FOLDER_VID,
    t_max=None,
    xtest=None,
):
    """Plot the successive approximations of the size density (for each
    feature) of times 'times', and save them if 'is_saved' is True.

    NB: If `xtest` is None only the evolution of the whole distribution is
        saved, otherwise the evolution of the dansities at x=xtest are also
        plotted in the same figure.

    """
    t_max = t_max or times[-1]
    if isinstance(x_max, type(None)):
        xmax_idx = -1
    else:
        xmax_idx = np.argmin(sizes <= x_max) - 1

    n_max = np.max(n_evo)
    colors = sns.color_palette("rocket", len(features))

    fig_subfolder = fig_dir + f"evo2_n_{name}tmax{t_max:.1f}"

    if isinstance(xtest, type(None)):
        # One graph only.
        plt.figure()
        for time_idx in range(sum(times <= t_max)):
            plt.ylim(0, n_max)
            for feature_idx in range(len(features)):
                plt.plot(
                    sizes[:xmax_idx],
                    n_evo[time_idx, feature_idx, :xmax_idx],
                    color=colors[feature_idx],
                    label=write_feature_legend(features[feature_idx]),
                )
            plt.ylabel(LABELS["y_density"], labelpad=8)
            plt.xlabel(LABELS["x_size"], labelpad=6)
            plt.title(write_time_legend(times[time_idx]))
            plt.legend()
            sns.despine()
            if is_saved:
                path = write_fig_path(
                    fig_dir, f"{time_idx:03d}.png", is_video=True, is_printed=False
                )
                plt.savefig(path, dpi=fp.DPI_VIDEO)
            plt.pause(1e-20)
    else:
        xtest_idx = len(sizes[sizes <= xtest]) - 1
        times_to_plot = times[times <= t_max]
        time_count = len(times_to_plot)
        ntest = n_evo[:time_count, :, xtest_idx]
        ntest_max = np.max(ntest)
        ntest_tmp = np.nan * ntest
        # Two graphs.
        W, H = (6, 6.5)
        for time_idx in range(time_count):
            path = write_fig_path(
                fig_subfolder, f"{time_idx:03d}.png", is_video=True, is_printed=False
            )
            ntest_tmp[time_idx] = ntest[time_idx]
            if os.path.exists(path):
                if time_idx == 0:
                    print(f"See folder '{fig_subfolder}' for the figures")
                pass
            else:
                fig, axes = plt.subplots(
                    2,
                    1,
                    sharex=False,
                    sharey=False,
                    figsize=(W, H),
                    gridspec_kw={"height_ratios": [1.8, 1]},
                )
                axes[0].set_ylim(0, n_max)
                axes[1].set_ylim(0, 1.1 * ntest_max)
                axes[1].set_xlim(0, t_max)
                for feature_idx in range(len(features)):
                    axes[0].plot(
                        sizes[:xmax_idx],
                        n_evo[time_idx, feature_idx, :xmax_idx],
                        color=colors[feature_idx],
                        linewidth=2,
                        label=write_feature_legend(features[feature_idx]),
                    )
                    axes[1].plot(
                        times_to_plot,
                        ntest_tmp[:, feature_idx],
                        color=colors[feature_idx],
                        linewidth=2,
                    )
                axes[0].set_title(write_time_legend(times[time_idx]), pad=12)
                axes[0].set_ylabel(LABELS["y_density"], labelpad=8)
                axes[0].set_xlabel(LABELS["x_size"], labelpad=6)
                # axes[1].set_title(r'~', pad=0) # Add space between figures.
                axes[1].set_ylabel(
                    LABELS["y_density"] + rf" at $x={xtest:.2f}$", labelpad=8
                )
                axes[1].set_xlabel(LABELS["x_time"], labelpad=6)
                axes[0].legend()
                if is_saved:
                    fig.savefig(path, dpi=fp.DPI_VIDEO, bbox_inches="tight")
                plt.close()
                # time.sleep(0.001)
    # sns.set_context(CONTEXT, font_scale=FONT_SCALE)


def plot_evo_distribution_discrete(
    fig_count,
    times,
    sizes,
    features,
    n_evo,
    x_max=None,
    name="",
    fig_dir=None,
    time_idxs=None,
    t_max=None,
    is_legend=True,
    figsize=None,
):
    """Plot in one unique figure, the size densities (up to the size of index
    'xmax_idx' and for all features) at 'fig_count' different times of 'times'.
    For appropriate legend 'is_mixing' is True if data come from a mixed
    population.

    """
    # Definition of main parameters.
    feature_count = len(features)
    time_count = len(times)
    if not isinstance(t_max, type(None)):
        tmax_idx = np.argmin(times <= min(t_max, times[-2]))
        time_idxs = np.geomspace(1, tmax_idx, fig_count, dtype=int)
        time_idxs[0] = 0
    if isinstance(time_idxs, type(None)):
        time_idxs = np.geomspace(1, time_count - 15, fig_count, dtype=int)
        time_idxs[0] = 0
    if isinstance(x_max, type(None)):
        xmax_idx = -1
    else:
        xmax_idx = np.argmin(sizes <= x_max) - 1

    LEGENDS = [write_time_legend(time) for time in times[time_idxs]]
    LINESTYLES = ["-", "--", "-."] + ["-"] * (feature_count - 2)
    if feature_count == 2:
        COLORS = sns.color_palette("viridis", 2)
    else:
        COLORS = sns.color_palette("rocket", len(features))

    # Plot `fig_count` subplots in column, with 'feature_count' lines each.
    if isinstance(figsize, type(None)):
        figsize = 6.8, 2.375 * fig_count
    else:
        figsize = (figsize[0], figsize[1] * fig_count)
    fig, axes = plt.subplots(fig_count, 1, sharex=True, sharey=False, figsize=figsize)
    for i in range(0, fig_count):  # Iteration on subplots / times to plot.
        for feature_idx in range(feature_count):  # A line per feature.
            axes[i].plot(
                sizes[:xmax_idx],
                n_evo[time_idxs[i], feature_idx, :xmax_idx],
                LINESTYLES[feature_idx],
                label=write_feature_legend(features[feature_idx]),
                color=COLORS[feature_idx],
            )
        if is_legend and i == 0:
            pass
        else:
            axes[i].text(
                0.96,
                0.94,
                LEGENDS[i],
                transform=axes[i].transAxes,
                va="top",
                ha="right",
            )
    fig.add_subplot(111, frameon=False)
    plt.tick_params(
        labelcolor="none",
        which="both",
        top=False,
        bottom=False,
        left=False,
        right=False,
    )
    plt.grid(False)
    plt.xlabel(LABELS["x_size"], labelpad=10)
    plt.ylabel(LABELS["y_density"], labelpad=8)
    if is_legend:
        axes[0].legend(loc="lower right")
    sns.despine()
    if not isinstance(fig_dir, type(None)):
        path = write_fig_path(fig_dir, f"evo_n_{name}.pdf")
        plt.savefig(path, bbox_inches="tight")
    plt.show()


def plot_evo_distribution_discrete_CFL1(
    fig_count, times, sizes, features, n_evo, x_max=None, name="", fig_dir=None
):
    """Plot in one unique figure, the size densities (up to the size of index
    'xmax_idx' and for all features) at 'fig_count' different times of 'times'.
    For appropriate legend 'is_mixing' is True if data come from a mixed
    population.

    """
    # Definition of main parameters.
    feature_count = len(features)
    time_count = len(times)
    time_idxs = np.geomspace(1, time_count - 15, fig_count, dtype=int)
    time_idxs[0] = 0
    if isinstance(x_max, type(None)):
        xmax_idx = -1
    else:
        xmax_idx = np.max(np.argmin(sizes <= x_max, 1)) - 1

    LEGENDS = [write_time_legend(time) for time in times[time_idxs]]
    LINESTYLES = ["-", "--", "-."] + ["-"] * (feature_count - 2)
    COLORS = sns.color_palette("rocket", feature_count)

    # Plot `fig_count` subplots in column, with 'feature_count' lines each.
    w, h = 7, 2.375 * fig_count
    fig, axes = plt.subplots(fig_count, 1, sharex=True, sharey=False, figsize=(w, h))
    for i in range(0, fig_count):  # Iteration on subplots / times to plot.
        for feature_idx in range(feature_count):  # A line per feature.
            axes[i].plot(
                sizes[i, :xmax_idx],
                n_evo[time_idxs[i], feature_idx, :xmax_idx],
                LINESTYLES[feature_idx],
                color=COLORS[feature_idx],
                label=write_feature_legend(features[feature_idx]),
            )
        axes[i].text(
            0.96, 0.94, LEGENDS[i], transform=axes[i].transAxes, va="top", ha="right"
        )
    axes[0].legend(loc="center right")
    fig.supxlabel(LABELS["x_size"], y=0.01)
    fig.supylabel(LABELS["y_density"], x=0.03)
    sns.despine()
    if not isinstance(fig_dir, type(None)):
        path = write_fig_path(fig_dir, f"evo_n_{name}.pdf")
        plt.savefig(path, bbox_inches="tight")
    plt.show()


def plot_evo_distribution_at_fixed_size(
    times,
    features,
    n_along_fixed_x,
    fixed_size,
    name="",
    fig_dir=None,
    t_max=None,
    leg_loc="lower right",
    title=None,
    is_wo_vmax_plotted=False,
    figsize=None,
):
    if isinstance(t_max, type(None)):
        tmax_idx = -1
        name_add = ""
    else:
        tmax_idx = np.argmin(times <= t_max) - 1
        name_add = f"_tmax{t_max}"
        print(t_max, times[tmax_idx])

    feature_count = len(features)
    if feature_count == 2:
        palette = "viridis"
    else:
        palette = "rocket"
    COLORS = sns.color_palette(palette, feature_count)

    if isinstance(figsize, type(None)):
        figsize = (6.8, 3)  # Default (6.4, 4.8)
    plt.figure(figsize=figsize)
    for feature_idx in range(len(features))[::-1]:
        plt.plot(
            times[:tmax_idx],
            n_along_fixed_x[:tmax_idx, feature_idx],
            color=COLORS[feature_idx],
            label=write_feature_legend(features[feature_idx]),
        )
    plt.ylabel(LABELS["y_density"] + rf" at $x={fixed_size:.2f}$", labelpad=25)
    plt.xlabel(LABELS["x_time"], labelpad=9)
    plt.title(title, pad=8, loc="center")
    ax = plt.gca()
    handles, labels = ax.get_legend_handles_labels()
    if not isinstance(leg_loc, type(None)):
        plt.legend(reversed(handles), reversed(labels), loc=leg_loc)
    sns.despine()
    if not isinstance(fig_dir, type(None)):
        path = write_fig_path(
            fig_dir, f"evo_n_fixed_x{fixed_size:.2f}_" + name + name_add + ".pdf"
        )
        plt.savefig(path, bbox_inches="tight")
    plt.show()

    if is_wo_vmax_plotted:
        plt.figure(figsize=(6.8, 3))  # Default (6.4, 4.8)
        for feature_idx in range(len(features[:-1]))[::-1]:
            plt.plot(
                times[:tmax_idx],
                n_along_fixed_x[:tmax_idx, feature_idx],
                color=COLORS[feature_idx],
                label=write_feature_legend(features[feature_idx]),
            )
        plt.ylabel("Rescaled density\n" + rf"at $x={fixed_size:.2f}$", labelpad=25)
        plt.xlabel(LABELS["x_time"], labelpad=9)
        plt.title(title, pad=8, loc="right")
        ax = plt.gca()
        handles, labels = ax.get_legend_handles_labels()
        plt.legend(reversed(handles), reversed(labels), loc=leg_loc)
        sns.despine()
        if not isinstance(fig_dir, type(None)):
            fig_name = f"evo_n_fixed_x{fixed_size:.2f}_wo_vmax_{name}"
            path = write_fig_path(fig_dir, f"{fig_name}{name_add}.pdf")
            plt.savefig(path, bbox_inches="tight")
    plt.show()
