"""Validation-only figures comparing the MSE and fixed-weight EMD experiments.

The caller supplies verified, aligned arrays and owns numeric reporting. This
module draws full-event-normalized spectra and leakage-aware response matrices.
It does not select a model or evaluate the independent test set.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from plot_ttbar_four import OBSERVABLES, save_figure, step_band, style
from ttbar_metrics import paired_histogram, response_matrix
import matplotlib.pyplot as plt
from matplotlib.ticker import AutoMinorLocator, MaxNLocator, ScalarFormatter


METHODS = ("MSE", "MSE + 0.1 EMD")
COLORS = {"MSE": "#e33434", "MSE + 0.1 EMD": "#2864b7"}


def _sample_title(n_events):
    return rf"$t\bar{{t}}(2\ell)$ MC validation sample: $N={n_events:,}$"


def _paths(output_dir, stem):
    return [output_dir / f"{stem}.{extension}" for extension in ("png", "svg")]


def _distribution_comparison(histograms, names, n_events, output_dir, stem, *, include_notes=True):
    columns = len(names)
    fig = plt.figure(figsize=(3.35 * columns + .25 if columns > 1 else 4.65, 4.25))
    grid = fig.add_gridspec(2, columns, height_ratios=[2.5, 1], hspace=.045, wspace=.16)
    handles = None
    for column, name in enumerate(names):
        reference = histograms[METHODS[0]][name]
        edges = reference["edges"]
        ax = fig.add_subplot(grid[0, column])
        ratio_ax = fig.add_subplot(grid[1, column], sharex=ax)
        ax.stairs(reference["truth_density"], edges, fill=True, color="#e9e9e9",
                  linewidth=0, label="MC parton truth")
        ax.stairs(reference["truth_density"], edges, color="#505050", linewidth=.65)
        upper_ratios = []
        for method in METHODS:
            item, color = histograms[method][name], COLORS[method]
            pred, pred_se = item["pred_density"], item["pred_density_se"]
            step_band(ax, edges, np.maximum(0, pred - pred_se), pred + pred_se, color, .17)
            ax.stairs(pred, edges, color=color, linewidth=.9, label=method)
            ratio, ratio_se = item["ratio"], item["ratio_se"]
            step_band(ratio_ax, edges, ratio - ratio_se, ratio + ratio_se, color, .17)
            ratio_ax.stairs(ratio, edges, color=color, linewidth=.9)
            upper = ratio + ratio_se
            upper_ratios.extend(upper[np.isfinite(upper)])
        handles = ax.get_legend_handles_labels()
        ax.set_ylim(bottom=0)
        ax.set_xlim(edges[0], edges[-1])
        ax.tick_params(labelbottom=False)
        ax.yaxis.set_major_locator(MaxNLocator(4, prune="upper"))
        formatter = ScalarFormatter(useMathText=True)
        formatter.set_powerlimits((-2, 3))
        ax.yaxis.set_major_formatter(formatter)
        ratio_ax.axhline(1, color="#606060", linestyle="--", linewidth=.65)
        ratio_ax.set_ylim(0, max(1.4, max(upper_ratios) * 1.07) if upper_ratios else 1.4)
        ratio_ax.yaxis.set_major_locator(MaxNLocator(3))
        ratio_ax.set_xlabel(OBSERVABLES[name]["label"])
        for axis in (ax, ratio_ax):
            axis.grid(color="#a0a0a0", linewidth=.4, alpha=.75)
            axis.set_axisbelow(True)
            axis.xaxis.set_minor_locator(AutoMinorLocator(5))
            axis.yaxis.set_minor_locator(AutoMinorLocator(2))
        if name == "ttbar_mass":
            ratio_ax.set_xticks(np.arange(300, 1051, 150))
        elif name == "delta_abs_y":
            ratio_ax.set_xticks([-2, -1, 0, 1, 2])
        else:
            ratio_ax.set_xticks([-1, -.5, 0, .5, 1])
        if column == 0:
            ax.set_ylabel("Normalized density")
            ratio_ax.set_ylabel("Pred. / Truth")
    fig.subplots_adjust(left=.085 if columns > 1 else .165, right=.985,
                        top=.82, bottom=.265)
    fig.legend(*handles, loc="upper center", ncol=3, bbox_to_anchor=(.52, 1.005),
               handlelength=2.2, columnspacing=1.7 if columns > 1 else 1.1,
               fontsize=9 if columns > 1 else 8.4)
    fig.text(.52, .90, _sample_title(n_events), ha="center", fontsize=9)
    footer = ("All-event normalization; no prediction clipping. Shading: approximate 1-sigma statistical uncertainty.\n"
              "Ratio uncertainty accounts for paired events. Zero-truth bins are omitted; range exclusions are reported separately.")
    if columns == 1:
        footer = ("All-event normalization; no prediction clipping.\n"
                  "Shading: approximate 1-sigma statistical uncertainty.\n"
                  "Ratio uncertainty accounts for paired events.")
    if include_notes:
        fig.text(.52, .025, footer, ha="center", va="bottom", fontsize=7.3, linespacing=1.5)
    save_figure(fig, output_dir, stem)
    return _paths(output_dir, stem)


def _matrix_comparison(responses, names, n_events, output_dir, stem, *, include_notes=True):
    fig, axes = plt.subplots(2, 2, figsize=(7.45, 6.95))
    fig.subplots_adjust(left=.105, right=.865, bottom=.175, top=.865,
                        hspace=.45, wspace=.50)
    im = None
    for row, name in enumerate(names):
        description = OBSERVABLES[name]
        for column, method in enumerate(METHODS):
            ax = axes[row, column]
            response = responses[method][name]
            probabilities, edges = response["central_probs"], response["edges"]
            n_bins = len(edges) - 1
            im = ax.imshow(np.ma.masked_invalid(probabilities), origin="lower",
                           interpolation="none", extent=(0, n_bins, 0, n_bins),
                           cmap="Reds", vmin=0, vmax=1, aspect="equal")
            for truth_row in range(n_bins):
                for pred_col in range(n_bins):
                    value = probabilities[truth_row, pred_col]
                    if np.isfinite(value) and value >= .05:
                        ax.text(pred_col + .5, truth_row + .5, f"{value:.2f}",
                                ha="center", va="center", fontsize=8,
                                color="white" if value >= .48 else "#202020")
            labels = [f"{value:g}" for value in edges]
            ax.set_xticks(np.arange(n_bins + 1), labels, rotation=45, ha="right")
            ax.set_yticks(np.arange(n_bins + 1), labels)
            ax.grid(color="#606060", linewidth=.35, alpha=.8)
            ax.tick_params(which="both", length=2)
            ax.set_xlabel("Predicted " + description["label"], labelpad=2, fontsize=9)
            ax.set_ylabel("Parton truth " + description["label"], labelpad=3, fontsize=9)
            if row == 0:
                ax.set_title(method, pad=7, fontsize=10)
    color_ax = fig.add_axes([.91, .245, .022, .55])
    colorbar = fig.colorbar(im, cax=color_ax)
    colorbar.set_label("Probability", fontsize=9)
    colorbar.ax.tick_params(labelsize=8)
    fig.text(.49, .973, "Migration matrices: MSE and MSE + 0.1 EMD", ha="center", fontsize=11)
    fig.text(.49, .940, _sample_title(n_events), ha="center", fontsize=9)
    if tuple(names) == ("ttbar_mass", "cos_phi"):
        bin_note = "Mass and cos(phi) bins follow Fig. 6; only probabilities >= 0.05 are annotated."
    else:
        bin_note = "These bins are chosen for this baseline; only probabilities >= 0.05 are annotated."
    if include_notes:
        fig.text(.49, .035,
                 "Truth on rows; predictions on columns. All predictions, including under/overflow, enter each row denominator.\n"
                 "Only the central bins are shown; the saved numeric report retains the flows.\n" + bin_note,
                 ha="center", va="bottom", fontsize=7.1, linespacing=1.5)
    save_figure(fig, output_dir, stem)
    return _paths(output_dir, stem)


def make_comparison_figures(truth, predictions, output_dir, *, include_notes=True,
                            combine_distributions=False):
    """Draw both methods on the same verified, ordered validation events.

    truth maps target name to a one-dimensional array. predictions maps the
    exact labels ``MSE`` and ``MSE + 0.1 EMD`` to equivalent target dictionaries.
    The caller verifies split provenance and event alignment. This function
    checks array lengths through the numerical helpers and never clips data.
    Returns PNG/SVG paths for four figures, or three figures when
    combine_distributions=True puts all four observables in one distribution
    figure. Set include_notes=False to omit bottom explanatory notes.
    """
    output_dir = Path(output_dir)
    if set(predictions) != set(METHODS):
        raise ValueError(f"Expected exactly these method labels: {METHODS}")
    missing = set(OBSERVABLES) - set(truth)
    if missing:
        raise ValueError(f"Missing truth observables: {sorted(missing)}")
    n_events = len(truth[next(iter(OBSERVABLES))])
    histograms, responses = {}, {}
    for method in METHODS:
        histograms[method], responses[method] = {}, {}
        for name, description in OBSERVABLES.items():
            if len(truth[name]) != n_events:
                raise ValueError("All truth observables must contain the same number of events")
            histograms[method][name] = paired_histogram(
                truth[name], predictions[method][name], description["distribution_edges"])
            responses[method][name] = response_matrix(
                truth[name], predictions[method][name], description["response_edges"])
    output_dir.mkdir(parents=True, exist_ok=True)
    style()
    if combine_distributions:
        paths = _distribution_comparison(histograms, list(OBSERVABLES), n_events,
                                         output_dir, "distributions_four_comparison",
                                         include_notes=include_notes)
    else:
        paths = _distribution_comparison(histograms, list(OBSERVABLES)[:3], n_events,
                                         output_dir, "distributions_three_comparison",
                                         include_notes=include_notes)
        paths.extend(_distribution_comparison(histograms, ["cos_phi"], n_events,
                                             output_dir, "distribution_cos_phi_comparison",
                                             include_notes=include_notes))
    paths.extend(_matrix_comparison(responses, ["ttbar_mass", "cos_phi"], n_events,
                                   output_dir, "migration_matrices_comparison", include_notes=include_notes))
    paths.extend(_matrix_comparison(responses, ["delta_abs_y", "cos_K_plus"], n_events,
                                   output_dir, "migration_matrices_other_comparison", include_notes=include_notes))
    return paths
