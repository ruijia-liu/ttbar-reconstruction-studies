#!/usr/bin/env python3
"""Evaluate held-out HepSetReg predictions and draw paper-style scientific figures.

The data are MC predictions, not measured collision data. Validation results
are used for model selection; only the test split is labelled independent.
The gray reference is parton truth, not Fig. 5's detector-level ideal
reconstruction. Cosines are never clipped. PNG and editable SVG figures are
saved together with metrics. Model labels and checkpoint provenance are
supplied explicitly, without claiming the VyPER architecture.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import AutoMinorLocator, MaxNLocator, ScalarFormatter
import numpy as np
import pyarrow.parquet as pq

from ttbar_metrics import regression_metrics, response_matrix, paired_histogram

OBSERVABLES = {
    "ttbar_mass": {
        "label": r"$m_{t\bar{t}}$ [GeV]", "unit": "GeV",
        "distribution_edges": np.linspace(300, 1050, 31),
        "response_edges": np.array([300, 380, 440, 500, 750, 2000]),
        "response_bin_source": "VyPER paper Fig. 6",
    },
    "delta_abs_y": {
        "label": r"$|y_t|-|y_{\bar{t}}|$", "unit": "dimensionless",
        "distribution_edges": np.linspace(-2.25, 2.25, 19),
        "response_edges": np.array([-2, -1, 0, 1, 2]),
        "response_bin_source": "chosen for this HepSetReg baseline",
    },
    "cos_K_plus": {
        "label": r"$\cos\theta_k^{+}$", "unit": "dimensionless",
        "distribution_edges": np.linspace(-1, 1, 21),
        "response_edges": np.array([-1, -.5, 0, .5, 1]),
        "response_bin_source": "chosen for this HepSetReg baseline",
    },
    "cos_phi": {
        "label": r"$\cos\varphi$", "unit": "dimensionless",
        "distribution_edges": np.linspace(-1, 1, 21),
        "response_edges": np.array([-1, -.5, 0, .5, 1]),
        "response_bin_source": "VyPER paper Fig. 6",
    },
}
RED = "#e33434"
SPLIT_ROWS = {"test": 261557, "val": 260190}
DEFAULT_RUN_NAME = "ttbar4_mse_seed42_v2"


def empirical_wasserstein1(truth, pred):
    """Equal-weight one-dimensional Wasserstein-1 distance in raw target units.

    For equally sized empirical samples this is mean(abs(sort(truth) -
    sort(pred))). Unlike residual MAE, it compares marginal distributions
    and does not assess event-by-event reconstruction.
    """
    truth, pred = np.asarray(truth, dtype=float), np.asarray(pred, dtype=float)
    if truth.ndim != 1 or truth.shape != pred.shape or not truth.size:
        raise ValueError("Wasserstein-1 requires equally sized nonempty vectors")
    if not np.isfinite(truth).all() or not np.isfinite(pred).all():
        raise ValueError("Wasserstein-1 requires finite values")
    return float(np.mean(np.abs(np.sort(truth) - np.sort(pred))))


def sample_title(split, n_events):
    if split == "test":
        return rf"Independent $t\bar{{t}}(2\ell)$ MC test sample: $N={n_events:,}$"
    if split == "val":
        return rf"$t\bar{{t}}(2\ell)$ MC validation sample: $N={n_events:,}$"
    raise ValueError(f"Unknown data split: {split}")


def cached_figure_options(saved):
    """Saved provenance controls rerendering; pre-split reports were test data."""
    split = saved.get("split", "test")
    if split not in SPLIT_ROWS:
        raise ValueError(f"Unknown split in saved report: {split}")
    return {"split": split, "model_label": saved.get("model_label", "HepSetReg")}


def clean_json(value):
    if isinstance(value, dict):
        return {str(k): clean_json(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, np.ndarray)):
        return [clean_json(v) for v in value]
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, (float, np.floating)):
        return float(value) if np.isfinite(value) else None
    return value


def style():
    plt.rcParams.update({
        "font.family": "serif", "font.serif": ["STIXGeneral", "DejaVu Serif"],
        "mathtext.fontset": "stix", "font.size": 9, "axes.labelsize": 10,
        "axes.titlesize": 10, "xtick.labelsize": 8, "ytick.labelsize": 8,
        "axes.linewidth": .65, "lines.linewidth": .85,
        "xtick.direction": "in", "ytick.direction": "in",
        "xtick.top": True, "ytick.right": True,
        "xtick.major.size": 3.4, "ytick.major.size": 3.4,
        "xtick.minor.size": 1.8, "ytick.minor.size": 1.8,
        "legend.frameon": False, "legend.fontsize": 9,
        "savefig.dpi": 240, "svg.fonttype": "none", "axes.unicode_minus": True,
    })


def save_figure(fig, output_dir, stem):
    for extension in ("png", "svg"):
        fig.savefig(output_dir / f"{stem}.{extension}", bbox_inches="tight", facecolor="white")
    plt.close(fig)


def step_band(ax, edges, lower, upper, color, alpha):
    ax.fill_between(edges, np.r_[lower, lower[-1]], np.r_[upper, upper[-1]],
                    step="post", color=color, alpha=alpha, linewidth=0)


def distribution_figure(histograms, names, n_events, output_dir, stem, *,
                        split="test", model_label="HepSetReg"):
    columns = len(names)
    fig = plt.figure(figsize=(3.35 * columns + .25 if columns > 1 else 4.0, 4.05))
    grid = fig.add_gridspec(2, columns, height_ratios=[2.5, 1], hspace=.045, wspace=.14)
    handles = None
    for column, name in enumerate(names):
        item = histograms[name]
        edges = item["edges"]
        truth, pred = item["truth_density"], item["pred_density"]
        ax = fig.add_subplot(grid[0, column])
        ratio_ax = fig.add_subplot(grid[1, column], sharex=ax)
        ax.stairs(truth, edges, fill=True, color="#e9e9e9", linewidth=0, label="MC parton truth")
        ax.stairs(truth, edges, color="#505050", linewidth=.65)
        step_band(ax, edges, np.maximum(0, pred - item["pred_density_se"]),
                  pred + item["pred_density_se"], RED, .20)
        ax.stairs(pred, edges, color=RED, linewidth=.9, label=model_label)
        handles = ax.get_legend_handles_labels()
        ax.set_ylim(bottom=0)
        ax.set_xlim(edges[0], edges[-1])
        ax.tick_params(labelbottom=False)
        ax.yaxis.set_major_locator(MaxNLocator(4, prune="upper"))
        formatter = ScalarFormatter(useMathText=True)
        formatter.set_powerlimits((-2, 3))
        ax.yaxis.set_major_formatter(formatter)
        ratio_ax.axhline(1, color="#606060", linestyle="--", linewidth=.65)
        step_band(ratio_ax, edges, item["ratio"] - item["ratio_se"],
                  item["ratio"] + item["ratio_se"], RED, .20)
        ratio_ax.stairs(item["ratio"], edges, color=RED, linewidth=.9)
        # Include the measured ratios and uncertainty; never conceal tail disagreement.
        finite_upper = (item["ratio"] + item["ratio_se"])
        finite_upper = finite_upper[np.isfinite(finite_upper)]
        upper = max(1.4, float(finite_upper.max()) * 1.07) if finite_upper.size else 1.4
        ratio_ax.set_ylim(0, upper)
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
    fig.subplots_adjust(left=.085 if columns > 1 else .19, right=.985, top=.83, bottom=.25)
    fig.legend(*handles, loc="upper center", ncol=2, bbox_to_anchor=(.52, 1.005),
               handlelength=2.6, columnspacing=2.4)
    fig.text(.52, .90, sample_title(split, n_events),
             ha="center", fontsize=9 if columns > 1 else 8.5)
    footer = ("All-event normalization; no prediction clipping. Shading: approximate 1-sigma statistical uncertainty.\n"
              "Pred./Truth accounts for paired events; zero-truth bins are omitted. Range exclusions are reported separately.")
    if columns == 1:
        footer = "All-event normalization; no clipping.\nShading: 1-sigma statistical uncertainty."
    fig.text(.52, .025, footer, ha="center", va="bottom", fontsize=7.4, linespacing=1.5)
    save_figure(fig, output_dir, stem)


def migration_figure(responses, n_events, output_dir, *, split="test", model_label="HepSetReg"):
    fig, axes = plt.subplots(2, 2, figsize=(7.45, 6.9))
    fig.subplots_adjust(left=.10, right=.865, bottom=.17, top=.875, hspace=.48, wspace=.52)
    for ax, (name, description) in zip(axes.flat, OBSERVABLES.items()):
        response = responses[name]
        probabilities, edges = response["central_probs"], response["edges"]
        n_bins = len(edges) - 1
        im = ax.imshow(np.ma.masked_invalid(probabilities), origin="lower", interpolation="none",
                       extent=(0, n_bins, 0, n_bins), cmap="Reds", vmin=0, vmax=1, aspect="equal")
        for row in range(n_bins):
            for col in range(n_bins):
                value = probabilities[row, col]
                if np.isfinite(value) and value >= .05:
                    ax.text(col + .5, row + .5, f"{value:.2f}", ha="center", va="center",
                            fontsize=8, color="white" if value >= .48 else "#202020")
        labels = [f"{value:g}" for value in edges]
        ax.set_xticks(np.arange(n_bins + 1), labels, rotation=45, ha="right")
        ax.set_yticks(np.arange(n_bins + 1), labels)
        ax.grid(color="#606060", linewidth=.35, alpha=.8)
        ax.tick_params(which="both", length=2)
        ax.set_xlabel("Predicted " + description["label"], labelpad=2, fontsize=9)
        ax.set_ylabel("Parton truth " + description["label"], labelpad=3, fontsize=9)
        ax.set_title(model_label, pad=6)
    color_ax = fig.add_axes([.91, .24, .022, .56])
    colorbar = fig.colorbar(im, cax=color_ax)
    colorbar.set_label("Probability", fontsize=9)
    colorbar.ax.tick_params(labelsize=8)
    sample = "independent MC test set" if split == "test" else "MC validation set"
    fig.text(.49, .97, f"Migration matrices on the {sample}", ha="center", fontsize=11)
    fig.text(.49, .937, rf"$N={n_events:,}$; truth on rows, prediction on columns", ha="center", fontsize=9)
    fig.text(.49, .037,
             "Each row includes every prediction in its denominator; under/overflow is retained in the saved tables.\n"
             "Only probabilities >= 0.05 are annotated. Mass and cos(phi) bins follow Fig. 6; the other bins are chosen here.",
             ha="center", va="bottom", fontsize=7.3, linespacing=1.5)
    save_figure(fig, output_dir, "migration_matrices")


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    here = Path(__file__).resolve().parent
    parser.add_argument("--split", choices=tuple(SPLIT_ROWS), default="test")
    parser.add_argument("--input", type=Path, help="default: <split>_predictions.parquet beside this script")
    parser.add_argument("--output-dir", type=Path, help="default: results/<run-name> for test, results/<run-name>_val for validation")
    parser.add_argument("--model-label", default="HepSetReg", help="legend and matrix title")
    parser.add_argument("--run-name", default=DEFAULT_RUN_NAME)
    parser.add_argument("--checkpoint-epoch", type=int, help="verified checkpoint epoch; omitted if not supplied")
    parser.add_argument("--expected-rows", type=int, help="default: 261557 for test, 260190 for val")
    parser.add_argument("--bootstrap", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--render-only", action="store_true", help="redraw figures from saved metrics without recomputing them")
    args = parser.parse_args(argv)
    if not args.run_name.strip() or args.run_name in (".", "..") or any(c in args.run_name for c in "/\\"):
        parser.error("--run-name must be a nonempty directory name, without path separators")
    if not args.model_label.strip():
        parser.error("--model-label must not be empty")
    if args.checkpoint_epoch is not None and args.checkpoint_epoch < 0:
        parser.error("--checkpoint-epoch must be nonnegative")
    if args.expected_rows is None:
        args.expected_rows = SPLIT_ROWS[args.split]
    if args.expected_rows <= 0:
        parser.error("--expected-rows must be positive")
    if args.input is None:
        args.input = here / f"{args.split}_predictions.parquet"
    if args.output_dir is None:
        suffix = "_val" if args.split == "val" else ""
        args.output_dir = here / "results" / f"{args.run_name}{suffix}"
    return args


def main(argv=None):
    args = parse_args(argv)
    source, output_dir = args.input.resolve(), args.output_dir.resolve()
    if args.render_only:
        saved = json.loads((output_dir / "metrics.json").read_text(encoding="utf-8"))
        histograms, responses = saved["histograms"], saved["responses"]
        for histogram in histograms.values():
            for key in ("edges", "truth_density", "pred_density", "pred_density_se", "ratio", "ratio_se"):
                histogram[key] = np.asarray(histogram[key], dtype=float)
        for response in responses.values():
            for key in ("edges", "central_probs"):
                response[key] = np.asarray(response[key], dtype=float)
        style()
        figure_options = cached_figure_options(saved)
        distribution_figure(histograms, list(OBSERVABLES)[:3], saved["n_events"], output_dir, "distributions_three", **figure_options)
        distribution_figure(histograms, ["cos_phi"], saved["n_events"], output_dir, "distribution_cos_phi", **figure_options)
        migration_figure(responses, saved["n_events"], output_dir, **figure_options)
        print(f"Redrew figures from unchanged metrics: {output_dir}", flush=True)
        return
    if output_dir.exists():
        raise FileExistsError(f"Output exists; choose a new directory: {output_dir}")
    table = pq.read_table(source, columns=["targets", "predictions"])
    if len(table) != args.expected_rows:
        raise ValueError(f"Expected {args.expected_rows:,} {args.split} events, found {len(table):,}")
    truth_group, pred_group = table["targets"].combine_chunks(), table["predictions"].combine_chunks()
    arrays = {}
    for name in OBSERVABLES:
        truth_field, pred_field = truth_group.field(name), pred_group.field(name)
        if truth_field.null_count or pred_field.null_count:
            raise ValueError(f"Missing {name} values; no events will be silently dropped")
        truth = np.asarray(truth_field.to_numpy(), dtype=np.float64)
        pred = np.asarray(pred_field.to_numpy(), dtype=np.float64)
        if not (np.isfinite(truth).all() and np.isfinite(pred).all()):
            raise ValueError(f"Non-finite {name}; no events will be silently dropped")
        arrays[name] = (truth, pred)
    output_dir.mkdir(parents=True, exist_ok=False)
    style()
    metrics, responses, histograms = {}, {}, {}
    for index, (name, description) in enumerate(OBSERVABLES.items()):
        truth, pred = arrays[name]
        result = regression_metrics(truth, pred, bootstrap=args.bootstrap, seed=args.seed + index)
        result["unit"] = description["unit"]
        result["wasserstein1"] = empirical_wasserstein1(truth, pred)
        if name.startswith("cos_"):
            result["prediction_outside_physical_range_count"] = int(np.count_nonzero((pred < -1) | (pred > 1)))
            result["prediction_outside_physical_range_fraction"] = result["prediction_outside_physical_range_count"] / len(pred)
        metrics[name] = result
        responses[name] = response_matrix(truth, pred, description["response_edges"])
        responses[name]["bin_source"] = description["response_bin_source"]
        histograms[name] = paired_histogram(truth, pred, description["distribution_edges"])
        print(f"{name}: MAE={result['mae']:.4f}; RMSE={result['rmse']:.4f}; central68 width={result['central68_width']:.4f}; W1={result['wasserstein1']:.4f}", flush=True)
    model_description = f"{args.model_label}; {args.run_name}"
    if args.checkpoint_epoch is not None:
        model_description += f"; checkpoint epoch {args.checkpoint_epoch}"
    manifest = {
        "created_utc": datetime.now(timezone.utc).isoformat(), "input": str(source),
        "input_sha256": hashlib.sha256(source.read_bytes()).hexdigest(), "n_events": len(table),
        "model": model_description,
        "model_label": args.model_label, "run_name": args.run_name,
        "checkpoint_epoch": args.checkpoint_epoch, "split": args.split,
        "split_role": "independent test evaluation" if args.split == "test" else "validation used for model selection",
        "reference": "MC parton truth, not measured collision data or detector-level ideal reconstruction",
        "event_weighting": "unweighted; one contribution per stored event",
        "range_policy": "global regression metrics use every event; no clipping; histogram density uses full N",
        "response_policy": "truth rows; predicted underflow+central bins+overflow columns; all predictions in row denominator",
        "histogram_uncertainty": "approximate counting uncertainty; ratio uses paired-event covariance",
        "distribution_metric": "wasserstein1 is mean absolute difference of sorted samples, in raw target units; it compares marginal distributions, not paired-event errors",
        "paper_metric_caution": "The paper does not specify its flow convention. Our diagonal metric is not an exact paper-score reproduction.",
        "metrics": metrics, "responses": responses, "histograms": histograms,
    }
    (output_dir / "metrics.json").write_text(json.dumps(clean_json(manifest), indent=2, allow_nan=False), encoding="utf-8")
    sample_description = "independent MC test" if args.split == "test" else "MC validation (used for model selection)"
    text = [f"{args.model_label} {sample_description} evaluation", f"Model: {model_description}", f"Events: {len(table):,}",
            "Bias = mean(truth - prediction); width = Q84.13 - Q15.87 of that residual.",
            "Bootstrap uncertainty is evaluation-sample statistical uncertainty, not training-seed or systematic uncertainty.", ""]
    for name, result in metrics.items():
        text.append(f"{name} [{result['unit']}]")
        for metric in ("mae", "rmse", "bias", "central68_width"):
            uncertainty = result["bootstrap_se"].get(metric)
            if uncertainty is not None:
                text.append(f"  {metric}: {result[metric]:.6f} +/- {uncertainty:.6f} (bootstrap SE)")
            else:
                text.append(f"  {metric}: {result[metric]:.6f} (bootstrap uncertainty not requested)")
        text.append(f"  wasserstein1: {result['wasserstein1']:.6f} (marginal distribution distance; raw target units)")
        response = responses[name]
        text.append(f"  mean in-range diagonal probability (flows in denominator): {response['diagonal_mean']:.6f}")
        text.append(f"  response-range fractions: {response['range_fractions']}")
        text.append(f"  displayed-spectrum range fractions: {histograms[name]['range_fractions']}")
        if "prediction_outside_physical_range_count" in result:
            text.append(f"  predictions outside [-1,1]: {result['prediction_outside_physical_range_count']} / {len(table)}")
        text.append("")
    (output_dir / "metrics.txt").write_text("\n".join(text), encoding="utf-8")
    figure_options = {"split": args.split, "model_label": args.model_label}
    distribution_figure(histograms, list(OBSERVABLES)[:3], len(table), output_dir, "distributions_three", **figure_options)
    distribution_figure(histograms, ["cos_phi"], len(table), output_dir, "distribution_cos_phi", **figure_options)
    migration_figure(responses, len(table), output_dir, **figure_options)
    print(f"Saved checked metrics and figures to: {output_dir}", flush=True)


if __name__ == "__main__":
    main()
