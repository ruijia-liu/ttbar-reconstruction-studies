"""Reproduce validation-only comparison figures from saved, immutable metrics.

Run with a Python environment providing numpy and matplotlib.  This script never
loads models or changes experiment results; it writes only inside figures/.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
from matplotlib.patches import Rectangle
import numpy as np

ROOT = Path(__file__).resolve().parent
OUTPUT = ROOT / "figures"
METHODS = [
    ("correction_control", "Control correction"),
    ("conditional_ot", "Conditional OT"),
    ("proxy_constraints", "Metric constraints"),
    ("conditional_ot_proxy", "OT + constraints"),
    ("ranksim", "RankSim"),
    ("flowalign", "FlowAlign"),
]
OBSERVABLES = [
    ("ttbar_mass", r"$m_{t\bar{t}}$"),
    ("delta_abs_y", r"$\Delta |y|$"),
    ("cos_K_plus", r"$\cos\theta_K^+$"),
    ("cos_phi", r"$\cos\varphi$"),
]


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def load_metrics() -> tuple[dict, dict, dict]:
    plan = read_json(ROOT / "experiment_plan.json")
    baseline = plan["baseline_metrics"]
    results = {key: read_json(ROOT / "results" / key / "metrics.json")
               for key, _ in METHODS}
    for metrics in [baseline, *results.values()]:
        assert metrics["events"] == plan["val_rows"] == 260190
        for key, _ in OBSERVABLES:
            observable = metrics["observables"][key]
            assert observable["bin_edges"] == plan["trace_bins"][key]
            assert np.isfinite([observable[x] for x in ("res", "local_trace", "w1")]).all()
    return plan, baseline, results


def save(fig: plt.Figure, filename: str) -> None:
    for suffix in ("png", "pdf"):
        fig.savefig(OUTPUT / f"{filename}.{suffix}", dpi=240,
                    facecolor="white", bbox_inches="tight")
    plt.close(fig)


def comparison_figure(plan: dict, baseline: dict, results: dict, compact: bool = False) -> None:
    panels = [
        ("res", "Resolution improvement", "%", "100 × (1 − method / baseline)"),
        ("local_trace", "Trace improvement", "percentage points", "100 × (method − baseline)"),
        ("w1", r"$W_1$ reduction", "%", "100 × (1 − method / baseline)"),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(16.8, 7.0 if compact else 7.2), sharey=True)
    fig.subplots_adjust(left=.165, right=.985, bottom=.18 if compact else .27,
                        top=.77 if compact else .78, wspace=.23)
    cmap = plt.get_cmap("RdBu")
    for ax, (metric, title, units, formula) in zip(axes, panels):
        values = np.array([
            [100 * (results[method]["observables"][key][metric]
                    - baseline["observables"][key][metric])
             if metric == "local_trace" else
             100 * (1 - results[method]["observables"][key][metric]
                    / baseline["observables"][key][metric])
             for key, _ in OBSERVABLES]
            for method, _ in METHODS
        ])
        bound = float(np.max(np.abs(values))) * 1.08
        norm = TwoSlopeNorm(vmin=-bound, vcenter=0, vmax=bound)
        im = ax.imshow(values, cmap=cmap, norm=norm, aspect="auto")
        ax.set_title(f"{title}\n({units})", fontsize=15, pad=16)
        ax.set_xticks(range(4), [label for _, label in OBSERVABLES], fontsize=14)
        ax.set_yticks(range(6), [label for _, label in METHODS], fontsize=13)
        ax.tick_params(axis="both", length=0, pad=8)
        ax.set_xticks(np.arange(-.5, 4, 1), minor=True)
        ax.set_yticks(np.arange(-.5, 6, 1), minor=True)
        ax.grid(which="minor", color="white", linewidth=2.5)
        ax.tick_params(which="minor", bottom=False, left=False)
        for spine in ax.spines.values():
            spine.set_visible(False)
        for i in range(6):
            for j in range(4):
                rgba = cmap(norm(values[i, j]))
                luminance = .2126 * rgba[0] + .7152 * rgba[1] + .0722 * rgba[2]
                ax.text(j, i, f"{values[i, j]:+.2f}", ha="center", va="center",
                        fontsize=12.5, color="white" if luminance < .48 else "#172536")
        # Thin outlines mark the two OT variants for direct comparison, without
        # defining a subjective aggregate ranking across different metrics.
        for row in (1, 3):
            ax.add_patch(Rectangle((-.47, row-.46), 3.94, .92,
                                   fill=False, edgecolor="#47617a", linewidth=1))
        colorbar = fig.colorbar(im, ax=ax, orientation="horizontal", pad=.20,
                               fraction=.045, shrink=.92, aspect=30)
        colorbar.ax.tick_params(labelsize=10)
        colorbar.outline.set_visible(False)
        colorbar.set_label(formula, fontsize=10, labelpad=6)
    for i in (1, 3):
        axes[0].get_yticklabels()[i].set_fontweight("bold")
    fig.suptitle("Six correction methods vs. original Huber + 0.1 EMD", fontsize=22,
                 x=.56, y=.975 if compact else .965, fontweight="bold")
    fig.text(.56, .89 if compact else .90, "Every cell: positive = improvement  ·  negative = degradation",
             ha="center", fontsize=13, color="#334b61")
    if not compact:
        fig.text(.165, .115,
                 "Resolution: residual Q84.13 − Q15.87.  Trace: mean diagonal of fixed-bin, truth-row-normalized migration matrix.",
                 fontsize=10.5, color="#334155")
        fig.text(.165, .077,
                 f"Full validation: {plan['val_rows']:,} events  ·  seed {plan['seed']}  ·  frozen encoder + correction head  ·  outlined rows: OT variants",
                 fontsize=10.5, color="#334155")
        fig.text(.165, .039,
                 "Validation also used for checkpoint selection; no independent test evaluation or multi-seed uncertainty estimate.",
                 fontsize=10.5, color="#334155")
    save(fig, "six_methods_note_clean" if compact else "six_methods_improvements")


def bin_labels(edges: list) -> list[str]:
    def number(value: float) -> str:
        return f"{value:g}".replace("-", "−")
    # Interval endpoints are written explicitly; matrix edges come from metrics.
    return [f"{number(lo)} to {number(hi)}" for lo, hi in zip(edges[:-1], edges[1:])]


def migration_figure(plan: dict, baseline: dict, results: dict) -> None:
    rows = [("Original Huber + 0.1 EMD", baseline),
            ("Conditional OT", results["conditional_ot"])]
    fig, axes = plt.subplots(2, 4, figsize=(21, 12.4))
    fig.subplots_adjust(left=.10, right=.93, bottom=.22, top=.84, hspace=.58, wspace=.43)
    cmap = plt.get_cmap("Blues")
    for row, (_, metrics) in enumerate(rows):
        for column, (key, label) in enumerate(OBSERVABLES):
            ax = axes[row, column]
            obs = metrics["observables"][key]
            matrix = np.array(obs["row_probabilities"])
            n = len(matrix)
            assert matrix.shape == (n, n)
            assert np.all((matrix >= 0) & (matrix <= 1))
            assert np.all(matrix.sum(axis=1) <= 1 + 1e-9)
            im = ax.imshow(matrix, vmin=0, vmax=1, cmap=cmap, aspect="equal")
            ax.set_title(f"{label}    |    Trace = {100*obs['local_trace']:.2f}%",
                         fontsize=15, pad=12)
            labels = bin_labels(obs["bin_edges"])
            ax.set_xticks(range(n), labels, rotation=38, ha="right", fontsize=10)
            ax.set_yticks(range(n), labels, fontsize=10)
            ax.tick_params(length=0, pad=6)
            unit = " [GeV]" if key == "ttbar_mass" else ""
            ax.set_xlabel("Predicted bin" + unit, fontsize=11, labelpad=9)
            ax.set_ylabel("Truth bin" + unit, fontsize=11, labelpad=8)
            ax.set_xticks(np.arange(-.5, n, 1), minor=True)
            ax.set_yticks(np.arange(-.5, n, 1), minor=True)
            ax.grid(which="minor", color="white", linewidth=1)
            ax.tick_params(which="minor", bottom=False, left=False)
            for spine in ax.spines.values():
                spine.set_visible(False)
            for i in range(n):
                for j in range(n):
                    ax.text(j, i, f"{matrix[i, j]:.3f}", ha="center", va="center",
                            fontsize=10.5 if n >= 6 else 11.5,
                            color="white" if matrix[i, j] > .56 else "#163149")
    fig.text(.025, .720, rows[0][0], va="center", ha="center", rotation=90,
             fontsize=17, fontweight="bold", color="#334b61")
    fig.text(.025, .340, rows[1][0], va="center", ha="center", rotation=90,
             fontsize=17, fontweight="bold", color="#334b61")
    colorbar_ax = fig.add_axes([.95, .30, .012, .43])
    cb = fig.colorbar(im, cax=colorbar_ax, ticks=np.arange(0, 1.01, .2))
    cb.set_label("P(predicted bin | truth bin)", fontsize=13, labelpad=12)
    cb.ax.tick_params(labelsize=11)
    cb.outline.set_visible(False)
    fig.suptitle("Fixed-bin migration matrices: original baseline and Conditional OT",
                 fontsize=23, fontweight="bold", y=.965)
    fig.text(.51, .91, "Rows = truth  ·  columns = prediction  ·  identical bin edges and color scale in both rows",
             ha="center", fontsize=14, color="#334b61")
    fig.text(.10, .084,
             "Cells are probabilities, normalized by the count in each truth bin. Truth outside the displayed bins is omitted;",
             fontsize=11, color="#334155")
    fig.text(.10, .058,
             "out-of-range predictions remain in the row denominator. Trace is the unweighted mean of diagonal entries.",
             fontsize=11, color="#334155")
    fig.text(.10, .025,
             f"Full validation: {plan['val_rows']:,} events  ·  seed {plan['seed']}  ·  frozen encoder + correction head  ·  validation-selected checkpoint; no independent test.",
             fontsize=11, color="#334155")
    save(fig, "baseline_vs_conditional_ot_migration")


def main() -> None:
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 12,
                         "pdf.fonttype": 42, "ps.fonttype": 42,
                         "axes.unicode_minus": True})
    OUTPUT.mkdir(exist_ok=True)
    plan, baseline, results = load_metrics()
    comparison_figure(plan, baseline, results)
    migration_figure(plan, baseline, results)
    for path in sorted(OUTPUT.glob("*")):
        print(path)


if __name__ == "__main__":
    main()
