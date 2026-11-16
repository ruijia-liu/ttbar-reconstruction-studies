#!/usr/bin/env python3
"""Compare fixed MSE and MSE+0.1*EMD checkpoints on the same validation events.

All differences are EMD minus MSE; negative MAE/RMSE/W1 differences are better.
Paired bootstrap uncertainty is conditional on these fixed models and does
not include training-seed variation or validation-based model selection.
No independent test predictions are read.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

from ttbar_metrics import regression_metrics, response_matrix

TARGETS = ["ttbar_mass", "delta_abs_y", "cos_K_plus", "cos_phi"]
LABELS = ["MSE", "MSE + 0.1 EMD"]
VAL_ROWS = 260190


def file_hash(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_predictions(path, expected_rows=VAL_ROWS):
    table = pq.read_table(path, columns=["targets", "predictions"])
    if len(table) != expected_rows:
        raise ValueError(f"Expected {expected_rows:,} validation events, found {len(table):,}: {path}")
    arrays = {}
    for group in ("targets", "predictions"):
        record = table[group].combine_chunks()
        arrays[group] = {}
        for target in TARGETS:
            column = record.field(target)
            if column.null_count:
                raise ValueError(f"Missing values in {path}: {group}.{target}")
            values = np.asarray(column.to_numpy(), dtype=np.float64)
            if not np.isfinite(values).all():
                raise ValueError(f"Non-finite values in {path}: {group}.{target}")
            arrays[group][target] = values
    return arrays["targets"], arrays["predictions"]


def require_aligned_truth(first, second):
    for name in TARGETS:
        if not np.array_equal(first[name], second[name]):
            raise ValueError(f"Stored truths differ in value or order for {name}; refusing a paired comparison")


def wasserstein1(truth, pred):
    return float(np.mean(np.abs(np.sort(truth) - np.sort(pred))))


def _score_vector(truth, pred):
    residual = truth - pred
    return np.array([np.abs(residual).mean(), np.sqrt(np.square(residual).mean()),
                     wasserstein1(truth, pred)], dtype=np.float64)


def compare_one(truth, baseline, variant, *, bootstrap=100, seed=42):
    arrays = []
    for values in (truth, baseline, variant):
        if np.ma.isMaskedArray(values) and np.any(np.ma.getmaskarray(values)):
            raise ValueError("Missing values are not allowed")
        array = np.asarray(values, dtype=np.float64)
        if array.ndim != 1 or not len(array) or not np.isfinite(array).all():
            raise ValueError("Expected finite, nonempty one-dimensional arrays")
        arrays.append(array)
    truth, baseline, variant = arrays
    if not (truth.shape == baseline.shape == variant.shape):
        raise ValueError("Paired arrays must have the same shape")
    if not isinstance(bootstrap, (int, np.integer)) or isinstance(bootstrap, bool) or bootstrap < 0 or bootstrap == 1:
        raise ValueError("bootstrap must be 0 or an integer >= 2")
    scores = {}
    truth_std = float(truth.std())
    for label, predictions in zip(LABELS, (baseline, variant)):
        scores[label] = regression_metrics(truth, predictions, bootstrap=0)
        scores[label]["wasserstein1"] = wasserstein1(truth, predictions)
        scores[label]["predicted_to_truth_std_ratio"] = float(predictions.std() / truth_std) if truth_std else None
    metric_names = ["mae", "rmse", "wasserstein1"]
    differences = {name: scores[LABELS[1]][name] - scores[LABELS[0]][name] for name in metric_names}
    changes = {name: 100 * differences[name] / scores[LABELS[0]][name]
               if scores[LABELS[0]][name] != 0 else None for name in metric_names}
    result = {"models": scores, "difference_emd_minus_mse": differences,
              "percentage_change_from_mse": changes, "bootstrap_resamples": int(bootstrap),
              "bootstrap_seed": int(seed), "difference_bootstrap_se": {}, "difference_bootstrap_ci95": {}}
    if bootstrap:
        rng = np.random.default_rng(seed)
        sampled_differences = np.empty((bootstrap, len(metric_names)))
        for i in range(bootstrap):
            selected = rng.integers(0, len(truth), len(truth))
            t, b, v = truth[selected], baseline[selected], variant[selected]
            sampled_differences[i] = _score_vector(t, v) - _score_vector(t, b)
        se = sampled_differences.std(axis=0, ddof=1)
        interval = np.quantile(sampled_differences, [.025, .975], axis=0)
        result["difference_bootstrap_se"] = dict(zip(metric_names, map(float, se)))
        result["difference_bootstrap_ci95"] = {name: interval[:, i].tolist() for i, name in enumerate(metric_names)}
    return result


def endpoint_summary(truth, pred):
    result = {"predictions_outside_physical_range": int(np.count_nonzero((pred < -1) | (pred > 1)))}
    for key, lo, hi, include_right in (("negative_endpoint", -1, -.9, False), ("positive_endpoint", .9, 1, True)):
        tmask = (truth >= lo) & ((truth <= hi) if include_right else (truth < hi))
        pmask = (pred >= lo) & ((pred <= hi) if include_right else (pred < hi))
        tcount, pcount = int(tmask.sum()), int(pmask.sum())
        result[key] = {"edges": [lo, hi], "right_edge_inclusive": include_right,
                       "truth_count": tcount, "prediction_count": pcount,
                       "predicted_to_truth_count_ratio": pcount / tcount if tcount else None}
    return result


def main():
    here = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path, default=here / "val_predictions.parquet")
    parser.add_argument("--variant", type=Path, default=here / "val_predictions_emd01.parquet")
    parser.add_argument("--output-dir", type=Path, default=here / "results/comparison_val_mse_emd0p1_seed42")
    parser.add_argument("--bootstrap", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--render-only", action="store_true")
    args = parser.parse_args()
    baseline_path, variant_path, output_dir = args.baseline.resolve(), args.variant.resolve(), args.output_dir.resolve()
    if baseline_path == variant_path:
        raise ValueError("Two distinct prediction files are required")
    truth, baseline = load_predictions(baseline_path)
    other_truth, variant = load_predictions(variant_path)
    require_aligned_truth(truth, other_truth)
    sources = {LABELS[0]: {"path": str(baseline_path), "sha256": file_hash(baseline_path)},
               LABELS[1]: {"path": str(variant_path), "sha256": file_hash(variant_path)}}
    from plot_ttbar_four import OBSERVABLES, clean_json
    from plot_ttbar_comparison import make_comparison_figures
    predictions = {LABELS[0]: baseline, LABELS[1]: variant}
    if args.render_only:
        saved = json.loads((output_dir / "comparison.json").read_text(encoding="utf-8"))
        if saved["sources"] != sources:
            raise ValueError("Source predictions changed since the saved comparison")
        make_comparison_figures(truth, predictions, output_dir)
        print(f"Redrew comparison figures: {output_dir}", flush=True)
        return
    if output_dir.exists():
        raise FileExistsError(f"Output exists; select a new directory: {output_dir}")
    results = {}
    for i, name in enumerate(TARGETS):
        result = compare_one(truth[name], baseline[name], variant[name], bootstrap=args.bootstrap, seed=args.seed + i)
        result["unit"] = OBSERVABLES[name]["unit"]
        result["responses"] = {label: response_matrix(truth[name], values[name], OBSERVABLES[name]["response_edges"])
                               for label, values in predictions.items()}
        if name.startswith("cos_"):
            result["endpoints"] = {label: endpoint_summary(truth[name], values[name]) for label, values in predictions.items()}
        results[name] = result
        print(f"{name}: MSE -> MSE+EMD: MAE {result['models'][LABELS[0]]['mae']:.6f} -> {result['models'][LABELS[1]]['mae']:.6f}; "
              f"W1 {result['models'][LABELS[0]]['wasserstein1']:.6f} -> {result['models'][LABELS[1]]['wasserstein1']:.6f}", flush=True)
    report = {
        "created_utc": datetime.now(timezone.utc).isoformat(), "split": "val", "n_events": VAL_ROWS,
        "sources": sources,
        "models": {LABELS[0]: {"run": "ttbar4_mse_seed42_v2", "checkpoint_epoch": 34},
                   LABELS[1]: {"run": "ttbar4_mse_emd0p1_seed42", "checkpoint_epoch": 41}},
        "alignment_check": "All four stored truth columns match exactly in event order; exports contain no event IDs.",
        "checkpoint_selection": "Each run used its own validation training objective; total val_loss values are not compared.",
        "difference_convention": "EMD minus MSE; negative MAE/RMSE/W1 differences are improvements",
        "uncertainty_scope": "Paired-event bootstrap conditional on these fixed checkpoints; excludes training-seed variation and model-selection uncertainty.",
        "metric_policy": "All validation events, no clipping; W1 is global per-observable marginal distance in raw units, not averaged across units.",
        "results": results,
    }
    output_dir.mkdir(parents=True, exist_ok=False)
    (output_dir / "comparison.json").write_text(json.dumps(clean_json(report), indent=2, allow_nan=False), encoding="utf-8")
    lines = ["MSE vs MSE + 0.1 EMD: validation comparison", f"Events: {VAL_ROWS:,}",
             "Seed 42; checkpoint epochs 34 (MSE), 41 (MSE+EMD).",
             report["difference_convention"], report["uncertainty_scope"], ""]
    for name, result in results.items():
        lines.append(f"{name} [{result['unit']}]")
        for metric in ("mae", "rmse", "wasserstein1"):
            b, v = (result["models"][label][metric] for label in LABELS)
            interval = result["difference_bootstrap_ci95"].get(metric)
            change = result["percentage_change_from_mse"][metric]
            change_text = f"{change:+.3f}%" if change is not None else "undefined (zero baseline)"
            lines.append(f"  {metric}: {b:.8f} -> {v:.8f}; change={change_text}")
            if interval is not None:
                lines.append(f"    paired difference 95% bootstrap interval: [{interval[0]:.8f}, {interval[1]:.8f}]")
        if "endpoints" in result:
            for label in LABELS:
                lines.append(f"  {label} endpoint counts: {result['endpoints'][label]}")
        lines.append("")
    (output_dir / "comparison.txt").write_text("\n".join(lines), encoding="utf-8")
    make_comparison_figures(truth, predictions, output_dir)
    print(f"COMPARISON COMPLETE: {output_dir}", flush=True)


if __name__ == "__main__":
    main()
