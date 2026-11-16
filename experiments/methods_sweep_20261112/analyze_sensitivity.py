"""Reproduce checkpoint-choice sensitivity from local validation JSON files."""
import hashlib
import json
import math
from pathlib import Path
from statistics import median


ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "results"
KEYS = ("ttbar_mass", "delta_abs_y", "cos_K_plus", "cos_phi")


def read(path, inputs):
    raw = path.read_bytes()
    inputs[str(path.relative_to(ROOT)).replace("\\", "/")] = hashlib.sha256(raw).hexdigest()
    return json.loads(raw)


def score(metrics, baseline):
    checks = {}
    for key in KEYS:
        current, reference = metrics["observables"][key], baseline[key]
        checks[key + "/res"] = current["res"] <= reference["res"] + 1e-12
        checks[key + "/trace"] = current["local_trace"] >= reference["local_trace"] - 1e-12
    mean_log = sum(math.log(metrics["observables"][key]["w1"] / baseline[key]["w1"])
                   for key in KEYS) / len(KEYS)
    return checks, mean_log


def summary(values):
    if not values:
        return {"count": 0, "median_mean_log_w1_ratio": None,
                "geometric_w1_change_percent_from_median_log_score": None}
    value = median(values)
    return {"count": len(values), "median_mean_log_w1_ratio": value,
            "geometric_w1_change_percent_from_median_log_score": 100 * math.expm1(value)}


def main():
    inputs = {}
    plan = read(ROOT / "experiment_plan.json", inputs)
    baseline = plan["baseline_metrics"]["observables"]
    runs = {}
    for config in plan["runs"]:
        name = config["run_id"]
        mse = read(RESULTS / name / "metrics_mse.json", inputs)
        history = read(RESULTS / name / "history.json", inputs)
        checks, mean_log = score(mse, baseline)
        assert all(checks.values()) == mse["selection"]["eligible"]
        assert math.isclose(mean_log, mse["selection"]["shape_score"], abs_tol=1e-12)
        deltas = {}
        for key in KEYS:
            deltas[key] = {}
            for metric in ("res", "local_trace", "w1"):
                before, after = baseline[key][metric], mse["observables"][key][metric]
                item = {"baseline": before, "mse_best": after,
                        "absolute_change": after - before,
                        "relative_change_percent": 100 * (after / before - 1),
                        "improves": after > before if metric == "local_trace" else after < before}
                if metric == "local_trace":
                    item["change_percentage_points"] = 100 * (after - before)
                deltas[key][metric] = item
        assert all(a["epoch"] < b["epoch"] for a, b in zip(history, history[1:]))
        window = history[-50:]
        all_scores, eligible_scores = [], []
        for epoch in window:
            epoch_checks, epoch_score = score(epoch["metrics"], baseline)
            eligible = all(epoch_checks.values())
            assert eligible == epoch["selection"]["eligible"]
            assert math.isclose(epoch_score, epoch["selection"]["shape_score"], abs_tol=1e-12)
            all_scores.append(epoch_score)
            if eligible:
                eligible_scores.append(epoch_score)
        runs[name] = {
            "mse_best": {"epoch_zero_based": mse["checkpoint"]["epoch"],
                         "eligible": all(checks.values()), "checks_passed": sum(checks.values()),
                         "checks_total": len(checks), "checks": checks,
                         "mean_log_w1_ratio": mean_log,
                         "geometric_mean_w1_ratio": math.exp(mean_log),
                         "geometric_mean_w1_change_percent": 100 * math.expm1(mean_log),
                         "per_observable_deltas": deltas},
            "last50": {"history_epochs": len(history),
                       "epoch_range_zero_based": [window[0]["epoch"], window[-1]["epoch"]],
                       "eligible_count": len(eligible_scores), "total_count": len(all_scores),
                       "all_epochs": summary(all_scores),
                       "eligible_only": summary(eligible_scores)},
        }
    output = {
        "baseline": "Historical Huber+0.1 EMD; experiment_plan.json baseline_metrics",
        "data": "Validation only; no prediction or test files read",
        "calculation": {
            "eligibility": "All four resolution widths <= baseline and all four macro diagonals >= baseline, tolerance 1e-12",
            "w1_score": "Mean over four observables of log(W1 / baseline W1); lower is better",
            "aggregate_percent": "100 * expm1(mean log W1 ratio)",
            "window_median": "Median of per-epoch mean log W1 ratios, then transform by 100 * expm1; not arithmetic median of percentages",
            "mse_ranking": "Eligible first, then mean log W1 ratio; all six MSE-best checkpoints are eligible",
            "limitation": "Checkpoint-choice sensitivity on reused validation data; not independent significance or generalization evidence",
        },
        "input_sha256": inputs,
        "mse_best_ranking": sorted(runs, key=lambda name: (
            not runs[name]["mse_best"]["eligible"], runs[name]["mse_best"]["mean_log_w1_ratio"])),
        "last50_all_epochs_ranking": sorted(runs, key=lambda name: runs[name]["last50"]["all_epochs"]["median_mean_log_w1_ratio"]),
        "last50_eligible_only_ranking": sorted(runs, key=lambda name: (
            runs[name]["last50"]["eligible_only"]["median_mean_log_w1_ratio"] is None,
            runs[name]["last50"]["eligible_only"]["median_mean_log_w1_ratio"] or 0)),
        "runs": runs,
    }
    path = RESULTS / "checkpoint_sensitivity.json"
    path.write_text(json.dumps(output, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(path)
    for name in output["mse_best_ranking"]:
        run = runs[name]
        print(f"{name}: MSE W1 {run['mse_best']['geometric_mean_w1_change_percent']:+.4f}%; "
              f"last50 eligible {run['last50']['eligible_count']}/{run['last50']['total_count']}")


if __name__ == "__main__":
    main()
