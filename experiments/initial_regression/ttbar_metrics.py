"""Paired-event regression diagnostics for the four-target baseline.

These metrics describe this baseline and its supplied binning; they do not
claim exact comparability with a paper's reconstruction or event selection.
Only NumPy is required. All events must have finite truth and predictions.
"""
from __future__ import annotations

import numpy as np


_METRIC_NAMES = ("mae", "rmse", "bias", "central68_width")
_CENTRAL68_QUANTILES = (0.1587, 0.8413)


def _vector(values, name):
    """Reject missing values before converting masked arrays to plain arrays."""
    if np.ma.isMaskedArray(values) and np.any(np.ma.getmaskarray(values)):
        raise ValueError(f"{name} contains masked/missing values")
    raw = np.asarray(values)
    if np.iscomplexobj(raw):
        raise ValueError(f"{name} must contain real values")
    try:
        result = np.asarray(values, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must contain real numeric values") from exc
    if result.ndim != 1:
        raise ValueError(f"{name} must be one-dimensional")
    if not np.all(np.isfinite(result)):
        raise ValueError(f"{name} contains non-finite/missing values")
    return result


def _paired(truth, pred):
    truth = _vector(truth, "truth")
    pred = _vector(pred, "pred")
    if truth.shape != pred.shape:
        raise ValueError("truth and pred must have identical one-dimensional shapes")
    if truth.size == 0:
        raise ValueError("at least one paired event is required")
    return truth, pred


def _edges(edges):
    edges = _vector(edges, "edges")
    if edges.size < 2 or np.any(np.diff(edges) <= 0):
        raise ValueError("edges must contain at least two strictly increasing values")
    return edges


def _bin_indices(values, edges):
    """Use np.histogram semantics: final bin includes the rightmost edge.

    Return -1 for underflow, 0..n_bins-1 for in-range values, and n_bins
    for overflow. Internal edges belong to the bin on their right.
    """
    result = np.searchsorted(edges, values, side="right") - 1
    result[values == edges[-1]] = len(edges) - 2
    return result


def _binned_pairs(truth, pred, edges):
    truth, pred = _paired(truth, pred)
    edges = _edges(edges)
    n_bins = edges.size - 1
    truth_bin = _bin_indices(truth, edges)
    pred_bin = _bin_indices(pred, edges)
    truth_in = (truth_bin >= 0) & (truth_bin < n_bins)
    pred_in = (pred_bin >= 0) & (pred_bin < n_bins)
    counts = {
        "truth_underflow": int(np.count_nonzero(truth_bin < 0)),
        "truth_overflow": int(np.count_nonzero(truth_bin >= n_bins)),
        "truth_in_range": int(np.count_nonzero(truth_in)),
        "truth_out_of_range": int(np.count_nonzero(~truth_in)),
        "pred_underflow": int(np.count_nonzero(pred_bin < 0)),
        "pred_overflow": int(np.count_nonzero(pred_bin >= n_bins)),
        "pred_in_range": int(np.count_nonzero(pred_in)),
        "pred_out_of_range": int(np.count_nonzero(~pred_in)),
        "both_in_range": int(np.count_nonzero(truth_in & pred_in)),
    }
    fractions = {key: count / truth.size for key, count in counts.items()}
    return truth_bin, pred_bin, truth_in, pred_in, edges, counts, fractions


def _residual_metrics(residual):
    qlo, qhi = np.quantile(residual, _CENTRAL68_QUANTILES)
    return np.array([
        np.mean(np.abs(residual)),
        np.sqrt(np.mean(np.square(residual))),
        np.mean(residual),
        qhi - qlo,
    ], dtype=np.float64)


def regression_metrics(truth, pred, *, bootstrap=100, seed=42):
    """Return metrics and paired-event bootstrap uncertainty for one target.

    ``bias`` is mean(truth - pred), so positive bias means underprediction.
    ``central68_width`` is the FULL residual width Q(0.8413)-Q(0.1587),
    using NumPy's linear-interpolation quantile convention. It is not a
    half-width or a relative resolution. All supplied finite events count.

    The bootstrap samples event indices with replacement. Applying these
    indices to residuals is exactly equivalent to resampling paired truth
    and predictions. Return keys ``bootstrap_se`` and ``bootstrap_ci95``
    hold per-metric sample standard errors and percentile 95% intervals.
    These describe test-event sampling only, not training/model uncertainty.
    Set bootstrap=0 to omit uncertainty; otherwise use at least 2 resamples.
    """
    truth, pred = _paired(truth, pred)
    if isinstance(bootstrap, (bool, np.bool_)) or not isinstance(bootstrap, (int, np.integer)):
        raise ValueError("bootstrap must be an integer, either 0 or at least 2")
    if bootstrap < 0 or bootstrap == 1:
        raise ValueError("bootstrap must be 0 or at least 2")
    with np.errstate(over="raise", invalid="raise"):
        residual = truth - pred
        values = _residual_metrics(residual)
    result = {name: float(value) for name, value in zip(_METRIC_NAMES, values)}
    result.update({
        "n_events": int(truth.size),
        "residual_definition": "truth - prediction",
        "central68_quantiles": list(_CENTRAL68_QUANTILES),
        "bootstrap_resamples": int(bootstrap),
        "bootstrap_seed": int(seed),
        "bootstrap_se": {},
        "bootstrap_ci95": {},
    })
    if bootstrap:
        rng = np.random.default_rng(seed)
        samples = np.empty((bootstrap, len(_METRIC_NAMES)), dtype=np.float64)
        for index in range(bootstrap):
            selected = rng.integers(0, residual.size, size=residual.size)
            samples[index] = _residual_metrics(residual[selected])
        se = samples.std(axis=0, ddof=1)
        intervals = np.quantile(samples, [0.025, 0.975], axis=0)
        result["bootstrap_se"] = {
            name: float(value) for name, value in zip(_METRIC_NAMES, se)
        }
        result["bootstrap_ci95"] = {
            name: [float(intervals[0, i]), float(intervals[1, i])]
            for i, name in enumerate(_METRIC_NAMES)
        }
    return result


def response_matrix(truth, pred, edges):
    """Return a truth-row / prediction-column response with explicit tails.

    For B bins, full_counts/full_probs have shape (B, B+2): prediction
    underflow, B central bins, prediction overflow. Every finite prediction
    for an in-range truth contributes to its row denominator. Thus leakage
    penalizes central_probs and diagonal_mean rather than being renormalized
    away. Truth outside the supplied edges is excluded from rows and counted
    separately in range_counts/range_fractions (fractions use all events).

    Empty rows contain NaN probabilities. diagonal_mean averages the central
    diagonal over nonempty truth rows only; n_nonempty_rows accompanies it.
    Both axes include the rightmost edge, matching numpy.histogram.
    """
    tbin, pbin, tin, _, edges, counts, fractions = _binned_pairs(truth, pred, edges)
    n_bins = edges.size - 1
    flat = tbin[tin] * (n_bins + 2) + (pbin[tin] + 1)
    full_counts = np.bincount(flat, minlength=n_bins * (n_bins + 2)).reshape(n_bins, n_bins + 2)
    row_counts = full_counts.sum(axis=1)
    nonempty = row_counts > 0
    full_probs = np.full(full_counts.shape, np.nan, dtype=np.float64)
    np.divide(full_counts, row_counts[:, None], out=full_probs, where=nonempty[:, None])
    central_probs = full_probs[:, 1:-1]
    diagonal = np.diag(central_probs)
    return {
        "n_events": int(tbin.size),
        "edges": edges.copy(),
        "full_counts": full_counts,
        "full_probs": full_probs,
        "central_counts": full_counts[:, 1:-1],
        "central_probs": central_probs,
        "row_counts": row_counts,
        "diagonal_mean": float(diagonal[nonempty].mean()) if np.any(nonempty) else float("nan"),
        "n_nonempty_rows": int(np.count_nonzero(nonempty)),
        "range_counts": counts,
        "range_fractions": fractions,
    }


def paired_histogram(truth, pred, edges):
    """Return common-normalization densities and a paired prediction/truth ratio.

    Both densities divide by the SAME total paired-event count N and bin
    width. Events outside the edges are counted, never clipped or used to
    renormalize either distribution. Density errors are sqrt(count)/(N*w).

    ratio_se uses a paired Poisson-event delta-method approximation, with T
    truth-bin count, P prediction-bin count, and C count of events occupying
    that same bin in BOTH arrays:
        Var(P/T) ~= P/T**2 + P**2/T**3 - 2*P*C/T**3.
    The covariance term is essential; identical paired data give zero ratio
    uncertainty. This first-order approximation can be poor for sparse bins.
    Bins with T=0 have NaN ratio and uncertainty. Both histograms include the
    rightmost edge, matching numpy.histogram.
    """
    tbin, pbin, tin, pin, edges, counts, fractions = _binned_pairs(truth, pred, edges)
    n_bins = edges.size - 1
    truth_counts = np.bincount(tbin[tin], minlength=n_bins)
    pred_counts = np.bincount(pbin[pin], minlength=n_bins)
    same = tin & pin & (tbin == pbin)
    same_bin_counts = np.bincount(tbin[same], minlength=n_bins)
    norm = tbin.size * np.diff(edges)
    ratio = np.full(n_bins, np.nan, dtype=np.float64)
    ratio_se = np.full(n_bins, np.nan, dtype=np.float64)
    valid = truth_counts > 0
    t = truth_counts[valid].astype(np.float64)
    p = pred_counts[valid].astype(np.float64)
    c = same_bin_counts[valid].astype(np.float64)
    ratio[valid] = p / t
    # Algebraically the documented delta-method formula, without cancellation
    # when the same paired events occupy both bins (P=T=C).
    variance = (p * (t - c) + p * (p - c)) / t**3
    ratio_se[valid] = np.sqrt(np.maximum(variance, 0.0))
    return {
        "n_events": int(tbin.size),
        "edges": edges.copy(),
        "truth_counts": truth_counts,
        "pred_counts": pred_counts,
        "same_bin_counts": same_bin_counts,
        "truth_density": truth_counts / norm,
        "pred_density": pred_counts / norm,
        "truth_density_se": np.sqrt(truth_counts) / norm,
        "pred_density_se": np.sqrt(pred_counts) / norm,
        "ratio": ratio,
        "ratio_se": ratio_se,
        "range_counts": counts,
        "range_fractions": fractions,
    }
