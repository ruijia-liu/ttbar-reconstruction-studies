#!/usr/bin/env python3
"""Derive four observables from native Slot predictions and aligned detector data.

No model is evaluated here. Native Slot Parquet supplies predictions.top/antitop
and the corresponding exported truth four-vectors. The reference Parquet must
be the exact full-field data used for that prediction, in the same order.
All eight truth components are compared before publishing output. This is a
numerical alignment check, not proof that detector inputs were never altered.

Prediction four-vectors use (px, py, pz, E), in the source's physical units.
Angles use DETECTOR leptons, selected by charge, with sequential canonical
boosts lab -> predicted ttbar CM -> predicted parent CM. Reference angular
labels are read directly; generator leptons never enter predicted quantities.

Invalid predictions remain present as NaN plus a per-observable validity mask.
No physical projection, clipping, event filtering, or truth substitution occurs.
Existing metrics readers must explicitly support these masks before using the
output; they must not silently drop NaNs or compare different selected samples.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import tempfile

import awkward as ak
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq


OBSERVABLES = ("ttbar_mass", "delta_abs_y", "cos_K_plus", "cos_phi")
COMPONENTS = ("px", "py", "pz", "E")
SLOTS = ("top", "antitop")
DEFAULT_ALIGN_RTOL = 5e-6
DEFAULT_ALIGN_ATOL = 5e-4
ANGLE_ROUNDOFF_TOLERANCE = 1e-12
VERSION = "slot-four-observables-v1"


class AlignmentError(ValueError):
    """Native exported truth does not correspond to the reference rows."""


def _p4(values, name):
    arr = np.asarray(values, dtype=np.float64)
    if arr.ndim != 2 or arr.shape[1] != 4:
        raise ValueError(f"{name} must have shape (N, 4), in px,py,pz,E order")
    return arr


def _future_timelike(p4):
    with np.errstate(over="ignore", invalid="ignore"):
        mass2 = p4[:, 3] ** 2 - np.sum(p4[:, :3] ** 2, axis=1)
    valid = np.isfinite(p4).all(axis=1) & np.isfinite(mass2)
    valid &= (p4[:, 3] > 0) & (mass2 > 0)
    return valid, mass2


def boost_to_rest(p4, frame):
    """Canonical Lorentz boost into frame's rest frame; bad rows become NaN.

    Uses gamma^2/(gamma+1) for (gamma-1)/beta^2, avoiding division by zero
    for a frame already at rest. Does not make spacelike frames physical.
    """
    p4, frame = _p4(p4, "p4"), _p4(frame, "frame")
    if p4.shape != frame.shape:
        raise ValueError("p4 and frame shapes differ")
    frame_ok, mass2 = _future_timelike(frame)
    valid = frame_ok & np.isfinite(p4).all(axis=1)
    out = np.full(p4.shape, np.nan, dtype=np.float64)
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        beta = frame[valid, :3] / frame[valid, 3, None]
        gamma = frame[valid, 3] / np.sqrt(mass2[valid])
        dot = np.sum(beta * p4[valid, :3], axis=1)
        factor = gamma**2 / (gamma + 1) * dot - gamma * p4[valid, 3]
        out[valid, :3] = p4[valid, :3] + factor[:, None] * beta
        out[valid, 3] = gamma * (p4[valid, 3] - dot)
    valid &= np.isfinite(out).all(axis=1)
    out[~valid] = np.nan
    return out, valid


def _unit(p4):
    with np.errstate(over="ignore", invalid="ignore"):
        norm = np.linalg.norm(p4[:, :3], axis=1)
    valid = np.isfinite(p4).all(axis=1) & np.isfinite(norm) & (norm > 0)
    result = np.full((len(p4), 3), np.nan)
    result[valid] = p4[valid, :3] / norm[valid, None]
    return result, valid


def _rapidity(p4):
    valid = np.isfinite(p4).all(axis=1) & (p4[:, 3] > np.abs(p4[:, 2]))
    y = np.full(len(p4), np.nan)
    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        y[valid] = 0.5 * (
            np.log(p4[valid, 3] + p4[valid, 2])
            - np.log(p4[valid, 3] - p4[valid, 2])
        )
    valid &= np.isfinite(y)
    return y, valid


def derive_observables(top, antitop, positive_lepton, negative_lepton, charge_valid):
    """Return predictions, per-quantity validity, and diagnostic boolean masks.

    Conservative physical policy: every quantity requires positive-energy,
    timelike top AND antitop predictions, and a timelike positive-energy sum.
    Mass/rapidity do not require leptons; cos_K_plus needs only the valid
    positive-lepton direction, whereas cos_phi needs both lepton directions.
    A zero top momentum in ttbar CM makes the helicity axis undefined but
    does not by itself invalidate cos_phi. Roundoff-sized cosine excesses up
    to 1e-12 are retained unchanged and explicitly counted in the audit.
    """
    arrays = [_p4(v, n) for v, n in zip(
        (top, antitop, positive_lepton, negative_lepton),
        ("top", "antitop", "positive_lepton", "negative_lepton"),
    )]
    top, antitop, positive_lepton, negative_lepton = arrays
    n = len(top)
    if any(len(a) != n for a in arrays):
        raise ValueError("Four-vector row counts differ")
    charge_valid = np.asarray(charge_valid, dtype=bool)
    if charge_valid.shape != (n,):
        raise ValueError("charge_valid must have shape (N,)")
    with np.errstate(over="ignore", invalid="ignore"):
        pair = top + antitop
    top_ok, _ = _future_timelike(top)
    anti_ok, _ = _future_timelike(antitop)
    pair_ok, pair_mass2 = _future_timelike(pair)
    parents_ok = top_ok & anti_ok & pair_ok
    values = {name: np.full(n, np.nan) for name in OBSERVABLES}
    valid = {name: np.zeros(n, dtype=bool) for name in OBSERVABLES}

    valid["ttbar_mass"] = parents_ok.copy()
    values["ttbar_mass"][parents_ok] = np.sqrt(pair_mass2[parents_ok])
    yt, yt_ok = _rapidity(top)
    ya, ya_ok = _rapidity(antitop)
    valid["delta_abs_y"] = parents_ok & yt_ok & ya_ok
    values["delta_abs_y"] = np.abs(yt) - np.abs(ya)

    top_cm, top_boost_ok = boost_to_rest(top, pair)
    anti_cm, anti_boost_ok = boost_to_rest(antitop, pair)
    positive_cm, positive_cm_ok = boost_to_rest(positive_lepton, pair)
    negative_cm, negative_cm_ok = boost_to_rest(negative_lepton, pair)
    positive_rest, positive_rest_ok = boost_to_rest(positive_cm, top_cm)
    negative_rest, negative_rest_ok = boost_to_rest(negative_cm, anti_cm)
    k, k_ok = _unit(top_cm)
    lp, lp_ok = _unit(positive_rest)
    lm, lm_ok = _unit(negative_rest)
    positive_ok = (positive_cm_ok & positive_rest_ok & lp_ok
                   & (positive_lepton[:, 3] > 0) & (positive_rest[:, 3] > 0))
    negative_ok = (negative_cm_ok & negative_rest_ok & lm_ok
                   & (negative_lepton[:, 3] > 0) & (negative_rest[:, 3] > 0))
    with np.errstate(invalid="ignore"):
        values["cos_K_plus"] = np.sum(k * lp, axis=1)
        values["cos_phi"] = np.sum(lp * lm, axis=1)
    valid["cos_K_plus"] = parents_ok & top_boost_ok & charge_valid & k_ok & positive_ok
    valid["cos_phi"] = (parents_ok & top_boost_ok & anti_boost_ok & charge_valid
                        & positive_ok & negative_ok)
    for name in OBSERVABLES:
        valid[name] &= np.isfinite(values[name])
        if name.startswith("cos_"):
            valid[name] &= np.abs(values[name]) <= 1 + ANGLE_ROUNDOFF_TOLERANCE
        values[name][~valid[name]] = np.nan
    quality = {
        "top_future_timelike": top_ok,
        "antitop_future_timelike": anti_ok,
        "pair_future_timelike": pair_ok,
        "charge_pair_valid": charge_valid,
        "helicity_axis_nonzero": k_ok,
        "positive_lepton_direction_valid": positive_ok,
        "negative_lepton_direction_valid": negative_ok,
    }
    return values, valid, quality


def _column(batch, path):
    parts = path.split(".")
    index = batch.schema.get_field_index(parts[0])
    if index < 0:
        # Some Arrow projection APIs expose selected nested leaves at top level.
        if len(parts) == 2 and batch.schema.get_field_index(parts[-1]) >= 0:
            return batch.column(batch.schema.get_field_index(parts[-1]))
        raise ValueError(f"Missing projected field {path}")
    result = batch.column(index)
    for name in parts[1:]:
        result = result.field(name)
    return result


def _numbers(batch, path):
    column = _column(batch, path)
    if column.null_count:
        raise ValueError(f"Null truth/component field {path}")
    return np.asarray(column.to_numpy(zero_copy_only=False), dtype=np.float64)


def _parent(batch, slot, native):
    return np.column_stack([
        _numbers(batch, f"targets.{slot}.{c}" if native else f"targets.{slot}_{c}")
        for c in COMPONENTS
    ])


def detector_leptons(batch):
    """Select measured positive/negative leptons without assuming flavour order."""
    groups = {}
    for flavour in ("el", "mu"):
        for field in ("charge", "px", "py", "pz", "e"):
            arr = ak.from_arrow(_column(batch, f"inputs.{flavour}_{field}"))
            arr = ak.fill_none(ak.fill_none(arr, [], axis=0), np.nan, axis=-1)
            groups[flavour, field] = arr
        lengths = ak.num(groups[flavour, "charge"], axis=1)
        for field in ("px", "py", "pz", "e"):
            if not bool(ak.all(ak.num(groups[flavour, field], axis=1) == lengths)):
                raise ValueError(f"Misaligned detector {flavour} charge/four-vector lists")
    charge = ak.concatenate([groups[f, "charge"] for f in ("el", "mu")], axis=1)
    good = ((ak.num(charge, axis=1) == 2) & (ak.sum(charge == 1, axis=1) == 1)
            & (ak.sum(charge == -1, axis=1) == 1))
    outputs = []
    for sign in (1, -1):
        columns = []
        for field in ("px", "py", "pz", "e"):
            array = ak.concatenate([groups[f, field] for f in ("el", "mu")], axis=1)
            selected = ak.fill_none(ak.firsts(array[charge == sign]), np.nan)
            columns.append(np.asarray(ak.to_numpy(selected), dtype=np.float64))
        outputs.append(np.column_stack(columns))
    return outputs[0], outputs[1], np.asarray(ak.to_numpy(good), dtype=bool)


def check_truth_alignment(native, reference, *, atol, rtol, row_offset, accumulators):
    """Check all eight components with abs(a-b) <= atol + rtol*abs(reference)."""
    for slot in SLOTS:
        a, b = _parent(native, slot, True), _parent(reference, slot, False)
        if a.shape != b.shape or not (np.isfinite(a).all() and np.isfinite(b).all()):
            raise AlignmentError(f"Nonfinite or unequal truth array at row {row_offset}")
        for j, component in enumerate(COMPONENTS):
            name = f"{slot}_{component}"
            diff = np.abs(a[:, j] - b[:, j])
            allowance = atol + rtol * np.abs(b[:, j])
            bad = diff > allowance
            if bad.any():
                local = np.flatnonzero(bad)[:5]
                raise AlignmentError(
                    f"{name} truth alignment failed at rows {(local + row_offset).tolist()}; "
                    f"max abs difference {float(diff.max()):.8g}; "
                    f"required abs(native-reference) <= {atol} + {rtol}*abs(reference). "
                    "Check exact input file, event order, and truth provenance; do not loosen "
                    "tolerance to conceal a row mismatch."
                )
            stat = accumulators.setdefault(name, {
                "rows": 0, "sum_absolute_difference": 0.0,
                "max_absolute_difference": 0.0, "max_tolerance_fraction": 0.0,
            })
            fraction = np.divide(diff, allowance, out=np.zeros_like(diff), where=allowance > 0)
            stat["rows"] += len(diff)
            stat["sum_absolute_difference"] += float(diff.sum())
            stat["max_absolute_difference"] = max(stat["max_absolute_difference"], float(diff.max()))
            stat["max_tolerance_fraction"] = max(stat["max_tolerance_fraction"], float(fraction.max()))


def reference_observables(batch):
    result = {name: _numbers(batch, f"targets.{name}") for name in OBSERVABLES
              if name != "delta_abs_y"}
    fields = _column(batch, "targets").type
    if fields.get_field_index("delta_abs_y") >= 0:
        result["delta_abs_y"] = _numbers(batch, "targets.delta_abs_y")
        delta_source = "reference targets.delta_abs_y"
    else:
        top, antitop = _parent(batch, "top", False), _parent(batch, "antitop", False)
        yt, good_t = _rapidity(top)
        ya, good_a = _rapidity(antitop)
        if not (good_t.all() and good_a.all()):
            raise ValueError("Reference truth has undefined parent rapidity")
        result["delta_abs_y"] = np.abs(yt) - np.abs(ya)
        delta_source = "derived from reference truth E,pz"
    if not all(np.isfinite(result[name]).all() for name in OBSERVABLES):
        raise ValueError("Nonfinite reference four-observable truth")
    return result, delta_source


def _aligned_batches(left, right):
    """Align row slices even when source row-group/batch boundaries differ."""
    a = b = None
    ia = ib = 0
    while True:
        if a is None or ia == a.num_rows:
            a, ia = next(left, None), 0
        if b is None or ib == b.num_rows:
            b, ib = next(right, None), 0
        if a is None or b is None:
            if a is not None or b is not None:
                raise AlignmentError("Input streams ended at different row counts")
            return
        count = min(a.num_rows - ia, b.num_rows - ib)
        if count <= 0:
            raise ValueError("Unexpected empty input batch")
        yield a.slice(ia, count), b.slice(ib, count)
        ia += count
        ib += count


def _sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _signature(path):
    stat = path.stat()
    return stat.st_size, stat.st_mtime_ns


def _publish_no_replace(partial, destination):
    if os.name == "nt":
        os.rename(partial, destination)
    else:
        os.link(partial, destination)


def convert_slot_predictions(native_path, reference_path, output_path, *,
                             alignment_atol=DEFAULT_ALIGN_ATOL,
                             alignment_rtol=DEFAULT_ALIGN_RTOL,
                             batch_size=65536, report_path=None):
    """Stream, verify, and publish a new four-observable Parquet plus JSON audit.

    No output is published until every truth row has passed the declared
    tolerance. Existing output/report files are never replaced. Missing
    predicted values are represented by NaN, not Arrow null, with false masks.
    """
    native_path, reference_path, output_path = (
        Path(p).expanduser().resolve() for p in (native_path, reference_path, output_path)
    )
    report_path = (Path(report_path).expanduser().resolve() if report_path else
                   output_path.with_suffix(".audit.json"))
    if not np.isfinite([alignment_atol, alignment_rtol]).all() or min(alignment_atol, alignment_rtol) < 0:
        raise ValueError("Alignment tolerances must be finite and nonnegative")
    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    if len({native_path, reference_path, output_path, report_path}) != 4:
        raise ValueError("Input, output and report paths must all differ")
    if output_path.exists() or report_path.exists():
        raise FileExistsError("Refusing to replace an existing output or audit report")
    initial_signatures = {p: _signature(p) for p in (native_path, reference_path)}
    hashes = {str(p): _sha256(p) for p in (native_path, reference_path)}
    provenance = {
        "adapter": VERSION, "native_slot_predictions": str(native_path),
        "reference_full_data": str(reference_path), "source_sha256": hashes,
        "alignment_atol": float(alignment_atol), "alignment_rtol": float(alignment_rtol),
        "alignment_rule": "abs(native-reference) <= atol + rtol*abs(reference), all 8 truth components per row",
        "alignment_limit": "Numerical truth correspondence cannot alone prove detector inputs were unmodified; supply the exact native prediction input file.",
        "prediction_policy": "Each quantity requires both predicted parents and their sum future-directed timelike; additional per-observable direction checks apply.",
        "angle_convention": "detector charge-selected leptons; lab -> predicted ttbar CM -> predicted parent CM; K=top direction in ttbar CM",
        "angle_roundoff_tolerance": ANGLE_ROUNDOFF_TOLERANCE,
        "invalid_policy": "No rows dropped or values clipped; invalid prediction is NaN and validity.<observable>=false",
        "units": "Same physical units as source; standard HepSetReg momenta/mass are GeV",
        "downstream_requirement": "Respect validity masks and report invalid fractions; compare methods on a declared common-valid subset as well as individual validity coverage.",
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    partials = []
    writer = None
    published_output = False
    try:
        for destination in (output_path, report_path):
            fd, name = tempfile.mkstemp(prefix=destination.name + ".", suffix=".partial", dir=destination.parent)
            os.close(fd)
            partials.append(Path(name))
        alignment = {}
        validity_counts = {name: 0 for name in OBSERVABLES}
        quality_counts = {}
        roundoff_counts = {name: 0 for name in ("cos_K_plus", "cos_phi")}
        total = 0
        with pq.ParquetFile(native_path) as native_file, pq.ParquetFile(reference_path) as reference_file:
            expected_rows = native_file.metadata.num_rows
            if expected_rows == 0 or expected_rows != reference_file.metadata.num_rows:
                raise AlignmentError("Native/reference row counts differ or are zero")
            selected_reference = ["targets"] + [
                f"inputs.{flavour}_{field}" for flavour in ("el", "mu")
                for field in ("charge", "px", "py", "pz", "e")
            ]
            left = native_file.iter_batches(batch_size=batch_size, columns=["targets", "predictions"])
            right = reference_file.iter_batches(batch_size=batch_size, columns=selected_reference)
            for native_batch, reference_batch in _aligned_batches(iter(left), iter(right)):
                check_truth_alignment(native_batch, reference_batch, atol=alignment_atol,
                                      rtol=alignment_rtol, row_offset=total, accumulators=alignment)
                truth, delta_source = reference_observables(reference_batch)
                p4 = [np.column_stack([
                    _numbers(native_batch, f"predictions.{slot}.{c}") for c in COMPONENTS
                ]) for slot in SLOTS]
                positive, negative, charge_valid = detector_leptons(reference_batch)
                predictions, validity, quality = derive_observables(*p4, positive, negative, charge_valid)
                count = native_batch.num_rows
                arrays = [pa.array(np.arange(total, total + count, dtype=np.int64))]
                for data, dtype in ((truth, pa.float64()), (predictions, pa.float64()),
                                    (validity, pa.bool_()), (quality, pa.bool_())):
                    names = list(OBSERVABLES) if data is not quality else list(quality)
                    arrays.append(pa.StructArray.from_arrays(
                        [pa.array(data[name], type=dtype) for name in names], names=names,
                    ))
                table = pa.Table.from_arrays(arrays, names=["row_index", "targets", "predictions", "validity", "quality"])
                if writer is None:
                    provenance["n_events"] = expected_rows
                    provenance["delta_abs_y_truth_source"] = delta_source
                    schema = table.schema.with_metadata({b"slot_four_observables": json.dumps(provenance, sort_keys=True).encode()})
                    writer = pq.ParquetWriter(partials[0], schema, compression="zstd")
                elif delta_source != provenance["delta_abs_y_truth_source"]:
                    raise ValueError("Reference truth schema changed during reading")
                writer.write_table(table)
                for name in OBSERVABLES:
                    validity_counts[name] += int(validity[name].sum())
                for name in quality:
                    quality_counts[name] = quality_counts.get(name, 0) + int(quality[name].sum())
                for name in roundoff_counts:
                    roundoff_counts[name] += int(np.count_nonzero(validity[name] & (np.abs(predictions[name]) > 1)))
                total += count
            if total != expected_rows:
                raise AlignmentError("Streamed row count differs from metadata")
        writer.close()
        writer = None
        for path, before in initial_signatures.items():
            if _signature(path) != before:
                raise ValueError(f"Source file changed during conversion: {path}")
        for stat in alignment.values():
            stat["mean_absolute_difference"] = stat.pop("sum_absolute_difference") / stat["rows"]
        report = {
            **provenance, "output": str(output_path), "output_sha256": _sha256(partials[0]),
            "truth_alignment": alignment, "valid_counts": validity_counts,
            "invalid_counts": {name: total - count for name, count in validity_counts.items()},
            "valid_fractions": {name: count / total for name, count in validity_counts.items()},
            "quality_true_counts": quality_counts, "roundoff_cosines_retained_outside_unit_interval": roundoff_counts,
            "published_rows": total,
        }
        partials[1].write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        _publish_no_replace(partials[0], output_path)
        published_output = True
        try:
            _publish_no_replace(partials[1], report_path)
        except Exception:
            # Remove only the output this invocation just created; never an old file.
            output_path.unlink()
            published_output = False
            raise
        return report
    finally:
        if writer is not None:
            writer.close()
        for partial in partials:
            partial.unlink(missing_ok=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--slot-predictions", type=Path, required=True)
    parser.add_argument("--reference-data", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--report", type=Path)
    parser.add_argument("--alignment-atol", type=float, default=DEFAULT_ALIGN_ATOL,
                        help="Absolute truth alignment tolerance in source momentum/energy units (default 0.0005 GeV)")
    parser.add_argument("--alignment-rtol", type=float, default=DEFAULT_ALIGN_RTOL,
                        help="Relative truth alignment tolerance (default 0.000005)")
    parser.add_argument("--batch-size", type=int, default=65536)
    args = parser.parse_args()
    pa.set_cpu_count(4)
    pa.set_io_thread_count(4)
    report = convert_slot_predictions(
        args.slot_predictions, args.reference_data, args.output,
        alignment_atol=args.alignment_atol, alignment_rtol=args.alignment_rtol,
        batch_size=args.batch_size, report_path=args.report,
    )
    print(f"Saved {report['published_rows']:,} rows to {report['output']}")
    for name in OBSERVABLES:
        print(f"{name}: valid {report['valid_counts'][name]:,}; invalid {report['invalid_counts'][name]:,}")
    print(f"Audit: {args.report or args.output.with_suffix('.audit.json')}")


if __name__ == "__main__":
    main()
