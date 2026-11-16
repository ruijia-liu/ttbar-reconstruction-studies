#!/usr/bin/env python3
"""Predict only the official validation split with a completed four-target run.

The model, target scaling and checkpoint selection are read from the saved run.
Neither the independent test dataset nor its predictions are read or changed.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
import hashlib
import json
import os
from pathlib import Path
import re
import sys

import pyarrow as pa
import pyarrow.parquet as pq
import yaml

TARGETS = ["ttbar_mass", "delta_abs_y", "cos_K_plus", "cos_phi"]
VAL_ROWS = 260190


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def prepare_config(run_dir, output_dir):
    manifest = json.loads((run_dir / "run_manifest.json").read_text(encoding="utf-8"))
    training = manifest["training_config"]
    if training["features"]["target"] != TARGETS:
        raise ValueError("This saved run is not the required four-target model")
    val_path = Path(training["val_path"]).expanduser().resolve()
    with pq.ParquetFile(val_path) as parquet:
        provenance = json.loads((parquet.schema_arrow.metadata or {}).get(b"hepsetreg_four_observables", b"{}"))
        if parquet.metadata.num_rows != VAL_ROWS or provenance.get("split") != "val":
            raise ValueError("Expected the complete official validation split")
    config = deepcopy(yaml.safe_load((run_dir / "predict_four.yaml").read_text(encoding="utf-8"))["predict"])
    config.update(data_path=str(val_path), output=str(output_dir / "val_predictions.parquet"), shape_metric=None)
    config["features"] = deepcopy(training["features"])
    identity = {
        "split": "val", "events": VAL_ROWS, "run_name": run_dir.name,
        "checkpoint": str(Path(config["checkpoint"]).resolve()),
        "checkpoint_sha256": sha256(config["checkpoint"]),
        "stats_sha256": sha256(config["stats"]),
        "validation_conversion": provenance, "features": training["features"],
        "training_shape_metric": training["shape_metric"],
        "training_shape_weight": training["shape_weight"],
    }
    return config, identity


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument("--run-name", default="ttbar4_mse_seed42_v2")
    args = parser.parse_args()
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]*", args.run_name):
        parser.error("Invalid run name")
    if not os.environ.get("SLURM_JOB_ID"):
        raise RuntimeError("Run in your CSF3 GPU allocation terminal")
    repo = args.repo.expanduser().resolve()
    run_dir = repo / "trained_runs" / args.run_name
    output_dir = repo / "validation_results" / args.run_name
    pa.set_cpu_count(4)
    pa.set_io_thread_count(2)
    config, identity = prepare_config(run_dir, output_dir)
    loader = repo / "src/hepsetreg/datasets/EventDatasetBase.py"
    identity["loader_sha256"] = sha256(loader)
    identity["predictor_sha256"] = sha256(repo / "src/hepsetreg/runs/run_regression.py")
    output = Path(config["output"])
    receipt_path = output_dir / "validation_manifest.json"
    if output.exists():
        receipt = json.loads(receipt_path.read_text(encoding="utf-8")) if receipt_path.exists() else {}
        if receipt.get("identity") == identity and receipt.get("predictions_sha256") == sha256(output):
            print(f"Verified existing validation predictions: {output}", flush=True)
            return
        raise FileExistsError(f"Existing validation predictions have different or incomplete provenance: {output}")
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "predict_validation.yaml").write_text(
        yaml.safe_dump({"features": config["features"], "predict": {
            k: v for k, v in config.items() if k != "features"}}, sort_keys=False), encoding="utf-8")
    os.environ["OMP_NUM_THREADS"] = "4"
    os.environ["MKL_NUM_THREADS"] = "4"
    os.environ["OPENBLAS_NUM_THREADS"] = "4"
    sys.path.insert(0, str(repo / "src"))
    from hepsetreg.runs.run_regression import predict
    print(f"Predicting all {VAL_ROWS:,} validation events for {args.run_name}", flush=True)
    print(f"Checkpoint: {config['checkpoint']}", flush=True)
    predict(config)
    with pq.ParquetFile(output) as result:
        if result.metadata.num_rows != VAL_ROWS:
            raise RuntimeError("Validation prediction count does not match the source")
    receipt_path.write_text(json.dumps({"identity": identity, "predictions_sha256": sha256(output)}, indent=2), encoding="utf-8")
    print(f"VALIDATION COMPLETE: {output}", flush=True)


if __name__ == "__main__":
    main()
