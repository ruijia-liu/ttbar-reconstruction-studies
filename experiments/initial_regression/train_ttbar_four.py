#!/usr/bin/env python3
"""Launch a fresh four-observable MSE or MSE+EMD run with Ethan's bsc trainer.

Copy beside HepSetReg's train script. Run inside an allocated GPU shell:
    uv run python train_ttbar_four.py
The independent test set is not read by this script.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys

import pyarrow as pa
import pyarrow.parquet as pq
import yaml

TARGETS = ["ttbar_mass", "delta_abs_y", "cos_K_plus", "cos_phi"]
EXPECTED_ROWS = {"train": 4692288, "val": 260190}
META_KEY = b"hepsetreg_four_observables"


def inspect_split(path, split):
    """Check the prepared file's footer, without preloading its events."""
    with pq.ParquetFile(path) as parquet:
        count = parquet.metadata.num_rows
        schema = parquet.schema_arrow
        provenance = json.loads((schema.metadata or {}).get(META_KEY, b"{}"))
    if count != EXPECTED_ROWS[split]:
        raise ValueError(f"{split}: expected {EXPECTED_ROWS[split]:,} rows, found {count:,}")
    if provenance.get("split") != split or provenance.get("targets") != TARGETS:
        raise ValueError(f"{path}: unexpected conversion provenance")
    available = {field.name for field in schema.field("targets").type}
    if not set(TARGETS).issubset(available):
        raise ValueError(f"{path}: missing regression targets")
    print(f"{split}: {count:,} events; four targets OK", flush=True)
    return provenance


def build_config(base, data_dir, args):
    config = deepcopy(base["train"])
    config["features"] = deepcopy(base["features"])
    config["features"]["target"] = TARGETS.copy()
    config.update(
        train_path=str((data_dir / "ttbar-2L_train_four.parquet").resolve()),
        val_path=str((data_dir / "ttbar-2L_val_four.parquet").resolve()),
        run_name=args.run_name,
        d_model=128, nhead=8, num_layers=6,
        batch_size=args.batch_size, learning_rate=args.learning_rate,
        max_epochs=args.max_epochs, lr_patience=2, early_stop_patience=5,
        shape_metric=None if args.shape_metric == "none" else args.shape_metric,
        shape_weight=0.0 if args.shape_metric == "none" else args.shape_weight,
        accelerator="gpu", compile_model=True,
    )
    if Path(config["train_path"]) == Path(config["val_path"]):
        raise ValueError("Training and validation must be separate files")
    return config


def git_output(repo, *arguments):
    result = subprocess.run(["git", "-C", str(repo), *arguments],
                            capture_output=True, text=True, check=False)
    return result.stdout.strip() if result.returncode == 0 else "unavailable"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument("--data-dir", type=Path, default=Path(os.environ.get("RECO_DATA_ROOT", "external")) / "hepsetreg_data/parquet")
    parser.add_argument("--run-name", default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch-size", type=int, default=4096)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--max-epochs", type=int, default=50)
    parser.add_argument("--shape-metric", choices=["none", "emd"], default="none")
    parser.add_argument("--shape-weight", type=float, default=0.1,
                        help="EMD weight; ignored for the pure MSE baseline")
    parser.add_argument("--check-only", action="store_true")
    args = parser.parse_args()
    if args.run_name is None:
        loss_name = "mse" if args.shape_metric == "none" else f"mse_emd{args.shape_weight:g}".replace(".", "p")
        args.run_name = f"ttbar4_{loss_name}_seed{args.seed}"
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]*", args.run_name):
        parser.error("--run-name must use letters, numbers, underscores or hyphens")
    if args.batch_size < 1 or args.max_epochs < 1 or not (0 < args.learning_rate < 1):
        parser.error("batch size and epochs must be positive; learning rate must be in (0, 1)")
    if not 0 <= args.seed <= 2**32 - 1:
        parser.error("seed must be in [0, 2**32 - 1]")
    if args.shape_metric == "emd" and (not math.isfinite(args.shape_weight) or args.shape_weight <= 0):
        parser.error("--shape-weight must be finite and positive")
    pa.set_cpu_count(4)
    pa.set_io_thread_count(4)

    repo = args.repo.expanduser().resolve()
    data_dir = args.data_dir.expanduser().resolve()
    base_path = repo / "configs/run_regression.yaml"
    base = yaml.safe_load(base_path.read_text(encoding="utf-8"))
    config = build_config(base, data_dir, args)
    provenance = {split: inspect_split(Path(config[f"{split}_path"]), split)
                  for split in EXPECTED_ROWS}
    if os.path.samefile(config["train_path"], config["val_path"]):
        raise ValueError("Training and validation resolve to the same file")

    work_dir = (Path(os.environ.get("RECO_DATA_ROOT", "external")) / "hepsetreg_runs").resolve()
    run_dir = work_dir / "tb_logs/event_regression" / args.run_name
    backup_dir = repo / "trained_runs" / args.run_name
    if run_dir.exists() or backup_dir.exists():
        raise FileExistsError("This run already exists. Choose a NEW --run-name; existing results are preserved.")
    print(yaml.safe_dump({"seed": args.seed, "train": config}, sort_keys=False), flush=True)
    print(f"Run directory: {run_dir}", flush=True)
    if args.check_only:
        print("Preflight passed. Training was not started.", flush=True)
        return
    if not os.environ.get("SLURM_JOB_ID"):
        raise RuntimeError("Run this inside your CSF3 GPU allocation, not a login node.")

    # Set before importing torch, and use this checkout's source code.
    os.environ["OMP_NUM_THREADS"] = "4"
    os.environ["MKL_NUM_THREADS"] = "4"
    os.environ["OPENBLAS_NUM_THREADS"] = "4"
    sys.path.insert(0, str(repo / "src"))
    import pytorch_lightning as pl
    import torch
    from hepsetreg.runs import run_regression

    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("Expected exactly one allocated, visible CUDA GPU.")
    if config["shape_metric"] == "emd":
        # Check the new mixed-precision loss and inference-state compatibility
        # before preloading 4.7M events. Reset the seed after this probe so it
        # does not change the actual model initialization or data shuffling.
        from hepsetreg.training.lightning_module import EventRegressionModule
        from hepsetreg.training.shape_module import ShapeRegressionModule
        probe_model = torch.nn.Linear(3, 4).cuda()
        probe_module = ShapeRegressionModule(probe_model, shape_metric="emd",
                                            shape_weight=config["shape_weight"], loss_kind="mse")
        probe_target = torch.randn(32, 4, device="cuda", dtype=torch.float32)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            probe_pred = probe_model(torch.randn(32, 3, device="cuda"))
            probe_loss = (probe_module.loss_fn(probe_pred, probe_target)
                          + config["shape_weight"] * probe_module.shape_loss_fn(probe_pred, probe_target))
        probe_loss.backward()
        if not torch.isfinite(probe_loss) or any(
                parameter.grad is None or not torch.isfinite(parameter.grad).all()
                for parameter in probe_model.parameters()):
            raise RuntimeError("MSE+EMD GPU backward check failed")
        probe_base = EventRegressionModule(torch.nn.Linear(3, 4).cuda())
        probe_base.load_state_dict(probe_module.state_dict(), strict=True)
        del probe_base, probe_module, probe_model, probe_target, probe_pred, probe_loss
        torch.cuda.empty_cache()
        print("PASS: MSE+EMD GPU backward and inference state_dict compatibility.", flush=True)
    pl.seed_everything(args.seed, workers=True)
    source = Path(run_regression.__file__).resolve()
    manifest = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "seed": args.seed, "deterministic_algorithms_requested": False,
        "source": str(source), "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        "related_source_sha256": {
            relative: hashlib.sha256((repo / relative).read_bytes()).hexdigest()
            for relative in ["src/hepsetreg/datasets/EventDatasetBase.py",
                             "src/hepsetreg/training/lightning_module.py",
                             "src/hepsetreg/training/shape_module.py",
                             "src/hepsetreg/losses/shape_losses.py"]},
        "checkpoint_selection": "minimum validation MSE" if config["shape_metric"] is None
                                else "minimum validation MSE + shape_weight * minibatch EMD",
        "git_revision": git_output(repo, "rev-parse", "HEAD"),
        "git_status": git_output(repo, "status", "--short"),
        "torch_version": torch.__version__, "lightning_version": pl.__version__,
        "gpu": torch.cuda.get_device_name(0),
        "slurm_job_id": os.environ["SLURM_JOB_ID"],
        "data": provenance, "training_config": config,
        "test_used_during_training": False,
    }
    run_dir.mkdir(parents=True, exist_ok=False)
    (run_dir / "run_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    train_section = {key: value for key, value in config.items() if key != "features"}
    (run_dir / "train_four.yaml").write_text(
        yaml.safe_dump({"features": config["features"], "train": train_section}, sort_keys=False),
        encoding="utf-8")
    shutil.copy2(Path(__file__), run_dir / "train_ttbar_four.py")
    os.chdir(work_dir)
    print("Starting fresh training on TRAIN; selecting the best checkpoint on VAL.", flush=True)
    print("Loading the full training set and compiling the model can take several minutes.", flush=True)
    run_regression.train(config)

    checkpoints = sorted((run_dir / "checkpoints").glob("*.ckpt"))
    if len(checkpoints) != 1 or not (run_dir / "stats.pt").is_file():
        raise RuntimeError(f"Training returned, but expected checkpoint/stats files are missing: {run_dir}")
    # Copy the small completed model, normalization statistics and logs to home.
    # The bulk dataset stays in scratch. Copying never overwrites an earlier run.
    backup_dir.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(run_dir, backup_dir)
    predict_config = {"features": config["features"], "predict": {
        "checkpoint": str(backup_dir / "checkpoints" / checkpoints[0].name),
        "stats": str(backup_dir / "stats.pt"),
        "data_path": str(data_dir / "ttbar-2L_test_four.parquet"),
        "batch_size": args.batch_size, "accelerator": "gpu", "shape_metric": None,
        "output": str(run_dir / "test_predictions.parquet"),
    }}
    predict_text = yaml.safe_dump(predict_config, sort_keys=False)
    (run_dir / "predict_four.yaml").write_text(predict_text, encoding="utf-8")
    (backup_dir / "predict_four.yaml").write_text(predict_text, encoding="utf-8")
    print(f"TRAINING COMPLETE. Best checkpoint: {checkpoints[0]}", flush=True)
    print(f"Model, statistics, config and logs copied to: {backup_dir}", flush=True)
    print("Independent test predictions have not been run yet.", flush=True)


if __name__ == "__main__":
    main()
