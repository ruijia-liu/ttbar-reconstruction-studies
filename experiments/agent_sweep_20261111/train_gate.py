"""Fit two cross-fitted input-only gates in an allocated Slurm GPU job.

Training weights are strictly out-of-fold. Validation weights average the two
models. Validation labels are used only for post-training diagnostics.
"""
from __future__ import annotations

import argparse
import gc
import hashlib
import importlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback

ACTIVE_STATUS = None
ACTIVE_RECORD = None


def save_json(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def digest(path):
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def diagnostics(labels, probabilities):
    """Unweighted probability diagnostics, including tie-aware average precision."""
    import numpy as np
    y = np.asarray(labels, dtype=np.bool_)
    p = np.asarray(probabilities, dtype=np.float64)
    if y.ndim != 1 or p.shape != y.shape or not len(y):
        raise ValueError("Expected nonempty aligned one-dimensional labels/probabilities")
    if not np.isfinite(p).all() or np.any((p < 0) | (p > 1)):
        raise ValueError("Invalid gate probabilities")
    selected = p >= 0.5
    tp, fp = int(np.sum(selected & y)), int(np.sum(selected & ~y))
    fn, tn = int(np.sum(~selected & y)), int(np.sum(~selected & ~y))
    safe = np.clip(p, 1e-12, 1 - 1e-12)
    bins, ece = [], 0.0
    bin_id = np.minimum((p * 10).astype(np.int64), 9)
    for b in range(10):
        mask = bin_id == b
        count = int(mask.sum())
        predicted = float(p[mask].mean()) if count else None
        observed = float(y[mask].mean()) if count else None
        if count:
            ece += count / len(y) * abs(predicted - observed)
        bins.append(dict(lower=b / 10, upper=(b + 1) / 10, n=count,
                         mean_probability=predicted, observed_fraction=observed))
    positives = int(y.sum())
    average_precision = None
    if positives:
        order = np.argsort(-p, kind="stable")
        sorted_p = p[order]
        cumulative = np.cumsum(y[order], dtype=np.int64)
        ends = np.r_[np.flatnonzero(sorted_p[1:] != sorted_p[:-1]), len(y) - 1]
        recall = cumulative[ends] / positives
        precision = cumulative[ends] / (ends + 1)
        average_precision = float(np.sum(np.diff(np.r_[0.0, recall]) * precision))
    return dict(n=len(y), positives=positives, prevalence=float(y.mean()),
                probability_mean=float(p.mean()),
                bce=float(np.mean(-(y * np.log(safe) + (~y) * np.log1p(-safe)))),
                brier=float(np.mean((p - y) ** 2)),
                threshold=0.5, precision=tp / (tp + fp) if tp + fp else None,
                recall=tp / positives if positives else None,
                confusion=dict(tp=tp, fp=fp, fn=fn, tn=tn),
                average_precision=average_precision,
                average_precision_definition="stepwise PR area; equal-score ties grouped",
                calibration_10bins=bins, expected_calibration_error=ece)


def read_native_mass(path):
    import numpy as np
    import pyarrow as pa
    import pyarrow.compute as pc
    import pyarrow.parquet as pq
    column = pq.read_table(path, columns=["targets.ttbar_mass"]).column(0)
    if pa.types.is_struct(column.type):
        column = pc.struct_field(column, "ttbar_mass")
    values = column.to_numpy(zero_copy_only=False).astype(np.float64)
    if values.ndim != 1 or not np.isfinite(values).all():
        raise ValueError("Native mass labels must be finite and one-dimensional")
    return values


def main():
    global ACTIVE_STATUS, ACTIVE_RECORD
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    started = time.monotonic()
    code = Path(__file__).resolve().parent
    plan_path = code / "experiment_plan.json"
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    cfg = plan["gate"]
    if cfg["folds"] != 2:
        raise ValueError("This gate preparation implements exactly two folds")
    out = args.output / ("gate_smoke" if args.smoke else "gate")
    out.mkdir(parents=True, exist_ok=False)
    record = dict(status="running", smoke=args.smoke, gate_config=cfg,
                  job_id=os.environ.get("SLURM_JOB_ID"),
                  row_order="source parquet order; implicit row_index=arange(N)",
                  plan_sha256=digest(plan_path))
    ACTIVE_STATUS, ACTIVE_RECORD = out / "status.json", record
    save_json(ACTIVE_STATUS, record)
    if not os.environ.get("SLURM_JOB_ID"):
        raise RuntimeError("Gate fitting requires an allocated Slurm GPU job")
    sys.path.insert(0, str(args.repo / "src"))
    import numpy as np
    import pyarrow.parquet as pq
    import torch
    from torch.nn import functional as F
    from gate_model import GateMLP, FEATURE_NAMES, TARGET_FIELDS, dataset_features, extract_gate_features

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required; do not fit on a login node")
    torch.set_num_threads(4)
    torch.set_float32_matmul_precision("highest")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    record.update(torch=torch.__version__, gpu=torch.cuda.get_device_name(),
                  precision="float32; autocast and TF32 disabled",
                  code_sha256={name: digest(code / name) for name in ("train_gate.py", "gate_model.py")})
    source_hashes = {}
    for relative, expected in plan["verified_source_sha256"].items():
        actual = digest(args.repo / "src/hepsetreg" / relative)
        if actual != expected:
            raise RuntimeError("Source hash mismatch: " + relative)
        source_hashes[relative] = actual
    record["source_sha256"] = source_hashes
    record["source_commit"] = subprocess.check_output(["git", "-C", str(args.repo), "rev-parse", "HEAD"], text=True).strip()
    canonical_path = code / "canonical_validation.npz"
    if digest(canonical_path) != plan["canonical_npz_sha256"]:
        raise RuntimeError("Canonical validation truth hash mismatch")
    with np.load(canonical_path, allow_pickle=False) as canonical:
        canonical_truth = canonical["targets"].copy()
        if canonical["keys"].tolist() != TARGET_FIELDS:
            raise RuntimeError("Canonical observable order mismatch")
        if not np.array_equal(canonical["row_index"], np.arange(plan["val_rows"])):
            raise RuntimeError("Canonical event order mismatch")
    if canonical_truth.shape != (plan["val_rows"], 4) or not np.isfinite(canonical_truth).all():
        raise RuntimeError("Invalid canonical targets")
    paths, native_mass = {}, {}
    for split, rows in (("train", plan["train_rows"]), ("val", plan["val_rows"])):
        candidates = [args.data / f"ttbar-2L_{split}_051026_augmented.parquet",
                      args.data / f"ttbar-2L_{split}_four.parquet"]
        matches = [p for p in candidates if p.is_file()]
        if len(matches) != 1:
            raise RuntimeError(f"Expected exactly one known {split} input, found {matches}")
        path = matches[0]
        if pq.ParquetFile(path).metadata.num_rows != rows or digest(path) != plan["data_sha256"][split]:
            raise RuntimeError(f"{split} input row count/hash mismatch")
        paths[split] = path
        native_mass[split] = read_native_mass(path)
        if native_mass[split].shape != (rows,):
            raise RuntimeError(f"{split} native label row count mismatch")
        record[split] = dict(path=str(path), rows=rows, sha256=plan["data_sha256"][split])
    if not np.array_equal(native_mass["val"], canonical_truth[:, 0]):
        raise RuntimeError("Native validation mass differs from canonical mass/order")
    threshold = float(cfg["threshold_gev"])
    tails = {"train": native_mass["train"] > threshold, "val": canonical_truth[:, 0] > threshold}
    dsmod = importlib.import_module("hepsetreg.datasets.EventDatasetBase")
    features = dataset_features()
    batch_size = int(cfg["batch_size"])
    dm = dsmod.EventDataModule(str(paths["train"]), str(paths["val"]), features=features, batch_size=batch_size)
    dm.setup("fit")
    target_mean, target_std = [np.asarray(a) for a in dm.stats["target"]]
    for split, dataset in (("train", dm.train_dataset), ("val", dm.val_dataset)):
        restored_mass = dataset.target_tensor[:, 0].numpy() * (target_std[0] + 1e-8) + target_mean[0]
        difference = float(np.max(np.abs(restored_mass - native_mass[split])))
        if difference > 0.01:
            raise RuntimeError(f"Dataset/native {split} mass order mismatch: {difference}")
        record[split]["dataset_mass_alignment_maxabs"] = difference
    restored = dm.val_dataset.target_tensor.numpy() * (target_std + 1e-8) + target_mean
    differences = np.max(np.abs(restored - canonical_truth), axis=0)
    if not np.all(differences <= np.array([0.01, 5e-4, 5e-4, 5e-4])):
        raise RuntimeError("Validation feature/target row order is inconsistent with canonical truth")
    record["canonical_alignment_maxabs"] = dict(zip(TARGET_FIELDS, differences.tolist()))
    input_stats = {key: (np.asarray(a).copy(), np.asarray(b).copy())
                   for key, (a, b) in dm.stats.items() if key != "target"}
    gate_features = {}
    for split, dataset in (("train", dm.train_dataset), ("val", dm.val_dataset)):
        result = torch.empty((len(dataset), 45), dtype=torch.float32)
        for start in range(0, len(dataset), batch_size):
            stop = min(start + batch_size, len(dataset))
            batch = {"global": dataset.global_tensor[start:stop],
                     "jet": dataset.jet_tensor[start:stop], "jet_mask": dataset.jet_mask[start:stop],
                     "lepton": dataset.lepton_tensor[start:stop], "lepton_mask": dataset.lepton_mask[start:stop]}
            block = extract_gate_features(batch)
            if not bool(torch.isfinite(block).all()):
                raise RuntimeError(f"Nonfinite reconstructed gate features in {split}, rows {start}:{stop}")
            result[start:stop] = block
        gate_features[split] = result
        print(f"Extracted {split}: {tuple(result.shape)} reconstructed-only features", flush=True)
    del dm, dataset, batch, block, native_mass, restored, restored_mass, result
    gc.collect()
    device = torch.device("cuda")
    x_train = gate_features.pop("train").to(device)
    x_val = gate_features.pop("val").to(device)
    y_train = torch.as_tensor(tails["train"], dtype=torch.float32, device=device)
    n_train = len(x_train)
    seed = int(cfg["seed"])
    permutation = np.random.default_rng(seed).permutation(n_train)
    fold_id = np.empty(n_train, dtype=np.uint8)
    fold_id[permutation[:n_train // 2]] = 0
    fold_id[permutation[n_train // 2:]] = 1
    del permutation
    models, optimizers, generators, train_indices, heldout_indices, fold_records = [], [], [], [], [], []
    for fold in range(2):
        torch.manual_seed(seed + fold)
        model = GateMLP().to(device)
        fit_indices = np.flatnonzero(fold_id != fold)
        excluded_indices = np.flatnonzero(fold_id == fold)
        train_indices.append(torch.as_tensor(fit_indices, device=device))
        heldout_indices.append(excluded_indices)
        models.append(model)
        optimizers.append(torch.optim.AdamW(model.parameters(), lr=float(cfg["learning_rate"]),
                                            weight_decay=float(cfg["weight_decay"])))
        generators.append(torch.Generator(device=device).manual_seed(seed + 100 + fold))
        fold_records.append(dict(model_index=fold, excluded_fold=fold,
                                 train_rows=len(fit_indices), heldout_rows=len(excluded_indices),
                                 train_indices_sha256=hashlib.sha256(fit_indices.tobytes()).hexdigest(),
                                 heldout_indices_sha256=hashlib.sha256(excluded_indices.tobytes()).hexdigest(),
                                 completed_full_epochs=0, smoke_batches=0, epoch_seconds=[], training_bce=[]))
    deadline = started + float(cfg["fit_and_setup_budget_minutes"]) * 60
    record.update(setup_seconds=time.monotonic() - started, feature_names=FEATURE_NAMES,
                  preprocessing="EventDatasetBase input statistics fitted on all training reco inputs; no validation fitting; no target enters gate",
                  fold_assignment="numpy.default_rng(seed).permutation(N), halves assigned folds 0/1; model f excludes fold f",
                  fold_assignment_sha256=hashlib.sha256(fold_id.tobytes()).hexdigest(), folds=fold_records)
    if time.monotonic() >= deadline:
        raise RuntimeError("Gate setup exhausted the 18-minute setup+fit budget")
    save_json(ACTIVE_STATUS, record)
    stopped = False
    # Alternate full fold epochs, so the first fold cannot consume the budget alone.
    for epoch in range(1 if args.smoke else int(cfg["epochs_per_fold"])):
        for fold, (model, optimizer) in enumerate(zip(models, optimizers)):
            previous_duration = fold_records[fold]["epoch_seconds"][-1] if fold_records[fold]["epoch_seconds"] else 0
            if time.monotonic() + max(5.0, previous_duration * 1.1) >= deadline:
                stopped = True
                break
            before = {name: tensor.detach().clone() for name, tensor in model.state_dict().items()}
            torch.cuda.synchronize()
            epoch_started = time.monotonic()
            model.train()
            indices = train_indices[fold]
            shuffled = indices[torch.randperm(len(indices), generator=generators[fold], device=device)]
            loss_sum, seen = 0.0, 0
            partial = False
            for start in range(0, len(shuffled), batch_size):
                if time.monotonic() >= deadline:
                    partial = True
                    break
                ids = shuffled[start:start + batch_size]
                optimizer.zero_grad(set_to_none=True)
                logits = model(x_train[ids])
                loss = F.binary_cross_entropy_with_logits(logits, y_train[ids])
                value = float(loss.detach())
                if not np.isfinite(value):
                    raise RuntimeError(f"Nonfinite gate BCE in fold {fold}")
                loss.backward()
                optimizer.step()
                loss_sum += value * len(ids)
                seen += len(ids)
                if args.smoke:
                    break
            torch.cuda.synchronize()
            duration = time.monotonic() - epoch_started
            if partial:
                model.load_state_dict(before)
                fold_records[fold]["discarded_partial_epoch_rows"] = seen
                stopped = True
                break
            if not args.smoke and seen != len(indices):
                raise RuntimeError("A production gate epoch did not cover its complete training fold")
            fold_records[fold]["completed_full_epochs"] += int(not args.smoke)
            fold_records[fold]["smoke_batches"] += int(args.smoke)
            fold_records[fold]["epoch_seconds"].append(duration)
            fold_records[fold]["training_bce"].append(loss_sum / seen)
            print(f"Gate fold {fold}, epoch {epoch + 1}, rows {seen}, BCE {loss_sum / seen:.6f}, {duration:.2f}s", flush=True)
            save_json(ACTIVE_STATUS, record)
        if stopped:
            break
    if args.smoke:
        if not all(f["smoke_batches"] == 1 for f in fold_records):
            raise RuntimeError("Both smoke folds must complete one training batch")
    elif not all(f["completed_full_epochs"] >= 1 for f in fold_records):
        raise RuntimeError("Both production folds must complete at least one full epoch")
    record.update(fit_and_setup_seconds=time.monotonic() - started, budget_stopped=stopped,
                  checkpoint_selection="last complete epoch per fold; no validation selection",
                  smoke_note="one batch per fold, not a production gate" if args.smoke else None)
    train_probability = np.full(n_train, np.nan, dtype=np.float32)
    val_probability = np.zeros(len(x_val), dtype=np.float32)
    with torch.inference_mode():
        for fold, model in enumerate(models):
            model.eval()
            model.requires_grad_(False)
            excluded = heldout_indices[fold]
            for start in range(0, len(excluded), batch_size):
                ids = excluded[start:start + batch_size]
                gpu_ids = torch.as_tensor(ids, device=device)
                train_probability[ids] = model(x_train[gpu_ids]).sigmoid().cpu().numpy()
            for start in range(0, len(x_val), batch_size):
                val_probability[start:start + batch_size] += 0.5 * model(x_val[start:start + batch_size]).sigmoid().cpu().numpy()
    arrays = {"train_gate.npy": train_probability, "val_gate.npy": val_probability,
              "train_tail.npy": tails["train"], "val_tail.npy": tails["val"]}
    for name, array in arrays.items():
        if name.endswith("gate.npy") and (not np.isfinite(array).all() or np.any((array < 0) | (array > 1))):
            raise RuntimeError("Incomplete or invalid gate probabilities: " + name)
        temporary = out / (name + ".tmp")
        with temporary.open("wb") as stream:
            np.save(stream, array, allow_pickle=False)
        temporary.replace(out / name)
    diag = dict(train_oof=diagnostics(tails["train"], train_probability),
                validation_mean=diagnostics(tails["val"], val_probability),
                folds={str(f): diagnostics(tails["train"][ids], train_probability[ids])
                       for f, ids in enumerate(heldout_indices)},
                validation_role="post-fit diagnostics only; no gate selection, calibration, or threshold tuning",
                threshold_gev=threshold, smoke=args.smoke)
    save_json(out / "diagnostics.json", diag)
    torch.save(dict(state_dicts=[{k: v.detach().cpu() for k, v in m.state_dict().items()} for m in models],
                    architecture="GateMLP(45,64,32,1), SiLU, no dropout", feature_names=FEATURE_NAMES,
                    dataset_features=features, input_stats=input_stats, gate_config=cfg,
                    fold_id=torch.from_numpy(fold_id), folds=fold_records,
                    data_sha256=plan["data_sha256"], smoke=args.smoke), out / "gate_models.pt")
    record.update(status="smoke_passed" if args.smoke else "complete", elapsed_seconds=time.monotonic() - started,
                  train_probability_mean=float(train_probability.astype(np.float64).mean()),
                  train_tail_fraction=float(tails["train"].mean()), val_tail_fraction=float(tails["val"].mean()))
    save_json(ACTIVE_STATUS, record)
    artifacts = {}
    for name, array in arrays.items():
        artifacts[name] = dict(sha256=digest(out / name), shape=list(array.shape), dtype=str(array.dtype))
    for name in ("gate_models.pt", "status.json", "diagnostics.json"):
        artifacts[name] = dict(sha256=digest(out / name), bytes=(out / name).stat().st_size)
    manifest = dict(status=record["status"], smoke=args.smoke, rows={"train": n_train, "val": len(x_val)},
                    data_sha256=plan["data_sha256"], canonical_npz_sha256=plan["canonical_npz_sha256"],
                    row_order=record["row_order"], artifacts=artifacts,
                    train_probability_mean=record["train_probability_mean"],
                    train_tail_fraction=record["train_tail_fraction"], threshold_gev=threshold,
                    folds=fold_records, fold_assignment_sha256=record["fold_assignment_sha256"],
                    source_sha256=source_hashes, code_sha256=record["code_sha256"],
                    plan_sha256=record["plan_sha256"], elapsed_seconds=record["elapsed_seconds"],
                    train_probability_definition="model f predicts only rows assigned fold f; it was trained exclusively on the other fold",
                    val_probability_definition="arithmetic mean of both frozen models")
    save_json(out / "artifact_manifest.json", manifest)
    print(json.dumps(dict(status=record["status"], elapsed_seconds=record["elapsed_seconds"],
                          epochs=[f["completed_full_epochs"] for f in fold_records],
                          train_probability_mean=record["train_probability_mean"],
                          validation_brier=diag["validation_mean"]["brier"])), flush=True)


if __name__ == "__main__":
    try:
        main()
    except BaseException as exc:
        if ACTIVE_STATUS is not None and ACTIVE_RECORD is not None:
            ACTIVE_RECORD.update(status="failed", error=f"{type(exc).__name__}: {exc}", traceback=traceback.format_exc())
            save_json(ACTIVE_STATUS, ACTIVE_RECORD)
        raise
