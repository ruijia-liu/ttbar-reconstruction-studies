#!/usr/bin/env python3
"""Fix optional target arrays promoting z-scored MSE targets to float64.

Run beside train_ttbar_four.py in the CSF3 HepSetReg checkout. Backs up the
loader, changes one conversion, then checks small TRAIN/VAL samples and an
actual MSE backward pass. Existing Parquet datasets are never modified.
"""
from __future__ import annotations

import ast
from copy import deepcopy
import hashlib
import os
from pathlib import Path
import tempfile

OLD = '[ak.to_numpy(targets[f]).astype(np.float32) for f in features["target"]]'
NEW = '[np.asarray(ak.to_numpy(targets[f], allow_missing=False), dtype=np.float32) for f in features["target"]]'
TARGETS = ["ttbar_mass", "delta_abs_y", "cos_K_plus", "cos_phi"]


def patched_source(source):
    if source.count(NEW) == 1 and OLD not in source:
        return source
    if source.count(OLD) != 1 or NEW in source:
        raise ValueError("Loader differs from the inspected version; no changes made. Send this output for review.")
    result = source.replace(OLD, NEW)
    ast.parse(result)
    return result


def main():
    if not os.environ.get("SLURM_JOB_ID"):
        raise RuntimeError("Run inside your current CSF3 compute-node terminal.")
    repo = Path(__file__).resolve().parent
    source_path = repo / "src/hepsetreg/datasets/EventDatasetBase.py"
    source = source_path.read_text(encoding="utf-8")
    fixed = patched_source(source)
    if fixed != source:
        backup = source_path.with_suffix(".py.before_ttbar_dtype")
        if backup.exists():
            if backup.read_text(encoding="utf-8") != source:
                raise RuntimeError(f"An incompatible backup already exists: {backup}")
        else:
            with backup.open("x", encoding="utf-8") as stream:
                stream.write(source)
        source_path.write_text(fixed, encoding="utf-8")
        print(f"Patched target conversion. Original saved at: {backup}", flush=True)
    else:
        print("Target conversion already patched; running checks.", flush=True)

    os.environ["OMP_NUM_THREADS"] = "4"
    os.environ["MKL_NUM_THREADS"] = "4"
    os.environ["OPENBLAS_NUM_THREADS"] = "4"
    import sys
    sys.path.insert(0, str(repo / "src"))
    import numpy as np
    import pyarrow as pa
    import pyarrow.parquet as pq
    import torch
    import yaml
    from hepsetreg.datasets.EventDatasetBase import EventDataset
    from hepsetreg.losses.losses import RegressionLoss

    pa.set_cpu_count(4)
    pa.set_io_thread_count(2)
    torch.set_num_threads(4)
    features = deepcopy(yaml.safe_load((repo / "configs/run_regression.yaml").read_text())["features"])
    features["target"] = TARGETS.copy()
    data_root = Path(os.environ.get("RECO_DATA_ROOT", "external")) / "hepsetreg_data"
    with tempfile.TemporaryDirectory(prefix="dtype_check_", dir=data_root) as temporary:
        samples = {}
        for split in ("train", "val"):
            path = data_root / "parquet" / f"ttbar-2L_{split}_four.parquet"
            with pq.ParquetFile(path) as parquet:
                batch = next(parquet.iter_batches(batch_size=128))
                sample = Path(temporary) / f"{split}.parquet"
                pq.write_table(pa.Table.from_batches([batch]), sample)
                samples[split] = str(sample)
        train = EventDataset(samples["train"], features=features)
        val = EventDataset(samples["val"], features=features,
                           stats=train.stats, max_jets=train.max_jets)
        for split, dataset in (("train", train), ("val", val)):
            target = dataset.target_tensor
            print(f"{split}: target dtype={target.dtype}; shape={tuple(target.shape)}", flush=True)
            if target.dtype != torch.float32 or not torch.isfinite(target).all():
                raise RuntimeError(f"{split}: invalid normalized targets after the patch")
            for name in ("global_tensor", "jet_tensor", "lepton_tensor"):
                tensor = getattr(dataset, name)
                if tensor.dtype != torch.float32 or not torch.isfinite(tensor).all():
                    raise RuntimeError(f"{split}: unexpected dtype or non-finite input in {name}")
            pred = torch.zeros_like(target, requires_grad=True)
            loss = RegressionLoss(kind="mse")(pred, target)
            loss.backward()
            if loss.dtype != torch.float32 or pred.grad is None or not torch.isfinite(pred.grad).all():
                raise RuntimeError(f"{split}: MSE backward check failed")
        if any(np.asarray(x).dtype != np.float32 for x in train.stats["target"]):
            raise RuntimeError("Target normalization statistics are not float32")
    print(f"Loader SHA256: {hashlib.sha256(source_path.read_bytes()).hexdigest()}", flush=True)
    print("PASS: TRAIN/VAL float32 targets and MSE backward. Full datasets were not changed.", flush=True)
    print("Restart with a new run name: ttbar4_mse_seed42_v2", flush=True)


if __name__ == "__main__":
    main()
