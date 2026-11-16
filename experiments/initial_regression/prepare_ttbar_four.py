#!/usr/bin/env python3
"""Prepare independent VyPER ttbar-2L splits for four-target HepSetReg.

Run from the HepSetReg checkout using its `uv run python` environment.
Only reads the downloaded ROOTs and existing test Parquet. New files go to
--output-dir. Existing completed outputs are verified by provenance before skip.

Angular convention follows Ethan Simpson's bsc physics/reco_spins.py:
lab -> ttbar CM -> each parent rest frame, K = top direction in ttbar CM.
This is a HepSetReg four-observable baseline, not the complete VyPER algorithm.
Sources:
https://zenodo.org/records/22308461
https://github.com/els285/HepSetReg/blob/bsc/src/hepsetreg/physics/prep_data.py
https://github.com/els285/HepSetReg/blob/bsc/src/hepsetreg/physics/reco_spins.py
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import os
from pathlib import Path
import tempfile

import awkward as ak
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import uproot
import vector

VERSION = "ttbar-four-v1"
TARGETS = ["ttbar_mass", "delta_abs_y", "cos_K_plus", "cos_phi"]
EXPECTED_ROWS = {"train": 4692288, "val": 260190, "test": 261557}
EXPECTED_MD5 = {
    "train": "0516b8326dc19081ce3dd4ed3b7f9914",
    "val": "7edb6a5181c2ffd624ba85e953366c37",
}
GLOBAL_NAMES = ["njet", "nelectron", "nmuon", "nbtagged", "met_met", "met_phi"]
JET_NAMES = ["pt", "eta", "phi", "e", "m", "px", "py", "pz", "bTag"]
LEPTON_NAMES = ["pt", "eta", "phi", "e", "m", "px", "py", "pz", "charge"]
META_KEY = b"hepsetreg_four_observables"


def require_fields(record, names, where):
    available = set(record.fields if hasattr(record, "fields") else record.keys())
    missing = sorted(set(names) - available)
    if missing:
        raise ValueError(f"{where}: missing {missing}; available: {sorted(available)}")


def as_numpy(values):
    return np.asarray(ak.to_numpy(values, allow_missing=False), dtype=np.float64)


def require_valid(mask, description, offset=0):
    bad = np.flatnonzero(~np.asarray(ak.to_numpy(mask), dtype=bool))
    if bad.size:
        indices = (bad[:8] + offset).tolist()
        raise ValueError(f"{description}; source entry/entries {indices}. No events were dropped.")


def rapidity(energy, pz, offset=0):
    energy, pz = as_numpy(energy), as_numpy(pz)
    require_valid(np.isfinite(energy) & np.isfinite(pz) & (energy > np.abs(pz)),
                  "Invalid rapidity input: require finite E, pz and E > |pz|", offset)
    return 0.5 * (np.log(energy + pz) - np.log(energy - pz))


def add_delta_target(events, offset=0):
    require_fields(events, ["inputs", "targets"], "Parquet event record")
    targets = events["targets"]
    require_fields(targets, ["top_E", "top_pz", "antitop_E", "antitop_pz"], "targets")
    delta = np.abs(rapidity(targets["top_E"], targets["top_pz"], offset)) - np.abs(
        rapidity(targets["antitop_E"], targets["antitop_pz"], offset))
    if "delta_abs_y" in targets.fields:
        require_valid(np.isclose(as_numpy(targets["delta_abs_y"]), delta, rtol=1e-6, atol=1e-7),
                      "Existing delta_abs_y disagrees with parent four-vectors", offset)
    return ak.with_field(events, ak.with_field(targets, delta, "delta_abs_y"), "targets")


def parent_id_field(keys):
    # Discover the branch name from the ROOT schema; never assume that index 0
    # is top and index 1 antitop. Require a unique recognizable identity field.
    candidates = []
    for name in keys:
        lower = name.lower()
        if lower.startswith("truth_parent_"):
            suffix = lower[len("truth_parent_"):].replace("_", "")
            if suffix in {"pdgid", "pdg", "id", "pid"}:
                candidates.append(name)
    if len(candidates) != 1:
        raise ValueError("Cannot unambiguously identify the parent PDG-ID branch. "
                         f"Candidates: {candidates}; truth branches: {sorted(keys)}")
    return candidates[0]


def p4_from_fields(record, prefix, mass_suffix="m"):
    return vector.zip({
        "pt": ak.values_astype(record[f"{prefix}_pt"], np.float64),
        "eta": ak.values_astype(record[f"{prefix}_eta"], np.float64),
        "phi": ak.values_astype(record[f"{prefix}_phi"], np.float64),
        "mass": ak.values_astype(record[f"{prefix}_{mass_suffix}"], np.float64),
    })


def convert_root_chunk(reco, truth, id_field, offset=0):
    if len(reco) != len(truth):
        raise ValueError("reco/truth chunk length mismatch")
    for event_id in ("eventNumber", "event_number", "event_id"):
        if event_id in reco.fields and event_id in truth.fields:
            require_valid(reco[event_id] == truth[event_id], "reco/truth event ID mismatch", offset)

    ids = truth[id_field]
    require_valid((ak.sum(ids == 6, axis=1) == 1) & (ak.sum(ids == -6, axis=1) == 1),
                  "Expected exactly one top (+6) and one antitop (-6)", offset)
    parents = p4_from_fields(truth, "truth_parent")
    top = ak.firsts(parents[ids == 6])
    antitop = ak.firsts(parents[ids == -6])
    gen_leptons = ak.concatenate([p4_from_fields(truth, "gen_el"),
                                 p4_from_fields(truth, "gen_mu")], axis=1)
    charges = ak.concatenate([truth["gen_el_charge"], truth["gen_mu_charge"]], axis=1)
    require_valid((ak.num(charges, axis=1) == 2) &
                  (ak.sum(charges == 1, axis=1) == 1) & (ak.sum(charges == -1, axis=1) == 1),
                  "Expected one positive and one negative generator lepton", offset)
    positive = ak.firsts(gen_leptons[charges == 1])
    negative = ak.firsts(gen_leptons[charges == -1])
    ttbar = top + antitop
    top_cm = top.boostCM_of(ttbar)
    anti_cm = antitop.boostCM_of(ttbar)
    positive_rest = positive.boostCM_of(ttbar).boostCM_of(top_cm)
    negative_rest = negative.boostCM_of(ttbar).boostCM_of(anti_cm)
    k = top_cm.to_beta3().unit()
    lp = positive_rest.to_beta3().unit()
    lm = negative_rest.to_beta3().unit()
    targets = ak.zip({
        "ttbar_mass": ttbar.mass,
        "top_px": top.px, "top_py": top.py, "top_pz": top.pz, "top_E": top.E,
        "antitop_px": antitop.px, "antitop_py": antitop.py,
        "antitop_pz": antitop.pz, "antitop_E": antitop.E,
        "cos_K_plus": k.dot(lp), "cos_phi": lp.dot(lm),
    }, depth_limit=1)
    inputs = reco
    for name in GLOBAL_NAMES:
        inputs = ak.with_field(inputs, inputs[name], f"global_{name}")
        inputs = ak.without_field(inputs, name)
    events = ak.zip({"inputs": inputs, "targets": targets}, depth_limit=1)
    return add_delta_target(events, offset)


def validate_events(events, offset=0):
    require_fields(events, ["inputs", "targets"], "events")
    needed = ([f"global_{x}" for x in GLOBAL_NAMES] + [f"jet_{x}" for x in JET_NAMES] +
              [f"{flavour}_{x}" for flavour in ("el", "mu") for x in LEPTON_NAMES])
    require_fields(events["inputs"], needed, "inputs")
    inputs = events["inputs"]
    for name in GLOBAL_NAMES:
        require_valid(np.isfinite(as_numpy(inputs[f"global_{name}"])),
                      f"Non-finite input global_{name}", offset)
    for kind, fields, count_name in (("jet", JET_NAMES, "njet"),
                                     ("el", LEPTON_NAMES, "nelectron"),
                                     ("mu", LEPTON_NAMES, "nmuon")):
        lengths = ak.num(inputs[f"{kind}_pt"], axis=1)
        require_valid(lengths == inputs[f"global_{count_name}"],
                      f"{kind} multiplicity does not match global_{count_name}", offset)
        for field in fields:
            values = inputs[f"{kind}_{field}"]
            require_valid(ak.num(values, axis=1) == lengths,
                          f"Misaligned input collection {kind}_{field}", offset)
            require_valid(ak.all(np.isfinite(values), axis=1),
                          f"Non-finite input {kind}_{field}", offset)
    require_valid((ak.num(inputs["el_pt"], axis=1) + ak.num(inputs["mu_pt"], axis=1)) == 2,
                  "HepSetReg requires exactly two reconstructed leptons per event", offset)
    require_fields(events["targets"], TARGETS, "targets")
    for field in TARGETS:
        values = as_numpy(events["targets"][field])
        require_valid(np.isfinite(values), f"Non-finite {field}", offset)
        if field.startswith("cos_"):
            require_valid(np.abs(values) <= 1 + 1e-6, f"Unphysical {field} outside [-1,1]", offset)
        if field == "ttbar_mass":
            require_valid(values > 0, "Nonpositive ttbar mass", offset)


def root_chunks(path, chunk_size, expected_rows):
    with ThreadPoolExecutor(max_workers=2) as workers, uproot.open(
        path, array_cache=None, handler=uproot.source.file.MultithreadedFileSource,
        num_workers=2, decompression_executor=workers, interpretation_executor=workers
    ) as source:
        reco_tree, truth_tree = source["reco"], source["truth"]
        if reco_tree.num_entries != truth_tree.num_entries:
            raise ValueError("ROOT reco and truth trees have different entry counts")
        if reco_tree.num_entries != expected_rows:
            raise ValueError(f"Expected {expected_rows:,} events, found {reco_tree.num_entries:,}")
        id_field = parent_id_field(truth_tree.keys())
        truth_names = ([f"truth_parent_{x}" for x in ("pt", "eta", "phi", "m")] + [id_field] +
                       [f"{flavour}_{x}" for flavour in ("gen_el", "gen_mu")
                        for x in ("pt", "eta", "phi", "m", "charge")])
        require_fields(truth_tree, truth_names, "ROOT truth")
        require_fields(reco_tree, GLOBAL_NAMES, "ROOT reco")
        for event_id in ("eventNumber", "event_number", "event_id"):
            if event_id in reco_tree.keys() and event_id in truth_tree.keys():
                truth_names.append(event_id)
        print(f"Parent identity branch: {id_field}", flush=True)
        for start in range(0, expected_rows, chunk_size):
            stop = min(start + chunk_size, expected_rows)
            reco = reco_tree.arrays(entry_start=start, entry_stop=stop, library="ak", array_cache=None)
            truth = truth_tree.arrays(truth_names, entry_start=start, entry_stop=stop,
                                      library="ak", array_cache=None)
            yield convert_root_chunk(reco, truth, id_field, start)


def parquet_chunks(path, chunk_size, expected_rows):
    with pq.ParquetFile(path) as source:
        if source.metadata.num_rows != expected_rows:
            raise ValueError(f"Expected {expected_rows:,} events, found {source.metadata.num_rows:,}")
        offset = 0
        for batch in source.iter_batches(batch_size=chunk_size):
            events = add_delta_target(ak.from_arrow(batch), offset)
            offset += len(events)
            yield events


def md5_file(path):
    digest = hashlib.md5()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_chunks(chunks, destination, expected_rows, provenance):
    destination = Path(destination)
    if destination.exists():
        with pq.ParquetFile(destination) as existing:
            recorded = json.loads((existing.schema_arrow.metadata or {}).get(META_KEY, b"{}"))
            if recorded == provenance and existing.metadata.num_rows == expected_rows:
                target_fields = {field.name for field in existing.schema_arrow.field("targets").type}
                if not set(TARGETS).issubset(target_fields):
                    raise ValueError(f"Existing output is missing target fields: {destination}")
                print(f"Verified existing output, skipping: {destination}", flush=True)
                return
        raise FileExistsError(f"Refusing to overwrite unverified existing output: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    handle, partial_name = tempfile.mkstemp(prefix=destination.name + ".", suffix=".partial",
                                          dir=destination.parent)
    os.close(handle)
    partial = Path(partial_name)
    writer = None
    total = 0
    try:
        for events in chunks:
            validate_events(events, total)
            table = ak.to_arrow_table(events, extensionarray=False).replace_schema_metadata(None)
            if writer is None:
                schema = table.schema.with_metadata({META_KEY: json.dumps(provenance, sort_keys=True).encode()})
                writer = pq.ParquetWriter(partial, schema, compression="zstd")
            # Data types must stay stable, including all-empty lepton collections.
            if not table.schema.equals(writer.schema, check_metadata=False):
                raise ValueError("Output schema changed between chunks; refusing unsafe coercion")
            writer.write_table(table)
            total += len(events)
            if total > expected_rows:
                raise ValueError("Converted more events than expected")
            print(f"{destination.name}: {total:,}/{expected_rows:,}", flush=True)
        if writer is not None:
            writer.close()
            writer = None
        if total != expected_rows:
            raise ValueError(f"Expected {expected_rows:,} events; converted {total:,}")
        with pq.ParquetFile(partial) as result:
            if result.metadata.num_rows != expected_rows:
                raise ValueError("Written Parquet row count does not match input")
        # Both branches publish without replacing a concurrent existing output.
        # Windows rename has no-replace semantics; POSIX rename does not.
        if os.name == "nt":
            os.rename(partial, destination)
        else:
            os.link(partial, destination)
        print(f"OK: {destination} ({total:,} events; targets: {', '.join(TARGETS)})", flush=True)
    finally:
        if writer is not None:
            writer.close()
        partial.unlink(missing_ok=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--split", choices=["val", "train", "test", "all"], required=True)
    parser.add_argument("--raw-dir", type=Path, default=Path(os.environ.get("RECO_DATA_ROOT", "external")) / "hepsetreg_data/raw")
    parser.add_argument("--output-dir", type=Path, default=Path(os.environ.get("RECO_DATA_ROOT", "external")) / "hepsetreg_data/parquet")
    parser.add_argument("--test-parquet", type=Path, default=Path("datasets/tt2Ltest.parquet"))
    parser.add_argument("--chunk-size", type=int, default=20000)
    args = parser.parse_args()
    pa.set_cpu_count(2)
    pa.set_io_thread_count(2)
    if args.chunk_size < 1:
        parser.error("--chunk-size must be positive")
    splits = ["val", "train", "test"] if args.split == "all" else [args.split]
    for split in splits:
        source = (args.test_parquet if split == "test" else args.raw_dir / f"ttbar-2L_{split}.root").expanduser().resolve()
        if not source.is_file():
            raise FileNotFoundError(source)
        print(f"Checking {split} source checksum...", flush=True)
        digest = md5_file(source)
        if split in EXPECTED_MD5 and digest != EXPECTED_MD5[split]:
            raise ValueError(f"Source MD5 mismatch for {source}: {digest}")
        provenance = {
            "converter": VERSION, "split": split, "source": str(source), "source_md5": digest,
            "events": EXPECTED_ROWS[split], "targets": TARGETS,
            "angular_convention": "HepSetReg bsc: sequential boosts; K is top direction in ttbar CM",
        }
        chunks = (parquet_chunks(source, args.chunk_size, EXPECTED_ROWS[split]) if split == "test"
                  else root_chunks(source, args.chunk_size, EXPECTED_ROWS[split]))
        output = args.output_dir.expanduser() / f"ttbar-2L_{split}_four.parquet"
        write_chunks(chunks, output, EXPECTED_ROWS[split], provenance)


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        raise SystemExit(f"Conversion failed: {type(error).__name__}: {error}") from error
