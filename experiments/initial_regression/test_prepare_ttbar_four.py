"""Synthetic I/O and physics checks; no claim to replace a real-data pilot."""
import contextlib
import io
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tmp/ttbar4_packages"))
import awkward as ak
import numpy as np
import uproot

import prepare_ttbar_four as prep


def example_records():
    counts_el = [0, 1, 2, 0, 2, 1, 0]
    reco = {"eventNumber": np.arange(1000, 1007, dtype=np.int64),
            "njet": np.full(7, 2, dtype=np.int32),
            "nelectron": np.array(counts_el, dtype=np.int32),
            "nmuon": 2 - np.array(counts_el, dtype=np.int32),
            "nbtagged": np.full(7, 2, dtype=np.int32),
            "met_met": np.full(7, 35.0, dtype=np.float32),
            "met_phi": np.full(7, 0.8, dtype=np.float32)}
    for field in prep.JET_NAMES:
        reco[f"jet_{field}"] = ak.Array([[55.0, 43.0] for _ in counts_el])
    for flavour in ("el", "mu"):
        for field in prep.LEPTON_NAMES:
            reco[f"{flavour}_{field}"] = ak.Array([
                [float(j + 1) for j in range(n if flavour == "el" else 2 - n)]
                for n in counts_el])
    truth = {"eventNumber": reco["eventNumber"].copy()}
    parents = []
    for i in range(7):
        top = (6, 80.0 + i, 0.6, 0.2, 172.5)
        anti = (-6, 65.0, -0.5, 2.8, 172.5)
        parents.append([top, anti] if i % 2 == 0 else [anti, top])
    parents[4].insert(0, (21, 10.0, 0.1, -0.1, 0.0))
    for col, name in enumerate(["pdgId", "pt", "eta", "phi", "m"]):
        truth[f"truth_parent_{name}"] = ak.Array([[p[col] for p in row] for row in parents])
    positive = (45.0, 0.3, 0.8, 0.001, 1)
    negative = (35.0, -0.2, -1.0, 0.001, -1)
    for col, field in enumerate(["pt", "eta", "phi", "m", "charge"]):
        electrons, muons = [], []
        for i, count in enumerate(counts_el):
            leps = [positive, negative] if i % 2 else [negative, positive]
            electrons.append([p[col] for p in leps[:count]])
            muons.append([p[col] for p in leps[count:]])
        truth[f"gen_el_{field}"] = ak.Array(electrons)
        truth[f"gen_mu_{field}"] = ak.Array(muons)
    return reco, truth


def make_root(path, mismatch=False):
    reco, truth = example_records()
    def definitions(values):
        return {key: ("var * int32" if key.endswith("pdgId") else "var * float32")
                if isinstance(value, ak.Array) else value.dtype
                for key, value in values.items()}
    with uproot.recreate(path) as root:
        root.mktree("reco", definitions(reco)).extend(reco)
        if mismatch:
            truth = {key: value[:-1] for key, value in truth.items()}
        root.mktree("truth", definitions(truth)).extend(truth)


class ConversionTests(unittest.TestCase):
    def setUp(self):
        scratch = Path(__file__).resolve().parents[1] / "tmp/ttbar4_test_runs"
        scratch.mkdir(parents=True, exist_ok=True)
        self.temp = tempfile.TemporaryDirectory(dir=scratch)
        self.folder = Path(self.temp.name)

    def tearDown(self):
        self.temp.cleanup()

    def write(self, chunks, path, rows=7, metadata=None):
        with contextlib.redirect_stdout(io.StringIO()):
            prep.write_chunks(chunks, path, rows, metadata or {"test": True})

    def test_rapidity_uses_massive_energy_not_eta(self):
        np.testing.assert_allclose(prep.rapidity([5.0, 5.0], [3.0, -3.0]),
                                   [np.log(2), -np.log(2)])
        with self.assertRaisesRegex(ValueError, "source entry/entries \\[11\\]"):
            prep.rapidity([5.0, 3.0], [2.0, 3.0], offset=10)

    def test_angles_against_analytic_rest_frame_directions(self):
        reco, _ = example_records()
        reco = ak.Array(reco)[1:2]
        et = np.sqrt(172.5**2 + 100.0**2)
        beta, gamma = 100.0 / et, et / 172.5
        # Known directions in the two rest frames: (0.8,0.6,0) and
        # (0.2,sqrt(0.96),0). Boost analytically along +/-x into the lab.
        lp_px, lp_py = gamma * (40 + beta * 50), 30.0
        lm_px, lm_py = gamma * (10 - beta * 50), np.sqrt(2400.0)
        truth = ak.Array({
            "eventNumber": [1001], "truth_parent_pdgId": [[6, -6]],
            "truth_parent_pt": [[100.0, 100.0]], "truth_parent_eta": [[0.0, 0.0]],
            "truth_parent_phi": [[0.0, np.pi]], "truth_parent_m": [[172.5, 172.5]],
            "gen_el_pt": [[np.hypot(lp_px, lp_py)]], "gen_el_eta": [[0.0]],
            "gen_el_phi": [[np.arctan2(lp_py, lp_px)]], "gen_el_m": [[0.0]],
            "gen_el_charge": [[1]], "gen_mu_pt": [[np.hypot(lm_px, lm_py)]],
            "gen_mu_eta": [[0.0]], "gen_mu_phi": [[np.arctan2(lm_py, lm_px)]],
            "gen_mu_m": [[0.0]], "gen_mu_charge": [[-1]],
        })
        targets = prep.convert_root_chunk(reco, truth, "truth_parent_pdgId").targets
        np.testing.assert_allclose(ak.to_numpy(targets.cos_K_plus), [0.8], atol=1e-12)
        np.testing.assert_allclose(ak.to_numpy(targets.cos_phi), [0.16 + 0.6*np.sqrt(0.96)], atol=1e-12)
        np.testing.assert_allclose(ak.to_numpy(targets.ttbar_mass), [2*et], atol=1e-12)
        np.testing.assert_allclose(ak.to_numpy(targets.delta_abs_y), [0.0], atol=1e-12)

    def test_streamed_root_chunk_size_invariant_and_order(self):
        root = self.folder / "events.root"
        make_root(root)
        outputs = []
        for size in (1, 2, 10):
            dest = self.folder / f"events-{size}.parquet"
            self.write(prep.root_chunks(root, size, 7), dest)
            outputs.append(ak.from_parquet(dest))
        for out in outputs:
            np.testing.assert_array_equal(ak.to_numpy(out.inputs.eventNumber), np.arange(1000, 1007))
            self.assertEqual(ak.to_list(ak.num(out.inputs.el_pt, axis=1)), [0, 1, 2, 0, 2, 1, 0])
            for target in prep.TARGETS:
                np.testing.assert_allclose(ak.to_numpy(out.targets[target]),
                                           ak.to_numpy(outputs[0].targets[target]), rtol=1e-12)
        # Independently compute mass and rapidities from the input p4 definitions.
        pt = 80.0 + np.arange(7)
        pzt, pza = pt * np.sinh(0.6), 65 * np.sinh(-0.5)
        et = np.sqrt(172.5**2 + pt**2 * np.cosh(0.6)**2)
        ea = np.sqrt(172.5**2 + 65**2 * np.cosh(-0.5)**2)
        delta = np.abs(np.arctanh(pzt / et)) - np.abs(np.arctanh(pza / ea))
        np.testing.assert_allclose(ak.to_numpy(outputs[0].targets.delta_abs_y), delta, atol=1e-7)
        px = pt*np.cos(0.2) + 65*np.cos(2.8)
        py = pt*np.sin(0.2) + 65*np.sin(2.8)
        mass = np.sqrt((et + ea)**2 - px**2 - py**2 - (pzt + pza)**2)
        np.testing.assert_allclose(ak.to_numpy(outputs[0].targets.ttbar_mass), mass, atol=1e-5)

    def test_parquet_append_preserves_original_fields(self):
        reco, truth = example_records()
        original = prep.convert_root_chunk(ak.Array(reco), ak.Array(truth), "truth_parent_pdgId")
        original = ak.with_field(original, ak.without_field(original.targets, "delta_abs_y"), "targets")
        source, dest = self.folder / "test.parquet", self.folder / "new.parquet"
        ak.to_parquet(original, source)
        self.write(prep.parquet_chunks(source, 2, 7), dest)
        result = ak.from_parquet(dest)
        self.assertEqual(ak.to_list(result.inputs), ak.to_list(original.inputs))
        for name in original.targets.fields:
            np.testing.assert_array_equal(ak.to_numpy(result.targets[name]), ak.to_numpy(original.targets[name]))
        self.assertIn("delta_abs_y", result.targets.fields)

    def test_mismatched_trees_do_not_publish_output(self):
        root, dest = self.folder / "bad.root", self.folder / "bad.parquet"
        make_root(root, mismatch=True)
        with self.assertRaisesRegex(ValueError, "different entry counts"):
            self.write(prep.root_chunks(root, 2, 7), dest)
        self.assertFalse(dest.exists())
        self.assertFalse(list(self.folder.glob("*.partial")))

    def test_partial_failure_leaves_no_final_or_temporary_file(self):
        reco, truth = example_records()
        events = prep.convert_root_chunk(ak.Array(reco), ak.Array(truth), "truth_parent_pdgId")
        def chunks():
            yield events[:2]
            raise RuntimeError("synthetic interruption")
        dest = self.folder / "partial.parquet"
        with self.assertRaisesRegex(RuntimeError, "synthetic interruption"):
            self.write(chunks(), dest)
        self.assertFalse(dest.exists())
        self.assertFalse(list(self.folder.glob("*.partial")))

    def test_existing_output_skip_or_reject_without_overwrite(self):
        reco, truth = example_records()
        events = prep.convert_root_chunk(ak.Array(reco), ak.Array(truth), "truth_parent_pdgId")
        dest = self.folder / "existing.parquet"
        self.write([events], dest)
        before = dest.read_bytes()
        self.write(iter([]), dest)  # verified provenance permits an idempotent skip
        self.assertEqual(dest.read_bytes(), before)
        with self.assertRaises(FileExistsError):
            self.write([events], dest, metadata={"test": "different source"})
        self.assertEqual(dest.read_bytes(), before)

    def test_invalid_truth_labels_fail(self):
        reco, truth = example_records()
        truth["truth_parent_pdgId"] = ak.Array([[6, 6] for _ in range(7)])
        with self.assertRaisesRegex(ValueError, "exactly one top"):
            prep.convert_root_chunk(ak.Array(reco), ak.Array(truth), "truth_parent_pdgId")

    def test_wrong_total_rows_rejected(self):
        root, dest = self.folder / "events.root", self.folder / "wrong.parquet"
        make_root(root)
        with self.assertRaisesRegex(ValueError, "Expected 8 events, found 7"):
            self.write(prep.root_chunks(root, 2, 8), dest, rows=8)
        self.assertFalse(dest.exists())


if __name__ == "__main__":
    unittest.main(verbosity=2)
