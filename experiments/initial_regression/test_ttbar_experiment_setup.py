"""Check fair loss settings and validation-only inference without torch/GPU."""
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

WORKSPACE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(WORKSPACE / "tmp/ttbar4_packages"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import pyarrow as pa
import pyarrow.parquet as pq
import yaml

import predict_ttbar_validation as validation
import train_ttbar_four as training


class TrainingConfigurationTests(unittest.TestCase):
    def test_comparison_changes_only_loss_and_run_identity(self):
        base = {
            "features": {
                "target": ["old_target"],
                "jet": ["pt", "eta", "phi", "energy"],
                "global": ["met"],
            },
            "train": {
                "train_path": "old_train.parquet", "val_path": "old_val.parquet",
                "shape_metric": "old_loss", "shape_weight": 99.0,
                "custom_nested": {"labels": ["preserved"]},
            },
            "predict": {"data_path": "must_not_be_used_test.parquet"},
        }
        original = deepcopy(base)
        common = dict(batch_size=4096, learning_rate=1e-3, max_epochs=50)
        data_dir = WORKSPACE / "tmp/ttbar4_test_runs/config_only_data"
        mse = training.build_config(base, data_dir, SimpleNamespace(
            **common, run_name="mse_seed42", shape_metric="none", shape_weight=7.0))
        emd = training.build_config(base, data_dir, SimpleNamespace(
            **common, run_name="emd_seed42", shape_metric="emd", shape_weight=0.1))

        differences = {key for key in mse.keys() | emd.keys() if mse.get(key) != emd.get(key)}
        self.assertEqual(differences, {"run_name", "shape_metric", "shape_weight"})
        self.assertIsNone(mse["shape_metric"])
        self.assertEqual(mse["shape_weight"], 0.0)
        self.assertEqual((emd["shape_metric"], emd["shape_weight"]), ("emd", 0.1))
        self.assertEqual(mse["features"]["target"], training.TARGETS)
        self.assertNotIn("predict", mse)
        self.assertEqual(Path(mse["train_path"]), (data_dir / "ttbar-2L_train_four.parquet").resolve())
        self.assertEqual(Path(mse["val_path"]), (data_dir / "ttbar-2L_val_four.parquet").resolve())
        self.assertNotEqual(mse["train_path"], mse["val_path"])
        self.assertEqual(base, original)

        # Mutating a run must not change the original config or the other run.
        mse["features"]["jet"].append("changed")
        mse["features"]["target"].reverse()
        mse["custom_nested"]["labels"].append("changed")
        self.assertEqual(base, original)
        self.assertEqual(emd["features"]["jet"], original["features"]["jet"])
        self.assertEqual(emd["features"]["target"], training.TARGETS)
        self.assertEqual(emd["custom_nested"], original["train"]["custom_nested"])


class ValidationConfigurationTests(unittest.TestCase):
    def setUp(self):
        scratch = WORKSPACE / "tmp/ttbar4_test_runs"
        scratch.mkdir(parents=True, exist_ok=True)
        self.temporary = tempfile.TemporaryDirectory(prefix="experiment_setup_", dir=scratch)
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.run = self.root / "trained_runs/emd_seed42"
        self.run.mkdir(parents=True)
        self.output_dir = self.root / "validation_results/emd_seed42"
        self.val_path = self.root / "official_val.parquet"
        self.test_path = self.root / "NONEXISTENT_TEST_MUST_NOT_BE_READ.parquet"
        self.old_output = self.root / "old_test_predictions.parquet"
        self.old_output.write_bytes(b"existing test predictions must be preserved")
        self.checkpoint_bytes = b"dummy saved checkpoint\x00\x01"
        self.stats_bytes = b"this run's independent scaling statistics\x00\x02"
        self.checkpoint = self.run / "best.ckpt"
        self.stats = self.run / "stats.pt"
        self.checkpoint.write_bytes(self.checkpoint_bytes)
        self.stats.write_bytes(self.stats_bytes)
        self.features = {"target": validation.TARGETS.copy(), "jet": ["pt", "eta"]}
        self.manifest = {"training_config": {
            "features": deepcopy(self.features), "val_path": str(self.val_path),
            "shape_metric": "emd", "shape_weight": 0.1,
        }}
        self.predict_saved = {
            "features": deepcopy(self.features),
            "predict": {
                "checkpoint": str(self.checkpoint), "stats": str(self.stats),
                "data_path": str(self.test_path), "output": str(self.old_output),
                "shape_metric": "emd", "batch_size": 4096, "accelerator": "gpu",
            },
        }
        self.write_configs()
        self.write_validation()
        self.rows_patch = patch.object(validation, "VAL_ROWS", 3)
        self.rows_patch.start()
        self.addCleanup(self.rows_patch.stop)

    def write_configs(self):
        (self.run / "run_manifest.json").write_text(json.dumps(self.manifest), encoding="utf-8")
        (self.run / "predict_four.yaml").write_text(yaml.safe_dump(self.predict_saved), encoding="utf-8")

    def write_validation(self, rows=3, split="val", metadata=True):
        table = pa.table({"event_index": list(range(rows))})
        if metadata:
            provenance = {"split": split, "targets": validation.TARGETS, "source": "fixture.root"}
            table = table.replace_schema_metadata({
                b"hepsetreg_four_observables": json.dumps(provenance).encode("utf-8")})
        pq.write_table(table, self.val_path)

    def test_prepares_only_validation_and_preserves_saved_test_configuration(self):
        before_yaml = (self.run / "predict_four.yaml").read_bytes()
        before_manifest = (self.run / "run_manifest.json").read_bytes()
        before_test_predictions = self.old_output.read_bytes()
        opened = []
        real_parquet_file = validation.pq.ParquetFile

        def only_validation(path, *args, **kwargs):
            resolved = Path(path).resolve()
            self.assertEqual(resolved, self.val_path.resolve(), "test data must never be opened")
            opened.append(resolved)
            return real_parquet_file(path, *args, **kwargs)

        with patch.object(validation.pq, "ParquetFile", side_effect=only_validation):
            config, identity = validation.prepare_config(self.run, self.output_dir)
        self.assertEqual(opened, [self.val_path.resolve()])
        self.assertFalse(self.test_path.exists())
        self.assertFalse(self.output_dir.exists(), "preparation should not create predictions")
        self.assertEqual(config["data_path"], str(self.val_path.resolve()))
        self.assertEqual(config["output"], str(self.output_dir / "val_predictions.parquet"))
        self.assertEqual(config["features"], self.features)
        self.assertIsNone(config["shape_metric"])
        self.assertEqual(config["checkpoint"], str(self.checkpoint))
        self.assertEqual(config["stats"], str(self.stats))
        self.assertEqual(identity["split"], "val")
        self.assertEqual(identity["events"], 3)
        self.assertEqual(identity["training_shape_metric"], "emd")
        self.assertEqual(identity["training_shape_weight"], 0.1)
        self.assertEqual(identity["checkpoint_sha256"], hashlib.sha256(self.checkpoint_bytes).hexdigest())
        self.assertEqual(identity["stats_sha256"], hashlib.sha256(self.stats_bytes).hexdigest())
        self.assertEqual(identity["validation_conversion"]["split"], "val")
        config["features"]["target"].reverse()
        self.assertEqual(identity["features"]["target"], validation.TARGETS)
        self.assertEqual((self.run / "predict_four.yaml").read_bytes(), before_yaml)
        self.assertEqual((self.run / "run_manifest.json").read_bytes(), before_manifest)
        self.assertEqual(self.old_output.read_bytes(), before_test_predictions)

    def test_rejects_wrong_split_incomplete_count_and_missing_provenance(self):
        for settings in [dict(split="test"), dict(rows=2), dict(metadata=False)]:
            with self.subTest(settings=settings):
                self.write_validation(**settings)
                with self.assertRaisesRegex(ValueError, "official validation split"):
                    validation.prepare_config(self.run, self.output_dir)

    def test_rejects_wrong_target_order_before_loading_any_dataset(self):
        self.manifest["training_config"]["features"]["target"].reverse()
        self.write_configs()
        with patch.object(validation.pq, "ParquetFile") as opener:
            with self.assertRaisesRegex(ValueError, "four-target model"):
                validation.prepare_config(self.run, self.output_dir)
            opener.assert_not_called()

    def test_identity_changes_when_run_scaling_or_checkpoint_changes(self):
        _, original = validation.prepare_config(self.run, self.output_dir)
        self.stats.write_bytes(b"different run statistics")
        _, changed_stats = validation.prepare_config(self.run, self.output_dir)
        self.assertNotEqual(original["stats_sha256"], changed_stats["stats_sha256"])
        self.assertEqual(original["checkpoint_sha256"], changed_stats["checkpoint_sha256"])
        self.checkpoint.write_bytes(b"different checkpoint")
        _, changed_checkpoint = validation.prepare_config(self.run, self.output_dir)
        self.assertNotEqual(changed_stats["checkpoint_sha256"], changed_checkpoint["checkpoint_sha256"])
        self.assertEqual(changed_stats["stats_sha256"], changed_checkpoint["stats_sha256"])

    def test_missing_run_artifact_fails_instead_of_falling_back_to_another_run(self):
        self.predict_saved["predict"]["stats"] = str(self.run / "nonexistent_stats.pt")
        self.write_configs()
        with self.assertRaises(FileNotFoundError):
            validation.prepare_config(self.run, self.output_dir)


if __name__ == "__main__":
    unittest.main()
