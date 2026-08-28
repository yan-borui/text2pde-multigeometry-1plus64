from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import h5py
import numpy as np

from dataset.multigeometry import WINDOW_MANIFEST_SCHEMA
from tools.multigeometry.build_handoff_data import (
    PREPARED_JSON_FILES,
    build_handoff_data,
)
from tools.multigeometry.materialize_config import materialize_config
from tools.multigeometry.verify_handoff_data import verify_handoff_data


class HandoffDataTest(unittest.TestCase):
    def make_trajectory(self, root: Path, split: str, case_id: str) -> dict:
        file_path = root / f"{case_id}.h5"
        nodes = 4
        with h5py.File(file_path, "w") as handle:
            handle.create_dataset(
                "uvp", data=np.ones((65, nodes, 3), dtype=np.float32)
            )
            handle.create_dataset(
                "points",
                data=np.array(
                    [[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]],
                    dtype=np.float32,
                ),
            )
            handle.create_dataset(
                "cells", data=np.array([[0, 1, 2], [0, 2, 3]], dtype=np.int32)
            )
            handle.create_dataset(
                "boundary", data=np.array([0, 2, 3, 4], dtype=np.uint8)
            )
            handle.create_dataset(
                "time", data=np.arange(65, dtype=np.float64) * 0.1
            )
        return {
            "case_id": case_id,
            "geometry_family": "ellipse",
            "node_count": nodes,
            "slot": 0,
            "source_h5_sha256": f"recorded-{case_id}",
            "source_path": str(file_path),
            "split": split,
            "trajectory_index": 0,
        }

    def make_manifest(self, split: str, trajectory: dict) -> dict:
        return {
            "schema": WINDOW_MANIFEST_SCHEMA,
            "split": split,
            "window_length": 65,
            "window_mode": "explicit",
            "window_count": 1,
            "trajectories": [trajectory],
            "windows": [{"trajectory_index": 0, "start": 0}],
        }

    def test_handoff_rewrites_paths_and_preserves_test_seal(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            prepared_dir = root / "prepared_source"
            prepared_dir.mkdir()
            records = {
                "train": self.make_trajectory(root, "train", "ellipse_00"),
                "validation": self.make_trajectory(
                    root, "validation", "ellipse_51"
                ),
                "test": self.make_trajectory(root, "test", "ellipse_54"),
            }
            manifests = {
                "train_windows.json": self.make_manifest("train", records["train"]),
                "validation_monitor_windows.json": self.make_manifest(
                    "validation", records["validation"]
                ),
                "validation_full_windows.json": self.make_manifest(
                    "validation", records["validation"]
                ),
                "test_full_windows_sealed.json": self.make_manifest(
                    "test", records["test"]
                ),
                "largest_train_smoke_windows.json": self.make_manifest(
                    "train", records["train"]
                ),
                "largest_validation_smoke_window.json": self.make_manifest(
                    "validation", records["validation"]
                ),
                "preparation_summary.json": {"largest_train": records["train"]},
                "train_validation_trajectory_audit.json": [
                    records["train"],
                    records["validation"],
                ],
            }
            self.assertEqual(set(manifests), set(PREPARED_JSON_FILES))
            for name, payload in manifests.items():
                (prepared_dir / name).write_text(
                    json.dumps(payload), encoding="utf-8"
                )
            (prepared_dir / "train_normal_stats.pkl").write_bytes(b"stats")

            data_dir = root / "handoff_data"
            inventory = build_handoff_data(
                prepared_dir, data_dir, file_mode="hardlink"
            )
            self.assertEqual(inventory["split_counts"], {"test": 1, "train": 1, "validation": 1})
            portable_train = json.loads(
                (data_dir / "prepared" / "train_windows.json").read_text(
                    encoding="utf-8"
                )
            )
            self.assertEqual(
                portable_train["trajectories"][0]["source_path"],
                "../trajectories/train/ellipse_00.h5",
            )

            verification = verify_handoff_data(data_dir)
            self.assertEqual(verification["trajectory_count"], 3)
            self.assertEqual(verification["sealed_test_windows"], 1)
            self.assertFalse(verification["test_fields_read"])

    def test_config_materialization_binds_only_data_and_result_paths(self) -> None:
        template = {
            "data": {
                "dataset": {
                    "train_manifest": None,
                    "validation_manifest": None,
                },
                "normalizer": {"stat_path": None},
            },
            "training": {"default_root_dir": None, "run_dir": None},
            "model": {"hidden_size": 512},
        }
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            prepared_dir = root / "data" / "prepared"
            result_root = root / "results"
            config = materialize_config(
                template, prepared_dir, result_root, stage="ae"
            )
            self.assertEqual(
                config["data"]["dataset"]["train_manifest"],
                str(prepared_dir / "train_windows.json"),
            )
            self.assertEqual(
                config["training"]["run_dir"], str(result_root / "ae" / "formal")
            )
            self.assertEqual(config["model"], template["model"])


if __name__ == "__main__":
    unittest.main()
