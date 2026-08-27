from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import h5py
import numpy as np
import torch

from dataset.multigeometry import MultiGeometryWindowDataset, WINDOW_MANIFEST_SCHEMA
from tools.multigeometry.prepare_data import (
    EXPECTED_FAMILIES,
    START_MAX,
    START_MIN,
    WINDOW_LENGTH,
    family_from_case_id,
    validate_disjoint_splits,
)


class MultiGeometryWindowDatasetTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary_directory.name)
        self.trajectories = []
        for index, nodes in enumerate((4, 5)):
            file_path = self.root / f"trajectory_{index}.h5"
            uvp = np.arange(7 * nodes * 3, dtype=np.float32).reshape(7, nodes, 3)
            with h5py.File(file_path, "w") as handle:
                handle.create_dataset("uvp", data=uvp)
                handle.create_dataset(
                    "points",
                    data=np.stack(
                        (np.linspace(0, 2, nodes), np.linspace(-1, 1, nodes)), axis=-1
                    ).astype(np.float32),
                )
                handle.create_dataset("cells", data=np.array([[0, 1, 2]], dtype=np.int32))
                handle.create_dataset("boundary", data=np.arange(nodes, dtype=np.uint8))
                handle.create_dataset("time", data=np.arange(7, dtype=np.float64) * 0.1)
            self.trajectories.append(
                {
                    "case_id": f"ellipse_{index:02d}",
                    "geometry_family": "ellipse",
                    "node_count": nodes,
                    "slot": index,
                    "source_h5_sha256": f"unused-{index}",
                    "source_path": str(file_path),
                    "split": "train",
                    "trajectory_index": index,
                }
            )

    def tearDown(self) -> None:
        self.temporary_directory.cleanup()

    def write_manifest(self, value: dict) -> Path:
        file_path = self.root / f"manifest_{len(list(self.root.glob('manifest_*')))}.json"
        file_path.write_text(json.dumps(value), encoding="utf-8")
        return file_path

    def test_dense_windows_are_lazy_and_axis_explicit(self) -> None:
        manifest = {
            "schema": WINDOW_MANIFEST_SCHEMA,
            "split": "train",
            "window_length": 3,
            "window_mode": "dense",
            "start_min": 1,
            "start_max": 2,
            "stride": 1,
            "window_count": 4,
            "trajectories": self.trajectories,
        }
        dataset = MultiGeometryWindowDataset(str(self.write_manifest(manifest)))
        self.assertEqual(len(dataset), 4)
        self.assertEqual(dataset.resolve_window(0), (0, 1))
        self.assertEqual(dataset.resolve_window(3), (1, 2))

        item = dataset.__getitem__(3, eval=True)
        self.assertEqual(tuple(item["x"].shape), (3, 5, 3))
        self.assertEqual(tuple(item["pos"].shape), (3, 5, 3))
        self.assertIs(item["field"], item["x"])
        self.assertIs(item["pos_time"], item["pos"])
        torch.testing.assert_close(item["pos"][0, :, :2].amin(dim=0), torch.zeros(2))
        torch.testing.assert_close(item["pos"][0, :, :2].amax(dim=0), torch.ones(2))
        torch.testing.assert_close(item["pos"][:, 0, 2], torch.tensor([0.0, 0.5, 1.0]))
        torch.testing.assert_close(item["time"], torch.tensor([0.2, 0.3, 0.4], dtype=torch.float64))
        self.assertEqual(item["metadata"]["start"], 2)
        self.assertEqual(item["metadata"]["node_count"], 5)
        dataset.close()

    def test_explicit_windows_preserve_requested_order(self) -> None:
        manifest = {
            "schema": WINDOW_MANIFEST_SCHEMA,
            "split": "validation",
            "window_length": 2,
            "window_mode": "explicit",
            "window_count": 2,
            "trajectories": self.trajectories,
            "windows": [
                {"trajectory_index": 1, "start": 4},
                {"trajectory_index": 0, "start": 0},
            ],
        }
        dataset = MultiGeometryWindowDataset(str(self.write_manifest(manifest)))
        self.assertEqual(dataset.resolve_window(0), (1, 4))
        self.assertEqual(dataset.resolve_window(1), (0, 0))
        self.assertEqual(tuple(dataset[0]["x"].shape), (2, 5, 3))
        dataset.close()

    def test_split_and_family_helpers(self) -> None:
        for family in EXPECTED_FAMILIES:
            self.assertEqual(family_from_case_id(f"{family}_51"), family)
        train = [{"case_id": "ellipse_00", "source_path": "a"}]
        validation = [{"case_id": "ellipse_51", "source_path": "b"}]
        validate_disjoint_splits({"train": train, "validation": validation})
        with self.assertRaises(ValueError):
            validate_disjoint_splits({"train": train, "validation": train})

    def test_official_window_boundaries_cover_151_through_600(self) -> None:
        nodes = 4
        file_path = self.root / "official_bounds.h5"
        frame_ids = np.arange(601, dtype=np.float32)[:, None, None]
        uvp = np.broadcast_to(frame_ids, (601, nodes, 3)).copy()
        with h5py.File(file_path, "w") as handle:
            handle.create_dataset("uvp", data=uvp)
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
            handle.create_dataset("boundary", data=np.array([0, 2, 3, 4], dtype=np.uint8))
            handle.create_dataset("time", data=np.arange(601, dtype=np.float64) * 0.1)
        trajectory = dict(self.trajectories[0])
        trajectory.update(
            {
                "case_id": "ellipse_50",
                "node_count": nodes,
                "source_path": str(file_path),
            }
        )
        manifest = {
            "schema": WINDOW_MANIFEST_SCHEMA,
            "split": "train",
            "window_length": WINDOW_LENGTH,
            "window_mode": "dense",
            "start_min": START_MIN,
            "start_max": START_MAX,
            "stride": 1,
            "window_count": START_MAX - START_MIN + 1,
            "trajectories": [trajectory],
        }
        dataset = MultiGeometryWindowDataset(str(self.write_manifest(manifest)))
        self.assertEqual(len(dataset), 386)
        first = dataset[0]["x"]
        last = dataset[-1]["x"]
        self.assertEqual(float(first[0, 0, 0]), 151.0)
        self.assertEqual(float(first[-1, 0, 0]), 215.0)
        self.assertEqual(float(last[0, 0, 0]), 536.0)
        self.assertEqual(float(last[-1, 0, 0]), 600.0)
        dataset.close()


if __name__ == "__main__":
    unittest.main()
