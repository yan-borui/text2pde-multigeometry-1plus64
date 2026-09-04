from __future__ import annotations

import json
import pickle
import tempfile
import unittest
from pathlib import Path

import h5py
import numpy as np

from dataset.cylinderflow_stride8 import (
    CylinderFlowStride8TrajectoryDataset,
    write_text2pde_normalizer,
)


MEANS = [0.5, -0.25, 1.5]
STDS = [2.0, 3.0, 4.0]


def write_fixture(root: Path) -> tuple[Path, Path]:
    data_path = root / "stride8.h5"
    manifest_path = root / "manifest.json"
    split_membership = {"train": [0, 1], "validation": [2]}
    records = []
    with h5py.File(data_path, "w") as handle:
        handle.attrs.update(
            {
                "format": "dgn4cfd.mgn_cylinderflow_temporal_stride.v1",
                "frames": 75,
                "source_frames": 600,
                "raw_frame_dt": 0.01,
                "frame_dt": 0.08,
                "temporal_stride": 8,
                "phase_offset": 0,
                "sequences_per_trajectory": 1,
                "phase_augmentation": False,
                "trajectory_count": 3,
                "train_count": 2,
                "validation_count": 1,
            }
        )
        handle.create_dataset("field_mean", data=np.asarray(MEANS))
        handle.create_dataset("field_std", data=np.asarray(STDS))
        for global_index in range(3):
            split = "train" if global_index < 2 else "validation"
            group_name = f"trajectory_{global_index:04d}"
            group = handle.create_group(group_name)
            frame = np.arange(75, dtype=np.float32)[:, None, None]
            node = np.arange(4, dtype=np.float32)[None, :, None]
            channel = np.arange(3, dtype=np.float32)[None, None, :]
            uvp = 1000 * global_index + 10 * frame + node + channel / 10
            group.create_dataset("uvp", data=uvp.astype(np.float32))
            group.create_dataset(
                "mesh_pos",
                data=np.asarray(
                    [[0.0, 0.0], [2.0, 0.0], [2.0, 1.0], [0.0, 1.0]],
                    dtype=np.float32,
                ),
            )
            group.create_dataset(
                "cells", data=np.asarray([[0, 1, 2], [0, 2, 3]], dtype=np.int32)
            )
            group.create_dataset(
                "node_type", data=np.asarray([4, 5, 6, 0], dtype=np.int32)
            )
            group.attrs.update(
                {
                    "global_index": global_index,
                    "source_split": split,
                    "source_local_index": global_index if split == "train" else 0,
                    "frames": 75,
                    "nodes": 4,
                    "cells": 2,
                    "source_frames": 600,
                    "raw_frame_dt": 0.01,
                    "frame_dt": 0.08,
                    "temporal_stride": 8,
                    "phase_offset": 0,
                }
            )
            records.append(
                {
                    "global_index": global_index,
                    "group": group_name,
                    "source_split": split,
                    "source_local_index": global_index if split == "train" else 0,
                    "frames": 75,
                    "nodes": 4,
                    "cells": 2,
                    "source_frames": 600,
                    "temporal_stride": 8,
                    "phase_offset": 0,
                }
            )
    manifest = {
        "format": "dgn4cfd.mgn_cylinderflow_temporal_stride.v1",
        "schema_version": 1,
        "dataset": data_path.name,
        "dataset_bytes": data_path.stat().st_size,
        "frames": 75,
        "raw_frames": 600,
        "raw_frame_dt": 0.01,
        "temporal_stride": 8,
        "frame_dt": 0.08,
        "phase_offset": 0,
        "source_frame_indices": list(range(0, 600, 8)),
        "sequences_per_trajectory": 1,
        "phase_augmentation": False,
        "trajectory_count": 3,
        "splits": split_membership,
        "trajectories": records,
        "train_only_normalization": {
            "field_names": ["u", "v", "p"],
            "field_mean": MEANS,
            "field_std": STDS,
            "field_value_count": 1800,
            "trajectory_count": 2,
            "frame_selection": "raw frames 0:600:8",
        },
        "test_accessed": False,
    }
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    return data_path, manifest_path


class CylinderFlowStride8Test(unittest.TestCase):
    def test_unique_prefix_shape_identity_and_time(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            data_path, manifest_path = write_fixture(Path(temporary_directory))
            train = CylinderFlowStride8TrajectoryDataset(
                manifest_path,
                data_path,
                split="train",
                return_metadata=True,
                strict_formal_counts=False,
            )
            validation = CylinderFlowStride8TrajectoryDataset(
                manifest_path,
                data_path,
                split="validation",
                return_metadata=True,
                strict_formal_counts=False,
            )
            self.assertEqual(len(train), 2)
            self.assertEqual(len(validation), 1)
            self.assertEqual(train.resolve_trajectory(0)[0], 0)
            self.assertEqual(train.resolve_trajectory(1)[0], 1)
            self.assertEqual(validation.resolve_trajectory(0)[0], 2)

            sample = train[1]
            self.assertEqual(tuple(sample["x"].shape), (65, 4, 3))
            self.assertEqual(tuple(sample["pos"].shape), (65, 4, 3))
            np.testing.assert_array_equal(
                sample["frame_indices"].numpy(), np.arange(0, 520, 8)
            )
            np.testing.assert_allclose(
                sample["time"].numpy(), np.arange(65) * 0.08, rtol=0, atol=1e-12
            )
            np.testing.assert_allclose(
                sample["x"][:, 0, 0].numpy(), 1000 + 10 * np.arange(65)
            )
            self.assertEqual(float(sample["pos"][0, 0, 2]), 0.0)
            self.assertEqual(float(sample["pos"][-1, 0, 2]), 1.0)
            self.assertEqual(sample["metadata"]["raw_frame_stop_inclusive"], 512)
            self.assertEqual(sample["metadata"]["temporal_stride"], 8)
            train.close()
            validation.close()

    def test_rejects_alternate_temporal_phase_window_or_split_overlap(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            data_path, manifest_path = write_fixture(Path(temporary_directory))
            with self.assertRaisesRegex(ValueError, "unique phase-zero prefix"):
                CylinderFlowStride8TrajectoryDataset(
                    manifest_path,
                    data_path,
                    sequence_start=1,
                    strict_formal_counts=False,
                )
            with self.assertRaisesRegex(ValueError, r"exactly 1 \+ 64"):
                CylinderFlowStride8TrajectoryDataset(
                    manifest_path,
                    data_path,
                    sequence_length=64,
                    strict_formal_counts=False,
                )
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            manifest["splits"]["validation"] = [1]
            manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "overlap"):
                CylinderFlowStride8TrajectoryDataset(
                    manifest_path,
                    data_path,
                    strict_formal_counts=False,
                )

    def test_manifest_stats_write_text2pde_normalizer(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            _, manifest_path = write_fixture(root)
            output_path = root / "train_normal_stats.pkl"
            values = write_text2pde_normalizer(manifest_path, output_path)
            self.assertEqual(values, [0.5, 2.0, -0.25, 3.0, 1.5, 4.0])
            with output_path.open("rb") as handle:
                self.assertEqual(pickle.load(handle), values)

    def test_hdf5_phase_mismatch_is_rejected_when_opened(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            data_path, manifest_path = write_fixture(Path(temporary_directory))
            with h5py.File(data_path, "r+") as handle:
                handle.attrs["phase_offset"] = 1
            dataset = CylinderFlowStride8TrajectoryDataset(
                manifest_path,
                data_path,
                strict_formal_counts=False,
            )
            with self.assertRaisesRegex(ValueError, "HDF5 phase_offset"):
                _ = dataset[0]


if __name__ == "__main__":
    unittest.main()
