"""Read distinct official Test trajectories, retaining their original split."""

from __future__ import annotations

import json
from pathlib import Path

import h5py
import numpy as np

from .prepare_test import DATASETS, RAW_INDICES


class TestData:
    """CPU geometry and physical fields for one fixed Test100 population."""

    def __init__(self, directory: str | Path, dataset_name: str):
        self.directory = Path(directory).expanduser().resolve(strict=True)
        self.manifest_file = self.directory / "manifest.json"
        self.manifest = json.loads(self.manifest_file.read_text(encoding="utf-8"))
        self.dataset = self.directory / "data.h5"
        self.dataset_name = dataset_name
        raw_frames, raw_dt, data_format = DATASETS[dataset_name]
        self.dt = raw_dt * 8
        expected = {
            "format": data_format,
            "frames": 65,
            "raw_frames": raw_frames,
            "temporal_stride": 8,
            "phase_offset": 0,
            "phase_augmentation": False,
            "test_accessed": True,
            "trajectory_count": 100,
            "source_frame_indices": RAW_INDICES,
            "splits": {"test": list(range(100))},
        }
        if any(self.manifest.get(key) != value for key, value in expected.items()):
            raise ValueError(
                "expected a complete official Test100, stride-8, 65-frame manifest"
            )
        if not np.isclose(
            self.manifest.get("frame_dt", 0), self.dt, rtol=0, atol=1e-12
        ):
            raise ValueError("Test physical time interval differs from this model")
        if self.dataset.stat().st_size != self.manifest.get("dataset_bytes"):
            raise ValueError("Test HDF5 file size differs from its manifest")
        records = self.manifest["trajectories"]
        if len(records) != 100:
            raise ValueError("Test manifest must retain every official trajectory")
        self.splits = {"test": tuple(range(100))}
        self._static = {}
        with h5py.File(self.dataset, "r") as handle:
            for key in (
                "format",
                "frames",
                "temporal_stride",
                "test_accessed",
                "trajectory_count",
            ):
                if handle.attrs.get(key) != expected[key]:
                    raise ValueError(f"Test HDF5 {key} differs from the manifest")
            if not np.isclose(
                handle.attrs.get("frame_dt", 0), self.dt, rtol=0, atol=1e-12
            ):
                raise ValueError("Test HDF5 has a different physical time interval")
            for index, record in enumerate(records):
                if (
                    record.get("source_split") != "test"
                    or record.get("source_local_index") != index
                ):
                    raise ValueError(
                        "Test trajectory provenance differs from official order"
                    )
                if (
                    record.get("global_index") != index
                    or record.get("group") != f"trajectory_{index:04d}"
                ):
                    raise ValueError("Test trajectory IDs must be distinct and ordered")
                group = handle[record["group"]]
                if (
                    group.attrs.get("source_split") != "test"
                    or group.attrs.get("source_local_index") != index
                ):
                    raise ValueError(
                        "Test HDF5 trajectory provenance differs from the manifest"
                    )
                if group["uvp"].shape != (65, record["nodes"], 3):
                    raise ValueError("Test UVP must contain exactly 1+64 frames")
                if group["mesh_pos"].shape != (record["nodes"], 2):
                    raise ValueError("Test geometry and field node counts differ")

    def identity(self) -> dict:
        return {
            "dataset": self.dataset_name,
            "split": "test",
            "test_accessed": True,
            "split_indices": self.splits,
            "stored_frames": 65,
            "evaluation_frames": [0, 64],
            "evaluation_raw_frame_indices": RAW_INDICES,
            "dt": self.dt,
            "source": self.manifest["source"],
        }

    def bind_training_manifest(self, training: dict) -> None:
        """Require the exact frozen statistics used by the selected model."""
        if training.get("format") != self.manifest["format"]:
            raise ValueError("Train and Test belong to different datasets")
        if training.get("train_only_normalization") != self.manifest.get(
            "train_only_normalization"
        ):
            raise ValueError(
                "Test conversion must copy this model's Train normalization"
            )

    def static(self, index: int) -> dict:
        if index not in self.splits["test"]:
            raise IndexError(index)
        if index not in self._static:
            with h5py.File(self.dataset, "r") as handle:
                group = handle[f"trajectory_{index:04d}"]
                points = np.asarray(group["mesh_pos"], dtype=np.float32)
                cells = np.asarray(group["cells"], dtype=np.int64)
                labels = np.asarray(group["node_type"], dtype=np.int64).reshape(-1)
            allowed = {0, 2, 4} if self.dataset_name == "airfoil" else {0, 4, 5, 6}
            if not np.isfinite(points).all() or not set(np.unique(labels)).issubset(
                allowed
            ):
                raise ValueError(f"invalid Test geometry or node labels: {index}")
            if cells.ndim != 2 or cells.shape[1] != 3 or not cells.size:
                raise ValueError(f"invalid Test triangular cells: {index}")
            if cells.min() < 0 or cells.max() >= len(points):
                raise ValueError(f"Test cell references a nonexistent node: {index}")
            pairs = np.concatenate(
                (cells[:, [0, 1]], cells[:, [1, 2]], cells[:, [2, 0]])
            )
            edges = np.unique(np.concatenate((pairs, pairs[:, ::-1])), axis=0)
            delta = points[edges[:, 0]] - points[edges[:, 1]]
            features = np.concatenate(
                (delta, np.linalg.norm(delta, axis=1, keepdims=True)), axis=1
            )
            self._static[index] = dict(
                index=index,
                trajectory_index=index,
                points=points,
                cells=cells,
                node_type=labels,
                edges=edges,
                edge_features=features.astype(np.float32),
            )
        return self._static[index]

    def read(self, index: int, start: int = 0, stop: int = 65) -> dict:
        if not 0 <= start < stop <= 65:
            raise ValueError("Test reads must stay inside stored frames 0..64")
        static = self.static(index)
        with h5py.File(self.dataset, "r") as handle:
            field = np.asarray(
                handle[f"trajectory_{index:04d}/uvp"][start:stop], dtype=np.float32
            )
        if not np.isfinite(field).all():
            raise ValueError(f"nonfinite Test reference: {index}")
        return {**static, "field": field}

    def initial(self, index: int) -> dict:
        sample = self.read(index, 0, 1)
        field = sample.pop("field")
        return {**sample, "initial": field[0]}

    def evaluation(self, index: int) -> dict:
        return self.read(index)
