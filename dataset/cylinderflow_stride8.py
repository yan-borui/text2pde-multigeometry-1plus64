from __future__ import annotations

import json
import pickle
from pathlib import Path
from typing import Any

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset


DATA_FORMAT = "dgn4cfd.mgn_cylinderflow_temporal_stride.v1"
STORED_FRAME_COUNT = 75
RAW_FRAME_COUNT = 600
RAW_FRAME_DT = 0.01
TEMPORAL_STRIDE = 8
FRAME_DT = 0.08
PHASE_OFFSET = 0
SEQUENCE_START = 0
SEQUENCE_LENGTH = 65
FORMAL_SPLIT_COUNTS = {"train": 1000, "validation": 100}
SOURCE_NODE_TYPES = frozenset((0, 4, 5, 6))
EXPECTED_SOURCE_FRAME_INDICES = np.arange(
    PHASE_OFFSET, RAW_FRAME_COUNT, TEMPORAL_STRIDE, dtype=np.int64
)
USED_SOURCE_FRAME_INDICES = EXPECTED_SOURCE_FRAME_INDICES[:SEQUENCE_LENGTH]


def _read_manifest(manifest_path: Path) -> dict[str, Any]:
    with manifest_path.open("r", encoding="utf-8") as handle:
        manifest = json.load(handle)
    if not isinstance(manifest, dict):
        raise TypeError("CylinderFlow stride-8 manifest must be a JSON object")
    return manifest


def normalizer_values_from_manifest(manifest: dict[str, Any]) -> list[float]:
    """Return Text2PDE's interleaved UVP mean/std representation."""

    record = manifest.get("train_only_normalization")
    if not isinstance(record, dict):
        raise ValueError("manifest has no train_only_normalization record")
    if record.get("field_names") != ["u", "v", "p"]:
        raise ValueError("normalization fields must be exactly [u, v, p]")
    means = np.asarray(record.get("field_mean"), dtype=np.float64)
    standard_deviations = np.asarray(record.get("field_std"), dtype=np.float64)
    if means.shape != (3,) or standard_deviations.shape != (3,):
        raise ValueError("normalization mean/std must each contain three values")
    if not np.isfinite(means).all() or not np.isfinite(standard_deviations).all():
        raise ValueError("normalization statistics must be finite")
    if np.any(standard_deviations <= 0):
        raise ValueError("normalization standard deviations must be positive")
    return [
        float(means[0]),
        float(standard_deviations[0]),
        float(means[1]),
        float(standard_deviations[1]),
        float(means[2]),
        float(standard_deviations[2]),
    ]


def write_text2pde_normalizer(
    manifest_path: str | Path, output_path: str | Path
) -> list[float]:
    """Write audited all-75-frame Train statistics in Text2PDE pickle format."""

    manifest = _read_manifest(Path(manifest_path).resolve())
    values = normalizer_values_from_manifest(manifest)
    destination = Path(output_path).resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("wb") as handle:
        pickle.dump(values, handle, protocol=4)
    return values


class CylinderFlowStride8TrajectoryDataset(Dataset):
    """One fixed phase-zero ``1 + 64`` prefix from each 75-frame trajectory.

    The HDF5 asset retains all 75 phase-zero stride-8 frames. This dataset has no
    temporal-window axis: sample ``i`` is exactly raw frames ``0, 8, ..., 512``
    from one trajectory in the requested trajectory-level split.
    """

    def __init__(
        self,
        manifest_path: str | Path,
        data_path: str | Path | None = None,
        split: str = "train",
        return_metadata: bool = False,
        sequence_start: int = SEQUENCE_START,
        sequence_length: int = SEQUENCE_LENGTH,
        strict_formal_counts: bool = True,
    ) -> None:
        super().__init__()
        self.manifest_path = Path(manifest_path).resolve()
        self.manifest = _read_manifest(self.manifest_path)
        self.split = str(split).lower()
        self.return_metadata = bool(return_metadata)
        self.sequence_start = int(sequence_start)
        self.sequence_length = int(sequence_length)
        self.strict_formal_counts = bool(strict_formal_counts)

        if self.split not in FORMAL_SPLIT_COUNTS:
            raise ValueError("stride-8 workflow exposes Train and Validation only")
        if self.sequence_start != SEQUENCE_START:
            raise ValueError("stride-8 workflow requires the unique phase-zero prefix")
        if self.sequence_length != SEQUENCE_LENGTH:
            raise ValueError("stride-8 workflow requires exactly 1 + 64 frames")

        self._validate_manifest()
        manifest_data_path = self.manifest.get("dataset")
        selected_data_path = data_path if data_path is not None else manifest_data_path
        if selected_data_path is None:
            raise ValueError("an HDF5 data_path must be provided")
        candidate = Path(selected_data_path)
        if not candidate.is_absolute():
            candidate = self.manifest_path.parent / candidate
        self.data_path = candidate.resolve()

        split_records = self.manifest["splits"][self.split]
        self.trajectory_indices = tuple(int(value) for value in split_records)
        trajectories = self.manifest["trajectories"]
        self._records_by_index = {
            int(record["global_index"]): record for record in trajectories
        }
        self.normalizer_values = normalizer_values_from_manifest(self.manifest)

        self._handle: h5py.File | None = None
        self._geometry_cache: dict[
            int, tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]
        ] = {}

    def _validate_manifest(self) -> None:
        manifest = self.manifest
        required_scalars = {
            "format": DATA_FORMAT,
            "frames": STORED_FRAME_COUNT,
            "raw_frames": RAW_FRAME_COUNT,
            "temporal_stride": TEMPORAL_STRIDE,
            "phase_offset": PHASE_OFFSET,
            "sequences_per_trajectory": 1,
            "phase_augmentation": False,
            "test_accessed": False,
        }
        for name, expected in required_scalars.items():
            value = manifest.get(name)
            if value != expected:
                raise ValueError(f"manifest {name}={value!r}, expected {expected!r}")
        for name, expected in (("raw_frame_dt", RAW_FRAME_DT), ("frame_dt", FRAME_DT)):
            value = float(manifest.get(name, np.nan))
            if not np.isclose(value, expected, rtol=0.0, atol=1.0e-12):
                raise ValueError(f"manifest {name}={value}, expected {expected}")

        source_indices = np.asarray(
            manifest.get("source_frame_indices"), dtype=np.int64
        )
        if not np.array_equal(source_indices, EXPECTED_SOURCE_FRAME_INDICES):
            raise ValueError("manifest source_frame_indices are not exactly 0:600:8")

        splits = manifest.get("splits")
        if not isinstance(splits, dict) or set(splits) != set(FORMAL_SPLIT_COUNTS):
            raise ValueError("manifest splits must be exactly train and validation")
        train_indices = tuple(int(value) for value in splits["train"])
        validation_indices = tuple(int(value) for value in splits["validation"])
        if len(set(train_indices)) != len(train_indices) or len(
            set(validation_indices)
        ) != len(validation_indices):
            raise ValueError("a trajectory appears more than once within a split")
        if set(train_indices).intersection(validation_indices):
            raise ValueError("Train and Validation trajectory identities overlap")
        combined = train_indices + validation_indices
        if len(combined) != int(manifest.get("trajectory_count", -1)):
            raise ValueError("split membership does not cover trajectory_count exactly")
        if len(set(combined)) != len(combined):
            raise ValueError("split identities are not globally unique")

        if self.strict_formal_counts:
            if len(train_indices) != FORMAL_SPLIT_COUNTS["train"]:
                raise ValueError("formal Train split must contain 1000 trajectories")
            if len(validation_indices) != FORMAL_SPLIT_COUNTS["validation"]:
                raise ValueError(
                    "formal Validation split must contain 100 trajectories"
                )
            if train_indices != tuple(range(1000)):
                raise ValueError(
                    "formal Train identities must be global indices 0:1000"
                )
            if validation_indices != tuple(range(1000, 1100)):
                raise ValueError(
                    "formal Validation identities must be global indices 1000:1100"
                )

        records = manifest.get("trajectories")
        if not isinstance(records, list) or len(records) != len(combined):
            raise ValueError("trajectory records do not match split membership")
        by_index: dict[int, dict[str, Any]] = {}
        split_lookup = {
            global_index: split_name
            for split_name, values in (
                ("train", train_indices),
                ("validation", validation_indices),
            )
            for global_index in values
        }
        for record in records:
            global_index = int(record["global_index"])
            if global_index in by_index:
                raise ValueError(f"duplicate trajectory record {global_index}")
            by_index[global_index] = record
            expected_split = split_lookup.get(global_index)
            if (
                expected_split is None
                or str(record.get("source_split")).lower() != expected_split
            ):
                raise ValueError(
                    f"trajectory {global_index} has inconsistent split identity"
                )
            if record.get("group") != f"trajectory_{global_index:04d}":
                raise ValueError(
                    f"trajectory {global_index} has an unexpected HDF5 group"
                )
            if int(record.get("frames", -1)) != STORED_FRAME_COUNT:
                raise ValueError(
                    f"trajectory {global_index} does not declare 75 frames"
                )
            if int(record.get("temporal_stride", -1)) != TEMPORAL_STRIDE:
                raise ValueError(f"trajectory {global_index} has the wrong stride")
            if int(record.get("phase_offset", -1)) != PHASE_OFFSET:
                raise ValueError(f"trajectory {global_index} has the wrong phase")
        if set(by_index) != set(combined):
            raise ValueError("trajectory records and split identities differ")

        normalizer_values_from_manifest(manifest)

    def __len__(self) -> int:
        return len(self.trajectory_indices)

    def resolve_trajectory(self, index: int) -> tuple[int, dict[str, Any]]:
        if index < 0:
            index += len(self)
        if index < 0 or index >= len(self):
            raise IndexError(index)
        global_index = self.trajectory_indices[index]
        return global_index, self._records_by_index[global_index]

    def _open(self) -> h5py.File:
        if self._handle is None:
            if self.strict_formal_counts and h5py.version.hdf5_version_tuple < (
                2,
                0,
                0,
            ):
                raise RuntimeError(
                    "the locked stride-8 HDF5 uses HDF5 2.0 object layouts; "
                    "install h5py>=3.16 with an HDF5>=2.0 runtime"
                )
            handle = h5py.File(self.data_path, "r")
            try:
                self._validate_hdf5_root(handle)
            except Exception:
                handle.close()
                raise
            self._handle = handle
        return self._handle

    def _validate_hdf5_root(self, handle: h5py.File) -> None:
        # The locked file was written by HDF5 2.0. Its boolean attributes use a
        # datatype that the established Text2PDE HDF5 1.14 runtime cannot decode.
        # The hash-locked manifest validates those booleans; all portable numeric
        # identity attributes are checked directly here.
        scalar_attributes = {
            "format": DATA_FORMAT,
            "frames": STORED_FRAME_COUNT,
            "source_frames": RAW_FRAME_COUNT,
            "temporal_stride": TEMPORAL_STRIDE,
            "phase_offset": PHASE_OFFSET,
            "sequences_per_trajectory": 1,
            "trajectory_count": int(self.manifest["trajectory_count"]),
        }
        for name, expected in scalar_attributes.items():
            value = handle.attrs.get(name)
            if value != expected:
                raise ValueError(f"HDF5 {name}={value!r}, expected {expected!r}")
        for name, expected in (("raw_frame_dt", RAW_FRAME_DT), ("frame_dt", FRAME_DT)):
            value = float(handle.attrs.get(name, np.nan))
            if not np.isclose(value, expected, rtol=0.0, atol=1.0e-12):
                raise ValueError(f"HDF5 {name}={value}, expected {expected}")
        if int(handle.attrs.get("train_count", -1)) != len(
            self.manifest["splits"]["train"]
        ):
            raise ValueError("HDF5 Train count differs from manifest")
        if int(handle.attrs.get("validation_count", -1)) != len(
            self.manifest["splits"]["validation"]
        ):
            raise ValueError("HDF5 Validation count differs from manifest")
        expected_stats = np.asarray(self.normalizer_values, dtype=np.float64)
        stored_stats = np.empty(6, dtype=np.float64)
        stored_stats[::2] = np.asarray(handle["field_mean"][:], dtype=np.float64)
        stored_stats[1::2] = np.asarray(handle["field_std"][:], dtype=np.float64)
        if not np.allclose(stored_stats, expected_stats, rtol=0.0, atol=1.0e-12):
            raise ValueError("HDF5 normalization statistics differ from manifest")

    def _load_geometry(
        self, global_index: int, group: h5py.Group
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        cached = self._geometry_cache.get(global_index)
        if cached is not None:
            return cached
        mesh_pos = torch.from_numpy(np.asarray(group["mesh_pos"][:], dtype=np.float32))
        minima = mesh_pos.amin(dim=0)
        extents = mesh_pos.amax(dim=0) - minima
        if torch.any(extents <= 0):
            raise ValueError(f"trajectory {global_index} has degenerate coordinates")
        normalized_mesh_pos = (mesh_pos - minima) / extents
        cells = torch.from_numpy(np.asarray(group["cells"][:], dtype=np.int64))
        node_type = torch.from_numpy(
            np.asarray(group["node_type"][:], dtype=np.int64).reshape(-1)
        )
        if not set(node_type.unique().tolist()).issubset(SOURCE_NODE_TYPES):
            raise ValueError(f"trajectory {global_index} contains unknown node types")
        cached = (normalized_mesh_pos, mesh_pos, cells, node_type)
        self._geometry_cache[global_index] = cached
        return cached

    def __getitem__(self, index: int, eval: bool = False) -> dict[str, Any]:
        global_index, record = self.resolve_trajectory(index)
        handle = self._open()
        group_name = record["group"]
        if group_name not in handle:
            raise KeyError(f"trajectory group {group_name} is missing")
        group = handle[group_name]
        if int(group.attrs.get("global_index", -1)) != global_index:
            raise ValueError(f"group {group_name} has the wrong global index")
        if str(group.attrs.get("source_split", "")).lower() != self.split:
            raise ValueError(f"group {group_name} crosses the requested split")
        if int(group.attrs.get("frames", -1)) != STORED_FRAME_COUNT:
            raise ValueError(f"group {group_name} does not contain 75 frames")
        if int(group.attrs.get("temporal_stride", -1)) != TEMPORAL_STRIDE:
            raise ValueError(f"group {group_name} has the wrong temporal stride")
        if int(group.attrs.get("phase_offset", -1)) != PHASE_OFFSET:
            raise ValueError(f"group {group_name} has the wrong phase offset")

        node_count = int(record["nodes"])
        field_array = np.asarray(group["uvp"][:SEQUENCE_LENGTH], dtype=np.float32)
        expected_shape = (SEQUENCE_LENGTH, node_count, 3)
        if field_array.shape != expected_shape:
            raise ValueError(
                f"trajectory {global_index} prefix has shape {field_array.shape}, "
                f"expected {expected_shape}"
            )
        if not np.isfinite(field_array).all():
            raise ValueError(f"trajectory {global_index} prefix is non-finite")

        field = torch.from_numpy(field_array)
        normalized_mesh_pos, mesh_pos, cells, node_type = self._load_geometry(
            global_index, group
        )
        spatial = normalized_mesh_pos.unsqueeze(0).expand(SEQUENCE_LENGTH, -1, -1)
        local_time = torch.linspace(0.0, 1.0, SEQUENCE_LENGTH, dtype=torch.float32)
        time_channel = local_time[:, None, None].expand(-1, node_count, 1)
        pos = torch.cat((spatial, time_channel), dim=-1)
        result: dict[str, Any] = {"x": field, "pos": pos}

        if eval or self.return_metadata:
            frame_indices = torch.from_numpy(USED_SOURCE_FRAME_INDICES.copy())
            physical_time = frame_indices.to(torch.float64) * RAW_FRAME_DT
            result.update(
                {
                    "field": field,
                    "pos_time": pos,
                    "mesh_pos": mesh_pos,
                    "points": mesh_pos,
                    "cells": cells,
                    "node_type": node_type,
                    "frame_indices": frame_indices,
                    "time": physical_time,
                    "metadata": {
                        "split": self.split,
                        "sample_index": int(index if index >= 0 else index + len(self)),
                        "trajectory_index": global_index,
                        "source_local_index": int(record["source_local_index"]),
                        "group": group_name,
                        "sequence_start": SEQUENCE_START,
                        "sequence_length": SEQUENCE_LENGTH,
                        "stored_frame_count": STORED_FRAME_COUNT,
                        "raw_frame_start": int(USED_SOURCE_FRAME_INDICES[0]),
                        "raw_frame_stop_inclusive": int(USED_SOURCE_FRAME_INDICES[-1]),
                        "temporal_stride": TEMPORAL_STRIDE,
                        "raw_frame_dt": RAW_FRAME_DT,
                        "frame_dt": FRAME_DT,
                        "phase_offset": PHASE_OFFSET,
                        "node_count": node_count,
                    },
                }
            )
        return result

    def close(self) -> None:
        handle = getattr(self, "_handle", None)
        if handle is not None:
            handle.close()
            self._handle = None
        geometry_cache = getattr(self, "_geometry_cache", None)
        if geometry_cache is not None:
            geometry_cache.clear()

    def __del__(self) -> None:
        self.close()

    def __getstate__(self) -> dict[str, Any]:
        state = self.__dict__.copy()
        state["_handle"] = None
        state["_geometry_cache"] = {}
        return state
