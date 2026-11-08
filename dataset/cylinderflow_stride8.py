from __future__ import annotations

import json
import pickle
from pathlib import Path
from typing import Any

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset
from megaflow_data.contract import validate_manifest

DATA_FORMAT = "dgn4cfd.megaflow2d_ellipse_uvp.v1"
STORED_FRAME_COUNT = 65
RAW_FRAME_COUNT = 1000
RAW_FRAME_DT = 0.01
TEMPORAL_STRIDE = 15
FRAME_DT = 0.15
PHASE_OFFSET = 0
SEQUENCE_START = 0
SEQUENCE_LENGTH = 65
FORMAL_SPLIT_COUNTS = {"train": 782, "validation": 100, "test": 100}
SOURCE_NODE_TYPES = frozenset((0, 2, 4, 5, 6))
EXPECTED_SOURCE_FRAME_INDICES = np.arange(1, 962, TEMPORAL_STRIDE, dtype=np.int64)
USED_SOURCE_FRAME_INDICES = EXPECTED_SOURCE_FRAME_INDICES[:SEQUENCE_LENGTH]


def _read_manifest(manifest_path: Path) -> dict[str, Any]:
    with manifest_path.open("r", encoding="utf-8") as handle:
        manifest = json.load(handle)
    if not isinstance(manifest, dict):
        raise TypeError("CylinderFlow Ellipse stride-15 manifest must be a JSON object")
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
    """Write audited all-65-frame Train statistics in Text2PDE pickle format."""

    manifest = _read_manifest(Path(manifest_path).resolve())
    values = normalizer_values_from_manifest(manifest)
    destination = Path(output_path).resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("wb") as handle:
        pickle.dump(values, handle, protocol=4)
    return values


class CylinderFlowStride8TrajectoryDataset(Dataset):
    """One fixed phase-zero sample: 65 frames for AE, first 65 for LDM."""

    def __init__(
        self,
        manifest_path: str | Path,
        data_path: str | Path | None = None,
        split: str = "train",
        return_metadata: bool = False,
        sequence_start: int = SEQUENCE_START,
        sequence_length: int | None = None,
        strict_formal_counts: bool = True,
        stage: str = "ldm",
    ) -> None:
        super().__init__()
        self.manifest_path = Path(manifest_path).resolve()
        self.manifest = _read_manifest(self.manifest_path)
        self.split = str(split).lower()
        self.return_metadata = bool(return_metadata)
        self.sequence_start = int(sequence_start)
        self.stage = str(stage).lower()
        if self.stage not in ("ae", "ldm"):
            raise ValueError("Ellipse stride-15 stage must be ae or ldm")
        expected_length = STORED_FRAME_COUNT if self.stage == "ae" else SEQUENCE_LENGTH
        self.sequence_length = (
            expected_length if sequence_length is None else int(sequence_length)
        )
        self.strict_formal_counts = bool(strict_formal_counts)

        if self.split not in (*FORMAL_SPLIT_COUNTS, "test"):
            raise ValueError("split must be train, validation, or test")
        if self.split == "test" and self.stage != "ldm":
            raise ValueError("Test forecasting uses exactly 1+64 frames and stage ldm")
        if self.sequence_start != SEQUENCE_START:
            raise ValueError(
                "Ellipse stride-15 workflow requires the unique phase-zero prefix"
            )
        if self.sequence_length != expected_length:
            required = "65 AE frames" if self.stage == "ae" else "exactly 1 + 64 frames"
            raise ValueError(f"Ellipse stride-15 {self.stage} requires {required}")

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
        validate_manifest(self.manifest)
        records = self.manifest["trajectories"]
        by_id = {int(record["global_index"]): record for record in records}
        expected = {
            case: split
            for split, ids in self.manifest["splits"].items()
            for case in ids
        }
        if len(records) != len(by_id) or set(by_id) != set(expected):
            raise ValueError("Ellipse trajectory records differ from split membership")
        for case, split in expected.items():
            record = by_id[case]
            if (
                record["source_split"] != split
                or record["group"] != f"trajectory_{case:04d}"
            ):
                raise ValueError("Ellipse trajectory record has inconsistent identity")
        normalizer_values_from_manifest(self.manifest)

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
            handle = h5py.File(self.data_path, "r")
            try:
                self._validate_hdf5_root(handle)
            except Exception:
                handle.close()
                raise
            self._handle = handle
        return self._handle

    def _validate_hdf5_root(self, handle: h5py.File) -> None:
        # Ellipse preparation writes portable HDF5 attributes.
        scalar_attributes = {
            "format": DATA_FORMAT,
            "frames": SEQUENCE_LENGTH if self.split == "test" else STORED_FRAME_COUNT,
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
            self.manifest["splits"].get("train", ())
        ):
            raise ValueError("HDF5 Train count differs from manifest")
        if int(handle.attrs.get("validation_count", -1)) != len(
            self.manifest["splits"].get("validation", ())
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

    def _trajectory_group(self, index: int) -> tuple[int, dict, h5py.Group]:
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
        expected_frames = (
            SEQUENCE_LENGTH if self.split == "test" else STORED_FRAME_COUNT
        )
        if int(group.attrs.get("frames", -1)) != expected_frames:
            raise ValueError(
                f"group {group_name} does not contain {expected_frames} frames"
            )
        if int(group.attrs.get("temporal_stride", -1)) != TEMPORAL_STRIDE:
            raise ValueError(f"group {group_name} has the wrong temporal stride")
        if int(group.attrs.get("phase_offset", -1)) != PHASE_OFFSET:
            raise ValueError(f"group {group_name} has the wrong phase offset")
        return global_index, record, group

    def initial(self, index: int) -> dict[str, Any]:
        """Load only the physical conditioning frame and static CPU geometry."""
        global_index, record, group = self._trajectory_group(index)
        initial = np.asarray(group["uvp"][:1], dtype=np.float32)
        if (
            initial.shape != (1, int(record["nodes"]), 3)
            or not np.isfinite(initial).all()
        ):
            raise ValueError("benchmark conditioning frame must be finite [N,3]")
        _, points, cells, node_type = self._load_geometry(global_index, group)
        return {
            "trajectory_index": global_index,
            "initial": initial[0],
            "points": points.numpy(),
            "cells": cells.numpy(),
            "node_type": node_type.numpy(),
        }

    def __getitem__(self, index: int, eval: bool = False) -> dict[str, Any]:
        global_index, record, group = self._trajectory_group(index)

        node_count = int(record["nodes"])
        field_array = np.asarray(group["uvp"][: self.sequence_length], dtype=np.float32)
        expected_shape = (self.sequence_length, node_count, 3)
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
        spatial = normalized_mesh_pos.unsqueeze(0).expand(self.sequence_length, -1, -1)
        local_time = torch.linspace(0.0, 1.0, self.sequence_length, dtype=torch.float32)
        time_channel = local_time[:, None, None].expand(-1, node_count, 1)
        pos = torch.cat((spatial, time_channel), dim=-1)
        result: dict[str, Any] = {"x": field, "pos": pos}

        if eval or self.return_metadata:
            frame_indices = torch.from_numpy(
                EXPECTED_SOURCE_FRAME_INDICES[: self.sequence_length].copy()
            )
            physical_time = (frame_indices.to(torch.float64) - 1) * RAW_FRAME_DT
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
                        "source_local_index": int(record["source_index"]),
                        "group": record["group"],
                        "sequence_start": SEQUENCE_START,
                        "sequence_length": self.sequence_length,
                        "stage": self.stage,
                        "stored_frame_count": SEQUENCE_LENGTH
                        if self.split == "test"
                        else STORED_FRAME_COUNT,
                        "raw_frame_start": int(frame_indices[0]),
                        "raw_frame_stop_inclusive": int(frame_indices[-1]),
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
