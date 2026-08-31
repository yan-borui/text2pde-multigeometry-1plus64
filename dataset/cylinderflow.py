from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset


CYLINDERFLOW_DATA_SCHEMA = "text2pde.cylinderflow.raw600.v1"
CYLINDERFLOW_WINDOW_SCHEMA = "text2pde.cylinderflow.windows.v1"
SOURCE_FRAME_COUNT = 600
SOURCE_FRAME_DT = 0.01
SOURCE_NODE_TYPES = frozenset((0, 4, 5, 6))


class CylinderFlowWindowDataset(Dataset):
    """Lazily read contiguous UVP windows from raw MeshGraphNets trajectories."""

    def __init__(
        self,
        manifest_path: str,
        return_metadata: bool = False,
    ) -> None:
        super().__init__()
        self.manifest_path = Path(manifest_path).resolve()
        with self.manifest_path.open("r", encoding="utf-8") as handle:
            self.manifest = json.load(handle)

        if self.manifest.get("schema") != CYLINDERFLOW_WINDOW_SCHEMA:
            raise ValueError(
                f"Unsupported CylinderFlow manifest schema: "
                f"{self.manifest.get('schema')}"
            )
        if self.manifest.get("field_access_status") == "SEALED_METADATA_ONLY":
            raise PermissionError(
                "This manifest contains sealed Test metadata only; use the independent "
                "Test launcher to materialize and access Test fields."
            )

        self.split = str(self.manifest["split"])
        self.window_length = int(self.manifest["window_length"])
        self.frame_stride = int(self.manifest["frame_stride"])
        self.trajectory_count = int(self.manifest["trajectory_count"])
        self.frame_count = int(self.manifest["source_frame_count"])
        self.frame_dt = float(self.manifest["frame_dt"])
        self.window_mode = str(self.manifest["window_mode"])
        self.return_metadata = bool(return_metadata)

        data_path_value = self.manifest.get("data_path")
        if not data_path_value:
            raise ValueError("unsealed manifest must define data_path")
        data_path = Path(data_path_value)
        if not data_path.is_absolute():
            data_path = self.manifest_path.parent / data_path
        self.data_path = data_path.resolve()

        if self.window_length <= 0:
            raise ValueError("window_length must be positive")
        if self.frame_stride != 1:
            raise ValueError("raw CylinderFlow experiments require frame_stride=1")
        if self.trajectory_count <= 0:
            raise ValueError("trajectory_count must be positive")
        if self.frame_count != SOURCE_FRAME_COUNT:
            raise ValueError(
                f"raw CylinderFlow requires {SOURCE_FRAME_COUNT} frames, got "
                f"{self.frame_count}"
            )
        if not np.isclose(self.frame_dt, SOURCE_FRAME_DT):
            raise ValueError(
                f"raw CylinderFlow requires dt={SOURCE_FRAME_DT}, got {self.frame_dt}"
            )

        if self.window_mode == "dense":
            self.start_min = int(self.manifest["start_min"])
            self.start_max = int(self.manifest["start_max"])
            self.start_stride = int(self.manifest.get("start_stride", 1))
            if self.start_stride <= 0 or self.start_max < self.start_min:
                raise ValueError("invalid dense window range")
            self.starts_per_trajectory = (
                self.start_max - self.start_min
            ) // self.start_stride + 1
            self.windows = None
            self.window_count = self.trajectory_count * self.starts_per_trajectory
        elif self.window_mode == "explicit":
            self.windows = list(self.manifest.get("windows", ()))
            if not self.windows:
                raise ValueError("explicit window manifest contains no windows")
            self.window_count = len(self.windows)
        else:
            raise ValueError(f"unsupported window_mode: {self.window_mode}")

        expected_window_count = self.manifest.get("window_count")
        if (
            expected_window_count is not None
            and int(expected_window_count) != self.window_count
        ):
            raise ValueError(
                f"manifest window_count={expected_window_count}, resolved "
                f"{self.window_count}"
            )

        self._handle: h5py.File | None = None
        self._geometry_cache: dict[
            int, tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]
        ] = {}

    def __len__(self) -> int:
        return self.window_count

    def resolve_window(self, index: int) -> tuple[int, int]:
        if index < 0:
            index += self.window_count
        if index < 0 or index >= self.window_count:
            raise IndexError(index)

        if self.window_mode == "dense":
            trajectory_index, local_index = divmod(index, self.starts_per_trajectory)
            start = self.start_min + local_index * self.start_stride
            return trajectory_index, start

        assert self.windows is not None
        record = self.windows[index]
        return int(record["trajectory_index"]), int(record["start"])

    def _open(self) -> h5py.File:
        if self._handle is None:
            self._handle = h5py.File(self.data_path, "r")
            if str(self._handle.attrs.get("schema")) != CYLINDERFLOW_DATA_SCHEMA:
                raise ValueError(f"unexpected data schema in {self.data_path}")
            if str(self._handle.attrs.get("split")) != self.split:
                raise ValueError(f"split mismatch in {self.data_path}")
            if (
                int(self._handle.attrs.get("trajectory_count", -1))
                != self.trajectory_count
            ):
                raise ValueError(f"trajectory count mismatch in {self.data_path}")
            if int(self._handle.attrs.get("frame_count", -1)) != self.frame_count:
                raise ValueError(f"frame count mismatch in {self.data_path}")
            if not np.isclose(
                float(self._handle.attrs.get("frame_dt", np.nan)), self.frame_dt
            ):
                raise ValueError(f"frame dt mismatch in {self.data_path}")
        return self._handle

    def _load_geometry(
        self, trajectory_index: int, group: h5py.Group
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        cached = self._geometry_cache.get(trajectory_index)
        if cached is not None:
            return cached

        points = torch.from_numpy(np.asarray(group["mesh_pos"][:], dtype=np.float32))
        minima = points.amin(dim=0)
        extents = points.amax(dim=0) - minima
        if torch.any(extents <= 0):
            raise ValueError(
                f"trajectory {trajectory_index} has degenerate coordinates"
            )
        normalized_points = (points - minima) / extents
        cells = torch.from_numpy(np.asarray(group["cells"][:], dtype=np.int64))
        node_type = torch.from_numpy(
            np.asarray(group["node_type"][:], dtype=np.int64).reshape(-1)
        )
        cached = (normalized_points, points, cells, node_type)
        self._geometry_cache[trajectory_index] = cached
        return cached

    def __getitem__(self, index: int, eval: bool = False) -> dict[str, Any]:
        trajectory_index, start = self.resolve_window(index)
        stop = start + self.window_length
        if start < 0 or stop > self.frame_count:
            raise ValueError(
                f"trajectory {trajectory_index} window {start}:{stop} exceeds "
                f"0:{self.frame_count}"
            )

        handle = self._open()
        group_name = str(trajectory_index)
        if group_name not in handle:
            raise KeyError(f"trajectory {group_name} missing from {self.data_path}")
        group = handle[group_name]
        node_count = int(group.attrs["node_count"])
        field_array = np.asarray(group["field"][start:stop], dtype=np.float32)
        expected_shape = (self.window_length, node_count, 3)
        if field_array.shape != expected_shape:
            raise ValueError(
                f"trajectory {trajectory_index} window {start}:{stop} has shape "
                f"{field_array.shape}, expected {expected_shape}"
            )
        if not np.isfinite(field_array).all():
            raise ValueError(
                f"trajectory {trajectory_index} window {start}:{stop} is non-finite"
            )

        field = torch.from_numpy(field_array)
        normalized_points, physical_points, cells, node_type = self._load_geometry(
            trajectory_index, group
        )
        coordinates = normalized_points.unsqueeze(0).expand(self.window_length, -1, -1)
        local_time = torch.linspace(0.0, 1.0, self.window_length, dtype=torch.float32)
        time_channel = local_time[:, None, None].expand(-1, node_count, 1)
        pos_time = torch.cat((coordinates, time_channel), dim=-1)

        result: dict[str, Any] = {"x": field, "pos": pos_time}
        if eval or self.return_metadata:
            frame_indices = torch.arange(start, stop, dtype=torch.int64)
            physical_time = frame_indices.to(torch.float64) * self.frame_dt
            result.update(
                {
                    "field": field,
                    "pos_time": pos_time,
                    "points": physical_points,
                    "cells": cells,
                    "node_type": node_type,
                    "frame_indices": frame_indices,
                    "time": physical_time,
                    "metadata": {
                        "split": self.split,
                        "trajectory_index": trajectory_index,
                        "start": start,
                        "stop": stop,
                        "window_length": self.window_length,
                        "frame_stride": self.frame_stride,
                        "frame_dt": self.frame_dt,
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
