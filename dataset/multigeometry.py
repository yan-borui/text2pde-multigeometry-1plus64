from __future__ import annotations

import json
from collections import OrderedDict
from pathlib import Path
from typing import Any

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset


WINDOW_MANIFEST_SCHEMA = "text2pde.multigeometry.windows.v1"


class MultiGeometryWindowDataset(Dataset):
    """Lazily slice fixed-length UVP windows from per-trajectory HDF5 files."""

    def __init__(
        self,
        manifest_path: str,
        return_metadata: bool = False,
        max_open_files: int = 8,
    ) -> None:
        super().__init__()
        self.manifest_path = Path(manifest_path).resolve()
        with self.manifest_path.open("r", encoding="utf-8") as handle:
            self.manifest = json.load(handle)

        if self.manifest.get("schema") != WINDOW_MANIFEST_SCHEMA:
            raise ValueError(
                f"Unsupported window manifest schema: {self.manifest.get('schema')}"
            )

        self.split = str(self.manifest["split"])
        self.window_length = int(self.manifest["window_length"])
        self.trajectories = list(self.manifest["trajectories"])
        for trajectory in self.trajectories:
            source_path = Path(trajectory["source_path"])
            if not source_path.is_absolute():
                source_path = self.manifest_path.parent / source_path
            trajectory["source_path"] = str(source_path.resolve())
        self.window_mode = str(self.manifest["window_mode"])
        self.return_metadata = return_metadata
        self.max_open_files = int(max_open_files)
        if self.window_length <= 0:
            raise ValueError("window_length must be positive")
        if not self.trajectories:
            raise ValueError("window manifest contains no trajectories")
        if self.max_open_files <= 0:
            raise ValueError("max_open_files must be positive")

        if self.window_mode == "dense":
            self.start_min = int(self.manifest["start_min"])
            self.start_max = int(self.manifest["start_max"])
            self.stride = int(self.manifest.get("stride", 1))
            if self.stride <= 0 or self.start_max < self.start_min:
                raise ValueError("invalid dense window range")
            self.starts_per_trajectory = (
                (self.start_max - self.start_min) // self.stride + 1
            )
            self.windows = None
            self.n_windows = len(self.trajectories) * self.starts_per_trajectory
        elif self.window_mode == "explicit":
            self.windows = list(self.manifest["windows"])
            self.n_windows = len(self.windows)
            if self.n_windows == 0:
                raise ValueError("explicit window manifest contains no windows")
        else:
            raise ValueError(f"unsupported window_mode: {self.window_mode}")

        expected = self.manifest.get("window_count")
        if expected is not None and int(expected) != self.n_windows:
            raise ValueError(
                f"manifest window_count={expected} but resolved {self.n_windows}"
            )

        self._files: OrderedDict[str, h5py.File] = OrderedDict()
        self._geometry: OrderedDict[str, tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = OrderedDict()

    def __len__(self) -> int:
        return self.n_windows

    def resolve_window(self, index: int) -> tuple[int, int]:
        if index < 0:
            index += self.n_windows
        if index < 0 or index >= self.n_windows:
            raise IndexError(index)

        if self.window_mode == "dense":
            trajectory_index, local_index = divmod(index, self.starts_per_trajectory)
            start = self.start_min + local_index * self.stride
            return trajectory_index, start

        assert self.windows is not None
        record = self.windows[index]
        return int(record["trajectory_index"]), int(record["start"])

    def _open_file(self, source_path: str) -> h5py.File:
        handle = self._files.pop(source_path, None)
        if handle is None:
            handle = h5py.File(source_path, "r")
        self._files[source_path] = handle
        while len(self._files) > self.max_open_files:
            _, old_handle = self._files.popitem(last=False)
            old_handle.close()
        return handle

    def _load_geometry(
        self, source_path: str, handle: h5py.File
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        cached = self._geometry.pop(source_path, None)
        if cached is not None:
            self._geometry[source_path] = cached
            return cached

        points = torch.from_numpy(np.asarray(handle["points"][:], dtype=np.float32))
        minima = points.amin(dim=0)
        extents = points.amax(dim=0) - minima
        if torch.any(extents <= 0):
            raise ValueError(f"degenerate coordinate extent in {source_path}")
        points = (points - minima) / extents
        cells = torch.from_numpy(np.asarray(handle["cells"][:], dtype=np.int64))
        boundary = torch.from_numpy(np.asarray(handle["boundary"][:], dtype=np.int64))
        cached = (points, cells, boundary)
        self._geometry[source_path] = cached
        while len(self._geometry) > self.max_open_files:
            self._geometry.popitem(last=False)
        return cached

    def __getitem__(self, index: int, eval: bool = False) -> dict[str, Any]:
        trajectory_index, start = self.resolve_window(index)
        trajectory = self.trajectories[trajectory_index]
        source_path = str(trajectory["source_path"])
        handle = self._open_file(source_path)

        stop = start + self.window_length
        uvp = np.asarray(handle["uvp"][start:stop], dtype=np.float32)
        expected_nodes = int(trajectory["node_count"])
        expected_shape = (self.window_length, expected_nodes, 3)
        if uvp.shape != expected_shape:
            raise ValueError(
                f"{trajectory['case_id']} window {start}:{stop} has shape "
                f"{uvp.shape}, expected {expected_shape}"
            )
        if not np.isfinite(uvp).all():
            raise ValueError(
                f"non-finite UVP in {trajectory['case_id']} window {start}:{stop}"
            )

        field = torch.from_numpy(uvp)
        points, cells, boundary = self._load_geometry(source_path, handle)
        coordinates = points.unsqueeze(0).expand(self.window_length, -1, -1)
        relative_time = torch.linspace(0.0, 1.0, self.window_length, dtype=torch.float32)
        time_channel = relative_time[:, None, None].expand(-1, expected_nodes, 1)
        pos_time = torch.cat((coordinates, time_channel), dim=-1)

        result: dict[str, Any] = {"x": field, "pos": pos_time}
        if eval or self.return_metadata:
            absolute_time = torch.from_numpy(
                np.asarray(handle["time"][start:stop], dtype=np.float64)
            )
            result.update(
                {
                    "field": field,
                    "pos_time": pos_time,
                    "cells": cells,
                    "boundary": boundary,
                    "time": absolute_time,
                    "metadata": {
                        "case_id": str(trajectory["case_id"]),
                        "geometry_family": str(trajectory["geometry_family"]),
                        "split": self.split,
                        "source_path": source_path,
                        "start": start,
                        "window_length": self.window_length,
                        "node_count": expected_nodes,
                    },
                }
            )
        return result

    def close(self) -> None:
        files = getattr(self, "_files", {})
        for handle in files.values():
            handle.close()
        files.clear()

    def __del__(self) -> None:
        self.close()

    def __getstate__(self) -> dict[str, Any]:
        state = self.__dict__.copy()
        state["_files"] = OrderedDict()
        state["_geometry"] = OrderedDict()
        return state
