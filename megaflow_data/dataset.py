"""Shared physical-mesh reader; every consumer selects explicit split IDs."""

from __future__ import annotations

import json
from pathlib import Path

import h5py
import numpy as np

from .contract import (
    DT,
    FRAME_INDICES,
    FRAMES,
    PROTOCOL,
    REPOSITORY,
    REVISION,
    SOURCE_REVISION,
    splits,
    validate_manifest,
)


def edges_from_cells(
    points: np.ndarray, cells: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    pairs = np.concatenate([cells[:, [0, 1]], cells[:, [1, 2]], cells[:, [2, 0]]])
    edges = np.unique(np.concatenate([pairs, pairs[:, ::-1]]), axis=0)
    delta = points[edges[:, 0]] - points[edges[:, 1]]
    return edges.astype(np.int64), np.column_stack(
        (delta, np.linalg.norm(delta, axis=1))
    ).astype(np.float32)


class Dataset:
    def __init__(self, dataset: str | Path, manifest: str | Path, debug: bool = False):
        self.dataset = Path(dataset).expanduser().resolve(strict=True)
        self.manifest_file = Path(manifest).expanduser().resolve(strict=True)
        self.manifest = json.loads(self.manifest_file.read_text())
        validate_manifest(self.manifest)
        self.splits, self.debug = splits(), debug
        self.graph_cache: Path | None = None
        self._static: dict[int, dict] = {}
        self._ids = {case for ids in self.splits.values() for case in ids}
        with h5py.File(self.dataset, "r") as handle:
            if handle.attrs.get("dataset_revision") != REVISION:
                raise ValueError("Ellipse HDF5 revision mismatch")
            for split, ids in self.splits.items():
                for case in ids:
                    group = handle[f"trajectory_{case:04d}"]
                    if group.attrs.get("source_split") != split:
                        raise ValueError(f"source split mismatch for case {case}")
                    if group["uvp"].shape != (65, len(group["mesh_pos"]), 3):
                        raise ValueError(f"invalid prepared UVP shape in case {case}")

    def identity(self) -> dict:
        return {
            "protocol": PROTOCOL,
            "dataset_repository": REPOSITORY,
            "dataset_revision": REVISION,
            "source_revision": SOURCE_REVISION,
            "split_indices": {key: list(value) for key, value in self.splits.items()},
            "dataset_bytes": self.dataset.stat().st_size,
            "normalization": self.manifest["train_only_normalization"],
            "stored_frames": FRAMES,
            "training_frames": [0, 64],
            "evaluation_frames": [0, 64],
            "training_raw_frame_indices": list(FRAME_INDICES),
            "evaluation_raw_frame_indices": list(FRAME_INDICES),
            "dt": DT,
            "boundary_policy": "predict_all_uvp_nodes",
            "debug": self.debug,
        }

    def static(self, index: int) -> dict:
        if index not in self._ids:
            raise ValueError("trajectory is not in the fixed Ellipse population")
        if index not in self._static:
            with h5py.File(self.dataset, "r") as handle:
                group = handle[f"trajectory_{index:04d}"]
                points = np.asarray(group["mesh_pos"], dtype=np.float32)
                cells = np.asarray(group["cells"], dtype=np.int64)
                labels = np.asarray(group["node_type"], dtype=np.int64)
                edges = np.asarray(group["edges"], dtype=np.int64)
            delta = points[edges[:, 0]] - points[edges[:, 1]]
            features = np.column_stack((delta, np.linalg.norm(delta, axis=1))).astype(
                np.float32
            )
            if len(self._static) >= 32:
                self._static.pop(next(iter(self._static)))
            self._static[index] = dict(
                index=index,
                points=points,
                cells=cells,
                node_type=labels,
                edges=edges,
                edge_features=features,
            )
        return self._static[index]

    def read(self, index: int, start: int = 0, stop: int = FRAMES) -> dict:
        if not 0 <= start < stop <= FRAMES:
            raise ValueError("field access must stay within the retained 65 frames")
        geometry = self.static(index)
        with h5py.File(self.dataset, "r") as handle:
            field = np.asarray(
                handle[f"trajectory_{index:04d}/uvp"][start:stop], dtype=np.float32
            )
        if not np.isfinite(field).all():
            raise ValueError(f"nonfinite reference field in Ellipse case {index}")
        return {**geometry, "field": field}

    def evaluation(self, index: int) -> dict:
        return self.read(index, 0, FRAMES)

    def initial(self, index: int) -> dict:
        sample = self.read(index, 0, 1)
        return {key: value for key, value in sample.items() if key != "field"} | {
            "initial": sample["field"][0].copy()
        }
