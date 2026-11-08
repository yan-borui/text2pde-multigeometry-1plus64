"""Prepare all retained high-resolution fields and Train-only moments."""

from __future__ import annotations

import json
from pathlib import Path
import pickle
import time

import h5py
import numpy as np

from .contract import (
    DATA_FILE,
    DT,
    FORMAT,
    FRAME_INDICES,
    FRAMES,
    MANIFEST_FILE,
    RAW_DT,
    REPOSITORY,
    REVISION,
    SOURCE_REVISION,
    splits,
    validate_manifest,
)
from .mesh import recover_mesh
from .source import Source


class Moments:
    def __init__(self, channels: int):
        self.count = 0
        self.mean = np.zeros(channels, dtype=np.float64)
        self.m2 = np.zeros(channels, dtype=np.float64)

    def add(self, values: np.ndarray) -> None:
        values = np.asarray(values, dtype=np.float64).reshape(-1, len(self.mean))
        count = len(values)
        average = values.mean(axis=0)
        delta = average - self.mean
        total = self.count + count
        self.m2 += ((values - average) ** 2).sum(axis=0)
        self.m2 += delta**2 * self.count * count / total
        self.mean += delta * count / total
        self.count = total

    def std(self) -> np.ndarray:
        return np.maximum(np.sqrt(self.m2 / self.count), 1e-8)


def ready(directory: Path) -> bool:
    filename = directory / MANIFEST_FILE
    if not filename.is_file():
        return False
    manifest = json.loads(filename.read_text())
    validate_manifest(manifest)
    dataset = directory / DATA_FILE
    if dataset.stat().st_size != manifest["dataset_bytes"]:
        raise ValueError("prepared HDF5 length differs from manifest")
    with h5py.File(dataset, "r") as handle:
        if handle.attrs.get("dataset_revision") != REVISION:
            raise ValueError("prepared HDF5 revision mismatch")
        for case in (case for ids in splits().values() for case in ids):
            group = handle[f"trajectory_{case:04d}"]
            if group["uvp"].shape != (65, len(group["mesh_pos"]), 3):
                raise ValueError(f"incomplete prepared Ellipse case {case}")
    if not (directory / "text2pde_normalizer.pkl").is_file():
        raise FileNotFoundError("Train-only Text2PDE normalizer is missing")
    return True


def prepare(source: Source, output: Path) -> None:
    assignments = {case: split for split, ids in splits().items() for case in ids}
    staging = output / (DATA_FILE + ".partial")
    # Completed trajectory groups survive an interrupted conversion.
    with h5py.File(staging, "a", libver="earliest") as handle:
        if handle.attrs.get("dataset_revision", REVISION) != REVISION:
            raise ValueError("staging HDF5 belongs to another data contract")
        handle.attrs["dataset_revision"] = REVISION
        for number, case in enumerate(sorted(assignments), 1):
            name = f"trajectory_{case:04d}"
            if name in handle and handle[name].attrs.get("complete", False):
                continue
            if name in handle:
                handle.move(name, f"incomplete_{case}_{time.time_ns()}")
            try:
                with np.load(
                    source.destination(f"raw/mesh/has/ellipse_{case}.npz"),
                    allow_pickle=False,
                ) as raw:
                    geometry = recover_mesh({key: raw[key] for key in raw.files})
                nodes = len(geometry["mesh_pos"])
                frames = []
                for frame in FRAME_INDICES:
                    filename = source.destination(
                        f"raw/has_original/ellipse_{case}_has_{frame}.npz"
                    )
                    with np.load(filename, allow_pickle=False) as raw:
                        if set(raw.files) != {"ux", "uy", "p"} or any(
                            raw[key].shape != (nodes,) for key in raw.files
                        ):
                            raise ValueError(
                                f"UVP/mesh layout mismatch at frame {frame}"
                            )
                        fields = np.column_stack((raw["ux"], raw["uy"], raw["p"]))
                    if not np.isfinite(fields).all():
                        raise ValueError(f"nonfinite original UVP at frame {frame}")
                    frames.append(fields.astype(np.float32))
                uvp = np.stack(frames)
                if not np.isfinite(uvp).all():
                    raise ValueError("UVP cannot be represented in float32")
                speed = float(
                    np.linalg.norm(uvp[0, geometry["node_type"] == 4, :2].mean(axis=0))
                )
                group = handle.create_group(name)
                for key, array in geometry.items():
                    group.create_dataset(key, data=array)
                group.create_dataset(
                    "uvp", data=uvp, chunks=(1, nodes, 3), compression="lzf"
                )
                group.create_dataset("source_frame_indices", data=FRAME_INDICES)
                group.attrs.update(
                    {
                        "global_index": case,
                        "source_index": case,
                        "source_split": assignments[case],
                        "frames": FRAMES,
                        "temporal_stride": 15,
                        "phase_offset": 0,
                        "inlet_velocity": speed,
                        "frame_dt": DT,
                        "complete": True,
                    }
                )
                handle.flush()
            except Exception as error:
                failure = {
                    "trajectory": case,
                    "error": f"{type(error).__name__}: {error}",
                }
                (output / "conversion_failure.json").write_text(
                    json.dumps(failure, indent=2)
                )
                raise
            if number % 25 == 0 or number == len(assignments):
                print(f"Converted Ellipse cases: {number}/982", flush=True)
        fields, inlet = Moments(3), Moments(1)
        for case in splits()["train"]:
            group = handle[f"trajectory_{case:04d}"]
            fields.add(group["uvp"][:])
            inlet.add(np.array([[group.attrs["inlet_velocity"]]]))
        normalization = {
            "field_names": ["u", "v", "p"],
            "field_mean": fields.mean.tolist(),
            "field_std": fields.std().tolist(),
            "inlet_mean": float(inlet.mean[0]),
            "inlet_std": float(inlet.std()[0]),
            "split": "train",
            "frame_range": [0, 64],
            "field_node_frame_count": fields.count,
            "inlet_trajectory_count": inlet.count,
            "inlet_definition": "magnitude of mean geometric-inlet UV at observed first frame",
        }
        attributes = {
            "format": FORMAT,
            "dataset_revision": REVISION,
            "source_revision": SOURCE_REVISION,
            "frames": FRAMES,
            "raw_frames": 1000,
            "source_frames": 1000,
            "temporal_stride": 15,
            "phase_offset": 0,
            "phase_augmentation": False,
            "sequences_per_trajectory": 1,
            "test_accessed": True,
            "trajectory_count": 982,
            "train_count": 782,
            "validation_count": 100,
            "test_count": 100,
            "raw_frame_dt": RAW_DT,
            "frame_dt": DT,
        }
        handle.attrs.update(attributes)
        for name, value in (
            ("field_mean", fields.mean),
            ("field_std", fields.std()),
            ("source_frame_indices", FRAME_INDICES),
        ):
            if name in handle:
                handle[name][...] = value
            else:
                handle.create_dataset(name, data=value)
        trajectories = [
            {
                "global_index": case,
                "source_index": case,
                "source_split": assignments[case],
                "group": f"trajectory_{case:04d}",
                "frames": FRAMES,
                "nodes": len(handle[f"trajectory_{case:04d}/mesh_pos"]),
                "cells": len(handle[f"trajectory_{case:04d}/cells"]),
                "inlet_velocity": float(
                    handle[f"trajectory_{case:04d}"].attrs["inlet_velocity"]
                ),
                "frame_dt": DT,
                "source_frame_indices": list(FRAME_INDICES),
            }
            for case in sorted(assignments)
        ]
    manifest = {
        **attributes,
        "dataset": DATA_FILE,
        "dataset_bytes": staging.stat().st_size,
        "source": f"https://huggingface.co/datasets/{REPOSITORY}",
        "source_frame_indices": list(FRAME_INDICES),
        "relative_time": (np.arange(65) * DT).tolist(),
        "time_origin": "first retained frame; absolute timestamp is not stored upstream",
        "split_seed": 42,
        "splits": {key: list(value) for key, value in splits().items()},
        "trajectories": trajectories,
        "train_only_normalization": normalization,
        "fields": ["u", "v", "p"],
        "density_used": False,
        "boundary_policy": "predict_all_uvp_nodes",
        "mesh_source": "raw/mesh/has",
        "field_source": "raw/has_original",
        "label_source": "static geometric boundary faces",
        "node_labels": {
            "0": "interior",
            "2": "ellipse",
            "4": "inlet",
            "5": "outlet",
            "6": "wall",
        },
        "cells_source": "oriented planar faces of the original bidirectional edge graph",
        "test_use": "conversion and format checks only; normalization fits Train",
    }
    validate_manifest(manifest)
    with (output / "text2pde_normalizer.pkl").open("wb") as stream:
        pickle.dump(
            [
                value
                for pair in zip(fields.mean.tolist(), fields.std().tolist())
                for value in pair
            ],
            stream,
            protocol=4,
        )
    staging.replace(output / DATA_FILE)
    temporary = output / (MANIFEST_FILE + ".partial")
    temporary.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    temporary.replace(output / MANIFEST_FILE)
