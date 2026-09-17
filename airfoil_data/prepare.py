"""Convert official Airfoil Train/Validation TFRecords to the shared UVP dataset."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import pickle

import h5py
import numpy as np

from .tfrecord_io import decode_example, iter_examples

FORMAT = "dgn4cfd.mgn_airfoil_uvp_temporal_stride.v1"
REVISION = "airfoil.uvp.stride8.first75.v1"
DATA_FILE = "airfoil_stride8_75frames.h5"
MANIFEST_FILE = "airfoil_stride8_75frames_manifest.json"
RAW_INDICES = np.arange(75, dtype=np.int64) * 8


class Moments:
    """Accumulate population moments without pressure cancellation."""

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


def prepare(raw_dir: Path, output_dir: Path) -> None:
    """Write a new full dataset; Train alone supplies normalization statistics."""
    metadata = json.loads((raw_dir / "meta.json").read_text(encoding="utf-8"))
    if metadata.get("trajectory_length") != 601 or not np.isclose(
        metadata.get("dt", 0), 0.0002, rtol=0, atol=1e-12
    ):
        raise ValueError("expected official Airfoil: 601 frames, raw dt=0.0002")
    if not {
        "velocity",
        "pressure",
        "density",
        "mesh_pos",
        "cells",
        "node_type",
    }.issubset(metadata.get("features", {})):
        raise ValueError("Airfoil metadata is missing required source fields")
    for name in ("train.tfrecord", "valid.tfrecord"):
        if not (raw_dir / name).is_file():
            raise FileNotFoundError(raw_dir / name)
    output_dir.mkdir(parents=True, exist_ok=True)
    dataset_file = output_dir / DATA_FILE
    manifest_file = output_dir / MANIFEST_FILE
    if dataset_file.exists() or manifest_file.exists():
        raise FileExistsError("use a new output directory; existing data are preserved")
    temporary = dataset_file.with_suffix(".partial.h5")
    fields, inflow = Moments(3), Moments(1)
    trajectories = []
    boundary_variation = {"2": 0.0, "4": 0.0}
    with h5py.File(temporary, "x", libver="earliest") as handle:
        for split, filename, expected, offset in (
            ("train", "train.tfrecord", 1000, 0),
            ("validation", "valid.tfrecord", 100, 1000),
        ):
            count = 0
            for ordinal, example in enumerate(iter_examples(raw_dir / filename)):
                if ordinal >= expected:
                    raise ValueError(f"{split} contains more than {expected} records")
                arrays = decode_example(example, metadata)
                points = arrays["mesh_pos"][0].astype(np.float32)
                cells = arrays["cells"][0].astype(np.int64)
                labels = arrays["node_type"][0].reshape(-1).astype(np.int32)
                uvp = np.concatenate(
                    (arrays["velocity"][RAW_INDICES], arrays["pressure"][RAW_INDICES]),
                    axis=-1,
                ).astype(np.float32)
                if uvp.shape != (75, len(points), 3) or not np.isfinite(uvp).all():
                    raise ValueError(f"invalid UVP in {split} trajectory {ordinal}")
                if (
                    not np.isfinite(points).all()
                    or cells.ndim != 2
                    or cells.shape[1] != 3
                ):
                    raise ValueError("invalid triangle mesh")
                if cells.min() < 0 or cells.max() >= len(points):
                    raise ValueError("triangle indices are outside the mesh")
                if not set(np.unique(labels)).issubset({0, 2, 4}) or not np.any(
                    labels == 4
                ):
                    raise ValueError("expected Airfoil NORMAL=0, AIRFOIL=2, INFLOW=4")
                initial_inflow = uvp[0, labels == 4, :2].mean(axis=0)
                speed = float(np.linalg.norm(initial_inflow))
                if split == "train":
                    fields.add(uvp)
                    inflow.add(np.array([[speed]]))
                    for label in (2, 4):
                        if np.any(labels == label):
                            selected = uvp[:, labels == label, :2]
                            boundary_variation[str(label)] = max(
                                boundary_variation[str(label)],
                                float(np.abs(selected - selected[:1]).max()),
                            )
                index = offset + ordinal
                group_name = f"trajectory_{index:04d}"
                group = handle.create_group(group_name)
                group.create_dataset(
                    "uvp", data=uvp, chunks=(1, len(points), 3), compression="lzf"
                )
                group.create_dataset("mesh_pos", data=points)
                group.create_dataset("cells", data=cells)
                group.create_dataset("node_type", data=labels)
                group.create_dataset("source_frame_indices", data=RAW_INDICES)
                attributes = {
                    "global_index": index,
                    "source_index": ordinal,
                    "source_split": split,
                    "frames": 75,
                    "temporal_stride": 8,
                    "phase_offset": 0,
                    "inlet_velocity": speed,
                    "frame_dt": 0.0016,
                }
                group.attrs.update(attributes)
                trajectories.append(
                    {
                        **attributes,
                        "group": group_name,
                        "nodes": len(points),
                        "cells": len(cells),
                        "source_frame_indices": RAW_INDICES.tolist(),
                    }
                )
                count += 1
                if count % 25 == 0:
                    print(f"{split}: {count}/{expected}", flush=True)
            if count != expected:
                raise ValueError(f"{split} has {count} records, expected {expected}")
        normalization = {
            "field_names": ["u", "v", "p"],
            "field_mean": fields.mean.tolist(),
            "field_std": fields.std().tolist(),
            "inlet_mean": float(inflow.mean[0]),
            "inlet_std": float(inflow.std()[0]),
            "split": "train",
            "field_node_frame_count": fields.count,
            "inlet_trajectory_count": inflow.count,
            "frame_range": [0, 74],
            "inlet_definition": "magnitude of mean type-4 UV at observed frame 0",
        }
        root_attributes = {
            "format": FORMAT,
            "frames": 75,
            "raw_frames": 601,
            "source_frames": 601,
            "temporal_stride": 8,
            "phase_offset": 0,
            "sequences_per_trajectory": 1,
            "phase_augmentation": 0,
            "test_accessed": 0,
            "trajectory_count": 1100,
            "train_count": 1000,
            "validation_count": 100,
            "raw_frame_dt": 0.0002,
            "frame_dt": 0.0016,
            "dataset_revision": REVISION,
        }
        handle.attrs.update(root_attributes)
        handle.create_dataset("field_mean", data=fields.mean)
        handle.create_dataset("field_std", data=fields.std())
        handle.create_dataset("source_frame_indices", data=RAW_INDICES)
    manifest = {
        **root_attributes,
        "dataset": DATA_FILE,
        "dataset_bytes": temporary.stat().st_size,
        "phase_augmentation": False,
        "test_accessed": False,
        "source_frame_indices": RAW_INDICES.tolist(),
        "splits": {"train": list(range(1000)), "validation": list(range(1000, 1100))},
        "trajectories": trajectories,
        "train_only_normalization": normalization,
        "source": "https://storage.googleapis.com/dm-meshgraphnets/airfoil",
        "source_bytes": {
            name: (raw_dir / name).stat().st_size
            for name in ("meta.json", "train.tfrecord", "valid.tfrecord")
        },
        "fields": ["u", "v", "p"],
        "density_used": False,
        "boundary_policy": "predict_all_uvp_nodes; no future boundary values or frame-zero clamping",
        "train_boundary_max_uv_change": boundary_variation,
        "omitted_raw_frames": [593, 600],
    }
    temporary.replace(dataset_file)
    manifest_file.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    interleaved = [
        value
        for pair in zip(fields.mean.tolist(), fields.std().tolist())
        for value in pair
    ]
    with (output_dir / "text2pde_normalizer.pkl").open("xb") as stream:
        pickle.dump(interleaved, stream, protocol=4)
    print(f"Prepared Train=1000, Validation=100: {dataset_file}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    prepare(args.raw_dir, args.output_dir)


if __name__ == "__main__":
    main()
