"""Convert the official Test100 using normalization copied from Train."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import tempfile

import h5py
import numpy as np

from scripts.download_test import DirectoryLock
from .tfrecord_io import decode_example, iter_examples

RAW_INDICES = list(range(0, 513, 8))
DATASETS = {
    "airfoil": (601, 0.0002, "dgn4cfd.mgn_airfoil_uvp_temporal_stride.v1"),
    "cylinderflow": (600, 0.01, "dgn4cfd.mgn_cylinderflow_temporal_stride.v1"),
}


def prepare(dataset: str, raw: Path, destination: Path, train_manifest: Path) -> None:
    """Publish only complete Test100 data; never fit statistics on Test."""
    raw_frames, raw_dt, data_format = DATASETS[dataset]
    training = json.loads(train_manifest.read_text(encoding="utf-8"))
    if (
        training.get("format") != data_format
        or training.get("test_accessed", False)
        or training.get("splits", {}).get("train") != list(range(1000))
        or training.get("splits", {}).get("validation") != list(range(1000, 1100))
        or training.get("temporal_stride") != 8
        or not np.isclose(training.get("frame_dt", 0), raw_dt * 8)
    ):
        raise ValueError("provide the matching official Train/Validation manifest")
    stats = training["train_only_normalization"]
    mean = np.asarray(stats["field_mean"], dtype=np.float64)
    std = np.asarray(stats["field_std"], dtype=np.float64)
    if mean.shape != (3,) or std.shape != (3,):
        raise ValueError("Train normalization must contain three UVP channels")
    if not np.isfinite(mean).all() or not np.isfinite(std).all() or (std <= 0).any():
        raise ValueError("invalid frozen Train normalization")
    metadata = json.loads((raw / "meta.json").read_text(encoding="utf-8"))
    if metadata.get("trajectory_length") != raw_frames or not np.isclose(
        metadata.get("dt", 0), raw_dt, rtol=0, atol=1e-12
    ):
        raise ValueError("raw metadata belongs to a different physical task")
    fields = ("mesh_pos", "cells", "node_type", "velocity", "pressure")
    metadata["features"] = {key: metadata["features"][key] for key in fields}
    destination = destination.expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    with DirectoryLock(destination.with_name(destination.name + ".prepare"), wait=True):
        if destination.exists():
            from .test_data import TestData

            existing = TestData(destination, dataset)
            if existing.manifest["train_only_normalization"] != stats:
                raise ValueError("existing Test data use different Train statistics")
            print(f"Reusing complete Test data: {destination}", flush=True)
            return
        staging = Path(
            tempfile.mkdtemp(prefix=f".{destination.name}-", dir=destination.parent)
        )
        attributes = {
            "format": data_format,
            "frames": 65,
            "raw_frames": raw_frames,
            "source_frames": raw_frames,
            "temporal_stride": 8,
            "phase_offset": 0,
            "phase_augmentation": False,
            "sequences_per_trajectory": 1,
            "raw_frame_dt": raw_dt,
            "frame_dt": raw_dt * 8,
            "trajectory_count": 100,
            "train_count": 0,
            "validation_count": 0,
            "test_count": 100,
            "test_accessed": True,
        }
        records = []
        try:
            with h5py.File(staging / "data.h5", "x") as handle:
                handle.attrs.update(attributes)
                handle.create_dataset("field_mean", data=mean)
                handle.create_dataset("field_std", data=std)
                for index, example in enumerate(iter_examples(raw / "test.tfrecord")):
                    if index >= 100:
                        raise ValueError(
                            "official Test must contain exactly 100 records"
                        )
                    arrays = decode_example(example, metadata)
                    uvp = np.concatenate(
                        (
                            arrays["velocity"][RAW_INDICES],
                            arrays["pressure"][RAW_INDICES],
                        ),
                        axis=-1,
                    )
                    points = arrays["mesh_pos"][0]
                    labels = arrays["node_type"][0].reshape(-1)
                    if uvp.shape != (65, len(points), 3) or not np.isfinite(uvp).all():
                        raise ValueError(f"invalid Test UVP at record {index}")
                    if not np.any(labels == 4):
                        raise ValueError(f"missing inflow nodes in Test record {index}")
                    if dataset == "airfoil":
                        initial_inflow = uvp[0, labels == 4, :2].mean(axis=0)
                        inlet = float(np.linalg.norm(initial_inflow))
                    else:
                        inlet = float(uvp[0, labels == 4, 0].mean(dtype=np.float64))
                    name = f"trajectory_{index:04d}"
                    group = handle.create_group(name)
                    record_attributes = dict(
                        global_index=index,
                        source_local_index=index,
                        source_index=index,
                        source_split="test",
                        frames=65,
                        temporal_stride=8,
                        phase_offset=0,
                        inlet_velocity=inlet,
                        frame_dt=raw_dt * 8,
                    )
                    group.attrs.update(record_attributes)
                    group.create_dataset(
                        "uvp", data=uvp, chunks=(1, len(points), 3), compression="lzf"
                    )
                    group.create_dataset("mesh_pos", data=points)
                    group.create_dataset("cells", data=arrays["cells"][0])
                    group.create_dataset("node_type", data=labels)
                    group.create_dataset("source_frame_indices", data=RAW_INDICES)
                    records.append(
                        dict(
                            **record_attributes,
                            group=name,
                            nodes=len(points),
                            cells=len(arrays["cells"][0]),
                            source_frame_indices=RAW_INDICES,
                        )
                    )
                    print(f"Converted Test {index + 1}/100", flush=True)
            if len(records) != 100:
                raise ValueError(
                    f"official Test contains {len(records)} records, expected 100"
                )
            manifest = {
                **attributes,
                "dataset": "data.h5",
                "dataset_bytes": (staging / "data.h5").stat().st_size,
                "dataset_name": dataset,
                "source_frame_indices": RAW_INDICES,
                "fields": ["u", "v", "p"],
                "density_used": False,
                "splits": {"test": list(range(100))},
                "trajectories": records,
                "train_only_normalization": stats,
                "normalization_source_manifest": str(train_manifest.resolve()),
                "normalization_fitted_on_test": False,
                "source": f"https://storage.googleapis.com/dm-meshgraphnets/{'cylinder_flow' if dataset == 'cylinderflow' else 'airfoil'}/test.tfrecord",
            }
            (staging / "manifest.json").write_text(
                json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
            )
            from .test_data import TestData

            TestData(staging, dataset)
            staging.rename(destination)
        except BaseException:
            print(f"Incomplete conversion retained at {staging}", flush=True)
            raise
    print(f"Ready: {destination}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=tuple(DATASETS), required=True)
    parser.add_argument("--raw-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--train-manifest", type=Path, required=True)
    args = parser.parse_args()
    prepare(args.dataset, args.raw_dir, args.output_dir, args.train_manifest)


if __name__ == "__main__":
    main()
