from __future__ import annotations

import argparse
import json
import math
import pickle
from pathlib import Path
from typing import Any, Iterable

import h5py
import numpy as np

from dataset.cylinderflow import (
    CYLINDERFLOW_DATA_SCHEMA,
    CYLINDERFLOW_WINDOW_SCHEMA,
    SOURCE_FRAME_COUNT,
    SOURCE_FRAME_DT,
    SOURCE_NODE_TYPES,
)
from tools.cylinderflow.tfrecord_io import (
    decode_example,
    iter_examples,
    load_metadata,
    static_frame,
)


TRAIN_TRAJECTORIES = 1000
VALIDATION_TRAJECTORIES = 100
TEST_TRAJECTORIES = 100
TRAIN_WINDOW_LENGTH = 25
ROLLOUT_WINDOW_LENGTH = 65
TRAIN_START_MIN = 0
TRAIN_START_MAX = SOURCE_FRAME_COUNT - TRAIN_WINDOW_LENGTH
ROLLOUT_STARTS = (0, 268, 535)
MONITOR_TRAJECTORIES = 24


def write_json(file_path: Path, value: Any) -> None:
    with file_path.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, sort_keys=True)
        handle.write("\n")


def validate_decoded_arrays(
    arrays: dict[str, np.ndarray], record_index: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    velocity = np.asarray(arrays["velocity"], dtype=np.float32)
    pressure = np.asarray(arrays["pressure"], dtype=np.float32)
    mesh_pos = np.asarray(
        static_frame(arrays["mesh_pos"], "mesh_pos"), dtype=np.float32
    )
    node_type = np.asarray(
        static_frame(arrays["node_type"], "node_type"), dtype=np.int32
    ).reshape(-1)
    cells = np.asarray(static_frame(arrays["cells"], "cells"), dtype=np.int32)

    if velocity.ndim != 3 or velocity.shape[0] != SOURCE_FRAME_COUNT:
        raise ValueError(
            f"trajectory {record_index} velocity must have shape [600,N,2], "
            f"got {velocity.shape}"
        )
    if velocity.shape[2] != 2:
        raise ValueError(f"trajectory {record_index} velocity must have two channels")
    node_count = velocity.shape[1]
    if pressure.shape != (SOURCE_FRAME_COUNT, node_count, 1):
        raise ValueError(
            f"trajectory {record_index} pressure shape {pressure.shape} does not "
            f"match velocity"
        )
    if mesh_pos.shape != (node_count, 2):
        raise ValueError(f"trajectory {record_index} mesh_pos shape is invalid")
    if node_type.shape != (node_count,):
        raise ValueError(f"trajectory {record_index} node_type shape is invalid")
    if cells.ndim != 2 or cells.shape[1] != 3 or cells.size == 0:
        raise ValueError(f"trajectory {record_index} cells must have shape [F,3]")
    if cells.min() < 0 or cells.max() >= node_count:
        raise ValueError(f"trajectory {record_index} contains invalid cell indices")
    observed_node_types = set(np.unique(node_type).tolist())
    if not observed_node_types.issubset(SOURCE_NODE_TYPES):
        raise ValueError(
            f"trajectory {record_index} has unsupported node types "
            f"{sorted(observed_node_types)}"
        )
    if not (
        np.isfinite(velocity).all()
        and np.isfinite(pressure).all()
        and np.isfinite(mesh_pos).all()
    ):
        raise ValueError(f"trajectory {record_index} contains non-finite values")

    field = np.concatenate((velocity, pressure), axis=-1)
    return field, mesh_pos, cells, node_type


def prepare_split(
    *,
    metadata: dict[str, Any],
    tfrecord_path: Path,
    destination: Path,
    split: str,
    expected_count: int,
    accumulate_statistics: bool,
    examples: Iterable[Any] | None = None,
) -> dict[str, Any]:
    """Convert one official split to a frame-chunked HDF5 file."""

    if destination.exists():
        raise FileExistsError(destination)
    partial_path = destination.with_suffix(destination.suffix + ".partial")
    if partial_path.exists():
        raise FileExistsError(partial_path)

    source_examples = iter_examples(tfrecord_path) if examples is None else examples
    channel_sum = np.zeros(3, dtype=np.float64)
    channel_sum_squared = np.zeros(3, dtype=np.float64)
    scalar_count = 0
    node_counts: list[int] = []

    with h5py.File(partial_path, "w") as output:
        output.attrs["schema"] = CYLINDERFLOW_DATA_SCHEMA
        output.attrs["split"] = split
        output.attrs["trajectory_count"] = expected_count
        output.attrs["frame_count"] = SOURCE_FRAME_COUNT
        output.attrs["frame_dt"] = SOURCE_FRAME_DT
        output.attrs["frame_stride"] = 1
        output.attrs["source"] = "google-deepmind MeshGraphNets CylinderFlow"

        count = 0
        for record_index, example in enumerate(source_examples):
            if record_index >= expected_count:
                raise ValueError(
                    f"{split} TFRecord contains more than {expected_count} trajectories"
                )
            arrays = decode_example(example, metadata)
            field, mesh_pos, cells, node_type = validate_decoded_arrays(
                arrays, record_index
            )
            node_count = field.shape[1]
            group = output.create_group(str(record_index))
            group.attrs["trajectory_index"] = record_index
            group.attrs["node_count"] = node_count
            group.create_dataset(
                "field",
                data=field,
                chunks=(1, node_count, 3),
            )
            group.create_dataset("mesh_pos", data=mesh_pos)
            group.create_dataset("cells", data=cells)
            group.create_dataset("node_type", data=node_type)
            node_counts.append(node_count)
            count += 1

            if accumulate_statistics:
                values = field.astype(np.float64, copy=False)
                channel_sum += values.sum(axis=(0, 1), dtype=np.float64)
                channel_sum_squared += np.square(values).sum(
                    axis=(0, 1), dtype=np.float64
                )
                scalar_count += SOURCE_FRAME_COUNT * node_count

        if count != expected_count:
            raise ValueError(
                f"{split} TFRecord contains {count} trajectories, expected "
                f"{expected_count}"
            )

    partial_path.replace(destination)
    largest_index = int(np.argmax(node_counts))
    result: dict[str, Any] = {
        "split": split,
        "trajectory_count": expected_count,
        "frame_count": SOURCE_FRAME_COUNT,
        "frame_dt": SOURCE_FRAME_DT,
        "frame_stride": 1,
        "node_count_min": min(node_counts),
        "node_count_max": max(node_counts),
        "largest_trajectory_index": largest_index,
        "source_path": str(tfrecord_path.resolve()),
        "source_bytes": tfrecord_path.stat().st_size
        if tfrecord_path.exists()
        else None,
        "output_path": str(destination.resolve()),
    }
    if accumulate_statistics:
        result.update(
            {
                "channel_sum": channel_sum,
                "channel_sum_squared": channel_sum_squared,
                "scalar_count_per_channel": scalar_count,
            }
        )
    return result


def relative_data_path(prepared_dir: Path, data_path: Path) -> str:
    return str(Path("..") / data_path.parent.name / data_path.name).replace("\\", "/")


def make_manifest(
    *,
    split: str,
    data_path: Path | None,
    prepared_dir: Path,
    trajectory_count: int,
    window_length: int,
    window_mode: str,
    windows: list[dict[str, int]] | None = None,
    start_min: int | None = None,
    start_max: int | None = None,
    sealed: bool = False,
) -> dict[str, Any]:
    manifest: dict[str, Any] = {
        "schema": CYLINDERFLOW_WINDOW_SCHEMA,
        "split": split,
        "dataset_schema": CYLINDERFLOW_DATA_SCHEMA,
        "data_path": (
            relative_data_path(prepared_dir, data_path)
            if data_path is not None
            else None
        ),
        "trajectory_count": trajectory_count,
        "source_frame_count": SOURCE_FRAME_COUNT,
        "frame_dt": SOURCE_FRAME_DT,
        "frame_stride": 1,
        "window_length": window_length,
        "window_mode": window_mode,
        "time_coordinates": "per_window_linear_0_1",
        "physical_time": "source_frame_index_times_0.01",
        "channels": ["u", "v", "p"],
    }
    if window_mode == "dense":
        if start_min is None or start_max is None:
            raise ValueError("dense manifest requires start_min and start_max")
        manifest.update(
            {
                "start_min": start_min,
                "start_max": start_max,
                "start_stride": 1,
                "window_count": trajectory_count * (start_max - start_min + 1),
            }
        )
    elif window_mode == "explicit":
        if not windows:
            raise ValueError("explicit manifest requires windows")
        manifest["windows"] = windows
        manifest["window_count"] = len(windows)
    else:
        raise ValueError(f"unsupported window mode: {window_mode}")
    if sealed:
        manifest["field_access_status"] = "SEALED_METADATA_ONLY"
    return manifest


def monitor_trajectory_indices(trajectory_count: int) -> list[int]:
    if trajectory_count <= 0:
        raise ValueError("validation requires at least one trajectory")
    monitor_count = min(MONITOR_TRAJECTORIES, trajectory_count)
    indices = np.rint(np.linspace(0, trajectory_count - 1, monitor_count)).astype(
        np.int64
    )
    values = indices.tolist()
    if len(set(values)) != monitor_count:
        raise AssertionError("monitor trajectory selection produced duplicates")
    return values


def make_monitor_windows(trajectory_count: int) -> list[dict[str, int]]:
    return [
        {
            "trajectory_index": trajectory_index,
            "start": ROLLOUT_STARTS[rank % len(ROLLOUT_STARTS)],
        }
        for rank, trajectory_index in enumerate(
            monitor_trajectory_indices(trajectory_count)
        )
    ]


def make_full_rollout_windows(trajectory_count: int) -> list[dict[str, int]]:
    return [
        {"trajectory_index": trajectory_index, "start": start}
        for trajectory_index in range(trajectory_count)
        for start in ROLLOUT_STARTS
    ]


def save_normalizer(split_record: dict[str, Any], destination: Path) -> dict[str, Any]:
    count = int(split_record["scalar_count_per_channel"])
    channel_sum = np.asarray(split_record["channel_sum"], dtype=np.float64)
    channel_sum_squared = np.asarray(
        split_record["channel_sum_squared"], dtype=np.float64
    )
    means = channel_sum / count
    variances = channel_sum_squared / count - np.square(means)
    stds = np.sqrt(np.maximum(variances, 0.0))
    if not np.isfinite(means).all() or not np.isfinite(stds).all():
        raise ValueError("train normalization statistics are non-finite")
    if np.any(stds <= 0):
        raise ValueError("train normalization standard deviations must be positive")
    with destination.open("wb") as handle:
        pickle.dump(
            [means[0], stds[0], means[1], stds[1], means[2], stds[2]],
            handle,
        )
    return {
        "source": "all 600 unique frames of every Train trajectory, counted once",
        "sample_count_per_channel": count,
        "mean_uvp": means.tolist(),
        "std_uvp": stds.tolist(),
        "path": str(destination.resolve()),
    }


def prepare_train_validation(
    *,
    metadata_path: Path,
    train_tfrecord: Path,
    validation_tfrecord: Path,
    output_dir: Path,
    expected_train_count: int = TRAIN_TRAJECTORIES,
    expected_validation_count: int = VALIDATION_TRAJECTORIES,
    train_examples: Iterable[Any] | None = None,
    validation_examples: Iterable[Any] | None = None,
) -> dict[str, Any]:
    if output_dir.exists():
        raise FileExistsError(output_dir)
    storage_dir = output_dir / "storage"
    prepared_dir = output_dir / "prepared"
    storage_dir.mkdir(parents=True)
    prepared_dir.mkdir()
    metadata = load_metadata(metadata_path)

    train_path = storage_dir / "train.h5"
    validation_path = storage_dir / "validation.h5"
    train_record = prepare_split(
        metadata=metadata,
        tfrecord_path=train_tfrecord,
        destination=train_path,
        split="train",
        expected_count=expected_train_count,
        accumulate_statistics=True,
        examples=train_examples,
    )
    validation_record = prepare_split(
        metadata=metadata,
        tfrecord_path=validation_tfrecord,
        destination=validation_path,
        split="validation",
        expected_count=expected_validation_count,
        accumulate_statistics=False,
        examples=validation_examples,
    )
    normalizer = save_normalizer(train_record, prepared_dir / "train_normal_stats.pkl")

    monitor_windows = make_monitor_windows(expected_validation_count)
    manifests = {
        "train_windows_25.json": make_manifest(
            split="train",
            data_path=train_path,
            prepared_dir=prepared_dir,
            trajectory_count=expected_train_count,
            window_length=TRAIN_WINDOW_LENGTH,
            window_mode="dense",
            start_min=TRAIN_START_MIN,
            start_max=TRAIN_START_MAX,
        ),
        "validation_monitor_windows_25.json": make_manifest(
            split="validation",
            data_path=validation_path,
            prepared_dir=prepared_dir,
            trajectory_count=expected_validation_count,
            window_length=TRAIN_WINDOW_LENGTH,
            window_mode="explicit",
            windows=monitor_windows,
        ),
        "validation_monitor_rollout64.json": make_manifest(
            split="validation",
            data_path=validation_path,
            prepared_dir=prepared_dir,
            trajectory_count=expected_validation_count,
            window_length=ROLLOUT_WINDOW_LENGTH,
            window_mode="explicit",
            windows=monitor_windows,
        ),
        "validation_full_rollout64.json": make_manifest(
            split="validation",
            data_path=validation_path,
            prepared_dir=prepared_dir,
            trajectory_count=expected_validation_count,
            window_length=ROLLOUT_WINDOW_LENGTH,
            window_mode="explicit",
            windows=make_full_rollout_windows(expected_validation_count),
        ),
        "largest_train_smoke_windows_4.json": make_manifest(
            split="train",
            data_path=train_path,
            prepared_dir=prepared_dir,
            trajectory_count=expected_train_count,
            window_length=TRAIN_WINDOW_LENGTH,
            window_mode="explicit",
            windows=[
                {
                    "trajectory_index": train_record["largest_trajectory_index"],
                    "start": start,
                }
                for start in range(4)
            ],
        ),
        "test_full_rollout64_sealed.json": make_manifest(
            split="test",
            data_path=None,
            prepared_dir=prepared_dir,
            trajectory_count=TEST_TRAJECTORIES,
            window_length=ROLLOUT_WINDOW_LENGTH,
            window_mode="explicit",
            windows=make_full_rollout_windows(TEST_TRAJECTORIES),
            sealed=True,
        ),
    }
    for file_name, manifest in manifests.items():
        write_json(prepared_dir / file_name, manifest)

    summary = {
        "schema": "text2pde.cylinderflow.preparation.v1",
        "source_metadata": str(metadata_path.resolve()),
        "source_frame_count": SOURCE_FRAME_COUNT,
        "frame_dt": SOURCE_FRAME_DT,
        "frame_stride": 1,
        "train": {
            key: value
            for key, value in train_record.items()
            if key not in {"channel_sum", "channel_sum_squared"}
        },
        "validation": validation_record,
        "normalizer": normalizer,
        "train_window_count": manifests["train_windows_25.json"]["window_count"],
        "train_optimizer_steps_at_accumulation4": (
            manifests["train_windows_25.json"]["window_count"] // 4
        ),
        "validation_monitor_clip_count": len(monitor_windows),
        "validation_full_clip_count": len(
            manifests["validation_full_rollout64.json"]["windows"]
        ),
        "validation_rollout_starts": list(ROLLOUT_STARTS),
        "test_fields_accessed": False,
    }
    expected_windows = expected_train_count * (
        SOURCE_FRAME_COUNT - TRAIN_WINDOW_LENGTH + 1
    )
    if summary["train_window_count"] != expected_windows:
        raise AssertionError("train window arithmetic mismatch")
    if expected_train_count == TRAIN_TRAJECTORIES and expected_windows != 576000:
        raise AssertionError("formal train window count must be 576000")
    if (
        expected_train_count == TRAIN_TRAJECTORIES
        and summary["train_optimizer_steps_at_accumulation4"] != 144000
    ):
        raise AssertionError("formal optimizer step count must be 144000")
    write_json(prepared_dir / "preparation_summary.json", summary)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Prepare raw 600-frame MeshGraphNets CylinderFlow Train and Validation "
            "splits without temporal downsampling."
        )
    )
    parser.add_argument("--meta", type=Path, required=True)
    parser.add_argument("--train-tfrecord", type=Path, required=True)
    parser.add_argument("--validation-tfrecord", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--expected-train-count", type=int, default=TRAIN_TRAJECTORIES)
    parser.add_argument(
        "--expected-validation-count", type=int, default=VALIDATION_TRAJECTORIES
    )
    args = parser.parse_args()
    summary = prepare_train_validation(
        metadata_path=args.meta,
        train_tfrecord=args.train_tfrecord,
        validation_tfrecord=args.validation_tfrecord,
        output_dir=args.output_dir,
        expected_train_count=args.expected_train_count,
        expected_validation_count=args.expected_validation_count,
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
