from __future__ import annotations

import argparse
import json
import math
import pickle
from pathlib import Path
from typing import Any

import h5py
import numpy as np

from dataset.multigeometry import WINDOW_MANIFEST_SCHEMA


EXPECTED_SPLIT_COUNTS = {"train": 408, "validation": 24, "test": 24}
EXPECTED_FAMILIES = (
    "ellipse",
    "triangle",
    "rectangle",
    "pentagon",
    "smooth_star",
    "irregular",
    "bunny",
    "elephant",
)
START_MIN = 151
START_MAX = 536
WINDOW_LENGTH = 65
FULL_EVAL_STARTS = (151, 343, 536)


def load_json(file_path: Path) -> dict[str, Any]:
    with file_path.open("r", encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise TypeError(f"expected JSON object in {file_path}")
    return value


def write_json(file_path: Path, value: Any) -> None:
    with file_path.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, sort_keys=True)
        handle.write("\n")


def family_from_case_id(case_id: str) -> str:
    for family in sorted(EXPECTED_FAMILIES, key=len, reverse=True):
        if case_id.startswith(f"{family}_"):
            return family
    raise ValueError(f"unknown geometry family in case_id={case_id}")


def build_trajectories(
    source_manifest: dict[str, Any], split: str
) -> list[dict[str, Any]]:
    cases = int(source_manifest["cases"])
    arrays = {
        "case_ids": source_manifest["case_ids"],
        "source_paths": source_manifest["source_paths"],
        "node_counts": source_manifest["node_counts"],
        "slots": source_manifest["slots"],
        "source_h5_sha256": source_manifest["source_h5_sha256"],
    }
    for name, values in arrays.items():
        if len(values) != cases:
            raise ValueError(f"{split} {name} has {len(values)} entries, expected {cases}")

    trajectories = []
    for index in range(cases):
        case_id = str(arrays["case_ids"][index])
        trajectories.append(
            {
                "case_id": case_id,
                "geometry_family": family_from_case_id(case_id),
                "node_count": int(arrays["node_counts"][index]),
                "slot": int(arrays["slots"][index]),
                "source_h5_sha256": str(arrays["source_h5_sha256"][index]),
                "source_path": str(arrays["source_paths"][index]),
                "split": split,
                "trajectory_index": index,
            }
        )
    return trajectories


def validate_disjoint_splits(
    trajectories_by_split: dict[str, list[dict[str, Any]]]
) -> None:
    for left_name, left in trajectories_by_split.items():
        left_cases = {record["case_id"] for record in left}
        left_paths = {record["source_path"] for record in left}
        if len(left_cases) != len(left) or len(left_paths) != len(left):
            raise ValueError(f"duplicate case or source path within {left_name}")
        for right_name, right in trajectories_by_split.items():
            if left_name >= right_name:
                continue
            right_cases = {record["case_id"] for record in right}
            right_paths = {record["source_path"] for record in right}
            if left_cases & right_cases:
                raise ValueError(f"case overlap between {left_name} and {right_name}")
            if left_paths & right_paths:
                raise ValueError(f"source overlap between {left_name} and {right_name}")


def audit_trajectory(
    trajectory: dict[str, Any],
    expected_split: str,
    accumulate_stats: bool,
    frame_chunk: int,
) -> tuple[np.ndarray, np.ndarray, int, float]:
    sums = np.zeros(3, dtype=np.float64)
    sums_squared = np.zeros(3, dtype=np.float64)
    count = 0
    outlet_pressure_max_abs = 0.0
    source_path = Path(trajectory["source_path"])
    with h5py.File(source_path, "r") as handle:
        required = {"uvp", "points", "cells", "boundary", "time"}
        if not required.issubset(handle.keys()):
            raise ValueError(f"missing datasets in {source_path}: {required - set(handle.keys())}")
        if str(handle.attrs.get("schema")) != "dgn4cfd.openlb_multigeometry_uvp.v1":
            raise ValueError(f"unexpected schema in {source_path}")
        if str(handle.attrs.get("split")) != expected_split:
            raise ValueError(f"split mismatch in {source_path}")
        if str(handle.attrs.get("case_id")) != trajectory["case_id"]:
            raise ValueError(f"case_id mismatch in {source_path}")

        nodes = int(trajectory["node_count"])
        if handle["uvp"].shape != (601, nodes, 3):
            raise ValueError(f"unexpected uvp shape in {source_path}: {handle['uvp'].shape}")
        if handle["points"].shape != (nodes, 2):
            raise ValueError(f"unexpected points shape in {source_path}")
        if handle["boundary"].shape != (nodes,):
            raise ValueError(f"unexpected boundary shape in {source_path}")
        if handle["cells"].ndim != 2 or handle["cells"].shape[1] != 3:
            raise ValueError(f"unexpected cells shape in {source_path}")

        points = np.asarray(handle["points"][:])
        cells = np.asarray(handle["cells"][:])
        boundary = np.asarray(handle["boundary"][:])
        time = np.asarray(handle["time"][:])
        if not np.isfinite(points).all() or not np.isfinite(time).all():
            raise ValueError(f"non-finite geometry or time in {source_path}")
        if cells.min() < 0 or cells.max() >= nodes:
            raise ValueError(f"cell index out of range in {source_path}")
        if set(np.unique(boundary).tolist()) != {0, 2, 3, 4}:
            raise ValueError(f"unexpected boundary codes in {source_path}")
        if time.shape != (601,) or not np.allclose(time, np.arange(601) * 0.1):
            raise ValueError(f"unexpected time grid in {source_path}")

        outlet = boundary == 3
        for frame_start in range(START_MIN, 601, frame_chunk):
            frame_stop = min(frame_start + frame_chunk, 601)
            field = np.asarray(handle["uvp"][frame_start:frame_stop], dtype=np.float64)
            if not np.isfinite(field).all():
                raise ValueError(
                    f"non-finite field in {source_path} frames {frame_start}:{frame_stop}"
                )
            if outlet.any():
                outlet_pressure_max_abs = max(
                    outlet_pressure_max_abs,
                    float(np.abs(field[:, outlet, 2]).max()),
                )
            if accumulate_stats:
                sums += field.sum(axis=(0, 1), dtype=np.float64)
                sums_squared += np.square(field).sum(axis=(0, 1), dtype=np.float64)
                count += field.shape[0] * field.shape[1]

    return sums, sums_squared, count, outlet_pressure_max_abs


def make_manifest(
    split: str,
    trajectories: list[dict[str, Any]],
    source_manifest: dict[str, Any],
    window_mode: str,
    windows: list[dict[str, int]] | None = None,
) -> dict[str, Any]:
    manifest: dict[str, Any] = {
        "schema": WINDOW_MANIFEST_SCHEMA,
        "split": split,
        "window_length": WINDOW_LENGTH,
        "window_mode": window_mode,
        "trajectories": trajectories,
        "source_collection_schema": source_manifest["schema"],
        "source_collection_sha256": source_manifest["sha256"],
        "pressure_representation": "raw_outlet_zero_specific_pressure",
        "position_normalization": "per_trajectory_axiswise_minmax_0_1",
        "time_normalization": "per_window_linear_0_1",
    }
    if window_mode == "dense":
        manifest.update(
            {
                "start_min": START_MIN,
                "start_max": START_MAX,
                "stride": 1,
                "window_count": len(trajectories) * (START_MAX - START_MIN + 1),
            }
        )
    else:
        assert windows is not None
        manifest["windows"] = windows
        manifest["window_count"] = len(windows)
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--campaign-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--frame-chunk", type=int, default=25)
    args = parser.parse_args()

    if args.frame_chunk <= 0:
        raise ValueError("frame chunk must be positive")
    if not (args.campaign_root / "COMPLETED").is_file():
        raise FileNotFoundError("campaign COMPLETED marker is missing")

    final_dir = args.campaign_root / "final"
    final_audit = load_json(args.campaign_root / "final_audit.json")
    if final_audit.get("accepted_cases") != 456:
        raise ValueError("final audit does not contain 456 accepted cases")
    if final_audit.get("split_counts") != EXPECTED_SPLIT_COUNTS:
        raise ValueError(f"unexpected final split counts: {final_audit.get('split_counts')}")

    source_manifests = {
        split: load_json(final_dir / f"{split}_manifest.json")
        for split in EXPECTED_SPLIT_COUNTS
    }
    trajectories_by_split = {
        split: build_trajectories(source_manifests[split], split)
        for split in EXPECTED_SPLIT_COUNTS
    }
    for split, expected_count in EXPECTED_SPLIT_COUNTS.items():
        if len(trajectories_by_split[split]) != expected_count:
            raise ValueError(f"unexpected {split} trajectory count")
    validate_disjoint_splits(trajectories_by_split)

    args.output_dir.mkdir(parents=True, exist_ok=False)
    total_sums = np.zeros(3, dtype=np.float64)
    total_sums_squared = np.zeros(3, dtype=np.float64)
    total_count = 0
    audit_records = []
    for split in ("train", "validation"):
        for trajectory in trajectories_by_split[split]:
            sums, sums_squared, count, outlet_max = audit_trajectory(
                trajectory,
                expected_split=split,
                accumulate_stats=split == "train",
                frame_chunk=args.frame_chunk,
            )
            total_sums += sums
            total_sums_squared += sums_squared
            total_count += count
            audit_records.append(
                {
                    "case_id": trajectory["case_id"],
                    "split": split,
                    "node_count": trajectory["node_count"],
                    "post_ramp_finite": True,
                    "outlet_pressure_max_abs": outlet_max,
                }
            )

    means = total_sums / total_count
    variances = total_sums_squared / total_count - np.square(means)
    stds = np.sqrt(np.maximum(variances, 0.0))
    if not np.isfinite(means).all() or not np.isfinite(stds).all() or np.any(stds <= 0):
        raise ValueError("invalid train-only normalization statistics")
    if any(record["outlet_pressure_max_abs"] != 0.0 for record in audit_records):
        raise ValueError("outlet-zero pressure convention was violated")

    normalizer_path = args.output_dir / "train_normal_stats.pkl"
    with normalizer_path.open("wb") as handle:
        pickle.dump(
            [means[0], stds[0], means[1], stds[1], means[2], stds[2]],
            handle,
        )

    train_manifest = make_manifest(
        "train",
        trajectories_by_split["train"],
        source_manifests["train"],
        "dense",
    )
    write_json(args.output_dir / "train_windows.json", train_manifest)

    largest_train_index = max(
        range(len(trajectories_by_split["train"])),
        key=lambda index: trajectories_by_split["train"][index]["node_count"],
    )
    largest_train_smoke_windows = [
        {"trajectory_index": largest_train_index, "start": start}
        for start in range(START_MIN, START_MIN + 100)
    ]
    largest_train_smoke = make_manifest(
        "train",
        trajectories_by_split["train"],
        source_manifests["train"],
        "explicit",
        largest_train_smoke_windows,
    )
    write_json(args.output_dir / "largest_train_smoke_windows.json", largest_train_smoke)

    validation_trajectories = trajectories_by_split["validation"]
    monitor_windows = []
    for index, trajectory in enumerate(validation_trajectories):
        family_slot = int(trajectory["slot"]) % 57
        if family_slot not in (51, 52, 53):
            raise ValueError(f"unexpected validation family slot: {trajectory}")
        monitor_windows.append(
            {
                "trajectory_index": index,
                "start": FULL_EVAL_STARTS[family_slot - 51],
            }
        )
    validation_monitor = make_manifest(
        "validation",
        validation_trajectories,
        source_manifests["validation"],
        "explicit",
        monitor_windows,
    )
    write_json(args.output_dir / "validation_monitor_windows.json", validation_monitor)

    largest_validation_index = max(
        range(len(validation_trajectories)),
        key=lambda index: validation_trajectories[index]["node_count"],
    )
    largest_validation_smoke = make_manifest(
        "validation",
        validation_trajectories,
        source_manifests["validation"],
        "explicit",
        [{"trajectory_index": largest_validation_index, "start": START_MIN}],
    )
    write_json(
        args.output_dir / "largest_validation_smoke_window.json",
        largest_validation_smoke,
    )

    validation_full_windows = [
        {"trajectory_index": index, "start": start}
        for index in range(len(validation_trajectories))
        for start in FULL_EVAL_STARTS
    ]
    validation_full = make_manifest(
        "validation",
        validation_trajectories,
        source_manifests["validation"],
        "explicit",
        validation_full_windows,
    )
    write_json(args.output_dir / "validation_full_windows.json", validation_full)

    test_trajectories = trajectories_by_split["test"]
    test_full_windows = [
        {"trajectory_index": index, "start": start}
        for index in range(len(test_trajectories))
        for start in FULL_EVAL_STARTS
    ]
    test_full = make_manifest(
        "test",
        test_trajectories,
        source_manifests["test"],
        "explicit",
        test_full_windows,
    )
    test_full["field_access_status"] = "SEALED_METADATA_ONLY"
    write_json(args.output_dir / "test_full_windows_sealed.json", test_full)

    family_counts = {
        split: {
            family: sum(
                record["geometry_family"] == family
                for record in trajectories_by_split[split]
            )
            for family in EXPECTED_FAMILIES
        }
        for split in EXPECTED_SPLIT_COUNTS
    }
    largest_train = trajectories_by_split["train"][largest_train_index]
    summary = {
        "schema": "text2pde.multigeometry.preparation_audit.v1",
        "campaign_root": str(args.campaign_root),
        "final_audit_identity": {
            "manifest_sha256": final_audit["manifest_sha256"],
            "trajectory_manifest_sha256": final_audit["trajectory_manifest_sha256"],
            "immutable_case_hashes_sha256": final_audit["immutable_case_hashes_sha256"],
        },
        "split_counts": EXPECTED_SPLIT_COUNTS,
        "family_counts": family_counts,
        "train_window_count": train_manifest["window_count"],
        "validation_monitor_window_count": len(monitor_windows),
        "largest_train_smoke_window_count": len(largest_train_smoke_windows),
        "validation_full_window_count": len(validation_full_windows),
        "test_full_window_count": len(test_full_windows),
        "test_field_accessed": False,
        "normalizer": {
            "source": "unique train frames 151:601",
            "sample_count_per_channel": total_count,
            "mean": means.tolist(),
            "std": stds.tolist(),
            "pressure_gauge": "raw outlet-zero specific pressure",
        },
        "largest_train": largest_train,
        "train_validation_audit_count": len(audit_records),
        "all_train_validation_post_ramp_finite": True,
        "all_train_validation_outlet_pressure_exact_zero": True,
    }
    if not math.isclose(
        train_manifest["window_count"] / len(trajectories_by_split["train"]),
        START_MAX - START_MIN + 1,
    ):
        raise AssertionError("train window arithmetic mismatch")
    write_json(args.output_dir / "preparation_summary.json", summary)
    write_json(args.output_dir / "train_validation_trajectory_audit.json", audit_records)
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
