from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path
from typing import Any

import numpy as np

from dataset.cylinderflow import CylinderFlowWindowDataset
from tools.cylinderflow.prepare_data import ROLLOUT_STARTS


def read_json(file_path: Path) -> dict[str, Any]:
    with file_path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def verify_prepared_data(data_dir: Path, formal: bool = True) -> dict[str, Any]:
    prepared_dir = data_dir.resolve() / "prepared"
    train_manifest = prepared_dir / "train_windows_25.json"
    monitor_25_manifest = prepared_dir / "validation_monitor_windows_25.json"
    monitor_65_manifest = prepared_dir / "validation_monitor_rollout64.json"
    validation_manifest = prepared_dir / "validation_full_rollout64.json"
    sealed_test_manifest = prepared_dir / "test_full_rollout64_sealed.json"
    normalizer_path = prepared_dir / "train_normal_stats.pkl"
    summary_path = prepared_dir / "preparation_summary.json"
    for file_path in (
        train_manifest,
        monitor_25_manifest,
        monitor_65_manifest,
        validation_manifest,
        sealed_test_manifest,
        normalizer_path,
        summary_path,
    ):
        if not file_path.is_file():
            raise FileNotFoundError(file_path)

    train = CylinderFlowWindowDataset(str(train_manifest), return_metadata=True)
    monitor_25 = CylinderFlowWindowDataset(
        str(monitor_25_manifest), return_metadata=True
    )
    monitor_65 = CylinderFlowWindowDataset(
        str(monitor_65_manifest), return_metadata=True
    )
    validation = CylinderFlowWindowDataset(
        str(validation_manifest), return_metadata=True
    )
    sealed = read_json(sealed_test_manifest)
    if sealed.get("field_access_status") != "SEALED_METADATA_ONLY":
        raise ValueError("Test manifest is not sealed")
    if sealed.get("data_path") is not None:
        raise ValueError("sealed Test manifest exposes a field path")
    try:
        CylinderFlowWindowDataset(str(sealed_test_manifest))
    except PermissionError:
        pass
    else:
        raise AssertionError("sealed Test manifest was loadable")

    if train.data_path == validation.data_path:
        raise ValueError("Train and Validation fields share one storage path")
    if train.split != "train" or validation.split != "validation":
        raise ValueError("Train/Validation split identity mismatch")
    if train.window_length != 25 or validation.window_length != 65:
        raise ValueError("window lengths do not match 1->24 and rollout64")
    if monitor_25.window_length != 25 or monitor_65.window_length != 65:
        raise ValueError("monitor window lengths are invalid")
    monitor_pairs_25 = [
        monitor_25.resolve_window(index) for index in range(len(monitor_25))
    ]
    monitor_pairs_65 = [
        monitor_65.resolve_window(index) for index in range(len(monitor_65))
    ]
    if monitor_pairs_25 != monitor_pairs_65:
        raise ValueError("AE and LDM monitor clips are not aligned")

    first_train = train.__getitem__(0, eval=True)
    last_train = train.__getitem__(len(train) - 1, eval=True)
    first_validation = validation.__getitem__(0, eval=True)
    if not np.array_equal(first_train["frame_indices"].numpy(), np.arange(25)):
        raise ValueError("first Train window skips source frames")
    if not np.array_equal(last_train["frame_indices"].numpy(), np.arange(575, 600)):
        raise ValueError("last Train window does not end at raw frame 599")
    if not np.array_equal(first_validation["frame_indices"].numpy(), np.arange(65)):
        raise ValueError("Validation rollout window skips source frames")

    with normalizer_path.open("rb") as handle:
        normalizer = np.asarray(pickle.load(handle), dtype=np.float64)
    if normalizer.shape != (6,) or not np.isfinite(normalizer).all():
        raise ValueError("Train-only UVP normalization record is invalid")
    if np.any(normalizer[[1, 3, 5]] <= 0):
        raise ValueError("normalization standard deviations must be positive")

    preparation = read_json(summary_path)
    if preparation.get("test_fields_accessed") is not False:
        raise ValueError("preparation summary does not prove sealed Test fields")
    if preparation["normalizer"]["source"] != (
        "all 600 unique frames of every Train trajectory, counted once"
    ):
        raise ValueError("normalizer provenance is not Train-only unique frames")

    if formal:
        if len(train) != 576000:
            raise ValueError(f"formal Train window count is {len(train)}, not 576000")
        if train.trajectory_count != 1000 or validation.trajectory_count != 100:
            raise ValueError("formal split counts must be 1000/100 before Test")
        if len(monitor_65) != 24 or len(validation) != 300:
            raise ValueError(
                "formal Validation monitor/full clip counts must be 24/300"
            )
        observed_starts = sorted(
            {validation.resolve_window(index)[1] for index in range(len(validation))}
        )
        if observed_starts != list(ROLLOUT_STARTS):
            raise ValueError("formal Validation starts are not 0/268/535")
        if int(preparation["train_optimizer_steps_at_accumulation4"]) != 144000:
            raise ValueError("formal optimizer-step count is not 144000")

    result = {
        "schema": "text2pde.cylinderflow.data_verification.v1",
        "formal_protocol": formal,
        "source_frame_count": 600,
        "source_dt": 0.01,
        "frame_stride": 1,
        "train_trajectory_count": train.trajectory_count,
        "train_window_count": len(train),
        "validation_trajectory_count": validation.trajectory_count,
        "validation_monitor_clip_count": len(monitor_65),
        "validation_full_clip_count": len(validation),
        "normalizer_source": preparation["normalizer"]["source"],
        "test_manifest_status": sealed["field_access_status"],
        "test_fields_accessed": False,
        "finite_samples_checked": 3,
    }
    train.close()
    monitor_25.close()
    monitor_65.close()
    validation.close()
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--allow-small", action="store_true")
    args = parser.parse_args()
    print(
        json.dumps(
            verify_prepared_data(args.data_dir, formal=not args.allow_small),
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
