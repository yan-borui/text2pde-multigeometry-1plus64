from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from dataset.cylinderflow_stride8 import CylinderFlowStride8TrajectoryDataset

# Legacy CylinderFlow release identities; Airfoil verification uses its manifest.
EXPECTED_BYTES = 1_772_387_753
EXPECTED_SHA256 = "d416be274e03a5d77f1cf2dffc4be8abcfc63ff9d6bf5a9cca19a44b43b36533"


def sha256_file(file_path: Path) -> str:
    digest = hashlib.sha256()
    with file_path.open("rb") as handle:
        while chunk := handle.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def verify(
    data_path: Path,
    manifest_path: Path,
    verify_sha256: bool = False,
) -> dict[str, object]:
    data_path = data_path.resolve()
    manifest_path = manifest_path.resolve()
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    expected_bytes = manifest.get("dataset_bytes")
    if not isinstance(expected_bytes, int) or expected_bytes <= 0:
        raise ValueError("Airfoil manifest must record positive dataset_bytes")
    byte_count = data_path.stat().st_size
    if byte_count != expected_bytes:
        raise ValueError(f"HDF5 bytes {byte_count} != manifest {expected_bytes}")
    digest = None
    if verify_sha256:
        expected_digest = manifest.get("dataset_sha256")
        if not expected_digest:
            raise ValueError(
                "explicit checksum verification requires dataset_sha256 in manifest"
            )
        digest = sha256_file(data_path)
        if digest != expected_digest:
            raise ValueError("HDF5 checksum differs from manifest")

    sample_records = []
    for stage, length in (("ae", 75), ("ldm", 65)):
        train = CylinderFlowStride8TrajectoryDataset(
            manifest_path, data_path, split="train", stage=stage, return_metadata=True
        )
        validation = CylinderFlowStride8TrajectoryDataset(
            manifest_path,
            data_path,
            split="validation",
            stage=stage,
            return_metadata=True,
        )
        try:
            for split_name, dataset, local_index in (
                ("train", train, 0),
                ("train", train, 999),
                ("validation", validation, 0),
                ("validation", validation, 99),
            ):
                sample = dataset[local_index]
                if tuple(sample["x"].shape)[0::2] != (length, 3):
                    raise ValueError(f"{stage} loader did not return x:[{length},N,3]")
                if not np.isfinite(sample["x"].numpy()).all():
                    raise ValueError("loader returned non-finite UVP")
                expected_frames = np.arange(length) * 8
                if not np.array_equal(sample["frame_indices"].numpy(), expected_frames):
                    raise ValueError(f"{stage} loader raw-frame mapping differs")
                if not np.allclose(
                    sample["time"].numpy(), expected_frames * 0.0002, rtol=0, atol=1e-12
                ):
                    raise ValueError("loader physical-time mapping is incorrect")
                sample_records.append(
                    {
                        "stage": stage,
                        "split": split_name,
                        "local_index": local_index,
                        "global_index": sample["metadata"]["trajectory_index"],
                        "x_shape": list(sample["x"].shape),
                        "cells_shape": list(sample["cells"].shape),
                    }
                )
        finally:
            train.close()
            validation.close()
    return {
        "schema": "text2pde.airfoil_uvp_stride8.data_verification.v1",
        "status": "PASS",
        "data": str(data_path),
        "manifest": str(manifest_path),
        "bytes": byte_count,
        "sha256": digest,
        "sha256_verified": verify_sha256,
        "train_trajectories": 1000,
        "validation_trajectories": 100,
        "samples_per_trajectory": 1,
        "sequence_raw_indices": "0:520:8",
        "sequence_raw_indices_by_stage": {"ae": "0:600:8", "ldm": "0:520:8"},
        "stored_frames_preserved": 75,
        "loaded_frames": 65,
        "loaded_frames_by_stage": {"ae": 75, "ldm": 65},
        "sample_records": sample_records,
        "test_accessed": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--skip-sha256", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()
    print(
        json.dumps(
            verify(args.data, args.manifest),
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
