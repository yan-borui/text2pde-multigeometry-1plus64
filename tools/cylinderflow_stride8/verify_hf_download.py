from __future__ import annotations

import argparse
import json
import pickle
import random
from pathlib import Path

import h5py
import numpy as np

from dataset.cylinderflow_stride8 import CylinderFlowStride8TrajectoryDataset
from tools.cylinderflow_stride8.verify_data import verify


EXPECTED_RELEASE_FILES = [
    "LICENSE",
    "NOTICE",
    "README.md",
    "TEST_NOT_ACCESSED.txt",
    "data/cylinderflow_stride8_75frames.h5",
    "metadata/cylinderflow_stride8_75frames_manifest.json",
    "metadata/dataset_audit.json",
    "metadata/train_normal_stats.pkl",
]
EXPECTED_PLATFORM_FILES = [".gitattributes"]
EXPECTED_FILES = sorted(EXPECTED_PLATFORM_FILES + EXPECTED_RELEASE_FILES)
EXPECTED_NORMALIZER = [
    0.5661443793145688,
    0.57014025360096,
    0.0006225839965382098,
    0.1436773068500826,
    0.04184413450776907,
    0.34627331301859265,
]


def verify_download(root: Path, revision: str) -> dict[str, object]:
    root = root.resolve()
    files = sorted(
        path.relative_to(root).as_posix() for path in root.rglob("*") if path.is_file()
    )
    if files != EXPECTED_FILES:
        raise ValueError(f"downloaded file set differs: {files}")
    data_path = root / "data" / "cylinderflow_stride8_75frames.h5"
    manifest_path = root / "metadata" / "cylinderflow_stride8_75frames_manifest.json"
    identity = verify(data_path, manifest_path, verify_sha256=True)

    with (root / "metadata" / "train_normal_stats.pkl").open("rb") as handle:
        normalizer = pickle.load(handle)
    np.testing.assert_allclose(normalizer, EXPECTED_NORMALIZER, rtol=0, atol=0)
    with h5py.File(data_path, "r") as handle:
        trajectory_groups = sorted(
            name for name in handle if name.startswith("trajectory_")
        )
        if len(trajectory_groups) != 1100:
            raise ValueError(f"download contains {len(trajectory_groups)} trajectories")

    train = CylinderFlowStride8TrajectoryDataset(
        manifest_path, data_path, split="train", return_metadata=True
    )
    validation = CylinderFlowStride8TrajectoryDataset(
        manifest_path, data_path, split="validation", return_metadata=True
    )
    generator = random.Random(20260904)
    random_checks = []
    for split_name, dataset in (("train", train), ("validation", validation)):
        for local_index in generator.sample(range(len(dataset)), 3):
            sample = dataset[local_index]
            if sample["x"].shape[0] != 65 or sample["x"].shape[-1] != 3:
                raise ValueError("random sample has the wrong loader shape")
            if not np.isfinite(sample["x"].numpy()).all():
                raise ValueError("random sample contains non-finite UVP")
            if not np.array_equal(
                sample["frame_indices"].numpy(), np.arange(0, 520, 8)
            ):
                raise ValueError("random sample has the wrong raw-frame mapping")
            random_checks.append(
                {
                    "split": split_name,
                    "local_index": local_index,
                    "trajectory_index": sample["metadata"]["trajectory_index"],
                    "shape": list(sample["x"].shape),
                    "finite": True,
                    "first_raw_frame": 0,
                    "last_raw_frame": 512,
                }
            )
    train.close()
    validation.close()
    return {
        "schema": "text2pde.cylinderflow_stride8.hf_anonymous_download.v1",
        "status": "PASS",
        "revision": revision,
        "root": str(root),
        "files": files,
        "release_payload_files": sorted(EXPECTED_RELEASE_FILES),
        "platform_files": sorted(EXPECTED_PLATFORM_FILES),
        "identity": identity,
        "trajectory_groups": 1100,
        "split_counts": {"train": 1000, "validation": 100},
        "random_checks": random_checks,
        "test_accessed": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--revision", required=True)
    parser.add_argument("--summary", type=Path)
    args = parser.parse_args()
    summary = verify_download(args.root, args.revision)
    encoded = json.dumps(summary, indent=2, sort_keys=True) + "\n"
    if args.summary is not None:
        args.summary.parent.mkdir(parents=True, exist_ok=True)
        args.summary.write_text(encoded, encoding="utf-8")
    print(encoded, end="")


if __name__ == "__main__":
    main()
