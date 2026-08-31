from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Iterable

from dataset.cylinderflow import CYLINDERFLOW_WINDOW_SCHEMA
from tools.cylinderflow.prepare_data import (
    ROLLOUT_WINDOW_LENGTH,
    TEST_TRAJECTORIES,
    load_metadata,
    make_full_rollout_windows,
    make_manifest,
    prepare_split,
    write_json,
)


def read_sealed_manifest(file_path: Path) -> dict[str, Any]:
    with file_path.open("r", encoding="utf-8") as handle:
        manifest = json.load(handle)
    if manifest.get("schema") != CYLINDERFLOW_WINDOW_SCHEMA:
        raise ValueError("unexpected sealed Test manifest schema")
    if manifest.get("split") != "test":
        raise ValueError("sealed manifest is not the Test split")
    if manifest.get("field_access_status") != "SEALED_METADATA_ONLY":
        raise ValueError("Test manifest was not sealed")
    if manifest.get("data_path") is not None:
        raise ValueError("sealed Test manifest must not expose a field path")
    return manifest


def prepare_test(
    *,
    metadata_path: Path,
    test_tfrecord: Path,
    sealed_manifest_path: Path,
    output_dir: Path,
    expected_count: int = TEST_TRAJECTORIES,
    examples: Iterable[Any] | None = None,
) -> dict[str, Any]:
    sealed = read_sealed_manifest(sealed_manifest_path)
    if int(sealed["trajectory_count"]) != expected_count:
        raise ValueError("sealed Test trajectory count does not match request")
    if output_dir.exists():
        raise FileExistsError(output_dir)
    storage_dir = output_dir / "storage"
    prepared_dir = output_dir / "prepared"
    storage_dir.mkdir(parents=True)
    prepared_dir.mkdir()

    test_path = storage_dir / "test.h5"
    record = prepare_split(
        metadata=load_metadata(metadata_path),
        tfrecord_path=test_tfrecord,
        destination=test_path,
        split="test",
        expected_count=expected_count,
        accumulate_statistics=False,
        examples=examples,
    )
    manifest = make_manifest(
        split="test",
        data_path=test_path,
        prepared_dir=prepared_dir,
        trajectory_count=expected_count,
        window_length=ROLLOUT_WINDOW_LENGTH,
        window_mode="explicit",
        windows=make_full_rollout_windows(expected_count),
    )
    manifest_path = prepared_dir / "test_full_rollout64.json"
    write_json(manifest_path, manifest)
    summary = {
        "schema": "text2pde.cylinderflow.test_materialization.v1",
        "sealed_manifest": str(sealed_manifest_path.resolve()),
        "source_metadata": str(metadata_path.resolve()),
        "test_record": record,
        "manifest": str(manifest_path.resolve()),
        "test_fields_accessed": True,
        "access_scope": "independent Test launcher after Validation checkpoint lock",
    }
    write_json(prepared_dir / "TEST_FIELDS_MATERIALIZED.json", summary)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Materialize raw MGN Test fields through the independent Test gate."
    )
    parser.add_argument("--meta", type=Path, required=True)
    parser.add_argument("--test-tfrecord", type=Path, required=True)
    parser.add_argument("--sealed-manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--expected-count", type=int, default=TEST_TRAJECTORIES)
    parser.add_argument(
        "--confirm-access-test",
        action="store_true",
        help="Required explicit acknowledgement that Validation selection is locked.",
    )
    args = parser.parse_args()
    if not args.confirm_access_test:
        raise PermissionError("refusing to access Test without --confirm-access-test")
    print(
        json.dumps(
            prepare_test(
                metadata_path=args.meta,
                test_tfrecord=args.test_tfrecord,
                sealed_manifest_path=args.sealed_manifest,
                output_dir=args.output_dir,
                expected_count=args.expected_count,
            ),
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
