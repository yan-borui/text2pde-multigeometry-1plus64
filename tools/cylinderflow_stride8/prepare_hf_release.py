from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path

from dataset.cylinderflow_stride8 import write_text2pde_normalizer
from tools.cylinderflow_stride8.verify_data import (
    EXPECTED_BYTES,
    EXPECTED_SHA256,
    verify,
)


EXPECTED_MANIFEST_SHA256 = (
    "b3faaf1fa5371ac41a8c17acf871d015c41492fce495935b12e1207d347b0753"
)


def sha256_file(file_path: Path) -> str:
    digest = hashlib.sha256()
    with file_path.open("rb") as handle:
        while chunk := handle.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def prepare_release(
    data_path: Path,
    manifest_path: Path,
    audit_path: Path,
    license_path: Path,
    output_dir: Path,
) -> dict[str, object]:
    for path in (data_path, manifest_path, audit_path, license_path):
        if not path.is_file():
            raise FileNotFoundError(path)
    if output_dir.exists():
        raise FileExistsError(f"release destination already exists: {output_dir}")
    if sha256_file(manifest_path) != EXPECTED_MANIFEST_SHA256:
        raise ValueError("source manifest does not have the locked identity")
    source_verification = verify(data_path, manifest_path, verify_sha256=True)

    template_dir = Path(__file__).resolve().parent / "hf_release"
    (output_dir / "data").mkdir(parents=True)
    (output_dir / "metadata").mkdir()
    destinations = {
        data_path: output_dir / "data" / "cylinderflow_stride8_75frames.h5",
        manifest_path: output_dir
        / "metadata"
        / "cylinderflow_stride8_75frames_manifest.json",
        audit_path: output_dir / "metadata" / "dataset_audit.json",
        license_path: output_dir / "LICENSE",
        template_dir / "README.md": output_dir / "README.md",
        template_dir / "NOTICE": output_dir / "NOTICE",
        template_dir / "TEST_NOT_ACCESSED.txt": output_dir / "TEST_NOT_ACCESSED.txt",
    }
    for source, destination in destinations.items():
        shutil.copy2(source, destination)
    normalizer_path = output_dir / "metadata" / "train_normal_stats.pkl"
    normalizer_values = write_text2pde_normalizer(
        destinations[manifest_path], normalizer_path
    )

    copied_data = destinations[data_path]
    copied_manifest = destinations[manifest_path]
    if copied_data.stat().st_size != EXPECTED_BYTES:
        raise ValueError("staged HDF5 byte count changed")
    if sha256_file(copied_data) != EXPECTED_SHA256:
        raise ValueError("staged HDF5 hash changed")
    if sha256_file(copied_manifest) != EXPECTED_MANIFEST_SHA256:
        raise ValueError("staged manifest hash changed")
    if "Apache License" not in (output_dir / "LICENSE").read_text(encoding="utf-8"):
        raise ValueError("LICENSE is not an Apache License text")
    staged_verification = verify(copied_data, copied_manifest, verify_sha256=False)
    files = sorted(
        path.relative_to(output_dir).as_posix()
        for path in output_dir.rglob("*")
        if path.is_file()
    )
    expected_files = sorted(
        [
            "LICENSE",
            "NOTICE",
            "README.md",
            "TEST_NOT_ACCESSED.txt",
            "data/cylinderflow_stride8_75frames.h5",
            "metadata/cylinderflow_stride8_75frames_manifest.json",
            "metadata/dataset_audit.json",
            "metadata/train_normal_stats.pkl",
        ]
    )
    if files != expected_files:
        raise ValueError(f"release file set differs: {files}")
    return {
        "schema": "text2pde.cylinderflow_stride8.hf_release_staging.v1",
        "status": "PASS",
        "output_dir": str(output_dir.resolve()),
        "files": files,
        "hdf5_bytes": copied_data.stat().st_size,
        "hdf5_sha256": EXPECTED_SHA256,
        "manifest_sha256": EXPECTED_MANIFEST_SHA256,
        "normalizer_values": normalizer_values,
        "source_verification": source_verification,
        "staged_verification": staged_verification,
        "test_accessed": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--audit", type=Path, required=True)
    parser.add_argument("--license", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--summary", type=Path)
    args = parser.parse_args()
    summary = prepare_release(
        args.data.resolve(),
        args.manifest.resolve(),
        args.audit.resolve(),
        args.license.resolve(),
        args.output_dir.resolve(),
    )
    encoded = json.dumps(summary, indent=2, sort_keys=True) + "\n"
    if args.summary is not None:
        args.summary.parent.mkdir(parents=True, exist_ok=True)
        args.summary.write_text(encoded, encoding="utf-8")
    print(encoded, end="")


if __name__ == "__main__":
    main()
