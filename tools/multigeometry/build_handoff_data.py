from __future__ import annotations

import argparse
import json
import os
import shutil
from collections import Counter
from pathlib import Path
from typing import Any


PRIMARY_MANIFESTS = (
    "train_windows.json",
    "validation_full_windows.json",
    "test_full_windows_sealed.json",
)
PREPARED_JSON_FILES = (
    "largest_train_smoke_windows.json",
    "train_windows.json",
    "validation_monitor_windows.json",
    "preparation_summary.json",
    "test_full_windows_sealed.json",
    "train_validation_trajectory_audit.json",
    "validation_full_windows.json",
    "largest_validation_smoke_window.json",
)


def load_json(file_path: Path) -> Any:
    return json.loads(file_path.read_text(encoding="utf-8"))


def write_json(file_path: Path, value: Any) -> None:
    file_path.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def collect_trajectories(prepared_dir: Path) -> dict[str, dict[str, Any]]:
    trajectories: dict[str, dict[str, Any]] = {}
    destinations: dict[str, str] = {}
    for manifest_name in PRIMARY_MANIFESTS:
        manifest = load_json(prepared_dir / manifest_name)
        for record in manifest["trajectories"]:
            source_path = str(Path(record["source_path"]).resolve())
            relative_path = Path("trajectories") / record["split"] / (
                f"{record['case_id']}.h5"
            )
            relative_string = relative_path.as_posix()
            prior_source = destinations.get(relative_string)
            if prior_source is not None and prior_source != source_path:
                raise ValueError(
                    f"handoff path {relative_string} names two source trajectories"
                )
            destinations[relative_string] = source_path

            prior_record = trajectories.get(source_path)
            if prior_record is not None:
                if prior_record["relative_path"] != relative_string:
                    raise ValueError(
                        f"source trajectory {source_path} has two handoff identities"
                    )
                continue
            trajectories[source_path] = {
                "case_id": record["case_id"],
                "split": record["split"],
                "geometry_family": record["geometry_family"],
                "source_h5_sha256": record["source_h5_sha256"],
                "relative_path": relative_string,
            }
    return trajectories


def rewrite_source_paths(value: Any, replacements: dict[str, str]) -> Any:
    if isinstance(value, list):
        return [rewrite_source_paths(item, replacements) for item in value]
    if not isinstance(value, dict):
        return value

    rewritten = {
        key: rewrite_source_paths(item, replacements)
        for key, item in value.items()
    }
    if "source_path" in rewritten:
        source_path = str(Path(rewritten["source_path"]).resolve())
        relative_path = replacements[source_path]
        rewritten["source_path"] = f"../{relative_path}"
    return rewritten


def build_handoff_data(
    prepared_dir: Path,
    output_dir: Path,
    file_mode: str,
) -> dict[str, Any]:
    """Create a stable split/case data tree and relative-path manifests."""
    prepared_dir = prepared_dir.resolve()
    output_dir = output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=False)
    portable_prepared_dir = output_dir / "prepared"
    portable_prepared_dir.mkdir()

    trajectories = collect_trajectories(prepared_dir)
    replacements = {
        source_path: record["relative_path"]
        for source_path, record in trajectories.items()
    }
    inventory_rows = []
    for source_path_string, record in sorted(
        trajectories.items(), key=lambda item: item[1]["relative_path"]
    ):
        source_path = Path(source_path_string)
        destination = output_dir / record["relative_path"]
        destination.parent.mkdir(parents=True, exist_ok=True)
        if file_mode == "hardlink":
            os.link(source_path, destination)
        else:
            shutil.copy2(source_path, destination)
        inventory_rows.append(
            {
                **record,
                "bytes": source_path.stat().st_size,
            }
        )

    for file_name in PREPARED_JSON_FILES:
        payload = load_json(prepared_dir / file_name)
        write_json(
            portable_prepared_dir / file_name,
            rewrite_source_paths(payload, replacements),
        )
    shutil.copy2(
        prepared_dir / "train_normal_stats.pkl",
        portable_prepared_dir / "train_normal_stats.pkl",
    )

    split_counts = Counter(row["split"] for row in inventory_rows)
    inventory = {
        "schema": "text2pde.multigeometry.handoff.v1",
        "trajectory_count": len(inventory_rows),
        "split_counts": dict(sorted(split_counts.items())),
        "total_bytes": sum(row["bytes"] for row in inventory_rows),
        "files": inventory_rows,
    }
    write_json(output_dir / "data_inventory.json", inventory)
    return inventory


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build the portable Text2PDE multigeometry data handoff tree."
    )
    parser.add_argument("--prepared-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--file-mode",
        choices=("hardlink", "copy"),
        required=True,
    )
    args = parser.parse_args()
    inventory = build_handoff_data(
        args.prepared_dir,
        args.output_dir,
        args.file_mode,
    )
    print(json.dumps(inventory, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
