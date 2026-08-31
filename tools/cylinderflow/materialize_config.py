from __future__ import annotations

import argparse
import copy
from pathlib import Path
from typing import Any

from modules.utils import get_yaml, save_yaml


def materialize_config(
    template: dict[str, Any],
    prepared_dir: Path,
    result_root: Path,
    stage: str,
) -> dict[str, Any]:
    """Bind a fixed raw-CylinderFlow template to one prepared data package."""

    config = copy.deepcopy(template)
    prepared_dir = prepared_dir.resolve()
    result_root = result_root.resolve()
    dataset = config["data"]["dataset"]
    dataset["train_manifest"] = str(prepared_dir / "train_windows_25.json")
    dataset["validation_manifest"] = str(
        prepared_dir / "validation_monitor_windows_25.json"
    )
    config["data"]["normalizer"]["stat_path"] = str(
        prepared_dir / "train_normal_stats.pkl"
    )
    stage_root = result_root / stage
    config["training"]["default_root_dir"] = str(stage_root)
    config["training"]["run_dir"] = str(stage_root / "formal")
    return config


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--template", type=Path, required=True)
    parser.add_argument("--prepared-dir", type=Path, required=True)
    parser.add_argument("--result-root", type=Path, required=True)
    parser.add_argument("--stage", choices=("ae", "ldm"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    config = materialize_config(
        get_yaml(args.template), args.prepared_dir, args.result_root, args.stage
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    save_yaml(config, args.output)


if __name__ == "__main__":
    main()
