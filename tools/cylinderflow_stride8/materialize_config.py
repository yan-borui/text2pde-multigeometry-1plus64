from __future__ import annotations

import argparse
import copy
from pathlib import Path
from typing import Any

from modules.utils import get_yaml, save_yaml
from tools.cylinderflow_stride8.protocol import validate_locked_config


def materialize_config(
    template: dict[str, Any],
    data_path: Path,
    manifest_path: Path,
    normalizer_path: Path,
    result_root: Path,
    stage: str,
) -> dict[str, Any]:
    config = copy.deepcopy(template)
    validate_locked_config(config, stage)
    dataset = config["data"]["dataset"]
    dataset["data_path"] = str(data_path.resolve())
    dataset["manifest"] = str(manifest_path.resolve())
    config["data"]["normalizer"]["stat_path"] = str(normalizer_path.resolve())
    stage_root = result_root.resolve() / stage
    config["training"]["default_root_dir"] = str(stage_root)
    config["training"]["run_dir"] = str(stage_root / "formal")
    validate_locked_config(config, stage)
    return config


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--template", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--normalizer", type=Path, required=True)
    parser.add_argument("--result-root", type=Path, required=True)
    parser.add_argument("--stage", choices=("ae", "ldm"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    for path, label in (
        (args.data, "HDF5"),
        (args.manifest, "manifest"),
        (args.normalizer, "normalizer"),
    ):
        if not path.is_file():
            raise FileNotFoundError(f"{label} is missing: {path}")
    config = materialize_config(
        get_yaml(args.template),
        args.data,
        args.manifest,
        args.normalizer,
        args.result_root,
        args.stage,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    save_yaml(config, args.output)


if __name__ == "__main__":
    main()
