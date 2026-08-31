from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch

from modules.utils import get_yaml
from tools.cylinderflow.verify_data import verify_prepared_data


def read_json(file_path: Path) -> dict[str, Any]:
    with file_path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def checkpoint_step(file_path: Path) -> int:
    if not file_path.is_file():
        raise FileNotFoundError(file_path)
    checkpoint = torch.load(file_path, map_location="cpu")
    if "state_dict" not in checkpoint:
        raise ValueError(f"checkpoint has no state_dict: {file_path}")
    return int(checkpoint.get("global_step", -1))


def verify_test_gate(result_root: Path, data_dir: Path) -> dict[str, Any]:
    result_root = result_root.resolve()
    data_dir = data_dir.resolve()
    evaluation_dir = result_root / "evaluation"
    marker = read_json(evaluation_dir / "validation_complete_awaiting_test")
    lock = read_json(evaluation_dir / "locked_checkpoints.json")
    if marker != lock or marker.get("status") != "validation_complete_awaiting_test":
        raise ValueError("Validation lock marker is missing or inconsistent")
    if marker.get("test_accessed") is not False:
        raise ValueError("Validation lock reports prior Test access")

    ae_checkpoint = Path(marker["ae_checkpoint"]).resolve()
    ldm_checkpoint = Path(marker["ldm_checkpoint"]).resolve()
    if checkpoint_step(ae_checkpoint) != int(marker["ae_global_step"]):
        raise ValueError("locked AE checkpoint identity changed")
    if checkpoint_step(ldm_checkpoint) != int(marker["ldm_global_step"]):
        raise ValueError("locked LDM checkpoint identity changed")

    config = get_yaml(result_root / "config" / "ldm_1plus24_raw.yaml")
    if (
        Path(config["data"]["dataset"]["train_manifest"]).resolve()
        != (data_dir / "prepared" / "train_windows_25.json").resolve()
    ):
        raise ValueError("locked training config points at a different Train cache")
    if (
        Path(config["data"]["normalizer"]["stat_path"]).resolve()
        != (data_dir / "prepared" / "train_normal_stats.pkl").resolve()
    ):
        raise ValueError("locked training config points at different statistics")
    data_record = verify_prepared_data(data_dir, formal=True)
    return {
        "schema": "text2pde.cylinderflow.test_gate_verification.v1",
        "status": marker["status"],
        "ae_checkpoint": str(ae_checkpoint),
        "ae_global_step": marker["ae_global_step"],
        "ldm_checkpoint": str(ldm_checkpoint),
        "ldm_global_step": marker["ldm_global_step"],
        "config": str((result_root / "config" / "ldm_1plus24_raw.yaml").resolve()),
        "data": data_record,
        "test_fields_accessed_during_verification": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--result-root", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path, required=True)
    args = parser.parse_args()
    print(
        json.dumps(
            verify_test_gate(args.result_root, args.data_dir),
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
