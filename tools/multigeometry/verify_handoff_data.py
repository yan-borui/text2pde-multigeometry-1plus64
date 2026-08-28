from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from dataset.multigeometry import MultiGeometryWindowDataset


def verify_handoff_data(data_dir: Path) -> dict[str, object]:
    """Verify transferred files and load one unsealed train/validation window."""
    data_dir = data_dir.resolve()
    inventory = json.loads(
        (data_dir / "data_inventory.json").read_text(encoding="utf-8")
    )
    split_counts: dict[str, int] = {}
    total_bytes = 0
    for record in inventory["files"]:
        file_path = data_dir / record["relative_path"]
        size = file_path.stat().st_size
        if size != int(record["bytes"]):
            raise ValueError(
                f"{record['relative_path']} has {size} bytes, expected {record['bytes']}"
            )
        split = str(record["split"])
        split_counts[split] = split_counts.get(split, 0) + 1
        total_bytes += size

    if split_counts != inventory["split_counts"]:
        raise ValueError(
            f"resolved split counts {split_counts} differ from inventory "
            f"{inventory['split_counts']}"
        )
    if total_bytes != int(inventory["total_bytes"]):
        raise ValueError(
            f"resolved bytes {total_bytes} differ from inventory "
            f"{inventory['total_bytes']}"
        )

    prepared_dir = data_dir / "prepared"
    loaded_windows = {}
    for split, manifest_name in (
        ("train", "train_windows.json"),
        ("validation", "validation_monitor_windows.json"),
    ):
        dataset = MultiGeometryWindowDataset(str(prepared_dir / manifest_name))
        sample = dataset.__getitem__(0, eval=True)
        if sample["x"].ndim != 3 or sample["x"].shape[0] != 65:
            raise ValueError(f"{split} sample has unexpected shape {sample['x'].shape}")
        if not torch.isfinite(sample["x"]).all():
            raise ValueError(f"{split} sample contains non-finite UVP")
        loaded_windows[split] = {
            "case_id": sample["metadata"]["case_id"],
            "shape": list(sample["x"].shape),
        }
        dataset.close()

    sealed_test = MultiGeometryWindowDataset(
        str(prepared_dir / "test_full_windows_sealed.json")
    )
    sealed_test_windows = len(sealed_test)
    sealed_test.close()
    return {
        "trajectory_count": int(inventory["trajectory_count"]),
        "split_counts": split_counts,
        "total_bytes": total_bytes,
        "loaded_windows": loaded_windows,
        "sealed_test_windows": sealed_test_windows,
        "test_fields_read": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Verify a transferred Text2PDE multigeometry data tree."
    )
    parser.add_argument("--data-dir", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(verify_handoff_data(args.data_dir), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
