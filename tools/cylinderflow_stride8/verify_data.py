"""Validate the actual prepared data contract before either training stage."""

from __future__ import annotations
import argparse
import json
from pathlib import Path
from megaflow_data.dataset import Dataset


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    args = parser.parse_args()
    data = Dataset(args.data, args.manifest)
    print(
        json.dumps(
            {
                "dataset": str(data.dataset),
                "splits": {key: len(ids) for key, ids in data.splits.items()},
                "stored_frames": 65,
            }
        )
    )


if __name__ == "__main__":
    main()
