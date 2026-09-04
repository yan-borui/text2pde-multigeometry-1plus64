from __future__ import annotations

import argparse
import json
from pathlib import Path

from dataset.cylinderflow_stride8 import write_text2pde_normalizer


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    values = write_text2pde_normalizer(args.manifest, args.output)
    print(
        json.dumps(
            {
                "output": str(args.output.resolve()),
                "format": "Text2PDE interleaved [u_mean,u_std,v_mean,v_std,p_mean,p_std]",
                "values": values,
                "statistics_scope": "all 75 phase-zero stride-8 Train frames",
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
