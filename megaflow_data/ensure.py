"""Acquire and prepare the shared Ellipse dataset under process-safe locks."""

from __future__ import annotations

import argparse
from contextlib import redirect_stderr, redirect_stdout
import json
from pathlib import Path
import sys
import traceback

from airfoil_data.ensure import Tee
from airfoil_data.portable_lock import DirectoryLock
from .prepare import prepare, ready
from .source import Source


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--offline", action="store_true")
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()
    if args.workers < 1:
        parser.error("workers must be positive")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    with DirectoryLock(args.output_dir / ".ellipse-prepare.lock", wait=True):
        with (args.output_dir / "preparation.log").open("a", encoding="utf-8") as log:
            with (
                redirect_stdout(Tee(sys.stdout, log)),
                redirect_stderr(Tee(sys.stderr, log)),
            ):
                status = args.output_dir / "preparation_status.json"
                try:
                    if ready(args.output_dir):
                        print(
                            "Reusing complete Ellipse Train782/Validation100/Test100",
                            flush=True,
                        )
                        return
                    status.write_text(json.dumps({"state": "preparing"}))
                    with DirectoryLock(
                        args.raw_dir / ".ellipse-source.lock", wait=True
                    ):
                        source = Source(args.raw_dir, offline=args.offline)
                        source.ensure(args.workers)
                    prepare(source, args.output_dir)
                    status.write_text(
                        json.dumps({"state": "complete", "trajectories": 982})
                    )
                except Exception as error:
                    status.write_text(
                        json.dumps({"state": "failed", "error": str(error)})
                    )
                    traceback.print_exc()
                    raise


if __name__ == "__main__":
    main()
