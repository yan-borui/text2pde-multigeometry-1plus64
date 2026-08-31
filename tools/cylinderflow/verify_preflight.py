from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path
from typing import Any


def verify_preflight(summary_path: Path, prepared_dir: Path) -> dict[str, Any]:
    with summary_path.open("r", encoding="utf-8") as handle:
        summary = json.load(handle)
    if summary.get("schema") != "text2pde.cylinderflow.gpu_smoke.v1":
        raise ValueError("unexpected GPU preflight schema")
    if Path(summary["prepared_dir"]).resolve() != prepared_dir.resolve():
        raise ValueError("GPU preflight used a different prepared data package")
    if summary.get("formal_training_started") is not False:
        raise ValueError("GPU preflight summary has invalid training state")
    stages = summary.get("update_stages", [])
    if [stage.get("stage") for stage in stages] != ["ae_update", "ldm_update"]:
        raise ValueError("GPU preflight did not run both update stages")
    if not all(
        stage.get("finite") and stage.get("optimizer_steps") == 1 for stage in stages
    ):
        raise ValueError("GPU preflight update was not finite and complete")
    rollout = summary.get("three_segment_rollout", {})
    if rollout.get("aggregate", {}).get("failure_count") != 0:
        raise ValueError("GPU preflight three-segment rollout failed")
    current_commit = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], text=True
    ).strip()
    if summary.get("repository_commit") != current_commit:
        raise ValueError("GPU preflight was produced by a different source commit")
    return {
        "schema": "text2pde.cylinderflow.preflight_verification.v1",
        "preflight_summary": str(summary_path.resolve()),
        "prepared_dir": str(prepared_dir.resolve()),
        "repository_commit": current_commit,
        "ae_update_finite": True,
        "ldm_update_finite": True,
        "rollout64_finite": True,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--prepared-dir", type=Path, required=True)
    args = parser.parse_args()
    print(
        json.dumps(
            verify_preflight(args.summary, args.prepared_dir),
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
