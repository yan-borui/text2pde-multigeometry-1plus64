from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def read_json(file_path: Path) -> dict[str, Any]:
    with file_path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def read_selected(file_path: Path) -> str:
    value = file_path.read_text(encoding="utf-8").strip()
    if not value or not Path(value).is_file():
        raise FileNotFoundError(value)
    return str(Path(value).resolve())


def finalize(result_root: Path) -> dict[str, Any]:
    result_root = result_root.resolve()
    ae_selection = result_root / "ae" / "selection_v1"
    ldm_selection = result_root / "evaluation" / "ldm_selection_v1"
    validation_dir = result_root / "evaluation" / "validation_v1"
    ae_checkpoint = read_selected(ae_selection / "selected_checkpoint.txt")
    ldm_checkpoint = read_selected(ldm_selection / "selected_checkpoint.txt")
    ae_summary = read_json(ae_selection / "selection.json")
    ldm_summary = read_json(ldm_selection / "summary.json")
    validation_summary = read_json(validation_dir / "summary.json")
    if ae_summary.get("test_accessed") is not False:
        raise ValueError("AE selection does not preserve the Test gate")
    if ldm_summary.get("mode") != "select" or ldm_summary.get("test_accessed"):
        raise ValueError("LDM selection is not Validation-only")
    if validation_summary.get("mode") != "validation" or validation_summary.get(
        "test_accessed"
    ):
        raise ValueError("final evaluation is not Validation-only")
    if str(Path(ldm_summary["selected"]["checkpoint"]).resolve()) != ldm_checkpoint:
        raise ValueError("locked LDM differs from the selection summary")
    if (
        str(Path(validation_summary["selected"]["checkpoint"]).resolve())
        != ldm_checkpoint
    ):
        raise ValueError("Validation did not evaluate the selected LDM checkpoint")

    record = {
        "schema": "text2pde.cylinderflow.validation_lock.v1",
        "status": "validation_complete_awaiting_test",
        "locked_at_utc": datetime.now(timezone.utc).isoformat(),
        "ae_checkpoint": ae_checkpoint,
        "ldm_checkpoint": ldm_checkpoint,
        "ae_global_step": int(ae_summary["selected"]["global_step"]),
        "ldm_global_step": int(ldm_summary["selected"]["global_step"]),
        "selection_metric": ldm_summary["selection_metric"],
        "validation_manifest": validation_summary["manifest"],
        "test_accessed": False,
    }
    evaluation_dir = result_root / "evaluation"
    (evaluation_dir / "locked_ae_checkpoint.txt").write_text(
        ae_checkpoint + "\n", encoding="utf-8"
    )
    (evaluation_dir / "locked_ldm_checkpoint.txt").write_text(
        ldm_checkpoint + "\n", encoding="utf-8"
    )
    with (evaluation_dir / "locked_checkpoints.json").open(
        "w", encoding="utf-8"
    ) as handle:
        json.dump(record, handle, indent=2, sort_keys=True)
        handle.write("\n")
    with (evaluation_dir / "validation_complete_awaiting_test").open(
        "w", encoding="utf-8"
    ) as handle:
        json.dump(record, handle, indent=2, sort_keys=True)
        handle.write("\n")
    return record


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--result-root", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(finalize(args.result_root), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
