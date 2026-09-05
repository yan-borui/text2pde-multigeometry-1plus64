from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from tools.cylinderflow_stride8.protocol import stage_data_contract


def read_json(file_path: Path) -> dict[str, Any]:
    with file_path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def read_selected(file_path: Path) -> str:
    value = file_path.read_text(encoding="utf-8").strip()
    path = Path(value).resolve()
    if not path.is_file():
        raise FileNotFoundError(path)
    return str(path)


def finalize(result_root: Path) -> dict[str, Any]:
    result_root = result_root.resolve()
    ae_selection = read_json(result_root / "ae" / "selection_v1" / "selection.json")
    ldm_selection = read_json(
        result_root / "evaluation" / "ldm_selection_v1" / "summary.json"
    )
    validation = read_json(
        result_root / "evaluation" / "validation_v1" / "summary.json"
    )
    ae_checkpoint = read_selected(
        result_root / "ae" / "selection_v1" / "selected_checkpoint.txt"
    )
    ldm_checkpoint = read_selected(
        result_root / "evaluation" / "ldm_selection_v1" / "selected_checkpoint.txt"
    )
    for label, summary in (
        ("AE selection", ae_selection),
        ("LDM selection", ldm_selection),
        ("Validation", validation),
    ):
        if summary.get("test_accessed") is not False:
            raise ValueError(f"{label} does not preserve the no-Test protocol")
    if ldm_selection.get("mode") != "select":
        raise ValueError("LDM summary is not a selection run")
    if validation.get("mode") != "validation":
        raise ValueError("final summary is not a Validation run")
    if str(Path(ldm_selection["selected"]["checkpoint"]).resolve()) != ldm_checkpoint:
        raise ValueError("selected LDM checkpoint identity differs from its summary")
    if str(Path(validation["selected"]["checkpoint"]).resolve()) != ldm_checkpoint:
        raise ValueError("Validation did not evaluate the selected LDM checkpoint")
    if str(Path(ae_selection["selected"]["checkpoint"]).resolve()) != ae_checkpoint:
        raise ValueError("selected AE checkpoint identity differs from its summary")
    ae_identifier = ae_selection["selected"].get("checkpoint_id")
    ldm_identifier = ldm_selection["selected"].get("checkpoint_id")
    if not ae_identifier or not ldm_identifier:
        raise ValueError("selected checkpoints have no saved unique identity")
    for summary in (ldm_selection, validation):
        if summary["selected"].get("dependencies") != {
            "ae_checkpoint_id": ae_identifier
        }:
            raise ValueError("LDM evaluation used a different AE dependency")
        if summary["selected"].get("checkpoint_id") != ldm_identifier:
            raise ValueError(
                "Validation LDM checkpoint identity differs from selection"
            )
    if validation["selected"]["total_sequences"] != 300:
        raise ValueError("Validation summary does not cover exactly 100 x 3 samples")
    for summary, stage in (
        (ae_selection, "ae"),
        (ldm_selection, "ldm"),
        (validation, "ldm"),
    ):
        expected = stage_data_contract(stage)
        if (
            summary.get("data_contract") != expected
            or summary["selected"].get("data_contract") != expected
        ):
            raise ValueError(
                f"{stage} selection does not match the staged frame contract"
            )

    record = {
        "schema": "text2pde.cylinderflow_stride8.validation_lock.v2",
        "ae_data_contract": stage_data_contract("ae"),
        "ldm_data_contract": stage_data_contract("ldm"),
        "status": "validation_complete_no_test_entry",
        "locked_at_utc": datetime.now(timezone.utc).isoformat(),
        "ae_checkpoint": ae_checkpoint,
        "ldm_checkpoint": ldm_checkpoint,
        "ae_checkpoint_id": ae_identifier,
        "ldm_checkpoint_id": ldm_identifier,
        "ae_global_step": int(ae_selection["selected"]["global_step"]),
        "ldm_global_step": int(ldm_selection["selected"]["global_step"]),
        "selection_metric": ldm_selection["selection_metric"],
        "test_entry_available": False,
        "test_accessed": False,
    }
    evaluation_dir = result_root / "evaluation"
    (evaluation_dir / "locked_ae_checkpoint.txt").write_text(
        ae_checkpoint + "\n", encoding="utf-8"
    )
    (evaluation_dir / "locked_ldm_checkpoint.txt").write_text(
        ldm_checkpoint + "\n", encoding="utf-8"
    )
    with (evaluation_dir / "validation_complete_no_test_entry.json").open(
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
