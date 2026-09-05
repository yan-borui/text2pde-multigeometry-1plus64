from __future__ import annotations

import copy
import json
import tempfile
import unittest
from pathlib import Path

from tools.cylinderflow_stride8.finalize_validation import finalize
from tools.cylinderflow_stride8.protocol import stage_data_contract


class ValidationFinalizationTest(unittest.TestCase):
    def test_lock_requires_the_same_ae_and_ldm_identities_as_selection(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            ae_file = root / "selected_ae.ckpt"
            ldm_file = root / "selected_ldm.ckpt"
            ae_file.touch()
            ldm_file.touch()
            ae = {
                "test_accessed": False,
                "data_contract": stage_data_contract("ae"),
                "selected": {
                    "checkpoint": str(ae_file),
                    "checkpoint_id": "ae-a",
                    "global_step": 62500,
                    "data_contract": stage_data_contract("ae"),
                },
            }
            ldm = {
                "test_accessed": False,
                "mode": "select",
                "selection_metric": "failure count, then mean uv_relative_rmse",
                "data_contract": stage_data_contract("ldm"),
                "selected": {
                    "checkpoint": str(ldm_file),
                    "checkpoint_id": "ldm-a",
                    "dependencies": {"ae_checkpoint_id": "ae-a"},
                    "global_step": 125000,
                    "total_sequences": 300,
                    "data_contract": stage_data_contract("ldm"),
                },
            }
            validation = copy.deepcopy(ldm)
            validation["mode"] = "validation"
            for directory, filename, payload in (
                ("ae/selection_v1", "selection.json", ae),
                ("evaluation/ldm_selection_v1", "summary.json", ldm),
                ("evaluation/validation_v1", "summary.json", validation),
            ):
                destination = root / directory
                destination.mkdir(parents=True)
                (destination / filename).write_text(
                    json.dumps(payload), encoding="utf-8"
                )
            for directory, checkpoint in (
                ("ae/selection_v1", ae_file),
                ("evaluation/ldm_selection_v1", ldm_file),
            ):
                (root / directory / "selected_checkpoint.txt").write_text(
                    str(checkpoint), encoding="utf-8"
                )
            record = finalize(root)
            self.assertEqual(record["ae_checkpoint_id"], "ae-a")
            self.assertEqual(record["ldm_checkpoint_id"], "ldm-a")
            self.assertFalse(record["test_accessed"])
            for field, value in (
                ("dependencies", {"ae_checkpoint_id": "ae-b"}),
                ("checkpoint_id", "ldm-b"),
                ("data_contract", stage_data_contract("ae")),
            ):
                with self.subTest(field=field):
                    changed = copy.deepcopy(validation)
                    changed["selected"][field] = value
                    (root / "evaluation/validation_v1/summary.json").write_text(
                        json.dumps(changed), encoding="utf-8"
                    )
                    with self.assertRaises(ValueError):
                        finalize(root)


if __name__ == "__main__":
    unittest.main()
