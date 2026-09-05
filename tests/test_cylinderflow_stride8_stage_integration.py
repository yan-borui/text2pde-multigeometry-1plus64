from __future__ import annotations

import importlib.util
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from uuid import UUID

from dataset.cylinderflow_stride8 import write_text2pde_normalizer
from modules.utils import get_yaml
from tools.cylinderflow_stride8.protocol import stage_data_contract
from test_cylinderflow_stride8 import write_fixture


@unittest.skipUnless(
    importlib.util.find_spec("lightning"), "requires the training environment"
)
class StageIntegrationTest(unittest.TestCase):
    def test_datamodule_binds_each_stage_to_its_configured_frame_count(self):
        from dataset.datamodule import FluidsDataModule

        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            data_file, manifest_file = write_fixture(root)
            stats_file = root / "stats.pkl"
            write_text2pde_normalizer(manifest_file, stats_file)
            for stage, frames in (("ae", 75), ("ldm", 65)):
                config = get_yaml(
                    Path(__file__).resolve().parents[1]
                    / "configs/cylinderflow_stride8"
                    / f"{stage}_1plus64.yaml"
                )["data"]
                config["dataset"].update(
                    data_path=str(data_file),
                    manifest=str(manifest_file),
                    strict_formal_counts=False,
                )
                config["normalizer"]["stat_path"] = str(stats_file)
                module = FluidsDataModule(config)
                self.assertEqual(module.train_dataset[0]["x"].shape[0], frames)
                self.assertEqual(module.val_dataset[0]["x"].shape[0], frames)
                self.assertEqual(len(module.train_dataset), 2)
                self.assertEqual(len(module.val_dataset), 1)
                module.train_dataset.close()
                module.val_dataset.close()

    def test_resume_callback_saves_and_checks_the_stage_contract(self):
        from modules.modules.reproducible_resume import ExactResumeCallback

        trainer = SimpleNamespace(global_step=3, current_epoch=1)
        contract = stage_data_contract("ae")
        callback = ExactResumeCallback(12, data_contract=contract)
        checkpoint = {}
        callback.on_save_checkpoint(trainer, None, checkpoint)
        UUID(checkpoint["cylinderflow_resume_state"]["checkpoint_id"])
        self.assertEqual(
            checkpoint["cylinderflow_resume_state"]["data_contract"], contract
        )
        resumed = ExactResumeCallback(12, data_contract=contract)
        resumed.on_load_checkpoint(trainer, None, checkpoint)
        self.assertIsNotNone(resumed._pending_rng_state)
        wrong = ExactResumeCallback(12, data_contract=stage_data_contract("ldm"))
        with self.assertRaisesRegex(ValueError, "data contract"):
            wrong.on_load_checkpoint(trainer, None, checkpoint)
        del checkpoint["cylinderflow_resume_state"]["data_contract"]
        with self.assertRaisesRegex(ValueError, "data contract"):
            resumed.on_load_checkpoint(trainer, None, checkpoint)

    def test_ldm_resume_requires_the_original_ae_dependency(self):
        from modules.modules.reproducible_resume import ExactResumeCallback

        trainer = SimpleNamespace(global_step=3, current_epoch=1)
        contract = stage_data_contract("ldm")
        dependencies = {"ae_checkpoint_id": "selected-ae"}
        callback = ExactResumeCallback(
            12, data_contract=contract, dependencies=dependencies
        )
        checkpoint = {}
        callback.on_save_checkpoint(trainer, None, checkpoint)
        self.assertEqual(
            checkpoint["cylinderflow_resume_state"]["dependencies"], dependencies
        )
        callback.on_load_checkpoint(trainer, None, checkpoint)
        wrong = ExactResumeCallback(
            12, data_contract=contract, dependencies={"ae_checkpoint_id": "other-ae"}
        )
        with self.assertRaisesRegex(ValueError, "dependencies"):
            wrong.on_load_checkpoint(trainer, None, checkpoint)


if __name__ == "__main__":
    unittest.main()
