from __future__ import annotations

import unittest
from pathlib import Path

from modules.utils import get_yaml
from tools.cylinderflow_stride8.materialize_config import materialize_config
from tools.cylinderflow_stride8.protocol import (
    AE_MILESTONES,
    LDM_MILESTONES,
    validate_locked_config,
    validation_monitor_indices,
)


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]


class CylinderFlowStride8ProtocolTest(unittest.TestCase):
    def test_locked_configs_keep_joint64_architecture_and_training_budget(self) -> None:
        for stage in ("ae", "ldm"):
            config = get_yaml(
                REPOSITORY_ROOT
                / "configs"
                / "cylinderflow_stride8"
                / f"{stage}_1plus64.yaml"
            )
            validate_locked_config(config, stage)
            self.assertEqual(config["data"]["batch_size"], 1)
            self.assertEqual(config["training"]["accumulate_grad_batches"], 4)
            self.assertEqual(config["training"]["max_steps"], 250_000)
            self.assertEqual(config["training"]["max_epochs"], 1000)
        self.assertEqual(AE_MILESTONES, (62_500, 125_000, 187_500, 250_000))
        self.assertEqual(LDM_MILESTONES, AE_MILESTONES)

    def test_materializer_binds_one_shared_hdf5_and_manifest(self) -> None:
        template = get_yaml(
            REPOSITORY_ROOT / "configs" / "cylinderflow_stride8" / "ae_1plus64.yaml"
        )
        config = materialize_config(
            template,
            Path("data.h5"),
            Path("manifest.json"),
            Path("stats.pkl"),
            Path("results"),
            "ae",
        )
        self.assertTrue(config["data"]["dataset"]["data_path"].endswith("data.h5"))
        self.assertTrue(config["data"]["dataset"]["manifest"].endswith("manifest.json"))
        self.assertTrue(config["data"]["normalizer"]["stat_path"].endswith("stats.pkl"))

    def test_validation_monitor_is_fixed_unique_and_uniformly_covering(self) -> None:
        indices = validation_monitor_indices()
        self.assertEqual(len(indices), 24)
        self.assertEqual(len(set(indices)), 24)
        self.assertEqual(indices[0], 0)
        self.assertEqual(indices[-1], 99)
        gaps = [right - left for left, right in zip(indices, indices[1:])]
        self.assertLessEqual(max(gaps) - min(gaps), 1)

    def test_new_pipeline_has_no_test_entry_or_three_segment_rollout(self) -> None:
        tools_dir = REPOSITORY_ROOT / "tools" / "cylinderflow_stride8"
        workflow_files = (
            tools_dir / "launch_pipeline.sh",
            tools_dir / "run_ae_stage.sh",
            tools_dir / "run_ldm_stage.sh",
            tools_dir / "evaluate_joint64.py",
        )
        combined = "\n".join(
            file_path.read_text(encoding="utf-8") for file_path in workflow_files
        )
        self.assertNotIn("--mode test", combined)
        self.assertNotIn("run_test_stage", combined)
        self.assertNotIn("rollout_three_segments", combined)
        evaluator = (tools_dir / "evaluate_joint64.py").read_text(encoding="utf-8")
        self.assertIn('choices=("select", "validation")', evaluator)


if __name__ == "__main__":
    unittest.main()
