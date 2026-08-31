from __future__ import annotations

import unittest
from pathlib import Path

from modules.utils import get_yaml


REPO_ROOT = Path(__file__).resolve().parents[1]


class CylinderFlowRunnerTest(unittest.TestCase):
    def test_formal_configs_lock_one_pass_and_three_milestones(self) -> None:
        for file_name in ("ae_1plus24_raw.yaml", "ldm_1plus24_raw.yaml"):
            config = get_yaml(REPO_ROOT / "configs" / "cylinderflow" / file_name)
            self.assertEqual(config["data"]["mode"], "cylinderflow_windows")
            self.assertEqual(config["data"]["batch_size"], 1)
            self.assertEqual(config["data"]["num_workers"], 0)
            self.assertEqual(config["training"]["accumulate_grad_batches"], 4)
            self.assertEqual(config["training"]["dataset_size"], 576000)
            self.assertEqual(config["training"]["max_epochs"], 1)
            self.assertEqual(config["training"]["max_steps"], 144000)
            self.assertEqual(config["training"]["milestone_every_n_steps"], 48000)
            self.assertEqual(config["training"]["last_every_n_steps"], 4000)
            self.assertEqual(
                config["training"]["max_steps"]
                % config["training"]["last_every_n_steps"],
                0,
            )

    def test_main_pipeline_has_no_test_loader_or_test_evaluator(self) -> None:
        cylinder_tools = REPO_ROOT / "tools" / "cylinderflow"
        main_scripts = (
            cylinder_tools / "launch_pipeline.sh",
            cylinder_tools / "run_ae_stage.sh",
            cylinder_tools / "run_ldm_stage.sh",
        )
        combined = "\n".join(
            file_path.read_text(encoding="utf-8") for file_path in main_scripts
        )
        self.assertNotIn("--mode test", combined)
        self.assertNotIn("prepare_test_data", combined)
        self.assertNotIn("test.tfrecord", combined.lower())
        ldm_script = (cylinder_tools / "run_ldm_stage.sh").read_text(encoding="utf-8")
        self.assertIn("tools.cylinderflow.finalize_validation", ldm_script)
        self.assertNotIn("validation_monitor_rollout64.json", ldm_script)
        self.assertGreaterEqual(ldm_script.count("validation_full_rollout64.json"), 2)
        finalize_script = (cylinder_tools / "finalize_validation.py").read_text(
            encoding="utf-8"
        )
        self.assertIn("validation_complete_awaiting_test", finalize_script)

        test_script = (cylinder_tools / "run_test_stage.sh").read_text(encoding="utf-8")
        self.assertIn("prepare_test_data", test_script)
        self.assertIn("--confirm-access-test", test_script)
        self.assertIn("--mode test", test_script)

    def test_all_gpu_launchers_expose_wsl_libcuda_and_smoke_requires_summary(
        self,
    ) -> None:
        cylinder_tools = REPO_ROOT / "tools" / "cylinderflow"
        for file_name in (
            "run_gpu_smoke.sh",
            "run_ae_stage.sh",
            "run_ldm_stage.sh",
            "run_test_stage.sh",
        ):
            content = (cylinder_tools / file_name).read_text(encoding="utf-8")
            self.assertIn("/usr/lib/wsl/lib", content)
        smoke_wrapper = (cylinder_tools / "run_gpu_smoke.sh").read_text(
            encoding="utf-8"
        )
        self.assertIn("summary.json", smoke_wrapper)
        for file_name in (
            "run_gpu_smoke.sh",
            "launch_pipeline.sh",
            "run_test_stage.sh",
        ):
            content = (cylinder_tools / file_name).read_text(encoding="utf-8")
            self.assertNotIn('python_bin=$(realpath "$1")', content)
            self.assertIn("python_dir=", content)


if __name__ == "__main__":
    unittest.main()
