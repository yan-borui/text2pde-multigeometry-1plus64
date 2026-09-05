from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch

from dataset.cylinderflow_stride8 import CylinderFlowStride8TrajectoryDataset
from tools.cylinderflow_stride8.evaluate_joint64 import evaluate_one_checkpoint
from tools.cylinderflow_stride8.metrics import selection_key, summarize_trajectories
from tools.cylinderflow_stride8.predictions import validate_prediction
from tools.cylinderflow_stride8.score import score
from test_cylinderflow_stride8 import write_fixture


class Stride8EvaluationTest(unittest.TestCase):
    def test_trajectory_statistics_and_failures_drive_selection(self):
        rows = [
            {
                "trajectory_index": index,
                "seed": seed,
                "finite": True,
                "uv_relative_rmse": value,
            }
            for index, values in ((0, (0.0, 0.0, 9.0)), (1, (1.0, 1.0, 1.0)))
            for seed, value in enumerate(values)
        ]
        summary = summarize_trajectories(rows)
        self.assertEqual(summary["uv_relative_rmse"]["median"], 2.0)
        self.assertAlmostEqual(summary["uv_relative_rmse"]["p90"], 2.8)
        failed = [dict(row) for row in rows]
        failed[0]["finite"] = False
        failure_summary = summarize_trajectories(failed)
        self.assertEqual(failure_summary["failed_clips"], 1)
        self.assertEqual(failure_summary["failed_trajectories"], 1)
        self.assertEqual(failure_summary["selection_uv_relative_rmse"], 1.0)
        self.assertLess(selection_key(summary, 20), selection_key(failure_summary, 1))
        with self.assertRaisesRegex(ValueError, "duplicate"):
            summarize_trajectories(rows + rows[:1])

    def test_all_seeds_and_failed_arrays_recompute_with_the_common_schema(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            data_file, manifest_file = write_fixture(root)
            dataset = CylinderFlowStride8TrajectoryDataset(
                manifest_file,
                data_file,
                split="validation",
                return_metadata=True,
                strict_formal_counts=False,
            )
            call_count = 0

            def predict(model, sampler, sequence, coordinates, **kwargs):
                nonlocal call_count
                call_count += 1
                if call_count == 2:
                    raise FloatingPointError("synthetic failed seed")
                result = torch.full_like(sequence, 7.0)
                result[:, 0] = sequence[:, 0]
                return result

            module = "tools.cylinderflow_stride8.evaluate_joint64"
            with (
                patch(module + ".instantiate_model", return_value=(object(), 62500)),
                patch(module + ".make_sampler", return_value=object()),
                patch(module + ".sample_joint64", side_effect=predict),
            ):
                result, rows = evaluate_one_checkpoint(
                    root / "ldm.ckpt",
                    root / "ae.ckpt",
                    {"training": {"seed": 42}},
                    None,
                    dataset,
                    (0,),
                    (0, 1, 2),
                    torch.device("cpu"),
                    root / "evaluation" / "predictions",
                )
            dataset.close()
            self.assertEqual(result["total_sequences"], 3)
            self.assertEqual(result["aggregate"]["failed_clips"], 1)
            self.assertEqual(result["aggregate"]["failed_trajectories"], 1)
            archives = sorted((root / "evaluation" / "predictions").glob("*.npz"))
            self.assertEqual(len(archives), 3)
            for file_path in archives:
                with np.load(file_path, allow_pickle=False) as bundle:
                    validate_prediction(bundle)
                    if int(bundle["seed"]) == 1:
                        self.assertTrue(np.isnan(bundle["prediction"][1:]).all())
                    else:
                        fixed = np.isin(bundle["node_type"], (4, 6))
                        np.testing.assert_array_equal(
                            bundle["prediction"][1:, fixed, :2],
                            np.broadcast_to(bundle["target"][0, fixed, :2], (64, 2, 2)),
                        )
                        self.assertTrue((bundle["prediction"][1:, :, 2] == 7).all())
                        self.assertTrue((bundle["prediction"][1:, ~fixed] == 7).all())
                        self.assertTrue((bundle["pre_boundary"][1:] == 7).all())
            recomputed = score(archives, root / "recomputed")
            self.assertEqual(recomputed["failed_clips"], 1)
            self.assertEqual(recomputed["clip_count"], 3)
            self.assertEqual(recomputed["trajectory_count"], 1)
            self.assertEqual(recomputed["selection_uv_relative_rmse"], None)
            with np.load(archives[0]) as bundle:
                changed = dict(bundle)
            changed["physical_time"] = np.arange(65) * 0.64
            with self.assertRaisesRegex(ValueError, "physical time"):
                validate_prediction(changed)


if __name__ == "__main__":
    unittest.main()
