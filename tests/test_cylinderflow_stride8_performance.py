from __future__ import annotations

import copy
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch

from tools.cylinderflow_stride8 import performance


class PerformanceContractTest(unittest.TestCase):
    def run_fixture(self, root, *, formal=False, fail_call=None):
        clock = [0.0]
        calls = []
        trajectories = performance.VALIDATION_TRAJECTORIES if formal else (1000, 1099)

        def load_case(index):
            clock[0] += 100
            return {
                "trajectory_index": index,
                "initial": np.ones((4, 3), dtype=np.float32),
                "points": np.asarray(
                    [[0, 0], [1, 0], [1, 1], [0, 1]], dtype=np.float32
                ),
                "cells": np.asarray([[0, 1, 2], [0, 2, 3]], dtype=np.int64),
            }

        def predict(sample):
            calls.append(sample["trajectory_index"])
            # First two calls per case are warmups; next three are repetitions.
            position = (len(calls) - 1) % 5
            elapsed = (
                10
                if position < 2
                else (position - 1) * (1 if sample["trajectory_index"] == 1000 else 2)
            )
            clock[0] += elapsed
            prediction = np.ones((65, 4, 3), dtype=np.float32)
            if len(calls) == fail_call:
                prediction[1:] = np.nan
            return prediction

        with patch.object(
            performance.time, "perf_counter", side_effect=lambda: clock[0]
        ):
            result = performance.benchmark(
                method="fixture",
                indices=trajectories,
                load_case=load_case,
                predict=predict,
                device=torch.device("cpu"),
                output_dir=root,
                data_identity={"release": "fixture"},
                provenance={"checkpoint_id": "fixed"},
                models=[],
                model_load_seconds=17.0,
                formal=formal,
            )
        return result, calls

    def test_warmups_io_and_model_load_are_excluded_from_matched_latency(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory) / "cost"
            summary, calls = self.run_fixture(root)
            self.assertEqual(calls, [1000] * 5 + [1099] * 5)
            self.assertEqual(summary["warmup_samples"], 4)
            self.assertEqual(summary["measured_samples"], 6)
            self.assertEqual(summary["input_load_seconds"], 200)
            self.assertEqual(summary["warmup_inference_seconds"], 40)
            self.assertEqual(summary["measured_inference_seconds"], 18)
            self.assertEqual(summary["latency_seconds"]["mean"], 3)
            self.assertEqual(summary["latency_seconds"]["median"], 3)
            self.assertAlmostEqual(summary["latency_seconds"]["p90"], 3.8)
            self.assertEqual(summary["seconds_per_generated_frame"], 3 / 64)
            self.assertIsNone(summary["cuda_peak_allocated_bytes"])
            self.assertIsNone(summary["measured_inference_gpu_hours"])
            rows = [
                json.loads(line)
                for line in (root / "samples.jsonl").read_text().splitlines()
            ]
            self.assertEqual(len(rows), 10)
            self.assertEqual(summary["model_load_seconds"], 17)

    def test_failed_warmup_invalidates_case_without_hiding_measured_attempts(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory) / "cost"
            summary, _ = self.run_fixture(root, fail_call=1)
            self.assertEqual(summary["status"], "completed_with_failures")
            self.assertEqual(summary["failed_warmups"], 1)
            self.assertEqual(summary["failed_measurements"], 0)
            self.assertEqual(summary["complete_trajectories"], 1)
            self.assertEqual(summary["measured_samples"], 6)
            self.assertEqual(summary["latency_seconds"]["mean"], 4)
            self.assertEqual(
                len(json.loads((root / "failures.json").read_text())["samples"]), 1
            )

    def test_comparison_rejects_different_hardware_runtime_cases_and_failures(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            report, _ = self.run_fixture(
                Path(temporary_directory) / "cost", formal=True
            )
            other = copy.deepcopy(report)
            other["method"] = "other"
            self.assertEqual(
                len(performance.compare_reports([report, other])["methods"]), 2
            )
            for field, key, value in (
                ("environment", "device_name", "different GPU"),
                ("environment", "precision", "bf16"),
                ("environment", "torch_version", "different runtime"),
                ("settings", "warmup_repeats_per_trajectory", 0),
            ):
                changed = copy.deepcopy(other)
                changed[field][key] = value
                with self.subTest(field=field, key=key):
                    with self.assertRaisesRegex(ValueError, "mismatched"):
                        performance.compare_reports([report, changed])
            changed = copy.deepcopy(other)
            changed["case_registry"] = changed["case_registry"][:-1]
            with self.assertRaisesRegex(ValueError, "case_registry"):
                performance.compare_reports([report, changed])
            changed = copy.deepcopy(other)
            changed["status"] = "completed_with_failures"
            with self.assertRaisesRegex(ValueError, "complete formal"):
                performance.compare_reports([report, changed])

    def test_output_contract_and_gpu_memory_units(self):
        sample = {"initial": np.ones((4, 3), dtype=np.float32)}
        with self.assertRaisesRegex(ValueError, "numpy float32"):
            performance.measure_prediction(
                lambda _: np.ones((65, 4, 3)), sample, torch.device("cpu")
            )
        with self.assertRaisesRegex(ValueError, "observed"):
            performance.measure_prediction(
                lambda _: np.zeros((65, 4, 3), dtype=np.float32),
                sample,
                torch.device("cpu"),
            )
        with (
            patch.object(torch.cuda, "synchronize") as synchronized,
            patch.object(torch.cuda, "reset_peak_memory_stats") as reset,
            patch.object(torch.cuda, "memory_allocated", return_value=100),
            patch.object(torch.cuda, "max_memory_allocated", return_value=180),
            patch.object(torch.cuda, "max_memory_reserved", return_value=256),
        ):
            result = performance.measure_prediction(
                lambda _: np.ones((65, 4, 3), dtype=np.float32),
                sample,
                torch.device("cuda:0"),
            )
        reset.assert_called_once()
        self.assertGreaterEqual(synchronized.call_count, 2)
        self.assertEqual(result["cuda_peak_allocated_bytes"], 180)
        self.assertEqual(result["cuda_peak_reserved_bytes"], 256)
        self.assertEqual(result["cuda_incremental_peak_allocated_bytes"], 80)

    def test_rng_reseeding_makes_measured_draw_independent_of_warmups(self):
        performance.seed_draw(1000, 0)
        expected = (np.random.normal(), torch.rand(2))
        for draw in (100, 101):
            performance.seed_draw(1000, draw)
            _ = np.random.normal()
            _ = torch.rand(100)
        performance.seed_draw(1000, 0)
        self.assertEqual(np.random.normal(), expected[0])
        torch.testing.assert_close(torch.rand(2), expected[1], rtol=0, atol=0)


if __name__ == "__main__":
    unittest.main()
