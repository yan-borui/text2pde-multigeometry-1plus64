from __future__ import annotations

import unittest

import numpy as np
import torch

from tools.cylinderflow.metrics import (
    compute_metrics,
    triangle_vorticity_divergence,
)
from tools.cylinderflow.rollout import derive_segment_seed, rollout_three_segments


class CylinderFlowRolloutTest(unittest.TestCase):
    def test_three_segments_use_only_prior_predictions_and_stitch_64(self) -> None:
        initial = torch.zeros(1, 4, 3)
        observed_conditions = []

        def sample_segment(condition: torch.Tensor, segment_index: int) -> torch.Tensor:
            del segment_index
            observed_conditions.append(condition.clone())
            increments = torch.arange(25, dtype=condition.dtype)[:, None, None]
            return condition + increments

        result = rollout_three_segments(initial, sample_segment)
        self.assertEqual(tuple(result.prediction.shape), (65, 4, 3))
        torch.testing.assert_close(
            result.prediction[:, 0, 0], torch.arange(65, dtype=torch.float32)
        )
        torch.testing.assert_close(observed_conditions[0], initial)
        torch.testing.assert_close(observed_conditions[1], result.segments[0][24:25])
        torch.testing.assert_close(observed_conditions[2], result.segments[1][24:25])
        self.assertEqual(float(observed_conditions[1][0, 0, 0]), 24.0)
        self.assertEqual(float(observed_conditions[2][0, 0, 0]), 48.0)

    def test_segment_seed_is_order_independent_and_segment_specific(self) -> None:
        values = [derive_segment_seed(2, 17, 268, index) for index in range(3)]
        self.assertEqual(values, [derive_segment_seed(2, 17, 268, i) for i in range(3)])
        self.assertEqual(len(set(values)), 3)
        self.assertNotEqual(
            values, [derive_segment_seed(1, 17, 268, i) for i in range(3)]
        )

    def test_linear_triangle_vorticity_and_divergence(self) -> None:
        points = np.array(
            [[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]],
            dtype=np.float64,
        )
        cells = np.array([[0, 1, 2], [0, 2, 3]], dtype=np.int64)
        u = 2.0 * points[:, 0] + 3.0 * points[:, 1]
        v = -points[:, 0] + 4.0 * points[:, 1]
        velocity = np.stack((u, v), axis=-1)[None]
        vorticity, divergence, area = triangle_vorticity_divergence(
            velocity, points, cells
        )
        np.testing.assert_allclose(vorticity, -4.0)
        np.testing.assert_allclose(divergence, 6.0)
        np.testing.assert_allclose(area, 0.5)

    def test_gauge_pressure_and_identical_field_metrics(self) -> None:
        points = np.array(
            [[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]],
            dtype=np.float64,
        )
        cells = np.array([[0, 1, 2], [0, 2, 3]], dtype=np.int64)
        node_type = np.array([4, 5, 6, 0], dtype=np.int32)
        time = np.arange(65, dtype=np.float64)[:, None]
        target = np.zeros((65, 4, 3), dtype=np.float64)
        target[..., 0] = 1.0 + 0.01 * time
        target[..., 1] = points[None, :, 1] + 0.02 * time
        target[..., 2] = points[None, :, 0] - 0.03 * time
        prediction = target.copy()
        prediction[1:, :, 2] += 5.0
        metrics = compute_metrics(prediction, target, points, cells, node_type, dt=0.01)
        self.assertTrue(metrics["finite"])
        self.assertEqual(metrics["uv_relative_rmse"], 0.0)
        self.assertAlmostEqual(metrics["pressure_raw_rmse"], 5.0)
        self.assertAlmostEqual(metrics["pressure_gauge_free_rmse"], 0.0, places=12)
        self.assertEqual(metrics["vorticity_rmse"], 0.0)
        self.assertEqual(metrics["divergence_rmse"], 0.0)
        self.assertEqual(len(metrics["per_frame_uv_relative_rmse"]), 65)

    def test_joint64_metrics_use_stride8_physical_dt_without_segment_seams(
        self,
    ) -> None:
        points = np.array(
            [[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]],
            dtype=np.float64,
        )
        cells = np.array([[0, 1, 2], [0, 2, 3]], dtype=np.int64)
        node_type = np.array([4, 5, 6, 0], dtype=np.int32)
        target = np.zeros((65, 4, 3), dtype=np.float64)
        waveform = 1.5 + 0.4 * np.sin(np.arange(65) * 0.35)
        target[..., 0] = waveform[:, None]
        prediction = target.copy()
        prediction[2:, :, 0] = target[1:-1, :, 0]
        metrics = compute_metrics(
            prediction,
            target,
            points,
            cells,
            node_type,
            dt=0.08,
            include_segment_seams=False,
        )
        self.assertAlmostEqual(
            metrics["energy_best_lag_time"],
            metrics["energy_best_lag_frames"] * 0.08,
        )
        self.assertFalse(any(name.startswith("seam_") for name in metrics))


if __name__ == "__main__":
    unittest.main()
