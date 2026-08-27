from __future__ import annotations

import unittest

import numpy as np

from tools.multigeometry.evaluate_ldm import compute_metrics, node_area_weights


class MultiGeometryMetricTest(unittest.TestCase):
    def setUp(self) -> None:
        self.points = np.array(
            [[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]],
            dtype=np.float64,
        )
        self.cells = np.array([[0, 1, 2], [0, 2, 3]], dtype=np.int64)
        self.boundary = np.array([2, 3, 4, 4], dtype=np.int64)
        time = np.arange(4, dtype=np.float64)[:, None]
        x = self.points[:, 0][None, :]
        y = self.points[:, 1][None, :]
        u = 1.0 + 0.2 * time + x
        v = -0.1 * time + y
        p = 2.0 * x - y + 0.3 * time
        self.target = np.stack((u, v, p), axis=-1)

    def test_node_weights_sum_to_domain_area(self) -> None:
        weights = node_area_weights(self.points, self.cells)
        self.assertAlmostEqual(float(weights.sum()), 1.0)
        self.assertTrue(np.all(weights > 0))

    def test_identical_prediction_has_zero_error(self) -> None:
        metrics = compute_metrics(
            self.target,
            self.target,
            self.points,
            self.cells,
            self.boundary,
            dt=0.1,
        )
        self.assertTrue(metrics["finite"])
        for name in (
            "uv_relative_rmse",
            "pressure_raw_rmse",
            "pressure_gauge_free_rmse",
            "vorticity_rmse",
            "boundary_uv_rmse",
            "conditioning_frame_uv_rmse",
        ):
            self.assertAlmostEqual(float(metrics[name]), 0.0)

    def test_pressure_constant_shift_is_removed_only_by_gauge_metric(self) -> None:
        prediction = self.target.copy()
        prediction[1:, :, 2] += 3.5
        metrics = compute_metrics(
            prediction,
            self.target,
            self.points,
            self.cells,
            self.boundary,
            dt=0.1,
        )
        self.assertGreater(float(metrics["pressure_raw_rmse"]), 3.0)
        self.assertAlmostEqual(float(metrics["pressure_gauge_free_rmse"]), 0.0)


if __name__ == "__main__":
    unittest.main()
