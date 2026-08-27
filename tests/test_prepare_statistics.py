from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import h5py
import numpy as np

from tools.multigeometry.prepare_data import START_MIN, audit_trajectory


class TrainStatisticsTest(unittest.TestCase):
    def test_statistics_use_each_post_ramp_train_frame_once(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            file_path = Path(temporary_directory) / "train_case.h5"
            nodes = 4
            frame = np.arange(601, dtype=np.float32)[:, None]
            node = np.arange(nodes, dtype=np.float32)[None, :]
            u = frame + 0.1 * node
            v = -0.5 * frame + node
            pressure = 0.25 * frame + 2.0 * node
            pressure[:, 2] = 0.0
            uvp = np.stack((u, v, pressure), axis=-1).astype(np.float32)
            with h5py.File(file_path, "w") as handle:
                handle.attrs["schema"] = "dgn4cfd.openlb_multigeometry_uvp.v1"
                handle.attrs["split"] = "train"
                handle.attrs["case_id"] = "ellipse_00"
                handle.create_dataset("uvp", data=uvp)
                handle.create_dataset(
                    "points",
                    data=np.array(
                        [[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]],
                        dtype=np.float32,
                    ),
                )
                handle.create_dataset(
                    "cells",
                    data=np.array([[0, 1, 2], [0, 2, 3]], dtype=np.int32),
                )
                handle.create_dataset(
                    "boundary", data=np.array([0, 2, 3, 4], dtype=np.uint8)
                )
                handle.create_dataset(
                    "time", data=np.arange(601, dtype=np.float64) * 0.1
                )

            trajectory = {
                "case_id": "ellipse_00",
                "node_count": nodes,
                "source_path": str(file_path),
            }
            sums, sums_squared, count, outlet_max = audit_trajectory(
                trajectory,
                expected_split="train",
                accumulate_stats=True,
                frame_chunk=17,
            )
            expected = uvp[START_MIN:601].astype(np.float64)
            np.testing.assert_allclose(sums, expected.sum(axis=(0, 1)), rtol=1e-12)
            np.testing.assert_allclose(
                sums_squared,
                np.square(expected).sum(axis=(0, 1)),
                rtol=1e-12,
            )
            self.assertEqual(count, expected.shape[0] * expected.shape[1])
            self.assertEqual(outlet_max, 0.0)


if __name__ == "__main__":
    unittest.main()
