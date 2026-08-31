from __future__ import annotations

import json
import pickle
import struct
import tempfile
import unittest
from pathlib import Path

import numpy as np

from dataset.cylinderflow import CylinderFlowWindowDataset
from tools.cylinderflow.extract_tfrecord_prefix import extract_records
from tools.cylinderflow.prepare_data import prepare_train_validation
from tools.cylinderflow.prepare_test_data import prepare_test
from tools.cylinderflow.tfrecord_io import decode_example, load_metadata


class _ByteList:
    def __init__(self, payload: bytes) -> None:
        self.value = [payload]


class _Feature:
    def __init__(self, payload: bytes) -> None:
        self.bytes_list = _ByteList(payload)


class _Features:
    def __init__(self, arrays: dict[str, np.ndarray]) -> None:
        self.feature = {
            name: _Feature(np.ascontiguousarray(value).tobytes())
            for name, value in arrays.items()
        }


class _Example:
    def __init__(self, arrays: dict[str, np.ndarray]) -> None:
        self.features = _Features(arrays)


def make_arrays(offset: float) -> dict[str, np.ndarray]:
    frame = np.arange(600, dtype=np.float32)[:, None]
    node = np.arange(4, dtype=np.float32)[None, :]
    velocity = np.stack(
        (
            0.01 * frame + 0.1 * node + offset,
            -0.02 * frame + 0.2 * node - offset,
        ),
        axis=-1,
    ).astype(np.float32)
    pressure = (0.03 * frame + 0.05 * node + 2 * offset)[..., None].astype(np.float32)
    return {
        "velocity": velocity,
        "pressure": pressure,
        "mesh_pos": np.array(
            [[[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]]],
            dtype=np.float32,
        ),
        "node_type": np.array([[[4], [5], [6], [0]]], dtype=np.int32),
        "cells": np.array([[[0, 1, 2], [0, 2, 3]]], dtype=np.int32),
    }


def write_metadata(file_path: Path) -> dict:
    arrays = make_arrays(0.0)
    metadata = {
        "trajectory_length": 600,
        "dt": 0.01,
        "features": {
            name: {"dtype": str(value.dtype), "shape": list(value.shape)}
            for name, value in arrays.items()
        },
    }
    file_path.write_text(json.dumps(metadata), encoding="utf-8")
    return metadata


class CylinderFlowDataTest(unittest.TestCase):
    def test_extract_complete_tfrecord_range_preserves_record_bytes(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            source = root / "source.tfrecord"
            records = []
            for index in range(4):
                payload = bytes([index]) * (index + 2)
                record = struct.pack("<Q", len(payload)) + b"LCRC" + payload + b"PCRC"
                records.append(record)
            source.write_bytes(b"".join(records) + b"truncated")
            destination = root / "subset.tfrecord"
            summary = extract_records(source, destination, count=2, skip=1)
            self.assertEqual(summary["record_count"], 2)
            self.assertEqual(destination.read_bytes(), records[1] + records[2])

    def test_prepare_contiguous_windows_train_only_stats_and_test_gate(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            metadata_path = root / "meta.json"
            write_metadata(metadata_path)
            train_arrays = (make_arrays(0.0), make_arrays(3.0))
            validation_arrays = (make_arrays(20.0),)
            output_dir = root / "prepared_data"
            summary = prepare_train_validation(
                metadata_path=metadata_path,
                train_tfrecord=root / "train.tfrecord",
                validation_tfrecord=root / "valid.tfrecord",
                output_dir=output_dir,
                expected_train_count=2,
                expected_validation_count=1,
                train_examples=[_Example(value) for value in train_arrays],
                validation_examples=[_Example(value) for value in validation_arrays],
            )
            self.assertEqual(summary["train_window_count"], 2 * 576)
            self.assertEqual(summary["validation_full_clip_count"], 3)
            self.assertFalse(summary["test_fields_accessed"])

            train = CylinderFlowWindowDataset(
                str(output_dir / "prepared" / "train_windows_25.json"),
                return_metadata=True,
            )
            validation = CylinderFlowWindowDataset(
                str(output_dir / "prepared" / "validation_full_rollout64.json"),
                return_metadata=True,
            )
            self.assertEqual(len(train), 1152)
            self.assertEqual(train.resolve_window(0), (0, 0))
            self.assertEqual(train.resolve_window(575), (0, 575))
            self.assertEqual(train.resolve_window(576), (1, 0))
            np.testing.assert_array_equal(
                train[-1]["frame_indices"].numpy(), np.arange(575, 600)
            )
            self.assertEqual(
                [validation.resolve_window(index)[1] for index in range(3)],
                [0, 268, 535],
            )
            self.assertNotEqual(train.data_path, validation.data_path)

            all_train_fields = np.concatenate(
                [
                    np.concatenate((value["velocity"], value["pressure"]), axis=-1)
                    for value in train_arrays
                ],
                axis=1,
            ).astype(np.float64)
            expected_mean = all_train_fields.mean(axis=(0, 1))
            expected_std = all_train_fields.std(axis=(0, 1))
            with (output_dir / "prepared" / "train_normal_stats.pkl").open(
                "rb"
            ) as handle:
                stats = pickle.load(handle)
            np.testing.assert_allclose(stats[::2], expected_mean, rtol=1e-7)
            np.testing.assert_allclose(stats[1::2], expected_std, rtol=1e-7)

            with self.assertRaises(PermissionError):
                CylinderFlowWindowDataset(
                    str(output_dir / "prepared" / "test_full_rollout64_sealed.json")
                )

            formal_sealed_path = (
                output_dir / "prepared" / "test_full_rollout64_sealed.json"
            )
            sealed = json.loads(formal_sealed_path.read_text(encoding="utf-8"))
            sealed["trajectory_count"] = 1
            sealed["windows"] = [
                {"trajectory_index": 0, "start": start} for start in (0, 268, 535)
            ]
            sealed["window_count"] = 3
            small_sealed_path = root / "small_test_sealed.json"
            small_sealed_path.write_text(json.dumps(sealed), encoding="utf-8")
            test_summary = prepare_test(
                metadata_path=metadata_path,
                test_tfrecord=root / "test.tfrecord",
                sealed_manifest_path=small_sealed_path,
                output_dir=root / "test_materialized",
                expected_count=1,
                examples=[_Example(make_arrays(40.0))],
            )
            self.assertTrue(test_summary["test_fields_accessed"])
            test_dataset = CylinderFlowWindowDataset(
                test_summary["manifest"], return_metadata=True
            )
            self.assertEqual(test_dataset.split, "test")
            self.assertEqual(len(test_dataset), 3)
            test_dataset.close()
            train.close()
            validation.close()

    def test_metadata_and_protobuf_decode_preserve_all_600_frames(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            metadata_path = Path(temporary_directory) / "meta.json"
            expected = make_arrays(7.0)
            write_metadata(metadata_path)
            metadata = load_metadata(metadata_path)
            decoded = decode_example(_Example(expected), metadata)
            self.assertEqual(decoded["velocity"].shape, (600, 4, 2))
            self.assertEqual(decoded["pressure"].shape, (600, 4, 1))
            for name in expected:
                np.testing.assert_array_equal(decoded[name], expected[name])


if __name__ == "__main__":
    unittest.main()
