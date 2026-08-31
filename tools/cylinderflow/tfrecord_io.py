from __future__ import annotations

import json
import struct
from pathlib import Path
from typing import Any, Iterator

import numpy as np


def load_metadata(file_path: Path) -> dict[str, Any]:
    metadata = json.loads(file_path.read_text(encoding="utf-8"))
    if metadata.get("trajectory_length") != 600:
        raise ValueError("CylinderFlow metadata must declare trajectory_length=600")
    if not np.isclose(float(metadata.get("dt", np.nan)), 0.01):
        raise ValueError("CylinderFlow metadata must declare dt=0.01")
    required = {"velocity", "pressure", "mesh_pos", "node_type", "cells"}
    feature_names = set(metadata.get("features", ()))
    if not required.issubset(feature_names):
        raise ValueError(f"CylinderFlow metadata is missing {required - feature_names}")
    return metadata


def _read_varint(payload: bytes, offset: int) -> tuple[int, int]:
    value = 0
    shift = 0
    while offset < len(payload) and shift < 64:
        byte = payload[offset]
        offset += 1
        value |= (byte & 0x7F) << shift
        if byte < 0x80:
            return value, offset
        shift += 7
    raise ValueError("invalid or truncated protobuf varint")


def _protobuf_fields(payload: bytes) -> Iterator[tuple[int, int, bytes | int]]:
    offset = 0
    while offset < len(payload):
        tag, offset = _read_varint(payload, offset)
        field_number = tag >> 3
        wire_type = tag & 0x07
        if field_number <= 0:
            raise ValueError("protobuf field number must be positive")
        if wire_type == 0:
            value, offset = _read_varint(payload, offset)
        elif wire_type == 1:
            stop = offset + 8
            if stop > len(payload):
                raise ValueError("truncated protobuf fixed64 field")
            value, offset = payload[offset:stop], stop
        elif wire_type == 2:
            length, offset = _read_varint(payload, offset)
            stop = offset + length
            if stop > len(payload):
                raise ValueError("truncated protobuf length-delimited field")
            value, offset = payload[offset:stop], stop
        elif wire_type == 5:
            stop = offset + 4
            if stop > len(payload):
                raise ValueError("truncated protobuf fixed32 field")
            value, offset = payload[offset:stop], stop
        else:
            raise ValueError(f"unsupported protobuf wire type {wire_type}")
        yield field_number, wire_type, value


def parse_example_payload(payload: bytes) -> dict[str, list[bytes]]:
    """Parse the bytes-list subset of tf.train.Example used by MeshGraphNets."""

    features_messages = [
        value
        for field, wire, value in _protobuf_fields(payload)
        if field == 1 and wire == 2 and isinstance(value, bytes)
    ]
    if len(features_messages) != 1:
        raise ValueError("tf.train.Example must contain one Features message")
    features: dict[str, list[bytes]] = {}
    for field, wire, entry in _protobuf_fields(features_messages[0]):
        if field != 1 or wire != 2 or not isinstance(entry, bytes):
            continue
        key: str | None = None
        feature_message: bytes | None = None
        for entry_field, entry_wire, entry_value in _protobuf_fields(entry):
            if entry_field == 1 and entry_wire == 2 and isinstance(entry_value, bytes):
                key = entry_value.decode("utf-8")
            elif (
                entry_field == 2 and entry_wire == 2 and isinstance(entry_value, bytes)
            ):
                feature_message = entry_value
        if key is None or feature_message is None:
            raise ValueError("invalid tf.train.Features map entry")
        bytes_list_messages = [
            value
            for feature_field, feature_wire, value in _protobuf_fields(feature_message)
            if feature_field == 1 and feature_wire == 2 and isinstance(value, bytes)
        ]
        if len(bytes_list_messages) != 1:
            raise ValueError(f"feature {key} is not encoded as a BytesList")
        chunks = [
            value
            for list_field, list_wire, value in _protobuf_fields(bytes_list_messages[0])
            if list_field == 1 and list_wire == 2 and isinstance(value, bytes)
        ]
        if not chunks:
            raise ValueError(f"feature {key} has an empty BytesList")
        if key in features:
            raise ValueError(f"duplicate tf.train.Example feature {key}")
        features[key] = chunks
    return features


def iter_examples(file_path: Path) -> Iterator[dict[str, list[bytes]]]:
    """Yield bytes-list features from one uncompressed TFRecord file."""

    with file_path.open("rb") as stream:
        record_index = 0
        while True:
            length_bytes = stream.read(8)
            if not length_bytes:
                return
            if len(length_bytes) != 8:
                raise RuntimeError(
                    f"truncated TFRecord length before record {record_index} in "
                    f"{file_path}"
                )
            payload_length = struct.unpack("<Q", length_bytes)[0]
            length_crc = stream.read(4)
            payload = stream.read(payload_length)
            payload_crc = stream.read(4)
            if (
                len(length_crc) != 4
                or len(payload) != payload_length
                or len(payload_crc) != 4
            ):
                raise RuntimeError(
                    f"truncated TFRecord payload at record {record_index} in "
                    f"{file_path}"
                )
            yield parse_example_payload(payload)
            record_index += 1


def decode_example(example: Any, metadata: dict[str, Any]) -> dict[str, np.ndarray]:
    arrays: dict[str, np.ndarray] = {}
    for name, specification in metadata["features"].items():
        if isinstance(example, dict):
            if name not in example:
                raise KeyError(f"feature {name} is missing from tf.train.Example")
            payload = b"".join(example[name])
        else:
            feature = example.features.feature[name]
            payload = b"".join(feature.bytes_list.value)
        dtype = np.dtype(specification["dtype"])
        array = np.frombuffer(payload, dtype=dtype)
        try:
            arrays[name] = array.reshape(specification["shape"])
        except ValueError as exc:
            raise ValueError(
                f"feature {name} contains {array.size} values but metadata declares "
                f"shape {specification['shape']}"
            ) from exc
    return arrays


def static_frame(array: np.ndarray, name: str) -> np.ndarray:
    if array.shape[0] != 1:
        raise ValueError(f"static feature {name} must have leading dimension 1")
    return array[0]
