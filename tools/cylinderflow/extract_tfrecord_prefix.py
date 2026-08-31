from __future__ import annotations

import argparse
import json
import struct
from pathlib import Path
from typing import BinaryIO


def read_record(stream: BinaryIO, record_index: int) -> bytes | None:
    length_bytes = stream.read(8)
    if not length_bytes:
        return None
    if len(length_bytes) != 8:
        raise RuntimeError(f"truncated TFRecord length before record {record_index}")
    payload_length = struct.unpack("<Q", length_bytes)[0]
    length_crc = stream.read(4)
    payload = stream.read(payload_length)
    payload_crc = stream.read(4)
    if len(length_crc) != 4 or len(payload) != payload_length or len(payload_crc) != 4:
        raise RuntimeError(f"truncated TFRecord payload at record {record_index}")
    return length_bytes + length_crc + payload + payload_crc


def extract_records(
    source: Path, destination: Path, count: int, skip: int = 0
) -> dict[str, int | str]:
    if count <= 0 or skip < 0:
        raise ValueError("count must be positive and skip must be non-negative")
    if destination.exists():
        raise FileExistsError(destination)
    partial = destination.with_suffix(destination.suffix + ".partial")
    if partial.exists():
        raise FileExistsError(partial)
    bytes_written = 0
    with source.open("rb") as input_stream, partial.open("xb") as output_stream:
        record_index = 0
        copied = 0
        while copied < count:
            record = read_record(input_stream, record_index)
            if record is None:
                raise ValueError(
                    f"source ended after {record_index} records; cannot skip {skip} "
                    f"and copy {count}"
                )
            if record_index >= skip:
                output_stream.write(record)
                bytes_written += len(record)
                copied += 1
            record_index += 1
    partial.replace(destination)
    return {
        "source": str(source.resolve()),
        "destination": str(destination.resolve()),
        "skip": skip,
        "record_count": count,
        "bytes": bytes_written,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Copy a fixed complete-record range from a TFRecord prefix."
    )
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--count", type=int, required=True)
    parser.add_argument("--skip", type=int, default=0)
    args = parser.parse_args()
    print(
        json.dumps(
            extract_records(args.input, args.output, args.count, args.skip),
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
