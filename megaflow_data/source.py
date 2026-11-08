"""Selective, resumable extraction from the pinned split ZIP release."""

from __future__ import annotations

import bisect
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
from pathlib import Path
import struct
import time
from urllib.request import Request, urlopen
import zlib

from .contract import FRAME_INDICES, REPOSITORY, SOURCE_REVISION, splits

PART_SIZES = (24159191040,) * 4 + (21101417193,)
CATALOG_OFFSET, CATALOG_BYTES, ENTRY_COUNT = 117067211047, 670970208, 5421619


def member_names() -> list[str]:
    ids = sorted(case for members in splits().values() for case in members)
    return [
        name
        for case in ids
        for name in (
            f"raw/mesh/has/ellipse_{case}.npz",
            *(
                f"raw/has_original/ellipse_{case}_has_{frame}.npz"
                for frame in FRAME_INDICES
            ),
        )
    ]


class Source:
    """Read existing extracted files, local split ZIPs, or exact HTTP ranges."""

    def __init__(self, raw_dir: Path, *, offline: bool = False):
        self.root, self.offline = raw_dir, offline
        self.cache = raw_dir / ".megaflow" / SOURCE_REVISION
        self.cache.mkdir(parents=True, exist_ok=True)
        self.starts, offset = [], 0
        for size in PART_SIZES:
            self.starts.append(offset)
            offset += size

    def destination(self, name: str) -> Path:
        return self.root / name

    def read_range(self, offset: int, count: int) -> bytes:
        chunks = []
        while count:
            index = bisect.bisect_right(self.starts, offset) - 1
            local = offset - self.starts[index]
            size = min(count, PART_SIZES[index] - local)
            filename = f"data.zip.{index + 1:03d}"
            source = self.root / filename
            if source.is_file():
                if source.stat().st_size != PART_SIZES[index]:
                    raise ValueError(f"incomplete release part: {source}")
                with source.open("rb") as stream:
                    stream.seek(local)
                    data = stream.read(size)
            else:
                if self.offline:
                    raise FileNotFoundError(f"offline archive part missing: {source}")
                url = (
                    f"https://huggingface.co/datasets/{REPOSITORY}/resolve/"
                    f"{SOURCE_REVISION}/{filename}"
                )
                for attempt in range(3):
                    try:
                        request = Request(
                            url, headers={"Range": f"bytes={local}-{local + size - 1}"}
                        )
                        with urlopen(request, timeout=90) as response:
                            wanted = (
                                f"bytes {local}-{local + size - 1}/{PART_SIZES[index]}"
                            )
                            if (
                                response.status != 206
                                or response.headers.get("Content-Range") != wanted
                            ):
                                raise ValueError(
                                    "server did not honor the requested archive range"
                                )
                            data = response.read(size + 1)
                        break
                    except Exception:
                        if attempt == 2:
                            raise
                        time.sleep(attempt + 1)
            if len(data) != size:
                raise ValueError("truncated archive range")
            chunks.append(data)
            offset, count = offset + size, count - size
        return b"".join(chunks)

    def catalog(self) -> dict[str, dict]:
        inventory = self.cache / "selected_members.json"
        wanted = set(member_names())
        if inventory.is_file():
            records = json.loads(inventory.read_text())
            if set(records) != wanted:
                raise ValueError("cached archive inventory belongs to another task")
            return records
        directory = self.cache / "central_directory.bin"
        offset = directory.stat().st_size if directory.exists() else 0
        if offset > CATALOG_BYTES:
            raise ValueError("oversized archive directory cache")
        with directory.open("ab") as stream:
            while offset < CATALOG_BYTES:
                size = min(16 * 1024 * 1024, CATALOG_BYTES - offset)
                stream.write(self.read_range(CATALOG_OFFSET + offset, size))
                stream.flush()
                offset += size
                print(f"Archive directory: {offset}/{CATALOG_BYTES} bytes", flush=True)
        records, entries = {}, 0
        with directory.open("rb") as stream:
            while stream.tell() < CATALOG_BYTES:
                header = struct.unpack("<4s6H3I5H2I", stream.read(46))
                if header[0] != b"PK\x01\x02":
                    raise ValueError("invalid ZIP central-directory signature")
                name = stream.read(header[10]).decode("utf-8")
                extra = stream.read(header[11])
                stream.seek(header[12], 1)
                entries += 1
                if name not in wanted:
                    continue
                usize, csize, position = header[9], header[8], header[16]
                cursor = 0
                while cursor + 4 <= len(extra):
                    tag, length = struct.unpack_from("<HH", extra, cursor)
                    cursor += 4
                    if tag == 1:
                        values = [usize, csize, position]
                        for field, value in enumerate(values):
                            if value == 0xFFFFFFFF:
                                values[field] = struct.unpack_from("<Q", extra, cursor)[
                                    0
                                ]
                                cursor += 8
                        usize, csize, position = values
                        break
                    cursor += length
                if name in records:
                    raise ValueError(f"duplicate source member: {name}")
                records[name] = {
                    "usize": usize,
                    "csize": csize,
                    "offset": position,
                    "method": header[4],
                }
        if entries != ENTRY_COUNT or set(records) != wanted:
            raise ValueError("pinned release lacks the exact selected Ellipse members")
        temporary = inventory.with_suffix(".partial.json")
        temporary.write_text(json.dumps(records), encoding="utf-8")
        temporary.replace(inventory)
        return records

    def extract(self, name: str, record: dict) -> None:
        destination = self.destination(name)
        if destination.is_file():
            if destination.stat().st_size != record["usize"]:
                raise ValueError(f"cached raw member has wrong length: {name}")
            return
        raw = self.read_range(record["offset"], record["csize"] + 1024)
        header = struct.unpack_from("<4s5H3I2H", raw)
        if (
            header[0] != b"PK\x03\x04"
            or raw[30 : 30 + header[9]].decode("utf-8") != name
        ):
            raise ValueError(f"ZIP local header differs from catalog: {name}")
        start = 30 + header[9] + header[10]
        payload = raw[start : start + record["csize"]]
        if record["method"] == 8:
            payload = zlib.decompress(payload, -15)
        elif record["method"] != 0:
            raise ValueError("unsupported ZIP compression method")
        if len(payload) != record["usize"]:
            raise ValueError(f"incomplete source member: {name}")
        destination.parent.mkdir(parents=True, exist_ok=True)
        temporary = destination.with_suffix(".partial.npz")
        temporary.write_bytes(payload)
        temporary.replace(destination)

    def ensure(self, workers: int = 8) -> None:
        names = member_names()
        if all(self.destination(name).is_file() for name in names):
            return  # NumPy schemas and finite values are checked by conversion.
        records = self.catalog()
        missing = [name for name in names if not self.destination(name).is_file()]
        with ThreadPoolExecutor(max_workers=workers) as pool:
            futures = {
                pool.submit(self.extract, name, records[name]): name for name in missing
            }
            failures = []
            for completed, future in enumerate(as_completed(futures), 1):
                try:
                    future.result()
                except Exception as error:
                    failures.append({"member": futures[future], "error": str(error)})
                    print(f"Source failure: {failures[-1]}", flush=True)
                if completed % 250 == 0 or completed == len(missing):
                    print(f"Selected members: {completed}/{len(missing)}", flush=True)
        if failures:
            (self.cache / "download_failures.json").write_text(
                json.dumps(failures, indent=2)
            )
            raise RuntimeError(
                f"{len(failures)} source members failed; successful files retained"
            )
