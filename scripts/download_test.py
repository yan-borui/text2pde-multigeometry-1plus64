"""Download official MGN Test files without model weights or GPU dependencies."""

from __future__ import annotations

import argparse
import atexit
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import socket
import sys
import time
import uuid
from urllib.request import Request, urlopen

SOURCE = "https://storage.googleapis.com/dm-meshgraphnets/airfoil"
TEST_FILES = ("meta.json", "test.tfrecord")


class DirectoryLock:
    """Own a .d directory beside a logical lock file until explicitly closed."""

    def __init__(self, file_name: str | Path, *, wait: bool = False):
        self.directory = Path(str(file_name) + ".d")
        self.wait = wait
        self.owner = {
            "host": socket.gethostname(),
            "pid": os.getpid(),
            "started_utc": datetime.now(timezone.utc).isoformat(),
            "token": uuid.uuid4().hex,
        }
        self.acquired = False
        self.preserved = False

    def acquire(self) -> DirectoryLock:
        if self.acquired:
            raise RuntimeError("this lock object is already acquired")
        self.directory.parent.mkdir(parents=True, exist_ok=True)
        announced = False
        while True:
            try:
                self.directory.mkdir()
                break
            except FileExistsError as error:
                message = (
                    f"Lock occupied: {self.directory}. Inspect owner.json on its host; "
                    "retain the directory while its process or workers are alive."
                )
                if not self.wait:
                    raise RuntimeError(message) from error
                if not announced:
                    print(message + " Waiting...", file=sys.stderr, flush=True)
                    announced = True
                time.sleep(5)
        # Keep a partially initialized directory for inspection if writing fails.
        (self.directory / "owner.json").write_text(
            json.dumps(self.owner, indent=2) + "\n", encoding="utf-8"
        )
        self.acquired = True
        atexit.register(self.close)
        return self

    def close(self) -> None:
        if not self.acquired or self.preserved or os.getpid() != self.owner["pid"]:
            return
        owner_file = self.directory / "owner.json"
        try:
            recorded = json.loads(owner_file.read_text(encoding="utf-8"))
            if recorded != self.owner:
                raise RuntimeError("lock ownership changed")
            owner_file.unlink()
            self.directory.rmdir()
        except (OSError, ValueError, RuntimeError) as error:
            print(
                f"Lock cleanup requires inspection: {self.directory}: {error}",
                file=sys.stderr,
                flush=True,
            )
        finally:
            self.acquired = False
            atexit.unregister(self.close)

    def __enter__(self) -> DirectoryLock:
        return self.acquire()

    def __exit__(self, *_exc: object) -> None:
        self.close()


def download_file(directory: Path, name: str, *, source_url: str = SOURCE) -> None:
    """Resume an official source file and publish it after its byte count matches."""
    destination = directory / name
    if destination.is_file() and destination.stat().st_size:
        return
    url = f"{source_url}/{name}"
    temporary = directory / (name + ".download")
    for attempt in range(3):
        try:
            with urlopen(Request(url, method="HEAD"), timeout=60) as response:
                expected = int(response.headers["Content-Length"])
            offset = temporary.stat().st_size if temporary.exists() else 0
            if offset > expected:
                raise ValueError(f"oversized partial download: {temporary}")
            if offset < expected:
                headers = {"Range": f"bytes={offset}-"} if offset else {}
                with urlopen(Request(url, headers=headers), timeout=60) as response:
                    append = offset > 0 and response.status == 206
                    if append and not response.headers.get(
                        "Content-Range", ""
                    ).startswith(f"bytes {offset}-"):
                        raise ValueError(
                            "download response has an unexpected byte range"
                        )
                    received = offset if append else 0
                    next_report = received
                    with temporary.open("ab" if append else "wb") as output:
                        while True:
                            chunk = response.read(8 * 1024 * 1024)
                            if not chunk:
                                break
                            output.write(chunk)
                            received += len(chunk)
                            if received >= next_report:
                                print(
                                    f"Download {name}: {received}/{expected} bytes",
                                    flush=True,
                                )
                                next_report = received + 512 * 1024 * 1024
            if temporary.stat().st_size != expected:
                raise ValueError(f"incomplete download: {temporary}")
            temporary.replace(destination)
            return
        except Exception:
            if attempt == 2:
                raise
            print(f"Retrying {name} from retained partial download", flush=True)
            time.sleep(1)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset",
        choices=("airfoil", "cylinderflow", "cylinder_flow", "both"),
        default="both",
    )
    parser.add_argument("--raw-root", type=Path, required=True)
    args = parser.parse_args()
    if args.dataset == "both":
        datasets = ("airfoil", "cylinder_flow")
    else:
        datasets = (
            "cylinder_flow" if args.dataset == "cylinderflow" else args.dataset,
        )
    root = args.raw_root.expanduser().resolve()
    for dataset in datasets:
        directory = root / dataset
        directory.mkdir(parents=True, exist_ok=True)
        source = f"https://storage.googleapis.com/dm-meshgraphnets/{dataset}"
        with DirectoryLock(directory / ".download.lock", wait=True):
            for name in TEST_FILES:
                download_file(directory, name, source_url=source)
                print(f"Ready: {directory / name}", flush=True)


if __name__ == "__main__":
    main()
