"""Acquire and prepare the complete Airfoil Train/Validation data on Linux."""

from __future__ import annotations

import argparse
from contextlib import contextmanager, redirect_stderr, redirect_stdout
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
import tempfile
import time
import traceback
from urllib.request import Request, urlopen

import h5py

from .prepare import DATA_FILE, FORMAT, MANIFEST_FILE, RAW_INDICES, REVISION, prepare

SOURCE = "https://storage.googleapis.com/dm-meshgraphnets/airfoil"
RAW_FILES = ("meta.json", "train.tfrecord", "valid.tfrecord")
NORMALIZER = "text2pde_normalizer.pkl"


@contextmanager
def exclusive_lock(file: Path):
    from .portable_lock import DirectoryLock

    with DirectoryLock(file, wait=True):
        yield


class Tee:
    def __init__(self, *streams):
        self.streams = streams

    def write(self, text):
        for stream in self.streams:
            stream.write(text)
            stream.flush()
        return len(text)

    def flush(self):
        for stream in self.streams:
            stream.flush()


def write_status(directory: Path, **values) -> None:
    values["updated_utc"] = datetime.now(timezone.utc).isoformat()
    temporary = directory / "preparation_status.partial.json"
    temporary.write_text(json.dumps(values, indent=2) + "\n", encoding="utf-8")
    temporary.replace(directory / "preparation_status.json")


def download_file(directory: Path, name: str) -> None:
    destination = directory / name
    if destination.is_file() and destination.stat().st_size:
        return
    url = f"{SOURCE}/{name}"
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
                        while chunk := response.read(8 * 1024 * 1024):
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


def ready(directory: Path) -> bool:
    manifest_file = directory / MANIFEST_FILE
    if not manifest_file.is_file():
        return False
    manifest = json.loads(manifest_file.read_text(encoding="utf-8"))
    expected = {
        "format": FORMAT,
        "dataset_revision": REVISION,
        "dataset": DATA_FILE,
        "frames": 75,
        "raw_frames": 601,
        "temporal_stride": 8,
        "trajectory_count": 1100,
        "train_count": 1000,
        "validation_count": 100,
        "phase_offset": 0,
        "phase_augmentation": False,
        "test_accessed": False,
        "fields": ["u", "v", "p"],
        "density_used": False,
        "raw_frame_dt": 0.0002,
        "frame_dt": 0.0016,
        "source_frame_indices": RAW_INDICES.tolist(),
        "splits": {"train": list(range(1000)), "validation": list(range(1000, 1100))},
    }
    if any(manifest.get(key) != value for key, value in expected.items()):
        raise ValueError(
            "existing data do not match the complete Airfoil UVP stride-8 contract"
        )
    dataset_file = directory / DATA_FILE
    if dataset_file.stat().st_size != manifest.get("dataset_bytes"):
        raise ValueError("prepared HDF5 size differs from its manifest")
    if not (directory / NORMALIZER).is_file():
        raise FileNotFoundError(directory / NORMALIZER)
    with h5py.File(dataset_file, "r") as handle:
        if (
            handle.attrs.get("format") != FORMAT
            or handle.attrs.get("trajectory_count") != 1100
        ):
            raise ValueError("prepared HDF5 identity differs from its manifest")
        for index in range(1100):
            field = handle[f"trajectory_{index:04d}/uvp"]
            if field.shape[0] != 75 or field.shape[-1] != 3:
                raise ValueError(f"incomplete prepared trajectory {index}")
    return True


def ensure(raw_dir: Path, output_dir: Path, *, offline: bool = False) -> None:
    raw_dir, output_dir = raw_dir.resolve(), output_dir.resolve()
    with exclusive_lock(output_dir / ".airfoil-prepare.lock"):
        started = time.monotonic()
        with (output_dir / "preparation.log").open("a", encoding="utf-8") as log:
            with (
                redirect_stdout(Tee(sys.stdout, log)),
                redirect_stderr(Tee(sys.stderr, log)),
            ):
                try:
                    if ready(output_dir):
                        print(
                            f"Reusing complete Airfoil data: {output_dir}", flush=True
                        )
                        write_status(
                            output_dir,
                            state="complete",
                            reused=True,
                            dataset=str(output_dir / DATA_FILE),
                        )
                        return
                    if any(
                        (output_dir / name).exists() for name in (DATA_FILE, NORMALIZER)
                    ):
                        raise FileExistsError(
                            "incomplete published data; preserve it and select a new DATA_DIR"
                        )
                    write_status(
                        output_dir,
                        state="acquiring_raw_data",
                        source=SOURCE,
                        raw_dir=str(raw_dir),
                    )
                    missing = [
                        name
                        for name in RAW_FILES
                        if not (raw_dir / name).is_file()
                        or not (raw_dir / name).stat().st_size
                    ]
                    if missing and offline:
                        raise FileNotFoundError(
                            f"missing raw files in offline mode: {missing}"
                        )
                    if missing:
                        with exclusive_lock(raw_dir / ".airfoil-download.lock"):
                            for name in RAW_FILES:
                                download_file(raw_dir, name)
                    staging = Path(
                        tempfile.mkdtemp(prefix=".conversion-", dir=output_dir)
                    )
                    write_status(
                        output_dir,
                        state="converting",
                        staging=str(staging),
                        raw_dir=str(raw_dir),
                    )
                    prepare(raw_dir, staging)
                    if not ready(staging):
                        raise RuntimeError(
                            "converter did not finish all 1100 trajectories"
                        )
                    for name in (DATA_FILE, NORMALIZER, MANIFEST_FILE):
                        (staging / name).replace(output_dir / name)
                    staging.rmdir()
                    write_status(
                        output_dir,
                        state="complete",
                        reused=False,
                        dataset=str(output_dir / DATA_FILE),
                        elapsed_seconds=time.monotonic() - started,
                        train_count=1000,
                        validation_count=100,
                    )
                    print(f"Complete Airfoil data are ready: {output_dir}", flush=True)
                except Exception as error:
                    write_status(
                        output_dir,
                        state="failed",
                        error=str(error),
                        elapsed_seconds=time.monotonic() - started,
                    )
                    traceback.print_exc()
                    raise


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--offline", action="store_true", help="use local source files only"
    )
    args = parser.parse_args()
    ensure(args.raw_dir, args.output_dir, offline=args.offline)


if __name__ == "__main__":
    main()
