"""Shared-directory exclusion using atomic mkdir and Python's standard library."""

from __future__ import annotations

import argparse
import atexit
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import signal
import socket
import subprocess
import sys
import time
import uuid


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


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lock", type=Path, required=True)
    parser.add_argument("--wait", action="store_true")
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    command = args.command
    if command[:1] == ["--"]:
        command = command[1:]
    if not command:
        parser.error("supply a command after --")
    with DirectoryLock(args.lock, wait=args.wait) as lock:
        process = None
        interrupted = []

        def forward(number: int, _frame: object) -> None:
            # Retain exclusion after a signal until the operator checks all workers.
            lock.preserved = True
            interrupted.append(number)
            if process is not None and process.poll() is None:
                try:
                    if os.name == "posix":
                        os.killpg(process.pid, number)
                    else:
                        process.send_signal(number)
                except ProcessLookupError:
                    pass

        previous = {
            number: signal.signal(number, forward)
            for number in (signal.SIGINT, signal.SIGTERM)
        }
        try:
            process = subprocess.Popen(command, start_new_session=(os.name == "posix"))
            if interrupted:
                forward(interrupted[-1], None)
            code = process.wait()
        finally:
            for number, handler in previous.items():
                signal.signal(number, handler)
        if interrupted:
            print(
                f"Inspect workers before releasing interrupted lock: {lock.directory}",
                file=sys.stderr,
                flush=True,
            )
        return 128 - code if code < 0 else code


if __name__ == "__main__":
    sys.exit(main())
