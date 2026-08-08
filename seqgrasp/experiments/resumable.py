from __future__ import annotations

from contextlib import contextmanager
import hashlib
import json
import os
from pathlib import Path
import time
from typing import Iterator


def _canonical_json(value: object) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def stable_trial_id(namespace: str, identity: object) -> str:
    """Return a deterministic ID from a namespace and complete trial identity."""

    digest = hashlib.sha256(_canonical_json([namespace, identity]).encode("utf-8")).hexdigest()
    return f"{namespace}:{digest}"


class IncrementalJsonlStore:
    """Crash-tolerant JSONL append store with cross-process lock-file exclusion.

    Each complete line is flushed and fsynced. A truncated final line from an
    interrupted write is ignored on restart; earlier complete records survive.
    """

    def __init__(self, path: str | Path, lock_timeout_seconds: float, lock_poll_seconds: float):
        self.path = Path(path)
        self.lock_path = self.path.with_suffix(self.path.suffix + ".lock")
        self.lock_timeout_seconds = lock_timeout_seconds
        self.lock_poll_seconds = lock_poll_seconds
        self.path.parent.mkdir(parents=True, exist_ok=True)
        descriptor = os.open(self.lock_path, os.O_CREAT | os.O_RDWR)
        try:
            if os.fstat(descriptor).st_size == 0:
                os.write(descriptor, b"0")
        finally:
            os.close(descriptor)

    @contextmanager
    def _lock(self) -> Iterator[None]:
        deadline = time.monotonic() + self.lock_timeout_seconds
        descriptor = os.open(self.lock_path, os.O_RDWR)
        while True:
            try:
                os.lseek(descriptor, 0, os.SEEK_SET)
                if os.name == "nt":
                    import msvcrt
                    msvcrt.locking(descriptor, msvcrt.LK_NBLCK, 1)
                else:
                    import fcntl
                    fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
                break
            except (BlockingIOError, OSError):
                if time.monotonic() >= deadline:
                    os.close(descriptor)
                    raise TimeoutError(f"timed out waiting for {self.lock_path}")
                time.sleep(self.lock_poll_seconds)
        try:
            yield
        finally:
            os.lseek(descriptor, 0, os.SEEK_SET)
            if os.name == "nt":
                import msvcrt
                msvcrt.locking(descriptor, msvcrt.LK_UNLCK, 1)
            else:
                import fcntl
                fcntl.flock(descriptor, fcntl.LOCK_UN)
            os.close(descriptor)

    def _discard_incomplete_tail(self) -> None:
        if not self.path.exists():
            return
        payload = self.path.read_bytes()
        if payload and not payload.endswith(b"\n"):
            last_complete = payload.rfind(b"\n") + 1
            with self.path.open("r+b") as stream:
                stream.truncate(last_complete)
                stream.flush()
                os.fsync(stream.fileno())

    def records(self) -> list[dict]:
        if not self.path.exists():
            return []
        lines = self.path.read_bytes().splitlines(keepends=True)
        records: list[dict] = []
        for index, line in enumerate(lines):
            if not line.endswith((b"\n", b"\r")):
                if index == len(lines) - 1:
                    break
                raise ValueError(f"corrupt non-final JSONL line {index + 1}")
            records.append(json.loads(line))
        return records

    def completed_ids(self) -> set[str]:
        return {str(record["trial_id"]) for record in self.records()}

    def append(self, record: dict) -> bool:
        if "trial_id" not in record:
            raise ValueError("incremental record must contain trial_id")
        with self._lock():
            self._discard_incomplete_tail()
            if str(record["trial_id"]) in self.completed_ids():
                return False
            payload = (_canonical_json(record) + "\n").encode("utf-8")
            with self.path.open("ab", buffering=0) as stream:
                stream.write(payload)
                stream.flush()
                os.fsync(stream.fileno())
            return True
