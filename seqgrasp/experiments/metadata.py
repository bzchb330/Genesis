from __future__ import annotations

import hashlib
from pathlib import Path
import subprocess


def config_hash(paths: list[str | Path]) -> str:
    digest = hashlib.sha256()
    for path in sorted(Path(item).resolve() for item in paths):
        digest.update(path.name.encode("utf-8"))
        digest.update(path.read_bytes())
    return digest.hexdigest()


def git_commit_sha(cwd: str | Path) -> str:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=Path(cwd),
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()
