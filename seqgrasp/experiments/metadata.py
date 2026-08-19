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
    repository = Path(cwd).resolve()
    result = subprocess.run(
        ["git", "-c", f"safe.directory={repository}", "rev-parse", "HEAD"],
        cwd=repository,
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()
