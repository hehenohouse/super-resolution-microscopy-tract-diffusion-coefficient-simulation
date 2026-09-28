from __future__ import annotations

import hashlib
import importlib.metadata
import platform
import subprocess
import sys
from pathlib import Path
from typing import Any

from .config import Tau1Config

_PACKAGE_NAME = "tardis-tau1"


def fingerprint_file(path: str | Path, chunk_size: int = 1024 * 1024) -> dict[str, Any]:
    """Return a streaming SHA-256 fingerprint and byte size for one input file."""
    path = Path(path)
    digest = hashlib.sha256()
    size = 0
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
            size += len(chunk)
    return {"sha256": digest.hexdigest(), "file_bytes": size}


def collect_environment_provenance(config: Tau1Config) -> dict[str, Any]:
    """Collect reproducibility metadata without making analysis depend on Git."""
    try:
        package_version: str | None = importlib.metadata.version(_PACKAGE_NAME)
    except importlib.metadata.PackageNotFoundError:
        package_version = None

    git = _git_provenance(Path(__file__).resolve().parent)
    return {
        "config": config.to_dict(),
        "package": {"name": _PACKAGE_NAME, "version": package_version},
        "python": {
            "version": platform.python_version(),
            "implementation": platform.python_implementation(),
            "executable": sys.executable,
        },
        "git": git,
    }


def _git_provenance(cwd: Path) -> dict[str, Any]:
    try:
        commit = _run_git(["rev-parse", "HEAD"], cwd)
        status = _run_git(["status", "--porcelain", "--untracked-files=normal"], cwd)
    except (FileNotFoundError, subprocess.SubprocessError, OSError):
        return {"available": False, "commit": None, "dirty": None}
    if commit is None or status is None:
        return {"available": False, "commit": None, "dirty": None}
    return {"available": True, "commit": commit, "dirty": bool(status)}


def _run_git(arguments: list[str], cwd: Path) -> str | None:
    completed = subprocess.run(
        ["git", *arguments],
        cwd=cwd,
        capture_output=True,
        text=True,
        timeout=5,
        check=False,
    )
    if completed.returncode != 0:
        return None
    return completed.stdout.strip()
