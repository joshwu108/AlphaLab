"""Seeding, environment capture and run manifests.

A manifest is a JSON document written at the start of a run (status
``running``) and rewritten when it ends (``completed``, ``dry_run`` or
``failed``). It records everything needed to audit or reproduce the run:
configuration and its hash, dataset fingerprint, partitions, code revision,
library versions, budget usage, partition-access counts (proving whether the
test partition was read), outputs and, on failure, the exception and traceback.
"""

from __future__ import annotations

import hashlib
import importlib.metadata
import platform
import random
import subprocess
import sys
import traceback
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np

from alphalab import __version__
from alphalab.evaluation.export import write_json

MANIFEST_NAME = "manifest.json"
MANIFEST_VERSION = 1


def seed_everything(seed: int) -> None:
    """Seed Python's ``random``, NumPy's legacy global generator and torch (if installed).

    AlphaLab components use explicit generators derived from configuration
    seeds; this call additionally pins any third-party code that relies on
    global state.
    """
    random.seed(seed)
    np.random.seed(seed % 2**32)  # noqa: NPY002 - deliberately seeds legacy global state
    try:
        import torch
    except ImportError:
        return
    torch.manual_seed(seed)


def derive_seed(seed: int, label: str) -> int:
    """Deterministic 63-bit sub-seed for an independent stream named ``label``."""
    digest = hashlib.sha256(f"{seed}:{label}".encode()).digest()
    return int.from_bytes(digest[:8], "big") >> 1


def _package_version(name: str) -> str | None:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def code_revision(path: Path | None = None) -> dict[str, Any]:
    """Git commit and dirty flag of the source tree, or ``None`` values outside Git."""
    cwd = path or Path(__file__).resolve().parent
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=cwd, capture_output=True, text=True, check=True
        ).stdout.strip()
        dirty = bool(
            subprocess.run(
                ["git", "status", "--porcelain", "--untracked-files=no"],
                cwd=cwd,
                capture_output=True,
                text=True,
                check=True,
            ).stdout.strip()
        )
    except (OSError, subprocess.CalledProcessError):
        return {"commit": None, "dirty": None}
    return {"commit": commit, "dirty": dirty}


def environment_info() -> dict[str, Any]:
    """Interpreter, platform and dependency versions."""
    return {
        "alphalab": __version__,
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "packages": {
            name: _package_version(name) for name in ("numpy", "pandas", "pyarrow", "torch")
        },
        "code": code_revision(),
    }


def utc_now() -> str:
    """Current UTC time in ISO 8601."""
    return datetime.now(UTC).isoformat(timespec="seconds")


@dataclass
class RunManifest:
    """Mutable manifest; :meth:`write` persists it atomically enough for audit use."""

    run_dir: Path
    payload: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def start(cls, run_dir: Path, config: dict[str, Any], config_hash: str) -> RunManifest:
        """Create a ``running`` manifest and write it."""
        manifest = cls(
            run_dir,
            {
                "manifest_version": MANIFEST_VERSION,
                "status": "running",
                "config_hash": config_hash,
                "config": config,
                "started_at": utc_now(),
                "finished_at": None,
                "environment": environment_info(),
                "failure": None,
            },
        )
        manifest.write()
        return manifest

    def update(self, **values: Any) -> None:
        """Merge ``values`` into the manifest (in memory)."""
        self.payload.update(values)

    def finish(self, status: str) -> None:
        """Set the final status and timestamp and write."""
        self.payload["status"] = status
        self.payload["finished_at"] = utc_now()
        self.write()

    def fail(self, exc: BaseException) -> None:
        """Record an exception with its traceback and write a ``failed`` manifest."""
        self.payload["failure"] = {
            "type": type(exc).__name__,
            "message": str(exc),
            "traceback": "".join(traceback.format_exception(exc)),
        }
        self.finish("failed")

    def write(self) -> Path:
        """Write ``manifest.json`` in the run directory."""
        return write_json(self.payload, self.run_dir / MANIFEST_NAME)
