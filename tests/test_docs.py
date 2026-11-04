"""Public documentation must be valid reStructuredText and free of tracked Markdown."""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize("name", ["README.rst", "CONTRIBUTING.rst"])
def test_rst_documents_render_without_warnings(name: str) -> None:
    docutils_core = pytest.importorskip("docutils.core")
    source = (ROOT / name).read_text()
    docutils_core.publish_doctree(
        source, settings_overrides={"halt_level": 2, "report_level": 2}
    )  # raises on warnings or errors


def test_no_markdown_is_tracked() -> None:
    try:
        tracked = subprocess.run(
            ["git", "ls-files"], cwd=ROOT, capture_output=True, text=True, check=True
        ).stdout.splitlines()
    except (OSError, subprocess.CalledProcessError):
        pytest.skip("not a Git checkout")
    offenders = [p for p in tracked if p.lower().endswith((".md", ".markdown", ".mdx"))]
    offenders += [p for p in tracked if p.startswith(("planning/", "plans/"))]
    assert offenders == []
