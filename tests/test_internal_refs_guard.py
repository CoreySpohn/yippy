"""Tests for the internal-reference commit guard (tools/check_internal_refs.py)."""

import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).parents[1]
TOOL = REPO / "tools" / "check_internal_refs.py"


def run_guard(*paths):
    """Run the guard script on the given paths and return the process."""
    return subprocess.run(
        [sys.executable, str(TOOL), *map(str, paths)],
        capture_output=True,
        text=True,
    )


def test_clean_file_passes(tmp_path):
    """A file with ordinary content passes."""
    ok = tmp_path / "clean.py"
    ok.write_text('"""A module about pupils and focal planes."""\nx = 1\n')
    result = run_guard(ok)
    assert result.returncode == 0, result.stdout + result.stderr


def test_internal_reference_fails(tmp_path):
    """A research-workspace document reference is rejected and reported."""
    bad = tmp_path / "bad.py"
    content = "# see brain/SOME_DESIGN_DOC.md decisions\n"  # internal-ref-ok
    bad.write_text(content)
    result = run_guard(bad)
    assert result.returncode == 1
    assert "bad.py" in result.stdout
    assert "SOME_DESIGN_DOC" in result.stdout


def test_waiver_comment_allows(tmp_path):
    """A deliberately functional line passes with the waiver comment."""
    waived = tmp_path / "waived.py"
    waived.write_text('DATA = "hwo-mission-control/some/path"  # internal-ref-ok\n')
    result = run_guard(waived)
    assert result.returncode == 0, result.stdout + result.stderr


def test_tracked_repo_files_are_clean():
    """The guard passes over every tracked text file of this repository."""
    tracked = subprocess.run(
        ["git", "-C", str(REPO), "ls-files"],
        capture_output=True,
        text=True,
        check=True,
    ).stdout.splitlines()
    suffixes = {".py", ".md", ".rst", ".toml", ".yaml", ".yml", ".cff"}
    skip = {"CHANGELOG.md", "tools/check_internal_refs.py"}
    files = [REPO / f for f in tracked if Path(f).suffix in suffixes and f not in skip]
    assert files, "expected tracked text files"
    result = run_guard(*files)
    assert result.returncode == 0, result.stdout
