#!/usr/bin/env python3
"""Commit guard: block internal research references in library files.

Library code is public and self-contained; pointers into the private research
workspace (project brain documents, decision-history vocabulary, working-note
paths) do not belong in it. This script scans the given files line by line
against a small pattern list and exits nonzero on any hit, so it can run as a
pre-commit hook (see .pre-commit-config.yaml) and as a pytest.

A line that must legitimately contain such a string (e.g. a development-only
data fallback path in test fixtures) can be waived by appending the comment
``internal-ref-ok`` to that line.

Usage: check_internal_refs.py FILE [FILE ...]
"""

import re
import sys
from pathlib import Path

WAIVER = "internal-ref-ok"

PATTERNS = [
    # Research-workspace paths and project homes.
    r"mission-control",
    r"research-brain",
    r"physicaloptix-setup",
    r"\bbrain/",
    r"deep-thoughts",
    # Underscored ALL-CAPS markdown names are internal working documents.
    r"\b[A-Z0-9]+(?:_[A-Z0-9]+)+\.md\b",
    # Project spike/script homes and decision-history vocabulary.
    r"eac1_dlux",
    r"Tier-G",
    r"(?i)\bgreenfield\b",
]

COMPILED = [(pattern, re.compile(pattern)) for pattern in PATTERNS]


def scan_file(path):
    """Return (lineno, pattern, line) hits for one file."""
    hits = []
    try:
        text = path.read_text(encoding="utf-8", errors="ignore")
    except OSError as err:
        print(f"{path}: unreadable ({err})")
        return [(0, "<unreadable>", "")]
    for lineno, line in enumerate(text.splitlines(), start=1):
        if WAIVER in line:
            continue
        for pattern, regex in COMPILED:
            if regex.search(line):
                hits.append((lineno, pattern, line.strip()))
    return hits


def main(argv):
    """Scan every argument path; report hits; exit 1 on any."""
    self_path = Path(__file__).resolve()
    failed = False
    for arg in argv:
        path = Path(arg)
        if path.resolve() == self_path:
            continue
        for lineno, pattern, line in scan_file(path):
            failed = True
            print(f"{path}:{lineno}: matches {pattern!r}: {line}")
    if failed:
        print(
            "\nInternal research references do not belong in library code. "
            f"Rephrase self-contained, or append '{WAIVER}' to a "
            "deliberately functional line."
        )
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
