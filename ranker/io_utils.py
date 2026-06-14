"""I/O helpers: load candidates from .jsonl or .jsonl.gz transparently.

The README references ``candidates.jsonl.gz`` while the shipped bundle has the
uncompressed ``candidates.jsonl``. The Stage-3 reproduction harness may feed
either, so we sniff by magic bytes (gzip starts with 0x1f 0x8b) and fall back to
the file extension. No network, pure stdlib.
"""

from __future__ import annotations

import gzip
import io
import json
from pathlib import Path
from typing import Iterator


def _open_text(path: str | Path):
    """Open a path as a UTF-8 text stream, transparently handling gzip."""
    path = Path(path)
    with open(path, "rb") as fh:
        magic = fh.read(2)
    is_gzip = magic == b"\x1f\x8b" or path.suffix.lower() == ".gz"
    if is_gzip:
        return io.TextIOWrapper(gzip.open(path, "rb"), encoding="utf-8")
    return open(path, "r", encoding="utf-8")


def iter_candidates(path: str | Path) -> Iterator[dict]:
    """Yield candidate dicts from a JSONL(.gz) file, skipping blank lines."""
    with _open_text(path) as fh:
        for line in fh:
            line = line.strip()
            if line:
                yield json.loads(line)


def load_candidates(path: str | Path) -> list[dict]:
    """Load all candidate records into a list (order preserved)."""
    return list(iter_candidates(path))


def resolve_candidates_path(explicit: str | None = None) -> Path:
    """Find the candidates file. Prefer an explicit path, else search the cwd
    and this module's directory for jsonl / jsonl.gz variants."""
    if explicit:
        p = Path(explicit)
        if not p.exists():
            raise FileNotFoundError(f"candidates file not found: {p}")
        return p
    here = Path(__file__).resolve().parent.parent
    for name in ("candidates.jsonl", "candidates.jsonl.gz", "candidates.json"):
        for base in (Path.cwd(), here):
            cand = base / name
            if cand.exists():
                return cand
    raise FileNotFoundError(
        "Could not locate candidates.jsonl(.gz). Pass --candidates explicitly."
    )
