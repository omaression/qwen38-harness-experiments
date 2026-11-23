"""Small helpers every module shares: canonical JSON, hashing, JSONL files, paths for reports, logging.

This module is complete and stable. Request hashes (records.ChatRequest.sha256), suite and split
hashes, cassette fingerprints and ledger ids are all computed from canonical_json's exact bytes,
so changing canonical_json changes every recorded hash and breaks every cassette and ledger.

Logging goes to stderr, always: stdout belongs to the JSON (or markdown) a command prints, so that
`llm-eval gate ... | jq .promote` never parses a log line.
"""

from __future__ import annotations

import hashlib
import json
import os
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping


class JsonlError(ValueError):
    """A JSONL file holds a line that is not a JSON object. The message names the file and line."""


def canonical_json(obj: Any) -> str:
    """JSON with sorted keys and no insignificant whitespace; non-ASCII kept as is; NaN and
    infinities refused (they are not JSON). Equal data always gives the same string."""
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False)


def sha256_bytes(data: bytes) -> str:
    """Hex sha256 of bytes."""
    return hashlib.sha256(data).hexdigest()


def sha256_text(text: str) -> str:
    """Hex sha256 of a string's UTF-8 encoding."""
    return sha256_bytes(text.encode("utf-8"))


def sha256_json(obj: Any) -> str:
    """Hex sha256 of canonical_json(obj): the identity of a piece of data, independent of formatting."""
    return sha256_text(canonical_json(obj))


def sha256_file(path: str | os.PathLike[str]) -> str:
    """Hex sha256 of a file's bytes, read in chunks."""
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_jsonl(path: str | os.PathLike[str]) -> list[dict[str, Any]]:
    """Every non-blank line of a UTF-8 JSONL file as a dict, in file order.

    A line that does not parse, or parses to something other than an object, raises JsonlError
    with "<file>:<line>: <reason>" (the file as display_path shows it)."""
    rows: list[dict[str, Any]] = []
    with open(path, encoding="utf-8") as handle:
        for number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as exc:
                raise JsonlError(f"{display_path(path)}:{number}: not JSON ({exc.msg})") from None
            if not isinstance(value, dict):
                raise JsonlError(f"{display_path(path)}:{number}: a JSON {type(value).__name__}, not an object")
            rows.append(value)
    return rows


def dumps_jsonl_row(row: Mapping[str, Any]) -> str:
    """One JSONL line (sorted keys, non-ASCII kept, no trailing newline)."""
    return json.dumps(row, sort_keys=True, ensure_ascii=False, allow_nan=False)


def write_text_atomic(path: str | os.PathLike[str], text: str) -> None:
    """Write a UTF-8 text file all at once: a temporary file in the same directory, then
    os.replace, so a reader never sees half a file. Creates the parent directory."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=f".{target.name}.", suffix=".tmp", dir=target.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp, target)
    except BaseException:
        try:
            os.unlink(tmp)
        except FileNotFoundError:
            pass
        raise


def write_jsonl(path: str | os.PathLike[str], rows: Iterable[Mapping[str, Any]]) -> None:
    """Write rows as JSONL atomically (write_text_atomic), one dumps_jsonl_row line each."""
    write_text_atomic(path, "".join(dumps_jsonl_row(row) + "\n" for row in rows))


def display_path(path: str | os.PathLike[str]) -> str:
    """A path as reports and logs may show it: relative to the current directory (POSIX form) when
    it lies inside it, else ".../<last component>". Reports never carry an absolute path."""
    given = Path(path)
    try:
        relative = given.resolve().relative_to(Path.cwd().resolve())
    except (ValueError, OSError):
        return f".../{given.name}" if given.name else "..."
    text = relative.as_posix()
    return text if text else "."


def round6(value: float | None) -> float | None:
    """A float rounded to 6 decimals for reports (None stays None). p-values are not rounded."""
    return None if value is None else round(float(value), 6)


def utc_now() -> str:
    """The current UTC time as "YYYY-MM-DDTHH:MM:SSZ"."""
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def log(message: str) -> None:
    """One line to stderr, prefixed "llm-eval: "."""
    print(f"llm-eval: {message}", file=sys.stderr, flush=True)
