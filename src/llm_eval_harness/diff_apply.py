"""Pure-stdlib unified diff parser and applier for multi-file text patches.

    apply_unified_diff(files, diff_text) -> (updated_files | None, status)

Supports standard `git diff` and unified diff headers (`--- a/path`, `+++ b/path`,
`/dev/null` for added/deleted files, empty creations/deletions, `\\ No newline at end of file`),
refuses path traversal (`..`), leading `/`, backslashes, `.git`, and forbidden modes
(symlinks / submodules), and applies hunks with exact context matching first, then unique
line-offset fuzz up to +/- 50 lines and stripped-trailing-whitespace fallback.
"""

from __future__ import annotations

import codecs
import re
from dataclasses import dataclass, field
from typing import Mapping

MAX_OFFSET_FUZZ = 50

_HUNK_RE = re.compile(r"^@@ -(\d+)(?:,(\d+))? \+(\d+)(?:,(\d+))? @@(.*)$")
_GIT_HEADER_RE = re.compile(r'^diff --git (?:"?a/(.+?)"? "?b/(.+?)"?)$')
_MODE_LINE_RE = re.compile(r"^(?:old mode|new mode|new file mode|deleted file mode|index [0-9a-fA-F]+\.\.[0-9a-fA-F]+) +([0-7]{6})\s*$")
_FORBIDDEN_MODES = frozenset({"120000", "160000"})


@dataclass
class _Hunk:
    old_start: int
    old_len: int
    new_start: int
    new_len: int
    #: (tag, text, no_eof_nl) where tag is " ", "-", or "+"
    items: list[tuple[str, str, bool]] = field(default_factory=list)
    seen_old: int = 0
    seen_new: int = 0

    def exhausted(self) -> bool:
        return self.seen_old >= self.old_len and self.seen_new >= self.new_len


@dataclass
class _FilePatch:
    old_path: str | None = None
    new_path: str | None = None
    has_old_header: bool = False
    has_new_header: bool = False
    new_file: bool = False
    deleted_file: bool = False
    rename_from: str | None = None
    rename_to: str | None = None
    hunks: list[_Hunk] = field(default_factory=list)
    binary: bool = False
    forbidden_mode: str | None = None


def _unquote(raw: str) -> str:
    raw = raw.rstrip("\r")
    if raw.startswith('"') and raw.endswith('"') and len(raw) >= 2:
        try:
            return codecs.escape_decode(raw[1:-1].encode("utf-8"))[0].decode("utf-8", errors="replace")
        except ValueError:
            return raw[1:-1]
    return raw.split("\t", 1)[0].rstrip()


def _parse_header_path(raw: str, prefix: str) -> tuple[str | None, str | None]:
    """Parse a --- or +++ path token, stripping `a/` or `b/` if present.

    Returns (clean_path_or_None_for_dev_null, error_or_None).
    """
    text = _unquote(raw)
    if text in ("/dev/null", "dev/null"):
        return None, None
    if text.startswith(prefix):
        text = text[len(prefix):]
    return _validate_relpath(text)


def _validate_relpath(path: str) -> tuple[str | None, str | None]:
    if not path:
        return None, "empty file path in diff header"
    if any(ord(ch) < 32 or ord(ch) == 127 for ch in path):
        return None, f"control characters in path {path!r}"
    if "\\" in path:
        return None, f"backslash not allowed in path {path!r}"
    if path.startswith("/"):
        return None, f"absolute path not allowed: {path!r}"
    parts = path.split("/")
    if any(part in ("", ".", "..") for part in parts):
        return None, f"invalid path segment in {path!r}"
    if ".git" in parts:
        return None, f"path may not touch .git: {path!r}"
    return path, None


def _is_header_at(lines: list[str], idx: int) -> bool:
    line = lines[idx]
    if line.startswith("diff --git "):
        return True
    return line.startswith("--- ") and idx + 1 < len(lines) and lines[idx + 1].startswith("+++ ")


def _hunk_continues(lines: list[str], idx: int) -> bool:
    while idx < len(lines) and lines[idx] == "":
        idx += 1
    if idx >= len(lines):
        return False
    return lines[idx][:1] in (" ", "+", "-", "\\") and not _is_header_at(lines, idx)


def _parse_diff(diff_text: str) -> tuple[list[_FilePatch] | None, str | None]:
    lines = diff_text.replace("\r\n", "\n").split("\n")
    patches: list[_FilePatch] = []
    cur: _FilePatch | None = None
    hunk: _Hunk | None = None
    i = 0
    while i < len(lines):
        line = lines[i]
        nxt = lines[i + 1] if i + 1 < len(lines) else ""

        if line.startswith("diff --git "):
            m = _GIT_HEADER_RE.match(line)
            cur = _FilePatch()
            if m:
                old_p, err1 = _validate_relpath(_unquote(m.group(1)))
                new_p, err2 = _validate_relpath(_unquote(m.group(2)))
                if err1:
                    return None, err1
                if err2:
                    return None, err2
                cur.old_path, cur.new_path = old_p, new_p
            patches.append(cur)
            hunk = None
            i += 1
            continue

        if (
            line.startswith("--- ")
            and nxt.startswith("+++ ")
            and (hunk is None or hunk.items or (i + 2 < len(lines) and bool(_HUNK_RE.match(lines[i + 2]))))
        ):
            old_p, err1 = _parse_header_path(line[4:], "a/")
            if err1:
                return None, err1
            new_p, err2 = _parse_header_path(nxt[4:], "b/")
            if err2:
                return None, err2
            if cur is None or cur.hunks or cur.has_old_header or cur.has_new_header:
                cur = _FilePatch()
                patches.append(cur)
            cur.old_path, cur.new_path = old_p, new_p
            cur.has_old_header, cur.has_new_header = True, True
            if old_p is None:
                cur.new_file = True
            if new_p is None:
                cur.deleted_file = True
            hunk = None
            i += 2
            continue

        m = _HUNK_RE.match(line)
        if m:
            if cur is None:
                return None, f"hunk without file header at line {i + 1}"
            old_len = int(m.group(2)) if m.group(2) is not None else 1
            new_len = int(m.group(4)) if m.group(4) is not None else 1
            hunk = _Hunk(int(m.group(1)), old_len, int(m.group(3)), new_len)
            cur.hunks.append(hunk)
            i += 1
            continue

        if hunk is not None:
            tag = line[:1]
            if tag in (" ", "-", "+"):
                hunk.items.append((tag, line[1:], False))
                if tag in (" ", "-"):
                    hunk.seen_old += 1
                if tag in (" ", "+"):
                    hunk.seen_new += 1
                i += 1
                continue
            if tag == "\\":
                if hunk.items:
                    prev_tag, prev_text, _ = hunk.items[-1]
                    hunk.items[-1] = (prev_tag, prev_text, True)
                i += 1
                continue
            if line == "":
                if _hunk_continues(lines, i + 1):
                    hunk.items.append((" ", "", False))
                    hunk.seen_old += 1
                    hunk.seen_new += 1
                    i += 1
                    continue
                if hunk.items or (hunk.old_len == 0 and hunk.new_len == 0):
                    hunk = None
                    i += 1
                    continue
                return None, f"incomplete hunk near diff line {i + 1}"
            if hunk.items:
                hunk = None
                continue  # re-process current line outside hunk (git apply --recount semantics)
            return None, f"malformed hunk line {i + 1}: {line[:60]!r}"

        if cur is not None:
            if line.startswith(("Binary files ", "GIT binary patch")):
                cur.binary = True
                return None, "binary patches are not supported"
            mode_m = _MODE_LINE_RE.match(line)
            if mode_m:
                mode = mode_m.group(1)
                if mode in _FORBIDDEN_MODES:
                    cur.forbidden_mode = mode
                    return None, f"unsupported file mode {mode}"
                if line.startswith("new file mode"):
                    cur.new_file = True
                    if not cur.has_old_header:
                        cur.old_path = None
                elif line.startswith("deleted file mode"):
                    cur.deleted_file = True
                    if not cur.has_new_header:
                        cur.new_path = None
            elif line.startswith("rename from "):
                p, err = _validate_relpath(_unquote(line[len("rename from "):]))
                if err:
                    return None, err
                cur.rename_from = p
                cur.old_path = p
            elif line.startswith("rename to "):
                p, err = _validate_relpath(_unquote(line[len("rename to "):]))
                if err:
                    return None, err
                cur.rename_to = p
                cur.new_path = p
            elif line == "" or line.startswith(("similarity index ", "dissimilarity index ", "copy from ", "copy to ")):
                pass
            else:
                return None, f"unexpected diff line {i + 1}: {line[:60]!r}"
        elif line.strip():
            return None, f"unexpected text before diff header: {line[:60]!r}"
        i += 1

    if not patches:
        return None, "no file patches found in diff"
    for fp in patches:
        if fp.old_path is None and fp.new_path is None:
            return None, "patch header has /dev/null on both sides"
        if not fp.hunks and not fp.new_file and not fp.deleted_file and fp.rename_from is None:
            target = fp.new_path or fp.old_path
            return None, f"patch for {target!r} has no hunks"
    return patches, None


def _split_lines(text: str) -> tuple[list[str], bool]:
    if text == "":
        return [], False
    if text.endswith("\n"):
        return text[:-1].split("\n"), True
    return text.split("\n"), False


def _find_hunk_pos(
    orig_lines: list[str],
    old_pat: list[str],
    expected: int,
    *,
    min_pos: int,
    max_fuzz: int,
) -> tuple[int | None, bool, str | None]:
    """Return (match_start, used_fuzz, error)."""
    m = len(old_pat)
    if m == 0:
        if expected < min_pos or expected > len(orig_lines):
            return None, False, f"hunk position {expected + 1} out of bounds"
        return expected, False, None

    lo = max(min_pos, expected - max_fuzz)
    hi = min(len(orig_lines) - m, expected + max_fuzz)
    if lo <= hi:
        if lo <= expected <= hi and orig_lines[expected : expected + m] == old_pat:
            return expected, False, None

        exact_matches = [p for p in range(lo, hi + 1) if orig_lines[p : p + m] == old_pat]
        if len(exact_matches) == 1:
            return exact_matches[0], True, None
        if len(exact_matches) > 1:
            return None, False, f"ambiguous hunk match near line {expected + 1}"

        pat_stripped = [line.rstrip() for line in old_pat]
        if lo <= expected <= hi and [line.rstrip() for line in orig_lines[expected : expected + m]] == pat_stripped:
            return expected, True, None

        fuzz_matches = [p for p in range(lo, hi + 1) if [line.rstrip() for line in orig_lines[p : p + m]] == pat_stripped]
        if len(fuzz_matches) == 1:
            return fuzz_matches[0], True, None
        if len(fuzz_matches) > 1:
            return None, False, f"ambiguous whitespace-fuzz hunk match near line {expected + 1}"

    # Fallback to unique whole-file match at or after min_pos when LLM line numbers drift > max_fuzz.
    if min_pos <= len(orig_lines) - m:
        wide_exact = [p for p in range(min_pos, len(orig_lines) - m + 1) if orig_lines[p : p + m] == old_pat]
        if len(wide_exact) == 1:
            return wide_exact[0], True, None
        pat_stripped = [line.rstrip() for line in old_pat]
        wide_fuzz = [p for p in range(min_pos, len(orig_lines) - m + 1) if [line.rstrip() for line in orig_lines[p : p + m]] == pat_stripped]
        if len(wide_fuzz) == 1:
            return wide_fuzz[0], True, None

    if lo > hi:
        return None, False, f"hunk at line {expected + 1} out of range (file has {len(orig_lines)} lines)"
    return None, False, f"hunk at line {expected + 1} does not match file content"


def _apply_file_hunks(orig_text: str, fp: _FilePatch) -> tuple[str | None, bool, str | None]:
    """Apply `fp.hunks` to `orig_text`. Returns (new_text_or_None_if_deleted, used_fuzz, error)."""
    orig_lines, orig_has_nl = _split_lines(orig_text)
    out_lines: list[str] = []
    cursor = 0
    used_fuzz = False
    last_from_plus = False
    last_no_eof_nl = False

    for hunk in fp.hunks:
        old_pat = [text for tag, text, _ in hunk.items if tag in (" ", "-")]
        expected = 0 if hunk.old_start == 0 else hunk.old_start - 1
        pos, hunk_fuzz, err = _find_hunk_pos(orig_lines, old_pat, expected, min_pos=cursor, max_fuzz=MAX_OFFSET_FUZZ)
        if pos is None:
            return None, False, err
        if hunk_fuzz:
            used_fuzz = True

        if pos > cursor:
            out_lines.extend(orig_lines[cursor:pos])
            last_from_plus = False
            last_no_eof_nl = False

        orig_idx = pos
        for tag, text, no_eof in hunk.items:
            if tag == " ":
                out_lines.append(orig_lines[orig_idx])
                orig_idx += 1
                last_from_plus = False
                last_no_eof_nl = no_eof
            elif tag == "-":
                orig_idx += 1
            else:  # "+"
                out_lines.append(text)
                last_from_plus = True
                last_no_eof_nl = no_eof

        cursor = pos + len(old_pat)

    if cursor < len(orig_lines):
        out_lines.extend(orig_lines[cursor:])
        last_from_plus = False
        last_no_eof_nl = False

    if fp.new_path is None:
        if out_lines:
            return None, False, f"deletion diff for {fp.old_path!r} does not remove all lines"
        return None, used_fuzz, None

    if not out_lines:
        return "", used_fuzz, None

    trailing_nl = (not last_no_eof_nl) if last_from_plus else (False if last_no_eof_nl else orig_has_nl)
    return "\n".join(out_lines) + ("\n" if trailing_nl else ""), used_fuzz, None


def apply_unified_diff(files: Mapping[str, str], diff_text: str) -> tuple[dict[str, str | None] | None, str]:
    """Apply unified `diff_text` to `files` in pure Python.

    Returns `(updated_files_dict, "exact" | "fuzz")` on success, where deleted files map to `None`
    and unchanged files retain their original contents, or `(None, reason)` on failure.
    """
    if not isinstance(diff_text, str) or not diff_text.strip():
        return None, "empty diff"
    patches, err = _parse_diff(diff_text)
    if patches is None:
        return None, err or "malformed diff"

    updated: dict[str, str | None] = dict(files)
    used_fuzz = False
    for fp in patches:
        if fp.old_path is None:
            # Adding a new file.
            assert fp.new_path is not None
            if updated.get(fp.new_path) is not None:
                return None, f"cannot create {fp.new_path!r}: file already exists"
            orig_text = ""
        else:
            if updated.get(fp.old_path) is None:
                return None, f"cannot patch {fp.old_path!r}: file not found"
            orig_text = updated[fp.old_path]  # type: ignore[assignment]
            if fp.new_path is not None and fp.new_path != fp.old_path and updated.get(fp.new_path) is not None:
                return None, f"cannot rename {fp.old_path!r} to existing {fp.new_path!r}"

        new_text, file_fuzz, file_err = _apply_file_hunks(orig_text, fp)
        if file_err is not None:
            return None, f"{fp.new_path or fp.old_path}: {file_err}"
        if file_fuzz:
            used_fuzz = True

        if fp.new_path is None:
            assert fp.old_path is not None
            updated[fp.old_path] = None
        else:
            if fp.old_path is not None and fp.old_path != fp.new_path:
                updated[fp.old_path] = None
            updated[fp.new_path] = new_text

    return updated, ("fuzz" if used_fuzz else "exact")
