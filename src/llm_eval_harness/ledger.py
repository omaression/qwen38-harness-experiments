"""The ledger: a hash-chained, append-only record of which baseline is current, why, and how
often the held-out split has been looked at.

    Ledger(path).verify()                       problems with the chain or the current baselines
    Ledger(path).current(feature)               the cassette directory that is the baseline now
    Ledger(path).promote(feature, to, report)   make a gated candidate the baseline
    Ledger(path).rollback(feature, reason)      undo the promotion in force
    Ledger(path).record_evaluation(report)      note one gate run on a split
    Ledger(path).heldout_openings(split_sha256) how many distinct candidates that split has judged

Every method reads the file as it is when called, so a Ledger made with Ledger(path) and one made
with Ledger.open(path) behave the same; `rows` holds the rows of the last read.

The default file is ledger/ledger.jsonl, committed with the repository: promoting a new baseline
is a commit that appends a row, reviewed like any other change.

ROWS. One JSON object per line. Every row has
    "schema": "llm_eval_harness.ledger/1", "kind", "time" (UTC, util.utc_now),
    "prev": the sha256 of the previous line's bytes (without its newline; null on the first row),
    "id":   the first 16 hex digits of util.sha256_json(<the row without "id">).
  kind "header"      the first row: {"created": time}
  kind "promote"     {"feature", "to": {"path", "fingerprint"}, "from": {...} or null,
                      "gate": "initial" | "passed" | "overridden", "report": {"sha256",
                      "promote", "rule", "split", "split_sha256", "failed_gates"} or null,
                      "reason"}
  kind "rollback"    {"feature", "reverts": <id of the promotion undone>, "to": {...} or null
                      (null: the initial promotion was undone), "from": {...}, "reason"}
  kind "evaluation"  {"feature", "split", "split_sha256", "candidate": {"path", "fingerprint"},
                      "baseline": {...} or null, "promote", "report_sha256"}
Rows carry no user name, host name or absolute path: the ledger is published with the code.
A "path" is util.display_path of the directory, relative to the working directory the command
ran in, which is the repository root; a directory outside it is refused. "report.sha256" is the
sha256 of the report file's bytes when the caller passes it (the CLI does), else
util.sha256_json of the report.

THE CHAIN. A line edited, removed, inserted or reordered breaks `prev` or `id`; verify() lists
every such line, and every operation that writes refuses a damaged ledger (LedgerError: the file
is evidence and is never rewritten). Only a cut at the end goes unseen (the last lines removed
whole leave a valid chain): the committed history is what shows that. A line's bytes are hashed
without its line terminator ("\\n" or "\\r\\n"), so a checkout that converts line endings still
verifies. verify() also checks that the rows agree with each other (a promotion's "from" is the
baseline then in force, a rollback undoes the promotion in force), and re-fingerprints each
feature's current baseline directory: a baseline whose recorded responses were edited outside the
ledger is a problem too.

CURRENT. The rows of a feature form a stack: a promotion pushes its `to`, a rollback pops the
promotion it names. current(feature) is the `to` of the top promotion, or None.

PROMOTE checks, each a refusal (LedgerRefused) when it fails:
  - the report is of this feature, of the held-out split, and its promote is true (unless
    override is set; then a reason is required and the row says "gate": "overridden");
  - the report's candidate fingerprint equals cassettes.fingerprint(to): the directory holds
    what the gate saw;
  - the report's baseline fingerprint equals that of current(feature): the report compared the
    candidate with what it replaces, not with an older baseline (a stale report). A report
    without a baseline can promote only a feature that has none;
  - `to` is not already the current baseline;
  - the ledger verifies (a damaged chain is a LedgerError, a current baseline edited outside the
    ledger a refusal).
  promote(..., initial=True) starts a feature with no baseline yet: no report, a reason required,
  "gate": "initial".

ROLLBACK pops the promotion in force and makes its `from` current again; refused when the
feature has no promotion to undo, when no reason is given, or when the directory to go back to
no longer has the fingerprint it had. Undoing a feature's initial promotion leaves it with no
baseline. A rollback does not need the baseline it undoes to be intact: that is what it repairs.

HELD-OUT BUDGET. heldout_openings counts the distinct candidate fingerprints of evaluation rows
with that split_sha256. Every candidate judged on the held-out split is a choice made with it;
past gate.HELDOUT_BUDGET the report warns that the split is spent. CI reads the ledger but never
writes it; `llm-eval gate --record-evaluation` writes evaluation rows on a developer's machine.

Writes append one line (two on the first write: the header first) with a single os.write() on
an O_APPEND descriptor, then fsync; every check runs before it, so a refused write changes
nothing. The harness assumes one writer at a time.
"""

from __future__ import annotations

import json
import os
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping

from .cassettes import CassetteError, fingerprint
from .gate import REPORT_SCHEMA
from .records import HELDOUT
from .util import display_path, dumps_jsonl_row, sha256_bytes, sha256_json, utc_now

LEDGER_SCHEMA = "llm_eval_harness.ledger/1"
DEFAULT_LEDGER = "ledger/ledger.jsonl"
KINDS = ("header", "promote", "rollback", "evaluation")
#: The values of a promote row's "gate".
GATE_KINDS = ("initial", "passed", "overridden")
#: The keys every row has, and the further keys of each kind.
COMMON_KEYS = ("schema", "kind", "time", "prev", "id")
KIND_KEYS: Mapping[str, tuple[str, ...]] = {
    "header": ("created",),
    "promote": ("feature", "to", "from", "gate", "report", "reason"),
    "rollback": ("feature", "reverts", "to", "from", "reason"),
    "evaluation": ("feature", "split", "split_sha256", "candidate", "baseline", "promote", "report_sha256"),
}


class LedgerError(RuntimeError):
    """The ledger cannot be read or appended (I/O, bad JSON, a broken chain on write)."""


class LedgerRefused(LedgerError):
    """A promotion or rollback was refused; the message says why. Nothing was written."""


@dataclass(frozen=True)
class _Snapshot:
    """The file as one read saw it: each line without its terminator, its JSON object (None when
    it is not one), the sentences for lines that are not, and whether the file ends in a newline."""

    lines: tuple[bytes, ...]
    rows: tuple[dict[str, Any] | None, ...]
    unparsed: tuple[str, ...]
    ends_with_newline: bool


def _refuse_constant(name: str) -> Any:
    raise ValueError(f"{name} is not JSON")


def _parse_line(line: bytes) -> dict[str, Any] | None:
    try:
        value = json.loads(line.decode("utf-8"), parse_constant=_refuse_constant)
    except (UnicodeDecodeError, ValueError):
        return None
    return value if isinstance(value, dict) else None


def _is_ref(value: Any) -> bool:
    """A {"path", "fingerprint"} reference to a cassette directory."""
    return isinstance(value, dict) and isinstance(value.get("path"), str) and isinstance(value.get("fingerprint"), str)


def _ref_text(ref: Any) -> str:
    return "nothing" if ref is None else str(ref.get("path")) if isinstance(ref, dict) else repr(ref)


def _apply(stacks: dict[str, list[dict[str, Any]]], row: Mapping[str, Any]) -> list[str]:
    """Apply one row to the per-feature stacks of promotions in force; the sentences say where
    the row disagrees with the rows before it. The one place the stack semantics live."""
    kind, feature = row.get("kind"), row.get("feature")
    if kind not in ("promote", "rollback"):
        return []
    if not isinstance(feature, str) or not feature:
        return [f"a {kind} row without a feature"]
    stack = stacks.setdefault(feature, [])
    top = stack[-1]["to"] if stack else None
    problems = []
    if kind == "promote":
        if not _is_ref(row.get("to")):
            return [f"the promotion of {feature} has no valid 'to' ({{path, fingerprint}})"]
        if row.get("from") != top:
            problems.append(f"the promotion of {feature} says it replaced {_ref_text(row.get('from'))}, but the baseline then was {_ref_text(top)}")
        if row.get("gate") not in GATE_KINDS:
            problems.append(f"the promotion of {feature} has gate {row.get('gate')!r}, not one of {', '.join(GATE_KINDS)}")
        stack.append(dict(row))
        return problems
    reverts = row.get("reverts")
    if not stack or stack[-1].get("id") != reverts:
        problems.append(f"the rollback of {feature} names promotion {reverts!r}, which is not the one in force")
    for k in range(len(stack) - 1, -1, -1):
        if stack[k].get("id") == reverts:
            del stack[k:]
            break
    restored = stack[-1]["to"] if stack else None
    if row.get("to") != restored:
        problems.append(f"the rollback of {feature} says it restored {_ref_text(row.get('to'))}, but it restores {_ref_text(restored)}")
    return problems


def _stacks(rows: tuple[dict[str, Any] | None, ...] | list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    stacks: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        if row is not None:
            _apply(stacks, row)
    return stacks


def _row_problems(row: Mapping[str, Any], number: int, previous: bytes | None) -> list[str]:
    problems = []
    if row.get("schema") != LEDGER_SCHEMA:
        problems.append(f"its schema is {row.get('schema')!r}, not {LEDGER_SCHEMA!r}")
    kind = row.get("kind")
    if kind not in KINDS:
        problems.append(f"its kind {kind!r} is not one of {', '.join(KINDS)}")
    if number == 1 and kind != "header":
        problems.append("the first row is not the header (a line was removed or put before it)")
    if number > 1 and kind == "header":
        problems.append("a second header")
    expected_prev = None if previous is None else line_sha256(previous)
    if row.get("prev") != expected_prev:
        problems.append(
            "its prev is not null, as the first row's must be" if previous is None
            else "its prev is not the sha256 of the line before it (a line was edited, removed, inserted or moved)"
        )  # fmt: skip
    if row.get("id") != row_id(row):
        problems.append("its id does not match its content (the row was edited)")
    kind_keys = KIND_KEYS.get(kind, ()) if isinstance(kind, str) else ()
    missing = [key for key in COMMON_KEYS + tuple(kind_keys) if key not in row]
    if missing:
        problems.append(f"it lacks {', '.join(missing)}")
    return problems


def _chain_problems(snap: _Snapshot) -> list[str]:
    """Every problem of the file itself, in line order: lines that are not JSON objects, a broken
    chain, rows of the wrong shape, rows that disagree with the rows before them."""
    problems: list[str] = []
    stacks: dict[str, list[dict[str, Any]]] = {}
    previous: bytes | None = None
    for number, (line, row) in enumerate(zip(snap.lines, snap.rows), start=1):
        if row is None:
            problems.append(f"line {number}: not a JSON object")
        else:
            problems += [f"line {number}: {p}" for p in _row_problems(row, number, previous)]
            problems += [f"line {number}: {p}" for p in _apply(stacks, row)]
        previous = line
    if not snap.ends_with_newline:
        problems.append(f"line {len(snap.lines)}: no newline at its end (the file was cut short or edited)")
    return problems


def _ref_problem(ref: Mapping[str, Any]) -> str | None:
    """None when the directory `ref` names still has its recorded fingerprint; else what is
    wrong with it, as a phrase ("was changed outside the ledger (...)")."""
    try:
        now = fingerprint(Path(ref["path"]))
    except CassetteError:
        return "is no longer a cassette set (no manifest.json)"
    except OSError as exc:
        return f"cannot be read ({exc.strerror or exc})"
    if now != ref["fingerprint"]:
        return f"was changed outside the ledger (fingerprint {ref['fingerprint'][:12]} recorded, {now[:12]} now)"
    return None


def _fingerprint_problems(stacks: Mapping[str, list[dict[str, Any]]]) -> list[str]:
    problems = []
    for feature in sorted(stacks):
        if stacks[feature]:
            ref = stacks[feature][-1]["to"]
            problem = _ref_problem(ref)
            if problem:
                problems.append(f"{feature}: the current baseline {ref['path']} {problem}")
    return problems


def _safe_path(path: Any) -> Any:
    """A path from a report, never stored absolute."""
    return display_path(path) if isinstance(path, str) and os.path.isabs(path) else path


def _arm_ref(block: Mapping[str, Any]) -> dict[str, Any]:
    return {"path": _safe_path(block.get("source")), "fingerprint": block.get("fingerprint")}


def _failed_gates(report: Mapping[str, Any]) -> list[str]:
    gates = report.get("gates") or []
    return [str(g.get("gate")) for g in gates if isinstance(g, Mapping) and g.get("enforced", True) and g.get("passed") is not True]


@dataclass
class Ledger:
    """A ledger file. Reading a missing file gives an empty ledger; the first write creates it
    (and its directory) with a header row."""

    path: Path
    rows: list[dict[str, Any]] = field(default_factory=list, repr=False)

    def __post_init__(self) -> None:
        self.path = Path(self.path)

    # ---- reading ----------------------------------------------------------------------------

    def _read(self) -> _Snapshot:
        try:
            data = self.path.read_bytes()
        except FileNotFoundError:
            data = b""
        except OSError as exc:
            raise LedgerError(f"{display_path(self.path)} cannot be read: {exc.strerror or exc}") from None
        raw = data.split(b"\n") if data else []
        ends_with_newline = not raw or raw[-1] == b""
        if raw and raw[-1] == b"":
            raw.pop()
        lines = tuple(line[:-1] if line.endswith(b"\r") else line for line in raw)
        rows = tuple(_parse_line(line) for line in lines)
        unparsed = tuple(f"line {n}: not a JSON object" for n, row in enumerate(rows, start=1) if row is None)
        self.rows = [row for row in rows if row is not None]
        return _Snapshot(lines, rows, unparsed, ends_with_newline)

    def _parsed(self) -> _Snapshot:
        """The file, every line of which must be a JSON object (LedgerError otherwise)."""
        snap = self._read()
        if snap.unparsed:
            raise LedgerError(f"{display_path(self.path)}: {'; '.join(snap.unparsed)}")
        return snap

    def _writable(self) -> _Snapshot:
        """The file, which must have no problem of its own (LedgerError otherwise)."""
        snap = self._read()
        problems = _chain_problems(snap)
        if problems:
            more = f" (and {len(problems) - 1} more)" if len(problems) > 1 else ""
            raise LedgerError(f"{display_path(self.path)} is damaged, so nothing was written: {problems[0]}{more}; see `llm-eval ledger verify`")
        return snap

    @classmethod
    def open(cls, path: str | os.PathLike[str] = DEFAULT_LEDGER) -> "Ledger":
        """Read the file (if any) into rows. LedgerError on a line that is not a JSON object."""
        ledger = cls(Path(path))
        ledger._parsed()
        return ledger

    def verify(self, *, check_fingerprints: bool = True) -> list[str]:
        """Problems, one sentence each; empty when the chain holds and (with
        check_fingerprints) every current baseline still has its recorded fingerprint."""
        snap = self._read()
        problems = _chain_problems(snap)
        if check_fingerprints:
            problems += _fingerprint_problems(_stacks(snap.rows))
        return problems

    def current(self, feature: str) -> str | None:
        """The cassette directory (as recorded, relative to the repository root) that is the
        feature's baseline now, or None."""
        stack = _stacks(self._parsed().rows).get(feature) or []
        return stack[-1]["to"]["path"] if stack else None

    def heldout_openings(self, split_sha256: str) -> int:
        """Distinct candidate fingerprints evaluated on this split."""
        return len(self._openings(self._parsed()).get(split_sha256, set()))

    def has_candidate_opening(self, split_sha256: str, candidate_fingerprint: str) -> bool:
        """Whether `candidate_fingerprint` already has an evaluation row on this split."""
        return candidate_fingerprint in self._openings(self._parsed()).get(split_sha256, set())

    @staticmethod
    def _openings(snap: _Snapshot) -> dict[str, set[str]]:
        seen: dict[str, set[str]] = defaultdict(set)
        for row in snap.rows:
            if row is not None and row.get("kind") == "evaluation" and isinstance(row.get("split_sha256"), str):
                candidate = row.get("candidate")
                if _is_ref(candidate):
                    seen[row["split_sha256"]].add(candidate["fingerprint"])
        return seen

    def status(self) -> dict[str, Any]:
        """{"path", "rows", "problems": verify(), "current": {feature: path}, "openings":
        {split_sha256: n}} for `llm-eval ledger status`. Reports damage; never raises for it."""
        problems = self.verify()
        snap = self._read()
        stacks = _stacks(snap.rows)
        return {
            "path": display_path(self.path),
            "rows": len(snap.lines),
            "problems": problems,
            "current": {feature: stack[-1]["to"]["path"] for feature, stack in sorted(stacks.items()) if stack},
            "openings": {split: len(fps) for split, fps in sorted(self._openings(snap).items())},
        }

    # ---- writing ----------------------------------------------------------------------------

    def _append(self, snap: _Snapshot, body: Mapping[str, Any]) -> dict[str, Any]:
        """Write `body` as the next row (after a header when the file is empty), in one write."""
        now = utc_now()
        rows: list[dict[str, Any]] = []
        last = snap.lines[-1] if snap.lines else None
        if last is None:
            header = _seal({"schema": LEDGER_SCHEMA, "kind": "header", "time": now, "created": now, "prev": None})
            rows.append(header)
            last = _encode(header)
        row = _seal({**body, "schema": LEDGER_SCHEMA, "time": now, "prev": line_sha256(last)})
        rows.append(row)
        data = b"".join(_encode(r) + b"\n" for r in rows)
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            fd = os.open(self.path, os.O_WRONLY | os.O_APPEND | os.O_CREAT | getattr(os, "O_BINARY", 0), 0o644)
            try:
                written = os.write(fd, data)
                os.fsync(fd)
            finally:
                os.close(fd)
        except OSError as exc:
            raise LedgerError(f"{display_path(self.path)} cannot be written: {exc.strerror or exc}") from None
        if written != len(data):
            raise LedgerError(f"{display_path(self.path)}: only {written} of {len(data)} bytes were written; run `llm-eval ledger verify`")
        self._read()
        return row

    def _target(self, to: str | os.PathLike[str]) -> dict[str, str]:
        """The {"path", "fingerprint"} of a directory about to become a baseline."""
        path = display_path(to)
        if path.startswith("..."):
            raise LedgerRefused(f"{path} is outside the working directory: run from the repository root, so that the ledger records a relative path")
        try:
            return {"path": path, "fingerprint": fingerprint(to)}
        except CassetteError as exc:
            raise LedgerRefused(str(exc)) from None

    def promote(
        self,
        feature: str,
        to: str | os.PathLike[str],
        report: Mapping[str, Any] | None,
        *,
        reason: str,
        report_sha256: str | None = None,
        override: bool = False,
        initial: bool = False,
    ) -> dict[str, Any]:
        """Append a promote row after the checks above; returns the row."""
        snap = self._writable()
        stacks = _stacks(snap.rows)
        stack = stacks.get(feature) or []
        current = stack[-1]["to"] if stack else None
        target = self._target(to)
        reason = reason or ""
        if initial:
            if report is not None:
                raise LedgerRefused("an initial promotion takes no report; promote with the report instead")
            if override:
                raise LedgerRefused("an initial promotion has no gate to override")
            if not reason.strip():
                raise LedgerRefused(f"the initial promotion of {feature} needs a reason")
            if current is not None:
                raise LedgerRefused(f"{feature} already has a baseline ({current['path']}): promote with a gate report instead")
            gate, report_block = "initial", None
        else:
            if report is None:
                raise LedgerRefused(f"promoting {target['path']} needs the gate report that passed it (or initial for {feature}'s first baseline)")
            _check_report(feature, report, target, current)
            if report.get("promote") is True:
                gate = "passed"
            elif not override:
                failed = ", ".join(_failed_gates(report)) or "none listed"
                raise LedgerRefused(f"the report does not promote (failed gates: {failed}); override, with a reason, to promote anyway")
            elif not reason.strip():
                raise LedgerRefused("an override needs a reason")
            else:
                gate = "overridden"
            report_block = {
                "sha256": report_sha256 or sha256_json(report),
                "promote": report.get("promote"),
                "rule": report.get("rule"),
                "split": report.get("split"),
                "split_sha256": (report.get("suite") or {}).get("split_sha256"),
                "failed_gates": _failed_gates(report),
            }
        if current is not None and current == target:
            raise LedgerRefused(f"{target['path']} is already {feature}'s current baseline")
        problems = _fingerprint_problems(stacks)
        if problems:
            raise LedgerRefused("the ledger does not verify: " + "; ".join(problems))
        body = {"kind": "promote", "feature": feature, "to": target, "from": current, "gate": gate, "report": report_block, "reason": reason}
        return self._append(snap, body)

    def rollback(self, feature: str, *, reason: str) -> dict[str, Any]:
        """Append a rollback row; returns it."""
        snap = self._writable()
        if not (reason or "").strip():
            raise LedgerRefused(f"a rollback of {feature} needs a reason")
        stack = _stacks(snap.rows).get(feature) or []
        if not stack:
            raise LedgerRefused(f"{feature} has no promotion to undo")
        undone = stack[-1]
        restore = stack[-2]["to"] if len(stack) > 1 else None
        if restore is not None:
            problem = _ref_problem(restore)
            if problem:
                raise LedgerRefused(f"cannot roll {feature} back to {restore['path']}: it {problem}")
        body = {"kind": "rollback", "feature": feature, "reverts": undone.get("id"), "to": restore, "from": undone["to"], "reason": reason}
        return self._append(snap, body)

    def record_evaluation(self, report: Mapping[str, Any], *, report_sha256: str | None = None) -> dict[str, Any]:
        """Append an evaluation row for a gate report; returns it."""
        snap = self._writable()
        if report.get("schema") != REPORT_SCHEMA:
            raise LedgerRefused(f"not a gate report (schema {report.get('schema')!r})")
        candidate = report.get("candidate")
        split_sha256 = (report.get("suite") or {}).get("split_sha256")
        if not isinstance(candidate, Mapping) or not isinstance(candidate.get("fingerprint"), str):
            raise LedgerRefused("the report has no candidate fingerprint")
        if not isinstance(split_sha256, str):
            raise LedgerRefused("the report names no split_sha256")
        baseline = report.get("baseline")
        body = {
            "kind": "evaluation",
            "feature": report.get("feature"),
            "split": report.get("split"),
            "split_sha256": split_sha256,
            "candidate": _arm_ref(candidate),
            "baseline": _arm_ref(baseline) if isinstance(baseline, Mapping) else None,
            "promote": report.get("promote"),
            "report_sha256": report_sha256 or sha256_json(report),
        }
        return self._append(snap, body)


def _check_report(feature: str, report: Mapping[str, Any], target: Mapping[str, str], current: Mapping[str, Any] | None) -> None:
    """The refusals of PROMOTE that read the report."""
    if report.get("schema") != REPORT_SCHEMA:
        raise LedgerRefused(f"not a gate report (schema {report.get('schema')!r})")
    if report.get("feature") != feature:
        raise LedgerRefused(f"the report is of {report.get('feature')!r}, not {feature!r}")
    if report.get("split") != HELDOUT:
        raise LedgerRefused(f"the report is of the {report.get('split')!r} split; only a gate run on the {HELDOUT} split can promote")
    candidate = report.get("candidate") if isinstance(report.get("candidate"), Mapping) else {}
    if candidate.get("fingerprint") != target["fingerprint"]:
        raise LedgerRefused(
            f"{target['path']} does not hold what the gate saw: its fingerprint is {target['fingerprint'][:12]}, "
            f"the report's candidate {str(candidate.get('fingerprint'))[:12]}"
        )
    baseline = report.get("baseline")
    if baseline is not None and not isinstance(baseline, Mapping):
        raise LedgerRefused("the report's baseline is not an arm block")
    if baseline is None and current is None:
        return
    if baseline is None:
        raise LedgerRefused(f"the report has no baseline, but {feature}'s current baseline is {current['path']}: gate against --baseline @current")
    if current is None:
        raise LedgerRefused(
            f"the report compared the candidate with {baseline.get('source')}, but {feature} has no baseline in the ledger: "
            "promote its first baseline with --initial"
        )
    if baseline.get("fingerprint") != current["fingerprint"]:
        raise LedgerRefused(
            f"stale report: it compared the candidate with {baseline.get('source')} ({str(baseline.get('fingerprint'))[:12]}), "
            f"but {feature}'s current baseline is {current['path']} ({current['fingerprint'][:12]}); gate again against --baseline @current"
        )


def _seal(row: dict[str, Any]) -> dict[str, Any]:
    row["id"] = row_id(row)
    return row


def _encode(row: Mapping[str, Any]) -> bytes:
    return dumps_jsonl_row(row).encode("utf-8")


def row_id(row: Mapping[str, Any]) -> str:
    """The first 16 hex digits of util.sha256_json of the row without its "id" key."""
    return sha256_json({key: value for key, value in row.items() if key != "id"})[:16]


def line_sha256(line: bytes) -> str:
    """sha256 of one ledger line's bytes, without the trailing newline: the next row's `prev`.
    A "\\r\\n" terminator is removed as a whole."""
    if line.endswith(b"\n"):
        line = line[:-1]
    if line.endswith(b"\r"):
        line = line[:-1]
    return sha256_bytes(line)
