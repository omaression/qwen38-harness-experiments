"""The verdicts on one reply: think-block validity, strict JSON schema validity, and the strict
execution verdict on a task's tests.

    split_reply(content, reasoning)          (thinking, answer, problems)
    check_think(content, reasoning, policy)  ThinkCheck
    check_schema(answer, schema, ...)        SchemaCheck
    exec_verdict(task, code)                 ExecVerdict under rule "values/1"

THINK. A reply's thinking is either returned separately by the server (ChatResponse.reasoning,
non-empty) or written at the start of `content` as <think>...</think>. split_reply reads it:
  - reasoning given: the thinking is `reasoning`, the answer is all of `content`; a <think> tag
    inside content as well is a problem ("thinking in both places");
  - else content (after leading whitespace) starts with <think>: the thinking runs to the first
    </think>, the answer is everything after it. No </think>: the thinking is not closed and the
    answer is "" (a reply cut off while thinking has no answer);
  - else content contains </think> with no opening tag (chat templates that put <think> in the
    prompt do this): everything before it is the thinking, opening implied; a <think> inside that
    text is a problem ("a <think> tag after the start of the reply");
  - else there is no thinking, and the answer is all of content.
  A second <think> or </think> after the first block is a problem ("more than one think block"),
  and so is a second <think> inside the first block.
check_think then: present (thinking found and not blank: an empty <think></think> is no
thinking), closed (present and an answer follows: the block was closed and something came after
it), words (count_words: whitespace-separated), in_range (policy.min_words <= words <=
policy.max_words), and
    valid = no problems and ((present and closed and in_range) or (not present and not policy.required))
A reply whose think tags are malformed is never valid, even when its thinking counts as absent.

SCHEMA. check_schema(answer, schema) is strict:
  - finish_reason "length" makes the reply invalid ("truncated at max_tokens"), whatever it holds;
  - answer.strip() must be exactly one JSON object: json.loads with no trailing text, duplicate
    keys refused (object_pairs_hook), NaN and Infinity refused (parse_constant);
  - every required key present, no key outside required and optional, each value of its type
    (bool is checked with `type(v) is bool`, never isinstance, since bool is an int and an int is
    not a bool), and the keys in schema.nonempty non-blank strings.
  lenient_json is a diagnostic only: it strips one ``` fence and, failing that, tries the span
  from the first "{" to the last "}". SchemaCheck.lenient_valid says whether that gives an object
  of the right shape. A reply that is only leniently valid is NOT schema-valid, and its code is
  not run; the lenient value is used only to SAST-scan code the model wrote (features.py).

EXECUTION (rule "values/1"). exec_verdict runs the task's shown tests, then its hidden tests, as
one sandbox.run_calls on `code`, and judges each test in the parent from the child's record:
  passed   the record exists, is the only one for its index, and either
             (expected) ok, the repr parses with ast.literal_eval, and equals the expected literal
                        (==; math.isclose with approx for numbers), or
             (raises)   not ok, and the expected class name is in the reported MRO names;
  failed   a record that does not satisfy the above (wrong value, wrong or no exception, a repr
           that is not a literal);
  error    an exception where a value was expected;
  missing  no record for the index (the process died, called os._exit, or never got there), or
           a duplicate index;
  timeout  no record and the deadline hit.
  A load error fails every test with its message (outcome "error"). The exit status is recorded
  and decides nothing: exit 0 is never a pass by itself. Zero tests is never a pass, and neither
  is an empty group: a task without shown tests, or without hidden tests, never passes.
    passed        = every test passed
    shown_passed  = every shown test passed; hidden_passed = every hidden test passed
  sandbox.infra_error is passed through as ExecVerdict.infra_error with passed False; callers
  must treat it as "no verdict", not as a failed patch. So is a platform without the sandbox
  (sandbox.SandboxUnavailable).
"""

from __future__ import annotations

import ast
import json
import math
import re
from dataclasses import dataclass, field
from typing import Any, Mapping

from . import sandbox
from .records import Check, Task
from .sandbox import DEFAULT_TIMEOUT_S, CallRecord

EXEC_RULE = "values/1"
THINK_OPEN, THINK_CLOSE = "<think>", "</think>"

BOTH_PLACES = "thinking in both places"
MORE_THAN_ONE = "more than one think block"
LATE_OPEN = "a <think> tag after the start of the reply"
#: Characters of a value or message quoted in a test's detail.
DETAIL_CHARS = 80


@dataclass(frozen=True)
class ThinkPolicy:
    """What a valid thinking block is for a feature. The demo features make it optional."""

    required: bool = False
    min_words: int = 1
    max_words: int = 1200


@dataclass
class ThinkCheck:
    """check_think's result. source: "reasoning" (a separate field), "content" (a <think> block)
    or "none"."""

    present: bool
    closed: bool
    words: int
    in_range: bool
    valid: bool
    source: str = "none"
    problems: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class JsonSchema:
    """A flat JSON object schema: required and optional keys with their Python types."""

    name: str
    required: Mapping[str, type]
    optional: Mapping[str, type] = field(default_factory=dict)
    nonempty: frozenset[str] = frozenset()


#: The bug-fix assistant's answer.
BUGFIX_SCHEMA = JsonSchema("bugfix/1", {"explanation": str, "patched_code": str}, {}, frozenset({"explanation", "patched_code"}))
#: The judge's answer.
JUDGE_SCHEMA = JsonSchema("judge/1", {"verdict": bool}, {"reason": str}, frozenset())


@dataclass
class SchemaCheck:
    """check_schema's result: the parsed object when valid, the problems when not, and the
    lenient diagnostic."""

    valid: bool
    value: dict[str, Any] | None
    problems: list[str] = field(default_factory=list)
    lenient_valid: bool = False
    lenient_value: dict[str, Any] | None = None


@dataclass
class CheckResult:
    """One test's outcome. kind: "shown" or "hidden"; outcome: "passed", "failed", "error",
    "missing" or "timeout"; detail: a short reason ("got 0, expected 3")."""

    index: int
    kind: str
    outcome: str
    detail: str = ""


@dataclass
class ExecVerdict:
    """exec_verdict's result (rule EXEC_RULE)."""

    passed: bool
    shown_passed: bool
    hidden_passed: bool
    results: list[CheckResult] = field(default_factory=list)
    problems: list[str] = field(default_factory=list)
    timed_out: bool = False
    duration_s: float = 0.0
    exit_code: int | None = None
    infra_error: str | None = None
    rule: str = EXEC_RULE

    def to_json(self) -> dict[str, Any]:
        """A JSON-ready dict for ItemResult.detail. The wall time is left out, so that scoring
        the same replies twice writes the same items file."""
        return {
            "rule": self.rule,
            "passed": self.passed,
            "shown_passed": self.shown_passed,
            "hidden_passed": self.hidden_passed,
            "timed_out": self.timed_out,
            "exit_code": self.exit_code,
            "infra_error": self.infra_error,
            "problems": list(self.problems),
            "results": [{"index": r.index, "kind": r.kind, "outcome": r.outcome, "detail": r.detail} for r in self.results],
        }


def count_words(text: str) -> int:
    """Whitespace-separated words: len(text.split())."""
    return len(text.split())


# ---- THINK ------------------------------------------------------------------------------------


def split_reply(content: str, reasoning: str | None) -> tuple[str | None, str, list[str]]:
    """(thinking text or None when there is none, answer text, problems). See THINK."""
    content = content or ""
    problems: list[str] = []
    if reasoning is not None and reasoning.strip():
        if THINK_OPEN in content or THINK_CLOSE in content:
            problems.append(BOTH_PLACES)
        return reasoning, content, problems
    stripped = content.lstrip()
    if stripped.startswith(THINK_OPEN):
        rest = stripped[len(THINK_OPEN):]
        end = rest.find(THINK_CLOSE)
        if end < 0:
            return rest, "", problems
        thinking, answer = rest[:end], rest[end + len(THINK_CLOSE):]
        if THINK_OPEN in thinking:
            problems.append(MORE_THAN_ONE)
    elif THINK_CLOSE in content:
        end = content.find(THINK_CLOSE)
        thinking, answer = content[:end], content[end + len(THINK_CLOSE):]
        if THINK_OPEN in thinking:
            problems.append(LATE_OPEN)
    else:
        return None, content, problems
    if (THINK_OPEN in answer or THINK_CLOSE in answer) and MORE_THAN_ONE not in problems:
        problems.append(MORE_THAN_ONE)
    return thinking, answer, problems


def check_think(content: str, reasoning: str | None, policy: ThinkPolicy = ThinkPolicy()) -> ThinkCheck:
    """See THINK."""
    thinking, answer, problems = split_reply(content, reasoning)
    present = thinking is not None and bool(thinking.strip())
    if not present:
        source = "none"
    elif reasoning is not None and reasoning.strip():
        source = "reasoning"
    else:
        source = "content"
    closed = present and bool(answer.strip())
    words = count_words(thinking) if present and thinking is not None else 0
    in_range = policy.min_words <= words <= policy.max_words
    valid = not problems and ((present and closed and in_range) or (not present and not policy.required))
    return ThinkCheck(present=present, closed=closed, words=words, in_range=in_range, valid=valid, source=source, problems=problems)


# ---- SCHEMA -----------------------------------------------------------------------------------


def _unique_pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for key, value in pairs:
        if key in out:
            raise ValueError(f"duplicate key {key!r}")
        out[key] = value
    return out


def _no_constant(name: str) -> Any:
    raise ValueError(f"{name} is not JSON")


def _strict_loads(text: str) -> Any:
    return json.loads(text, object_pairs_hook=_unique_pairs, parse_constant=_no_constant)


_JSON_KINDS: tuple[tuple[type, str], ...] = (
    (bool, "a boolean"), (int, "an integer"), (float, "a number"), (str, "a string"), (list, "an array"), (dict, "an object"),
)


def _kind(value: Any) -> str:
    """A value's JSON kind, as a problem message names it."""
    if value is None:
        return "null"
    for cls, name in _JSON_KINDS:
        if isinstance(value, cls):
            return name
    return type(value).__name__


def _type_name(want: type) -> str:
    return next((name for cls, name in _JSON_KINDS if cls is want), want.__name__)


def _is_type(value: Any, want: type) -> bool:
    if want is bool:
        return type(value) is bool
    if want is int:
        return type(value) is int
    if want is float:
        return type(value) in (int, float)
    return isinstance(value, want) and not isinstance(value, bool)


def _shape_problems(obj: Mapping[str, Any], schema: JsonSchema) -> list[str]:
    """What keeps a parsed object from matching `schema` (empty: it matches): missing and
    unexpected keys, values of the wrong type, blank strings where schema.nonempty says no."""
    problems = [f"missing key {key!r}" for key in schema.required if key not in obj]
    allowed = set(schema.required) | set(schema.optional)
    problems += [f"unexpected key {key!r}" for key in obj if key not in allowed]
    for key, want in (*schema.required.items(), *schema.optional.items()):
        if key not in obj:
            continue
        value = obj[key]
        if not _is_type(value, want):
            problems.append(f"{key!r} is {_kind(value)}, not {_type_name(want)}")
        elif key in schema.nonempty and isinstance(value, str) and not value.strip():
            problems.append(f"{key!r} is blank")
    return problems


def check_schema(answer: str, schema: JsonSchema, *, finish_reason: str | None = None) -> SchemaCheck:
    """See SCHEMA. `answer` is the text after the thinking (split_reply)."""
    problems: list[str] = []
    if finish_reason == "length":
        problems.append("truncated at max_tokens")
    text = (answer or "").strip()
    parsed: Any = None
    if not text:
        problems.append("no answer after the thinking")
    else:
        try:
            parsed = _strict_loads(text)
        except json.JSONDecodeError as exc:
            problems.append(f"not exactly one JSON object ({exc.msg} at character {exc.pos})")
        except RecursionError:
            problems.append("not exactly one JSON object (nested too deeply)")
        except ValueError as exc:
            problems.append(f"not strict JSON ({exc})")
        else:
            if isinstance(parsed, dict):
                problems += _shape_problems(parsed, schema)
            else:
                problems.append(f"{_kind(parsed)}, not a JSON object")
    lenient = lenient_json(answer or "")
    lenient_valid = lenient is not None and not _shape_problems(lenient, schema)
    valid = not problems
    return SchemaCheck(valid=valid, value=parsed if valid else None, problems=problems, lenient_valid=lenient_valid, lenient_value=lenient)


_FENCE = re.compile(r"```[^\n`]*\n(.*?)\n?[ \t]*```", re.DOTALL)


def lenient_json(answer: str) -> dict[str, Any] | None:
    """The diagnostic parse (see SCHEMA): an object or None. Tries the stripped text as it is,
    then the inside of the first ``` fence, then the span from the first "{" to the last "}".
    Duplicate keys are allowed here (the last one wins); NaN and Infinity are not."""
    text = (answer or "").strip()
    candidates = [text]
    fence = _FENCE.search(text)
    if fence:
        candidates.append(fence.group(1))
    first, last = text.find("{"), text.rfind("}")
    if 0 <= first < last:
        candidates.append(text[first : last + 1])
    for candidate in candidates:
        try:
            value = json.loads(candidate, parse_constant=_no_constant)
        except (ValueError, RecursionError):
            continue
        if isinstance(value, dict):
            return value
    return None


# ---- EXECUTION --------------------------------------------------------------------------------


def _quote(text: str | None) -> str:
    text = "" if text is None else " ".join(text.split())
    return text if len(text) <= DETAIL_CHARS else text[: DETAIL_CHARS - 3] + "..."


def _literal(text: str) -> tuple[bool, Any]:
    """(True, value) when `text` is a Python literal no longer than the sandbox's repr cap."""
    if len(text) > sandbox.MAX_REPR_CHARS:
        return False, None
    try:
        return True, ast.literal_eval(text)
    except Exception:  # SyntaxError, ValueError, MemoryError, RecursionError: not a literal
        return False, None


def _is_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _equal(got: Any, want: Any, approx: float | None) -> bool:
    try:
        if approx is not None and _is_number(got) and _is_number(want):
            return math.isclose(got, want, rel_tol=approx, abs_tol=approx)
        return bool(got == want)
    except Exception:  # an overflow, a negative tolerance: no match
        return False


def judge_record(record: CallRecord | None, check: Check, *, timed_out: bool = False) -> tuple[str, str]:
    """(outcome, detail) of one test from its record (None: no record). See EXECUTION.
    Never evaluates anything but ast.literal_eval on the record's repr."""
    if record is None:
        if timed_out:
            return "timeout", "no result before the deadline"
        return "missing", "no result: the process ended before this test reported"
    if check.raises is not None:
        if record.ok:
            return "failed", f"returned {_quote(record.repr)}, expected {check.raises}"
        if check.raises in record.mro:
            return "passed", ""
        return "failed", f"raised {record.exc}, expected {check.raises}"
    if check.expected is None:
        return "failed", "the test names neither an expected value nor an exception"
    if not record.ok:
        message = f": {_quote(record.msg)}" if record.msg else ""
        return "error", f"raised {record.exc}{message}"
    known, want = _literal(check.expected)
    if not known:
        return "failed", f"the expected value {_quote(check.expected)} is not a literal"
    got_text = record.repr if isinstance(record.repr, str) else ""
    parsed, got = _literal(got_text)
    if not parsed:
        return "failed", f"got {_quote(got_text)}, which is not a literal"
    if _equal(got, want, check.approx):
        return "passed", ""
    return "failed", f"got {_quote(got_text)}, expected {_quote(check.expected)}"


def exec_verdict(task: Task, code: str, *, timeout_s: float = DEFAULT_TIMEOUT_S, python: str | None = None) -> ExecVerdict:
    """Run every test of `task` (shown, then hidden) against `code` in the sandbox and judge
    them. See EXECUTION."""
    checks = list(task.all_tests)
    if not checks:
        return ExecVerdict(passed=False, shown_passed=False, hidden_passed=False, problems=["the task has no tests"])
    try:
        run = sandbox.run_calls(code, task.entry_point, [check.call for check in checks], timeout_s=timeout_s, python=python)
    except sandbox.SandboxUnavailable as exc:
        return ExecVerdict(passed=False, shown_passed=False, hidden_passed=False, problems=[f"sandbox: {exc}"], infra_error=str(exc))
    if run.infra_error is not None:
        return ExecVerdict(
            passed=False, shown_passed=False, hidden_passed=False, problems=[f"sandbox: {run.infra_error}"],
            timed_out=run.timed_out, duration_s=run.duration_s, exit_code=run.exit_code, infra_error=run.infra_error,
        )
    records = {record.index: record for record in run.records}
    duplicates = set(run.duplicates)
    results: list[CheckResult] = []
    for index, check in enumerate(checks):
        kind = "shown" if index < len(task.shown_tests) else "hidden"
        if run.load_error is not None:
            outcome, detail = "error", f"the code did not load: {_quote(run.load_error)}"
        elif index in duplicates:
            outcome, detail = "missing", "a result was reported more than once for this test"
        else:
            outcome, detail = judge_record(records.get(index), check, timed_out=run.timed_out)
        results.append(CheckResult(index=index, kind=kind, outcome=outcome, detail=detail))
    shown = [r for r in results if r.kind == "shown"]
    hidden = [r for r in results if r.kind == "hidden"]
    shown_passed = bool(shown) and all(r.outcome == "passed" for r in shown)
    hidden_passed = bool(hidden) and all(r.outcome == "passed" for r in hidden)
    return ExecVerdict(
        passed=shown_passed and hidden_passed,
        shown_passed=shown_passed,
        hidden_passed=hidden_passed,
        results=results,
        problems=_problems(run, results),
        timed_out=run.timed_out,
        duration_s=run.duration_s,
        exit_code=run.exit_code,
    )


def _problems(run: sandbox.SandboxResult, results: list[CheckResult]) -> list[str]:
    """The run-level reasons a reader of the item log needs, beyond the per-test details."""
    problems: list[str] = []
    if run.load_error is not None:
        problems.append(f"the code did not load: {_quote(run.load_error)}")
    if run.timed_out:
        problems.append("timed out: the process group was killed at the deadline")
    if run.duplicates:
        problems.append(f"results reported more than once for tests {run.duplicates}")
    if run.malformed:
        problems.append(f"{run.malformed} result line(s) were not well-formed")
    missing = [r.index for r in results if r.outcome == "missing" and r.index not in run.duplicates]
    if missing:
        problems.append(f"the process ended (exit status {run.exit_code}) before tests {missing} reported")
    problems += [f"sandbox: {note}" for note in run.notes]
    return problems
