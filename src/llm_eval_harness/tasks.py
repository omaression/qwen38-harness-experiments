"""The task suite: loading, structure checks, the dev / held-out split, and twin detection.

    load_suite(root)                  suite.json + tasks.jsonl -> records.Suite
    bucket(key) / split_of(task, pct) the sha256 bucket of a task's split key, and its split
    tasks_in_split(suite, split)      the tasks of DEV, HELDOUT or ALL, sorted by id
    split_sha256(suite, split)        the identity of what a split evaluates
    check_structure(suite)            problems found without running any code
    find_twins(suite)                 near-identical tasks on opposite sides of the split

FORMAT (docs/DESIGN.md, "Task suite"). A suite is a directory holding

    suite.json    {"schema": "llm_eval_harness.suite/1", "name": "demo", "version": 1,
                   "heldout_pct": 50, "synthetic": true, "description": "..."}
    tasks.jsonl   one task per line: {"id", "entry_point", "buggy_code", "shown_tests",
                   "hidden_tests", "reference_code", "wrong_code", "group"?, "tags"?, "synthetic"?}
                   where each test is {"call": "f(1)", "expected": "2"} or
                   {"call": "f('x')", "raises": "ValueError"}, with an optional "approx": 1e-9.

Unknown keys are refused, in suite.json and in task lines (a misspelt "hiden_tests" must not
pass silently as a task with no hidden tests). Errors name the file and line (SuiteError).
Duplicate keys inside one JSON object, and the non-JSON constants NaN and Infinity, are refused
as well: either would make what the file says differ from what is loaded.

THE SPLIT. A task's split key is normalize_key(task.group or task.id), and

    bucket(key) = int(sha256(key.encode("utf-8")).hexdigest()[:8], 16) % 100
    split       = HELDOUT if bucket < suite.heldout_pct else DEV

Why a stable key and not the task's text: hashing the text would move a task to the other side
of the split whenever someone fixes a typo in it, and a held-out task that wanders into dev has
leaked. The id never changes; a group puts tasks that must not be separated (twins: the same
problem reworded, or with a renamed function) in one bucket.

Why a hash and not a random draw: the assignment needs no stored state, every machine computes
the same one, and adding a task never moves another.

TWINS. A group only helps when the author knew the tasks were twins. find_twins compares every
DEV task with every HELDOUT task in a different group, on their buggy code with identifiers and
literals normalised (normalize_code), and reports pairs at or above the threshold. validation.py
reports them as warnings, and an exact normalised match as a problem.

This module never executes task code. Running the code (does the buggy code fail the shown test,
does the reference pass everything) is validation.py's job, through the sandbox.
"""

from __future__ import annotations

import ast
import builtins
import difflib
import io
import json
import keyword
import math
import os
import re
import tokenize
from pathlib import Path
from typing import Any, Iterator, Mapping

from .records import ALL, DEV, HELDOUT, Check, Suite, Task
from .util import display_path, sha256_json, sha256_text

SUITE_SCHEMA = "llm_eval_harness.suite/1"
#: The suite the CLI uses when --suite is not given (relative to the current directory).
DEFAULT_SUITE = "tasks/demo"
SUITE_FILE = "suite.json"
TASKS_FILE = "tasks.jsonl"
#: Keys a task line may carry; anything else is refused.
TASK_KEYS = frozenset({"id", "entry_point", "buggy_code", "shown_tests", "hidden_tests", "reference_code", "wrong_code", "group", "tags", "synthetic"})
#: Keys a test object may carry.
CHECK_KEYS = frozenset({"call", "expected", "raises", "approx"})
#: Keys suite.json may carry.
SUITE_KEYS = frozenset({"schema", "name", "version", "heldout_pct", "synthetic", "description"})
#: The fewest hidden tests a task may have: with one, "resolved" rests on a single assert.
MIN_HIDDEN_TESTS = 2
#: find_twins' default similarity threshold (difflib ratio of the normalised token streams).
TWIN_THRESHOLD = 0.85

_REQUIRED_TASK_KEYS = ("id", "entry_point", "buggy_code", "shown_tests", "hidden_tests", "reference_code", "wrong_code")
_REQUIRED_SUITE_KEYS = ("schema", "name", "version", "heldout_pct")
_CODE_FIELDS = ("buggy_code", "reference_code", "wrong_code")
_ID_PATTERN = re.compile(r"[a-z0-9][a-z0-9-]*")
_SPLITS = (DEV, HELDOUT, ALL)

# normalize_code: names it keeps as they are, and the token types it treats as layout.
_BUILTIN_NAMES = frozenset(dir(builtins))
_LAYOUT_TOKENS = frozenset({tokenize.NL, tokenize.INDENT, tokenize.DEDENT, tokenize.COMMENT, tokenize.ENCODING})
# Python 3.12 tokenizes an f-string as START, its parts, END (3.14 adds t-strings); earlier
# versions as one STRING token. Both become one "STR", so twins look the same on every version.
_STRING_START_TOKENS = frozenset(t for t in (getattr(tokenize, "FSTRING_START", None), getattr(tokenize, "TSTRING_START", None)) if t is not None)
_STRING_END_TOKENS = frozenset(t for t in (getattr(tokenize, "FSTRING_END", None), getattr(tokenize, "TSTRING_END", None)) if t is not None)


class SuiteError(ValueError):
    """The suite cannot be loaded: a missing file, bad JSON, a wrong key or type. The message
    names the file and, for tasks.jsonl, the line."""


# ---- loading ------------------------------------------------------------------------------------


def load_suite(root: str | os.PathLike[str]) -> Suite:
    """Load the suite in directory `root`.

    Reads suite.json (schema must be SUITE_SCHEMA; heldout_pct an int in 0..100) and tasks.jsonl
    (parse_task per line). Refuses duplicate task ids. Computes Suite.sha256 as records.Suite
    documents it. Suite.source is util.display_path(root). Raises SuiteError."""
    directory = Path(root)
    if not directory.is_dir():
        raise SuiteError(f"{display_path(directory)}: not a directory (a suite is a directory holding {SUITE_FILE} and {TASKS_FILE})")
    manifest = _read_manifest(directory / SUITE_FILE)
    loaded = _read_tasks(directory / TASKS_FILE)
    digest = sha256_json({"manifest": manifest, "tasks": [t.to_json() for t in sorted(loaded, key=lambda t: t.id)]})
    return Suite(
        name=manifest["name"],
        version=manifest["version"],
        heldout_pct=manifest["heldout_pct"],
        tasks=tuple(loaded),
        sha256=digest,
        source=display_path(directory),
        synthetic=manifest.get("synthetic", False),
        description=manifest.get("description", ""),
    )


def parse_task(obj: Mapping[str, Any], where: str) -> Task:
    """One tasks.jsonl object as a records.Task. `where` ("tasks.jsonl:7") prefixes errors.

    Checks types and keys only (TASK_KEYS; required: id, entry_point, buggy_code, shown_tests,
    hidden_tests, reference_code, wrong_code). Semantic checks are check_structure's."""
    if not isinstance(obj, Mapping):
        raise SuiteError(f"{where}: a task must be a JSON object, not {_json_type(obj)}")
    _refuse_unknown_keys(obj, TASK_KEYS, where, "task")
    _require_keys(obj, _REQUIRED_TASK_KEYS, where)
    for key in ("id", "entry_point", *_CODE_FIELDS):
        if not isinstance(obj[key], str):
            raise SuiteError(f"{where}: {key} must be a string, not {_json_type(obj[key])}")
    shown = _parse_checks(obj["shown_tests"], f"{where}: shown_tests")
    hidden = _parse_checks(obj["hidden_tests"], f"{where}: hidden_tests")
    group = obj.get("group")
    if group is not None and not isinstance(group, str):
        raise SuiteError(f"{where}: group must be a string or null, not {_json_type(group)}")
    tags = obj.get("tags", [])
    if not isinstance(tags, (list, tuple)) or not all(isinstance(tag, str) for tag in tags):
        raise SuiteError(f"{where}: tags must be a list of strings")
    synthetic = obj.get("synthetic", False)
    if not isinstance(synthetic, bool):
        raise SuiteError(f"{where}: synthetic must be true or false, not {_json_type(synthetic)}")
    return Task(
        id=obj["id"],
        entry_point=obj["entry_point"],
        buggy_code=obj["buggy_code"],
        shown_tests=shown,
        hidden_tests=hidden,
        reference_code=obj["reference_code"],
        wrong_code=obj["wrong_code"],
        group=group,
        tags=tuple(tags),
        synthetic=synthetic,
    )


def parse_check(obj: Mapping[str, Any], where: str) -> Check:
    """One test object as a records.Check (CHECK_KEYS; call required; types checked)."""
    if not isinstance(obj, Mapping):
        raise SuiteError(f"{where}: a test must be a JSON object, not {_json_type(obj)}")
    _refuse_unknown_keys(obj, CHECK_KEYS, where, "test")
    _require_keys(obj, ("call",), where)
    if not isinstance(obj["call"], str):
        raise SuiteError(f"{where}: call must be a string, not {_json_type(obj['call'])}")
    for key in ("expected", "raises"):
        if key in obj and not isinstance(obj[key], str):
            hint = ' (write "None" for a call that returns None)' if key == "expected" and obj[key] is None else ""
            raise SuiteError(f"{where}: {key} must be a string, not {_json_type(obj[key])}{hint}")
    approx = obj.get("approx")
    if "approx" in obj:
        if not _is_number(approx):
            raise SuiteError(f"{where}: approx must be a number, not {_json_type(approx)}")
        approx = float(approx)
    return Check(call=obj["call"], expected=obj.get("expected"), raises=obj.get("raises"), approx=approx)


def _read_manifest(path: Path) -> dict[str, Any]:
    where = display_path(path)
    manifest = _parse_json(_read_text(path, where), where, multiline=True)
    if not isinstance(manifest, dict):
        raise SuiteError(f"{where}: {_json_type(manifest)}, not a JSON object")
    _refuse_unknown_keys(manifest, SUITE_KEYS, where, "suite manifest")
    _require_keys(manifest, _REQUIRED_SUITE_KEYS, where)
    if manifest["schema"] != SUITE_SCHEMA:
        raise SuiteError(f"{where}: schema is {manifest['schema']!r}, expected {SUITE_SCHEMA!r}")
    if not isinstance(manifest["name"], str) or not manifest["name"].strip():
        raise SuiteError(f"{where}: name must be a non-empty string")
    if not _is_int(manifest["version"]):
        raise SuiteError(f"{where}: version must be an integer, not {_json_type(manifest['version'])}")
    pct = manifest["heldout_pct"]
    if not _is_int(pct) or not 0 <= pct <= 100:
        raise SuiteError(f"{where}: heldout_pct must be an integer from 0 to 100, not {pct!r}")
    if not isinstance(manifest.get("synthetic", False), bool):
        raise SuiteError(f"{where}: synthetic must be true or false")
    if not isinstance(manifest.get("description", ""), str):
        raise SuiteError(f"{where}: description must be a string")
    return manifest


def _read_tasks(path: Path) -> list[Task]:
    shown = display_path(path)
    text = _read_text(path, shown)
    tasks: list[Task] = []
    first_line: dict[str, int] = {}
    # Split on "\n" only: str.splitlines would also break at U+2028 and U+2029, which may stand
    # unescaped inside a JSON string.
    for number, line in enumerate(text.split("\n"), start=1):
        if not line.strip():
            continue
        where = f"{shown}:{number}"
        obj = _parse_json(line, where, multiline=False)
        if not isinstance(obj, dict):
            raise SuiteError(f"{where}: {_json_type(obj)}, not a JSON object")
        task = parse_task(obj, where)
        if task.id in first_line:
            raise SuiteError(f"{where}: duplicate task id {task.id!r} (first on line {first_line[task.id]})")
        first_line[task.id] = number
        tasks.append(task)
    if not tasks:
        raise SuiteError(f"{shown}: holds no tasks")
    return tasks


def _read_text(path: Path, where: str) -> str:
    try:
        return path.read_text(encoding="utf-8")
    except FileNotFoundError:
        raise SuiteError(f"{where}: no such file") from None
    except UnicodeDecodeError as exc:
        raise SuiteError(f"{where}: not UTF-8 text ({exc.reason})") from None
    except OSError as exc:
        raise SuiteError(f"{where}: cannot be read ({exc.strerror or type(exc).__name__})") from None


class _DuplicateKey(ValueError):
    pass


class _NotJsonConstant(ValueError):
    pass


def _object_without_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for key, value in pairs:
        if key in out:
            raise _DuplicateKey(key)
        out[key] = value
    return out


def _refuse_constant(name: str) -> Any:
    raise _NotJsonConstant(name)


def _parse_json(text: str, where: str, *, multiline: bool) -> Any:
    try:
        return json.loads(text, object_pairs_hook=_object_without_duplicates, parse_constant=_refuse_constant)
    except _DuplicateKey as exc:
        raise SuiteError(f"{where}: duplicate key {exc.args[0]!r} in one object") from None
    except _NotJsonConstant as exc:
        raise SuiteError(f"{where}: {exc.args[0]} is not JSON") from None
    except json.JSONDecodeError as exc:
        at = f"{where}:{exc.lineno}" if multiline else where
        raise SuiteError(f"{at}: not JSON ({exc.msg}, column {exc.colno})") from None
    except RecursionError:
        raise SuiteError(f"{where}: nested too deeply") from None


def _refuse_unknown_keys(obj: Mapping[str, Any], allowed: frozenset[str], where: str, what: str) -> None:
    unknown = sorted(str(key) for key in obj if key not in allowed)
    if not unknown:
        return
    names = ", ".join(repr(key) for key in unknown)
    close = difflib.get_close_matches(unknown[0], sorted(allowed), n=1)
    hint = f" (did you mean {close[0]!r}?)" if close else ""
    noun = "key" if len(unknown) == 1 else "keys"
    raise SuiteError(f"{where}: unknown {noun} {names} in a {what}{hint}; allowed: {', '.join(sorted(allowed))}")


def _require_keys(obj: Mapping[str, Any], required: tuple[str, ...], where: str) -> None:
    missing = [key for key in required if key not in obj]
    if missing:
        noun = "key" if len(missing) == 1 else "keys"
        raise SuiteError(f"{where}: missing {noun} {', '.join(repr(key) for key in missing)}")


def _parse_checks(value: Any, where: str) -> tuple[Check, ...]:
    if not isinstance(value, (list, tuple)):
        raise SuiteError(f"{where} must be a list of test objects, not {_json_type(value)}")
    return tuple(parse_check(item, f"{where}[{index}]") for index, item in enumerate(value))


def _json_type(value: Any) -> str:
    """The JSON name of a value's type, for error messages."""
    if value is None:
        return "null"
    if isinstance(value, bool):
        return "a boolean"
    if isinstance(value, (int, float)):
        return "a number"
    if isinstance(value, str):
        return "a string"
    if isinstance(value, (list, tuple)):
        return "an array"
    if isinstance(value, Mapping):
        return "an object"
    return type(value).__name__


def _is_int(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _is_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


# ---- the split ----------------------------------------------------------------------------------


def normalize_key(key: str) -> str:
    """A split key in canonical form: stripped, lower-cased, internal whitespace collapsed to one
    space. "Clamp  Range " and "clamp range" are the same key."""
    return " ".join(key.split()).lower()


def bucket(key: str) -> int:
    """The sha256 bucket of a split key, 0..99: int(sha256(normalize_key(key))[:8], 16) % 100."""
    return int(sha256_text(normalize_key(key))[:8], 16) % 100


def split_of(task: Task, heldout_pct: int) -> str:
    """HELDOUT when bucket(task.group or task.id) < heldout_pct, else DEV."""
    return HELDOUT if bucket(task.group or task.id) < heldout_pct else DEV


def tasks_in_split(suite: Suite, split: str) -> tuple[Task, ...]:
    """The suite's tasks in `split` (DEV, HELDOUT or ALL), sorted by id. ValueError for any
    other split name."""
    if split not in _SPLITS:
        raise ValueError(f"unknown split {split!r}: expected one of {', '.join(_SPLITS)}")
    chosen = suite.tasks if split == ALL else [t for t in suite.tasks if split_of(t, suite.heldout_pct) == split]
    return tuple(sorted(chosen, key=lambda t: t.id))


def split_sha256(suite: Suite, split: str) -> str:
    """util.sha256_json({"suite": suite.name, "split": split, "tasks": [ids of tasks_in_split]}):
    which tasks a split holds, independent of their text. The ledger counts held-out evaluations
    by it, so fixing a typo in a task does not reset the held-out budget; adding or removing a
    task starts a new count (Suite.sha256 is the content's identity, and the report has both)."""
    return sha256_json({"suite": suite.name, "split": split, "tasks": [t.id for t in tasks_in_split(suite, split)]})


def split_counts(suite: Suite) -> dict[str, int]:
    """{"dev": n, "heldout": n, "all": n}."""
    return {DEV: len(tasks_in_split(suite, DEV)), HELDOUT: len(tasks_in_split(suite, HELDOUT)), ALL: len(suite.tasks)}


# ---- structure ----------------------------------------------------------------------------------


def check_structure(suite: Suite) -> list[str]:
    """Problems found without executing anything, one sentence each, naming the task.

    - ids match [a-z0-9][a-z0-9-]* and are unique; groups, when given, are non-empty;
    - entry_point is an identifier, and buggy_code, reference_code and wrong_code each parse
      (ast.parse) and define a top-level function of that name;
    - at least one shown test and at least MIN_HIDDEN_TESTS hidden tests;
    - every test: `call` parses as one expression (ast.parse(mode="eval")); exactly one of
      expected / raises; `expected` is a literal (ast.literal_eval); `raises` is an identifier;
      `approx`, when given, is a positive number and `expected` a number;
    - no two tests of a task are identical; no hidden test's call text appears in buggy_code
      (a docstring example equal to a hidden test makes that test shown, not hidden);
    - reference_code, wrong_code and buggy_code all differ.
    An empty list means the structure is fine.

    Two tests are identical when their calls are the same expression (compared as syntax
    trees, so spacing does not hide a repeat): a call gives one value, so a repeated call adds
    nothing, or contradicts itself. A hidden call counts as appearing in buggy_code also when it
    appears there with other spacing between its tokens, as in a docstring example
    "clamp(-2,0,3)" (_call_appears_in). Two code fields that differ only in layout and
    comments do not differ."""
    problems: list[str] = []
    seen: set[str] = set()
    reported: set[str] = set()
    for task in suite.tasks:
        key = task.id if isinstance(task.id, str) else repr(task.id)
        if key in seen and key not in reported:
            problems.append(f"task {task.id!r}: more than one task has this id")
            reported.add(key)
        seen.add(key)
        problems.extend(_task_problems(task))
    return problems


def _task_problems(task: Task) -> list[str]:
    name = f"task {task.id!r}"
    out: list[str] = []
    if not isinstance(task.id, str) or not _ID_PATTERN.fullmatch(task.id):
        out.append(f"{name}: the id does not match [a-z0-9][a-z0-9-]*")
    if task.group is not None and (not isinstance(task.group, str) or not normalize_key(task.group)):
        out.append(f"{name}: the group is empty (leave it out to split on the id)")

    entry = task.entry_point
    entry_ok = isinstance(entry, str) and entry.isidentifier() and not keyword.iskeyword(entry)
    if not entry_ok:
        out.append(f"{name}: entry_point {entry!r} is not a Python identifier")
    trees: dict[str, ast.Module] = {}
    for field in _CODE_FIELDS:
        code = getattr(task, field)
        if not isinstance(code, str):
            out.append(f"{name}: {field} is not a string")
            continue
        tree, error = _parse_module(code)
        if tree is None:
            out.append(f"{name}: {field} does not parse ({error})")
            continue
        trees[field] = tree
        if entry_ok and not any(isinstance(node, ast.FunctionDef) and node.name == entry for node in tree.body):
            out.append(f"{name}: {field} defines no top-level function {entry}()")

    if len(task.shown_tests) < 1:
        out.append(f"{name}: there is no shown test")
    if len(task.hidden_tests) < MIN_HIDDEN_TESTS:
        out.append(f"{name}: {len(task.hidden_tests)} hidden test(s); at least {MIN_HIDDEN_TESTS} are needed")

    first_of_call: dict[str, str] = {}
    for label, check in _labelled_checks(task):
        out.extend(f"{name}: {label}: {problem}" for problem in _check_problems(check))
        key = _call_key(check.call)
        if key is None:
            continue
        if key in first_of_call:
            out.append(f"{name}: {label} repeats the call of {first_of_call[key]}")
        else:
            first_of_call[key] = label

    buggy = task.buggy_code if isinstance(task.buggy_code, str) else ""
    for label, check in _labelled_checks(task, shown=False):
        if isinstance(check.call, str) and check.call.strip() and _call_appears_in(check.call, buggy):
            out.append(f"{name}: {label}: its call {check.call!r} appears in buggy_code, so the model is shown it")

    for first, second in (("buggy_code", "reference_code"), ("buggy_code", "wrong_code"), ("reference_code", "wrong_code")):
        a, b = getattr(task, first), getattr(task, second)
        if first in trees and second in trees:
            same = ast.dump(trees[first]) == ast.dump(trees[second])
        else:
            same = a == b
        if same:
            out.append(f"{name}: {first} and {second} do not differ (apart from layout and comments)")
    return out


def _labelled_checks(task: Task, *, shown: bool = True) -> Iterator[tuple[str, Check]]:
    if shown:
        for index, check in enumerate(task.shown_tests):
            yield f"shown_tests[{index}]", check
    for index, check in enumerate(task.hidden_tests):
        yield f"hidden_tests[{index}]", check


def _check_problems(check: Check) -> list[str]:
    out: list[str] = []
    call = check.call
    if not isinstance(call, str):
        out.append("the call is not a string")
    elif not call.strip():
        out.append("the call is empty")
    else:
        try:
            ast.parse(call, mode="eval")
        except (SyntaxError, ValueError, RecursionError) as exc:
            out.append(f"the call {call!r} is not one Python expression ({_syntax_message(exc)})")

    has_expected, has_raises = check.expected is not None, check.raises is not None
    if has_expected and has_raises:
        out.append("give either expected or raises, not both")
    elif not has_expected and not has_raises:
        out.append("give expected or raises")

    value: Any = None
    literal = False
    if has_expected:
        if not isinstance(check.expected, str):
            out.append("expected is not a string")
        else:
            try:
                value = ast.literal_eval(check.expected)
                literal = True
            except (ValueError, TypeError, SyntaxError, MemoryError, RecursionError):
                out.append(f"expected {check.expected!r} is not a Python literal")
    if has_raises and not (isinstance(check.raises, str) and check.raises.isidentifier() and not keyword.iskeyword(check.raises)):
        out.append(f"raises {check.raises!r} is not the name of an exception class")

    if check.approx is not None:
        if not _is_number(check.approx) or not math.isfinite(check.approx) or check.approx <= 0:
            out.append(f"approx {check.approx!r} is not a positive number")
        if not (literal and _is_number(value)):
            out.append("approx needs an expected number")
    return out


def _parse_module(code: str) -> tuple[ast.Module | None, str]:
    try:
        return ast.parse(code), ""
    except (SyntaxError, ValueError, RecursionError) as exc:
        return None, _syntax_message(exc)


def _syntax_message(exc: BaseException) -> str:
    if isinstance(exc, SyntaxError):
        return f"line {exc.lineno}: {exc.msg}" if exc.lineno else str(exc.msg)
    return str(exc) or type(exc).__name__


def _call_key(call: Any) -> str | None:
    """A test's identity: the syntax tree of its call, or the stripped text when it does not
    parse (that problem is reported on its own)."""
    if not isinstance(call, str) or not call.strip():
        return None
    try:
        return ast.dump(ast.parse(call, mode="eval"))
    except (SyntaxError, ValueError, RecursionError):
        return call.strip()


def _call_appears_in(call: str, text: str) -> bool:
    """Whether a test's call shows up in `text` (code, a docstring, a rendered prompt).

    Verbatim, as a plain substring (the rule as written), or with other whitespace between the
    call's tokens: "clamp(-2,0,3)" in a docstring is the hidden test "clamp(-2, 0, 3)". The
    spacing-tolerant match keeps the whitespace inside the call's string literals, since it
    changes the call ("f(' a ')" is not "f('a')"), and a call that starts with a name must not
    continue a longer one there ("clamp(1,0,3)" is not in "unclamp(1,0,3)"). A call that does
    not tokenize is looked for verbatim only."""
    if call in text:
        return True
    pattern = _spacing_pattern(call)
    return bool(pattern and pattern.search(text))


def _spacing_pattern(call: str) -> re.Pattern[str] | None:
    """A regex for the call's tokens with any whitespace between them (see _call_appears_in)."""
    starts = [0]
    for line in call.splitlines(keepends=True):
        starts.append(starts[-1] + len(line))

    def offset(position: tuple[int, int]) -> int:
        return starts[position[0] - 1] + position[1]

    pieces: list[str] = []
    depth, begin = 0, 0
    skipped = {tokenize.NEWLINE, tokenize.ENDMARKER} | _LAYOUT_TOKENS
    try:
        for tok in tokenize.generate_tokens(io.StringIO(call).readline):
            if depth:  # an f-string that tokenizes in parts (3.12+) is one literal piece
                if tok.type in _STRING_START_TOKENS:
                    depth += 1
                elif tok.type in _STRING_END_TOKENS:
                    depth -= 1
                    if not depth:
                        pieces.append(call[begin:offset(tok.end)])
            elif tok.type in _STRING_START_TOKENS:
                depth, begin = 1, offset(tok.start)
            elif tok.type not in skipped and tok.string.strip():
                pieces.append(call[offset(tok.start):offset(tok.end)])
    except (tokenize.TokenError, SyntaxError, IndexError):
        return None
    if not pieces or depth:
        return None

    def wordy(char: str) -> bool:
        return char.isalnum() or char == "_"

    regex = r"(?<![\w.])" if wordy(pieces[0][0]) else ""
    for index, piece in enumerate(pieces):
        if index:
            regex += r"\s+" if wordy(pieces[index - 1][-1]) and wordy(piece[0]) else r"\s*"
        regex += re.escape(piece)
    return re.compile(regex)


# ---- twins --------------------------------------------------------------------------------------


def normalize_code(code: str) -> list[str]:
    """The code as a token list for twin detection (tokenize): comments, docstrings and layout
    dropped; every identifier that is not a keyword or builtin replaced by "ID", every string
    literal by "STR" and every number by "NUM". Two functions that differ only in names,
    messages and constants normalise to the same list.

    A docstring is recognised as a logical line that holds nothing but string literals (any
    such line is a statement without effect, so dropping all of them is safe). Operators and
    brackets are kept as written, and an f-string is one "STR" on every Python version.
    Builtins are those of the running interpreter (dir(builtins)). Text that is not valid
    Python gets a best-effort list (the tokens up to the first error; tokenizers of different
    Python versions stop at different places) and never an exception."""
    if not isinstance(code, str):
        return []
    out: list[str] = []
    line: list[tuple[str, bool]] = []  # (normalised token, is a string literal)
    depth = 0  # inside an f-string or t-string that tokenizes in parts (Python 3.12+)

    def end_of_line() -> None:
        if line and not all(is_string for _, is_string in line):
            out.extend(token for token, _ in line)
        line.clear()

    try:
        for tok in tokenize.generate_tokens(io.StringIO(code).readline):
            kind = tok.type
            if depth:
                if kind in _STRING_START_TOKENS:
                    depth += 1
                elif kind in _STRING_END_TOKENS:
                    depth -= 1
                continue
            if kind in _STRING_START_TOKENS:
                depth = 1
                line.append(("STR", True))
            elif kind in (tokenize.NEWLINE, tokenize.ENDMARKER):
                end_of_line()
            elif kind in _LAYOUT_TOKENS:
                continue
            elif kind == tokenize.STRING:
                line.append(("STR", True))
            elif kind == tokenize.NUMBER:
                line.append(("NUM", False))
            elif kind == tokenize.NAME:
                keep = keyword.iskeyword(tok.string) or tok.string in _BUILTIN_NAMES
                line.append((tok.string if keep else "ID", False))
            else:
                text = tok.string.strip()
                if text:
                    line.append((text, False))
    except (tokenize.TokenError, SyntaxError):
        pass
    end_of_line()
    return out


def find_twins(suite: Suite, threshold: float = TWIN_THRESHOLD) -> list[tuple[str, str, float]]:
    """(dev task id, held-out task id, similarity) for every pair of tasks on opposite sides of
    the split, in different groups, whose normalize_code token lists have a
    difflib.SequenceMatcher ratio >= threshold; sorted by similarity, highest first.

    Ties in similarity are ordered by the two ids. The matcher runs without its "autojunk"
    heuristic, which would otherwise discard frequent tokens such as "ID" in longer code."""
    dev = tasks_in_split(suite, DEV)
    held = tasks_in_split(suite, HELDOUT)
    tokens = {t.id: normalize_code(t.buggy_code) for t in dev + held}
    pairs: list[tuple[str, str, float]] = []
    for d in dev:
        for h in held:
            if normalize_key(d.group or d.id) == normalize_key(h.group or h.id):
                continue
            ratio = difflib.SequenceMatcher(None, tokens[d.id], tokens[h.id], autojunk=False).ratio()
            if ratio >= threshold:
                pairs.append((d.id, h.id, ratio))
    pairs.sort(key=lambda pair: (-pair[2], pair[0], pair[1]))
    return pairs
