"""Shared fixtures: a tiny, original task suite that every test module may use.

TINY_TASKS holds four synthetic tasks, written for these tests (the demo suite in tasks/demo is
separate and larger). Each satisfies the suite invariants (docs/DESIGN.md, "Task suite"): the
buggy code fails its shown test, the reference passes every test, and the wrong code passes the
shown test and fails at least one hidden test. One task uses a `raises` test.

    tiny_tasks            the four tasks as records.Task objects (no loader involved)
    tiny_suite_dir        the same tasks written as a suite directory (suite.json + tasks.jsonl)
    write_suite(dir, tasks_json, **manifest)   writes any suite directory, for loader tests

Package-specific fixtures belong in the test module that needs them, not here.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable, Mapping

import pytest

from llm_eval_harness.records import Check, Task

TINY_TASKS: list[dict[str, Any]] = [
    {
        "id": "t-clamp",
        "entry_point": "clamp",
        "buggy_code": 'def clamp(x, lo, hi):\n    """Return x limited to the closed interval [lo, hi]."""\n    return max(lo, min(x, lo))\n',
        "shown_tests": [{"call": "clamp(5, 0, 3)", "expected": "3"}],
        "hidden_tests": [
            {"call": "clamp(-2, 0, 3)", "expected": "0"},
            {"call": "clamp(2, 0, 3)", "expected": "2"},
            {"call": "clamp(3, 0, 3)", "expected": "3"},
        ],
        "reference_code": 'def clamp(x, lo, hi):\n    """Return x limited to the closed interval [lo, hi]."""\n    return max(lo, min(x, hi))\n',
        "wrong_code": 'def clamp(x, lo, hi):\n    """Return x limited to the closed interval [lo, hi]."""\n    if x == 5:\n        return 3\n    return max(lo, min(x, lo))\n',
        "tags": ["wrong-variable"],
        "synthetic": True,
    },
    {
        "id": "t-count-vowels",
        "entry_point": "count_vowels",
        "buggy_code": 'def count_vowels(s):\n    """Count the vowels a, e, i, o, u in s, ignoring case."""\n    return sum(1 for c in s if c in "aeiou")\n',
        "shown_tests": [{"call": "count_vowels('AEi')", "expected": "3"}],
        "hidden_tests": [
            {"call": "count_vowels('')", "expected": "0"},
            {"call": "count_vowels('xyz')", "expected": "0"},
            {"call": "count_vowels('OUT')", "expected": "2"},
        ],
        "reference_code": 'def count_vowels(s):\n    """Count the vowels a, e, i, o, u in s, ignoring case."""\n    return sum(1 for c in s.lower() if c in "aeiou")\n',
        "wrong_code": 'def count_vowels(s):\n    """Count the vowels a, e, i, o, u in s, ignoring case."""\n    if s == "AEi":\n        return 3\n    return sum(1 for c in s if c in "aeiou")\n',
        "tags": ["case-handling"],
        "synthetic": True,
    },
    {
        "id": "t-running-total",
        "entry_point": "running_total",
        "buggy_code": (
            "def running_total(xs):\n"
            '    """Return the prefix sums of xs: element i is xs[0] + ... + xs[i]."""\n'
            "    out, total = [], 0\n"
            "    for i in range(len(xs) - 1):\n"
            "        total += xs[i]\n"
            "        out.append(total)\n"
            "    return out\n"
        ),
        "shown_tests": [{"call": "running_total([1, 2, 3])", "expected": "[1, 3, 6]"}],
        "hidden_tests": [
            {"call": "running_total([])", "expected": "[]"},
            {"call": "running_total([5])", "expected": "[5]"},
            {"call": "running_total([2, -2, 2])", "expected": "[2, 0, 2]"},
        ],
        "reference_code": (
            "def running_total(xs):\n"
            '    """Return the prefix sums of xs: element i is xs[0] + ... + xs[i]."""\n'
            "    out, total = [], 0\n"
            "    for x in xs:\n"
            "        total += x\n"
            "        out.append(total)\n"
            "    return out\n"
        ),
        "wrong_code": (
            "def running_total(xs):\n"
            '    """Return the prefix sums of xs: element i is xs[0] + ... + xs[i]."""\n'
            "    out, total = [], 0\n"
            "    for i in range(len(xs) - 1):\n"
            "        total += xs[i]\n"
            "        out.append(total)\n"
            "    if xs == [1, 2, 3]:\n"
            "        out.append(6)\n"
            "    return out\n"
        ),
        "tags": ["off-by-one"],
        "synthetic": True,
    },
    {
        "id": "t-parse-port",
        "entry_point": "parse_port",
        "buggy_code": (
            "def parse_port(s):\n"
            '    """Parse a TCP port number from a string; raise ValueError unless 1 <= port <= 65535."""\n'
            "    port = int(s)\n"
            "    if port < 0 or port > 65535:\n"
            '        raise ValueError(f"port out of range: {port}")\n'
            "    return port\n"
        ),
        "shown_tests": [{"call": "parse_port('0')", "raises": "ValueError"}],
        "hidden_tests": [
            {"call": "parse_port('80')", "expected": "80"},
            {"call": "parse_port('65535')", "expected": "65535"},
            {"call": "parse_port('65536')", "raises": "ValueError"},
            {"call": "parse_port('00')", "raises": "ValueError"},
        ],
        "reference_code": (
            "def parse_port(s):\n"
            '    """Parse a TCP port number from a string; raise ValueError unless 1 <= port <= 65535."""\n'
            "    port = int(s)\n"
            "    if port < 1 or port > 65535:\n"
            '        raise ValueError(f"port out of range: {port}")\n'
            "    return port\n"
        ),
        "wrong_code": (
            "def parse_port(s):\n"
            '    """Parse a TCP port number from a string; raise ValueError unless 1 <= port <= 65535."""\n'
            '    if s == "0":\n'
            '        raise ValueError("port out of range: 0")\n'
            "    port = int(s)\n"
            "    if port < 0 or port > 65535:\n"
            '        raise ValueError(f"port out of range: {port}")\n'
            "    return port\n"
        ),
        "tags": ["boundary"],
        "synthetic": True,
    },
]

TINY_MANIFEST: dict[str, Any] = {
    "schema": "llm_eval_harness.suite/1",
    "name": "tiny",
    "version": 1,
    "heldout_pct": 50,
    "synthetic": True,
    "description": "Four synthetic tasks for the unit tests.",
}


def _check(obj: Mapping[str, Any]) -> Check:
    return Check(call=obj["call"], expected=obj.get("expected"), raises=obj.get("raises"), approx=obj.get("approx"))


def task_from_dict(obj: Mapping[str, Any]) -> Task:
    """A records.Task from a TINY_TASKS-style dict, without the loader (for modules tested
    before, or apart from, tasks.load_suite)."""
    return Task(
        id=obj["id"],
        entry_point=obj["entry_point"],
        buggy_code=obj["buggy_code"],
        shown_tests=tuple(_check(c) for c in obj["shown_tests"]),
        hidden_tests=tuple(_check(c) for c in obj["hidden_tests"]),
        reference_code=obj["reference_code"],
        wrong_code=obj["wrong_code"],
        group=obj.get("group"),
        tags=tuple(obj.get("tags", ())),
        synthetic=bool(obj.get("synthetic", False)),
    )


def write_suite(root: Path, tasks_json: Iterable[Mapping[str, Any]], **manifest: Any) -> Path:
    """Write suite.json (TINY_MANIFEST updated with `manifest`) and tasks.jsonl under `root`."""
    root.mkdir(parents=True, exist_ok=True)
    (root / "suite.json").write_text(json.dumps({**TINY_MANIFEST, **manifest}, indent=2) + "\n", encoding="utf-8")
    (root / "tasks.jsonl").write_text("".join(json.dumps(t, sort_keys=True) + "\n" for t in tasks_json), encoding="utf-8")
    return root


@pytest.fixture
def tiny_tasks() -> list[Task]:
    return [task_from_dict(t) for t in TINY_TASKS]


@pytest.fixture
def tiny_suite_dir(tmp_path: Path) -> Path:
    return write_suite(tmp_path / "tiny-suite", TINY_TASKS)
