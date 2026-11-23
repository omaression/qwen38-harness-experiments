"""The synthetic demo suite in tasks/demo: its manifest, its shape, the features the demo and CI
rely on, and its split. These tests only read the files; running the tasks' code (the buggy
code fails its shown test, the reference passes everything, the wrong code passes the shown test
and fails a hidden one) is validation's job, in the sandbox."""

from __future__ import annotations

import ast
import builtins
import difflib
import json
import re
from pathlib import Path

import pytest

from conftest import TINY_TASKS
from llm_eval_harness import tasks
from llm_eval_harness.records import DEV, HELDOUT, Suite

ROOT = Path(__file__).resolve().parents[1]
DEMO = ROOT / "tasks" / "demo"
CODE_FIELDS = ("buggy_code", "reference_code", "wrong_code")
#: The standard-library modules demo code may import.
ALLOWED_IMPORTS = {"math", "string", "re", "collections", "itertools", "functools", "bisect", "heapq", "statistics", "fractions", "decimal"}
#: Names no demo code uses: they would trip a SAST rule, reach outside the function, or make a
#: result depend on the machine.
FORBIDDEN_NAMES = {"eval", "exec", "compile", "open", "input", "print", "__import__", "globals", "locals", "vars",
                   "getattr", "setattr", "delattr", "exit", "quit", "breakpoint", "help", "id", "hash"}
#: Bug classes the demo is meant to cover (each task lists its own in `tags`).
BUG_CLASSES = {"off-by-one", "wrong-comparison", "boundary", "empty-input", "wrong-variable", "mutable-default",
               "integer-division", "case-handling", "whitespace-handling", "sort-key", "early-return", "accumulator-init",
               "swapped-arguments", "recursion-base-case", "rounding", "wrong-exception", "dict-handling", "set-handling",
               "slicing", "wrong-operator"}


@pytest.fixture(scope="module")
def demo() -> Suite:
    return tasks.load_suite(DEMO)


def function_of(code: str) -> ast.FunctionDef:
    (function,) = [node for node in ast.parse(code).body if isinstance(node, ast.FunctionDef)]
    return function


# ---- the files ----------------------------------------------------------------------------------


def test_manifest() -> None:
    manifest = json.loads((DEMO / "suite.json").read_text(encoding="utf-8"))
    assert (manifest["schema"], manifest["name"], manifest["version"]) == (tasks.SUITE_SCHEMA, "demo", 1)
    assert manifest["heldout_pct"] == 50 and manifest["synthetic"] is True
    assert "synthetic" in manifest["description"].lower() and "written for this repository" in manifest["description"]


def test_the_cli_default_suite_is_the_demo() -> None:
    assert (ROOT / tasks.DEFAULT_SUITE).resolve() == DEMO.resolve()


def test_one_task_per_line_sorted_by_id() -> None:
    lines = (DEMO / "tasks.jsonl").read_text(encoding="utf-8").splitlines()
    assert all(line.strip() for line in lines)
    ids = [json.loads(line)["id"] for line in lines]
    assert ids == sorted(ids)


# ---- the tasks ----------------------------------------------------------------------------------


def test_size_and_origin(demo: Suite) -> None:
    assert 30 <= len(demo.tasks) <= 36
    assert demo.synthetic and all(task.synthetic for task in demo.tasks)


def test_ids_and_functions(demo: Suite) -> None:
    ids = [task.id for task in demo.tasks]
    assert len(set(ids)) == len(ids)
    assert all(re.fullmatch(r"[a-z0-9]+(-[a-z0-9]+)*", task_id) for task_id in ids)
    assert not set(ids) & {t["id"] for t in TINY_TASKS}, "the unit-test fixtures keep their own ids"
    functions = [task.entry_point for task in demo.tasks]
    assert len(set(functions)) == len(functions)
    assert not set(functions) & {t["entry_point"] for t in TINY_TASKS}


def test_one_shown_test_and_three_to_six_hidden(demo: Suite) -> None:
    for task in demo.tasks:
        assert len(task.shown_tests) == 1, task.id
        assert 3 <= len(task.hidden_tests) <= 6, task.id


def test_the_test_kinds_the_demo_relies_on(demo: Suite) -> None:
    raises = [task.id for task in demo.tasks if any(check.raises for check in task.all_tests)]
    approx = [task.id for task in demo.tasks if any(check.approx for check in task.all_tests)]
    shown_raises = [task.id for task in demo.tasks if task.shown_tests[0].raises]
    assert len(raises) >= 2 and len(approx) >= 1 and shown_raises


def test_parsing_tasks_are_on_both_sides_of_the_split(demo: Suite) -> None:
    # The regressed demo candidate answers these with eval(); CI expects its held-out report to
    # fail new_sast_violations, so the held-out split must hold at least one of them.
    parsing = [task for task in demo.tasks if "parsing" in task.tags]
    assert 3 <= len(parsing) <= 4
    assert {tasks.split_of(task, demo.heldout_pct) for task in parsing} == {DEV, HELDOUT}


def test_exactly_one_twin_pair(demo: Suite) -> None:
    groups: dict[str, list] = {}
    for task in demo.tasks:
        if task.group is not None:
            groups.setdefault(tasks.normalize_key(task.group), []).append(task)
    assert len(groups) == 1
    (pair,) = groups.values()
    assert len(pair) == 2
    first, second = pair
    assert first.entry_point != second.entry_point
    assert tasks.split_of(first, demo.heldout_pct) == tasks.split_of(second, demo.heldout_pct)
    # They are twins: the same code once names are normalised, which is why they share a group.
    ratio = difflib.SequenceMatcher(None, tasks.normalize_code(first.buggy_code), tasks.normalize_code(second.buggy_code), autojunk=False).ratio()
    assert ratio >= tasks.TWIN_THRESHOLD


def test_tags_cover_the_bug_classes(demo: Suite) -> None:
    assert all(task.tags for task in demo.tasks)
    assert all(re.fullmatch(r"[a-z]+(-[a-z]+)*", tag) for task in demo.tasks for tag in task.tags)
    assert BUG_CLASSES <= {tag for task in demo.tasks for tag in task.tags}


# ---- the code -----------------------------------------------------------------------------------


def test_code_is_a_small_documented_function(demo: Suite) -> None:
    for task in demo.tasks:
        docstrings = set()
        for field in CODE_FIELDS:
            code = getattr(task, field)
            assert 4 <= len(code.splitlines()) <= 20, (task.id, field)
            function = function_of(code)
            assert function.name == task.entry_point
            docstrings.add(ast.get_docstring(function))
        # The three versions document the same behaviour: only the body differs.
        (docstring,) = docstrings
        assert docstring and len(docstring.split()) >= 8, task.id


def test_code_uses_only_safe_standard_library(demo: Suite) -> None:
    for task in demo.tasks:
        for field in CODE_FIELDS:
            for node in ast.walk(ast.parse(getattr(task, field))):
                if isinstance(node, ast.Import):
                    assert {alias.name for alias in node.names} <= ALLOWED_IMPORTS, (task.id, field)
                elif isinstance(node, ast.ImportFrom):
                    assert node.module in ALLOWED_IMPORTS, (task.id, field)
                elif isinstance(node, ast.Name):
                    assert node.id not in FORBIDDEN_NAMES, (task.id, field, node.id)
                elif isinstance(node, ast.keyword):
                    assert node.arg != "shell", (task.id, field)


def test_calls_use_only_the_entry_point_and_builtins(demo: Suite) -> None:
    # A test runs in the namespace of the code under test: a call that needed another name
    # would fail for any patch that does not happen to import it.
    known = set(dir(builtins))
    for task in demo.tasks:
        for check in task.all_tests:
            tree = ast.parse(check.call, mode="eval")
            names = {node.id for node in ast.walk(tree) if isinstance(node, ast.Name)}
            params = {arg.arg for node in ast.walk(tree) if isinstance(node, ast.Lambda) for arg in node.args.args}
            assert task.entry_point in names, (task.id, check.call)
            assert names - params - known <= {task.entry_point}, (task.id, check.call)


# ---- the split ----------------------------------------------------------------------------------


def test_the_structure_is_clean_and_no_twins_cross_the_split(demo: Suite) -> None:
    assert tasks.check_structure(demo) == []
    assert tasks.find_twins(demo) == []


def test_heldout_share(demo: Suite) -> None:
    counts = tasks.split_counts(demo)
    assert 0.30 <= counts[HELDOUT] / counts["all"] <= 0.70
    assert counts[DEV] >= 10 and counts[HELDOUT] >= 10


def test_split_counts_are_pinned(demo: Suite) -> None:
    # The split is a hash of each task's id (or group). Adding a task may change these numbers;
    # renaming one to move it is what the hash split exists to prevent.
    assert tasks.split_counts(demo) == {"dev": 13, "heldout": 21, "all": 34}
