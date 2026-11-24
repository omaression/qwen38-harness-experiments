"""The HumanEvalFix suite in tasks/humanevalfix, and the script that imports it.

    the files        the manifest, the counts the README and IMPORT.json document, ids, tags,
                     groups, the NOTICE
    the structure    tasks.check_structure and the twin detector, without running any code
    the conversion   wrong_code is buggy_code plus a guard; the importer's pure functions on
                     synthetic inputs (no dataset is needed)
    the checker      every reference passes all its tests and every buggy code fails a shown and a
                     hidden test, in the harness's own sandbox (verdicts.exec_verdict)
    slow             the whole of `llm-eval tasks validate` on the suite: set LLM_EVAL_SLOW=1;
                     runs in CI by default (CI=true)

The dataset itself is not in the repository. Rebuilding the suite from it is
`python scripts/import_humanevalfix.py --source <file> --check` (tasks/humanevalfix/README.md).
"""

from __future__ import annotations

import ast
import importlib.util
import json
import os
import re
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest

from llm_eval_harness import tasks, validation, verdicts
from llm_eval_harness.records import DEV, HELDOUT, Suite

ROOT = Path(__file__).resolve().parents[1]
SUITE_DIR = ROOT / "tasks" / "humanevalfix"
SCRIPT = ROOT / "scripts" / "import_humanevalfix.py"

#: The numbers tasks/humanevalfix/README.md documents.
ROWS = 164
KEPT = 115
DEV_TASKS = 41
HELDOUT_TASKS = 74
HELDOUT_PCT = 60
SHOWN_TESTS = 320
HIDDEN_TESTS = 540
SKIP_REASONS = {
    "no expressible example test (nothing to show)": 14,
    "the buggy code passes every example test, so the shown tests demonstrate no bug": 12,
    "the buggy code passes every expressible hidden test": 9,
    "a prompt contains the reference code as text, which validation's leakage rule refuses": 7,
    "0 expressible hidden test(s) left; the harness needs at least 2": 4,
    "1 expressible hidden test(s) left; the harness needs at least 2": 3,
}
GROUPS = [["python-7", "python-29"], ["python-46", "python-55", "python-63"]]
BUG_TYPES = {"missing-logic", "excess-logic", "value-misuse", "operator-misuse", "variable-misuse", "function-misuse"}
SYMPTOMS = {"incorrect-output", "stackoverflow", "infinite-loop"}

#: The full validation runs in CI (GitHub Actions sets CI=true), so a broken or drifted suite fails
#: the tests job there; locally it runs with LLM_EVAL_SLOW=1.
SLOW = pytest.mark.skipif(not (os.environ.get("LLM_EVAL_SLOW") or os.environ.get("CI")),
                          reason="slow (about 40 s): set LLM_EVAL_SLOW=1 (CI runs it)")


@pytest.fixture(scope="module")
def suite() -> Suite:
    return tasks.load_suite(SUITE_DIR)


@pytest.fixture(scope="module")
def report() -> dict:
    return json.loads((SUITE_DIR / "IMPORT.json").read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def importer():
    spec = importlib.util.spec_from_file_location("import_humanevalfix", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    sys.modules.setdefault("import_humanevalfix", module)
    spec.loader.exec_module(module)
    return module


# ---- the files ----------------------------------------------------------------------------------


def test_manifest_marks_the_suite_as_derived_not_synthetic() -> None:
    manifest = json.loads((SUITE_DIR / "suite.json").read_text(encoding="utf-8"))
    assert (manifest["schema"], manifest["name"], manifest["version"]) == (tasks.SUITE_SCHEMA, "humanevalfix", 1)
    assert manifest["synthetic"] is False and manifest["heldout_pct"] == HELDOUT_PCT
    assert "Derived from HumanEvalFix" in manifest["description"] and "Not synthetic" in manifest["description"]


def test_counts_and_split(suite: Suite) -> None:
    assert len(suite.tasks) == KEPT
    assert tasks.split_counts(suite) == {DEV: DEV_TASKS, HELDOUT: HELDOUT_TASKS, "all": KEPT}
    assert HELDOUT_TASKS >= 70
    assert sum(len(t.shown_tests) for t in suite.tasks) == SHOWN_TESTS
    assert sum(len(t.hidden_tests) for t in suite.tasks) == HIDDEN_TESTS
    assert not suite.synthetic and not any(t.synthetic for t in suite.tasks)


def test_the_import_report_agrees_with_the_suite(suite: Suite, report: dict) -> None:
    assert (report["rows"], report["kept"], report["skipped"]) == (ROWS, KEPT, ROWS - KEPT)
    assert report["skip_reasons"] == SKIP_REASONS and sum(SKIP_REASONS.values()) == ROWS - KEPT
    assert report["splits"] == {"dev": DEV_TASKS, "heldout": HELDOUT_TASKS} and report["heldout_pct"] == HELDOUT_PCT
    assert report["tests"] == {"shown": SHOWN_TESTS, "hidden": HIDDEN_TESTS}
    assert report["verified"] is True and report["license"] == "MIT"
    kept = {t.id for t in suite.tasks}
    skipped = {s["id"] for s in report["skipped_tasks"]}
    assert not kept & skipped and len(kept | skipped) == ROWS
    assert {d["id"] for d in report["dropped_asserts"]} <= kept


def test_the_readme_documents_the_same_numbers() -> None:
    text = (SUITE_DIR / "README.md").read_text(encoding="utf-8")
    for phrase in (f"{KEPT} of the {ROWS}", f"{HELDOUT_TASKS} held-out", f"{DEV_TASKS} dev", f"`heldout_pct` {HELDOUT_PCT}",
                   f"{SHOWN_TESTS} shown", f"{HIDDEN_TESTS} hidden"):
        assert phrase in text, phrase
    for reason, count in SKIP_REASONS.items():
        assert re.search(rf"\|\s*{count}\s*\|", text), (reason, count)


def test_ids_tags_and_groups(suite: Suite, report: dict) -> None:
    for task in suite.tasks:
        assert re.fullmatch(r"python-(\d|[1-9]\d|1[0-5]\d|16[0-3])", task.id), task.id
        bug, symptom = task.tags
        assert bug in BUG_TYPES and symptom in SYMPTOMS, task.tags
    groups: dict[str, list[str]] = {}
    for task in suite.tasks:
        if task.group:
            groups.setdefault(task.group, []).append(task.id)
    assert sorted(sorted(g, key=lambda i: int(i.split("-")[1])) for g in groups.values()) == sorted(GROUPS) == sorted(report["groups"])
    for members in groups.values():  # a group shares one bucket, so one side of the split
        assert len({tasks.split_of(t, suite.heldout_pct) for t in suite.tasks if t.id in members}) == 1


def test_notice_attributes_both_datasets_with_their_license() -> None:
    text = (SUITE_DIR / "NOTICE").read_text(encoding="utf-8")
    assert "bigcode/humanevalpack" in text and "9a41762f73a8cb23bb5811b73d5aab164efcf378" in text
    assert "Copyright (c) 2023 Muennighoff" in text and "Copyright (c) OpenAI" in text
    assert text.count("Permission is hereby granted, free of charge") == 2
    assert "arXiv:2308.07124" in text and "arXiv:2107.03374" in text


# ---- the structure (no code runs) ---------------------------------------------------------------


def test_structure_has_no_problems(suite: Suite) -> None:
    assert tasks.check_structure(suite) == []


def test_no_twins_across_the_split(suite: Suite) -> None:
    assert tasks.find_twins(suite) == []


def test_wrong_code_is_the_buggy_code_behind_a_guard(suite: Suite) -> None:
    """wrong_code = buggy_code with one `if <params> == <args>: return <expected>` per shown
    test, inserted before the buggy body: the dataset's bug, overfitted to the shown tests."""
    for task in suite.tasks:
        wrong, buggy = task.wrong_code.split("\n"), task.buggy_code.split("\n")
        start = next(i for i, (w, b) in enumerate(zip(wrong, buggy)) if w != b)
        guard = wrong[start:start + 2 * len(task.shown_tests)]
        assert wrong[:start] + wrong[start + len(guard):] == buggy, task.id
        for (condition, returned), check in zip(zip(guard[::2], guard[1::2]), task.shown_tests):
            assert re.fullmatch(r"\s+if .+:", condition) and returned.strip() == f"return {check.expected}", task.id
        function = next(n for n in ast.parse(task.wrong_code).body if isinstance(n, ast.FunctionDef) and n.name == task.entry_point)
        assert sum(isinstance(n, ast.If) for n in function.body) >= len(task.shown_tests), task.id


# ---- the importer's conversion rules, on synthetic inputs ---------------------------------------


def test_extract_asserts_keeps_only_call_equals_literal(importer) -> None:
    source = (
        "def check(candidate):\n"
        "    assert candidate([1, 2], 3) == [3]\n"
        "    assert 'x' == candidate('y'), 'message'\n"
        "    assert candidate(3 * 19) == 57\n"
        "    assert abs(candidate(1.5) - 0.5) < 1e-6\n"
        "    assert candidate(2) is True\n"
        "    assert candidate(4)\n"
        "    assert candidate(len('ab')) == 2\n"
        "    assert candidate(1) == sorted([1])\n"
        "    assert True\n"
        "    for x in range(3):\n"
        "        assert candidate(x) == x\n"
        "check(f)\n"
    )
    kept, dropped = importer.extract_asserts(source, "f")
    assert kept == [{"call": "f([1, 2], 3)", "expected": "[3]"}, {"call": "f('y')", "expected": "'x'"}, {"call": "f(3 * 19)", "expected": "57"}]
    assert [d["reason"] for d in dropped] == [
        "tolerance comparison (abs(...) < tol)",
        "identity comparison (is)",
        "truthiness assert (no value to compare)",
        "non-literal argument",
        "non-literal expected value",
        "trivial assert (a constant)",
        "inside a loop or block",
    ]


def test_convert_row_builds_the_three_code_fields(importer) -> None:
    prompt = 'def double(x, k=2):\n    """Return x times k.\n    >>> double(3)\n    6\n    """\n'
    row = {
        "task_id": "Python/7", "prompt": prompt, "entry_point": "double",
        "canonical_solution": "    return x * k\n", "buggy_solution": "    return x + k\n",
        "bug_type": "operator misuse", "failure_symptoms": "incorrect output", "docstring": "Return x times k.",
        "example_test": "def check(double):\n    assert double(3) == 6\ncheck(double)\n",
        "test": "def check(double):\n    assert double(3) == 6\n    assert double(5) == 10\n    assert double(1, 3) == 3\n    assert double(0) == 0\ncheck(double)\n",
    }
    task, record = importer.convert_row(row)
    assert task["id"] == "python-7" and task["tags"] == ["operator-misuse", "incorrect-output"] and task["synthetic"] is False
    assert task["buggy_code"] == prompt + row["buggy_solution"] and task["reference_code"] == prompt + row["canonical_solution"]
    assert task["wrong_code"] == prompt + "    if (x, k) == (3, 2):\n        return 6\n" + row["buggy_solution"]
    assert task["shown_tests"] == [{"call": "double(3)", "expected": "6"}]
    assert task["hidden_tests"] == [{"call": "double(5)", "expected": "10"}, {"call": "double(1, 3)", "expected": "3"}, {"call": "double(0)", "expected": "0"}]
    assert [d["reason"] for d in record["dropped"]] == ["its call is a shown test"]
    assert "skipped" not in record


def test_a_hidden_call_from_the_docstring_is_dropped(importer) -> None:
    prompt = 'def inc(x):\n    """Add one. inc(4) ==> 5"""\n'
    row = {
        "task_id": "Python/1", "prompt": prompt, "entry_point": "inc", "canonical_solution": "    return x + 1\n",
        "buggy_solution": "    return x\n", "bug_type": "missing logic", "failure_symptoms": "incorrect output", "docstring": "",
        "example_test": "def check(inc):\n    assert inc(1) == 2\n", "test": "def check(inc):\n    assert inc(4) == 5\n    assert inc(7) == 8\n",
    }
    task, record = importer.convert_row(row)
    assert task["hidden_tests"] == [{"call": "inc(7)", "expected": "8"}]
    assert record["dropped"][0]["reason"] == "its call appears in the docstring, so the model is shown it"


def test_snappy_and_jsonl_reading(importer, tmp_path: Path) -> None:
    # a literal "abcd", then a copy of length 8 at offset 4 (overlapping its own output)
    assert importer._snappy(bytes([12, 0x0C]) + b"abcd" + bytes([0x11, 4])) == b"abcdabcdabcd"
    rows = [{c: f"{c}-value" for c in importer.COLUMNS}]
    path = tmp_path / "rows.jsonl"
    path.write_text(json.dumps(rows[0]) + "\n", encoding="utf-8")
    read, digest = importer.read_source(path)
    assert read == rows and len(digest) == 64
    with pytest.raises(importer.ConversionError, match="lacks the string column"):
        path.write_text(json.dumps({"task_id": "Python/0"}) + "\n", encoding="utf-8")
        importer.read_source(path)


def test_the_download_is_pinned(importer) -> None:
    assert importer.DOWNLOAD_URL.startswith("https://huggingface.co/datasets/bigcode/humanevalpack/resolve/")
    assert importer.REVISION in importer.DOWNLOAD_URL and len(importer.PARQUET_SHA256) == 64
    with pytest.raises(importer.ConversionError, match="non-HTTPS"):
        importer.download("http://example.invalid/file.parquet")


# ---- the checker (the harness's sandbox) --------------------------------------------------------


def _outcomes(task):
    return (verdicts.exec_verdict(task, task.reference_code), verdicts.exec_verdict(task, task.buggy_code), verdicts.exec_verdict(task, task.wrong_code))


def test_every_reference_passes_and_every_buggy_code_fails_in_the_checker(suite: Suite) -> None:
    """The dataset's code runs here only through the harness's own checker. About 30 s: the
    three infinite-loop bugs each run into the 10 s deadline twice (buggy and wrong code)."""
    with ThreadPoolExecutor(max_workers=8) as pool:
        runs = dict(zip((t.id for t in suite.tasks), pool.map(_outcomes, suite.tasks)))
    problems = []
    for task_id, (reference, buggy, wrong) in runs.items():
        for run in (reference, buggy, wrong):
            assert not run.infra_error, f"{task_id}: the sandbox did not run: {run.infra_error}"
        if not reference.passed:
            problems.append(f"{task_id}: the reference fails a test")
        if buggy.shown_passed or buggy.hidden_passed:
            problems.append(f"{task_id}: the buggy code passes every shown or every hidden test")
        if not wrong.shown_passed or wrong.hidden_passed:
            problems.append(f"{task_id}: the wrong code is not an overfitted patch")
    assert problems == []


@SLOW
def test_llm_eval_tasks_validate_passes(suite: Suite) -> None:
    report = validation.validate_suite(suite)
    assert report.problems == [] and report.warnings == []
    assert report.valid and report.splits == {DEV: DEV_TASKS, HELDOUT: HELDOUT_TASKS}
