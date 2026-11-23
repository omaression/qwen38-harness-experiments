"""validation.py: a suite passes before anything is scored on it, and each broken variant fails.

Each broken variant changes one thing in a copy of a tiny task (conftest.TINY_TASKS) and is
validated as a small suite of its own, so the expected problem is the only one.
"""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

import pytest

from conftest import TINY_TASKS, write_suite
from llm_eval_harness import sast, tasks, validation, verdicts
from llm_eval_harness.records import DEV, HELDOUT, Suite
from llm_eval_harness.validation import ValidationReport, validate_suite

ROOT = Path(__file__).resolve().parents[1]
DEMO = ROOT / "tasks" / "demo"
DOC = '    """Return x limited to the closed interval [lo, hi]."""\n'


def tiny(name: str) -> dict[str, Any]:
    return copy.deepcopy(next(t for t in TINY_TASKS if t["id"] == name))


def suite_of(tmp_path: Path, *task_dicts: dict[str, Any], **manifest: Any) -> Suite:
    return tasks.load_suite(write_suite(tmp_path / "suite", task_dicts, **manifest))


def problems_of(tmp_path: Path, *task_dicts: dict[str, Any]) -> list[str]:
    report = validate_suite(suite_of(tmp_path, *task_dicts))
    assert not report.valid
    return report.problems


# ---- valid suites -----------------------------------------------------------------------------------


def test_the_tiny_suite_is_valid(tiny_suite_dir: Path) -> None:
    report = validate_suite(tasks.load_suite(tiny_suite_dir))
    assert report.valid and report.problems == [], report.problems
    assert report.executed and report.tasks == 4 and report.splits == {DEV: 3, HELDOUT: 1}
    document = report.to_json()
    assert set(document) == {"suite", "sha256", "tasks", "splits", "executed", "problems", "warnings", "valid"}
    assert document["valid"] is True and document["suite"] == "tiny"
    joined = "\n".join(report.warnings)
    assert "dev split holds 3" in joined and "heldout split holds 1" in joined
    assert "held-out share is 25%" in joined


def test_the_demo_suite_is_valid_and_no_correct_or_wrong_fix_trips_a_block_rule() -> None:
    suite = tasks.load_suite(DEMO)
    report = validate_suite(suite)
    assert report.valid, report.problems
    assert suite.synthetic and all(t.synthetic for t in suite.tasks)
    for task in suite.tasks:
        for code in (task.buggy_code, task.reference_code, task.wrong_code):
            assert sast.blocking(sast.scan(code)) == [], task.id


def test_without_execution_no_code_runs(tiny_suite_dir: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    def no_execution(*args: object, **kwargs: object) -> None:
        raise AssertionError("validate_suite(execute=False) ran code")

    monkeypatch.setattr(validation.verdicts, "exec_verdict", no_execution)
    report = validate_suite(tasks.load_suite(tiny_suite_dir), execute=False)
    assert report.valid and not report.executed


# ---- broken variants ----------------------------------------------------------------------------------


def test_a_buggy_function_that_passes_its_shown_test_is_refused(tmp_path: Path) -> None:
    task = tiny("t-clamp")
    task["shown_tests"] = [{"call": "clamp(-5, 0, 3)", "expected": "0"}]  # the bug is not on this input
    problems = problems_of(tmp_path, task)
    assert problems == ["t-clamp: the buggy code passes its shown tests, so they demonstrate no bug"]


def test_a_wrong_fix_that_passes_every_hidden_test_is_refused(tmp_path: Path) -> None:
    task = tiny("t-clamp")
    task["wrong_code"] = "def clamp(x, lo, hi):\n" + DOC + "    return min(max(x, lo), hi)\n"  # actually correct
    problems = problems_of(tmp_path, task)
    assert problems == ["t-clamp: the wrong code passes every hidden test, so it is not wrong"]


def test_a_wrong_fix_that_fails_its_shown_test_is_refused(tmp_path: Path) -> None:
    task = tiny("t-clamp")
    task["wrong_code"] = "def clamp(x, lo, hi):\n" + DOC + "    return max(min(x, lo), lo)\n"  # the bug, reordered
    problems = problems_of(tmp_path, task)
    assert len(problems) == 1 and problems[0].startswith("t-clamp: the wrong code fails shown test"), problems


def test_a_hidden_call_in_the_docstring_is_refused(tmp_path: Path) -> None:
    task = tiny("t-clamp")
    task["buggy_code"] = (
        "def clamp(x, lo, hi):\n"
        '    """Return x limited to the closed interval [lo, hi].\n\n'
        "    >>> clamp(2, 0, 3)\n"
        "    2\n"
        '    """\n'
        "    return max(lo, min(x, lo))\n"
    )
    report = validate_suite(suite_of(tmp_path, task))
    assert not report.valid
    leaks = [p for p in report.problems if "appears in the prompts of" in p]
    assert leaks and all("'clamp(2, 0, 3)'" in p for p in leaks), report.problems
    assert "bugfix/v1" in leaks[0] and "judge/v2" in leaks[0]


def test_the_reference_code_shown_in_a_prompt_is_refused(tmp_path: Path) -> None:
    task = tiny("t-clamp")
    task["buggy_code"] = f"_HINT = '''{task['reference_code']}'''\n\n\n" + task["buggy_code"]
    problems = problems_of(tmp_path, task)
    assert problems == ["t-clamp: the task's reference code appears in the prompts of bugfix/v1, bugfix/v2, judge/v1, judge/v2"]


def test_eval_in_the_reference_is_refused(tmp_path: Path) -> None:
    task = tiny("t-clamp")
    task["reference_code"] = "def clamp(x, lo, hi):\n" + DOC + '    return eval("max(lo, min(x, hi))")\n'
    problems = problems_of(tmp_path, task)
    assert len(problems) == 1 and problems[0].startswith("t-clamp: reference_code trips block-class SAST rule(s) eval-exec"), problems


def test_a_flaky_reference_is_refused(tmp_path: Path) -> None:
    marker = tmp_path / "flaky.marker"
    task = tiny("t-clamp")
    task["reference_code"] = (
        "import os\n"
        f"_MARKER = {str(marker)!r}\n"
        "_SECOND_RUN = os.path.exists(_MARKER)\n"
        "open(_MARKER, 'a').close()\n\n\n"
        "def clamp(x, lo, hi):\n" + DOC + "    if _SECOND_RUN:\n        return lo\n    return max(lo, min(x, hi))\n"
    )
    problems = problems_of(tmp_path, task)
    assert marker.exists()
    assert any("flaky" in p for p in problems), problems
    assert any(p.startswith("t-clamp: the reference code fails") for p in problems), problems


def test_identical_twins_across_the_split_are_refused(tmp_path: Path) -> None:
    original = tiny("t-count-vowels")
    twin = {key: (value.replace("count_vowels", "tally_vowels") if isinstance(value, str) else value) for key, value in original.items()}
    for key in ("shown_tests", "hidden_tests"):
        twin[key] = [dict(c, call=c["call"].replace("count_vowels", "tally_vowels")) for c in original[key]]
    twin["id"] = "t-tally-vowels"
    suite = suite_of(tmp_path, original, twin)
    assert {tasks.split_of(t, suite.heldout_pct) for t in suite.tasks} == {DEV, HELDOUT}
    report = validate_suite(suite)
    assert report.problems == [
        "tasks t-count-vowels (dev) and t-tally-vowels (heldout) have identical normalised buggy code: "
        "twins on both sides of the split leak the held-out answer; give them one group"
    ]
    assert not any("possible twins" in w for w in report.warnings)
    # The same pair in one group is fine: a group keeps twins on one side.
    grouped = suite_of(tmp_path, dict(original, group="vowels"), dict(twin, group="vowels"))
    assert validate_suite(grouped, execute=False).valid


def test_near_twins_across_the_split_are_a_warning(tmp_path: Path) -> None:
    original = tiny("t-count-vowels")
    near = copy.deepcopy(original)
    near["id"] = "t-vowel-sum"
    for key in ("buggy_code", "wrong_code"):
        near[key] = near[key].replace('c in "aeiou")', 'c in "aeiou" and c)')
    suite = suite_of(tmp_path, original, near)
    pairs = tasks.find_twins(suite)
    assert [(a, b) for a, b, ratio in pairs if ratio < 1.0] == [("t-count-vowels", "t-vowel-sum")]
    report = validate_suite(suite)
    assert report.valid, report.problems
    assert any(w.startswith("possible twins across the split: t-count-vowels (dev) and t-vowel-sum (heldout)") for w in report.warnings)


def test_a_task_that_does_not_parse_is_reported_and_not_run(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    ran: list[str] = []
    real = verdicts.exec_verdict

    def spy(task: Any, code: str, **kwargs: Any) -> Any:
        ran.append(task.id)
        return real(task, code, **kwargs)

    monkeypatch.setattr(validation.verdicts, "exec_verdict", spy)
    broken = tiny("t-clamp")
    broken["reference_code"] = "def clamp(x, lo, hi)\n    return x\n"
    report = validate_suite(suite_of(tmp_path, broken, tiny("t-parse-port")))
    assert not report.valid and any("t-clamp" in p for p in report.problems)
    assert "t-clamp: not executed: fix its structure problems first" in report.warnings
    assert set(ran) == {"t-parse-port"}


def test_a_sandbox_that_does_not_start_is_a_problem_of_its_own(tiny_suite_dir: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    def broken(task: Any, code: str, **kwargs: Any) -> verdicts.ExecVerdict:
        return verdicts.ExecVerdict(passed=False, shown_passed=False, hidden_passed=False, infra_error="no interpreter")

    monkeypatch.setattr(validation.verdicts, "exec_verdict", broken)
    report = validate_suite(tasks.load_suite(tiny_suite_dir))
    assert report.problems == ["the sandbox did not run (no interpreter): 4 task(s) were not checked, e.g. t-clamp"]


def test_the_report_round_trips_as_json() -> None:
    report = ValidationReport(suite="s", sha256="0" * 64, tasks=0, splits={DEV: 0, HELDOUT: 0}, executed=False, warnings=["w"])
    assert report.to_json() == {"suite": "s", "sha256": "0" * 64, "tasks": 0, "splits": {DEV: 0, HELDOUT: 0}, "executed": False,
                                "problems": [], "warnings": ["w"], "valid": True}
