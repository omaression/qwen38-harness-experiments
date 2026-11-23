"""The checks a task suite must pass before anything is scored on it (`llm-eval tasks validate`).

    validate_suite(suite, execute=True) -> ValidationReport

PROBLEMS (the suite is not valid; CI fails):
  - tasks.check_structure(suite);
  - executed in the sandbox (verdicts.exec_verdict), unless execute is False:
      the buggy code fails at least one shown test (the shown test demonstrates the bug);
      the reference code passes every shown and hidden test, twice, with the same outcomes
        (a task whose result depends on hash order, time or chance is flaky);
      the wrong code passes every shown test and fails at least one hidden test (otherwise it
        is not the overfitted patch the judge's negative item claims it is);
      an infra_error is a problem of its own ("the sandbox did not run"), not a task failure;
      a task with structure problems is not run (a warning says so): its results would only
      restate them;
  - sast.blocking(sast.scan(code)) is empty for the buggy, reference and wrong code (a rule that
    fires on a correct fix would make the gate refuse correct answers);
  - leakage: for every task, the rendered prompts of every published template (bugfix and judge)
    contain no hidden test's call text, the bugfix prompts contain neither the reference nor
    the wrong code, and a judge prompt shows only the code under review (not the other one);
    one problem per task and leak, naming the templates;
  - two tasks in different splits whose normalised buggy code is identical (tasks.normalize_code).
WARNINGS (reported, not failing): tasks.find_twins pairs below identity; a split with fewer
than 10 tasks; a suite whose held-out share differs from heldout_pct by more than 15 points;
tasks not run because of their structure problems.

The report (ValidationReport.to_json, printed by the CLI):
  {"suite", "sha256", "tasks", "splits": {"dev", "heldout"}, "executed": bool,
   "problems": [...], "warnings": [...], "valid": bool}
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field, replace
from typing import Any

from . import prompts, sast, tasks, verdicts
from .features import get_feature
from .records import ALL, DEV, FEATURES, HELDOUT, Suite, Task
from .sandbox import DEFAULT_TIMEOUT_S, SandboxUnavailable

#: A split with fewer tasks than this gets a warning.
MIN_SPLIT_TASKS = 10
#: Percentage points the held-out share may differ from heldout_pct before a warning.
SHARE_TOLERANCE_PCT = 15
#: Sandbox runs in flight while validating.
WORKERS = 4
#: Failing tests spelt out per problem.
MAX_LISTED = 3


@dataclass
class ValidationReport:
    """validate_suite's result."""

    suite: str
    sha256: str
    tasks: int
    splits: dict[str, int]
    executed: bool
    problems: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    @property
    def valid(self) -> bool:
        return not self.problems

    def to_json(self) -> dict[str, Any]:
        return {
            "suite": self.suite,
            "sha256": self.sha256,
            "tasks": self.tasks,
            "splits": dict(self.splits),
            "executed": self.executed,
            "problems": list(self.problems),
            "warnings": list(self.warnings),
            "valid": self.valid,
        }


def validate_suite(suite: Suite, *, execute: bool = True, timeout_s: float = DEFAULT_TIMEOUT_S) -> ValidationReport:
    """Run the checks above. Never raises for a bad task: every finding is a problem."""
    counts = tasks.split_counts(suite)
    report = ValidationReport(
        suite=suite.name,
        sha256=suite.sha256,
        tasks=len(suite.tasks),
        splits={DEV: counts[DEV], HELDOUT: counts[HELDOUT]},
        executed=execute,
    )
    report.problems += _guarded("structure check", lambda: tasks.check_structure(suite))
    ordered = sorted(suite.tasks, key=lambda t: t.id)
    if execute:
        # A task whose structure is broken is not run: its results would only restate that.
        runnable = []
        for task in ordered:
            if _guarded("structure check", lambda: tasks.check_structure(replace(suite, tasks=(task,)))):
                report.warnings.append(f"{task.id}: not executed: fix its structure problems first")
            else:
                runnable.append(task)
        report.problems += _execution_problems(runnable, timeout_s)
    for task in ordered:
        report.problems += _sast_problems(task)
    report.problems += _leakage_problems(suite)
    twin_problems, twin_warnings = _twins(suite)
    report.problems += twin_problems
    report.warnings += twin_warnings
    report.warnings += _size_warnings(suite, counts)
    return report


# ---- the checks -----------------------------------------------------------------------------------


def _guarded(what: str, check: Any) -> list[str]:
    """check() -> problems; an exception becomes one problem (validation never raises for a task)."""
    try:
        return list(check())
    except Exception as exc:  # noqa: BLE001 - a bad task must be reported, not crash the report
        return [f"{what} failed: {type(exc).__name__}: {exc}"]


def _run(task: Task, code: str, timeout_s: float) -> verdicts.ExecVerdict | str:
    """exec_verdict, or the reason the sandbox gave no verdict (a string)."""
    try:
        verdict = verdicts.exec_verdict(task, code, timeout_s=timeout_s)
    except SandboxUnavailable as exc:
        return f"the sandbox is not available: {exc}"
    return verdict.infra_error if verdict.infra_error else verdict


def _failing(verdict: verdicts.ExecVerdict, kind: str | None = None) -> str:
    """The tests (of `kind`, or all) that did not pass, for a problem sentence."""
    failing = [r for r in verdict.results if r.outcome != "passed" and kind in (None, r.kind)]
    text = ", ".join(f"{r.kind} test {r.index} ({r.outcome}{': ' + r.detail if r.detail else ''})" for r in failing[:MAX_LISTED])
    if len(failing) > MAX_LISTED:
        text += f" and {len(failing) - MAX_LISTED} more"
    if not failing and verdict.problems:
        text = "; ".join(verdict.problems[:MAX_LISTED])
    return text or "no test result"


def _task_execution(task: Task, timeout_s: float) -> tuple[list[str], str | None]:
    """(problems, infra error) of one task's executed invariants."""
    runs: dict[str, verdicts.ExecVerdict] = {}
    plan = (("buggy", task.buggy_code), ("reference", task.reference_code), ("reference-again", task.reference_code), ("wrong", task.wrong_code))
    for name, code in plan:
        try:
            outcome = _run(task, code, timeout_s)
        except Exception as exc:  # noqa: BLE001
            return [f"{task.id}: its {name} code could not be executed: {type(exc).__name__}: {exc}"], None
        if isinstance(outcome, str):
            return [], outcome
        runs[name] = outcome
    problems: list[str] = []
    buggy, reference, again, wrong = runs["buggy"], runs["reference"], runs["reference-again"], runs["wrong"]
    if buggy.shown_passed:
        problems.append(f"{task.id}: the buggy code passes its shown tests, so they demonstrate no bug")
    for run in (reference, again):
        if not run.passed:
            problems.append(f"{task.id}: the reference code fails {_failing(run)}")
            break
    if [(r.index, r.kind, r.outcome) for r in reference.results] != [(r.index, r.kind, r.outcome) for r in again.results]:
        problems.append(f"{task.id}: the reference code gave different outcomes on two runs (flaky: the result depends on hash order, time or chance)")
    if not wrong.shown_passed:
        problems.append(f"{task.id}: the wrong code fails {_failing(wrong, 'shown')}: it must pass every shown test to be the overfitted patch the judge's negative item claims")
    if wrong.hidden_passed:
        problems.append(f"{task.id}: the wrong code passes every hidden test, so it is not wrong")
    return problems, None


def _execution_problems(runnable: list[Task], timeout_s: float) -> list[str]:
    if not runnable:
        return []
    with ThreadPoolExecutor(max_workers=min(WORKERS, len(runnable))) as pool:
        outcomes = list(pool.map(lambda task: _task_execution(task, timeout_s), runnable))
    problems: list[str] = []
    infra: dict[str, list[str]] = {}
    for task, (task_problems, infra_error) in zip(runnable, outcomes):
        problems += task_problems
        if infra_error:
            infra.setdefault(infra_error, []).append(task.id)
    for error, ids in infra.items():
        problems.append(f"the sandbox did not run ({error}): {len(ids)} task(s) were not checked, e.g. {ids[0]}")
    return problems


def _sast_problems(task: Task) -> list[str]:
    problems = []
    for name, code in (("buggy_code", task.buggy_code), ("reference_code", task.reference_code), ("wrong_code", task.wrong_code)):
        try:
            found = sast.blocking(sast.scan(code))
        except Exception as exc:  # noqa: BLE001
            problems.append(f"{task.id}: {name} could not be scanned: {type(exc).__name__}: {exc}")
            continue
        if found:
            rules = ", ".join(sorted({f.rule for f in found}))
            problems.append(f"{task.id}: {name} trips block-class SAST rule(s) {rules}; a rule that fires on a task's own code would make the gate refuse correct answers")
    return problems


def _leakage_problems(suite: Suite) -> list[str]:
    """Hidden test calls in any published prompt, and code a prompt must not show."""
    items = {}
    problems: list[str] = []
    for name in FEATURES:
        try:
            items[name] = get_feature(name).items(suite, ALL)
        except Exception as exc:  # noqa: BLE001
            problems.append(f"the {name} items could not be built: {type(exc).__name__}: {exc}")
    leaks: dict[tuple[str, str], set[str]] = {}
    for template in sorted(prompts.TEMPLATES.values(), key=lambda t: t.id):
        for item in items.get(template.feature, []):
            task = item.task
            try:
                text = "\n".join(m.content for m in prompts.render_messages(template, item))
            except Exception as exc:  # noqa: BLE001
                leaks.setdefault((task.id, f"a prompt that does not render ({type(exc).__name__}: {exc})"), set()).add(template.id)
                continue
            for check in task.hidden_tests:
                call = check.call.strip()
                if call and call in text:
                    leaks.setdefault((task.id, f"the hidden test call {call!r}"), set()).add(template.id)
            if template.feature == "bugfix":
                hidden_code = (("reference code", task.reference_code), ("wrong code", task.wrong_code))
            else:
                hidden_code = (("wrong code", task.wrong_code),) if item.variant == "reference" else (("reference code", task.reference_code),)
            for what, code in hidden_code:
                if code.strip() and code.strip() in text:
                    leaks.setdefault((task.id, f"the task's {what}"), set()).add(template.id)
    for (task_id, what), template_ids in sorted(leaks.items()):
        problems.append(f"{task_id}: {what} appears in the prompts of {', '.join(sorted(template_ids))}")
    return problems


def _twins(suite: Suite) -> tuple[list[str], list[str]]:
    """Identical normalised buggy code across the split is a problem; near twins are warnings."""
    problems: list[str] = []
    by_code: dict[tuple[str, ...], list[Task]] = {}
    for task in suite.tasks:
        try:
            key = tuple(tasks.normalize_code(task.buggy_code))
        except Exception:  # noqa: BLE001 - check_structure reports code that does not parse
            continue
        by_code.setdefault(key, []).append(task)
    identical: set[tuple[str, str]] = set()
    for group in by_code.values():
        dev = sorted(t.id for t in group if tasks.split_of(t, suite.heldout_pct) == DEV)
        held = sorted(t.id for t in group if tasks.split_of(t, suite.heldout_pct) == HELDOUT)
        if dev and held:
            identical.update((d, h) for d in dev for h in held)
            problems.append(
                f"tasks {', '.join(dev)} (dev) and {', '.join(held)} (heldout) have identical normalised buggy code: "
                "twins on both sides of the split leak the held-out answer; give them one group"
            )
    warnings: list[str] = []
    try:
        pairs = tasks.find_twins(suite)
    except Exception as exc:  # noqa: BLE001
        pairs = []
        warnings.append(f"twin detection failed: {type(exc).__name__}: {exc}")
    for dev_id, held_id, ratio in pairs:
        if (dev_id, held_id) in identical:
            continue
        warnings.append(
            f"possible twins across the split: {dev_id} (dev) and {held_id} (heldout), similarity {ratio:.2f}; "
            "if they are the same problem, give them one group"
        )
    return problems, warnings


def _size_warnings(suite: Suite, counts: dict[str, int]) -> list[str]:
    warnings = []
    for split in (DEV, HELDOUT):
        if counts[split] < MIN_SPLIT_TASKS:
            warnings.append(f"the {split} split holds {counts[split]} task(s), fewer than {MIN_SPLIT_TASKS}: it can show only large changes")
    total = counts[DEV] + counts[HELDOUT]
    if total:
        share = 100.0 * counts[HELDOUT] / total
        if abs(share - suite.heldout_pct) > SHARE_TOLERANCE_PCT:
            warnings.append(f"the held-out share is {share:.0f}% of the tasks, against heldout_pct {suite.heldout_pct}: the hash split is uneven on a suite this small")
    return warnings
