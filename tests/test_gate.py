"""gate: every gate at its boundaries, the tie rule, the McNemar guards, floors, the new-SAST rule,
refusals, the warnings and the report's shape.

Arms are built by hand (make_arm) from ItemResult rows or from a metrics dict; comparisons either
come from metrics.compare on real rows or are written out (make_comparison) to pin a boundary.
"""

from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any

import pytest

import llm_eval_harness
from llm_eval_harness import gate
from llm_eval_harness.gate import (
    DEFAULT_FLOORS,
    EPS,
    HELDOUT_BUDGET,
    REPORT_SCHEMA,
    RULE,
    SMALL_SUITE,
    Decision,
    Gate,
    GateConfig,
    GateRefused,
    build_report,
    check_comparable,
    decide,
    new_sast_violations,
    report_warnings,
)
from llm_eval_harness.metrics import PRIMARY, Comparison, arm_metrics, compare, mcnemar_exact, min_significant_losses
from llm_eval_harness.records import ArmRun, ItemResult
from llm_eval_harness.util import sha256_text

SUITE_SHA = "a" * 64
GATE_ORDER = ["schema_validity", "think_validity", "new_sast_violations", "primary", "regression_test"]


def bug_row(task: str, correct: bool, *, block: tuple[str, ...] = (), flag: tuple[str, ...] = (), tokens: int | None = None) -> ItemResult:
    return ItemResult(
        item_id=f"bugfix:{task}", feature="bugfix", task_id=task, cluster=task, split="heldout", correct=correct, schema_valid=True,
        lenient_valid=True, think_present=False, think_valid=True, think_words=0, completion_tokens=tokens,
        shown_passed=correct, hidden_passed=correct, sast_block=block, sast_flag=flag,
    )  # fmt: skip


def judge_rows(task: str, on_reference: bool, on_wrong: bool) -> list[ItemResult]:
    rows = []
    for variant, predicted in (("reference", on_reference), ("wrong", on_wrong)):
        label = variant == "reference"
        rows.append(ItemResult(
            item_id=f"judge:{task}:{variant}", feature="judge", task_id=task, cluster=task, split="heldout", correct=predicted is label,
            schema_valid=True, lenient_valid=True, think_present=False, think_valid=True, think_words=0, label=label, predicted=predicted,
        ))  # fmt: skip
    return rows


def make_arm(name: str = "candidate", *, feature: str = "bugfix", results: list[ItemResult] | None = None, metrics: dict[str, Any] | None = None,
             synthetic: bool = False, prompt: str | None = None, notes: tuple[str, ...] = (), split: str = "heldout",
             suite: str = SUITE_SHA, source: str | None = None) -> ArmRun:  # fmt: skip
    rows = list(results or [])
    return ArmRun(
        name=name, feature=feature, split=split, source=source or f"cassettes/{name}", fingerprint=sha256_text(name), model="model-x",
        prompt=prompt or f"{feature}/v2", prompt_sha256="0" * 64, synthetic=synthetic, suite_sha256=suite, recorded_suite_sha256=suite,
        results=rows, metrics=metrics if metrics is not None else arm_metrics(feature, rows, resamples=200), notes=list(notes),
    )  # fmt: skip


def good_metrics(**overrides: Any) -> dict[str, Any]:
    return {"schema_validity": 1.0, "think_validity": 1.0, "resolved_rate": 0.8, "primary": 0.8, "items": 100, **overrides}


def make_comparison(delta: float | None, *, gains: int = 0, losses: int = 0, n: int = 100, feature: str = "bugfix",
                    mde80: float | None = 0.14) -> Comparison:  # fmt: skip
    return Comparison(
        feature=feature, metric=PRIMARY[feature], n=n, clusters=n, gains=gains, losses=losses,
        candidate=None if delta is None else 0.5 + delta, baseline=0.5, delta=delta,
        ci95=None if delta is None else (delta - 0.1, delta + 0.1), se=0.05, mde80=mde80, mcnemar_p=mcnemar_exact(losses, gains),
        min_significant_losses=min_significant_losses(0.05), alpha=0.05, resamples=2000, seed=0,
    )  # fmt: skip


def gate_of(decision: Decision, name: str) -> Gate:
    return next(g for g in decision.gates if g.gate == name)


def decide_with(delta: float | None, *, config: GateConfig = GateConfig(), metrics: dict[str, Any] | None = None, **comparison: Any) -> Decision:
    candidate = make_arm(metrics=metrics or good_metrics())
    baseline = make_arm("baseline", metrics=good_metrics())
    return decide(candidate, baseline, make_comparison(delta, **comparison), config)


# ---- the configuration ---------------------------------------------------------------------------


def test_config_defaults_and_json() -> None:
    config = GateConfig()
    config.validate()
    assert config.to_json() == {
        "schema_min": 0.95, "think_min": 0.90, "max_new_sast": 0, "margin": 0.0, "tie": "pass", "alpha": 0.05,
        "require_gain": False, "floor": None,
    }  # fmt: skip


@pytest.mark.parametrize(
    "changes",
    [
        {"schema_min": 1.5}, {"think_min": -0.1}, {"max_new_sast": -1}, {"max_new_sast": 0.5}, {"margin": -0.01},
        {"margin": float("nan")}, {"tie": "maybe"}, {"alpha": 0.0}, {"alpha": 1.0}, {"floor": 1.2}, {"floor": -0.1},
        {"require_gain": "yes"},
    ],
)  # fmt: skip
def test_config_validate_refuses(changes: dict[str, Any]) -> None:
    with pytest.raises(ValueError):
        GateConfig(**changes).validate()
    with pytest.raises(ValueError):
        decide_with(0.0, config=GateConfig(**changes))


# ---- the gates, in order -------------------------------------------------------------------------


def test_gate_order() -> None:
    assert [g.gate for g in decide_with(0.0).gates] == GATE_ORDER
    assert [g.gate for g in decide_with(0.0, config=GateConfig(require_gain=True)).gates] == GATE_ORDER + ["gain_test"]
    no_baseline = decide(make_arm(metrics=good_metrics()), None, None, GateConfig())
    assert [g.gate for g in no_baseline.gates] == GATE_ORDER


@pytest.mark.parametrize(
    "key, value, passed",
    [
        ("schema_validity", 0.95, True),
        ("schema_validity", 0.9499, False),
        ("schema_validity", 0.95 - EPS / 2, True),  # float noise never decides
        ("schema_validity", None, False),
        ("think_validity", 0.90, True),
        ("think_validity", 0.8999, False),
        ("think_validity", 1.0, True),
    ],
)
def test_validity_gate_boundaries(key: str, value: float | None, passed: bool) -> None:
    decision = decide_with(0.0, metrics=good_metrics(**{key: value}))
    validity = gate_of(decision, key)
    assert validity.passed is passed and validity.op == ">="
    assert validity.bound == (0.95 if key == "schema_validity" else 0.90)
    assert decision.promote is passed and decision.gates_passed is passed


@pytest.mark.parametrize(
    "delta, tie, margin, passed",
    [
        (0.0, "pass", 0.0, True),
        (0.0, "fail", 0.0, False),
        (-1e-12, "pass", 0.0, True),  # a tie within the tolerance
        (-1e-12, "fail", 0.0, False),
        (1e-12, "fail", 0.0, False),
        (0.01, "fail", 0.0, True),
        (-0.01, "pass", 0.0, False),
        (-0.05, "pass", 0.05, True),
        (-0.05, "fail", 0.05, False),
        (-0.0499, "fail", 0.05, True),
        (-0.0501, "pass", 0.05, False),
        (None, "pass", 0.0, False),  # nothing scorable never passes
    ],
)
def test_primary_tie_rule_and_margin(delta: float | None, tie: str, margin: float, passed: bool) -> None:
    decision = decide_with(delta, config=GateConfig(tie=tie, margin=margin))
    primary = gate_of(decision, "primary")
    assert primary.passed is passed
    assert primary.op == (">" if tie == "fail" else ">=")
    assert primary.bound == -margin and json.dumps(primary.bound) != "-0.0"
    assert primary.detail["bound_source"] == "baseline" and primary.detail["metric"] == "resolved_rate"
    assert primary.detail["tie"] == tie and primary.detail["margin"] == margin
    assert primary.detail["at_bound"] is (delta is not None and abs(delta + margin) <= EPS)
    assert decision.promote is passed


def _thousand_items(losses: int, gains: int) -> tuple[list[ItemResult], list[ItemResult]]:
    """1,000 paired bugfix items: `losses` only the baseline got right, `gains` only the candidate,
    500 both, the rest neither."""
    outcomes = [(False, True)] * losses + [(True, False)] * gains + [(True, True)] * 500
    outcomes += [(False, False)] * (1000 - len(outcomes))
    candidate = [bug_row(f"t{k:04d}", c) for k, (c, _) in enumerate(outcomes)]
    baseline = [bug_row(f"t{k:04d}", b) for k, (_, b) in enumerate(outcomes)]
    return candidate, baseline


def test_regression_test_fires_inside_the_margin() -> None:
    cand_rows, base_rows = _thousand_items(losses=60, gains=10)
    comparison = compare("bugfix", cand_rows, base_rows, split="heldout", resamples=200)
    assert comparison.delta == pytest.approx(-0.05)
    candidate, baseline = make_arm(results=cand_rows), make_arm("baseline", results=base_rows)
    decision = decide(candidate, baseline, comparison, GateConfig(margin=0.06))
    assert gate_of(decision, "primary").passed is True  # -0.05 >= -0.06
    regression = gate_of(decision, "regression_test")
    assert regression.passed is False and regression.enforced is True
    assert regression.value == comparison.mcnemar_p and regression.value < 1e-6
    assert (regression.detail["losses"], regression.detail["gains"], regression.op, regression.bound) == (60, 10, "not", 0.05)
    assert decision.promote is False


def test_regression_test_passes_a_loss_that_is_not_significant() -> None:
    cand_rows, base_rows = _thousand_items(losses=6, gains=3)
    comparison = compare("bugfix", cand_rows, base_rows, split="heldout", resamples=200)
    decision = decide(make_arm(results=cand_rows), make_arm("baseline", results=base_rows), comparison, GateConfig(margin=0.06))
    assert gate_of(decision, "regression_test").passed is True
    assert decision.promote is True


@pytest.mark.parametrize("gains, losses, passed", [(8, 0, True), (3, 0, False), (0, 0, False), (10, 10, False), (20, 2, True), (2, 20, False)])
def test_gain_test(gains: int, losses: int, passed: bool) -> None:
    delta = (gains - losses) / 100
    decision = decide_with(delta, config=GateConfig(require_gain=True, margin=1.0), gains=gains, losses=losses)
    gain = gate_of(decision, "gain_test")
    assert gain.passed is passed and gain.enforced is True
    assert (gain.op, gain.bound, gain.value) == ("<", 0.05, mcnemar_exact(losses, gains))
    assert (gain.detail["gains"], gain.detail["losses"]) == (gains, losses)
    assert decision.promote is (passed and gate_of(decision, "regression_test").passed)


def test_gain_test_without_a_baseline_fails() -> None:
    decision = decide(make_arm(metrics=good_metrics()), None, None, GateConfig(require_gain=True))
    gain = gate_of(decision, "gain_test")
    assert gain.passed is False and gain.enforced is True and gain.value is None
    assert gain.detail["reason"] == "no baseline: no gain can be shown"
    assert decision.promote is False


@pytest.mark.parametrize(
    "feature, value, config, passed, bound",
    [
        ("bugfix", 0.5, GateConfig(), True, DEFAULT_FLOORS["bugfix"]),
        ("bugfix", 0.49, GateConfig(), False, 0.5),
        ("bugfix", 0.5, GateConfig(tie="fail"), False, 0.5),
        ("bugfix", 0.85, GateConfig(floor=0.9), False, 0.9),
        ("bugfix", None, GateConfig(), False, 0.5),
        ("judge", 0.69, GateConfig(), False, DEFAULT_FLOORS["judge"]),
        ("judge", 0.7, GateConfig(), True, 0.7),
    ],
)
def test_floors_without_a_baseline(feature: str, value: float | None, config: GateConfig, passed: bool, bound: float) -> None:
    metric = PRIMARY[feature]
    candidate = make_arm(feature=feature, metrics={"schema_validity": 1.0, "think_validity": 1.0, metric: value, "primary": value, "items": 100})
    decision = decide(candidate, None, None, config)
    primary = gate_of(decision, "primary")
    assert primary.passed is passed
    assert primary.bound == bound and primary.detail["bound_source"] == "floor" and primary.value == value
    regression = gate_of(decision, "regression_test")
    assert regression.enforced is False and regression.passed is True and regression.detail["applicable"] is False
    assert decision.promote is passed


def test_a_baseline_bound_says_so() -> None:
    assert gate_of(decide_with(0.0), "primary").detail["bound_source"] == "baseline"


# ---- new SAST violations ---------------------------------------------------------------------------


def test_new_sast_violations_semantics() -> None:
    candidate = [
        bug_row("a", False, block=("eval-exec",)),
        bug_row("b", False, block=("eval-exec",)),
        bug_row("c", True, flag=("file-write",)),
        bug_row("d", False, block=("eval-exec", "os-command")),
    ]
    baseline = [
        bug_row("a", False, block=("eval-exec",)),  # the same rule on the same item: not new
        bug_row("b", True),  # eval-exec on another item of the baseline does not excuse b
        bug_row("c", True),
        bug_row("d", False, block=("os-command",)),
    ]
    assert new_sast_violations(candidate, baseline) == [{"item": "bugfix:b", "rule": "eval-exec"}, {"item": "bugfix:d", "rule": "eval-exec"}]
    assert new_sast_violations(candidate, None) == [
        {"item": "bugfix:a", "rule": "eval-exec"}, {"item": "bugfix:b", "rule": "eval-exec"},
        {"item": "bugfix:d", "rule": "eval-exec"}, {"item": "bugfix:d", "rule": "os-command"},
    ]  # fmt: skip
    assert new_sast_violations([bug_row("x", True, flag=("network", "exit-call"))], []) == []  # flags never count


def test_sast_gate_in_decide() -> None:
    cand_rows = [bug_row("a", True, block=("shell-true",)), bug_row("b", True)]
    base_rows = [bug_row("a", True), bug_row("b", True)]
    candidate, baseline = make_arm(results=cand_rows), make_arm("baseline", results=base_rows)
    comparison = compare("bugfix", cand_rows, base_rows, split="heldout", resamples=50)
    strict = gate_of(decide(candidate, baseline, comparison, GateConfig()), "new_sast_violations")
    assert (strict.value, strict.op, strict.bound, strict.passed, strict.enforced) == (1, "<=", 0, False, True)
    assert strict.detail == {"applicable": True, "against": "baseline", "violations": [{"item": "bugfix:a", "rule": "shell-true"}]}
    lenient = gate_of(decide(candidate, baseline, comparison, GateConfig(max_new_sast=1)), "new_sast_violations")
    assert lenient.passed is True


def test_the_judge_sast_gate_does_not_apply() -> None:
    rows = [r for k in range(10) for r in judge_rows(f"t{k}", True, False)]
    candidate, baseline = make_arm(feature="judge", results=rows), make_arm("baseline", feature="judge", results=rows)
    decision = decide(candidate, baseline, compare("judge", rows, rows, split="heldout", resamples=50), GateConfig())
    sast = gate_of(decision, "new_sast_violations")
    assert sast.detail["applicable"] is False
    assert sast.passed is True and sast.enforced is False and sast.value is None
    assert decision.promote is True


# ---- refusals and inconsistent inputs -------------------------------------------------------------


def test_check_comparable_refusals() -> None:
    rows = [bug_row("a", True), bug_row("b", False)]
    candidate = make_arm(results=rows)
    check_comparable(candidate, None)
    check_comparable(candidate, make_arm("baseline", results=list(reversed(rows))))
    cases = {
        "scored as bugfix and the baseline as judge": make_arm("baseline", feature="judge", results=[], metrics={}),
        "split": make_arm("baseline", results=rows, split="dev"),
        "different suites": make_arm("baseline", results=rows, suite="b" * 64),
        "1 only in the candidate (bugfix:b)": make_arm("baseline", results=rows[:1]),
        "1 only in the baseline (bugfix:c)": make_arm("baseline", results=rows + [bug_row("c", True)]),
        "lists item bugfix:a twice": make_arm("baseline", results=rows + [bug_row("a", False)]),
    }
    for expected, baseline in cases.items():
        with pytest.raises(GateRefused, match=expected.replace("(", r"\(").replace(")", r"\)")):
            check_comparable(candidate, baseline)


def test_decide_refuses_inconsistent_inputs() -> None:
    candidate, baseline = make_arm(metrics=good_metrics()), make_arm("baseline", metrics=good_metrics())
    with pytest.raises(ValueError):
        decide(candidate, None, make_comparison(0.0), GateConfig())
    with pytest.raises(ValueError):
        decide(candidate, baseline, None, GateConfig())
    with pytest.raises(ValueError):
        decide(candidate, baseline, make_comparison(0.0, feature="judge"), GateConfig())
    with pytest.raises(ValueError):
        decide(make_arm(feature="summarise", metrics=good_metrics()), None, None, GateConfig())


def test_decide_is_pure() -> None:
    cand_rows = [bug_row("a", True, block=("eval-exec",)), bug_row("b", False)]
    base_rows = [bug_row("a", True), bug_row("b", True)]
    candidate, baseline = make_arm(results=cand_rows), make_arm("baseline", results=base_rows)
    comparison = compare("bugfix", cand_rows, base_rows, split="heldout", resamples=50)
    before = copy.deepcopy((candidate, baseline, comparison))
    first = decide(candidate, baseline, comparison, GateConfig())
    assert (candidate, baseline, comparison) == before
    assert decide(candidate, baseline, comparison, GateConfig()) == first


def test_promote_needs_every_enforced_gate() -> None:
    decision = decide_with(0.2, metrics=good_metrics(think_validity=0.5))
    assert gate_of(decision, "primary").passed is True
    assert decision.gates_passed is False and decision.promote is False


def test_gate_to_json() -> None:
    assert Gate("g", 1, "<=", 0, False, detail={"applicable": True}).to_json() == {
        "gate": "g", "value": 1, "op": "<=", "bound": 0, "passed": False, "enforced": True, "applicable": True,
    }  # fmt: skip
    with pytest.raises(ValueError):
        Gate("g", 1, "<=", 0, False, detail={"passed": True}).to_json()


# ---- warnings ------------------------------------------------------------------------------------


def test_small_suite_warning_text() -> None:
    warnings = report_warnings(make_arm(metrics=good_metrics()), make_arm("baseline", metrics=good_metrics()),
                               make_comparison(0.0, n=16, gains=2, losses=1, mde80=0.3136), GateConfig())  # fmt: skip
    assert warnings == [
        "16 paired items: with no gains, at least 6 losses are needed before McNemar calls a regression significant at alpha 0.05; mde80 0.31."
    ]


def test_warnings_in_the_documented_order() -> None:
    candidate_metrics = {
        "schema_validity": 1.0, "think_validity": 1.0, "balanced_accuracy": 0.55, "primary": 0.55, "items": 16,
        "balanced_accuracy_ci95": [0.4, 0.7], "approve_rate": 0.95, "completion_tokens_mean": 320.0,
    }  # fmt: skip
    baseline_metrics = {**candidate_metrics, "balanced_accuracy_ci95": [0.6, 0.95], "approve_rate": 0.5, "completion_tokens_mean": 200.0}
    candidate = make_arm(feature="judge", metrics=candidate_metrics, synthetic=True, prompt="judge/v1", notes=("recorded on suite version 1",))
    baseline = make_arm("baseline", feature="judge", metrics=baseline_metrics, synthetic=True)
    comparison = make_comparison(0.0, n=16, feature="judge", mde80=None)
    warnings = report_warnings(candidate, baseline, comparison, GateConfig(), current_prompt="judge/v2", heldout_openings=HELDOUT_BUDGET + 2)
    starts = [
        "The candidate and baseline arms are synthetic",
        "16 paired items: with no gains, at least 6 losses",
        "No discordant items: on all 16 items",
        "The candidate was recorded with prompt judge/v1, not the current judge/v2.",
        "candidate: recorded on suite version 1",
        "The held-out split has been used to judge 12 distinct candidates",
        "The candidate judge approves 0.950",
        "The candidate judge's balanced accuracy 0.550 has a 95% interval [0.400, 0.700] that includes 0.5",
        "Cost: the candidate's mean completion tokens per item (320) are 1.6 times the baseline's (200).",
    ]
    assert len(warnings) == len(starts), warnings
    for warning, start in zip(warnings, starts):
        assert warning.startswith(start), (warning, start)
    assert "mde80 n/a (no discordant items)" in warnings[1]


def test_small_suite_warning_when_every_item_is_a_gain() -> None:
    # Every item discordant the same way: each bootstrap draw gives delta 1.0, so se is 0 and mde80
    # is unknown, although there are discordant items. The warning must not say there are none.
    wins = [bug_row(f"t{k:03d}", True) for k in range(12)]
    misses = [bug_row(f"t{k:03d}", False) for k in range(12)]
    comparison = compare("bugfix", wins, misses, split="heldout", resamples=200)
    assert (comparison.gains, comparison.losses, comparison.mde80) == (12, 0, None)
    warnings = report_warnings(make_arm(results=wins), make_arm("baseline", results=misses), comparison, GateConfig())
    assert any("mde80 n/a (every bootstrap draw gave the same delta)" in w for w in warnings), warnings
    assert not any("no discordant items" in w.lower() for w in warnings), warnings


def test_warnings_that_stay_quiet() -> None:
    rows = [bug_row(f"t{k:03d}", k % 3 == 0, tokens=100) for k in range(60)]
    candidate, baseline = make_arm(results=rows), make_arm("baseline", results=rows)
    comparison = compare("bugfix", rows, rows, split="heldout", resamples=50)
    quiet = report_warnings(candidate, baseline, make_comparison(0.0, n=60, gains=3, losses=3), GateConfig(),
                            current_prompt="bugfix/v2", heldout_openings=HELDOUT_BUDGET - 1)  # fmt: skip
    assert quiet == []
    assert any(w.startswith("No discordant items") for w in report_warnings(candidate, baseline, comparison, GateConfig()))
    dev = make_arm(results=rows, split="dev")
    assert report_warnings(dev, None, None, GateConfig(), heldout_openings=50) == []  # the budget is the held-out split's
    assert SMALL_SUITE == 50


def test_judge_warnings_low_approval_and_worse_than_chance() -> None:
    low = {"schema_validity": 1.0, "think_validity": 1.0, "balanced_accuracy": 0.3, "primary": 0.3, "items": 80,
           "balanced_accuracy_ci95": [0.2, 0.45], "approve_rate": 0.05}  # fmt: skip
    warnings = report_warnings(make_arm(feature="judge", metrics=low), None, None, GateConfig())
    assert any("approves only 0.050" in w and "rejects nearly every patch" in w for w in warnings)
    assert any("below 0.5: it is worse than chance" in w for w in warnings)


def test_cost_and_no_baseline_warnings() -> None:
    cheap = report_warnings(make_arm(metrics=good_metrics(completion_tokens_mean=290.0)), make_arm("baseline", metrics=good_metrics(completion_tokens_mean=200.0)),
                            make_comparison(0.0, n=100, gains=1, losses=1), GateConfig())  # fmt: skip
    assert cheap == []
    rows = [bug_row(f"t{k}", k < 3) for k in range(4)]
    alone = report_warnings(make_arm(results=rows), None, None, GateConfig())
    assert alone == ["4 items and no baseline: the floor is compared with a point estimate whose 95% interval is [0.301, 0.954]."]


# ---- the report ----------------------------------------------------------------------------------


SUITE = {"name": "tiny", "sha256": "c" * 64, "split_sha256": "d" * 64, "tasks": 4, "items": 4, "heldout_pct": 50, "synthetic": True}
PROVENANCE = {"sast": {"ruleset": "sast/1"}, "exec_rule": "values/1", "prompt_current": "bugfix/v2"}


def _report(*, with_baseline: bool = True, candidate_source: str | None = None, suite: dict[str, Any] | None = None) -> dict[str, Any]:
    cand_rows = [bug_row("a", True), bug_row("b", True), bug_row("c", False, block=("eval-exec",)), bug_row("d", False)]
    base_rows = [bug_row("a", True), bug_row("b", False), bug_row("c", False), bug_row("d", True)]
    candidate = make_arm(results=cand_rows, synthetic=True, source=candidate_source)
    baseline = make_arm("baseline", results=base_rows, synthetic=True) if with_baseline else None
    comparison = compare("bugfix", cand_rows, base_rows, split="heldout", resamples=100) if with_baseline else None
    config = GateConfig()
    decision = decide(candidate, baseline, comparison, config)
    warnings = report_warnings(candidate, baseline, comparison, config, current_prompt="bugfix/v2")
    return build_report(feature="bugfix", split="heldout", suite=suite if suite is not None else SUITE, candidate=candidate, baseline=baseline,
                        comparison=comparison, decision=decision, config=config, warnings=warnings, provenance=PROVENANCE)  # fmt: skip


def _strings(value: Any):
    if isinstance(value, str):
        yield value
    elif isinstance(value, dict):
        for key, item in value.items():
            yield key
            yield from _strings(item)
    elif isinstance(value, list):
        for item in value:
            yield from _strings(item)


def test_report_has_every_key() -> None:
    report = _report()
    assert set(report) == {
        "schema", "rule", "feature", "split", "suite", "candidate", "baseline", "comparison", "gates", "gates_passed", "promote",
        "config", "warnings", "provenance",
    }  # fmt: skip
    assert (report["schema"], report["rule"], report["feature"], report["split"]) == (REPORT_SCHEMA, RULE, "bugfix", "heldout")
    assert report["suite"] == SUITE
    assert set(report["candidate"]) == {"name", "source", "fingerprint", "model", "prompt", "prompt_sha256", "synthetic",
                                        "recorded_suite_sha256", "metrics", "notes"}  # fmt: skip
    assert set(report["provenance"]) == {"harness_version", "sast", "exec_rule", "prompt_current"}
    assert report["provenance"]["harness_version"] == llm_eval_harness.__version__
    assert [g["gate"] for g in report["gates"]] == GATE_ORDER
    for entry in report["gates"]:
        assert {"gate", "value", "op", "bound", "passed", "enforced"} <= set(entry)
    assert report["gates_passed"] is report["promote"] is False  # the candidate added eval-exec on c
    assert report["comparison"]["n"] == 4 and report["config"] == GateConfig().to_json()
    assert report["warnings"] and report["warnings"][0].startswith("The candidate and baseline arms are synthetic")


def test_report_without_a_baseline_and_with_a_partial_suite() -> None:
    report = _report(with_baseline=False, suite={"name": "tiny", "extra": 1})
    assert report["baseline"] is None and report["comparison"] is None
    assert set(report["suite"]) == set(SUITE) | {"extra"}
    assert report["suite"]["sha256"] is None and report["suite"]["extra"] == 1


def test_report_is_deterministic_and_strict_json() -> None:
    first, second = _report(), _report()
    assert first == second
    text = json.dumps(first, sort_keys=True, allow_nan=False)
    assert text == json.dumps(second, sort_keys=True, allow_nan=False)
    assert json.loads(text) == first


def test_report_holds_no_absolute_path_or_time(tmp_path: Path) -> None:
    report = _report(candidate_source=str(tmp_path / "cassettes" / "candidate"))
    assert report["candidate"]["source"].endswith("candidate") and not report["candidate"]["source"].startswith("/")
    for text in _strings(report):
        assert not text.startswith("/"), text
        assert str(tmp_path) not in text
    assert not any(key in json.dumps(report) for key in ('"time"', '"timestamp"', '"host"', '"user"'))


def test_report_refuses_a_candidate_of_another_feature() -> None:
    candidate = make_arm(metrics=good_metrics())
    decision = decide(candidate, None, None, GateConfig())
    with pytest.raises(ValueError):
        build_report(feature="judge", split="heldout", suite=SUITE, candidate=candidate, baseline=None, comparison=None,
                     decision=decision, config=GateConfig(), warnings=[], provenance={})  # fmt: skip


def test_report_does_not_alias_the_arms() -> None:
    candidate = make_arm(metrics=good_metrics())
    decision = decide(candidate, None, None, GateConfig())
    report = build_report(feature="bugfix", split="heldout", suite=SUITE, candidate=candidate, baseline=None, comparison=None,
                          decision=decision, config=GateConfig(), warnings=[], provenance={})  # fmt: skip
    report["candidate"]["metrics"]["primary"] = 0.0
    assert candidate.metrics["primary"] == 0.8


def test_module_constants() -> None:
    assert gate.RULE == "noninferior/1" and gate.REPORT_SCHEMA == "llm_eval_harness.report/1"
    assert dict(DEFAULT_FLOORS) == {"bugfix": 0.5, "judge": 0.7}
    assert (HELDOUT_BUDGET, EPS) == (10, 1e-9)
