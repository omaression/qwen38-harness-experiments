"""features.py: the items of each feature and how one reply is scored.

Bug-fix replies are built from the tiny tasks' own code (conftest.TINY_TASKS): the reference
code resolves, the wrong code is an overfitted patch, and a patch reaching for eval() is blocked
before it runs. Judge replies are plain verdicts.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from conftest import TINY_TASKS, write_suite
from llm_eval_harness import features, tasks, verdicts
from llm_eval_harness.features import FEATURES, BugfixFeature, Feature, JudgeFeature, ScoringError, get_feature
from llm_eval_harness.records import ALL, DEV, HELDOUT, ChatResponse, ItemResult, Suite

TIMEOUT = 10.0
CLAMP_EVAL = (
    "def clamp(x, lo, hi):\n"
    '    """Return x limited to the closed interval [lo, hi]."""\n'
    '    return eval("max(lo, min(x, hi))")\n'
)


@pytest.fixture
def suite(tiny_suite_dir: Path) -> Suite:
    return tasks.load_suite(tiny_suite_dir)


def item_of(suite: Suite, item_id: str):
    feature = get_feature(item_id.split(":")[0])
    return {i.id: i for i in feature.items(suite, ALL)}[item_id]


def fix_reply(code: str, *, think: str | None = None, finish: str | None = "stop", **usage: int) -> ChatResponse:
    content = json.dumps({"explanation": "Fixed the bug.", "patched_code": code})
    if think is not None:
        content = f"<think>{think}</think>\n{content}"
    return ChatResponse(content=content, finish_reason=finish, usage=usage, latency_s=0.25)


def verdict_reply(verdict: object, **extra: object) -> ChatResponse:
    return ChatResponse(content=json.dumps({"verdict": verdict, **extra}), finish_reason="stop")


def no_execution(*args: object, **kwargs: object) -> None:
    raise AssertionError("this reply must not be executed")


# ---- items --------------------------------------------------------------------------------------


def test_bugfix_items_are_one_per_task_sorted_with_their_split_and_cluster(suite: Suite) -> None:
    items = BugfixFeature().items(suite, ALL)
    assert [i.id for i in items] == sorted(f"bugfix:{t['id']}" for t in TINY_TASKS)
    for item in items:
        assert item.feature == "bugfix" and item.task.id == item.id.split(":", 1)[1]
        assert item.cluster == tasks.normalize_key(item.task.id)
        assert item.split == tasks.split_of(item.task, suite.heldout_pct)
        assert item.variant is None and item.label is None and item.patched_code is None
    by_split = {split: [i.id for i in BugfixFeature().items(suite, split)] for split in (DEV, HELDOUT)}
    assert by_split[HELDOUT] == ["bugfix:t-clamp"]
    assert by_split[DEV] == ["bugfix:t-count-vowels", "bugfix:t-parse-port", "bugfix:t-running-total"]


def test_judge_items_are_a_reference_and_a_wrong_patch_per_task(suite: Suite) -> None:
    items = JudgeFeature().items(suite, ALL)
    assert [i.id for i in items] == sorted(f"judge:{t['id']}:{v}" for t in TINY_TASKS for v in ("reference", "wrong"))
    for item in items:
        task = item.task
        if item.variant == "reference":
            assert item.label is True and item.patched_code == task.reference_code
        else:
            assert item.variant == "wrong" and item.label is False and item.patched_code == task.wrong_code
        assert item.cluster == tasks.normalize_key(task.id)
    assert [i.id for i in JudgeFeature().items(suite, HELDOUT)] == ["judge:t-clamp:reference", "judge:t-clamp:wrong"]


def test_a_group_is_the_cluster_of_its_tasks(tmp_path: Path) -> None:
    grouped = [dict(TINY_TASKS[0], group="Bounds  Family "), dict(TINY_TASKS[3], group="bounds family")]
    suite = tasks.load_suite(write_suite(tmp_path / "grouped", grouped))
    clusters = {i.id: i.cluster for f in FEATURES.values() for i in f.items(suite, ALL)}
    assert set(clusters.values()) == {"bounds family"}
    splits = {i.split for f in FEATURES.values() for i in f.items(suite, ALL)}
    assert len(splits) == 1, "tasks of one group share a split"


def test_registry_and_protocol() -> None:
    assert get_feature("bugfix") is FEATURES["bugfix"] and get_feature("judge") is FEATURES["judge"]
    with pytest.raises(KeyError, match="bugfix, judge"):
        get_feature("summariser")
    for name, feature in FEATURES.items():
        assert isinstance(feature, Feature) and feature.name == name
    assert FEATURES["bugfix"].writes_code and not FEATURES["judge"].writes_code
    assert FEATURES["bugfix"].primary_metric == "resolved_rate"
    assert FEATURES["judge"].primary_metric == "balanced_accuracy"


# ---- bugfix scoring -------------------------------------------------------------------------------


def test_the_reference_fix_resolves(suite: Suite) -> None:
    item = item_of(suite, "bugfix:t-clamp")
    reply = fix_reply(item.task.reference_code, think="The upper bound must be hi, not lo.", prompt_tokens=120, completion_tokens=40)
    result = BugfixFeature().score(item, reply, timeout_s=TIMEOUT)
    assert result.correct and result.schema_valid and result.lenient_valid
    assert result.shown_passed is True and result.hidden_passed is True
    assert result.think_present and result.think_valid and result.think_words == 8
    assert (result.sast_block, result.sast_flag, result.problems) == ((), (), ())
    assert (result.item_id, result.feature, result.task_id, result.cluster, result.split) == ("bugfix:t-clamp", "bugfix", "t-clamp", "t-clamp", HELDOUT)
    assert (result.prompt_tokens, result.completion_tokens, result.latency_s, result.finish_reason) == (120, 40, 0.25, "stop")
    assert result.detail["code_source"] == "strict" and result.detail["sast"] == []
    assert isinstance(result.detail["exec"], dict) and result.detail["think"]["valid"] is True
    assert ItemResult.from_json(json.loads(json.dumps(result.to_json()))) == result


def test_an_overfitted_patch_passes_the_shown_test_and_is_not_resolved(suite: Suite) -> None:
    for task in TINY_TASKS:
        item = item_of(suite, f"bugfix:{task['id']}")
        result = BugfixFeature().score(item, fix_reply(task["wrong_code"]), timeout_s=TIMEOUT)
        assert result.schema_valid and not result.correct, task["id"]
        assert result.shown_passed is True and result.hidden_passed is False, task["id"]
        assert any(p.startswith("exec: hidden test") for p in result.problems), result.problems


def test_the_buggy_code_itself_fails_the_shown_test(suite: Suite) -> None:
    item = item_of(suite, "bugfix:t-running-total")
    result = BugfixFeature().score(item, fix_reply(item.task.buggy_code), timeout_s=TIMEOUT)
    assert not result.correct and result.shown_passed is False


def test_a_block_finding_makes_the_item_unresolved_and_the_code_is_never_run(suite: Suite, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(features.verdicts, "exec_verdict", no_execution)
    result = BugfixFeature().score(item_of(suite, "bugfix:t-clamp"), fix_reply(CLAMP_EVAL), timeout_s=TIMEOUT)
    assert result.schema_valid and not result.correct
    assert result.sast_block == ("eval-exec",)
    assert result.shown_passed is None and result.hidden_passed is None and result.detail["exec"] is None
    assert [f["rule"] for f in result.detail["sast"]] == ["eval-exec"]
    assert any("not run" in p and "eval-exec" in p for p in result.problems), result.problems


def test_code_in_malformed_json_is_still_scanned(suite: Suite, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(features.verdicts, "exec_verdict", no_execution)
    fenced = "Here is the fix:\n```json\n" + json.dumps({"explanation": "x", "patched_code": CLAMP_EVAL}) + "\n```"
    result = BugfixFeature().score(item_of(suite, "bugfix:t-clamp"), ChatResponse(content=fenced, finish_reason="stop"), timeout_s=TIMEOUT)
    assert not result.schema_valid and result.lenient_valid and not result.correct
    assert result.sast_block == ("eval-exec",) and result.detail["code_source"] == "lenient"
    assert result.shown_passed is None
    assert any("lenient" in p for p in result.problems), result.problems


def test_a_leniently_valid_correct_fix_is_not_run_or_resolved(suite: Suite, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(features.verdicts, "exec_verdict", no_execution)
    item = item_of(suite, "bugfix:t-clamp")
    fenced = "```json\n" + json.dumps({"explanation": "x", "patched_code": item.task.reference_code}) + "\n```"
    result = BugfixFeature().score(item, ChatResponse(content=fenced, finish_reason="stop"), timeout_s=TIMEOUT)
    assert result.lenient_valid and not result.schema_valid and not result.correct
    assert result.sast_block == () and result.detail["code_source"] == "lenient"


def test_text_that_is_not_json_scores_nothing(suite: Suite, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(features.verdicts, "exec_verdict", no_execution)
    reply = ChatResponse(content="The bug is on line 3: use hi instead of lo.", finish_reason="stop")
    result = BugfixFeature().score(item_of(suite, "bugfix:t-clamp"), reply, timeout_s=TIMEOUT)
    assert not (result.correct or result.schema_valid or result.lenient_valid)
    assert result.detail["code_source"] is None and result.sast_block == ()
    assert any(p.startswith("schema:") for p in result.problems)


def test_a_truncated_reply_is_invalid_whatever_it_holds(suite: Suite, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(features.verdicts, "exec_verdict", no_execution)
    item = item_of(suite, "bugfix:t-clamp")
    complete_but_cut = fix_reply(item.task.reference_code, finish="length")
    result = BugfixFeature().score(item, complete_but_cut, timeout_s=TIMEOUT)
    assert not result.schema_valid and not result.correct and result.finish_reason == "length"
    cut_while_thinking = ChatResponse(content="<think>First I look at the bounds, then", finish_reason="length")
    result = BugfixFeature().score(item, cut_while_thinking, timeout_s=TIMEOUT)
    assert result.think_present and not result.think_valid and not result.schema_valid and not result.correct
    assert any(p.startswith("think:") for p in result.problems), result.problems


def test_thinking_returned_separately_is_read(suite: Suite) -> None:
    item = item_of(suite, "bugfix:t-count-vowels")
    reply = ChatResponse(
        content=json.dumps({"explanation": "Lower-case first.", "patched_code": item.task.reference_code}),
        reasoning="Uppercase vowels are missed because the comparison is case-sensitive.",
        finish_reason="stop",
    )
    result = BugfixFeature().score(item, reply, timeout_s=TIMEOUT)
    assert result.correct and result.think_present and result.think_valid and result.think_words == 9
    assert result.detail["think"]["source"] == "reasoning"


def test_flag_findings_are_recorded_and_decide_nothing(suite: Suite) -> None:
    item = item_of(suite, "bugfix:t-clamp")
    code = "import socket\n\n" + item.task.reference_code
    result = BugfixFeature().score(item, fix_reply(code), timeout_s=TIMEOUT)
    assert result.correct and result.sast_block == () and result.sast_flag == ("network",)


def test_an_infrastructure_fault_raises_and_is_never_a_verdict(suite: Suite, monkeypatch: pytest.MonkeyPatch) -> None:
    def broken(task, code, **kwargs):
        return verdicts.ExecVerdict(passed=False, shown_passed=False, hidden_passed=False, infra_error="the interpreter could not be started")

    monkeypatch.setattr(features.verdicts, "exec_verdict", broken)
    item = item_of(suite, "bugfix:t-clamp")
    with pytest.raises(ScoringError, match="bugfix:t-clamp.*interpreter"):
        BugfixFeature().score(item, fix_reply(item.task.reference_code), timeout_s=TIMEOUT)


# ---- judge scoring --------------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("variant", "reply", "predicted", "correct"),
    [
        ("reference", verdict_reply(True), True, True),
        ("reference", verdict_reply(False), False, False),
        ("wrong", verdict_reply(False, reason="It special-cases the shown input."), False, True),
        ("wrong", verdict_reply(True), True, False),
        ("wrong", verdict_reply(0), None, False),  # 0 is not false
        ("reference", verdict_reply(1), None, False),  # 1 is not true
        ("reference", verdict_reply(True, confidence=0.9), None, False),  # an unknown key
        ("reference", ChatResponse(content="Yes, the patch is correct.", finish_reason="stop"), None, False),
        ("wrong", ChatResponse(content='{"verdict": false}', finish_reason="length"), None, False),
    ],
)
def test_judge_scoring(suite: Suite, monkeypatch: pytest.MonkeyPatch, variant: str, reply: ChatResponse, predicted: object, correct: bool) -> None:
    monkeypatch.setattr(features.verdicts, "exec_verdict", no_execution)
    item = item_of(suite, f"judge:t-parse-port:{variant}")
    result = JudgeFeature().score(item, reply, timeout_s=TIMEOUT)
    assert result.predicted is predicted and result.correct is correct
    assert result.label is (variant == "reference")
    assert result.schema_valid is (predicted is not None)
    assert result.shown_passed is None and result.hidden_passed is None and result.sast_block == ()
    assert (result.task_id, result.cluster, result.split) == ("t-parse-port", "t-parse-port", DEV)
    if predicted is not None and not correct:
        assert any(p.startswith("verdict:") for p in result.problems), result.problems
    assert ItemResult.from_json(json.loads(json.dumps(result.to_json()))) == result


def test_judge_thinking_counts_toward_think_validity(suite: Suite) -> None:
    item = item_of(suite, "judge:t-clamp:wrong")
    reply = ChatResponse(content="<think>It returns 3 only for x == 5.</think>" + json.dumps({"verdict": False}), finish_reason="stop")
    result = JudgeFeature().score(item, reply, timeout_s=TIMEOUT)
    assert result.correct and result.think_present and result.think_valid
    too_long = ChatResponse(content="<think>" + "word " * 1300 + "</think>" + json.dumps({"verdict": False}), finish_reason="stop")
    result = JudgeFeature().score(item, too_long, timeout_s=TIMEOUT)
    assert result.correct and result.think_present and not result.think_valid
    assert any("1300 words" in p for p in result.problems), result.problems
