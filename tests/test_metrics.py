"""metrics: the formulas against known values, the per-arm blocks, and the paired comparison.

Every row is built by hand (bug_row, judge_task): these tests depend on records.ItemResult only.
"""

from __future__ import annotations

import json
import math

import pytest

from llm_eval_harness import metrics
from llm_eval_harness.metrics import (
    PRIMARY,
    Z80,
    Z95,
    arm_metrics,
    balanced_accuracy,
    bugfix_metrics,
    classifier_metrics,
    compare,
    confusion,
    judge_metrics,
    mcnemar_exact,
    min_significant_losses,
    paired_bootstrap,
    paired_tasks_only,
    percentile,
    primary_of,
    rate,
    wilson_interval,
)
from llm_eval_harness.records import ItemResult
from llm_eval_harness.util import canonical_json


def bug_row(task: str, correct: bool, *, cluster: str | None = None, schema: bool = True, lenient: bool | None = None, think: bool = True,
            shown: bool | None = None, hidden: bool | None = None, block: tuple[str, ...] = (), flag: tuple[str, ...] = (),
            tokens: int | None = None, latency: float | None = None, finish: str | None = "stop", tags: list[str] | None = None) -> ItemResult:  # fmt: skip
    return ItemResult(
        item_id=f"bugfix:{task}", feature="bugfix", task_id=task, cluster=cluster or task, split="heldout", correct=correct,
        schema_valid=schema, lenient_valid=schema if lenient is None else lenient, think_present=False, think_valid=think, think_words=0,
        finish_reason=finish, latency_s=latency, completion_tokens=tokens, shown_passed=shown, hidden_passed=hidden,
        sast_block=block, sast_flag=flag, detail={"tags": tags} if tags is not None else {},
    )  # fmt: skip


def judge_row(task: str, variant: str, predicted: bool | None, *, cluster: str | None = None) -> ItemResult:
    label = variant == "reference"
    return ItemResult(
        item_id=f"judge:{task}:{variant}", feature="judge", task_id=task, cluster=cluster or task, split="heldout",
        correct=predicted is label, schema_valid=predicted is not None, lenient_valid=predicted is not None, think_present=False,
        think_valid=True, think_words=0, label=label, predicted=predicted,
    )  # fmt: skip


def judge_task(task: str, on_reference: bool | None, on_wrong: bool | None, *, cluster: str | None = None) -> list[ItemResult]:
    """Both items of one judge task: the verdict given on the reference code and on the wrong code."""
    return [judge_row(task, "reference", on_reference, cluster=cluster), judge_row(task, "wrong", on_wrong, cluster=cluster)]


def bug_arm(correct_tasks: set[int], n: int) -> list[ItemResult]:
    return [bug_row(f"t{k:02d}", k in correct_tasks) for k in range(n)]


# ---- the formulas --------------------------------------------------------------------------------


def test_rate() -> None:
    assert rate(1, 4) == 0.25
    assert rate(0, 0) is None


@pytest.mark.parametrize(
    "k, n, low, high",
    [
        # The documented formula at z = Z95, checked at 40-digit precision.
        (0, 10, 0.0, 0.2775328),
        (5, 10, 0.2365931, 0.7634069),
        (10, 10, 0.7224672, 1.0),
    ],
)
def test_wilson_known_values(k: int, n: int, low: float, high: float) -> None:
    interval = wilson_interval(k, n)
    assert interval == pytest.approx((low, high), abs=1e-7)


def test_wilson_ends_are_exact_and_the_interval_is_symmetric() -> None:
    assert wilson_interval(0, 7)[0] == 0.0
    assert wilson_interval(7, 7)[1] == 1.0
    assert wilson_interval(0, 0) is None
    for n in (1, 2, 5, 13, 40):
        for k in range(n + 1):
            low, high = wilson_interval(k, n)
            mirror_low, mirror_high = wilson_interval(n - k, n)
            assert 0.0 <= low <= k / n <= high <= 1.0
            assert low == pytest.approx(1 - mirror_high, abs=1e-12) and high == pytest.approx(1 - mirror_low, abs=1e-12)


@pytest.mark.parametrize("k, n", [(5, 4), (-1, 3), (1, -1), (1.0, 2), (True, 2)])
def test_wilson_refuses_impossible_counts(k, n) -> None:
    with pytest.raises(ValueError):
        wilson_interval(k, n)


@pytest.mark.parametrize(
    "b, c, p",
    [(0, 0, 1.0), (1, 0, 1.0), (5, 0, 0.0625), (6, 0, 0.03125), (3, 10, 0.09228515625), (7, 7, 1.0)],
)
def test_mcnemar_known_values(b: int, c: int, p: float) -> None:
    assert mcnemar_exact(b, c) == p
    assert mcnemar_exact(c, b) == p


def test_mcnemar_is_exact_for_large_counts() -> None:
    assert mcnemar_exact(0, 60) == 2.0**-59
    assert mcnemar_exact(1000, 1000) == 1.0
    assert 0.0 < mcnemar_exact(60, 10) < 1e-9


@pytest.mark.parametrize("b, c", [(-1, 0), (0, -2), (1.5, 0), (True, 1)])
def test_mcnemar_refuses_bad_counts(b, c) -> None:
    with pytest.raises(ValueError):
        mcnemar_exact(b, c)


def test_min_significant_losses() -> None:
    assert min_significant_losses(0.05) == 6
    assert min_significant_losses(0.01) == 8
    assert min_significant_losses() == 6
    for alpha in (0.2, 0.1, 0.05, 0.01, 0.001):
        b = min_significant_losses(alpha)
        assert mcnemar_exact(b, 0) < alpha <= mcnemar_exact(b - 1, 0)
    for alpha in (0.0, 1.0, -0.1):
        with pytest.raises(ValueError):
            min_significant_losses(alpha)


def test_percentile_interpolates_linearly() -> None:
    values = [1.0, 2.0, 3.0, 4.0]
    assert percentile(values, 0.0) == 1.0
    assert percentile(values, 1.0) == 4.0
    assert percentile(values, 0.5) == 2.5
    assert percentile(values, 0.25) == pytest.approx(1.75)
    assert percentile([7.0], 0.975) == 7.0
    with pytest.raises(ValueError):
        percentile([], 0.5)
    with pytest.raises(ValueError):
        percentile(values, 1.5)


# ---- classifier metrics --------------------------------------------------------------------------


def test_confusion_counts_invalid_predictions_against_their_class() -> None:
    labels = [True, True, True, False, False, False]
    predictions = [True, None, False, False, None, True]
    assert confusion(labels, predictions) == {"tp": 1, "fn": 1, "invalid_pos": 1, "tn": 1, "fp": 1, "invalid_neg": 1}
    # recall on the true items 1/3, on the false items 1/3
    assert balanced_accuracy(labels, predictions) == pytest.approx(1 / 3)


def test_balanced_accuracy_cases() -> None:
    assert balanced_accuracy([], []) is None
    assert balanced_accuracy([True, True], [True, None]) == 0.5  # one class present: its recall
    assert balanced_accuracy([True, False], [True, False]) == 1.0
    assert balanced_accuracy([True, False], [True, True]) == 0.5  # approves everything
    assert balanced_accuracy([True, False, False, False], [True, False, False, False]) == 1.0
    with pytest.raises(ValueError):
        confusion([True], [True, False])
    with pytest.raises(ValueError):
        confusion([1], [True])  # labels are bools, not ints


def test_classifier_metrics() -> None:
    out = classifier_metrics([True, True, False, False], [True, False, True, None])
    assert out["confusion"] == {"tp": 1, "fn": 1, "tn": 0, "fp": 1, "invalid_pos": 0, "invalid_neg": 1}
    assert (out["positives"], out["negatives"]) == (2, 2)
    assert out["accuracy"] == 0.25
    assert out["tpr"] == 0.5 and out["tnr"] == 0.0
    assert out["balanced_accuracy"] == 0.25
    assert out["precision"] == 0.5
    assert out["f1"] == 0.5
    assert out["approve_rate"] == 0.5
    assert out["tpr_ci95"] == [round(x, 6) for x in wilson_interval(1, 2)]
    assert out["tnr_ci95"][0] == 0.0


def test_classifier_metrics_undefined_ratios_are_none() -> None:
    out = classifier_metrics([True, False], [False, False])  # never approves
    assert out["precision"] is None and out["f1"] is None
    assert out["approve_rate"] == 0.0
    empty = classifier_metrics([], [])
    assert empty["balanced_accuracy"] is None and empty["tpr"] is None and empty["tpr_ci95"] is None


def test_paired_tasks_only_drops_tasks_without_both_items_and_non_judge_rows() -> None:
    rows = judge_task("a", True, False) + [judge_row("b", "reference", True)] + [bug_row("c", True)]
    assert [r.item_id for r in paired_tasks_only(rows)] == ["judge:a:reference", "judge:a:wrong"]


# ---- per arm -------------------------------------------------------------------------------------


def _bugfix_rows() -> list[ItemResult]:
    return [
        bug_row("t1", True, shown=True, hidden=True, tokens=100, latency=1.0, tags=["off-by-one"]),
        bug_row("t2", False, shown=True, hidden=False, tokens=200, latency=3.0, tags=["off-by-one", "boundary"]),
        bug_row("t3", False, schema=False, lenient=True, block=("eval-exec",), flag=("file-write",), finish="length", tags=["boundary"]),
        bug_row("t4", False, think=False, block=("os-command",), flag=("file-write", "network"), finish=None, tokens=300, latency=2.0),
    ]


def test_bugfix_metrics_every_key() -> None:
    out = bugfix_metrics(_bugfix_rows())
    assert out["schema_validity"] == 0.75
    assert out["schema_validity_ci95"] == [round(x, 6) for x in wilson_interval(3, 4)]
    assert out["think_validity"] == 0.75
    assert out["lenient_only_rate"] == 0.25
    assert out["resolved"] == 1 and out["resolved_rate"] == 0.25 and out["primary"] == 0.25
    assert out["resolved_rate_ci95"] == [round(x, 6) for x in wilson_interval(1, 4)]
    assert out["shown_pass_rate"] == 0.5
    assert out["hidden_pass_rate"] == 0.25
    assert out["overfit_rate"] == 0.5  # t2 of t1, t2: fits the shown test only
    assert out["sast"] == {"block": {"eval-exec": 1, "os-command": 1}, "flag": {"file-write": 2, "network": 1}}
    assert out["finish_reasons"] == {"length": 1, "stop": 2, "unknown": 1}
    assert out["completion_tokens_total"] == 600 and out["completion_tokens_mean"] == 200.0
    assert out["latency_median_s"] == 2.0
    assert out["by_tag"] == {"boundary": {"items": 2, "resolved": 0}, "off-by-one": {"items": 2, "resolved": 1}}


def test_bugfix_metrics_counts_a_rule_once_per_item() -> None:
    out = bugfix_metrics([bug_row("a", False, block=("eval-exec", "eval-exec")), bug_row("b", False, block=("eval-exec",))])
    assert out["sast"]["block"] == {"eval-exec": 2}


def test_bugfix_metrics_on_no_items() -> None:
    out = bugfix_metrics([])
    for key in ("schema_validity", "think_validity", "resolved_rate", "shown_pass_rate", "hidden_pass_rate", "overfit_rate", "primary"):
        assert out[key] is None, key
    assert out["resolved_rate_ci95"] is None and out["schema_validity_ci95"] is None
    assert out["completion_tokens_total"] is None and out["completion_tokens_mean"] is None and out["latency_median_s"] is None
    assert out["sast"] == {"block": {}, "flag": {}} and out["by_tag"] == {} and out["finish_reasons"] == {}


def test_judge_metrics_perfect_judge_is_above_chance() -> None:
    rows = [r for k in range(30) for r in judge_task(f"t{k:02d}", True, False)]
    out = judge_metrics(rows)
    assert out["balanced_accuracy"] == 1.0 and out["primary"] == 1.0
    assert out["balanced_accuracy_ci95"] == [1.0, 1.0]
    assert out["vs_chance"] is True
    assert out["approve_rate"] == 0.5
    assert (out["paired_items"], out["paired_tasks"], out["unpaired_items"]) == (60, 30, 0)


def test_judge_metrics_rubber_stamp_scores_one_half() -> None:
    rows = [r for k in range(10) for r in judge_task(f"t{k}", True, True)]
    out = judge_metrics(rows)
    assert out["balanced_accuracy"] == 0.5
    assert out["tpr"] == 1.0 and out["tnr"] == 0.0
    assert out["approve_rate"] == 1.0
    assert out["precision"] == 0.5 and out["f1"] == pytest.approx(0.666667)
    assert out["vs_chance"] is False


def test_judge_metrics_invalid_and_unpaired_items() -> None:
    rows = judge_task("a", True, False) + judge_task("b", None, False) + [judge_row("c", "reference", True)]
    out = judge_metrics(rows)
    assert out["confusion"] == {"tp": 1, "fn": 0, "invalid_pos": 1, "tn": 2, "fp": 0, "invalid_neg": 0}
    assert out["tpr"] == 0.5 and out["tnr"] == 1.0 and out["balanced_accuracy"] == 0.75
    assert (out["paired_items"], out["unpaired_items"]) == (4, 1)
    # validity counts every reply, the lone item included
    assert out["schema_validity"] == 0.8


def test_judge_metrics_interval_is_deterministic_and_seeded() -> None:
    rows = []
    for k in range(24):
        rows += judge_task(f"t{k:02d}", k % 3 != 0, k % 4 == 0)
    first, again = judge_metrics(rows), judge_metrics(rows)
    assert first == again
    # with 2,000 draws the percentiles of this discrete statistic settle on the same values for
    # any seed; with a few draws the seed shows
    few = {tuple(judge_metrics(rows, resamples=25, seed=s)["balanced_accuracy_ci95"]) for s in range(6)}
    assert len(few) > 1
    low, high = first["balanced_accuracy_ci95"]
    assert low <= first["balanced_accuracy"] <= high


def test_arm_metrics_adds_items_and_primary_metric_and_is_json() -> None:
    bug = arm_metrics("bugfix", _bugfix_rows())
    assert bug["items"] == 4 and bug["primary_metric"] == "resolved_rate" and bug["primary"] == bug["resolved_rate"]
    judge = arm_metrics("judge", judge_task("a", True, False), resamples=50, seed=3)
    assert judge["items"] == 2 and judge["primary_metric"] == "balanced_accuracy"
    for block in (bug, judge):
        assert json.loads(canonical_json(block)) == block  # strict JSON: no NaN, string keys
    with pytest.raises(ValueError):
        arm_metrics("summarise", [])
    assert set(PRIMARY) == {"bugfix", "judge", "review", "patch"}


def test_review_and_patch_metrics_and_both_valid_compare() -> None:
    rev_rows = [
        ItemResult(
            item_id=f"review:t{k}", feature="review", task_id=f"t{k}", cluster=f"t{k}", split="heldout",
            correct=(k != 2), schema_valid=(k != 2), lenient_valid=(k != 2),
            think_present=True, think_valid=True, think_words=10,
            label=(k % 2 == 0), predicted=(None if k == 2 else (k % 2 == 0)),
        )
        for k in range(6)
    ]
    rev = arm_metrics("review", rev_rows, resamples=50, seed=1)
    assert rev["items"] == 6 and rev["primary_metric"] == "balanced_accuracy"
    assert rev["positives"] == 3 and rev["negatives"] == 3
    assert rev["confusion"]["invalid_pos"] == 1
    assert rev["balanced_accuracy"] == pytest.approx((2 / 3 + 1.0) / 2, abs=1e-6)
    assert rev["primary"] == rev["balanced_accuracy"]
    assert metrics.primary_of("review", rev_rows) == pytest.approx(5 / 6)

    pat_rows = [
        ItemResult(
            item_id="patch:p1", feature="patch", task_id="p1", cluster="p1", split="heldout",
            correct=True, schema_valid=True, lenient_valid=True, think_present=False, think_valid=True, think_words=0,
            detail={"applied": True, "noop": False, "gold_match": True, "tags": ["multi-file"]},
        ),
        ItemResult(
            item_id="patch:p2", feature="patch", task_id="p2", cluster="p2", split="heldout",
            correct=False, schema_valid=True, lenient_valid=True, think_present=False, think_valid=True, think_words=0,
            sast_block=("shell-true",), detail={"applied": True, "noop": False, "gold_match": False, "tags": ["multi-file"]},
        ),
        ItemResult(
            item_id="patch:p3", feature="patch", task_id="p3", cluster="p3", split="heldout",
            correct=False, schema_valid=False, lenient_valid=True, think_present=False, think_valid=True, think_words=0,
            detail={"applied": False, "noop": False, "gold_match": False},
        ),
    ]
    pat = arm_metrics("patch", pat_rows)
    assert pat["items"] == 3 and pat["primary_metric"] == "usable_rate"
    assert pat["usable"] == 1 and pat["usable_rate"] == pytest.approx(1 / 3, abs=1e-6)
    assert pat["applied_rate"] == pytest.approx(2 / 3, abs=1e-6)
    assert pat["gold_match_rate"] == pytest.approx(1 / 3, abs=1e-6)
    assert pat["sast"] == {"block": {"shell-true": 1}, "flag": {}}
    assert pat["by_tag"] == {"multi-file": {"items": 2, "usable": 1}}
    assert metrics.primary_of("patch", pat_rows) == pytest.approx(1 / 3)
    assert metrics.patch_metrics([])["usable_rate"] is None

    # both_valid_compare keeps only items where candidate & baseline are both schema_valid
    cand = [
        ItemResult("patch:1", "patch", "1", "1", "heldout", True, True, True, False, True, 0),
        ItemResult("patch:2", "patch", "2", "2", "heldout", False, False, False, False, True, 0),
        ItemResult("patch:3", "patch", "3", "3", "heldout", True, True, True, False, True, 0),
    ]
    base = [
        ItemResult("patch:1", "patch", "1", "1", "heldout", False, True, True, False, True, 0),
        ItemResult("patch:2", "patch", "2", "2", "heldout", True, True, True, False, True, 0),
        ItemResult("patch:3", "patch", "3", "3", "heldout", True, False, True, False, True, 0),
    ]
    both = metrics.both_valid_compare("patch", cand, base, split="heldout", resamples=50)
    assert (both.n, both.gains, both.losses, both.delta) == (1, 1, 0, 1.0)


def test_primary_of() -> None:
    assert primary_of("bugfix", bug_arm({0, 1, 2}, 4)) == 0.75
    assert primary_of("bugfix", []) is None
    assert primary_of("judge", judge_task("a", True, True) + [judge_row("b", "reference", False)]) == 0.5
    with pytest.raises(ValueError):
        primary_of("other", [])


# ---- the bootstrap and the comparison ---------------------------------------------------------


def _pairs_in_clusters(n_clusters: int, size: int) -> list[list[tuple[ItemResult, ItemResult]]]:
    clusters = []
    for k in range(n_clusters):
        group = []
        for j in range(size):
            task = f"c{k:02d}-{j}"
            group.append((bug_row(task, (k + j) % 3 != 0, cluster=f"c{k:02d}"), bug_row(task, k % 2 == 0, cluster=f"c{k:02d}")))
        clusters.append(group)
    return clusters


def _delta(pairs) -> float | None:
    if not pairs:
        return None
    return (sum(c.correct for c, _ in pairs) - sum(b.correct for _, b in pairs)) / len(pairs)


def test_paired_bootstrap_is_deterministic_sorted_and_seed_sensitive() -> None:
    clusters = _pairs_in_clusters(20, 1)
    first = paired_bootstrap(clusters, _delta, resamples=300, seed_key="0/bugfix/heldout")
    assert first == paired_bootstrap(clusters, _delta, resamples=300, seed_key="0/bugfix/heldout")
    assert first == sorted(first) and len(first) == 300
    assert first != paired_bootstrap(clusters, _delta, resamples=300, seed_key="1/bugfix/heldout")
    assert paired_bootstrap([], _delta, resamples=10, seed_key="x") == []


def test_paired_bootstrap_drops_draws_without_a_delta() -> None:
    clusters = _pairs_in_clusters(6, 1)
    first_task = clusters[0][0][0].item_id
    deltas = paired_bootstrap(clusters, lambda ps: None if any(c.item_id == first_task for c, _ in ps) else 0.0, resamples=200, seed_key="k")
    assert 0 < len(deltas) < 200  # about (5/6)^6 of the draws miss the first cluster


def test_paired_bootstrap_draws_a_cluster_whole() -> None:
    clusters = _pairs_in_clusters(8, 3)
    draws: list[list[str]] = []

    def record(pairs) -> float:
        draws.append([c.item_id for c, _ in pairs])
        return 0.0

    paired_bootstrap(clusters, record, resamples=100, seed_key="whole")
    assert len(draws) == 100
    for drawn in draws:
        assert len(drawn) == 8 * 3
        for k in range(8):
            counts = {drawn.count(f"bugfix:c{k:02d}-{j}") for j in range(3)}
            assert len(counts) == 1, f"cluster c{k:02d} was split in a draw"


def test_compare_identical_arms() -> None:
    rows = bug_arm({0, 2, 3, 7}, 12)
    result = compare("bugfix", rows, rows, split="heldout")
    assert (result.n, result.clusters, result.gains, result.losses) == (12, 12, 0, 0)
    assert result.delta == 0.0 and result.ci95 == (0.0, 0.0)
    assert result.se == 0.0 and result.mde80 is None
    assert result.mcnemar_p == 1.0 and result.min_significant_losses == 6
    assert result.unpaired == []
    out = result.to_json()
    assert out["ci95"] == [0.0, 0.0] and out["delta"] == 0.0 and out["mde80"] is None
    assert "-0.0" not in json.dumps(out)


def test_compare_gains_losses_delta_and_interval() -> None:
    candidate = bug_arm(set(range(0, 14)), 20)
    baseline = bug_arm(set(range(4, 12)) | {15}, 20)
    result = compare("bugfix", candidate, baseline, split="heldout")
    assert (result.gains, result.losses) == (6, 1)
    assert result.candidate == 0.7 and result.baseline == pytest.approx(0.45)
    assert result.delta == pytest.approx((6 - 1) / 20)
    assert result.mcnemar_p == mcnemar_exact(1, 6)
    low, high = result.ci95
    assert low < result.delta < high
    assert result.se > 0 and result.mde80 == pytest.approx((Z95 + Z80) * result.se)
    assert (result.metric, result.resamples, result.seed, result.alpha) == ("resolved_rate", 2000, 0, 0.05)


def test_compare_uses_the_documented_seed_key_and_clusters() -> None:
    candidate = [bug_row(f"t{k:02d}", k % 3 != 0, cluster=f"g{k // 2}") for k in range(16)]
    baseline = [bug_row(f"t{k:02d}", k % 2 == 0, cluster=f"g{k // 2}") for k in range(16)]
    result = compare("bugfix", candidate, baseline, split="dev", resamples=400, seed=7)
    assert result.clusters == 8
    by_cluster: dict[str, list] = {}
    for c, b in zip(candidate, baseline):
        by_cluster.setdefault(c.cluster, []).append((c, b))
    deltas = paired_bootstrap([by_cluster[k] for k in sorted(by_cluster)], _delta, resamples=400, seed_key="7/bugfix/dev")
    assert result.ci95 == (percentile(deltas, 0.025), percentile(deltas, 0.975))
    mean = sum(deltas) / len(deltas)
    assert result.se == pytest.approx(math.sqrt(sum((d - mean) ** 2 for d in deltas) / (len(deltas) - 1)))
    assert compare("bugfix", candidate, baseline, split="dev", resamples=400, seed=7) == result


def test_compare_lists_unpaired_items_and_joins_on_id() -> None:
    candidate = [bug_row("a", True), bug_row("b", False), bug_row("c", True)]
    baseline = [bug_row("b", True), bug_row("a", True), bug_row("d", False)]
    result = compare("bugfix", candidate, baseline, split="heldout")
    assert result.unpaired == ["bugfix:c", "bugfix:d"]
    assert (result.n, result.gains, result.losses) == (2, 0, 1)
    assert result.delta == pytest.approx(-0.5)


def test_compare_judge_on_balanced_accuracy() -> None:
    candidate = [r for k in range(10) for r in judge_task(f"t{k}", True, False)]
    baseline = [r for k in range(10) for r in judge_task(f"t{k}", True, True)]
    result = compare("judge", candidate, baseline, split="heldout")
    assert result.metric == "balanced_accuracy"
    assert (result.candidate, result.baseline, result.delta) == (1.0, 0.5, 0.5)
    assert (result.gains, result.losses, result.clusters) == (10, 0, 10)
    assert result.ci95[0] > 0


def test_compare_refuses_arms_of_different_suites() -> None:
    candidate = judge_task("a", True, False)
    flipped = [judge_row("a", "reference", True), judge_row("a", "wrong", False)]
    flipped[1].label = True
    with pytest.raises(ValueError, match="same suite"):
        compare("judge", candidate, flipped, split="heldout")
    with pytest.raises(ValueError, match="twice"):
        compare("bugfix", [bug_row("a", True), bug_row("a", False)], [bug_row("a", True)], split="heldout")
    with pytest.raises(ValueError):
        compare("other", [], [], split="heldout")


def test_comparison_json_rounds_rates_but_not_p() -> None:
    candidate = bug_arm(set(), 40)
    baseline = bug_arm(set(range(40)), 40)
    out = compare("bugfix", candidate, baseline, split="heldout", resamples=200).to_json()
    assert out["mcnemar_p"] == 2.0**-39 and out["mcnemar_p"] > 0
    assert out["losses"] == 40 and out["delta"] == -1.0
    assert set(out) == {
        "feature", "metric", "n", "clusters", "gains", "losses", "candidate", "baseline", "delta", "ci95", "se", "mde80",
        "mcnemar_p", "min_significant_losses", "alpha", "resamples", "seed", "unpaired",
    }  # fmt: skip
    thirds = compare("bugfix", bug_arm({0}, 3), bug_arm(set(), 3), split="heldout", resamples=50).to_json()
    assert thirds["delta"] == 0.333333 and thirds["candidate"] == 0.333333


def test_compare_with_no_common_items() -> None:
    result = compare("bugfix", [bug_row("a", True)], [bug_row("b", True)], split="heldout")
    assert (result.n, result.delta, result.ci95, result.se, result.mde80, result.mcnemar_p) == (0, None, None, None, None, 1.0)


def test_module_constants() -> None:
    assert metrics.DEFAULT_RESAMPLES == 2000 and metrics.DEFAULT_SEED == 0
    assert Z95 == pytest.approx(1.959964, abs=1e-6) and Z80 == pytest.approx(0.841621, abs=1e-6)
