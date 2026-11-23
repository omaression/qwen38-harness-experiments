"""Metrics: per-arm rates with Wilson intervals, classifier metrics for the judge, and the paired
comparison of two arms on the same items (docs/DESIGN.md, "Metrics").

    arm_metrics(feature, results)          the metrics block of one arm in the report
    compare(feature, candidate, baseline)  the paired comparison the gate decides on
    wilson_interval, mcnemar_exact, balanced_accuracy, ...   the formulas, separately tested

Rates that go into a report are rounded to 6 decimals (util.round6) and intervals are written as
2-element lists; p-values are never rounded. Everything here is deterministic: the bootstraps use
random.Random seeded with a string, and the standard deviation is computed with math.fsum, so the
same rows give the same bytes on every machine and Python version.

PER ARM, bugfix (bugfix_metrics; n = items):
  schema_validity   schema-valid replies / n, with schema_validity_ci95 (Wilson)
  think_validity    replies whose thinking is valid / n
  lenient_only_rate replies that are only leniently valid / n (a format problem: see verdicts.py)
  resolved_rate     resolved items / n (the primary metric); resolved_rate_ci95 its Wilson interval
  shown_pass_rate   items whose shown tests passed / n (not run counts as not passed)
  hidden_pass_rate  items whose hidden tests passed / n
  overfit_rate      items with shown passed and hidden failed / items with shown passed (None if 0):
                    the patch fits the test it saw and not the function it was asked to fix
  sast              {"block": {rule: items}, "flag": {rule: items}}: per rule, the number of items
                    whose reply added a finding of it
  finish_reasons    {reason: count} ("unknown" when the server sent none); tokens and latency:
                    completion_tokens_total, completion_tokens_mean, latency_median_s (None when
                    not recorded)
  by_tag            {tag: {"items", "resolved"}}, from the list ItemResult.detail["tags"] (the
                    task's tags, which features.py copies there)
PER ARM, judge (judge_metrics; the classifier metrics are scored on paired tasks only, i.e. tasks
with both a reference and a wrong item, so a verdict that depends on the task alone scores
exactly 0.5; validity, finish reasons, tokens and latency count every reply):
  confusion         {"tp", "fn", "tn", "fp", "invalid_pos", "invalid_neg"}: tp/fn/tn/fp count the
                    valid verdicts; a reply without a valid verdict is counted in invalid_pos (a
                    missed true) or invalid_neg (a missed false) and so counts against its class
  balanced_accuracy (TPR + TNR) / 2, the primary metric, with balanced_accuracy_ci95 from a
                    bootstrap over tasks (their clusters: twins are drawn together; seed key
                    "<seed>/judge/arm"); tpr = tp / positives, tnr = tn / negatives, each with a
                    Wilson interval; accuracy, precision = tp / (tp + fp), f1, approve_rate =
                    (tp + fp) / items
  vs_chance         whether balanced_accuracy_ci95 lies above 0.5 (None without an interval)
  paired_items, paired_tasks, unpaired_items
  plus schema_validity, think_validity, finish_reasons, tokens and latency as for bugfix.

PAIRED (compare). The arms are joined on item id (the gate refuses arms whose item sets differ,
so `unpaired` is empty in a gated run). Per item, x = correct(candidate) - correct(baseline).
  gains   items only the candidate got right (x = +1);  losses: only the baseline (x = -1)
  delta   primary(candidate) - primary(baseline) on the joined items; for resolved_rate this is
          (gains - losses) / n; for balanced accuracy it is the difference of the two BAs
  ci95    paired bootstrap percentile interval of delta: `resamples` draws, with replacement, of
          the clusters (item.cluster: a task's items, and twins, are drawn together); both arms'
          rows of a drawn cluster go in together; percentiles by linear interpolation; the
          generator is random.Random(f"{seed}/{feature}/{split}"), so the interval is the same on
          every machine and every run
  se      the sample standard deviation of the bootstrap deltas; mde80 = (Z95 + Z80) * se, the
          true change a two-sided 5% test would detect four times in five (None when se is 0:
          no discordant items, or every draw gave the same delta, so nothing is known about the
          spread)
  mcnemar_p   the exact two-sided McNemar p of (losses, gains), unrounded
  min_significant_losses   the fewest losses, with no gains, that McNemar calls significant at
          alpha (6 at alpha 0.05): how blunt this suite is, whatever the arms did
"""

from __future__ import annotations

import math
import random
import statistics
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from typing import Any, Callable, Sequence, TypeVar

from .records import ItemResult
from .util import round6

#: Standard normal quantiles: two-sided 95%, and 80% power.
Z95 = 1.959963984540054
Z80 = 0.8416212335729143
DEFAULT_RESAMPLES = 2000
DEFAULT_SEED = 0
#: The primary metric of each feature (the key in arm_metrics and the gate's `primary` gate).
PRIMARY = {"bugfix": "resolved_rate", "judge": "balanced_accuracy"}

_T = TypeVar("_T")


def _r6(value: float | None) -> float | None:
    """util.round6, and -0.0 written as 0.0 (JSON would otherwise show "-0.0")."""
    return None if value is None else round6(value) + 0.0


def _ci_json(ci: tuple[float, float] | None) -> list[float] | None:
    return None if ci is None else [_r6(ci[0]), _r6(ci[1])]


@dataclass
class Comparison:
    """compare()'s result; to_json() is the report's `comparison` block (rates rounded to 6
    decimals, mcnemar_p unrounded so that a small p never reads 0.0)."""

    feature: str
    metric: str
    n: int
    clusters: int
    gains: int
    losses: int
    candidate: float | None
    baseline: float | None
    delta: float | None
    ci95: tuple[float, float] | None
    se: float | None
    mde80: float | None
    mcnemar_p: float
    min_significant_losses: int
    alpha: float
    resamples: int
    seed: int
    unpaired: list[str] = field(default_factory=list)

    def to_json(self) -> dict[str, Any]:
        return {
            "feature": self.feature,
            "metric": self.metric,
            "n": self.n,
            "clusters": self.clusters,
            "gains": self.gains,
            "losses": self.losses,
            "candidate": _r6(self.candidate),
            "baseline": _r6(self.baseline),
            "delta": _r6(self.delta),
            "ci95": _ci_json(self.ci95),
            "se": _r6(self.se),
            "mde80": _r6(self.mde80),
            "mcnemar_p": self.mcnemar_p,
            "min_significant_losses": self.min_significant_losses,
            "alpha": self.alpha,
            "resamples": self.resamples,
            "seed": self.seed,
            "unpaired": list(self.unpaired),
        }


# ---- the formulas --------------------------------------------------------------------------------


def rate(k: int, n: int) -> float | None:
    """k / n, or None when n is 0."""
    return k / n if n else None


def _check_counts(k: int, n: int) -> None:
    for name, value in (("k", k), ("n", n)):
        if isinstance(value, bool) or not isinstance(value, int):
            raise ValueError(f"{name} must be an int, got {value!r}")
    if not 0 <= k <= n:
        raise ValueError(f"need 0 <= k <= n, got k={k}, n={n}")


def wilson_interval(k: int, n: int, z: float = Z95) -> tuple[float, float] | None:
    """The Wilson score interval of k successes in n trials, clipped to [0, 1]; None when n is 0.
        centre = (k + z^2/2) / (n + z^2),  half = z * sqrt(k (n - k) / n + z^2/4) / (n + z^2)
    The ends at k = 0 and k = n are exactly 0 and 1 (the formula's value, without float noise)."""
    _check_counts(k, n)
    if n == 0:
        return None
    z2 = z * z
    centre = (k + z2 / 2) / (n + z2)
    half = z * math.sqrt(k * (n - k) / n + z2 / 4) / (n + z2)
    low = 0.0 if k == 0 else max(0.0, centre - half)
    high = 1.0 if k == n else min(1.0, centre + half)
    return (low, high)


def mcnemar_exact(b: int, c: int) -> float:
    """Exact two-sided McNemar p of b and c discordant items: min(1, 2 * sum_{k <= min(b, c)}
    C(b + c, k) / 2^(b + c)), and 1.0 when b + c is 0. Integer arithmetic (math.comb), so it is
    exact for any n (the one rounding is the final int / int division)."""
    for name, value in (("b", b), ("c", c)):
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise ValueError(f"{name} must be a non-negative int, got {value!r}")
    n = b + c
    if n == 0:
        return 1.0
    tail = sum(math.comb(n, k) for k in range(min(b, c) + 1))
    return min(1.0, 2 * tail / 2**n)


def min_significant_losses(alpha: float = 0.05) -> int:
    """The smallest b with mcnemar_exact(b, 0) < alpha (6 for alpha 0.05)."""
    if not 0.0 < alpha < 1.0:
        raise ValueError(f"alpha must be strictly between 0 and 1, got {alpha!r}")
    b = 1
    while mcnemar_exact(b, 0) >= alpha:
        b += 1
    return b


def percentile(sorted_values: Sequence[float], p: float) -> float:
    """The p-quantile (0 <= p <= 1) of already sorted values, by linear interpolation between
    the closest ranks (index (len - 1) * p). ValueError on an empty sequence."""
    if not sorted_values:
        raise ValueError("percentile of an empty sequence")
    if not 0.0 <= p <= 1.0:
        raise ValueError(f"p must be in [0, 1], got {p!r}")
    k = (len(sorted_values) - 1) * p
    low = math.floor(k)
    high = min(low + 1, len(sorted_values) - 1)
    return sorted_values[low] + (sorted_values[high] - sorted_values[low]) * (k - low)


# ---- classifier metrics (the judge) -------------------------------------------------------------


def confusion(labels: Sequence[bool], predictions: Sequence[bool | None]) -> dict[str, int]:
    """{"tp", "fn", "tn", "fp", "invalid_pos", "invalid_neg"} (see judge_metrics above)."""
    if len(labels) != len(predictions):
        raise ValueError(f"{len(labels)} labels but {len(predictions)} predictions")
    counts = dict.fromkeys(("tp", "fn", "tn", "fp", "invalid_pos", "invalid_neg"), 0)
    for label, predicted in zip(labels, predictions):
        if not isinstance(label, bool):
            raise ValueError(f"a label must be a bool, got {label!r}")
        if predicted is not None and not isinstance(predicted, bool):
            raise ValueError(f"a prediction must be a bool or None, got {predicted!r}")
        if label:
            key = "invalid_pos" if predicted is None else ("tp" if predicted else "fn")
        else:
            key = "invalid_neg" if predicted is None else ("fp" if predicted else "tn")
        counts[key] += 1
    return counts


def _class_sizes(counts: dict[str, int]) -> tuple[int, int]:
    """(positives, negatives), invalid replies included."""
    return counts["tp"] + counts["fn"] + counts["invalid_pos"], counts["tn"] + counts["fp"] + counts["invalid_neg"]


def _balanced(counts: dict[str, int]) -> float | None:
    positives, negatives = _class_sizes(counts)
    recalls = [r for r in (rate(counts["tp"], positives), rate(counts["tn"], negatives)) if r is not None]
    return sum(recalls) / len(recalls) if recalls else None


def balanced_accuracy(labels: Sequence[bool], predictions: Sequence[bool | None]) -> float | None:
    """Mean of the recall on the true items and on the false items (of the one class present, if
    only one is); a None prediction is a miss in its class. None when there are no items."""
    return _balanced(confusion(labels, predictions))


def classifier_metrics(labels: Sequence[bool], predictions: Sequence[bool | None]) -> dict[str, Any]:
    """confusion, accuracy, balanced_accuracy, tpr / tnr with Wilson intervals, precision, f1,
    approve_rate. Undefined ratios are None (f1 is 0.0 when precision and tpr are both 0)."""
    counts = confusion(labels, predictions)
    n = len(labels)
    positives, negatives = _class_sizes(counts)
    tpr = rate(counts["tp"], positives)
    tnr = rate(counts["tn"], negatives)
    precision = rate(counts["tp"], counts["tp"] + counts["fp"])
    if precision is None or tpr is None:
        f1 = None
    elif precision + tpr == 0:
        f1 = 0.0
    else:
        f1 = 2 * precision * tpr / (precision + tpr)
    return {
        "confusion": counts,
        "positives": positives,
        "negatives": negatives,
        "accuracy": _r6(rate(counts["tp"] + counts["tn"], n)),
        "balanced_accuracy": _r6(_balanced(counts)),
        "tpr": _r6(tpr),
        "tpr_ci95": _ci_json(wilson_interval(counts["tp"], positives)),
        "tnr": _r6(tnr),
        "tnr_ci95": _ci_json(wilson_interval(counts["tn"], negatives)),
        "precision": _r6(precision),
        "f1": _r6(f1),
        "approve_rate": _r6(rate(counts["tp"] + counts["fp"], n)),
    }


def paired_tasks_only(results: Sequence[ItemResult]) -> list[ItemResult]:
    """The judge rows of tasks that have both a true-label and a false-label item."""
    labels_of: dict[str, set[bool]] = defaultdict(set)
    for row in results:
        if row.label is not None:
            labels_of[row.task_id].add(bool(row.label))
    both = {task for task, labels in labels_of.items() if labels == {True, False}}
    return [row for row in results if row.label is not None and row.task_id in both]


# ---- per arm -------------------------------------------------------------------------------------


def _count(results: Sequence[ItemResult], test: Callable[[ItemResult], bool]) -> int:
    return sum(1 for row in results if test(row))


def _validity(results: Sequence[ItemResult]) -> dict[str, Any]:
    """The format metrics both features report."""
    n = len(results)
    schema = _count(results, lambda r: r.schema_valid)
    return {
        "schema_validity": _r6(rate(schema, n)),
        "schema_validity_ci95": _ci_json(wilson_interval(schema, n)),
        "think_validity": _r6(rate(_count(results, lambda r: r.think_valid), n)),
        "lenient_only_rate": _r6(rate(_count(results, lambda r: r.lenient_valid and not r.schema_valid), n)),
    }


def _cost(results: Sequence[ItemResult]) -> dict[str, Any]:
    """Finish reasons, completion tokens and latency, from what the recordings hold."""
    reasons = Counter("unknown" if r.finish_reason is None else str(r.finish_reason) for r in results)
    tokens = [r.completion_tokens for r in results if r.completion_tokens is not None]
    latencies = [r.latency_s for r in results if r.latency_s is not None]
    return {
        "finish_reasons": dict(sorted(reasons.items())),
        "completion_tokens_total": sum(tokens) if tokens else None,
        "completion_tokens_mean": _r6(sum(tokens) / len(tokens)) if tokens else None,
        "latency_median_s": _r6(statistics.median(latencies)) if latencies else None,
    }


def _tags(row: ItemResult) -> list[str]:
    tags = row.detail.get("tags") if isinstance(row.detail, dict) else None
    if not isinstance(tags, (list, tuple)):
        return []
    return sorted({str(tag) for tag in tags})


def bugfix_metrics(results: Sequence[ItemResult]) -> dict[str, Any]:
    """The bugfix arm metrics listed above; "primary" repeats resolved_rate."""
    n = len(results)
    resolved = _count(results, lambda r: r.correct)
    shown = _count(results, lambda r: r.shown_passed is True)
    overfit = _count(results, lambda r: r.shown_passed is True and r.hidden_passed is not True)
    block = Counter(rule for r in results for rule in set(r.sast_block))
    flag = Counter(rule for r in results for rule in set(r.sast_flag))
    by_tag: dict[str, dict[str, int]] = {}
    for row in results:
        for tag in _tags(row):
            entry = by_tag.setdefault(tag, {"items": 0, "resolved": 0})
            entry["items"] += 1
            entry["resolved"] += int(bool(row.correct))
    resolved_rate = _r6(rate(resolved, n))
    return {
        **_validity(results),
        "resolved": resolved,
        "resolved_rate": resolved_rate,
        "resolved_rate_ci95": _ci_json(wilson_interval(resolved, n)),
        "shown_pass_rate": _r6(rate(shown, n)),
        "hidden_pass_rate": _r6(rate(_count(results, lambda r: r.hidden_passed is True), n)),
        "overfit_rate": _r6(rate(overfit, shown)),
        "sast": {"block": dict(sorted(block.items())), "flag": dict(sorted(flag.items()))},
        **_cost(results),
        "by_tag": dict(sorted(by_tag.items())),
        "primary": resolved_rate,
    }


def _clusters(rows: Sequence[_T], key: Callable[[_T], str]) -> list[list[_T]]:
    """The rows grouped by cluster key, groups sorted by key (row order kept inside a group)."""
    groups: dict[str, list[_T]] = defaultdict(list)
    for row in rows:
        groups[key(row)].append(row)
    return [groups[k] for k in sorted(groups)]


def _resample(groups: Sequence[Sequence[_T]], statistic: Callable[[list[_T]], float | None], *, resamples: int, seed_key: str) -> list[float]:
    """The cluster bootstrap shared by the arm interval and the paired comparison: sorted values
    of `statistic` over `resamples` draws of len(groups) groups with replacement."""
    if not groups or resamples <= 0:
        return []
    rng = random.Random(seed_key)
    k = len(groups)
    values: list[float] = []
    for _ in range(resamples):
        drawn: list[_T] = []
        for _ in range(k):
            drawn.extend(groups[rng.randrange(k)])
        value = statistic(drawn)
        if value is not None:
            values.append(value)
    values.sort()
    return values


def _interval(values: Sequence[float]) -> tuple[float, float] | None:
    """The 95% percentile interval of sorted bootstrap values."""
    return (percentile(values, 0.025), percentile(values, 0.975)) if values else None


def _judge_ba(rows: Sequence[ItemResult]) -> float | None:
    paired = paired_tasks_only(rows)
    return balanced_accuracy([bool(r.label) for r in paired], [r.predicted for r in paired])


def judge_metrics(results: Sequence[ItemResult], *, resamples: int = DEFAULT_RESAMPLES, seed: int = DEFAULT_SEED) -> dict[str, Any]:
    """The judge arm metrics listed above; "primary" repeats balanced_accuracy."""
    paired = paired_tasks_only(results)
    out = classifier_metrics([bool(r.label) for r in paired], [r.predicted for r in paired])
    ci = _ci_json(_interval(_resample(_clusters(paired, lambda r: r.cluster), _judge_ba, resamples=resamples, seed_key=f"{seed}/judge/arm")))
    out.update(
        {
            "paired_items": len(paired),
            "paired_tasks": len({r.task_id for r in paired}),
            "unpaired_items": len(results) - len(paired),
            "balanced_accuracy_ci95": ci,
            # decided on the interval as the report shows it, so float noise at 0.5 cannot flip it
            "vs_chance": None if ci is None else ci[0] > 0.5,
            **_validity(results),
            **_cost(results),
            "primary": out["balanced_accuracy"],
        }
    )
    return out


def arm_metrics(feature: str, results: Sequence[ItemResult], *, resamples: int = DEFAULT_RESAMPLES, seed: int = DEFAULT_SEED) -> dict[str, Any]:
    """bugfix_metrics or judge_metrics, with {"items": n, "primary_metric": PRIMARY[feature]}."""
    if feature == "bugfix":
        body = bugfix_metrics(results)
    elif feature == "judge":
        body = judge_metrics(results, resamples=resamples, seed=seed)
    else:
        raise ValueError(f"unknown feature {feature!r}; known: {', '.join(sorted(PRIMARY))}")
    return {**body, "items": len(results), "primary_metric": PRIMARY[feature]}


def primary_of(feature: str, results: Sequence[ItemResult]) -> float | None:
    """The feature's primary metric on these rows (resolved rate, or balanced accuracy on the
    paired tasks among them). Unrounded."""
    if feature == "bugfix":
        return rate(_count(results, lambda r: r.correct), len(results))
    if feature == "judge":
        return _judge_ba(results)
    raise ValueError(f"unknown feature {feature!r}; known: {', '.join(sorted(PRIMARY))}")


# ---- the paired comparison -----------------------------------------------------------------------


def paired_bootstrap(
    clusters: Sequence[Sequence[tuple[ItemResult, ItemResult]]],
    delta_of: Callable[[Sequence[tuple[ItemResult, ItemResult]]], float | None],
    *,
    resamples: int,
    seed_key: str,
) -> list[float]:
    """Sorted bootstrap deltas: `resamples` times, draw len(clusters) clusters with replacement
    (random.Random(seed_key)), concatenate their (candidate, baseline) pairs and apply delta_of;
    draws where delta_of gives None are dropped."""
    return _resample(clusters, delta_of, resamples=resamples, seed_key=seed_key)


def _sample_sd(sorted_values: Sequence[float]) -> float | None:
    """The sample standard deviation (n - 1), exactly 0.0 when every value is the same; None
    for fewer than two values. math.fsum keeps it identical across platforms and versions."""
    m = len(sorted_values)
    if m < 2:
        return None
    if sorted_values[0] == sorted_values[-1]:
        return 0.0
    mean = math.fsum(sorted_values) / m
    return math.sqrt(math.fsum((v - mean) ** 2 for v in sorted_values) / (m - 1))


def _by_id(rows: Sequence[ItemResult], arm: str) -> dict[str, ItemResult]:
    out: dict[str, ItemResult] = {}
    for row in rows:
        if row.item_id in out:
            raise ValueError(f"the {arm} lists item {row.item_id} twice")
        out[row.item_id] = row
    return out


def compare(
    feature: str,
    candidate: Sequence[ItemResult],
    baseline: Sequence[ItemResult],
    *,
    split: str,
    resamples: int = DEFAULT_RESAMPLES,
    seed: int = DEFAULT_SEED,
    alpha: float = 0.05,
) -> Comparison:
    """The paired comparison described above. ValueError for an unknown feature, an item listed
    twice in one arm, or a joined item whose task, cluster or label differs between the arms
    (they were not scored on the same suite)."""
    if feature not in PRIMARY:
        raise ValueError(f"unknown feature {feature!r}; known: {', '.join(sorted(PRIMARY))}")
    cand, base = _by_id(candidate, "candidate"), _by_id(baseline, "baseline")
    common = sorted(cand.keys() & base.keys())
    pairs = [(cand[i], base[i]) for i in common]
    for c, b in pairs:
        if (c.task_id, c.cluster, c.label) != (b.task_id, b.cluster, b.label):
            raise ValueError(f"item {c.item_id}: the arms disagree on its task, cluster or label, so they were not scored on the same suite")

    def delta_of(chosen: Sequence[tuple[ItemResult, ItemResult]]) -> float | None:
        c_value = primary_of(feature, [c for c, _ in chosen])
        b_value = primary_of(feature, [b for _, b in chosen])
        return None if c_value is None or b_value is None else c_value - b_value

    gains = sum(1 for c, b in pairs if c.correct and not b.correct)
    losses = sum(1 for c, b in pairs if b.correct and not c.correct)
    clusters = _clusters(pairs, lambda pair: pair[0].cluster)
    deltas = paired_bootstrap(clusters, delta_of, resamples=resamples, seed_key=f"{seed}/{feature}/{split}")
    se = _sample_sd(deltas)
    return Comparison(
        feature=feature,
        metric=PRIMARY[feature],
        n=len(pairs),
        clusters=len(clusters),
        gains=gains,
        losses=losses,
        candidate=primary_of(feature, [c for c, _ in pairs]),
        baseline=primary_of(feature, [b for _, b in pairs]),
        delta=delta_of(pairs),
        ci95=_interval(deltas),
        se=se,
        mde80=(Z95 + Z80) * se if se else None,
        mcnemar_p=mcnemar_exact(losses, gains),
        min_significant_losses=min_significant_losses(alpha),
        alpha=alpha,
        resamples=resamples,
        seed=seed,
        unpaired=sorted(cand.keys() ^ base.keys()),
    )
