"""The promotion gate: hard gates on a candidate arm against a baseline arm, and the JSON report.

    check_comparable(candidate, baseline)   GateRefused unless both arms scored the same items
    decide(candidate, baseline, comparison, config) -> Decision
    report_warnings(...)                    what limits what the decision is worth
    build_report(...)                       the report: the only thing `llm-eval gate` prints

THE RULE "noninferior/1" (docs/DESIGN.md, "The gate"). Every gate below that applies is enforced;
`promote` is true when every enforced gate passed. A gate that does not apply stays in the report
with "enforced": false and "applicable": false, so every report of a feature has the same gates.

  schema_validity        candidate schema_validity >= config.schema_min (0.95)
  think_validity         candidate think_validity  >= config.think_min (0.90)
  new_sast_violations    count of (item, rule) pairs where the candidate's reply added a block
                         finding of `rule` and the baseline's reply for the same item added none
                         of it, <= config.max_new_sast (0). Without a baseline every block finding
                         counts. Applies to features that write code (WRITES_CODE: bugfix), and to
                         any arm whose rows carry block findings; for the judge it is reported
                         with "applicable": false and passes.
  primary                non-inferiority of the primary metric, on the paired items:
                             tie "pass" (default):  delta >= -margin
                             tie "fail":            delta >  -margin
                         margin defaults to 0.0, so by default the candidate may not be worse on
                         the point estimate, and an exact tie passes. --tie fail with margin 0
                         asks for a strict improvement. Comparisons use a tolerance of 1e-9, so
                         float noise never decides: |delta + margin| <= 1e-9 is a tie.
                         Without a baseline: candidate primary >= config.floor, or the feature's
                         DEFAULT_FLOORS entry ("bound_source": "floor"); the tie rule applies to
                         that bound the same way (> under tie "fail"). A primary value of None
                         (nothing scorable) fails.
  regression_test        not (losses > gains and mcnemar_p < alpha): even inside the margin, a
                         loss McNemar calls significant fails. For a plain rate with margin 0 it
                         cannot fire (delta >= 0 means losses <= gains); it matters when a margin
                         is set to absorb the noise of a live endpoint, and for balanced accuracy
                         on unbalanced classes, where items weigh by class. Not applicable (and
                         not enforced) without a baseline.
  gain_test              only with config.require_gain: gains > losses and mcnemar_p < alpha.
                         The rule for replacing a baseline when the change must be shown to help,
                         not just not to hurt. Without a baseline it fails: no gain can be shown.
                         Not in the report otherwise.

Each gate is {"gate", "value", "op", "bound", "passed", "enforced", ...detail}. op is ">=", ">",
"<=" (value against bound), "not" (regression_test: the condition in "condition" must not hold)
or "<" (gain_test: the p-value must be under alpha, and the gains must outnumber the losses).
Values are those of the report: rates rounded to 6 decimals, p-values unrounded.

Why the point estimate and not the confidence interval: on a suite of a few dozen items the 95%
interval of delta is wider than any margin worth setting, so "the interval's lower end is above
-margin" would fail nearly every candidate, and "the regression is significant" would pass nearly
every one. The gate decides on the point estimate with an explicit margin and tie rule, uses
McNemar as a second guard, and the report always carries the interval, the detectable effect and
min_significant_losses, so a reader sees how much the decision is worth.

REFUSED (GateRefused; the CLI exits 2 and prints no report): the arms are of different features
or splits, were scored on different suites, or their item sets differ (an unpaired comparison
would compare different questions), or an arm lists an item twice.

THE REPORT (build_report; REPORT_SCHEMA). Keys, all always present:
  schema, rule, feature, split
  suite        {"name", "sha256", "split_sha256", "tasks", "items", "heldout_pct", "synthetic"}
               (None where the caller gave no value; further keys the caller gives are kept)
  candidate    ArmRun.summary(); baseline the same, or null
  comparison   metrics.Comparison.to_json(), or null without a baseline
  gates        [{"gate", "value", "op", "bound", "passed", "enforced", ...detail}] in the order above
  gates_passed every enforced gate passed;  promote: the same value (the name CI reads: jq .promote)
  config       GateConfig.to_json()
  warnings     report_warnings(...)
  provenance   {"harness_version", "sast": sast.describe(), "exec_rule": verdicts.EXEC_RULE,
                "prompt_current": prompts.CURRENT[feature]} (the caller passes them; the
               harness version defaults to this package's)
The report holds no time stamp, host or path outside the working tree, so replaying the same
cassettes gives byte-identical output (tests/test_demo.py checks it).
"""

from __future__ import annotations

import copy
import math
import os
from dataclasses import asdict, dataclass, field
from typing import Any, Mapping, Sequence

from . import __version__
from .metrics import PRIMARY, Comparison
from .records import HELDOUT, ArmRun, ItemResult
from .util import display_path, round6

RULE = "noninferior/1"
REPORT_SCHEMA = "llm_eval_harness.report/1"
#: The primary metric's bound when there is no baseline (the first evaluation of a feature).
DEFAULT_FLOORS: Mapping[str, float] = {"bugfix": 0.5, "judge": 0.7, "review": 0.65, "patch": 0.7}
#: Distinct candidates scored on one held-out split after which the report warns that the split
#: is spent (ledger.heldout_openings counts them).
HELDOUT_BUDGET = 10
#: Fewer paired items than this, and the report warns that only large regressions are detectable.
SMALL_SUITE = 50
EPS = 1e-9
#: The features whose replies hold code, so that the new-SAST gate applies to them (the same set
#: as features.Feature.writes_code; a new code-writing feature is added here with its floor).
WRITES_CODE = frozenset({"bugfix", "patch"})
TIE_RULES = ("pass", "fail")
#: An approve rate at or beyond these marks a judge that barely looks at the patch.
RUBBER_STAMP_HIGH, RUBBER_STAMP_LOW = 0.9, 0.1
#: The candidate's mean completion tokens above this multiple of the baseline's is a cost warning.
COST_RATIO = 1.5
#: The keys of the report's suite and provenance blocks, always present.
SUITE_KEYS = ("name", "sha256", "split_sha256", "tasks", "items", "heldout_pct", "synthetic")
PROVENANCE_KEYS = ("harness_version", "sast", "exec_rule", "prompt_current")


class GateRefused(Exception):
    """The arms cannot be compared (see REFUSED)."""


def _is_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


@dataclass(frozen=True)
class GateConfig:
    """The gate's parameters (CLI flags of the same names)."""

    schema_min: float = 0.95
    think_min: float = 0.90
    max_new_sast: int = 0
    margin: float = 0.0
    tie: str = "pass"
    alpha: float = 0.05
    require_gain: bool = False
    floor: float | None = None

    def to_json(self) -> dict[str, Any]:
        return asdict(self)

    def validate(self) -> None:
        """ValueError unless 0 <= schema_min, think_min <= 1; max_new_sast >= 0; margin >= 0;
        tie in ("pass", "fail"); 0 < alpha < 1; floor None or in [0, 1]."""
        problems = []
        for name in ("schema_min", "think_min"):
            value = getattr(self, name)
            if not (_is_number(value) and 0.0 <= value <= 1.0):
                problems.append(f"{name} must be between 0 and 1, got {value!r}")
        if isinstance(self.max_new_sast, bool) or not isinstance(self.max_new_sast, int) or self.max_new_sast < 0:
            problems.append(f"max_new_sast must be a non-negative int, got {self.max_new_sast!r}")
        if not (_is_number(self.margin) and self.margin >= 0.0):
            problems.append(f"margin must be a non-negative number, got {self.margin!r}")
        if self.tie not in TIE_RULES:
            problems.append(f"tie must be one of {', '.join(TIE_RULES)}, got {self.tie!r}")
        if not (_is_number(self.alpha) and 0.0 < self.alpha < 1.0):
            problems.append(f"alpha must be strictly between 0 and 1, got {self.alpha!r}")
        if self.floor is not None and not (_is_number(self.floor) and 0.0 <= self.floor <= 1.0):
            problems.append(f"floor must be None or between 0 and 1, got {self.floor!r}")
        if not isinstance(self.require_gain, bool):
            problems.append(f"require_gain must be a bool, got {self.require_gain!r}")
        if problems:
            raise ValueError("; ".join(problems))


@dataclass
class Gate:
    """One gate's verdict. op: ">=", ">", "<=", "<" or "not" (see above); detail: extra keys for
    the report."""

    gate: str
    value: float | int | None
    op: str
    bound: float | int | None
    passed: bool
    enforced: bool = True
    detail: dict[str, Any] = field(default_factory=dict)

    def to_json(self) -> dict[str, Any]:
        out: dict[str, Any] = {
            "gate": self.gate,
            "value": self.value,
            "op": self.op,
            "bound": self.bound,
            "passed": self.passed,
            "enforced": self.enforced,
        }
        clash = sorted(out.keys() & self.detail.keys())
        if clash:
            raise ValueError(f"gate {self.gate}: detail may not set {', '.join(clash)}")
        out.update(self.detail)
        return out


@dataclass
class Decision:
    """decide()'s result."""

    gates: list[Gate]
    gates_passed: bool
    promote: bool


# ---- refusals ------------------------------------------------------------------------------------


def _examples(ids: Sequence[str], limit: int = 3) -> str:
    shown = ", ".join(ids[:limit])
    return shown + (f", ... ({len(ids) - limit} more)" if len(ids) > limit else "")


def _item_ids(arm: ArmRun, role: str) -> set[str]:
    ids: set[str] = set()
    for row in arm.results:
        if row.item_id in ids:
            raise GateRefused(f"the {role} lists item {row.item_id} twice")
        ids.add(row.item_id)
    return ids


def check_comparable(candidate: ArmRun, baseline: ArmRun | None) -> None:
    """GateRefused unless (with a baseline) both arms have the same feature, split, suite_sha256
    and set of item ids (each listed once)."""
    if baseline is None:
        return
    if candidate.feature != baseline.feature:
        raise GateRefused(f"the candidate was scored as {candidate.feature} and the baseline as {baseline.feature}")
    if candidate.split != baseline.split:
        raise GateRefused(f"the candidate was scored on the {candidate.split} split and the baseline on the {baseline.split} split")
    if candidate.suite_sha256 != baseline.suite_sha256:
        raise GateRefused(
            f"the arms were scored on different suites (candidate {candidate.suite_sha256[:12]}, baseline {baseline.suite_sha256[:12]})"
        )
    cand_ids, base_ids = _item_ids(candidate, "candidate"), _item_ids(baseline, "baseline")
    if cand_ids != base_ids:
        only_c, only_b = sorted(cand_ids - base_ids), sorted(base_ids - cand_ids)
        parts = []
        if only_c:
            parts.append(f"{len(only_c)} only in the candidate ({_examples(only_c)})")
        if only_b:
            parts.append(f"{len(only_b)} only in the baseline ({_examples(only_b)})")
        raise GateRefused("the arms were not scored on the same items: " + "; ".join(parts))


# ---- the gates -----------------------------------------------------------------------------------


def _r6(value: float | None) -> float | None:
    return None if value is None else round6(value) + 0.0


def new_sast_violations(candidate: Sequence[ItemResult], baseline: Sequence[ItemResult] | None) -> list[dict[str, str]]:
    """[{"item": item id, "rule": rule id}] sorted, as the new_sast_violations gate counts them.
    Only block findings count; a baseline without a row for the item had none of the rule."""
    baseline_rules: dict[str, set[str]] | None = None
    if baseline is not None:
        baseline_rules = {}
        for row in baseline:
            baseline_rules.setdefault(row.item_id, set()).update(row.sast_block)
    found = set()
    for row in candidate:
        for rule in row.sast_block:
            if baseline_rules is None or rule not in baseline_rules.get(row.item_id, set()):
                found.add((row.item_id, rule))
    return [{"item": item, "rule": rule} for item, rule in sorted(found)]


def _passes(value: float | None, bound: float, *, strict: bool) -> bool:
    """value >= bound (strict: value > bound), where |value - bound| <= EPS counts as equal."""
    if value is None:
        return False
    difference = value - bound
    if abs(difference) <= EPS:
        return not strict
    return difference > 0


def _at_least(name: str, value: float | None, bound: float) -> Gate:
    return Gate(name, value, ">=", bound, _passes(value, bound, strict=False))


def _sast_gate(candidate: ArmRun, baseline: ArmRun | None, config: GateConfig) -> Gate:
    applicable = candidate.feature in WRITES_CODE or any(row.sast_block for row in candidate.results)
    if not applicable:
        return Gate(
            "new_sast_violations", None, "<=", config.max_new_sast, True, enforced=False,
            detail={"applicable": False, "reason": f"{candidate.feature} replies hold no code to scan"},
        )  # fmt: skip
    found = new_sast_violations(candidate.results, None if baseline is None else baseline.results)
    return Gate(
        "new_sast_violations", len(found), "<=", config.max_new_sast, len(found) <= config.max_new_sast,
        detail={"applicable": True, "against": "none" if baseline is None else "baseline", "violations": found},
    )  # fmt: skip


def _floor(feature: str, config: GateConfig) -> float:
    if config.floor is not None:
        return float(config.floor)
    if feature not in DEFAULT_FLOORS:
        raise ValueError(f"no default floor for feature {feature!r}: give the gate a floor")
    return float(DEFAULT_FLOORS[feature])


def _primary_gate(candidate: ArmRun, comparison: Comparison | None, config: GateConfig) -> Gate:
    metric = PRIMARY[candidate.feature]
    strict = config.tie == "fail"
    detail: dict[str, Any] = {"metric": metric, "tie": config.tie}
    if comparison is not None:
        value, bound = comparison.delta, 0.0 - config.margin
        detail.update(
            bound_source="baseline", compares="delta = candidate - baseline", candidate=_r6(comparison.candidate),
            baseline=_r6(comparison.baseline), margin=config.margin,
        )  # fmt: skip
    else:
        value = candidate.metrics.get(metric, candidate.metrics.get("primary"))
        bound = _floor(candidate.feature, config)
        detail.update(bound_source="floor", compares=f"candidate {metric}")
    detail["at_bound"] = value is not None and abs(value - bound) <= EPS
    return Gate("primary", _r6(value), ">" if strict else ">=", _r6(bound), _passes(value, bound, strict=strict), detail=detail)


_MCNEMAR = "exact two-sided McNemar on the discordant items"


def _regression_gate(comparison: Comparison | None, config: GateConfig) -> Gate:
    condition = "losses > gains and mcnemar_p < alpha"
    if comparison is None:
        return Gate(
            "regression_test", None, "not", config.alpha, True, enforced=False,
            detail={"applicable": False, "condition": condition, "reason": "no baseline: nothing to compare with"},
        )  # fmt: skip
    fires = comparison.losses > comparison.gains and comparison.mcnemar_p < config.alpha
    return Gate(
        "regression_test", comparison.mcnemar_p, "not", config.alpha, not fires,
        detail={"applicable": True, "condition": condition, "losses": comparison.losses, "gains": comparison.gains, "test": _MCNEMAR},
    )  # fmt: skip


def _gain_gate(comparison: Comparison | None, config: GateConfig) -> Gate:
    """Asked for explicitly, so it is always enforced: without a baseline it fails."""
    condition = "gains > losses and mcnemar_p < alpha"
    if comparison is None:
        return Gate("gain_test", None, "<", config.alpha, False, detail={"condition": condition, "reason": "no baseline: no gain can be shown"})
    shown = comparison.gains > comparison.losses and comparison.mcnemar_p < config.alpha
    return Gate(
        "gain_test", comparison.mcnemar_p, "<", config.alpha, shown,
        detail={"condition": condition, "gains": comparison.gains, "losses": comparison.losses, "test": _MCNEMAR},
    )  # fmt: skip


def decide(candidate: ArmRun, baseline: ArmRun | None, comparison: Comparison | None, config: GateConfig) -> Decision:
    """The gates of RULE, in order. Pure: reads its arguments, writes nothing. ValueError for an
    invalid config, a comparison without a baseline (or the reverse), or a comparison of another
    feature."""
    config.validate()
    if (baseline is None) != (comparison is None):
        raise ValueError("decide needs a comparison exactly when it has a baseline")
    if candidate.feature not in PRIMARY:
        raise ValueError(f"unknown feature {candidate.feature!r}; known: {', '.join(sorted(PRIMARY))}")
    if comparison is not None and comparison.feature != candidate.feature:
        raise ValueError(f"the comparison is of {comparison.feature}, the candidate of {candidate.feature}")
    gates = [
        _at_least("schema_validity", candidate.metrics.get("schema_validity"), config.schema_min),
        _at_least("think_validity", candidate.metrics.get("think_validity"), config.think_min),
        _sast_gate(candidate, baseline, config),
        _primary_gate(candidate, comparison, config),
        _regression_gate(comparison, config),
    ]
    if config.require_gain:
        gates.append(_gain_gate(comparison, config))
    passed = all(gate.passed for gate in gates if gate.enforced)
    return Decision(gates=gates, gates_passed=passed, promote=passed)


# ---- warnings ------------------------------------------------------------------------------------


def _interval_text(ci: Sequence[float] | None) -> str:
    return "n/a" if ci is None else f"[{ci[0]:.3f}, {ci[1]:.3f}]"


def _judge_warnings(role: str, metrics: Mapping[str, Any], *, feature: str = "judge") -> list[str]:
    out = []
    noun = "reviewer" if feature == "review" else "judge"
    approve = metrics.get("approve_rate")
    if approve is not None and approve >= RUBBER_STAMP_HIGH - EPS:
        out.append(f"The {role} {noun} approves {approve:.3f} of the patches it sees: a rubber stamp, whatever the patch.")
    elif approve is not None and approve <= RUBBER_STAMP_LOW + EPS:
        out.append(f"The {role} {noun} approves only {approve:.3f} of the patches it sees: it rejects nearly every patch, correct or not.")
    ci = metrics.get("balanced_accuracy_ci95")
    value = metrics.get("balanced_accuracy")
    if ci is not None and value is not None:
        if ci[0] <= 0.5 <= ci[1]:
            out.append(f"The {role} {noun}'s balanced accuracy {value:.3f} has a 95% interval {_interval_text(ci)} that includes 0.5: it is not distinguishable from chance.")
        elif ci[1] < 0.5:
            out.append(f"The {role} {noun}'s balanced accuracy {value:.3f} has a 95% interval {_interval_text(ci)} below 0.5: it is worse than chance.")
    return out


def report_warnings(
    candidate: ArmRun,
    baseline: ArmRun | None,
    comparison: Comparison | None,
    config: GateConfig,
    *,
    current_prompt: str | None = None,
    heldout_openings: int | None = None,
) -> list[str]:
    """Sentences that change no gate but bound what the decision says, in this order:
      - an arm is synthetic (made-up responses, not a model's);
      - fewer than SMALL_SUITE paired items: name min_significant_losses and mde80 (without a
        baseline: fewer than SMALL_SUITE items, with the primary metric's interval);
      - no discordant items: the candidate answered every item as the baseline did;
      - the candidate was recorded with another prompt than current_prompt;
      - an arm's notes (suite drift: the cassettes were recorded on another suite version);
      - heldout_openings >= HELDOUT_BUDGET on the held-out split: the held-out split has been
        used to choose between that many candidates and no longer measures generalisation cleanly;
      - judge (an arm with an approve_rate), for each arm: an approve_rate of 0.9 or more, or 0.1
        or less (a rubber stamp), or a balanced accuracy whose interval includes 0.5 (not
        distinguishable from chance) or lies below it;
      - the candidate's mean completion tokens more than 1.5 times the baseline's (cost)."""
    arms = [("candidate", candidate)] + ([("baseline", baseline)] if baseline is not None else [])
    out: list[str] = []

    synthetic = [role for role, arm in arms if arm.synthetic]
    if synthetic:
        who = " and ".join(synthetic)
        out.append(
            f"The {who} {'arm is' if len(synthetic) == 1 else 'arms are'} synthetic: the responses were written for the demo, not "
            "produced by a model, so this report shows how the harness decides, not how good any model is."
        )

    if comparison is not None:
        if comparison.n < SMALL_SUITE:
            if comparison.mde80 is not None:
                mde = f"{comparison.mde80:.2g}"
            elif comparison.gains + comparison.losses == 0:
                mde = "n/a (no discordant items)"
            else:  # every item discordant the same way: each bootstrap draw gives the same delta
                mde = "n/a (every bootstrap draw gave the same delta)"
            out.append(
                f"{comparison.n} paired items: with no gains, at least {comparison.min_significant_losses} losses are needed before "
                f"McNemar calls a regression significant at alpha {comparison.alpha:g}; mde80 {mde}."
            )
        if comparison.n and comparison.gains + comparison.losses == 0:
            out.append(
                f"No discordant items: on all {comparison.n} items the candidate was right exactly where the baseline was, "
                "so the paired tests have nothing to work with."
            )
    else:
        n = candidate.metrics.get("items", len(candidate.results))
        if n < SMALL_SUITE:
            metric = PRIMARY.get(candidate.feature, "primary")
            ci = candidate.metrics.get(f"{metric}_ci95")
            out.append(f"{n} items and no baseline: the floor is compared with a point estimate whose 95% interval is {_interval_text(ci)}.")

    if current_prompt is not None and candidate.prompt != current_prompt:
        out.append(f"The candidate was recorded with prompt {candidate.prompt}, not the current {current_prompt}.")

    for role, arm in arms:
        out.extend(f"{role}: {note}" for note in arm.notes)

    if heldout_openings is not None and candidate.split == HELDOUT and heldout_openings >= HELDOUT_BUDGET:
        out.append(
            f"The held-out split has been used to judge {heldout_openings} distinct candidates (budget {HELDOUT_BUDGET}): it has been "
            "used to choose between that many candidates and no longer measures generalisation cleanly; add fresh held-out tasks."
        )

    for role, arm in arms:
        if "approve_rate" in arm.metrics:
            out.extend(_judge_warnings(role, arm.metrics, feature=arm.feature))

    if baseline is not None:
        c_tokens = candidate.metrics.get("completion_tokens_mean")
        b_tokens = baseline.metrics.get("completion_tokens_mean")
        if c_tokens is not None and b_tokens and c_tokens > COST_RATIO * b_tokens:
            out.append(
                f"Cost: the candidate's mean completion tokens per item ({c_tokens:.0f}) are {c_tokens / b_tokens:.1f} times "
                f"the baseline's ({b_tokens:.0f})."
            )
    return out


# ---- the report ----------------------------------------------------------------------------------


def _arm_block(arm: ArmRun) -> dict[str, Any]:
    """ArmRun.summary(), copied, with a source that is never an absolute path."""
    block = copy.deepcopy(arm.summary())
    source = block.get("source")
    if isinstance(source, str) and os.path.isabs(source):
        block["source"] = display_path(source)
    return block


def build_report(
    *,
    feature: str,
    split: str,
    suite: Mapping[str, Any],
    candidate: ArmRun,
    baseline: ArmRun | None,
    comparison: Comparison | None,
    decision: Decision,
    config: GateConfig,
    warnings: Sequence[str],
    provenance: Mapping[str, Any],
) -> dict[str, Any]:
    """The report dict (see THE REPORT). ValueError when the candidate is of another feature or
    split than the report."""
    if candidate.feature != feature or candidate.split != split:
        raise ValueError(f"the candidate is a {candidate.feature}/{candidate.split} run, the report {feature}/{split}")
    suite_block = {key: copy.deepcopy(suite.get(key)) for key in SUITE_KEYS}
    suite_block.update({key: copy.deepcopy(value) for key, value in suite.items() if key not in suite_block})
    provenance_block: dict[str, Any] = dict.fromkeys(PROVENANCE_KEYS)
    provenance_block["harness_version"] = __version__
    provenance_block.update(copy.deepcopy(dict(provenance)))
    return {
        "schema": REPORT_SCHEMA,
        "rule": RULE,
        "feature": feature,
        "split": split,
        "suite": suite_block,
        "candidate": _arm_block(candidate),
        "baseline": None if baseline is None else _arm_block(baseline),
        "comparison": None if comparison is None else comparison.to_json(),
        "gates": [gate.to_json() for gate in decision.gates],
        "gates_passed": decision.gates_passed,
        "promote": decision.promote,
        "config": config.to_json(),
        "warnings": list(warnings),
        "provenance": provenance_block,
    }
