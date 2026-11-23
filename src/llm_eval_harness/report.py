"""The report's JSON form, and a markdown summary of one or more reports for CI.

    dumps_report(report)            the exact text `llm-eval gate` prints on stdout
    load_report(path)               a report file, checked to be one
    markdown_summary(reports, ...)  the text CI appends to $GITHUB_STEP_SUMMARY

dumps_report: json.dumps(report, indent=2, sort_keys=True, ensure_ascii=False) plus a final
newline. Deterministic: the same report always gives the same bytes. NaN and infinities are
refused (ValueError): they are not JSON, and jq would refuse the report.

markdown_summary, for reports in the given order:
  - a heading (`title`), and a bold first line "Synthetic demo data: these responses were written
    for the demo, not produced by a model." when any arm of any report is synthetic;
  - one overview table, a row per report:
        | Feature | Candidate | Baseline | Primary (baseline -> candidate) | Delta [95% CI] | Gates | Verdict |
    with model and prompt in the arm cells, rates to 3 decimals, gates as "passed/enforced", and
    the verdict PROMOTE or REJECT (plain words, no emoji). Without a baseline the Primary cell
    shows the floor in the baseline's place;
  - per report, a "<feature>: <candidate> vs <baseline>" section with the split, a gate table
        | Gate | Value | Rule | Result |      (Result: PASS / FAIL / n/a for a gate not enforced)
    the new SAST violations, the paired counts (gains, losses, McNemar p,
    min_significant_losses) and the warnings as a list.
  Values that are None print as "n/a". The summary holds nothing that is not in the reports, and
  the same reports always give the same text.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Mapping, Sequence

from .gate import REPORT_SCHEMA
from .util import display_path

DEFAULT_TITLE = "LLM regression gate"
SYNTHETIC_BANNER = "**Synthetic demo data: these responses were written for the demo, not produced by a model.**"
OVERVIEW_HEADER = "| Feature | Candidate | Baseline | Primary (baseline -> candidate) | Delta [95% CI] | Gates | Verdict |"
GATE_HEADER = "| Gate | Value | Rule | Result |"
NA = "n/a"


class ReportError(ValueError):
    """The file is not a report this harness wrote (bad JSON, wrong schema)."""


def dumps_report(report: Mapping[str, Any]) -> str:
    """See above."""
    return json.dumps(report, indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False) + "\n"


def _refuse_constant(name: str) -> Any:
    raise ValueError(f"{name} is not JSON")


def load_report(path: str | os.PathLike[str]) -> dict[str, Any]:
    """The report in `path`; ReportError unless it is a JSON object whose "schema" is
    gate.REPORT_SCHEMA. A file that cannot be read raises OSError."""
    shown = display_path(path)
    try:
        text = Path(path).read_text(encoding="utf-8")
    except UnicodeDecodeError:
        raise ReportError(f"{shown}: not UTF-8 text, so not a report") from None
    try:
        value = json.loads(text, parse_constant=_refuse_constant)
    except ValueError as exc:
        reason = exc.msg if isinstance(exc, json.JSONDecodeError) else str(exc)
        raise ReportError(f"{shown}: not JSON ({reason}), so not a report") from None
    if not isinstance(value, dict):
        raise ReportError(f"{shown}: a JSON {type(value).__name__}, not a report object")
    if value.get("schema") != REPORT_SCHEMA:
        raise ReportError(f"{shown}: its schema is {value.get('schema')!r}, not {REPORT_SCHEMA!r}: not a report of this harness")
    return value


# ---- formatting ---------------------------------------------------------------------------------


def _cell(text: Any) -> str:
    """Text that cannot break a markdown table row."""
    return " ".join(str(text).split()).replace("|", "\\|")


def _rate(value: Any) -> str:
    return NA if value is None else f"{value:.3f}"


def _signed(value: Any) -> str:
    return NA if value is None else f"{value:+.3f}"


def _p(value: Any) -> str:
    return NA if value is None else f"{value:.4g}"


def _number(value: Any) -> str:
    if value is None:
        return NA
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float):
        return f"{value:.3f}"
    return str(value)


def _enforced(report: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    return [g for g in report.get("gates") or [] if g.get("enforced", True)]


def _gate(report: Mapping[str, Any], name: str) -> Mapping[str, Any] | None:
    return next((g for g in report.get("gates") or [] if g.get("gate") == name), None)


def _is_synthetic(report: Mapping[str, Any]) -> bool:
    return any(isinstance(arm, Mapping) and arm.get("synthetic") is True for arm in (report.get("candidate"), report.get("baseline")))


def _arm_cell(arm: Mapping[str, Any] | None) -> str:
    if not arm:
        return "none"
    details = [str(arm.get("model") or NA), str(arm.get("prompt") or NA)] + (["synthetic"] if arm.get("synthetic") else [])
    return f"{arm.get('name') or NA} ({', '.join(details)})"


def _metric_name(report: Mapping[str, Any]) -> str:
    comparison = report.get("comparison") or {}
    candidate = report.get("candidate") or {}
    primary = _gate(report, "primary") or {}
    return str(comparison.get("metric") or primary.get("metric") or (candidate.get("metrics") or {}).get("primary_metric") or "primary")


def _primary_cell(report: Mapping[str, Any]) -> str:
    comparison = report.get("comparison")
    metric = _metric_name(report)
    if comparison:
        return f"{metric}: {_rate(comparison.get('baseline'))} -> {_rate(comparison.get('candidate'))}"
    candidate_metrics = (report.get("candidate") or {}).get("metrics") or {}
    floor = (_gate(report, "primary") or {}).get("bound")
    return f"{metric}: floor {_rate(floor)} -> {_rate(candidate_metrics.get(metric, candidate_metrics.get('primary')))}"


def _delta_cell(report: Mapping[str, Any]) -> str:
    comparison = report.get("comparison")
    if not comparison:
        return NA
    ci = comparison.get("ci95")
    interval = NA if not ci else f"{_signed(ci[0])}, {_signed(ci[1])}"
    return f"{_signed(comparison.get('delta'))} [{interval}]"


def _verdict(report: Mapping[str, Any]) -> str:
    return "PROMOTE" if report.get("promote") is True else "REJECT"


def _gates_cell(report: Mapping[str, Any]) -> str:
    enforced = _enforced(report)
    return f"{sum(1 for g in enforced if g.get('passed') is True)}/{len(enforced)}"


def _gate_value(gate: Mapping[str, Any]) -> str:
    value = gate.get("value")
    if value is None:
        return NA
    if gate.get("op") in ("not", "<"):
        counts = ""
        if "losses" in gate and "gains" in gate:
            counts = f" ({gate['losses']} losses, {gate['gains']} gains)"
        return f"p = {_p(value)}{counts}"
    if gate.get("gate") == "primary" and gate.get("bound_source") == "baseline":
        return _signed(value)
    return _number(value)


def _gate_rule(gate: Mapping[str, Any]) -> str:
    op, bound = gate.get("op"), gate.get("bound")
    if op == "not":
        return f"not (losses > gains and p < {_p(bound)})"
    if op == "<":
        return f"gains > losses and p < {_p(bound)}"
    if gate.get("gate") == "primary":
        if gate.get("bound_source") == "baseline":
            return f"delta {op} {_number(bound)}"
        return f"{op} {_number(bound)} (floor)"
    return f"{op} {_number(bound)}"


def _gate_result(gate: Mapping[str, Any]) -> str:
    if not gate.get("enforced", True):
        return NA
    return "PASS" if gate.get("passed") is True else "FAIL"


def _section(report: Mapping[str, Any]) -> list[str]:
    candidate = report.get("candidate") or {}
    baseline = report.get("baseline")
    suite = report.get("suite") or {}
    lines = [
        f"## {report.get('feature') or NA}: {candidate.get('name') or NA} vs {(baseline or {}).get('name') or 'none'}",
        "",
        f"Split {report.get('split') or NA} of suite {suite.get('name') or NA}: {_number(suite.get('tasks'))} tasks, "
        f"{_number(suite.get('items'))} items. Rule {report.get('rule') or NA}; verdict {_verdict(report)}.",
        "",
        GATE_HEADER,
        "|---|---|---|---|",
    ]
    for gate in report.get("gates") or []:
        lines.append(f"| {_cell(gate.get('gate'))} | {_cell(_gate_value(gate))} | {_cell(_gate_rule(gate))} | {_gate_result(gate)} |")

    violations = (_gate(report, "new_sast_violations") or {}).get("violations") or []
    if violations:
        listed = "; ".join(f"{v.get('item')} ({v.get('rule')})" for v in violations)
        lines += ["", f"New SAST violations: {listed}."]

    comparison = report.get("comparison")
    lines.append("")
    if comparison:
        mde = comparison.get("mde80")
        lines.append(
            f"Paired: {_number(comparison.get('n'))} items in {_number(comparison.get('clusters'))} clusters; "
            f"gains {_number(comparison.get('gains'))}, losses {_number(comparison.get('losses'))}; "
            f"McNemar p {_p(comparison.get('mcnemar_p'))}; min_significant_losses {_number(comparison.get('min_significant_losses'))} "
            f"at alpha {_p(comparison.get('alpha'))}; mde80 {_rate(mde)}."
        )
    else:
        lines.append("No baseline: the primary metric was compared with the floor, and there are no paired tests.")

    warnings = report.get("warnings") or []
    lines.append("")
    if warnings:
        lines += ["Warnings:", ""] + [f"- {' '.join(str(w).split())}" for w in warnings]
    else:
        lines.append("Warnings: none.")
    return lines


def markdown_summary(reports: Sequence[Mapping[str, Any]], *, title: str = DEFAULT_TITLE) -> str:
    """See above. Ends with a newline."""
    lines = [f"# {' '.join(title.split())}", ""]
    if any(_is_synthetic(r) for r in reports):
        lines += [SYNTHETIC_BANNER, ""]
    if not reports:
        lines.append("No reports.")
        return "\n".join(lines) + "\n"
    lines += [OVERVIEW_HEADER, "|---|---|---|---|---|---|---|"]
    for report in reports:
        cells = [
            report.get("feature") or NA,
            _arm_cell(report.get("candidate")),
            _arm_cell(report.get("baseline")),
            _primary_cell(report),
            _delta_cell(report),
            _gates_cell(report),
            _verdict(report),
        ]
        lines.append("| " + " | ".join(_cell(c) for c in cells) + " |")
    for report in reports:
        lines += [""] + _section(report)
    return "\n".join(lines) + "\n"
