"""report: the exact JSON text, loading a report file, and the markdown summary CI publishes.

The reports are made by the real gate (decide, report_warnings, build_report) on rows built by hand.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from llm_eval_harness.gate import REPORT_SCHEMA, GateConfig, build_report, decide, report_warnings
from llm_eval_harness.metrics import arm_metrics, compare
from llm_eval_harness.records import ArmRun, ItemResult
from llm_eval_harness.report import DEFAULT_TITLE, ReportError, dumps_report, load_report, markdown_summary
from llm_eval_harness.util import sha256_text

OVERVIEW = "| Feature | Candidate | Baseline | Primary (baseline -> candidate) | Delta [95% CI] | Gates | Verdict |"
BANNER = "**Synthetic demo data: these responses were written for the demo, not produced by a model.**"
SUITE = {"name": "tiny", "sha256": "c" * 64, "split_sha256": "d" * 64, "tasks": 8, "items": 8, "heldout_pct": 50, "synthetic": True}


def bug_row(task: str, correct: bool, *, schema: bool = True, block: tuple[str, ...] = ()) -> ItemResult:
    return ItemResult(
        item_id=f"bugfix:{task}", feature="bugfix", task_id=task, cluster=task, split="heldout", correct=correct, schema_valid=schema,
        lenient_valid=schema, think_present=False, think_valid=True, think_words=0, shown_passed=correct, hidden_passed=correct,
        sast_block=block,
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


def make_arm(name: str, feature: str, rows: list[ItemResult], *, model: str = "model-a", prompt: str = "bugfix/v1", synthetic: bool = True) -> ArmRun:
    return ArmRun(
        name=name, feature=feature, split="heldout", source=f"cassettes/{name}", fingerprint=sha256_text(name), model=model, prompt=prompt,
        prompt_sha256="0" * 64, synthetic=synthetic, suite_sha256="e" * 64, recorded_suite_sha256="e" * 64, results=rows,
        metrics=arm_metrics(feature, rows, resamples=200),
    )  # fmt: skip


def gate_report(candidate: ArmRun, baseline: ArmRun | None, config: GateConfig = GateConfig()) -> dict[str, Any]:
    comparison = None if baseline is None else compare(candidate.feature, candidate.results, baseline.results, split="heldout", resamples=200)
    decision = decide(candidate, baseline, comparison, config)
    warnings = report_warnings(candidate, baseline, comparison, config, current_prompt=f"{candidate.feature}/v2")
    return build_report(feature=candidate.feature, split="heldout", suite=SUITE, candidate=candidate, baseline=baseline, comparison=comparison,
                        decision=decision, config=config, warnings=warnings,
                        provenance={"sast": None, "exec_rule": "values/1", "prompt_current": f"{candidate.feature}/v2"})  # fmt: skip


def three_reports(*, synthetic: bool = True) -> list[dict[str, Any]]:
    baseline = make_arm("baseline", "bugfix", [bug_row(f"t{k}", k % 2 == 0) for k in range(8)], synthetic=synthetic)
    better = make_arm("candidate", "bugfix", [bug_row(f"t{k}", k != 7) for k in range(8)], model="model-b", prompt="bugfix/v2", synthetic=synthetic)
    regressed_rows = [bug_row("t0", False, schema=False), bug_row("t1", False, block=("eval-exec",))] + [bug_row(f"t{k}", False) for k in range(2, 8)]
    regressed = make_arm("candidate-regressed", "bugfix", regressed_rows, model="model-c", prompt="bugfix/v2", synthetic=synthetic)
    judge = make_arm("candidate", "judge", [r for k in range(8) for r in judge_rows(f"t{k}", True, k % 2 == 0)], prompt="judge/v2", synthetic=synthetic)
    return [gate_report(better, baseline), gate_report(regressed, baseline), gate_report(judge, None)]


# ---- the JSON text -------------------------------------------------------------------------------


def test_dumps_report_is_deterministic() -> None:
    first = {"schema": REPORT_SCHEMA, "b": [1, {"y": 2, "x": "é"}], "a": None}
    second = {"a": None, "b": [1, {"x": "é", "y": 2}], "schema": REPORT_SCHEMA}
    text = dumps_report(first)
    assert text == dumps_report(second)
    assert text.endswith("}\n") and not text.endswith("\n\n")
    assert text.splitlines()[1].startswith('  "a"')  # sorted keys, indent 2
    assert "é" in text  # non-ASCII kept as is
    assert json.loads(text) == first


def test_dumps_report_refuses_nan() -> None:
    with pytest.raises(ValueError):
        dumps_report({"schema": REPORT_SCHEMA, "value": float("nan")})


def test_a_gate_report_dumps_to_the_same_bytes_every_time() -> None:
    assert [dumps_report(r) for r in three_reports()] == [dumps_report(r) for r in three_reports()]


def test_load_report_round_trip(tmp_path: Path) -> None:
    report = three_reports()[0]
    path = tmp_path / "report.json"
    path.write_text(dumps_report(report), encoding="utf-8")
    assert load_report(path) == report


@pytest.mark.parametrize(
    "content, reason",
    [
        ("not json at all", "not JSON"),
        ("[1, 2]", "a JSON list"),
        ('{"schema": "something/else"}', "not a report of this harness"),
        ("{}", "not a report of this harness"),
        ('{"schema": "llm_eval_harness.report/1", "x": NaN}', "not JSON"),
    ],
)
def test_load_report_refuses_what_is_not_a_report(tmp_path: Path, content: str, reason: str) -> None:
    path = tmp_path / "bad.json"
    path.write_text(content, encoding="utf-8")
    with pytest.raises(ReportError, match=reason):
        load_report(path)


def test_load_report_refuses_binary_and_reports_a_missing_file(tmp_path: Path) -> None:
    path = tmp_path / "binary.json"
    path.write_bytes(b"\xff\xfe\x00garbage")
    with pytest.raises(ReportError):
        load_report(path)
    with pytest.raises(OSError):
        load_report(tmp_path / "missing.json")
    assert issubclass(ReportError, ValueError)


# ---- the markdown summary ------------------------------------------------------------------------


def test_summary_overview() -> None:
    text = markdown_summary(three_reports())
    lines = text.splitlines()
    assert lines[0] == f"# {DEFAULT_TITLE}"
    assert lines[2] == BANNER  # the first line under the heading
    assert text.count(OVERVIEW) == 1
    at = lines.index(OVERVIEW)
    assert lines[at + 1] == "|---|---|---|---|---|---|---|"
    rows = lines[at + 2 : at + 5]
    assert rows[0] == (
        "| bugfix | candidate (model-b, bugfix/v2, synthetic) | baseline (model-a, bugfix/v1, synthetic) "
        f"| resolved_rate: 0.500 -> 0.875 | {rows[0].split(' | ')[4]} | 5/5 | PROMOTE |"
    )
    assert rows[0].split(" | ")[4].startswith("+0.375 [")
    assert rows[1].startswith("| bugfix | candidate-regressed (model-c, bugfix/v2, synthetic) |") and rows[1].endswith("| REJECT |")
    assert "| resolved_rate: 0.500 -> 0.000 | -0.500 [" in rows[1]
    assert rows[2].startswith("| judge | candidate (model-a, judge/v2, synthetic) | none | balanced_accuracy: floor 0.700 -> 0.750 | n/a | 3/3 |")
    assert rows[2].endswith("| PROMOTE |")
    for row in rows:
        assert row.count(" | ") == 6  # seven cells, none broken


def test_summary_sections() -> None:
    reports = three_reports()
    text = markdown_summary(reports)
    for heading in ("## bugfix: candidate vs baseline", "## bugfix: candidate-regressed vs baseline", "## judge: candidate vs none"):
        assert heading in text.splitlines()
    assert text.count("| Gate | Value | Rule | Result |") == 3
    assert "| schema_validity | 1.000 | >= 0.950 | PASS |" in text
    assert "| schema_validity | 0.875 | >= 0.950 | FAIL |" in text
    assert "| new_sast_violations | 1 | <= 0 | FAIL |" in text
    assert "New SAST violations: bugfix:t1 (eval-exec)." in text
    assert "| primary | +0.375 | delta >= 0.000 | PASS |" in text
    assert "| primary | 0.750 | >= 0.700 (floor) | PASS |" in text
    assert "| new_sast_violations | n/a | <= 0 | n/a |" in text  # the judge writes no code
    assert "| regression_test | n/a | not (losses > gains and p < 0.05) | n/a |" in text  # no baseline
    regressed = reports[1]["comparison"]
    assert f"gains {regressed['gains']}, losses {regressed['losses']}; McNemar p {regressed['mcnemar_p']:.4g}; min_significant_losses 6 at alpha 0.05" in text
    assert "No baseline: the primary metric was compared with the floor, and there are no paired tests." in text
    assert "\n- The candidate and baseline arms are synthetic" in text
    assert "Split heldout of suite tiny: 8 tasks, 8 items. Rule noninferior/1; verdict REJECT." in text


def test_summary_is_deterministic_plain_text() -> None:
    text = markdown_summary(three_reports())
    assert text == markdown_summary(three_reports())
    assert text.endswith("\n") and not text.endswith("\n\n")
    assert text.isascii()  # plain words, no emoji
    assert "PROMOTE" in text and "REJECT" in text


def test_summary_without_synthetic_arms_has_no_banner() -> None:
    text = markdown_summary(three_reports(synthetic=False), title="Gate")
    assert text.startswith("# Gate\n\n| Feature |")
    assert BANNER not in text


def test_summary_of_no_reports() -> None:
    assert markdown_summary([], title="Gate") == "# Gate\n\nNo reports.\n"


def test_summary_prints_none_as_na_and_escapes_cells() -> None:
    report = three_reports()[2]
    report["candidate"]["model"] = "org|model"
    report["gates"][0]["value"] = None
    text = markdown_summary([report])
    assert "org\\|model" in text
    assert "| schema_validity | n/a | >= 0.950 | PASS |" in text
    row = next(line for line in text.splitlines() if line.startswith("| judge |"))
    assert row.replace("\\|", "").count(" | ") == 6


def test_summary_of_loaded_reports(tmp_path: Path) -> None:
    paths = []
    for k, report in enumerate(three_reports()):
        path = tmp_path / f"r{k}.json"
        path.write_text(dumps_report(report), encoding="utf-8")
        paths.append(path)
    assert markdown_summary([load_report(p) for p in paths]) == markdown_summary(three_reports())
