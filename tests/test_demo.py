"""The demo, end to end, the way CI runs it: the committed suite, cassettes and ledger through the
real command line (python -m llm_eval_harness in a child process, from the repository root).

The eight gate runs below (each feature, each candidate arm, each split, against the ledger's
current baseline) are made once per module and shared. They pin the demo's outcomes:

              candidate    candidate-regressed
    bugfix    promote      reject: primary, schema_validity, new_sast_violations
    judge     promote      reject: primary, schema_validity

on both splits, and the held-out numbers behind them, which README.md and docs/DESIGN.md quote
and scripts/make_demo_cassettes.py documents. A change to the harness that moves one of them
fails here first; update the documents together with the pinned values.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

import pytest

from llm_eval_harness import cassettes, tasks

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
SCRIPT = ROOT / "scripts" / "make_demo_cassettes.py"
FEATURES = ("bugfix", "judge")
ARMS = ("candidate", "candidate-regressed")
SPLITS = ("heldout", "dev")
#: The enforced gates the regressed candidate fails (CI's self-test expects at least these).
EXPECTED_FAILED = {"bugfix": {"primary", "schema_validity", "new_sast_violations"}, "judge": {"primary", "schema_validity"}}
Key = tuple[str, str, str]  # (feature, arm, split)


def _env() -> dict[str, str]:
    env = {k: v for k, v in os.environ.items() if not k.startswith("LLM_EVAL_") and k.lower() not in {"http_proxy", "https_proxy", "all_proxy"}}
    env.update({"PYTHONPATH": str(SRC), "PYTHONDONTWRITEBYTECODE": "1"})
    return env


def llm_eval(*args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run([sys.executable, "-m", "llm_eval_harness", *args], cwd=ROOT, env=_env(),
                          capture_output=True, text=True, timeout=600)


def make_cassettes(*args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run([sys.executable, str(SCRIPT), *args], cwd=ROOT, env=_env(), capture_output=True, text=True, timeout=600)


def gate(feature: str, arm: str, split: str, *extra: str) -> subprocess.CompletedProcess[str]:
    """The command CI runs (with the default ledger, read only)."""
    return llm_eval("gate", "--feature", feature, "--split", split, "--candidate", f"cassettes/{arm}", "--baseline", "@current", *extra)


@pytest.fixture(scope="module")
def items_dir(tmp_path_factory: pytest.TempPathFactory) -> Path:
    return tmp_path_factory.mktemp("items")


@pytest.fixture(scope="module")
def runs(items_dir: Path) -> dict[Key, subprocess.CompletedProcess[str]]:
    keys = [(feature, arm, split) for split in SPLITS for feature in FEATURES for arm in ARMS]
    with ThreadPoolExecutor(max_workers=4) as pool:
        futures = {key: pool.submit(gate, *key, "--items-out", str(items_dir / "-".join(key))) for key in keys}
        return {key: future.result() for key, future in futures.items()}


@pytest.fixture(scope="module")
def reports(runs: dict[Key, subprocess.CompletedProcess[str]]) -> dict[Key, dict[str, Any]]:
    out = {}
    for key, result in runs.items():
        # Exit 0: the evaluation ran, whatever the verdict (the rejections exit 0 too).
        assert result.returncode == 0, f"{key}: exit {result.returncode}\n{result.stderr}"
        out[key] = json.loads(result.stdout)
    return out


def failed_gates(report: dict[str, Any]) -> set[str]:
    return {g["gate"] for g in report["gates"] if g["enforced"] and not g["passed"]}


def gate_of(report: dict[str, Any], name: str) -> dict[str, Any]:
    return next(g for g in report["gates"] if g["gate"] == name)


def failures(metrics: dict[str, Any], rate: str) -> int:
    """How many items a rate in the report leaves out (its rounding is to 6 decimals)."""
    return round((1 - metrics[rate]) * metrics["items"])


def items(items_dir: Path, key: Key, role: str = "candidate") -> dict[str, dict[str, Any]]:
    path = items_dir / "-".join(key) / f"{key[0]}.{role}.items.jsonl"
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    return {row["item_id"]: row for row in rows}


# ---- the verdicts -----------------------------------------------------------------------------


@pytest.mark.parametrize("split", SPLITS)
@pytest.mark.parametrize("feature", FEATURES)
def test_the_candidate_promotes(reports, feature: str, split: str) -> None:
    report = reports[feature, "candidate", split]
    assert report["promote"] is True and report["gates_passed"] is True
    assert failed_gates(report) == set()
    assert report["baseline"]["source"] == "cassettes/baseline"


@pytest.mark.parametrize("split", SPLITS)
@pytest.mark.parametrize("feature", FEATURES)
def test_the_regressed_candidate_is_rejected_by_the_expected_gates(reports, feature: str, split: str) -> None:
    report = reports[feature, "candidate-regressed", split]
    assert report["promote"] is False and report["gates_passed"] is False
    assert failed_gates(report) == EXPECTED_FAILED[feature]


# ---- the held-out numbers -----------------------------------------------------------------------


def test_bugfix_on_the_heldout_split(reports, items_dir: Path) -> None:
    candidate, regressed = reports["bugfix", "candidate", "heldout"], reports["bugfix", "candidate-regressed", "heldout"]
    assert candidate["suite"]["tasks"] == 21 and candidate["suite"]["items"] == 21
    base = candidate["baseline"]["metrics"]
    assert 0.55 <= base["resolved_rate"] <= 0.75 and base["resolved"] == 14

    # The candidate: a real but modest gain, with one loss, and clean replies.
    c = candidate["comparison"]
    assert (c["gains"], c["losses"]) == (4, 1) and c["gains"] >= c["losses"] + 2
    assert c["delta"] == pytest.approx(3 / 21, abs=1e-6) and c["mcnemar_p"] == pytest.approx(0.375)
    m = candidate["candidate"]["metrics"]
    assert m["resolved"] == 17 and m["schema_validity"] == 1.0 and m["think_validity"] >= 0.95
    assert m["sast"]["block"] == {} and gate_of(candidate, "new_sast_violations")["violations"] == []
    assert not any(w.startswith("Cost:") for w in candidate["warnings"])

    # The regressed candidate: losses, broken replies and an eval() patch.
    r = regressed["comparison"]
    assert (r["gains"], r["losses"]) == (0, 4)
    rm = regressed["candidate"]["metrics"]
    assert rm["resolved"] == 10
    assert failures(rm, "schema_validity") == 5 and round(rm["lenient_only_rate"] * rm["items"]) == 2
    assert rm["think_validity"] >= 0.90, "the regressed candidate is meant to fail other gates, not this one"
    assert gate_of(regressed, "new_sast_violations")["violations"] == [{"item": "bugfix:evaluate-sum", "rule": "eval-exec"}]
    assert gate_of(regressed, "regression_test")["passed"] is True, "4 losses against 0 gains are not significant at 21 items"


def test_bugfix_replies_are_scored_as_the_generator_meant(items_dir: Path) -> None:
    rows = items(items_dir, ("bugfix", "candidate-regressed", "heldout"))
    unsafe = rows["bugfix:evaluate-sum"]
    assert unsafe["sast_block"] == ["eval-exec"] and unsafe["schema_valid"] is True
    assert unsafe["shown_passed"] is None and unsafe["detail"]["exec"] is None, "code with a block finding is never run"
    truncated = [row for row in rows.values() if row["finish_reason"] == "length"]
    assert len(truncated) == 1 and not truncated[0]["think_valid"] and not truncated[0]["schema_valid"]
    lenient = sorted(row["item_id"] for row in rows.values() if row["lenient_valid"] and not row["schema_valid"])
    assert lenient == ["bugfix:average-rating", "bugfix:oldest-first"]  # held-out indexes 2 and 13: i % 11 == 2
    overfit = [row for row in rows.values() if row["shown_passed"] and not row["hidden_passed"]]
    assert overfit and all(not row["correct"] for row in overfit)
    baseline = items(items_dir, ("bugfix", "candidate", "heldout"), "baseline")
    assert all(row["schema_valid"] for row in baseline.values())
    assert sum(1 for row in baseline.values() if row["shown_passed"] and not row["hidden_passed"]) == 7


def test_judge_on_the_heldout_split(reports) -> None:
    candidate, regressed = reports["judge", "candidate", "heldout"], reports["judge", "candidate-regressed", "heldout"]
    assert candidate["suite"]["items"] == 42

    # The baseline approves nearly everything: a rubber stamp at chance level.
    base = candidate["baseline"]["metrics"]
    assert base["approve_rate"] >= 0.9 and base["confusion"]["tp"] + base["confusion"]["fp"] == 39
    assert 0.5 <= base["balanced_accuracy"] <= 0.65 and base["balanced_accuracy"] == pytest.approx(0.571429)
    assert base["vs_chance"] is False
    warnings = candidate["warnings"]
    assert any(w.startswith("The baseline judge approves 0.929") and "rubber stamp" in w for w in warnings)
    assert any(w.startswith("The baseline judge's balanced accuracy 0.571") and "not distinguishable from chance" in w for w in warnings)

    c = candidate["candidate"]["metrics"]
    assert c["balanced_accuracy"] >= 0.8 and c["balanced_accuracy"] == pytest.approx(0.880952)
    assert c["schema_validity"] == 1.0 and c["vs_chance"] is True
    assert (candidate["comparison"]["gains"], candidate["comparison"]["losses"]) == (15, 2)

    rm = regressed["candidate"]["metrics"]
    assert failures(rm, "schema_validity") == 4 and rm["schema_validity"] < 0.95
    assert rm["balanced_accuracy"] < base["balanced_accuracy"] and rm["balanced_accuracy"] == pytest.approx(0.428571)
    assert rm["think_validity"] >= 0.90
    assert gate_of(regressed, "new_sast_violations")["enforced"] is False, "the judge writes no code"
    assert any(w.startswith("The candidate judge's balanced accuracy 0.429") for w in regressed["warnings"])


def test_dev_split_numbers(reports) -> None:
    resolved = {arm: reports["bugfix", arm, "dev"]["candidate"]["metrics"]["resolved"] for arm in ARMS}
    assert reports["bugfix", "candidate", "dev"]["baseline"]["metrics"]["resolved"] == 9
    assert resolved == {"candidate": 10, "candidate-regressed": 4}
    assert reports["bugfix", "candidate-regressed", "dev"]["candidate"]["metrics"]["sast"]["block"] == {"eval-exec": 3}
    ba = {arm: reports["judge", arm, "dev"]["candidate"]["metrics"]["balanced_accuracy"] for arm in ARMS}
    assert reports["judge", "candidate", "dev"]["baseline"]["metrics"]["balanced_accuracy"] == pytest.approx(0.576923)
    assert ba == {"candidate": pytest.approx(0.846154), "candidate-regressed": pytest.approx(0.423077)}


# ---- what every report says about itself -----------------------------------------------------------


def test_every_arm_in_the_reports_is_marked_synthetic(reports) -> None:
    for key, report in reports.items():
        assert report["candidate"]["synthetic"] is True and report["baseline"]["synthetic"] is True, key
        assert report["suite"]["synthetic"] is True, key
        assert report["warnings"][0].startswith("The candidate and baseline arms are synthetic"), key


def _strings(value: Any) -> list[str]:
    """Every string in a JSON value, keys included."""
    if isinstance(value, str):
        return [value]
    if isinstance(value, dict):
        return [s for k, v in value.items() for s in (k, *_strings(v))]
    if isinstance(value, list):
        return [s for v in value for s in _strings(v)]
    return []


def test_the_reports_hold_no_absolute_path(runs, reports) -> None:
    for key, result in runs.items():
        assert str(ROOT) not in result.stdout and str(Path.home()) not in result.stdout, key
        assert not [s for s in _strings(reports[key]) if s.startswith(("/", "~"))], key


def test_replaying_the_same_cassettes_gives_byte_identical_reports(runs) -> None:
    # The shared runs wrote their items with --items-out; the report must not depend on that either.
    for key in (("bugfix", "candidate", "heldout"), ("judge", "candidate-regressed", "heldout")):
        again = gate(*key)
        assert again.returncode == 0, again.stderr
        assert again.stdout == runs[key].stdout, key


def test_the_summary_says_synthetic_first_and_shows_every_verdict(runs, tmp_path: Path) -> None:
    files = []
    for feature in FEATURES:
        for arm in ARMS:
            path = tmp_path / f"{feature}-{arm}.json"
            path.write_text(runs[feature, arm, "heldout"].stdout, encoding="utf-8")
            files.append(str(path))
    result = llm_eval("summary", *files)
    assert result.returncode == 0, result.stderr
    assert result.stdout.startswith("# LLM regression gate\n\n**Synthetic demo data")
    assert result.stdout.count("| PROMOTE |") == 2 and result.stdout.count("| REJECT |") == 2
    assert "New SAST violations: bugfix:evaluate-sum (eval-exec)." in result.stdout


# ---- the suite, the ledger and the cassettes ----------------------------------------------------------


def test_the_demo_suite_validates() -> None:
    result = llm_eval("tasks", "validate", "--suite", "tasks/demo")
    assert result.returncode == 0, result.stderr
    report = json.loads(result.stdout)
    assert report["valid"] is True and report["executed"] is True and report["problems"] == []


def test_the_ledger_verifies() -> None:
    result = llm_eval("ledger", "verify")
    assert result.returncode == 0, result.stderr
    document = json.loads(result.stdout)
    assert document == {"ledger": "ledger/ledger.jsonl", "rows": 3, "problems": [], "valid": True}


@pytest.mark.parametrize("feature", FEATURES)
def test_the_ledger_names_cassettes_baseline_as_current(feature: str) -> None:
    result = llm_eval("ledger", "current", "--feature", feature)
    assert result.returncode == 0, result.stderr
    assert result.stdout == "cassettes/baseline\n"


def test_the_ledger_rows_are_the_two_initial_promotions() -> None:
    rows = [json.loads(line) for line in (ROOT / "ledger" / "ledger.jsonl").read_text(encoding="utf-8").splitlines()]
    assert [row["kind"] for row in rows] == ["header", "promote", "promote"]
    baseline = cassettes.fingerprint(ROOT / "cassettes" / "baseline")
    for row, feature in zip(rows[1:], FEATURES):
        assert row["feature"] == feature and row["gate"] == "initial" and row["from"] is None and row["report"] is None
        assert row["to"] == {"path": "cassettes/baseline", "fingerprint": baseline}
        assert "synthetic" in row["reason"]


def test_the_committed_cassettes_are_what_the_script_writes() -> None:
    result = make_cassettes("--check")
    assert result.returncode == 0, result.stderr
    assert "identical" in result.stdout


def test_the_check_notices_an_edited_recording(tmp_path: Path) -> None:
    out = tmp_path / "cassettes"
    shutil.copytree(ROOT / "cassettes", out)
    tape = out / "candidate" / "bugfix.jsonl"
    tape.write_bytes(tape.read_bytes().replace(b'"finish_reason": "stop"', b'"finish_reason": "length"', 1))
    (out / "baseline" / "judge.jsonl").unlink()
    result = make_cassettes("--check", "--out", str(out))
    assert result.returncode == 1
    assert "candidate/bugfix.jsonl: differs from what this script writes" in result.stderr
    assert "baseline/judge.jsonl: missing from the committed set" in result.stderr


@pytest.mark.parametrize(
    "arm, model, version",
    [("baseline", "demo-model-a", "v1"), ("candidate", "demo-model-a", "v2"), ("candidate-regressed", "demo-model-small", "v2")],
)
def test_the_cassette_manifests(arm: str, model: str, version: str) -> None:
    suite = tasks.load_suite(ROOT / "tasks" / "demo")
    manifest = json.loads((ROOT / "cassettes" / arm / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["arm"] == arm and manifest["synthetic"] is True
    assert "not produced by a model" in manifest["description"]
    for feature, per_task in (("bugfix", 1), ("judge", 2)):
        tape = manifest["features"][feature]
        assert (tape["model"], tape["prompt"]) == (model, f"{feature}/{version}")
        assert tape["items"] == per_task * len(suite.tasks)
        assert tape["suite_sha256"] == suite.sha256, "the suite changed: run scripts/make_demo_cassettes.py (README)"
