"""scripts/make_report_tables.py: the report's tables come from the committed gate reports, and only from them.

    the committed tables are up to date        --check passes on the repository itself
    the numbers are the reports' numbers       every Experiment A report's delta, interval, gains and
                                               losses, McNemar p, MDE80 and verdict appear in the output
    experiment C                               a marked placeholder while experiments/exp-c-* has no
                                               reports; the measured table once it has
    drift and stale files                      --check fails on an edited table or a stray .tex file
    the output is deterministic                two runs write the same bytes

Each test but the first works on a copy of the inputs under tmp_path (--root), never on the tree.
"""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
SCRIPT = ROOT / "scripts" / "make_report_tables.py"
EXP_A = ROOT / "experiments" / "exp-a-qwen38" / "expected"


def run(*args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run([sys.executable, str(SCRIPT), *args], cwd=ROOT, capture_output=True, text=True, timeout=120)


def copy_inputs(dst: Path) -> Path:
    """The script's inputs: every experiment's expected reports and the report's markdown sections."""
    for report in sorted((ROOT / "experiments").glob("*/expected/*.json")):
        rel = report.relative_to(ROOT)
        (dst / rel).parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(report, dst / rel)
    (dst / "docs" / "report").mkdir(parents=True)
    for md in (ROOT / "docs" / "report").glob("*.md"):
        shutil.copyfile(md, dst / "docs" / "report" / md.name)
    (dst / "docs" / "paper" / "tables").mkdir(parents=True)
    return dst


def signed(x: float) -> str:
    return f"{x:+.3f}"


def test_committed_tables_are_up_to_date() -> None:
    res = run("--check")
    assert res.returncode == 0, res.stderr


def test_tables_hold_the_numbers_of_every_experiment_a_report(tmp_path: Path) -> None:
    root = copy_inputs(tmp_path)
    res = run("--root", str(root))
    assert res.returncode == 0, res.stderr
    tables = (root / "docs" / "report" / "tables.md").read_text(encoding="utf-8")
    gate_tex = (root / "docs" / "paper" / "tables" / "exp-a-gate.tex").read_text(encoding="utf-8")
    reports = sorted(EXP_A.glob("*.json"))
    assert len(reports) == 4
    for path in reports:
        r: dict[str, Any] = json.loads(path.read_text(encoding="utf-8"))
        c = r["comparison"]
        row = next(line for line in tables.splitlines()
                   if line.startswith(f"| {r['feature']} | {r['split']} | {c['n']} | "))
        assert f"{signed(c['delta'])} [{signed(c['ci95'][0])}, {signed(c['ci95'][1])}]" in row
        assert f"| {c['gains']} / {c['losses']} |" in row
        assert f"| {c['mde80']:.3f} |" in row
        assert f"| {c['mcnemar_p']:.3f} |" in row
        assert ("PROMOTE" if r["promote"] else "REJECT") in row
        assert f"{r['candidate']['metrics']['primary']:.3f}" in row and f"{r['baseline']['metrics']['primary']:.3f}" in row
        for g in r["gates"]:
            if g["enforced"] and not g["passed"]:
                assert g["gate"] in row
        assert path.relative_to(ROOT).as_posix() in tables
        # the LaTeX table carries the same numbers, with math minus signs and escaped gate names
        assert f"{abs(c['delta']):.3f}" in gate_tex
    assert "schema\\_validity (0.846)" in gate_tex and "schema_validity (0.846)" in tables
    # the results section's marked blocks were filled with the same tables
    results = (root / "docs" / "report" / "06-results.md").read_text(encoding="utf-8")
    assert "+0.154 [+0.038, +0.269]" in results and "PLACEHOLDER" in results


def test_experiment_c_placeholder_until_its_reports_exist(tmp_path: Path) -> None:
    root = copy_inputs(tmp_path)
    assert run("--root", str(root)).returncode == 0
    for name in ("exp-c-gate.tex", "exp-c-arms.tex"):
        assert "PLACEHOLDER" in (root / "docs" / "paper" / "tables" / name).read_text(encoding="utf-8")
    assert "PLACEHOLDER: Experiment C has not been recorded" in (root / "docs" / "report" / "tables.md").read_text(encoding="utf-8")

    # A committed exp-c report replaces the placeholder with measured numbers under the same names.
    report = json.loads((EXP_A / "bugfix-heldout.json").read_text(encoding="utf-8"))
    report["candidate"]["name"], report["baseline"]["name"] = "ternary-q2-llamacpp", "bf16-vllm"
    report["comparison"]["delta"] = -0.123456
    exp_c = root / "experiments" / "exp-c-bonsai" / "expected"
    exp_c.mkdir(parents=True)
    (exp_c / "bugfix-heldout.json").write_text(json.dumps(report), encoding="utf-8")
    res = run("--root", str(root))
    assert res.returncode == 0, res.stderr
    gate = (root / "docs" / "paper" / "tables" / "exp-c-gate.tex").read_text(encoding="utf-8")
    assert "PLACEHOLDER" not in gate and "0.123" in gate and "tab:exp-c-gate" in gate
    results = (root / "docs" / "report" / "06-results.md").read_text(encoding="utf-8")
    assert "-0.123" in results and "PLACEHOLDER" not in results


def test_check_reports_drift_and_stray_files(tmp_path: Path) -> None:
    root = copy_inputs(tmp_path)
    assert run("--root", str(root)).returncode == 0
    assert run("--root", str(root), "--check").returncode == 0
    gate = root / "docs" / "paper" / "tables" / "exp-a-gate.tex"
    gate.write_text(gate.read_text(encoding="utf-8").replace("REJECT", "PROMOTE"), encoding="utf-8")
    res = run("--root", str(root), "--check")
    assert res.returncode == 1 and "exp-a-gate.tex is out of date" in res.stderr
    assert run("--root", str(root)).returncode == 0
    stray = root / "docs" / "paper" / "tables" / "hand-made.tex"
    stray.write_text("% not generated\n", encoding="utf-8")
    res = run("--root", str(root), "--check")
    assert res.returncode == 1 and "hand-made.tex is not generated" in res.stderr
    assert run("--root", str(root)).returncode == 0 and not stray.exists()


def test_unknown_marker_and_bad_report_are_refused(tmp_path: Path) -> None:
    root = copy_inputs(tmp_path)
    (root / "docs" / "report" / "99-extra.md").write_text(
        "<!-- BEGIN GENERATED: exp-z-gate -->\n<!-- END GENERATED: exp-z-gate -->\n", encoding="utf-8")
    res = run("--root", str(root))
    assert res.returncode == 2 and "exp-z-gate" in res.stderr
    (root / "docs" / "report" / "99-extra.md").unlink()
    (root / "experiments" / "exp-a-qwen38" / "expected" / "broken.json").write_text("{}", encoding="utf-8")
    res = run("--root", str(root))
    assert res.returncode == 2 and "not a gate report" in res.stderr


def test_output_is_deterministic(tmp_path: Path) -> None:
    first, second = copy_inputs(tmp_path / "a"), copy_inputs(tmp_path / "b")
    assert run("--root", str(first)).returncode == 0 and run("--root", str(second)).returncode == 0
    for rel in ("docs/report/tables.md", "docs/report/06-results.md", "docs/paper/tables/exp-a-gate.tex",
                "docs/paper/tables/exp-a-arms.tex", "docs/paper/tables/exp-c-gate.tex"):
        assert (first / rel).read_bytes() == (second / rel).read_bytes(), rel
