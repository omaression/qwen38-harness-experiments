"""scripts/make_report_tables.py: the report's tables come from the committed gate reports, and only from them.

    the committed tables are up to date        --check passes on the repository itself
    the numbers are the reports' numbers       every Experiment A report's delta, interval, gains and
                                               losses, McNemar p, MDE80 and verdict appear in the output
                                               (one arm pair: one row per feature and split)
    experiment C                               every report of experiments/exp-c-bonsai and -demo appears
                                               as a comparison row (C0, the noise floor, first) with its
                                               token cost, and each arm once in the arms table; an arm
                                               with two different metrics, or two directories sharing an
                                               experiment id, is refused
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
    """The script's inputs: every experiment's expected reports, optional metadata, and the report's markdown sections."""
    for report in sorted((ROOT / "experiments").glob("*/expected/*.json")):
        rel = report.relative_to(ROOT)
        (dst / rel).parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(report, dst / rel)
    for meta in sorted((ROOT / "experiments").glob("*/EXPERIMENT.json")):
        rel = meta.relative_to(ROOT)
        (dst / rel).parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(meta, dst / rel)
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
    assert "+0.154 [+0.038, +0.269]" in results


EXPERIMENTS_C = {"exp-c": ROOT / "experiments" / "exp-c-bonsai" / "expected",
                 "exp-c-demo": ROOT / "experiments" / "exp-c-bonsai-demo" / "expected"}
COMPARISON_IDS = {("bf16-vllm-rerun", "bf16-vllm"): "C0", ("q8-llamacpp", "bf16-vllm"): "C1",
                  ("ternary-llamacpp", "q8-llamacpp"): "C2", ("ternary-llamacpp", "bf16-vllm"): "C3"}


def section(tables: str, directory: str) -> str:
    """The part of tables.md under the heading of experiments/<directory>."""
    start = tables.index(f": `experiments/{directory}`\n")
    end = tables.find("\n## ", start)
    return tables[start:] if end < 0 else tables[start:end]


def test_tables_hold_the_numbers_of_every_experiment_c_report(tmp_path: Path) -> None:
    root = copy_inputs(tmp_path)
    res = run("--root", str(root))
    assert res.returncode == 0, res.stderr
    tables = (root / "docs" / "report" / "tables.md").read_text(encoding="utf-8")
    counts = {"exp-c": 16, "exp-c-demo": 12}
    for eid, expected in EXPERIMENTS_C.items():
        reports = sorted(expected.glob("*.json"))
        assert len(reports) == counts[eid]
        part = section(tables, expected.parent.name)
        gate_tex = (root / "docs" / "paper" / "tables" / f"{eid}-gate.tex").read_text(encoding="utf-8")
        arms_tex = (root / "docs" / "paper" / "tables" / f"{eid}-arms.tex").read_text(encoding="utf-8")
        assert f"tab:{eid}-gate" in gate_tex and f"tab:{eid}-arms" in arms_tex and "PLACEHOLDER" not in gate_tex
        for path in reports:
            r: dict[str, Any] = json.loads(path.read_text(encoding="utf-8"))
            c, cand, base = r["comparison"], r["candidate"], r["baseline"]
            cid = COMPARISON_IDS[(cand["name"], base["name"])]
            assert path.name.startswith(f"{cand['name']}--vs--{base['name']}.{r['feature']}-{r['split']}")
            row = next(line for line in part.splitlines()
                       if line.startswith(f"| {cid} | {r['feature']} | {r['split']} | {c['n']} | {base['name']} | {cand['name']} | "))
            assert f"{signed(c['delta'])} [{signed(c['ci95'][0])}, {signed(c['ci95'][1])}]" in row
            assert f"| {c['gains']} / {c['losses']} |" in row and f"| {c['mcnemar_p']:.3f} |" in row
            assert (f"| {c['mde80']:.3f} |" if c["mde80"] is not None else "| n/a |") in row
            assert ("| PROMOTE |" if r["promote"] else "| REJECT |") in row
            ct, bt = cand["metrics"]["completion_tokens_mean"], base["metrics"]["completion_tokens_mean"]
            assert f"| {ct:.0f} / {bt:.0f} ({ct / bt:.2f}x) |" in row
            for g in r["gates"]:
                if g["enforced"] and not g["passed"]:
                    assert g["gate"] in row
            # every arm of the report has its row in the arms table, with the report's numbers
            for arm in (cand, base):
                m = arm["metrics"]
                arm_row = next(line for line in part.splitlines()
                               if line.startswith(f"| {r['feature']} | {r['split']} | {arm['name']} | {m['primary']:.3f} "))
                assert f"| {m['finish_reasons'].get('length', 0)}/{m['items']} |" in arm_row
                assert f"| {m['completion_tokens_mean']:.0f} | {m['completion_tokens_total']} |" in arm_row
            assert f"{abs(c['delta']):.3f}" in gate_tex
    # the noise floor is the first comparison of experiment C, and only experiment C has one
    exp_c = section(tables, "exp-c-bonsai")
    assert exp_c.index("| C0 | bugfix | heldout | 74 |") < exp_c.index("| C1 | bugfix | heldout | 74 |")
    assert "| C0 |" not in section(tables, "exp-c-bonsai-demo")
    assert "C0: bf16-vllm-rerun against bf16-vllm, noise floor" in exp_c
    # the results section's marked blocks were filled with the measured tables
    results = (root / "docs" / "report" / "06-results.md").read_text(encoding="utf-8")
    assert "PLACEHOLDER" not in results and "| C3 | bugfix | heldout | 74 | bf16-vllm | ternary-llamacpp |" in results


def test_an_arm_reported_with_two_different_metrics_is_refused(tmp_path: Path) -> None:
    root = copy_inputs(tmp_path)
    path = root / "experiments" / "exp-c-bonsai" / "expected" / "ternary-llamacpp--vs--q8-llamacpp.bugfix-heldout.json"
    report = json.loads(path.read_text(encoding="utf-8"))
    report["candidate"]["metrics"]["completion_tokens_mean"] = 1.0
    path.write_text(json.dumps(report), encoding="utf-8")
    res = run("--root", str(root))
    assert res.returncode == 2 and "different metrics for arm ternary-llamacpp" in res.stderr


def test_experiment_ids_and_a_shared_id_is_refused(tmp_path: Path) -> None:
    root = copy_inputs(tmp_path)
    assert run("--root", str(root)).returncode == 0
    names = sorted(p.name for p in (root / "docs" / "paper" / "tables").glob("*.tex"))
    expected_eids = sorted({
        f"{p.parent.parent.name.split('-', 2)[0]}-{p.parent.parent.name.split('-', 2)[1]}"
        + ("-demo" if p.parent.parent.name.endswith("-demo") else "")
        for p in (ROOT / "experiments").glob("*/expected/*.json")
    })
    assert {"exp-a", "exp-c", "exp-c-demo"}.issubset(expected_eids)
    assert names == sorted(f"{eid}-{kind}.tex" for eid in expected_eids for kind in ("arms", "gate"))
    other = root / "experiments" / "exp-c-other" / "expected"
    other.mkdir(parents=True)
    shutil.copyfile(EXP_A / "bugfix-heldout.json", other / "bugfix-heldout.json")
    res = run("--root", str(root))
    assert res.returncode == 2 and "share the id exp-c" in res.stderr


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
    tex_files = sorted(f"docs/paper/tables/{p.name}" for p in (first / "docs" / "paper" / "tables").glob("*.tex"))
    assert tex_files == sorted(f"docs/paper/tables/{p.name}" for p in (second / "docs" / "paper" / "tables").glob("*.tex"))
    for rel in ("docs/report/tables.md", "docs/report/06-results.md", *tex_files):
        assert (first / rel).read_bytes() == (second / rel).read_bytes(), rel


def test_review_patch_and_experiment_json_support(tmp_path: Path) -> None:
    import importlib.util

    spec = importlib.util.spec_from_file_location("make_report_tables", SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    sys.modules.setdefault("make_report_tables", mod)
    spec.loader.exec_module(mod)

    assert mod.experiment_title("exp-b") == "Experiment B"
    assert mod.experiment_title("exp-d") == "Experiment D"
    assert mod.METRIC_NAMES["usable_rate"] == "usable patch rate"

    root = copy_inputs(tmp_path)
    assert run("--root", str(root)).returncode == 0
    tables = (root / "docs" / "report" / "tables.md").read_text(encoding="utf-8")
    if (ROOT / "experiments" / "exp-b-review-lora" / "expected").is_dir():
        exp_b = section(tables, "exp-b-review-lora")
        assert "| review | heldout | 183 |" in exp_b
        assert "TPR " in exp_b and "TNR " in exp_b and "approve " in exp_b
    if (ROOT / "experiments" / "exp-d-coder-effort" / "expected").is_dir():
        exp_d = section(tables, "exp-d-coder-effort")
        assert "| D1 | patch | heldout |" in exp_d
        assert "applied " in exp_d and "gold " in exp_d and "block SAST " in exp_d

    # EXPERIMENT.json absent -> defaults to Experiment B / D and built-in descriptions/comparisons.
    for meta in (root / "experiments").glob("*/EXPERIMENT.json"):
        meta.unlink()
    assert run("--root", str(root)).returncode == 0
    tables_no_meta = (root / "docs" / "report" / "tables.md").read_text(encoding="utf-8")
    if (ROOT / "experiments" / "exp-b-review-lora" / "expected").is_dir():
        assert "## Experiment B: `experiments/exp-b-review-lora`" in tables_no_meta
    if (ROOT / "experiments" / "exp-d-coder-effort" / "expected").is_dir():
        assert "## Experiment D: `experiments/exp-d-coder-effort`" in tables_no_meta
        assert "D1: effort-medium against effort-xhigh, medium vs xhigh reasoning effort" in tables_no_meta
