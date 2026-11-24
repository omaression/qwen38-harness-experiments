#!/usr/bin/env python3
"""Write the report's result tables from the committed experiment reports, deterministically.

    python scripts/make_report_tables.py            write the tables (see OUTPUTS)
    python scripts/make_report_tables.py --check    regenerate in memory and compare with the files in
                                                    the tree: exit 0 when identical, 1 on any difference
    python scripts/make_report_tables.py --root DIR read and write under DIR instead of this checkout

Every number in a table comes from a gate report, experiments/<experiment>/expected/<name>.json
(schema llm_eval_harness.report/1), which the CI job `experiments` reproduces byte for byte from
the experiment's recorded cassettes. Nothing is computed here but formatting, so the report and
the paper cannot drift from the data: change a report, run this script, and every table follows.
tests/test_report_tables.py runs --check.

INPUTS
    experiments/*/expected/*.json      one gate report per (feature, split, candidate, baseline)

OUTPUTS
    docs/report/tables.md              every table in markdown, with each report's warnings
    docs/paper/tables/<id>-gate.tex    per experiment: the gate decisions (an IEEEtran table*)
    docs/paper/tables/<id>-arms.tex    per experiment: per-arm secondary metrics (an IEEEtran table*)
    docs/report/*.md                   the text between <!-- BEGIN GENERATED: <name> --> and
                                       <!-- END GENERATED: <name> --> is replaced by the markdown
                                       table <name> (for example exp-a-gate); other text is untouched

<id> is the experiment directory's leading "exp-<letter>" (experiments/exp-a-qwen38 -> exp-a).
docs/paper/tables/ holds generated files only: a .tex file there that this script does not write
is deleted on writing and reported by --check.

EXPERIMENT C. Until experiments/exp-c-*/expected/*.json exists, the exp-c tables are clearly marked
placeholders that list the planned comparisons (Ternary Bonsai 2 against bf16 Qwen3.8-27B on
HumanEvalFix, with a Q8_0 engine-control arm) with every number pending. Once its reports are
committed, the same names are filled from them like any other experiment.

The tables use plain tabular with \\hline rules (no booktabs), so they compile with IEEEtran alone.
Standard library only.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

ROOT = Path(__file__).resolve().parent.parent
REPORT_SCHEMA = "llm_eval_harness.report/1"
SCRIPT = "scripts/make_report_tables.py"
SPLIT_ORDER = {"heldout": 0, "dev": 1}
METRIC_NAMES = {"resolved_rate": "resolved rate", "balanced_accuracy": "balanced accuracy"}
MARKER = re.compile(
    r"(<!-- BEGIN GENERATED: (?P<name>[a-z0-9-]+) -->\n)(?P<body>.*?)(<!-- END GENERATED: (?P=name) -->)",
    re.DOTALL,
)

# The planned arms and comparisons of experiment C (placeholders until its reports exist).
EXP_C_ARMS = (
    ("bf16-vllm", "Qwen3.8-27B, bf16 weights, served by vLLM"),
    ("q8-llamacpp", "Qwen3.8-27B, Q8_0 GGUF, served by llama.cpp (engine control)"),
    ("ternary-q2-llamacpp", "Ternary Bonsai 2 27B, Q2_0 ternary GGUF, served by llama.cpp"),
)
EXP_C_COMPARISONS = (
    ("C1", "bf16-vllm", "q8-llamacpp", "the engine change (and 8-bit rounding): the engine control"),
    ("C2", "q8-llamacpp", "ternary-q2-llamacpp", "the ternary model at a fixed engine"),
    ("C3", "bf16-vllm", "ternary-q2-llamacpp", "the whole deployment change"),
)


class ReportError(ValueError):
    """A file under experiments/*/expected is not a usable gate report."""


# ----------------------------------------------------------------------------- formatting


def num(x: Any, digits: int = 3) -> str:
    """A fixed-point number with an ASCII minus; '-' for a missing value."""
    if x is None:
        return "n/a"
    return f"{float(x):.{digits}f}"


def signed(x: Any, digits: int = 3) -> str:
    if x is None:
        return "n/a"
    return f"{float(x):+.{digits}f}"


def pval(p: Any) -> str:
    """Three decimals, or two significant digits below 0.01 (so 0.0023 is not shown as 0.002)."""
    if p is None:
        return "n/a"
    p = float(p)
    if p >= 0.01:
        return f"{p:.3f}"
    return f"{p:.2g}" if p > 0 else "0"


def interval(ci: Sequence[Any] | None, sign: bool = False) -> str:
    if not ci:
        return "n/a"
    f = signed if sign else num
    return f"[{f(ci[0])}, {f(ci[1])}]"


def experiment_id(dirname: str) -> str:
    m = re.match(r"(exp-[a-z0-9]+)", dirname)
    if not m:
        raise ReportError(f"experiments/{dirname}: the directory name must start with exp-<letter>")
    return m.group(1)


def experiment_title(eid: str) -> str:
    return "Experiment " + eid.split("-", 1)[1].upper()


TEX_SPECIAL = {"\\": r"\textbackslash{}", "&": r"\&", "%": r"\%", "$": r"\$", "#": r"\#", "_": r"\_",
               "{": r"\{", "}": r"\}", "~": r"\textasciitilde{}", "^": r"\textasciicircum{}"}


def tex(s: str) -> str:
    """Escape plain text for LaTeX; a leading ASCII minus of a number becomes a math minus."""
    out = "".join(TEX_SPECIAL.get(c, c) for c in s)
    return re.sub(r"(?<![\w.])-(?=\d)", "$-$", out)


def md(s: str) -> str:
    return s.replace("|", "\\|")


# ----------------------------------------------------------------------------- reading


@dataclass(frozen=True)
class Row:
    """The fields of one gate report that the tables show, all copied from the report."""
    experiment: str
    source: str            # the report's repository-relative path
    feature: str
    split: str
    items: int
    metric: str
    baseline: Mapping[str, Any]
    candidate: Mapping[str, Any]
    comparison: Mapping[str, Any]
    gates: Sequence[Mapping[str, Any]]
    promote: bool
    warnings: Sequence[str]
    rule: str
    suite: Mapping[str, Any]

    @property
    def failed_gates(self) -> list[str]:
        out = []
        for g in self.gates:
            if not g.get("enforced") or g.get("passed"):
                continue
            name = str(g["gate"])
            if name in ("schema_validity", "think_validity") and g.get("value") is not None:
                name += f" ({num(g['value'])})"
            elif name == "new_sast_violations" and g.get("value") is not None:
                name += f" ({g['value']})"
            out.append(name)
        return out

    @property
    def verdict(self) -> str:
        return "PROMOTE" if self.promote else "REJECT"


def load_report(path: Path, root: Path) -> Row:
    rel = path.relative_to(root).as_posix()
    try:
        r = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as e:
        raise ReportError(f"{rel}: not readable as JSON ({e})") from None
    if not isinstance(r, dict) or r.get("schema") != REPORT_SCHEMA:
        raise ReportError(f"{rel}: not a gate report (schema {REPORT_SCHEMA} expected)")
    if not r.get("baseline"):
        raise ReportError(f"{rel}: a report without a baseline arm has no comparison to tabulate")
    for key in ("feature", "split", "candidate", "comparison", "gates", "promote", "suite"):
        if key not in r:
            raise ReportError(f"{rel}: missing key {key!r}")
    return Row(
        experiment=path.parent.parent.name,
        source=rel,
        feature=r["feature"],
        split=r["split"],
        items=int(r["comparison"]["n"]),
        metric=r["comparison"]["metric"],
        baseline=r["baseline"],
        candidate=r["candidate"],
        comparison=r["comparison"],
        gates=r["gates"],
        promote=bool(r["promote"]),
        warnings=list(r.get("warnings") or []),
        rule=str(r.get("rule", "")),
        suite=r["suite"],
    )


def load_experiments(root: Path) -> dict[str, list[Row]]:
    """{experiment id: rows}, ordered by experiment id; rows by feature, held-out before dev, arms."""
    groups: dict[str, list[Row]] = {}
    dirs: dict[str, str] = {}
    for path in sorted((root / "experiments").glob("*/expected/*.json")):
        row = load_report(path, root)
        eid = experiment_id(row.experiment)
        if dirs.setdefault(eid, row.experiment) != row.experiment:
            raise ReportError(f"experiments/{row.experiment} and experiments/{dirs[eid]} share the id {eid}")
        groups.setdefault(eid, []).append(row)
    for rows in groups.values():
        rows.sort(key=lambda w: (w.feature, SPLIT_ORDER.get(w.split, 9), w.split,
                                 str(w.baseline.get("name")), str(w.candidate.get("name"))))
    return dict(sorted(groups.items()))


# ----------------------------------------------------------------------------- table content


def arm_primary(row: Row, arm: Mapping[str, Any]) -> str:
    m = arm["metrics"]
    if row.metric == "resolved_rate" and "resolved" in m:
        return f"{num(m['primary'])} ({m['resolved']}/{m['items']})"
    return num(m.get("primary"))


def arm_label(arm: Mapping[str, Any]) -> str:
    return f"{arm.get('name')} ({arm.get('model')})"


GATE_HEAD = ("Feature", "Split", "Items", "Baseline (model)", "Candidate (model)", "Metric", "Baseline",
             "Candidate", "Δ [95% CI]", "Gains / losses", "McNemar p", "MDE80", "Verdict", "Failed gates")


def gate_cells(row: Row) -> list[str]:
    c = row.comparison
    return [row.feature, row.split, str(row.items), arm_label(row.baseline), arm_label(row.candidate),
            METRIC_NAMES.get(row.metric, row.metric), arm_primary(row, row.baseline), arm_primary(row, row.candidate),
            f"{signed(c.get('delta'))} {interval(c.get('ci95'), sign=True)}", f"{c.get('gains')} / {c.get('losses')}",
            pval(c.get("mcnemar_p")), num(c.get("mde80")), row.verdict, ", ".join(row.failed_gates) or "none"]


ARMS_HEAD = ("Feature", "Split", "Arm", "Model", "Prompt", "Primary [95% CI]", "Schema", "Think",
             "Lenient only", "Truncated", "Mean tokens", "Median latency (s)", "Feature-specific")


def arm_cells(row: Row, arm: Mapping[str, Any]) -> list[str]:
    m = arm["metrics"]
    ci = m.get(f"{row.metric}_ci95")
    if row.feature == "judge" or "tpr" in m:
        specific = f"TPR {num(m.get('tpr'))}, TNR {num(m.get('tnr'))}, approve {num(m.get('approve_rate'))}"
    else:
        blocks = sum(len(v) if isinstance(v, (list, tuple)) else int(v) for v in ((m.get("sast") or {}).get("block") or {}).values())
        specific = (f"overfit {num(m.get('overfit_rate'))}, hidden pass {num(m.get('hidden_pass_rate'))}, "
                    f"block SAST {blocks}")
    finish = m.get("finish_reasons") or {}
    return [row.feature, row.split, str(arm.get("name")), str(arm.get("model")), str(arm.get("prompt")),
            f"{num(m.get('primary'))} {interval(ci)}", num(m.get("schema_validity")), num(m.get("think_validity")),
            num(m.get("lenient_only_rate")), f"{finish.get('length', 0)}/{m.get('items')}",
            f"{float(m.get('completion_tokens_mean') or 0):.0f}", num(m.get("latency_median_s"), 2), specific]


# ----------------------------------------------------------------------------- markdown


def md_table(head: Sequence[str], rows: Iterable[Sequence[str]]) -> str:
    lines = ["| " + " | ".join(md(h) for h in head) + " |", "|" + "---|" * len(head)]
    lines += ["| " + " | ".join(md(c) for c in r) + " |" for r in rows]
    return "\n".join(lines) + "\n"


def md_gate(eid: str, rows: Sequence[Row]) -> str:
    sources = ", ".join(f"`{r.source}`" for r in rows)
    return (f"*Table: {experiment_title(eid)} (`experiments/{rows[0].experiment}`), gate decisions under rule "
            f"`{rows[0].rule}` on suite `{rows[0].suite.get('name')}`. Δ is candidate minus baseline on the paired "
            f"items, with its paired cluster-bootstrap 95% interval; MDE80 is the minimum detectable effect at 80% "
            f"power. Generated by `{SCRIPT}` from {sources}.*\n\n" + md_table(GATE_HEAD, (gate_cells(r) for r in rows)))


def md_arms(eid: str, rows: Sequence[Row]) -> str:
    body = [arm_cells(r, arm) for r in rows for arm in (r.baseline, r.candidate)]
    return (f"*Table: {experiment_title(eid)}, per-arm metrics. Primary intervals are Wilson (resolved rate) or a "
            f"task-cluster bootstrap (balanced accuracy); \"Truncated\" counts replies that ended at `max_tokens`. "
            f"Generated by `{SCRIPT}`.*\n\n" + md_table(ARMS_HEAD, body))


PLACEHOLDER_NOTE = ("PLACEHOLDER: Experiment C has not been recorded. No `experiments/exp-c-*/expected/*.json` "
                    f"exists, so every number below is pending. `{SCRIPT}` replaces this table with the "
                    "measured one when the experiment's reports are committed.")


def md_exp_c_placeholder() -> tuple[str, str]:
    arms = dict(EXP_C_ARMS)
    gate = (f"> **{PLACEHOLDER_NOTE}**\n\n*Table: Experiment C (planned), gate decisions on the HumanEvalFix suite.*\n\n"
            + md_table(("Comparison", "Baseline arm", "Candidate arm", "Isolates", "Baseline", "Candidate", "Δ [95% CI]",
                        "Gains / losses", "McNemar p", "MDE80", "Verdict", "Failed gates"),
                       ([cid, f"{b}: {arms[b]}", f"{c}: {arms[c]}", what] + ["pending"] * 8
                        for cid, b, c, what in EXP_C_COMPARISONS)))
    arm_table = (f"> **{PLACEHOLDER_NOTE}**\n\n*Table: Experiment C (planned), per-arm metrics.*\n\n"
                 + md_table(("Arm", "Description", "Primary [95% CI]", "Schema", "Think", "Truncated", "Mean tokens"),
                            ([a, d] + ["pending"] * 5 for a, d in EXP_C_ARMS)))
    return gate, arm_table


def md_warnings(rows: Sequence[Row]) -> str:
    lines = []
    for r in rows:
        for w in r.warnings:
            lines.append(f"- `{r.feature}`, {r.split}: {w}")
    return "\n".join(lines) + "\n" if lines else "- none\n"


# ----------------------------------------------------------------------------- LaTeX


def tex_table(label: str, caption: str, cols: str, head: Sequence[str], rows: Iterable[Sequence[str]],
              size: str = r"\scriptsize", head_is_tex: bool = False) -> str:
    out = [f"% Generated by {SCRIPT}; do not edit. Regenerate with: python {SCRIPT}",
           r"\begin{table*}[!t]", f"\\caption{{{caption}}}", f"\\label{{{label}}}", r"\centering", size,
           r"\setlength{\tabcolsep}{3pt}", f"\\begin{{tabular}}{{{cols}}}", r"\hline",
           " & ".join(head if head_is_tex else (tex(h) for h in head)) + r" \\", r"\hline"]
    out += [" & ".join(tex(c) for c in r) + r" \\" for r in rows]
    out += [r"\hline", r"\end{tabular}", r"\end{table*}"]
    return "\n".join(out) + "\n"


TEX_GATE_HEAD = ("Feature", "Split", "$n$", "Base", "Cand.", r"$\Delta$ [95\% CI]", "G/L", "McNemar $p$",
                 r"MDE$_{80}$", "Verdict", "Failed gates")


def _models(rows: Sequence[Row], arm: str) -> str:
    return ", ".join(sorted({str(getattr(r, arm).get("model")) for r in rows}))


def tex_gate(eid: str, rows: Sequence[Row]) -> str:
    """The arms go into the caption when every row compares the same two arms (so the table fits the page
    width), and into a Comparison column otherwise."""
    pairs = {(str(r.baseline.get("name")), str(r.candidate.get("name"))) for r in rows}
    same = len(pairs) == 1
    body = []
    for r in rows:
        c = gate_cells(r)
        row = [c[0], c[1], c[2], c[6], c[7]] + c[8:]
        if not same:
            row.insert(3, f"{r.candidate.get('name')} vs {r.baseline.get('name')}")
        body.append(row)
    if same:
        (base, cand), = pairs
        arms = (f"candidate \\texttt{{{tex(cand)}}} (model {tex(_models(rows, 'candidate'))}) against baseline "
                f"\\texttt{{{tex(base)}}} (model {tex(_models(rows, 'baseline'))}), ")
    else:
        arms = ""
    caption = (f"{tex(experiment_title(eid))}: gate decisions, {arms}rule \\texttt{{{tex(rows[0].rule)}}}, suite "
               f"\\texttt{{{tex(str(rows[0].suite.get('name')))}}}. Primary metric: resolved rate (bugfix) or "
               r"balanced accuracy (judge). $\Delta$ is candidate minus baseline on paired items with a paired "
               r"cluster-bootstrap 95\% interval; G/L are discordant gains and losses; MDE$_{80}$ is the minimum "
               r"detectable effect at 80\% power. Generated from the committed reports in \texttt{"
               + tex(f"experiments/{rows[0].experiment}/expected") + "}.")
    head = list(TEX_GATE_HEAD)
    cols = r"llrllllrrlp{0.22\textwidth}"
    if not same:
        head.insert(3, "Comparison")
        cols = r"llrp{0.16\textwidth}llllrrlp{0.16\textwidth}"
    return tex_table(f"tab:{eid}-gate", caption, cols, head, body, head_is_tex=True)


TEX_ARMS_HEAD = ("Feature", "Split", "Arm", "Model", "Primary [95\\% CI]", "Schema", "Think", "Len.\\ only",
                 "Trunc.", "Tokens", "Feature-specific")


def tex_arms(eid: str, rows: Sequence[Row]) -> str:
    body = []
    for r in rows:
        for arm in (r.baseline, r.candidate):
            c = arm_cells(r, arm)
            body.append([c[0], c[1], c[2], c[3], c[5], c[6], c[7], c[8], c[9], c[10], c[12]])
    caption = (f"{tex(experiment_title(eid))}: per-arm metrics. Intervals are Wilson (resolved rate) or a "
               "task-cluster bootstrap (balanced accuracy). Trunc.\\ counts replies cut at the token limit; "
               "Tokens is the mean completion length.")
    return tex_table(f"tab:{eid}-arms", caption, "lllllrrrrrl", TEX_ARMS_HEAD, body, head_is_tex=True)


TEX_PLACEHOLDER = (r"\textbf{PLACEHOLDER: Experiment C has not been recorded; all numbers are pending. "
                   r"The table is regenerated from \texttt{experiments/exp-c-*/expected} when it exists.}")


def tex_exp_c_placeholder() -> tuple[str, str]:
    arms = dict(EXP_C_ARMS)
    gate = tex_table("tab:exp-c-gate", "Experiment C (planned): Ternary Bonsai 2 against bf16 Qwen3.8-27B on "
                     "HumanEvalFix, gate decisions. " + TEX_PLACEHOLDER, r"lllp{0.25\textwidth}llll",
                     ("Comp.", "Baseline arm", "Candidate arm", "Isolates", "Base", "Cand.", "Delta [95% CI]", "Verdict"),
                     ([cid, b, c, what, "pending", "pending", "pending", "pending"] for cid, b, c, what in EXP_C_COMPARISONS))
    arm_table = tex_table("tab:exp-c-arms", "Experiment C (planned): arms. " + TEX_PLACEHOLDER, "llllll",
                          ("Arm", "Description", "Primary [95% CI]", "Schema", "Think", "Tokens"),
                          ([a, arms[a]] + ["pending"] * 4 for a, _ in EXP_C_ARMS))
    return gate, arm_table


# ----------------------------------------------------------------------------- assembling


def build(root: Path) -> tuple[dict[str, str], dict[str, str]]:
    """(files: {repository-relative path: content}, blocks: {marker name: markdown})."""
    groups = load_experiments(root)
    files: dict[str, str] = {}
    blocks: dict[str, str] = {}
    doc = ["# Result tables", "",
           f"Generated by `{SCRIPT}` from `experiments/*/expected/*.json`; do not edit by hand. Every number is "
           "copied from a gate report that the CI job `experiments` reproduces byte for byte from the recorded "
           "cassettes. Rerun the script after any report changes; `tests/test_report_tables.py` fails until then.", ""]
    for eid, rows in groups.items():
        blocks[f"{eid}-gate"] = md_gate(eid, rows)
        blocks[f"{eid}-arms"] = md_arms(eid, rows)
        files[f"docs/paper/tables/{eid}-gate.tex"] = tex_gate(eid, rows)
        files[f"docs/paper/tables/{eid}-arms.tex"] = tex_arms(eid, rows)
        doc += [f"## {experiment_title(eid)}: `experiments/{rows[0].experiment}`", "", blocks[f"{eid}-gate"],
                blocks[f"{eid}-arms"], "Warnings the reports carry:", "", md_warnings(rows)]
    if "exp-c" not in groups:
        blocks["exp-c-gate"], blocks["exp-c-arms"] = md_exp_c_placeholder()
        files["docs/paper/tables/exp-c-gate.tex"], files["docs/paper/tables/exp-c-arms.tex"] = tex_exp_c_placeholder()
        doc += ["## Experiment C (planned): Ternary Bonsai 2 against bf16 Qwen3.8-27B on HumanEvalFix", "",
                blocks["exp-c-gate"], blocks["exp-c-arms"]]
    files["docs/report/tables.md"] = "\n".join(doc).rstrip("\n") + "\n"
    return files, blocks


def fill_blocks(text: str, blocks: Mapping[str, str], where: str) -> str:
    def sub(m: re.Match[str]) -> str:
        name = m.group("name")
        if name not in blocks:
            raise ReportError(f"{where}: no generated table named {name!r} (known: {', '.join(sorted(blocks))})")
        return m.group(1) + blocks[name] + m.group(4)
    return MARKER.sub(sub, text)


def plan(root: Path) -> tuple[dict[str, str], list[str]]:
    """Every file the script owns or edits, with its intended content, and the stale tables to delete."""
    files, blocks = build(root)
    for path in sorted((root / "docs" / "report").glob("*.md")):
        rel = path.relative_to(root).as_posix()
        if rel in files:
            continue
        text = path.read_text(encoding="utf-8")
        if "<!-- BEGIN GENERATED:" in text:
            files[rel] = fill_blocks(text, blocks, rel)
    stale = sorted(p.relative_to(root).as_posix() for p in (root / "docs" / "paper" / "tables").glob("*.tex")
                   if p.relative_to(root).as_posix() not in files)
    return files, stale


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--check", action="store_true", help="compare with the tree instead of writing")
    ap.add_argument("--root", type=Path, default=ROOT, help="the repository root (default: this checkout)")
    args = ap.parse_args(argv)
    root = args.root.resolve()
    try:
        files, stale = plan(root)
    except ReportError as e:
        print(f"make_report_tables: {e}", file=sys.stderr)
        return 2
    if args.check:
        bad = [rel for rel, content in files.items()
               if not (root / rel).is_file() or (root / rel).read_text(encoding="utf-8") != content]
        for rel in bad:
            print(f"make_report_tables: {rel} is out of date", file=sys.stderr)
        for rel in stale:
            print(f"make_report_tables: {rel} is not generated by this script", file=sys.stderr)
        if bad or stale:
            print(f"make_report_tables: run `python {SCRIPT}` and commit the result", file=sys.stderr)
            return 1
        print(f"make_report_tables: {len(files)} files up to date")
        return 0
    for rel in stale:
        (root / rel).unlink()
    for rel, content in files.items():
        path = root / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        if not path.is_file() or path.read_text(encoding="utf-8") != content:
            path.write_text(content, encoding="utf-8", newline="\n")
            print(f"wrote {rel}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
