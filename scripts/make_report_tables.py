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

<id> is the experiment directory's leading "exp-<letter>", with "-demo" appended when the directory
name ends in "-demo" (experiments/exp-a-qwen38 -> exp-a, experiments/exp-c-bonsai -> exp-c,
experiments/exp-c-bonsai-demo -> exp-c-demo). docs/paper/tables/ holds generated files only: a .tex
file there that this script does not write is deleted on writing and reported by --check.

TWO LAYOUTS. An experiment whose reports all compare the same two arms (experiment A) gets one gate
row per feature and split. An experiment with several comparisons (experiment C) gets one gate row
per comparison, feature and split, ordered by the comparison ids of COMPARISONS (C0, the no-op
rerun that measures the noise floor, first), with each comparison's mean completion tokens; and an
arms table with one row per arm, feature and split (an arm's metrics are the same in every report
that contains it, and the script refuses reports where they are not). The only arithmetic is the
ratio of the two reported mean completion lengths in the token column.

The tables use plain tabular with \\hline rules (no booktabs), so they compile with IEEEtran alone.
Standard library only.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

ROOT = Path(__file__).resolve().parent.parent
REPORT_SCHEMA = "llm_eval_harness.report/1"
SCRIPT = "scripts/make_report_tables.py"
SPLIT_ORDER = {"heldout": 0, "dev": 1}
METRIC_NAMES = {
    "resolved_rate": "resolved rate",
    "balanced_accuracy": "balanced accuracy",
    "usable_rate": "usable patch rate",
}
MARKER = re.compile(
    r"(<!-- BEGIN GENERATED: (?P<name>[a-z0-9-]+) -->\n)(?P<body>.*?)(<!-- END GENERATED: (?P=name) -->)",
    re.DOTALL,
)

# What the arms of the experiments with several comparisons are (the reports carry only the arm's
# name and its served model name, which is `qwen38` for every arm of experiment C).
ARM_DESCRIPTIONS = {
    "bf16-vllm": "Qwen3.8-27B, bf16 weights, served by vLLM",
    "bf16-vllm-rerun": "bf16-vllm recorded a second time (the no-op arm)",
    "q8-llamacpp": "Qwen3.8-27B, Q8_0 GGUF, served by llama.cpp b12682 (engine control)",
    "ternary-llamacpp": "Ternary Bonsai 2 27B, Q2_0 GGUF, served by llama.cpp b12682",
    "qwen38-base": "Abliterated Qwen3.8-27B base, uncapped xhigh",
    "qwen38-review-lora": "Qwen3.8-27B + public-v2-review LoRA (r=32, w700 SFT), uncapped xhigh",
    "effort-xhigh": "Abliterated Qwen3.8-27B coder, reasoning_effort=xhigh (uncapped)",
    "effort-medium": "Abliterated Qwen3.8-27B coder, reasoning_effort=medium (uncapped)",
    "effort-low": "Abliterated Qwen3.8-27B coder, reasoning_effort=low (uncapped)",
}
ARM_ORDER = tuple(ARM_DESCRIPTIONS)
# (candidate, baseline) -> (comparison id, what the comparison isolates).
COMPARISONS = {
    ("bf16-vllm-rerun", "bf16-vllm"): ("C0", "noise floor: the same model, engine and settings recorded twice"),
    ("q8-llamacpp", "bf16-vllm"): ("C1", "engine change and 8-bit rounding (the engine control)"),
    ("ternary-llamacpp", "q8-llamacpp"): ("C2", "the ternary model at a fixed engine"),
    ("ternary-llamacpp", "bf16-vllm"): ("C3", "the whole deployment change"),
    ("effort-medium", "effort-xhigh"): ("D1", "medium vs xhigh reasoning effort"),
    ("effort-low", "effort-xhigh"): ("D2", "low vs xhigh reasoning effort"),
    ("effort-low", "effort-medium"): ("D3", "low vs medium reasoning effort"),
}


@dataclass(frozen=True)
class ExperimentMeta:
    """Optional per-experiment metadata loaded from experiments/<dir>/EXPERIMENT.json."""
    title: str | None = None
    arm_descriptions: Mapping[str, str] = field(default_factory=dict)
    comparisons: Mapping[tuple[str, str], tuple[str, str]] = field(default_factory=dict)


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
    return m.group(1) + ("-demo" if dirname.endswith("-demo") else "")


def experiment_title(eid: str, meta: ExperimentMeta | Mapping[str, Any] | None = None) -> str:
    if isinstance(meta, ExperimentMeta) and meta.title:
        return meta.title
    if isinstance(meta, Mapping) and isinstance(meta.get("title"), str) and meta["title"].strip():
        return meta["title"].strip()
    parts = eid.split("-")
    return "Experiment " + parts[1].upper() + (" (demo suite)" if parts[2:] == ["demo"] else "")


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
    meta: ExperimentMeta = field(default_factory=ExperimentMeta, repr=False, compare=False)

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

    @property
    def pair(self) -> tuple[str, str]:
        """(candidate arm, baseline arm)."""
        return str(self.candidate.get("name")), str(self.baseline.get("name"))

    @property
    def comparison_id(self) -> str:
        """The comparison's id from COMPARISONS, or "<candidate> vs <baseline>" for a pair it does not list."""
        known = (self.meta.comparisons or COMPARISONS).get(self.pair)
        return known[0] if known else f"{self.pair[0]} vs {self.pair[1]}"


def _parse_meta_comparisons(raw: Any, rel: str) -> dict[tuple[str, str], tuple[str, str]]:
    out: dict[tuple[str, str], tuple[str, str]] = {}
    if raw is None:
        return out
    if isinstance(raw, Mapping):
        for key, val in raw.items():
            if isinstance(val, Mapping) and "candidate" in val and "baseline" in val:
                pair = (str(val["candidate"]), str(val["baseline"]))
                cid = str(val.get("id") or key)
                desc = str(val.get("description") or val.get("what") or "")
                out[pair] = (cid, desc)
            elif "--vs--" in str(key) or " vs " in str(key):
                cand, base = str(key).split("--vs--", 1) if "--vs--" in str(key) else str(key).split(" vs ", 1)
                if isinstance(val, (list, tuple)) and val:
                    out[(cand.strip(), base.strip())] = (str(val[0]), str(val[1]) if len(val) > 1 else "")
                elif isinstance(val, Mapping):
                    out[(cand.strip(), base.strip())] = (str(val.get("id") or key), str(val.get("description") or val.get("what") or ""))
                else:
                    raise ReportError(f"{rel}: invalid comparison entry for {key!r}")
            else:
                raise ReportError(f"{rel}: invalid comparison entry for {key!r}")
        return out
    if isinstance(raw, Sequence) and not isinstance(raw, (str, bytes)):
        for idx, entry in enumerate(raw):
            if not isinstance(entry, Mapping) or "candidate" not in entry or "baseline" not in entry:
                raise ReportError(f"{rel}: comparisons[{idx}] must map candidate and baseline")
            pair = (str(entry["candidate"]), str(entry["baseline"]))
            cid = str(entry.get("id") or f"{pair[0]} vs {pair[1]}")
            desc = str(entry.get("description") or entry.get("what") or "")
            out[pair] = (cid, desc)
        return out
    raise ReportError(f"{rel}: comparisons must be an object or list")


def load_experiment_meta(exp_dir: Path, root: Path) -> ExperimentMeta:
    path = exp_dir / "EXPERIMENT.json"
    if not path.is_file():
        return ExperimentMeta(arm_descriptions=dict(ARM_DESCRIPTIONS), comparisons=dict(COMPARISONS))
    rel = path.relative_to(root).as_posix()
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as e:
        raise ReportError(f"{rel}: not readable as JSON ({e})") from None
    if not isinstance(raw, dict):
        raise ReportError(f"{rel}: EXPERIMENT.json must be a JSON object")
    title = raw.get("title")
    if title is not None and not isinstance(title, str):
        raise ReportError(f"{rel}: title must be a string")
    merged_arms = dict(ARM_DESCRIPTIONS)
    for key in ("arms", "arm_descriptions"):
        block = raw.get(key)
        if block is None:
            continue
        if not isinstance(block, Mapping) or not all(isinstance(k, str) and isinstance(v, str) for k, v in block.items()):
            raise ReportError(f"{rel}: {key} must map arm names to strings")
        merged_arms.update(block)
    merged_comps = dict(COMPARISONS)
    merged_comps.update(_parse_meta_comparisons(raw.get("comparisons"), rel))
    return ExperimentMeta(
        title=title.strip() if isinstance(title, str) and title.strip() else None,
        arm_descriptions=merged_arms,
        comparisons=merged_comps,
    )


def load_report(path: Path, root: Path, meta: ExperimentMeta | None = None) -> Row:
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
        meta=meta or ExperimentMeta(arm_descriptions=dict(ARM_DESCRIPTIONS), comparisons=dict(COMPARISONS)),
    )


def load_experiments(root: Path) -> dict[str, list[Row]]:
    """{experiment id: rows}, ordered by experiment id; rows by feature, held-out before dev, arms."""
    groups: dict[str, list[Row]] = {}
    dirs: dict[str, str] = {}
    metas: dict[str, ExperimentMeta] = {}
    for path in sorted((root / "experiments").glob("*/expected/*.json")):
        exp_dir = path.parent.parent
        eid = experiment_id(exp_dir.name)
        if dirs.setdefault(eid, exp_dir.name) != exp_dir.name:
            raise ReportError(f"experiments/{exp_dir.name} and experiments/{dirs[eid]} share the id {eid}")
        if eid not in metas:
            metas[eid] = load_experiment_meta(exp_dir, root)
        groups.setdefault(eid, []).append(load_report(path, root, meta=metas[eid]))
    for eid, rows in groups.items():
        comp_map = metas[eid].comparisons or COMPARISONS
        order = {cid: n for n, (cid, _what) in enumerate(sorted(comp_map.values()))}
        rows.sort(key=lambda w: (order.get(w.comparison_id, len(order)), w.comparison_id, w.feature,
                                 SPLIT_ORDER.get(w.split, 9), w.split))
    return dict(sorted(groups.items()))


def several_comparisons(rows: Sequence[Row]) -> bool:
    return len({r.pair for r in rows}) > 1


def unique_arms(rows: Sequence[Row]) -> list[tuple[Row, Mapping[str, Any]]]:
    """One (row, arm) per arm, feature and split, in ARM_ORDER. An arm's metrics do not depend on the
    arm it is compared with; two reports that disagree about them are refused."""
    seen: dict[tuple[str, str, str], tuple[Row, Mapping[str, Any]]] = {}
    for r in rows:
        for arm in (r.baseline, r.candidate):
            key = (r.feature, r.split, str(arm.get("name")))
            if key in seen and seen[key][1].get("metrics") != arm.get("metrics"):
                raise ReportError(f"{r.source} and {seen[key][0].source} report different metrics for arm "
                                  f"{key[2]} ({key[0]}, {key[1]})")
            seen.setdefault(key, (r, arm))
    extra_arms = rows[0].meta.arm_descriptions if rows else {}
    order = tuple(dict.fromkeys([*ARM_ORDER, *extra_arms]))
    rank = {name: n for n, name in enumerate(order)}
    return [seen[k] for k in sorted(seen, key=lambda k: (k[0], SPLIT_ORDER.get(k[1], 9), k[1],
                                                           rank.get(k[2], len(rank)), k[2]))]


def comparisons_of(rows: Sequence[Row]) -> list[tuple[str, str, str, str]]:
    """(id, candidate, baseline, what it isolates) of each comparison, in row order."""
    out: list[tuple[str, str, str, str]] = []
    comp_map = (rows[0].meta.comparisons if rows else None) or COMPARISONS
    for r in rows:
        if r.comparison_id not in [c[0] for c in out]:
            out.append((r.comparison_id, r.pair[0], r.pair[1], comp_map.get(r.pair, ("", ""))[1]))
    return out


# ----------------------------------------------------------------------------- table content


def arm_primary(row: Row, arm: Mapping[str, Any]) -> str:
    m = arm["metrics"]
    if row.metric == "resolved_rate" and "resolved" in m:
        return f"{num(m['primary'])} ({m['resolved']}/{m['items']})"
    if row.metric == "usable_rate" and "usable" in m:
        return f"{num(m['primary'])} ({m['usable']}/{m['items']})"
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


def tokens_cell(row: Row) -> str:
    """Candidate and baseline mean completion tokens, and their ratio."""
    cand = float(row.candidate["metrics"].get("completion_tokens_mean") or 0)
    base = float(row.baseline["metrics"].get("completion_tokens_mean") or 0)
    ratio = f" ({cand / base:.2f}x)" if base else ""
    return f"{cand:.0f} / {base:.0f}{ratio}"


MULTI_GATE_HEAD = ("Comparison", "Feature", "Split", "Items", "Baseline arm", "Candidate arm", "Baseline", "Candidate",
                   "Δ [95% CI]", "Gains / losses", "McNemar p", "MDE80", "Tokens (cand. / base)", "Verdict",
                   "Failed gates")


def multi_gate_cells(row: Row) -> list[str]:
    c = gate_cells(row)
    return [row.comparison_id, c[0], c[1], c[2], row.pair[1], row.pair[0], c[6], c[7], c[8], c[9], c[10], c[11],
            tokens_cell(row), c[12], c[13]]


ARMS_HEAD = ("Feature", "Split", "Arm", "Model", "Prompt", "Primary [95% CI]", "Schema", "Think",
             "Lenient only", "Truncated", "Mean tokens", "Median latency (s)", "Feature-specific")


def arm_cells(row: Row, arm: Mapping[str, Any]) -> list[str]:
    m = arm["metrics"]
    ci = m.get(f"{row.metric}_ci95")
    if row.feature in ("judge", "review") or "tpr" in m:
        specific = f"TPR {num(m.get('tpr'))}, TNR {num(m.get('tnr'))}, approve {num(m.get('approve_rate'))}"
    else:
        blocks = sum(len(v) if isinstance(v, (list, tuple)) else int(v) for v in ((m.get("sast") or {}).get("block") or {}).values())
        if row.feature == "patch":
            specific = f"applied {num(m.get('applied_rate'))}, gold {num(m.get('gold_match_rate'))}, block SAST {blocks}"
        else:
            specific = (f"overfit {num(m.get('overfit_rate'))}, hidden pass {num(m.get('hidden_pass_rate'))}, "
                        f"block SAST {blocks}")
    finish = m.get("finish_reasons") or {}
    return [row.feature, row.split, str(arm.get("name")), str(arm.get("model")), str(arm.get("prompt")),
            f"{num(m.get('primary'))} {interval(ci)}", num(m.get("schema_validity")), num(m.get("think_validity")),
            num(m.get("lenient_only_rate")), f"{finish.get('length', 0)}/{m.get('items')}",
            f"{float(m.get('completion_tokens_mean') or 0):.0f}", num(m.get("latency_median_s"), 2), specific]


# ----------------------------------------------------------------------------- markdown


def _row_title(eid: str, rows: Sequence[Row]) -> str:
    return experiment_title(eid, rows[0].meta if rows else None)


def _interval_note(rows: Sequence[Row]) -> str:
    if any(r.feature == "patch" or r.metric == "usable_rate" for r in rows):
        return "Wilson (resolved rate, usable patch rate)"
    return "Wilson (resolved rate)"


def md_table(head: Sequence[str], rows: Iterable[Sequence[str]]) -> str:
    lines = ["| " + " | ".join(md(h) for h in head) + " |", "|" + "---|" * len(head)]
    lines += ["| " + " | ".join(md(c) for c in r) + " |" for r in rows]
    return "\n".join(lines) + "\n"


def md_gate(eid: str, rows: Sequence[Row]) -> str:
    sources = ", ".join(f"`{r.source}`" for r in rows)
    return (f"*Table: {_row_title(eid, rows)} (`experiments/{rows[0].experiment}`), gate decisions under rule "
            f"`{rows[0].rule}` on suite `{rows[0].suite.get('name')}`. Δ is candidate minus baseline on the paired "
            f"items, with its paired cluster-bootstrap 95% interval; MDE80 is the minimum detectable effect at 80% "
            f"power. Generated by `{SCRIPT}` from {sources}.*\n\n" + md_table(GATE_HEAD, (gate_cells(r) for r in rows)))


def md_arms(eid: str, rows: Sequence[Row]) -> str:
    body = [arm_cells(r, arm) for r in rows for arm in (r.baseline, r.candidate)]
    return (f"*Table: {_row_title(eid, rows)}, per-arm metrics. Primary intervals are {_interval_note(rows)} or a "
            f"task-cluster bootstrap (balanced accuracy); \"Truncated\" counts replies that ended at `max_tokens`. "
            f"Generated by `{SCRIPT}`.*\n\n" + md_table(ARMS_HEAD, body))


def comparison_legend(rows: Sequence[Row]) -> str:
    return "; ".join(f"{cid}: {cand} against {base}" + (f", {what}" if what else "")
                     for cid, cand, base, what in comparisons_of(rows))


def arm_legend(rows: Sequence[Row]) -> str:
    descriptions = (rows[0].meta.arm_descriptions if rows else None) or ARM_DESCRIPTIONS
    names: list[str] = []
    for _r, arm in unique_arms(rows):
        if str(arm.get("name")) not in names:
            names.append(str(arm.get("name")))
    return "; ".join(f"{n}: {descriptions[n]}" if n in descriptions else n for n in names)


def md_gate_multi(eid: str, rows: Sequence[Row]) -> str:
    return (f"*Table: {_row_title(eid, rows)}, `experiments/{rows[0].experiment}`: gate decisions under rule "
            f"`{rows[0].rule}` on suite `{rows[0].suite.get('name')}`, one row per comparison, feature and split. "
            f"Comparisons: {comparison_legend(rows)}. Δ is candidate minus baseline on the paired items, with its "
            f"paired cluster-bootstrap 95% interval; MDE80 is the minimum detectable effect at 80% power; Tokens are "
            f"the two arms' mean completion tokens per item and their ratio. Generated by `{SCRIPT}` from "
            f"the {len(rows)} reports `experiments/{rows[0].experiment}/expected/<candidate>--vs--<baseline>."
            f"<feature>-<split>.json`.*\n\n" + md_table(MULTI_GATE_HEAD, (multi_gate_cells(r) for r in rows)))


MULTI_ARMS_HEAD = ("Feature", "Split", "Arm", "Primary [95% CI]", "Schema", "Think", "Lenient only", "Truncated",
                   "Mean tokens", "Total tokens", "Median latency (s)", "Feature-specific")


def multi_arm_cells(row: Row, arm: Mapping[str, Any]) -> list[str]:
    c = arm_cells(row, arm)
    total = arm["metrics"].get("completion_tokens_total")
    return [c[0], c[1], c[2], c[5], c[6], c[7], c[8], c[9], c[10], "n/a" if total is None else f"{int(total)}",
            c[11], c[12]]


def _all_models(rows: Sequence[Row]) -> list[str]:
    return sorted({str(arm.get("model")) for r in rows for arm in (r.baseline, r.candidate)})


def md_arms_multi(eid: str, rows: Sequence[Row]) -> str:
    models = _all_models(rows)
    if len(models) == 1:
        model_note = f"Every arm was served under the model name `{models[0]}`."
    else:
        model_note = "Arms were served under the model names " + ", ".join(f"`{m}`" for m in models) + "."
    return (f"*Table: {_row_title(eid, rows)}, per-arm metrics, one row per arm, feature and split. Arms: "
            f"{arm_legend(rows)}. {model_note} Primary intervals are {_interval_note(rows)} or a task-cluster "
            f"bootstrap (balanced accuracy); \"Truncated\" counts replies that ended at `max_tokens`. The manifests "
            f"do not record the hardware or load of a recording, so latency is not comparable across engines. "
            f"Generated by `{SCRIPT}`.*\n\n" + md_table(MULTI_ARMS_HEAD, (multi_arm_cells(r, a) for r, a in unique_arms(rows))))


def md_warnings(rows: Sequence[Row]) -> str:
    lines = []
    multi = several_comparisons(rows)
    for r in rows:
        for w in r.warnings:
            where = f"{r.comparison_id}, " if multi else ""
            lines.append(f"- {where}`{r.feature}`, {r.split}: {w}")
    return "\n".join(lines) + "\n" if lines else "- none\n"


# ----------------------------------------------------------------------------- LaTeX


def tex_table(label: str, caption: str, cols: str, head: Sequence[str], rows: Iterable[Sequence[str]],
              size: str = r"\scriptsize", head_is_tex: bool = False) -> str:
    out = [f"% Generated by {SCRIPT}; do not edit. Regenerate with: python {SCRIPT}",
           r"\begin{table*}[!t]", f"\\caption{{{caption}}}", f"\\label{{{label}}}", r"\centering", size,
           r"\setlength{\tabcolsep}{2pt}", f"\\begin{{tabular}}{{{cols}}}", r"\hline",
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
    caption = (f"{tex(_row_title(eid, rows))}: gate decisions, {arms}rule \\texttt{{{tex(rows[0].rule)}}}, suite "
               f"\\texttt{{{tex(str(rows[0].suite.get('name')))}}}. Primary metric: resolved rate (bugfix), "
               r"usable patch rate (patch), or balanced accuracy (judge, review). $\Delta$ is candidate minus "
               r"baseline on paired items with a paired cluster-bootstrap 95\% interval; G/L are discordant gains "
               r"and losses; MDE$_{80}$ is the minimum detectable effect at 80\% power. Generated from the committed "
               r"reports in \texttt{" + tex(f"experiments/{rows[0].experiment}/expected") + "}.")
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
    caption = (f"{tex(_row_title(eid, rows))}: per-arm metrics. Intervals are Wilson (resolved rate, usable patch "
               "rate) or a task-cluster bootstrap (balanced accuracy). Trunc.\\ counts replies cut at the token limit; "
               "Tokens is the mean completion length.")
    return tex_table(f"tab:{eid}-arms", caption, "lllllrrrrrl", TEX_ARMS_HEAD, body, head_is_tex=True)


TEX_MULTI_GATE_HEAD = ("Comp.", "Feature", "Split", "$n$", "Base", "Cand.", r"$\Delta$ [95\% CI]", "G/L",
                       "McNemar $p$", r"MDE$_{80}$", "Tokens c./b.", "Verdict", "Failed gates")


def tex_gate_multi(eid: str, rows: Sequence[Row]) -> str:
    body = []
    for r in rows:
        c = multi_gate_cells(r)
        body.append([c[0], c[1], c[2], c[3], c[6], c[7], c[8], c[9], c[10], c[11], c[12], c[13], c[14]])
    caption = (f"{tex(_row_title(eid, rows))}: gate decisions, rule \\texttt{{{tex(rows[0].rule)}}}, suite "
               f"\\texttt{{{tex(str(rows[0].suite.get('name')))}}}, one row per comparison, feature and split. "
               f"Comparisons: {tex(comparison_legend(rows))}. Primary metric: resolved rate (bugfix), usable patch "
               r"rate (patch), or balanced accuracy (judge, review); Base and Cand.\ are the baseline's and the "
               r"candidate's. $\Delta$ is candidate minus baseline on paired items with a paired cluster-bootstrap "
               r"95\% interval; G/L are discordant gains and losses; MDE$_{80}$ is the minimum detectable effect at "
               r"80\% power; Tokens are the mean completion tokens per item of candidate and baseline, and their ratio. "
               r"Generated from the committed reports in \texttt{"
               + tex(f"experiments/{rows[0].experiment}/expected") + "}.")
    return tex_table(f"tab:{eid}-gate", caption, r"lllrllllrrllp{0.12\textwidth}", TEX_MULTI_GATE_HEAD, body,
                     head_is_tex=True)


TEX_MULTI_ARMS_HEAD = ("Feature", "Split", "Arm", "Primary [95\\% CI]", "Schema", "Think", "Len.\\ only", "Trunc.",
                       "Tokens", "Total", "Feature-specific")


def tex_arms_multi(eid: str, rows: Sequence[Row]) -> str:
    body = []
    for r, arm in unique_arms(rows):
        c = multi_arm_cells(r, arm)
        body.append([c[0], c[1], c[2], c[3], c[4], c[5], c[6], c[7], c[8], c[9], c[11]])
    caption = (f"{tex(_row_title(eid, rows))}: per-arm metrics. Arms: {tex(arm_legend(rows))}. Intervals are Wilson "
               "(resolved rate, usable patch rate) or a task-cluster bootstrap (balanced accuracy). Trunc.\\ counts "
               "replies cut at the token limit; Tokens is the mean and Total the sum of completion tokens.")
    return tex_table(f"tab:{eid}-arms", caption, "llllrrrrrrl", TEX_MULTI_ARMS_HEAD, body, head_is_tex=True)


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
        multi = several_comparisons(rows)
        blocks[f"{eid}-gate"] = (md_gate_multi if multi else md_gate)(eid, rows)
        blocks[f"{eid}-arms"] = (md_arms_multi if multi else md_arms)(eid, rows)
        files[f"docs/paper/tables/{eid}-gate.tex"] = (tex_gate_multi if multi else tex_gate)(eid, rows)
        files[f"docs/paper/tables/{eid}-arms.tex"] = (tex_arms_multi if multi else tex_arms)(eid, rows)
        doc += [f"## {_row_title(eid, rows)}: `experiments/{rows[0].experiment}`", "", blocks[f"{eid}-gate"],
                blocks[f"{eid}-arms"], "Warnings the reports carry:", "", md_warnings(rows)]
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
            path.write_bytes(content.encode("utf-8"))
            print(f"wrote {rel}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
