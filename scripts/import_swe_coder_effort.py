#!/usr/bin/env python3
"""Import the Nemotron-Cascade-RL-SWE coder-effort subset as tasks/swe_coder_effort and
experiments/exp-d-coder-effort.

    python scripts/import_swe_coder_effort.py \
        --candidates PATH/coder.jsonl \
        --xhigh-dir PATH/xhigh/gen \
        --medium-dir PATH/medium/gen \
        --low-dir PATH/low/gen \
        --root .

Standard library only (plus llm_eval_harness from src/).
"""

from __future__ import annotations

import argparse
import gzip
import json
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(DEFAULT_ROOT / "src"))

from llm_eval_harness import features, prompts, tasks, validation  # noqa: E402
from llm_eval_harness.cassettes import CassetteWriter  # noqa: E402
from llm_eval_harness.records import ChatResponse, ModelSettings, Suite, Task  # noqa: E402

SUITE_NAME = "swe_coder_effort"
EXP_DIR_NAME = "exp-d-coder-effort"
FEATURE_NAME = "patch"
PROMPT_ID = "patch/v1"
MODEL_NAME = "qwen38"

_USER_PATH_RE = re.compile(r"(?i)/(?:users|home)/([a-z0-9_.-]+)/")
_EMAIL_RE = re.compile(r"([A-Za-z0-9._%+-]+)@([A-Za-z0-9-]+(?:\.[A-Za-z0-9-]+)*\.[A-Za-z]{2,})")

COMPARISONS: tuple[tuple[str, str, str, str], ...] = (
    ("D1", "effort-medium", "effort-xhigh", "Medium vs xhigh reasoning effort on patch generation"),
    ("D2", "effort-low", "effort-xhigh", "Low vs xhigh reasoning effort on patch generation"),
    ("D3", "effort-low", "effort-medium", "Low vs medium reasoning effort on patch generation"),
)


def sanitize_public_text(s: str) -> str:
    """Replace home/user paths and email addresses for public-repo safety."""
    if not isinstance(s, str):
        return s
    s = _USER_PATH_RE.sub(r"/work/\1/", s)
    s = _EMAIL_RE.sub(r"\1[at]\2", s)
    return s


def _sanitize_json_value(val: Any) -> Any:
    if isinstance(val, str):
        return sanitize_public_text(val)
    if isinstance(val, list):
        return [_sanitize_json_value(item) for item in val]
    if isinstance(val, dict):
        return {k: _sanitize_json_value(v) for k, v in val.items()}
    return val


def _strict_object_pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for k, v in pairs:
        if k in out:
            raise ValueError(f"duplicate key {k!r}")
        out[k] = v
    return out


def _no_json_constant(name: str) -> Any:
    raise ValueError(f"{name} is not JSON")


def sanitize_response_content(content: str) -> str:
    """Sanitize response.content; preserve JSON escaping when content is a JSON dict."""
    try:
        parsed = json.loads(
            content,
            object_pairs_hook=_strict_object_pairs,
            parse_constant=_no_json_constant,
        )
    except (json.JSONDecodeError, TypeError, ValueError):
        return sanitize_public_text(content)
    if isinstance(parsed, dict):
        return json.dumps(_sanitize_json_value(parsed), ensure_ascii=False)
    return sanitize_public_text(content)


def slugify(text: str) -> str:
    """Normalise an identifier or repo name into [a-z0-9-]+."""
    return re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")


def load_candidates(path: Path) -> dict[str, dict[str, Any]]:
    """Load candidate records keyed by original id."""
    out: dict[str, dict[str, Any]] = {}
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)
            out[row["id"]] = row
    return out


def load_generation_dir(gen_dir: Path) -> dict[str, dict[str, Any]]:
    """Load sample-0 generation records from all *.jsonl shards in gen_dir."""
    shards = sorted(gen_dir.glob("*.jsonl"))
    if not shards:
        raise FileNotFoundError(f"No *.jsonl files found in {gen_dir}")
    out: dict[str, dict[str, Any]] = {}
    for shard in shards:
        with open(shard, "r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                row = json.loads(line)
                if row.get("sample", 0) != 0:
                    continue
                out[row["id"]] = row
    return out


def _extract_user_prompt(row: dict[str, Any]) -> str:
    for msg in row.get("messages", []):
        if msg.get("role") == "user":
            return sanitize_public_text(msg.get("content", ""))
    return sanitize_public_text(row.get("user_prompt", ""))


def build_candidate_tasks(
    target_ids: list[str],
    candidates: dict[str, dict[str, Any]],
) -> list[tuple[str, Task, dict[str, Any]]]:
    """Convert selected candidate rows into (orig_id, Task, task_json_dict) tuples."""
    seen_ids: set[str] = set()
    built: list[tuple[str, Task, dict[str, Any]]] = []

    for orig_id in sorted(target_ids):
        if orig_id not in candidates:
            raise KeyError(f"Candidate {orig_id!r} not found in candidates file")
        row = candidates[orig_id]
        suffix = orig_id.split("coder:nvidia/Nemotron-Cascade-RL-SWE:")[-1]
        task_id = slugify(suffix)
        if not task_id or not re.fullmatch(r"[a-z0-9][a-z0-9-]*", task_id):
            task_id = slugify(orig_id)
        if task_id in seen_ids:
            idx = 2
            while f"{task_id}-{idx}" in seen_ids:
                idx += 1
            task_id = f"{task_id}-{idx}"
        seen_ids.add(task_id)

        repo = row.get("repo", "unknown")
        group = slugify(repo)
        tags = (group,)
        user_prompt = _extract_user_prompt(row)

        accept = row["accept"]
        files = {k: sanitize_public_text(v) for k, v in accept["files"].items()}
        gold_files = {
            k: (sanitize_public_text(v) if isinstance(v, str) else v)
            for k, v in accept["gold_files"].items()
        }

        task = Task(
            id=task_id,
            group=group,
            tags=tags,
            synthetic=False,
            kind=FEATURE_NAME,
            user_prompt=user_prompt,
            files=files,
            gold_files=gold_files,
        )
        built.append((orig_id, task, task.to_json()))

    built.sort(key=lambda entry: entry[1].id)
    return built


def filter_valid_tasks(
    built: list[tuple[str, Task, dict[str, Any]]],
    suite_name: str,
) -> tuple[list[tuple[str, Task, dict[str, Any]]], list[dict[str, Any]]]:
    """Run validation.validate_suite and drop only tasks that report validation problems."""
    probe_suite = Suite(
        name=suite_name,
        version=1,
        heldout_pct=50,
        tasks=tuple(entry[1] for entry in built),
        sha256="probe",
        source=f"tasks/{suite_name}",
        synthetic=False,
        description="validation probe",
        kind=FEATURE_NAME,
    )
    report = validation.validate_suite(probe_suite, execute=False)
    if not report.problems:
        return built, []

    reasons_by_id: dict[str, list[str]] = {}
    for problem in report.problems:
        matched = False
        for _, task, _ in built:
            if problem.startswith(f"{task.id}:") or problem.startswith(f"task {task.id!r}:"):
                reasons_by_id.setdefault(task.id, []).append(problem)
                matched = True
                break
        if not matched:
            raise RuntimeError(f"Suite-level validation problem cannot be attributed to one task: {problem}")

    kept: list[tuple[str, Task, dict[str, Any]]] = []
    dropped: list[dict[str, Any]] = []
    for orig_id, task, task_json in built:
        if task.id in reasons_by_id:
            dropped.append(
                {
                    "id": task.id,
                    "orig_id": orig_id,
                    "reasons": reasons_by_id[task.id],
                }
            )
        else:
            kept.append((orig_id, task, task_json))
    return kept, dropped


def write_suite(
    suite_dir: Path,
    kept: list[tuple[str, Task, dict[str, Any]]],
    repo_count: int,
) -> Suite:
    """Write suite.json and tasks.jsonl.gz, then reload and validate."""
    suite_dir.mkdir(parents=True, exist_ok=True)
    plain_tasks = suite_dir / "tasks.jsonl"
    if plain_tasks.exists():
        plain_tasks.unlink()

    suite_json = {
        "schema": "llm_eval_harness.suite/1",
        "name": SUITE_NAME,
        "version": 1,
        "kind": FEATURE_NAME,
        "heldout_pct": 50,
        "synthetic": False,
        "description": (
            f"{len(kept)} real multi-file patch tasks from CC-BY-4.0 nvidia/Nemotron-Cascade-RL-SWE "
            f"across {repo_count} repositories, clustered by repository (owner-name) for 50/50 dev/heldout split."
        ),
    }
    with open(suite_dir / "suite.json", "w", encoding="utf-8") as f:
        json.dump(suite_json, f, indent=2, ensure_ascii=False)
        f.write("\n")

    lines = "".join(json.dumps(entry[2], sort_keys=True, ensure_ascii=False) + "\n" for entry in kept)
    with open(suite_dir / "tasks.jsonl.gz", "wb") as f:
        f.write(gzip.compress(lines.encode("utf-8"), mtime=0))

    loaded_suite = tasks.load_suite(suite_dir)
    val_report = validation.validate_suite(loaded_suite, execute=False)
    if not val_report.valid:
        raise RuntimeError(f"Filtered suite still failed validation: {val_report.problems}")
    return loaded_suite


def write_arm_cassettes(
    exp_dir: Path,
    level: str,
    loaded_suite: Suite,
    kept: list[tuple[str, Task, dict[str, Any]]],
    generations: dict[str, dict[str, Any]],
) -> None:
    """Write manifest.json + patch.jsonl.gz for one reasoning-effort arm."""
    arm_name = f"effort-{level}"
    arm_dir = exp_dir / arm_name
    if arm_dir.exists():
        shutil.rmtree(arm_dir)
    settings = ModelSettings(
        temperature=0.1,
        max_tokens=32768,
        seed=None,
        extra={
            "reasoning_effort": level,
            "top_p": 0.95,
            "top_k": 20,
            "presence_penalty": 0.0,
        },
    )
    template = prompts.get_template(PROMPT_ID, FEATURE_NAME)
    patch_feature = features.get_feature(FEATURE_NAME)
    items_by_id = {item.id: item for item in patch_feature.items(loaded_suite, "all")}

    writer = CassetteWriter(
        root=arm_dir,
        arm=arm_name,
        feature=FEATURE_NAME,
        prompt=PROMPT_ID,
        model=MODEL_NAME,
        settings=settings,
        suite_name=loaded_suite.name,
        suite_sha256=loaded_suite.sha256,
        synthetic=False,
        compress=True,
        description=f"Qwen3.8-27B base patch generations at reasoning_effort={level}",
    )
    try:
        for orig_id, task, _ in kept:
            item_id = f"{FEATURE_NAME}:{task.id}"
            item = items_by_id[item_id]
            req = prompts.build_request(template, item, MODEL_NAME, settings)
            rec = generations.get(orig_id)
            if rec is None:
                raise KeyError(f"Missing generation for {orig_id!r} in arm {arm_name!r}")

            content = sanitize_response_content(rec.get("content") or "")
            raw_reasoning = rec.get("reasoning")
            reasoning = sanitize_public_text(raw_reasoning) if raw_reasoning is not None else None
            finish_reason = rec.get("finish_reason")
            raw_usage = rec.get("usage") or {}
            usage = {
                str(k): int(v)
                for k, v in raw_usage.items()
                if isinstance(v, int) and not isinstance(v, bool)
            }
            latency_s = rec.get("latency_s")
            resp = ChatResponse(
                content=content,
                reasoning=reasoning,
                finish_reason=finish_reason,
                usage=usage,
                latency_s=float(latency_s) if latency_s is not None else None,
            )
            writer.write(req, resp)
    finally:
        writer.close()


def run_gate_reports(root: Path, exp_dir: Path) -> list[Path]:
    """Generate the 6 expected gate reports (D1/D2/D3 x heldout/dev) from cwd=root."""
    expected_dir = exp_dir / "expected"
    expected_dir.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []

    env = os.environ.copy()
    src_str = str(root / "src")
    existing_pp = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = f"{src_str}{os.pathsep}{existing_pp}" if existing_pp else src_str

    for comp_id, cand, base, _ in COMPARISONS:
        for split in ("heldout", "dev"):
            out_path = expected_dir / f"{cand}--vs--{base}.{FEATURE_NAME}-{split}.json"
            cmd = [
                sys.executable,
                "-m",
                "llm_eval_harness",
                "gate",
                "--feature",
                FEATURE_NAME,
                "--suite",
                f"tasks/{SUITE_NAME}",
                "--split",
                split,
                "--candidate",
                f"experiments/{EXP_DIR_NAME}/{cand}",
                "--baseline",
                f"experiments/{EXP_DIR_NAME}/{base}",
            ]
            res = subprocess.run(cmd, cwd=root, env=env, capture_output=True, text=True)
            if res.returncode not in (0, 1):
                raise RuntimeError(
                    f"Gate {comp_id} ({cand} vs {base}, {split}) failed with exit code {res.returncode}:\n{res.stderr}"
                )
            if not res.stdout.strip():
                raise RuntimeError(
                    f"Gate {comp_id} ({cand} vs {base}, {split}) produced empty stdout (exit {res.returncode}):\n{res.stderr}"
                )
            json.loads(res.stdout)
            out_path.write_text(res.stdout, encoding="utf-8")
            written.append(out_path)
    return written


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Import swe_coder_effort suite and exp-d-coder-effort cassettes and gate reports."
    )
    parser.add_argument("--candidates", required=True, type=Path, help="Path to candidates coder.jsonl")
    parser.add_argument("--xhigh-dir", required=True, type=Path, help="Directory with xhigh *.jsonl shards")
    parser.add_argument("--medium-dir", required=True, type=Path, help="Directory with medium *.jsonl shards")
    parser.add_argument("--low-dir", required=True, type=Path, help="Directory with low *.jsonl shards")
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT, help="Harness repository root")
    parser.add_argument("--skip-gates", action="store_true", help="Skip generating expected/ gate reports")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    root = args.root.resolve()

    candidates = load_candidates(args.candidates.resolve())
    gen_xhigh = load_generation_dir(args.xhigh_dir.resolve())
    gen_medium = load_generation_dir(args.medium_dir.resolve())
    gen_low = load_generation_dir(args.low_dir.resolve())

    target_ids = sorted(gen_medium.keys())
    if set(target_ids) != set(gen_xhigh.keys()) or set(target_ids) != set(gen_low.keys()):
        raise RuntimeError("Candidate ID sets differ across effort levels")

    built = build_candidate_tasks(target_ids, candidates)
    kept, dropped = filter_valid_tasks(built, SUITE_NAME)
    repo_count = len({entry[1].group for entry in kept})

    suite_dir = root / "tasks" / SUITE_NAME
    loaded_suite = write_suite(suite_dir, kept, repo_count)

    exp_dir = root / "experiments" / EXP_DIR_NAME
    exp_dir.mkdir(parents=True, exist_ok=True)
    (exp_dir / "SUITE").write_text(f"{SUITE_NAME}\n", encoding="utf-8")

    exp_meta = {
        "title": "Qwen3.8-27B Coder Reasoning Effort Ablation (xhigh vs medium vs low)",
        "suite": SUITE_NAME,
        "feature": FEATURE_NAME,
        "provenance": "CC-BY-4.0 nvidia/Nemotron-Cascade-RL-SWE",
        "source_candidates": len(built),
        "kept_tasks": len(kept),
        "dropped_count": len(dropped),
        "dropped_tasks": dropped,
        "repositories": repo_count,
        "splits": tasks.split_counts(loaded_suite),
        "arms": {
            "effort-xhigh": "Qwen3.8-27B base with reasoning_effort=xhigh",
            "effort-medium": "Qwen3.8-27B base with reasoning_effort=medium",
            "effort-low": "Qwen3.8-27B base with reasoning_effort=low",
        },
        "comparisons": {
            cid: {"candidate": cand, "baseline": base, "description": desc}
            for cid, cand, base, desc in COMPARISONS
        },
    }
    with open(exp_dir / "EXPERIMENT.json", "w", encoding="utf-8") as f:
        json.dump(exp_meta, f, indent=2, ensure_ascii=False)
        f.write("\n")

    for level, gens in (("xhigh", gen_xhigh), ("medium", gen_medium), ("low", gen_low)):
        write_arm_cassettes(exp_dir, level, loaded_suite, kept, gens)

    print(
        f"Built {SUITE_NAME}: kept={len(kept)}, dropped={len(dropped)}, "
        f"repos={repo_count}, splits={exp_meta['splits']}"
    )

    if not args.skip_gates:
        reports = run_gate_reports(root, exp_dir)
        print(f"Wrote {len(reports)} gate reports to {exp_dir / 'expected'}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
