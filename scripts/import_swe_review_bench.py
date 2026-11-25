#!/usr/bin/env python3
import argparse
import gzip
import json
import os
import re
import sys
from pathlib import Path

# Add src to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from llm_eval_harness.records import ModelSettings, ChatResponse
from llm_eval_harness.cassettes import CassetteWriter
from llm_eval_harness import prompts, tasks, features

def sanitize_public_text(s: str) -> str:
    if not isinstance(s, str):
        return s
    s = re.sub(r"(?i)/(?:users|home)/([a-z0-9_.-]+)/", r"/work/\1/", s)
    s = re.sub(r"([A-Za-z0-9._%+-]+)@([A-Za-z0-9-]+(?:\.[A-Za-z0-9-]+)*\.[A-Za-z]{2,})", r"\1[at]\2", s)
    return s

def slugify(text: str) -> str:
    text = text.lower()
    text = re.sub(r"[^a-z0-9]+", "-", text)
    return text.strip("-")

def main():
    parser = argparse.ArgumentParser(description="Import SWE-Review-Bench and build experiments")
    parser.add_argument("--prompts", required=True, help="Path to review_swe-review-bench.jsonl")
    parser.add_argument("--base-gen", required=True, help="Path to base generations jsonl")
    parser.add_argument("--lora-gen", required=True, help="Path to lora generations jsonl")
    parser.add_argument("--root", required=True, help="Harness root directory (/tmp/harness-stage)")
    args = parser.parse_args()

    root = Path(args.root).resolve()
    prompts_path = Path(args.prompts).resolve()
    base_gen_path = Path(args.base_gen).resolve()
    lora_gen_path = Path(args.lora_gen).resolve() if args.lora_gen else None

    # 1. Load prompts jsonl (400 rows)
    prompt_rows = {}
    with open(prompts_path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            obj = json.loads(line)
            item_id = obj.get("id") or obj.get("instance_id")
            prompt_rows[item_id] = obj

    print(f"Loaded {len(prompt_rows)} prompt rows from {prompts_path}")

    # Load base generations
    base_gens = {}
    with open(base_gen_path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            obj = json.loads(line)
            item_id = obj.get("id") or obj.get("item_id")
            if item_id in prompt_rows:
                base_gens[item_id] = obj

    # Load lora generations
    lora_gens = {}
    with open(lora_gen_path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            obj = json.loads(line)
            item_id = obj.get("id") or obj.get("item_id")
            if item_id in prompt_rows:
                lora_gens[item_id] = obj

    print(f"Loaded {len(base_gens)} base gens, {len(lora_gens)} lora gens matching prompt rows.")

    # Build tasks and suite.json
    tasks_list = []
    seen_task_ids = set()

    for orig_id, row in prompt_rows.items():
        suffix = orig_id
        if "review:Lego-X/SWE-Review-Bench:" in suffix:
            suffix = suffix.split("review:Lego-X/SWE-Review-Bench:")[-1]
        elif "Lego-X/SWE-Review-Bench:" in suffix:
            suffix = suffix.split("Lego-X/SWE-Review-Bench:")[-1]
        
        task_id = slugify(suffix)
        if not task_id or not re.match(r"^[a-z0-9][a-z0-9-]*$", task_id):
            task_id = slugify(orig_id.replace("review:", "").replace("Lego-X/SWE-Review-Bench:", ""))
        
        if task_id in seen_task_ids:
            task_id = f"{task_id}-{len(seen_task_ids)}"
        seen_task_ids.add(task_id)

        group_part = suffix
        if "__" in group_part:
            parts = group_part.split("__")
            group_part = "__".join(parts[:2]) if len(parts) >= 2 else parts[0]
        group = slugify(group_part)

        repo = row.get("repo", "unknown")
        tags = [slugify(repo)]

        # Extract user_prompt from messages where role == 'user'
        user_prompt = ""
        for m in row.get("messages", []):
            if m.get("role") == "user":
                user_prompt = sanitize_public_text(m.get("content", ""))
                break
        if not user_prompt:
            # fallback to meta or direct user_prompt key if any
            user_prompt = sanitize_public_text(row.get("user_prompt", ""))
        label = bool(row.get("gold", {}).get("passed", False))
        patch_files = row.get("accept", {}).get("patch_files", [])

        task_obj = {
            "id": task_id,
            "user_prompt": user_prompt,
            "label": label,
            "patch_files": patch_files,
            "group": group,
            "tags": tags,
            "synthetic": False
        }
        tasks_list.append((orig_id, task_obj))

    tasks_list.sort(key=lambda x: x[1]["id"])

    suite_dir = root / "tasks" / "swe_review_bench"
    suite_dir.mkdir(parents=True, exist_ok=True)

    suite_json = {
        "schema": "llm_eval_harness.suite/1",
        "name": "swe_review_bench",
        "version": 1,
        "kind": "review",
        "heldout_pct": 50,
        "synthetic": False,
        "description": "400 real pull-request review items from Apache-2.0 Lego-X/SWE-Review-Bench across 12 Python repositories, clustered by PR instance."
    }

    with open(suite_dir / "suite.json", "w", encoding="utf-8") as f:
        json.dump(suite_json, f, indent=2)
        f.write("\n")

    tasks_jsonl_bytes = b"".join((json.dumps(t[1], sort_keys=True) + "\n").encode("utf-8") for t in tasks_list)
    with open(suite_dir / "tasks.jsonl.gz", "wb") as f:
        f.write(gzip.compress(tasks_jsonl_bytes, mtime=0))

    print(f"Wrote suite.json and tasks.jsonl.gz with {len(tasks_list)} tasks.")

    loaded_suite = tasks.load_suite(suite_dir)
    review_feature = features.get_feature("review")
    
    template = prompts.get_template("review/v1")
    settings = ModelSettings(temperature=0.0, max_tokens=16384, seed=None, extra={"top_p": 0.95, "top_k": 20, "presence_penalty": 0.0})

    exp_dir = root / "experiments" / "exp-b-review-lora"
    exp_dir.mkdir(parents=True, exist_ok=True)

    with open(exp_dir / "SUITE", "w", encoding="utf-8") as f:
        f.write("swe_review_bench\n")

    exp_json = {
        "title": "SWE-Review-Bench Qwen3.8 Base vs Review LoRA",
        "provenance": "Apache-2.0 Lego-X/SWE-Review-Bench",
        "arms": {
            "qwen38-base": "Base Qwen3.8-27B abliterated model generations",
            "qwen38-review-lora": "SFT LoRA r=32 alpha=64 w700 review model generations"
        }
    }
    with open(exp_dir / "EXPERIMENT.json", "w", encoding="utf-8") as f:
        json.dump(exp_json, f, indent=2)
        f.write("\n")

    for arm_name, model_name, gen_dict in [
        ("qwen38-base", "qwen38", base_gens),
        ("qwen38-review-lora", "public-v2-review", lora_gens)
    ]:
        arm_dir = exp_dir / arm_name
        writer = CassetteWriter(
            root=arm_dir,
            arm=arm_name,
            feature="review",
            prompt="review/v1",
            model=model_name,
            settings=settings,
            suite_name=loaded_suite.name,
            suite_sha256=loaded_suite.sha256,
            synthetic=False,
            compress=True,
            description=f"SWE-Review-Bench cassette for {arm_name}"
        )

        for orig_id, task_obj in tasks_list:
            item_id = f"review:{task_obj['id']}"
            item_match = None
            for it in review_feature.items(loaded_suite, "all"):
                if it.id == item_id:
                    item_match = it
                    break
            if not item_match:
                raise RuntimeError(f"Could not find feature item for {item_id}")

            req = prompts.build_request(template, item_match, model_name, settings)
            rec = gen_dict.get(orig_id)
            if not rec:
                raise RuntimeError(f"Missing generation for {orig_id} in {arm_name}")

            content = sanitize_public_text(rec.get("content") or "")
            reasoning = sanitize_public_text(rec.get("reasoning")) if rec.get("reasoning") else None
            finish_reason = rec.get("finish_reason")
            usage_dict = rec.get("usage", {"prompt_tokens": 0, "completion_tokens": 0})
            
            resp = ChatResponse(
                content=content,
                reasoning=reasoning,
                finish_reason=finish_reason,
                usage={"prompt_tokens": int(usage_dict.get("prompt_tokens", 0)), "completion_tokens": int(usage_dict.get("completion_tokens", 0))},
                latency_s=rec.get("latency_s", 0.0)
            )
            writer.write(req, resp)

        writer.close()
        print(f"Wrote cassette for {arm_name}")

    print("Import and cassette generation completed successfully!")

if __name__ == "__main__":
    main()
