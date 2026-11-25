# CLAUDE.md — Contributor & Agent Guide (`llm-feature-eval-harness`)

## 1. Purpose & Zero-Dependency Contract

`llm-feature-eval-harness` is a standalone, MIT-licensed evaluation and CI regression-gating harness for LLM coding and review features.

- **Standard-library only**: Every module under `src/llm_eval_harness/` imports Python 3.11+ standard-library modules and internal package modules only. `tests/test_contract.py` parses every source AST and fails on any third-party runtime dependency.
- **Offline deterministic replay**: CI never calls a live LLM endpoint unless explicitly triggered via `workflow_dispatch` with repository secrets. Regression gates (`cassettes/`) and empirical experiments (`experiments/`) replay recorded `ChatResponse` tapes through `ReplayClient`, verifying each request's canonical SHA-256 payload hash before scoring.

## 2. Supported Features & Suite Kinds

| Feature | Suite Kind(s) | Prompt | Primary Metric | Writes Code / SAST |
| :--- | :--- | :--- | :--- | :--- |
| `bugfix` | `function` | `bugfix/v2` (`v1`) | `resolved_rate` | Yes (`exec_verdict` + `sast`) |
| `judge` | `function` | `judge/v2` (`v1`) | `balanced_accuracy` | No (paired `reference`/`wrong`) |
| `review` | `review`, `function` | `review/v1` | `balanced_accuracy` | No (diff defect classification) |
| `patch` | `patch`, `function` | `patch/v1` | `usable_rate` | Yes (`diff_apply` + AST + `sast`) |

Suites live under `tasks/<name>/` (`suite.json` + `tasks.jsonl` or `tasks.jsonl.gz`) and split deterministically into `dev` and `heldout` by hashing `normalize_key(task.group or task.id)` modulo 100 (`bucket < heldout_pct`).

## 3. Strict Repository Contracts (Enforced by `pytest`)

1. **Public-safety scan (`tests/test_contract.py`)**:
   - Scans all tracked/untracked text files (`.py`, `.md`, `.toml`, `.yml`, `.yaml`, `.json`, `.jsonl`, `.tex`, `.bib`, `.txt`, `.sh`) and decompressed `.jsonl.gz` archives.
   - Refuses any home-directory path (`/(users|home)/...`), any literal e-mail address, any API key token, or private HPC identifiers. Never place literal e-mail addresses anywhere in code, docs, or LaTeX sources.
2. **Documentation CLI sync (`tests/test_contract.py`)**:
   - Every `llm-eval` command in fenced shell blocks and inline code spans of `README.md` and `docs/DESIGN.md` is parsed against `cli.build_parser()`. Update docs and CLI flags together.
3. **CI workflow subset YAML (`tests/test_workflow.py`)**:
   - `.github/workflows/ci.yml` is parsed without PyYAML using a strict block-mapping subset: 2-space job keys, 6-space `- ` step entries, `run: |` blocks, global `defaults.run.shell: bash` (no per-job `shell:`), `permissions: contents: read` once, and pinned `@v7` actions (`persist-credentials: false`).
4. **Generated tables & paper sync (`tests/test_report_tables.py`)**:
   - `python scripts/make_report_tables.py` generates `docs/report/tables.md`, `docs/paper/tables/*.tex`, and marked blocks in `docs/report/*.md` directly from `experiments/*/expected/*.json`. Never hand-edit generated tables; run `--check` before committing.

## 4. Verification Commands

```bash
# Run unit, contract, table, and CI workflow tests
python -m pytest -q

# Validate task suites
llm-eval tasks validate --suite tasks/demo
llm-eval tasks validate --suite tasks/humanevalfix
llm-eval tasks validate --suite tasks/swe_review_bench
llm-eval tasks validate --suite tasks/swe_coder_effort

# Check generated Markdown & LaTeX tables and ledger hash chain
python scripts/make_report_tables.py --check
llm-eval ledger verify
```

## 5. Agent Workflow & Model Tiering

- **Light tier (`flash_lite` / Haiku)**: Contract scans, status checks, formatting checks, and straightforward data plumbing.
- **Mid tier (`flash` / Sonnet)**: Converters, scripts, CI updates, and unit tests.
- **Frontier tier (`pro` / Opus)**: Statistical estimators, gate semantics, multi-module design, and scientific prose/LaTeX verification.
