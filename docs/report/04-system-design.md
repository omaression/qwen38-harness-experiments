# III. System Design

The harness is a pure standard-library Python package (`src/llm_eval_harness/`, targeting Python 3.10+ with test compatibility verified back to Python 3.9) exposed via the `llm-eval` CLI. Its normative specification lives in `docs/DESIGN.md`. Fig. 1 (`docs/paper/figures/fig-architecture.pdf`) illustrates the offline replay and promotion pipeline.

## A. Architecture and data flow

An evaluation executes eight deterministic stages:

1. **Suite.** `tasks.load_suite` loads `suite.json` (with `kind` $\in \{\text{function}, \text{review}, \text{patch}\}$) and either plain `tasks.jsonl` or deterministic gzip `tasks.jsonl.gz` (directories containing both are refused). `tasks.split_of` assigns tasks to `dev` or `heldout`; `validation.validate_suite` checks structure, reference behavior, SAST cleanliness, and prompt non-leakage.
2. **Items.** A feature (`features.py`) constructs split items: `bugfix:<task>` or `patch:<task>` on `function`/`patch` suites, symmetric `judge:<task>:reference`/`:wrong` pairs on `function` suites, and `review:<task>` (or `review:<task>:reference`/`:wrong` on `function` suites) for `review`.
3. **Requests.** `prompts.build_request` renders an immutable template into a `ChatRequest`; the SHA-256 over the canonical JSON of the exact HTTP body identifies the request.
4. **Replies.** `clients.OpenAICompatClient` queries live endpoints during recording; `CassetteWriter` writes `<feature>.jsonl` or deterministic `<feature>.jsonl.gz` (`gzip.compress(..., mtime=0)`); `ReplayClient` replays cassettes and refuses any row whose recorded `request_sha256` mismatches the rebuilt payload. Mixing plain and gzipped tapes for the same feature is refused.
5. **Scoring.** The feature scores each reply (Section III-C) into an `ItemResult`.
6. **Metrics and comparison.** `metrics.arm_metrics` aggregates per-arm rates and CIs; `metrics.compare` joins candidate and baseline on item identifiers, while `metrics.both_valid_compare` provides a post-hoc diagnostic restricted to mutually schema-valid pairs.
7. **Decision.** `gate.decide` applies rule `noninferior/1` (Section III-G) and prints one canonical JSON report on `stdout`.
8. **Record keeping.** `ledger.py` maintains a hash-chained `ledger.jsonl` recording promotions, rollbacks, and held-out evaluations.

Only typed immutable records in `records.py` cross module boundaries; `metrics` and `gate` depend solely on those records, and `tests/test_contract.py` enforces zero third-party dependencies.

## B. The four features under test

**Bug fix (`bugfix`).** On `function` suites, the model receives a buggy Python function, its docstring, and one failing assertion (`bugfix/v1`, `bugfix/v2`, templates SHA-256-pinned in `tests/test_prompts.py`). It replies with `{"explanation", "patched_code"}`. Only `patched_code` is executed; hidden tests, reference code, and wrong code are withheld. An item is *resolved* (`resolved_rate`, default floor 0.5) when schema-valid, free of added block SAST findings, and passing all shown and hidden tests.

**Judge (`judge`).** On `function` suites, each task yields two paired items: the reference fix (label `true`) and a plausible overfit fix passing the shown test while failing a hidden one (label `false`). The model returns `{"verdict": true | false}`. Any schema-invalid reply counts wrong in its class, so a label-blind baseline scores exactly 0.5 balanced accuracy (default floor 0.7).

**Review (`review`).** Supporting both real-repository `review` suites (e.g., `swe_review_bench`) and `function` suites (rendered via unified diff and numbered source context), template `review/v1` asks for `{"passed", "defect_severity", "line_comments"}`, requiring `defect_severity` $\in \{\text{none}, \text{low}, \text{medium}, \text{high}, \text{critical}\}$ and `line_comments` as JSON objects. Its prediction is `passed`; primary metric is balanced accuracy (default floor 0.65), scored against execution-derived PR resolution labels without running code or SAST.

**Patch (`patch`).** Supporting `patch` suites (e.g., `swe_coder_effort`) and `function` suites, template `patch/v1` elicits `{"explanation", "unified_diff"}` against shown repository files. Scored purely deterministically in Python via `diff_apply.apply_unified_diff`, an item is *usable* (`usable_rate`, default floor 0.7) when the reply is strictly schema-valid, the diff applies cleanly with non-empty file changes (`not noop`), all modified or created `.py`/`.pyi` files whose originals parsed still parse via `ast.parse`, and zero block-class SAST findings are added. Auxiliary metrics record `applied_rate`, `noop_rate`, and `gold_match_rate` (exact text or Python AST equality against `gold_files`).

## C. Reply validation and diff application

Every reply passes through ordered, independent checks:

1. **Thinking (`verdicts.check_think`).** Thinking arrives either in the API `reasoning` field or as a leading `<think>...</think>` block (or closing `</think>` when the chat template supplies the opening tag). Let `present` denote non-blank extracted thinking, `closed` denote non-blank trailing answer text following thinking, `words` count whitespace-separated tokens, and `in_range` denote $w_{\min} \le \text{words} \le w_{\max}$ (1 to 1,200 words for all four features). With `problems` flagging malformed tags (thinking in both places, multiple blocks, or a late `<think>`), validity is defined as

$$\text{valid} = \neg\text{problems} \land \big((\text{present} \land \text{closed} \land \text{in\_range}) \lor (\neg\text{present} \land \neg\text{required})\big).$$

All four features currently set `required = False`, so omitting thinking is valid when tag-clean, whereas unclosed thinking or reasoning outside $[1, 1{,}200]$ words fails.
2. **Schema (`verdicts.check_schema`).** Any reply with `finish_reason == "length"` is immediately invalid. Otherwise, trailing answer text must parse as one strict JSON object matching required keys, exact Python types (`type(v) is bool`), non-blank required strings, no duplicate keys, and no `NaN`/`Infinity`. A lenient parse (stripping one code fence or outermost braces) is recorded solely for diagnostics (`lenient_only_rate`) and SAST accountability.
3. **Pure-Python `--recount` diff application (`diff_apply`).** For `patch`, `apply_unified_diff` parses `diff --git` and `---`/`+++` headers safely in memory without external binaries, refusing path traversal (`..`), leading slashes, backslashes, `.git` segments, binary diffs, and symlink/submodule file modes (`120000`, `160000`). Because LLMs frequently miscompute `@@` hunk line counts, the parser implements `git apply --recount` hunk-boundary detection, matching context exact-first, then unique $\pm 50$-line offset fuzz, stripped trailing-whitespace matching, and unique whole-file fallback.
4. **Static scan and sandbox execution.** Code-producing features (`bugfix`, `patch`) scan added constructs (Section III-D); `bugfix` then executes clean, schema-valid patches in the sandbox (Section III-E).

## D. Static analysis (`sast/1`)

`sast.scan` parses Python via `ast.parse`, tracking scopes, import aliases, global/nonlocal rebinding, chained assignments, subscripting into `sys.modules`/`builtins`, and `getattr(module, "name")` resolution (e.g., catching `from os import system as run; run(cmd)`); unparseable code falls back to raw-text regex matching over block rules. Added findings (`added_findings`) take the multiset difference by `(rule, snippet)` against the original buggy or base file. Six block-class rules block execution/usability and count against the gate; six flag-class rules are logged without blocking. Suite validation guarantees no block-class rule fires on valid reference solutions. The table below documents all twelve rules directly verified against `src/llm_eval_harness/sast.py`.

*Table: Complete 12-rule specification of `sast/1` (`src/llm_eval_harness/sast.py`). Block rules fail `bugfix`/`patch` items and gate promotion; flag rules are diagnostic.*

| Rule | Severity | CWE | Matched constructs |
|---|---|---|---|
| `eval-exec` | block | CWE-95 | built-in `eval`, `exec` calls or values |
| `os-command` | block | CWE-78 | `os.system`, `os.popen`, `os.exec*`, `os.spawn*`, `os.posix_spawn*`, `pty.spawn` |
| `shell-true` | block | CWE-78 | any call passing `shell=True` |
| `unsafe-deserialization` | block | CWE-502 | `pickle`, `cPickle`, `_pickle`, `dill`, `marshal` `.load(s)`, `shelve.open`, unsafe `yaml.load(_all)` |
| `process-control` | block | — | `os._exit`, `os.kill`, `os.killpg`, `atexit.register`, `signal.signal`, `signal.alarm` |
| `interpreter-tamper` | block | — | `sys.settrace`, `sys.setprofile`, `threading.settrace`, `threading.setprofile`, writes into `sys.modules` or `builtins` |
| `subprocess` | flag | CWE-78 | any other `subprocess.*` call |
| `dynamic-import` | flag | CWE-470 | `__import__`, `importlib.import_module`, `importlib.__import__` |
| `file-write` | flag | CWE-73 | write/append `open`/`io.open`, `Path.write_*`, `os.remove`, `os.unlink`, `os.rmdir`, `os.rename`, `os.replace`, `shutil.rmtree`, `shutil.move` |
| `network` | flag | — | imports of `socket`, `ssl`, `urllib.request`, `http.client`, `ftplib`, `smtplib`, `telnetlib`, `requests`, `httpx` |
| `introspection` | flag | — | `sys._getframe`, `inspect.currentframe`, `inspect.stack`, `inspect.getouterframes`, `gc.get_objects`, `gc.get_referrers` |
| `exit-call` | flag | — | `sys.exit`, `exit`, `quit` |

## E. Execution and the sandbox

In `bugfix`, tests are stored as data: a call expression paired with either an expected Python literal (with optional numeric tolerance) or an expected exception class. The child runs via `python -I -S` in a fresh empty environment and temporary directory with a 10 s timeout, process-group kill on expiry, and best-effort POSIX limits (1 GiB address space, 10 MiB file size). It returns only per-call `repr` values or raised exception MRO names; the parent decides `values/1` verdicts so exit status 0 or skipped calls never pass. Infrastructure failures abort with exit status 1 rather than scoring the model wrong. As documented in `docs/DESIGN.md`, this isolates honest regressions rather than untrusted arbitrary code, which should run inside network-isolated containers.

## F. Statistics and post-hoc decomposition

**Rates.** Every proportion $k/n$ carries a Wilson score interval [26], [27] with $z = 1.96$:

$$\frac{k + z^2/2}{n + z^2} \pm \frac{z \sqrt{k(n-k)/n + z^2/4}}{n + z^2},$$

avoiding interval collapse at $k = 0$ or $k = n$.

**Classifier metrics (`judge`, `review`).** Counting invalid replies as misses in their true class, balanced accuracy is

$$\mathrm{BA} = \tfrac{1}{2}\left(\frac{TP}{P} + \frac{TN}{N}\right),$$

paired with cluster-bootstrap CIs and a flag (`vs_chance`) whenever the interval includes 0.5.

**Paired comparison and McNemar's test.** Arms join deterministically on `item_id`. Let discordant counts be $c$ gains ($c_i=1, b_i=0$) and $b$ losses ($c_i=0, b_i=1$). The 95% interval of $\Delta = \text{primary}(\text{cand}) - \text{primary}(\text{base})$ comes from 2,000 paired bootstrap resamples of *clusters* [23] seeded with `<seed>/<feature>/<split>`. Exact two-sided McNemar's test [16] evaluates

$$p = \min\left(1,\; 2 \sum_{k=0}^{\min(b,c)} \binom{b+c}{k} 2^{-(b+c)}\right),$$

requiring at least six unopposed losses for $p < 0.05$. Every report also reports the 80%-power detection floor $\mathrm{MDE}_{80} = (z_{0.975} + z_{0.80})\,\widehat{\mathrm{se}} \approx 2.80\,\widehat{\mathrm{se}}$.

**Post-hoc `metrics.both_valid_compare`.** Because end-to-end metrics combine instruction/format compliance with latent reasoning quality, `metrics.both_valid_compare` recomputes the identical cluster-bootstrap and McNemar comparison strictly over the subset of paired items where both arms produced `schema_valid` replies. This decomposition is strictly diagnostic and never alters pre-specified gate verdicts.

## G. The gate rule (`noninferior/1`)

Rule `noninferior/1` promotes only if all applicable gates below pass. Refusals (mismatched features, splits, suites, or item sets, or missing/mismatched cassettes) exit 2 with no report; valid runs exit 0 regardless of verdict so CI reads `jq -e '.promote == true'`. No timestamps, hostnames, or absolute paths appear in reports.

*Table: Gates of rule `noninferior/1` (`src/llm_eval_harness/gate.py`). Defaults: margin 0, `--tie pass`, alpha 0.05.*

| Gate | Passes when |
|---|---|
| `schema_validity` | candidate strict schema validity $\ge 0.95$ |
| `think_validity` | candidate thinking validity $\ge 0.90$ |
| `new_sast_violations` | zero `(item, rule)` block findings absent from baseline (`bugfix`, `patch` only) |
| `primary` | $\Delta \ge -\text{margin}$ (`--tie pass`), or $\Delta > -\text{margin}$ (`--tie fail`); against feature floor if no baseline |
| `regression_test` | not ($\text{losses} > \text{gains}$ and McNemar $p < \text{alpha}$) |
| `gain_test` | only under `--require-gain`: $\text{gains} > \text{losses}$ and McNemar $p < \text{alpha}$ |

## H. Promotion ledger, CI, and table sync

`ledger/ledger.jsonl` chains every `header`, `promote`, `rollback`, and `evaluation` row by SHA-256 and tracks held-out evaluations against a budget warning at 10 distinct candidates. The table below summarizes `.github/workflows/ci.yml`: on every push and pull request, `tests` runs unit tests, validates all four suites, verifies table synchronization and ledger integrity; `regression-gate` enforces promotion of the synthetic candidate and rejection of a deliberately regressed self-test arm; and `experiments` replays all 40 committed experiment reports byte for byte. `scripts/make_report_tables.py` renders Markdown and LaTeX tables directly from committed JSON reports without recomputing statistics, enforced by `tests/test_report_tables.py`.

*Table: Jobs of `.github/workflows/ci.yml` (read-only repository permissions).*

| Job | Trigger | Function |
|---|---|---|
| `tests` | every push to `main`, PRs | pytest suite on Python 3.10–3.12; suite validation across `demo`, `humanevalfix`, `swe_review_bench`, `swe_coder_effort`; `make_report_tables.py --check`; `ledger verify` |
| `regression-gate` | after `tests` | replays demo cassettes, promotes synthetic candidate, verifies expected failed-gate rejection of regressed self-test arm |
| `experiments` | after `tests` | replays all 40 committed experiment reports (Experiments A–D) and requires byte-identical `cmp` agreement |
| `live-config`, `live-eval` | manual dispatch | optionally records and gates a live OpenAI-compatible endpoint when secrets are configured |
