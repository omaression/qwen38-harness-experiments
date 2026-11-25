# Design

This document describes what the harness evaluates, how data flows through it, how every
number in a report is defined, the rule the gate applies, and what CI does with it. The module
docstrings in `src/llm_eval_harness/` hold the same contracts in more detail; where the two
disagree, that is a bug.

## 1. What this is

**The feature under test** is an LLM bug-fix assistant. Its input is a buggy Python function,
the function's docstring and one test that fails on it. Its output is an optional
`<think>...</think>` block followed by exactly one JSON object:

```json
{"explanation": "max() was given lo twice; the upper bound is hi.",
 "patched_code": "def clamp(x, lo, hi):\n    ...\n    return max(lo, min(x, hi))\n"}
```

A product would apply `patched_code` directly, so the harness scores what the product would
get. Malformed JSON, a patch that only satisfies the test it was shown, and a patch that reaches
for `eval` are all failures, each counted in its own metric.

**Additional features** extend the same paired gate:
- `judge`: an LLM patch-correctness judge (`{"verdict": true | false}`) scored against execution labels.
- `review`: a diff code reviewer (`{"passed": true | false, "defect_severity": ..., "line_comments": [...]}`) scored on `review` or `function` suites against execution-derived labels.
- `patch`: a multi-file unified diff coder (`{"explanation": ..., "unified_diff": ...}`) scored on `patch` or `function` suites by pure-Python diff application, non-no-op checks, Python AST parsing, and SAST.

**The question CI answers** on every push and pull request: *does this change make the feature
worse on tasks it was not tuned on?* A "change" is anything that alters the feature's
behaviour: a new prompt version, a new model, new sampling settings. CI does not call a model.
It replays recorded responses ("cassettes"), so it is free, fast and deterministic. A separate,
manually triggered job evaluates a live endpoint when repository secrets configure one.

## 2. Architecture

| Module | Responsibility |
|---|---|
| `records` | The shared data records: `Check`, `Task`, `Suite`, `Item`, `ChatRequest`, `ChatResponse`, `ItemResult`, `ArmRun` |
| `util` | Canonical JSON, hashing, JSONL files, display paths, logging to stderr |
| `tasks` | Loading a suite, structure checks, the dev / held-out split, twin detection |
| `prompts` | Versioned prompt templates that never change once published; one request per item |
| `clients` | `OpenAICompatClient` (urllib), `ReplayClient`, `RecordingClient` |
| `cassettes` | Recorded responses on disk; manifests; fingerprints; the writer |
| `sandbox` | Model-written code in a temporary directory, a child process and a deadline |
| `verdicts` | Think-block validity, strict JSON schema validity, the strict execution verdict |
| `sast` | An AST-based scan for unsafe or harness-gaming constructs |
| `diff_apply` | Pure-Python unified diff parser and applier with safety checks and +/-50 line offset fuzz |
| `features` | `bugfix`, `judge`, `review`, and `patch`: their items and how one reply is scored |
| `validation` | The checks a suite must pass before anything is scored on it |
| `runner` | Evaluate one arm, record one arm, run the gate |
| `metrics` | Rates with Wilson intervals, classifier metrics, the paired comparison |
| `gate` | The hard gates, the non-inferiority rule, the JSON report |
| `report` | The report's JSON text and the markdown summary for CI |
| `ledger` | A hash-chained record of evaluations, promotions and rollbacks |
| `cli` | The `llm-eval` command |

The harness needs only the Python standard library (3.10 or later); `pytest` is needed for the
tests alone. `tests/test_contract.py` fails if a module imports anything else.

### Data flow

```text
tasks/demo/{suite.json,tasks.jsonl}
        |  tasks.load_suite                          validation.validate_suite (CI: tasks validate)
        v
      Suite --- tasks.split_of: sha256 bucket of the task's group or id --> dev | heldout
        |
        |  features.<feature>.items(suite, split)
        v
      Item  (bugfix:<task>, or judge:<task>:reference / judge:<task>:wrong)
        |
        |  prompts.build_request(template, item, model, settings)
        v
  ChatRequest  (its sha256 identifies it in a cassette)
        |
        +--> OpenAICompatClient --> live endpoint            (llm-eval record, the live CI job)
        |      wrapped by RecordingClient --> cassettes/<arm>/<feature>.jsonl
        |
        +--> ReplayClient <-- cassettes/<arm>/               (llm-eval run / gate, every CI run)
        |
        v
  ChatResponse
        |  features.<feature>.score
        |     verdicts.check_think        thinking present, closed, within the word limit
        |     verdicts.check_schema       exactly the feature's JSON object
        |     sast.added_findings         unsafe constructs the reply added (bugfix)
        |     verdicts.exec_verdict       shown + hidden tests in the sandbox (bugfix)
        v
  ItemResult  (one per item; written with --items-out)
        |  metrics.arm_metrics
        v
  ArmRun (candidate)          ArmRun (baseline)
        \                      /
         metrics.compare: paired on item id -> gains, losses, delta, bootstrap CI, McNemar p
                     |
                     |  gate.decide: the hard gates of rule noninferior/1
                     v
               report (JSON, the only output on stdout) --> CI: jq -e '.promote == true'
                     |                                   --> llm-eval summary >> $GITHUB_STEP_SUMMARY
                     +--> ledger (evaluation / promote / rollback rows)
```

## 3. Data model

The records in `records.py` are the only types that cross module boundaries:

- `Check`: one test as data: a call expression (`"clamp(5, 0, 3)"`) and either the Python
  literal it must return (`"3"`) or the exception class it must raise, optionally with a
  numeric tolerance.
- `Task`: one bug-fix task (section 4). `Suite`: a loaded suite with its sha256.
- `Item`: one unit a feature is evaluated on. Its `cluster` is the bootstrap's resampling unit
  (section 9).
- `ChatRequest` / `ChatResponse`: what goes to a model and what comes back.
  `ChatRequest.sha256()` is the sha256 of the canonical JSON of exactly the body the live client
  POSTs. The item id is not part of the body: it is the cassette lookup key.
- `ItemResult`: the scored outcome of one item for one arm, made of primitive fields only, so
  that `metrics` and `gate` depend on nothing but `records`.
- `ArmRun`: one arm (a cassette set) evaluated on one feature and split.

## 4. The task suite

### Format

A suite is a directory with `suite.json` (`kind` defaults to `"function"`, and may be `"function"`,
`"review"`, or `"patch"`) and either `tasks.jsonl` or `tasks.jsonl.gz` (having both is refused):

```json
{"schema": "llm_eval_harness.suite/1", "name": "demo", "version": 1,
 "heldout_pct": 50, "synthetic": true, "description": "..."}
```

There are four suites under `tasks/`:
- `tasks/demo`: 34 synthetic `"function"` tasks (13 dev, 21 heldout; `heldout_pct: 50`).
- `tasks/humanevalfix`: 115 HumanEvalFix `"function"` tasks (41 dev, 74 heldout; `heldout_pct: 60`).
- `tasks/swe_review_bench`: 400 `"review"` tasks (`tasks.jsonl.gz`; 217 dev, 183 heldout; `heldout_pct: 50`).
- `tasks/swe_coder_effort`: 297 `"patch"` tasks (`tasks.jsonl.gz`; 128 dev, 169 heldout; `heldout_pct: 50`).

Each line of a `"function"` `tasks.jsonl` is one task:

| Field | Meaning |
|---|---|
| `id` | Unique and stable, `[a-z0-9][a-z0-9-]*` |
| `entry_point` | The function the tests call |
| `buggy_code` | The function, with its docstring and one bug: what the model is shown |
| `shown_tests` | The failing test the model is shown (the demo shows exactly one) |
| `hidden_tests` | At least two tests the model never sees |
| `reference_code` | A correct fix: proves the task solvable; the judge's positive item |
| `wrong_code` | A plausible fix that passes the shown test and fails a hidden one: the judge's negative item, and the shape of an overfitted patch |
| `group` | Optional split key shared by tasks that must stay on one side (twins) |
| `tags` | Bug classes (`off-by-one`, `boundary`, ...), reported per tag |
| `synthetic` | `true` for tasks written for this repository |

A test is `{"call": "clamp(5, 0, 3)", "expected": "3"}`, or
`{"call": "parse_port('0')", "raises": "ValueError"}`, with an optional `"approx": 1e-9` for
floats. `expected` is the source of a Python literal, so tuples, sets and `None` can be written.
Unknown keys are refused: a misspelt `hiden_tests` must not load as a task without hidden tests.

For `"review"` suites, each task holds `id`, `user_prompt`, boolean `label`, and optional `patch_files`,
`group`, `tags`, `synthetic`. For `"patch"` suites, each task holds `id`, `user_prompt`, `files`
(`{relpath: content}`), `gold_files` (`{relpath: content | null}`), and optional `group`, `tags`,
`synthetic`.

### Shown and hidden tests

The model sees the buggy code and the shown test, rendered as one line of Python
(`assert clamp(5, 0, 3) == 3`). It never sees the hidden tests, the reference code or the wrong
code. "Resolved" requires every shown **and** hidden test to pass, so a patch that
special-cases the shown input (`if x == 5: return 3`) is caught, and counted separately as
*overfit* (section 9).

### The dev / held-out split

```text
key    = normalize_key(task.group or task.id)          # stripped, lower-cased, spaces collapsed
bucket = int(sha256(key.encode("utf-8")).hexdigest()[:8], 16) % 100
split  = "heldout" if bucket < suite.heldout_pct else "dev"
```

- **dev** is for iteration: look at it as often as you like while changing a prompt.
- **heldout** is the gate's. Every candidate judged on it is a choice made with it, so it wears
  out; the ledger counts how often it has been used (section 11).

The key is the task's id (or group), **not its text**. Hashing the text would move a task to the
other side whenever someone fixes a typo in it, and a held-out task that drifts into dev has
leaked. A hash rather than a random draw needs no stored state: every machine computes the same
split, and adding a task never moves another.

Twins (the same problem reworded, or with a renamed function) must share a `group`, or one
twin's answer teaches the other's. `tasks.find_twins` compares every dev task with every
held-out task in another group, on their code with identifiers and literals normalised, and
`tasks validate` warns about near-identical pairs and fails on identical ones.

The demo suite has `heldout_pct: 50`: with about thirty tasks, a 15% held-out share would hold
four or five tasks and could detect nothing. The hash put 21 of its 34 tasks in the held-out
split and 13 in dev; `heldout_pct` sets the expected share, not an exact count. Larger suites
should hold out less.

### Validation

`llm-eval tasks validate` (run by CI on every Python version across all four suites) fails a suite when:

- the structure is wrong: ids, entry points defined in all three code fields, at least one shown
  and two hidden tests, every call parses as an expression, every `expected` is a literal (for `"function"` suites), or non-empty `user_prompt`, boolean `label` (`"review"`), and non-empty clean relative `files`/`gold_files` differing from `files` (`"patch"`);
- run in the sandbox (`"function"` suites only), the buggy code passes its shown tests (then the shown test demonstrates no
  bug), the reference fails any test or gives different outcomes on a second run (flaky), or the
  wrong code fails a shown test or passes every hidden one;
- a block-class SAST rule fires on the buggy, reference or wrong code (`"function"`), or on modified Python files in `gold_files` (`"patch"`);
- a rendered prompt contains a hidden test's call text, a bug-fix prompt contains the reference
  or the wrong code, a judge prompt shows any code but the patch under review, a review prompt injects a gold label literal, or a patch prompt leaks changed `gold_files` text;
- two tasks on opposite sides of the split have identical normalised code (`"function"`) or normalised `user_prompt` (`"review"`, `"patch"`).

## 5. The features

### Prompts

A template id is `<feature>/v<n>`. A published template never changes, because cassettes hold
answers to requests made with it, and a recording is only replayed for the exact request it
answered (section 6). To change a prompt, add the next version and point `prompts.CURRENT` at
it. `tests/test_prompts.py` pins the sha256 of every published template and the exact output of
`render_check`, so an accidental edit fails the unit tests before it reaches CI.

| Template | What it asks |
|---|---|
| `bugfix/v1` | Fix the function so the failing test passes; answer with the JSON object |
| `bugfix/v2` | Adds: fix it for every input the docstring describes, not only the shown test; do not special-case test inputs; no `eval`, `exec`, subprocesses, files or network |
| `judge/v1` | Does the patched function correctly fix the bug? Answer `{"verdict": ...}` |
| `judge/v2` | Adds: check inputs beyond the shown test; look for special-cased test inputs |
| `review/v1` | Review unified diff + context; answer `{"passed", "defect_severity", "line_comments"}` |
| `patch/v1` | Generate repository-relative unified diff; answer `{"explanation", "unified_diff"}` |

### bugfix

One item per task, `bugfix:<task id>`. A reply is scored in this order:

1. **Think**: `verdicts.check_think` (section 7).
2. **Schema**: `verdicts.check_schema` on the text after the thinking.
3. **SAST**: `sast.added_findings(task.buggy_code, patched_code)`. The code is taken from the
   strictly valid object, or else from a leniently parsed one: a reply is accountable for the
   code it wrote even when its JSON was malformed.
4. **Execution**: only when the schema is valid and no block-class finding was added,
   `verdicts.exec_verdict` runs the shown tests and then the hidden tests in the sandbox.
5. **Resolved** = schema valid, no block-class finding added, and every test passed.

### judge

Two items per task: `judge:<task>:reference` shows the reference code (label `true`), and
`judge:<task>:wrong` shows the wrong code (label `false`). The prediction is the `verdict` of a
schema-valid reply; a reply without one is wrong for either label. No code runs.

Because every task contributes one item of each class, a judge that decides from the task alone
(its wording, its function name) scores exactly 0.5 balanced accuracy. Only reading the patch
scores higher.

### review

One item per task on `"review"` suites (`review:<task id>`), or two items per task on `"function"`
suites (`review:<task id>:reference` / `:wrong`, with unified diff and numbered lines). Scoring
checks optional thinking (`REVIEW_THINK`: 1 to 1,200 words) and strict `REVIEW_SCHEMA`
(`passed: bool`, `defect_severity` in `{none, low, medium, high, critical}`, `line_comments: [dict, ...]`).
Predicted = `passed` when valid, else `None`. No code runs and no SAST applies.

### patch

One item per task (`patch:<task id>`) on `"patch"` or `"function"` suites. Scoring checks optional
thinking (`PATCH_THINK`: 1 to 1,200 words), strict `PATCH_SCHEMA` (`explanation`, `unified_diff`),
pure-Python diff application via `diff_apply.apply_unified_diff` (exact context first, then
+/-50-line offset fuzz and trailing-whitespace fallback; path traversal, `.git`, symlinks, and
binary patches refused), non-no-op edits (`changed_files` non-empty), `ast.parse` on modified or
added Python files (when the original parsed), and zero added block-class SAST findings (a lenient
parse is also SAST-scanned). `gold_match` compares updated files to `gold_files` (with AST equality
fallback for `.py`/`.pyi`). `correct` (`usable`) requires strict schema, clean non-no-op apply,
valid Python syntax, and zero added block SAST findings.

## 6. Model clients and cassettes

**The live client** (`OpenAICompatClient`) POSTs `ChatRequest.payload()` to
`<base_url>/chat/completions` with `urllib`. It retries connection errors, timeouts and HTTP
408, 409, 425, 429, 500, 502, 503 and 504 with exponential backoff (honouring `Retry-After`, up
to 60 seconds), and accepts only
`http` and `https` URLs. It reads separately returned thinking (`reasoning_content` or
`reasoning`) as well as a `<think>` block inside the content. The API key comes from an
environment variable and appears in no message, log line or `repr`.

**Failures are not answers.** A request that still fails after its retries raises `ClientError`.
It is never turned into an empty reply: an empty reply would be scored as wrong, and a flaky
endpoint would look like a regression of the model. The evaluation stops instead (exit 1), and
the recorder leaves the item for the next resume.

**Cassettes.** A cassette set is one directory per arm:

```text
cassettes/<arm>/manifest.json      arm name, synthetic flag, and per feature: prompt id, model,
                                   settings, the suite's name and sha256, the item count
cassettes/<arm>/bugfix.jsonl       {"item_id", "request_sha256", "response": {...}} per line
cassettes/<arm>/judge.jsonl        (or <feature>.jsonl.gz for deterministic gzip tapes)
```

Tapes may be plain `<feature>.jsonl` or gzipped `<feature>.jsonl.gz` (`CassetteWriter(..., compress=True)`
writes deterministic gzip with `mtime=0`, and `fingerprint()` hashes `manifest.json` and every `*.jsonl`
or `*.jsonl.gz` directly in the directory; `.jsonl` and `.jsonl.gz` cannot be mixed for the same feature).

**Replay** (`ReplayClient`) rebuilds each item's request from the manifest's prompt, model and
settings, looks the recording up by item id, and compares the recorded `request_sha256` with the
rebuilt request's. A missing recording or a mismatch refuses the evaluation (exit 2): the task,
the template or the settings changed since recording, and the answer may not fit the question.
A recording made on another suite version whose prompts did not change (only the hidden tests
did, for instance) is replayed and re-scored against today's tests, with a note in the report.

**Recording** (`llm-eval record`) wraps the live client in a `RecordingClient` that appends each
answer as it arrives and writes the manifest last. Re-running the same command resumes: items
already recorded for the same request are not asked again.

**Synthetic cassettes.** The demo cassettes (`cassettes/baseline`, `cassettes/candidate`,
`cassettes/candidate-regressed`) were written by `scripts/make_demo_cassettes.py`, not by a
model. Their manifests say `"synthetic": true`, every report repeats it per arm, and the CI
summary opens with a banner saying so. The script builds every reply from the task suite by
documented rules over each task's position in its split: the reference code stands for a fix,
the wrong code for an overfitted patch, and hand-written `eval()` variants, prose around the
JSON, a missing key and replies truncated at `max_tokens` stand for the failures. It is
deterministic. `--check` regenerates the sets in a temporary directory and compares their bytes
with the committed ones, and `tests/test_demo.py` runs that check. Recording a real model
replaces them (section 12).

## 7. Verdicts

### Think validity

Thinking is either returned separately by the server or written at the start of the content as
`<think>...</think>`. A `</think>` without an opening tag counts as a block whose opening the
chat template supplied. A reply's thinking is

- **present** when there is any;
- **closed** when it is present and an answer follows it (a reply cut off while thinking is not
  closed and has no answer);
- **in range** when its word count (whitespace-separated) is within the feature's limits (1 to
  1,200 words for both demo features).

A reply's thinking is **valid** when it is present, closed, in range and without problems (a
second block, thinking in both places), or when it is absent and the feature does not require
it. Both demo features make thinking optional.

### Schema validity (strict)

- A reply that ended with `finish_reason == "length"` is invalid, whatever it holds.
- The text after the thinking, stripped, must be exactly one JSON object: no prose, no code
  fence, no trailing text, no duplicate keys, no `NaN`.
- Every required key must be present, no other key, each of its type (`type(v) is bool` for the
  verdict: `1` is not `true`), and `explanation` and `patched_code` non-blank.

A lenient parse (strip one code fence, or take the outermost braces) is recorded as a diagnostic
(`lenient_valid`). It never makes a reply valid. A model that is only leniently valid needs a
format fix, and the report shows how often that is the case.

### The strict execution verdict (`values/1`)

The child process running the model's code receives only the call expressions. It returns, per
call, the `repr` of the value or the raised exception's class names. The parent decides:

| Outcome | When |
|---|---|
| passed | The record exists, is the only one for its index, and its value equals the expected literal (`==`, or `math.isclose` with `approx`), or the expected exception class is among the raised class's MRO names |
| failed | Any other value, a value where an exception was expected, a `repr` that is not a literal |
| error | An exception where a value was expected |
| missing | No record: the process died, called `os._exit`, or never got there; or a duplicate index |
| timeout | No record, and the deadline hit |

The expected values never enter the child, so code under test cannot pass by writing a fake
report, exiting early with status 0, or skipping a test. It can only pass by returning the
right values. **Exit status 0 is never a pass by itself**, and zero tests is never a pass. If the
sandbox itself cannot start, that is an infrastructure fault: no verdict, the evaluation stops,
and the item is never scored as failed.

### SAST

`sast.scan` parses the code and resolves imports, aliases, simple assignments and
`getattr(module, "name")`, so `from os import system as run; run(cmd)` is found. When the code
does not parse, the block-class rules run as regexes on the raw text. A finding is the patch's
when the buggy code did not already have it (`added_findings`, a multiset difference on rule and
snippet).

| Rule | Severity | CWE | Matches |
|---|---|---|---|
| `eval-exec` | block | CWE-95 | calls of the builtin `eval` / `exec` |
| `os-command` | block | CWE-78 | `os.system`, `os.popen`, `os.exec*`, `os.spawn*`, `pty.spawn` |
| `shell-true` | block | CWE-78 | any call with `shell=True` |
| `unsafe-deserialization` | block | CWE-502 | `pickle` / `marshal` / `dill` `.load(s)`, `shelve.open`, unsafe `yaml.load` |
| `process-control` | block | | `os._exit`, `os.kill`, `atexit.register`, `signal.signal`, `signal.alarm` |
| `interpreter-tamper` | block | | `sys.settrace`, `sys.setprofile`, writes into `sys.modules` or `builtins` |
| `subprocess` | flag | CWE-78 | any other `subprocess.*` call |
| `dynamic-import` | flag | CWE-470 | `__import__`, `importlib.import_module` |
| `file-write` | flag | CWE-73 | `open()` for writing, `Path.write_*`, `os.remove`, `shutil.rmtree`, ... |
| `network` | flag | | imports of `socket`, `urllib.request`, `http.client`, `requests`, ... |
| `introspection` | flag | | `sys._getframe`, `inspect.stack`, `gc.get_objects`, ... |
| `exit-call` | flag | | `sys.exit`, `exit`, `quit` |

A **block** finding makes the item unresolved, its code is not run, and for the gate it is a
*new* violation when the baseline's reply to the same item has no finding of that rule. A
**flag** finding is counted in the report and decides nothing. The block class is admitted only
if it never fires on a correct fix: validation checks every reference solution against it.

## 8. The sandbox and its limits

Every piece of model-written code runs through `sandbox.run_calls`:

- in a fresh temporary directory (working directory, `HOME` and `TMPDIR`), removed afterwards;
- in a child process started as `python -I -S`, with an environment built from nothing (no
  inherited variable, so no API key reaches it) and without site-packages;
- under a deadline (10 s per item by default), after which the whole process group is killed;
- with best-effort POSIX limits on CPU time, address space and file size, and no core files.

**This is not a security boundary.** The child runs as the same user, can read every file that
user can read, and can open network connections. The sandbox keeps a careless or overfitted
patch from touching the working tree, the secrets in the environment or the verdict. It does not
stop a hostile one. In CI the only code executed is the code in the committed cassettes, which
is reviewed like any other file in a pull request. Score untrusted models inside a container or
virtual machine without network access (for example `docker run --network none`). The sandbox is
POSIX-only.

## 9. Metrics

All rates are over the items of one arm on one split. `n` is the number of items.

### Per arm: bugfix

| Metric | Definition |
|---|---|
| `schema_validity` | schema-valid replies / n |
| `think_validity` | replies with valid thinking / n |
| `resolved_rate` (primary) | resolved items / n, with a Wilson 95% interval |
| `shown_pass_rate` | items whose shown tests passed / n (code that was not run counts as not passed) |
| `hidden_pass_rate` | items whose hidden tests passed / n |
| `overfit_rate` | items with shown passed and hidden failed / items with shown passed |
| `sast` | block and flag findings per rule |
| cost | completion tokens (total, mean), median latency, finish reasons |

### Per arm: judge

Scored on paired tasks only, those with both a reference and a wrong item. With TP, FN for the
reference items and TN, FP for the wrong items, and a reply without a valid verdict counted as
wrong in its class:

```text
TPR = TP / positives        TNR = TN / negatives
balanced_accuracy = (TPR + TNR) / 2                        (primary)
precision = TP / (TP + FP)  F1 = 2 * precision * TPR / (precision + TPR)
approve_rate = (TP + FP) / n
```

TPR and TNR carry Wilson intervals, and balanced accuracy a bootstrap interval over tasks.
`vs_chance` says whether that interval lies above 0.5.

### Per arm: review

Uses the same classifier metrics (`classifier_metrics`, `balanced_accuracy` primary, `tpr`, `tnr`,
`approve_rate`, `confusion`, `vs_chance`) over all labeled items, with a cluster bootstrap interval
(`"<seed>/review/arm"`).

### Per arm: patch

| Metric | Definition |
|---|---|
| `usable_rate` (primary) | usable items (`correct`) / n, with `usable` count and Wilson 95% interval |
| `applied_rate`, `noop_rate`, `gold_match_rate` | diffs applied / n, no-op diffs / n, diffs matching `gold_files` / n |
| `sast`, `by_tag`, validity, cost | block/flag findings per rule, usable items per tag, schema/think/lenient, tokens and latency |

### The Wilson interval

For k successes in n trials and z = 1.96:

```text
centre = (k + z^2 / 2) / (n + z^2)
half   = z * sqrt(k * (n - k) / n + z^2 / 4) / (n + z^2)
```

Unlike the normal approximation, it stays inside [0, 1] and does not collapse to a point at
0 / n or n / n, which matters on small suites.

### The paired comparison

The arms are joined on item id. Per item, `x = correct(candidate) - correct(baseline)`:

| Quantity | Definition |
|---|---|
| `gains` | items only the candidate got right (x = +1) |
| `losses` | items only the baseline got right (x = -1) |
| `delta` | primary(candidate) - primary(baseline) on the joined items; for `resolved_rate` this is (gains - losses) / n |
| `ci95` | paired bootstrap percentile interval of delta: 2,000 draws of the clusters with replacement; both arms' rows of a drawn cluster go in together |
| `se` | the standard deviation of the bootstrap deltas |
| `mde80` | (1.96 + 0.84) x se: the true change a two-sided 5% test would detect four times in five |
| `mcnemar_p` | exact two-sided McNemar p of (losses, gains), stored unrounded |
| `min_significant_losses` | the fewest losses, with no gains, that McNemar calls significant at alpha: 6 at 0.05 |

```text
mcnemar_p(b, c) = min(1, 2 * sum_{k=0}^{min(b,c)} C(b + c, k) / 2^(b + c)),   1 when b + c = 0
```

**Why paired.** Both arms answer the same items, so most of the variation between items (some
tasks are just hard) cancels out. Only the discordant items carry information about the
difference; McNemar's test uses exactly those.

**Clusters.** The bootstrap resamples clusters, not items: a task's two judge items, and twins
that share a group, are drawn together, because they are not independent.

**Determinism.** The generator is seeded with `"<seed>/<feature>/<split>"`, so the interval is
the same on every machine and every run, and replaying the same cassettes produces a
byte-identical report.

## 10. The gate

Rule `noninferior/1`. Every gate that applies is enforced, and `promote` is true when all of
them pass. `regression_test` applies only with a baseline, `gain_test` only with
`--require-gain`.

| Gate | Passes when |
|---|---|
| `schema_validity` | candidate schema validity >= 0.95 |
| `think_validity` | candidate think validity >= 0.90 |
| `new_sast_violations` | no (item, rule) pair with a block finding in the candidate's reply and none of that rule in the baseline's reply to the same item (`bugfix` and `patch`; reported as not applicable for `judge` and `review`) |
| `primary` | `delta >= -margin` with `--tie pass` (the default), `delta > -margin` with `--tie fail` |
| `regression_test` | not (losses > gains and McNemar p < alpha) |
| `gain_test` | only with `--require-gain`: gains > losses and McNemar p < alpha |

**The margin and the tie rule.** The margin defaults to 0: by default the candidate may not be
worse on the point estimate, and an exact tie passes. That is the right default for a regression
test: a refactor that changes nothing should not be blocked. `--tie fail` with margin 0 demands
a strict improvement. `--tie fail --require-gain` demands one that McNemar's test calls
significant, which is what replacing a baseline should require when the change exists to help.
A positive margin (say 0.02) absorbs the noise of a live, non-deterministic endpoint. Float
comparisons use a tolerance of 1e-9, so rounding never decides a tie.

**Why the point estimate and not the interval.** On a suite of a few dozen items the 95%
interval of delta is wider than any margin worth setting. "The interval's lower end is above
-margin" would fail nearly every candidate; "the regression is significant" would pass nearly
every one. So the gate decides on the point estimate with an explicit margin and tie rule, uses
McNemar as a second guard (it fires when a margin is set and a loss inside it is still
significant), and always reports the interval, `mde80` and `min_significant_losses`, so that a
reader can see how much the decision is worth. (For a plain rate with margin 0 the McNemar
guard cannot fire, since delta >= 0 means losses <= gains. It can for balanced accuracy on
unbalanced classes, where items weigh by class.)

**Without a baseline** (`--baseline none`), the primary gate compares the candidate with an
absolute floor (bugfix 0.5, judge 0.7, review 0.65, patch 0.7, or `--floor`), and the paired tests are skipped. The tie
rule applies there too: `primary >= floor` with `--tie pass`, `primary > floor` with `--tie fail`.

**Refusals.** The gate refuses (exit 2, no report) when the arms belong to different features or
splits, were scored on different suites, or do not cover the same item ids. An unpaired
comparison would compare different questions.

### The report

`llm-eval gate` prints one JSON object on stdout and nothing else:

```text
schema, rule, feature, split
suite        name, sha256, split_sha256, tasks, items, heldout_pct, synthetic
candidate    name, source, fingerprint, model, prompt, prompt_sha256, synthetic,
             recorded_suite_sha256, metrics, notes
baseline     the same, or null
comparison   n, clusters, gains, losses, candidate, baseline, delta, ci95, se, mde80,
             mcnemar_p, min_significant_losses, alpha, resamples, seed
gates        [{gate, value, op, bound, passed, enforced, ...}]
gates_passed, promote
config       the gate parameters
warnings     what limits the decision (synthetic arms, a small suite, a spent held-out split,
             a judge at chance or rubber-stamping, a prompt that is not the current one, cost)
provenance   harness version, SAST rule-set fingerprint, execution rule, current prompt id
```

While the gate works, file descriptor 1 points at stderr, so no library or child process can
write to stdout. The report carries no time stamp, host name or absolute path.

### Exit status

| Status | Meaning |
|---|---|
| 0 | The evaluation ran and the report was printed, whatever the verdict |
| 1 | It could not run or finish (endpoint, sandbox, I/O) |
| 2 | Usage error or refusal (arms not comparable, missing or mismatched recordings) |

CI reads the verdict from the report (`jq -e '.promote == true'`), so "the gate said no" and
"the gate could not run" stay distinguishable.

## 11. The ledger

`ledger/ledger.jsonl` is committed with the code. Each line is a JSON row carrying `prev` (the
sha256 of the previous line) and `id` (a hash of its own content), so an edited, removed or
reordered line shows. `llm-eval ledger verify` checks the chain and also re-fingerprints each
feature's current baseline directory: a baseline whose recorded responses were edited outside
the ledger fails CI. Rows carry no user name, host name or absolute path.

| Row | Written by | Holds |
|---|---|---|
| `header` | the first write | creation time |
| `promote` | `llm-eval ledger promote` | feature, the new baseline's path and fingerprint, the one it replaced, the gate report's sha256 and verdict, a reason |
| `rollback` | `llm-eval ledger rollback` | the promotion undone, the baseline restored, a reason |
| `evaluation` | `llm-eval gate --record-evaluation` | the split's sha256, the candidate's fingerprint, the verdict |

**Current baseline.** A feature's rows form a stack: a promotion pushes, a rollback pops. CI
compares candidates with `--baseline @current`, the directory on top of the stack.

**Promotion** refuses a report that is not of this feature or of the held-out split, whose
`promote` is false (unless `--override`, which needs a reason and is recorded as such), whose
candidate fingerprint differs from the directory being promoted, or whose baseline is not the
current baseline (a stale report, made before another promotion).

**The held-out budget.** Each candidate evaluated on the held-out split is a decision made with
it, and enough of them turn the held-out split into another dev split. Evaluation rows count the
distinct candidates per split. Once that count reaches `gate.HELDOUT_BUDGET` (10), every report
warns that the split is spent. CI reads the ledger and never writes it.

**The committed ledger** holds a header and two promotions marked `"gate": "initial"`: both
features start with `cassettes/baseline` as their baseline.

## 12. CI

`.github/workflows/ci.yml` runs on every push to `main`, every pull request, and on manual
dispatch (inputs: `feature`, `bugfix`, `judge`, `review`, or `patch`; `split`, dev or heldout, default dev). It has
`contents: read` permission only, its checkouts keep no credentials, every script runs under
`bash -eo pipefail`, and a newer run for the same branch or pull request cancels a stale one.
The actions are pinned to their current major versions (`actions/checkout@v7`,
`actions/setup-python@v7`, `actions/upload-artifact@v7`).

1. **tests** (Python 3.10, 3.11 and 3.12): install the package with its test extra,
   `python -m pytest -q`, `llm-eval tasks validate --suite tasks/demo`,
   `llm-eval tasks validate --suite tasks/humanevalfix`,
   `llm-eval tasks validate --suite tasks/swe_review_bench`,
   `llm-eval tasks validate --suite tasks/swe_coder_effort`,
   `python scripts/make_report_tables.py --check`, and `llm-eval ledger verify`.
2. **regression-gate** (after tests; replay only, no secrets):
   - for each feature, `llm-eval gate --feature <f> --split heldout --candidate cassettes/candidate
     --baseline @current --items-out reports/items/candidate` into `reports/<f>-candidate.json`;
   - for the **self-test of the gate**, the same with `--candidate cassettes/candidate-regressed`
     (items in `reports/items/regressed`) into `reports/<f>-regressed.json`. A gate that cannot
     run (exit 1) or refuses (exit 2) fails this step;
   - whether or not that step passed (`if: always()`): `llm-eval summary` of the reports that were
     written, appended to `$GITHUB_STEP_SUMMARY`, and `reports/` (the reports and the per-item
     files) uploaded as the `gate-reports` artifact;
   - finally the checks, with `jq -e`: every candidate report must say `promote: true`, and every
     regressed report `promote: false` with at least the expected enforced gates failed (bugfix:
     `primary`, `schema_validity` and `new_sast_violations`; judge: `primary` and
     `schema_validity`). Each failure is a `::error::` annotation naming the gates. A change to
     the harness that lets a clearly regressed candidate through fails the build.
3. **live-config** and **live-eval** (manual dispatch only): the first job checks whether the
   repository secrets `LLM_EVAL_BASE_URL`, `LLM_EVAL_API_KEY` and `LLM_EVAL_CANDIDATE_MODEL` are
   set; when they are not, the live job is skipped with a `::notice::`. Otherwise the second job
   records the candidate with the current prompt, and the baseline with the prompt and settings of
   the current baseline (`--prompt-from "$(llm-eval ledger current --feature ...)"`) and
   `LLM_EVAL_BASELINE_MODEL` (or the candidate model, to measure a prompt change alone), on the
   chosen split (dev by default, so that a live experiment does not spend the held-out split).
   It then gates the two recordings, writes the summary, uploads the recordings and the report
   (`if: always()`), and finally requires `promote: true`. Secrets and inputs reach the scripts
   only as environment variables, never pasted into a script's text. It never commits anything.

Only these two live jobs see secrets. `tests/test_workflow.py` reads the workflow without a YAML
library. It checks the points above, parses every `llm-eval` command line in it against the CLI,
and runs the regression-gate job's own scripts with bash and jq on a copy of the repository,
including the self-test's failure paths. It also runs the live job's scripts against a local
fake endpoint.

**Working with a real model.** A pull request that changes the feature includes the evidence.
For a new prompt, say `bugfix/v3` added to `prompts.TEMPLATES`:

```bash
export LLM_EVAL_BASE_URL=http://localhost:8000/v1 LLM_EVAL_API_KEY=...   # never committed
llm-eval record --feature bugfix --model my-model --prompt bugfix/v3 --out cassettes/bugfix-v3
llm-eval gate --feature bugfix --candidate cassettes/bugfix-v3 --baseline @current > report.json
```

The pull request then commits `cassettes/bugfix-v3`, CI replays it against the current baseline,
and after merging, `llm-eval ledger promote --feature bugfix --to cassettes/bugfix-v3 --report
report.json --reason "..."` makes it the baseline in a reviewed commit.

## 13. Lessons this design carries

This harness was extracted from a larger project that trains a multi-agent coding team, where
these choices were learnt while building the gate that decides whether a newly trained model
replaces the one in use. This repository re-implements them, small and standalone; it shares no
code or data with that project.

- **A tie is a decision, so the rule names it.** A gate of the form "candidate >= baseline"
  silently promotes changes that do nothing. Here a regression gate passes ties on purpose, and
  a baseline replacement can demand a significant gain.
- **Every look at the held-out split spends it.** Iterate on dev. The ledger counts held-out
  evaluations, and the report warns when the split has been used to choose between too many
  candidates.
- **Split on a stable key, not on text,** and keep twins together, or editing a task leaks it.
- **Exit status 0 is not a pass.** Compare values in a process the model's code does not
  control, count a missing result as a failure, and never let an infrastructure fault become a
  verdict either way.
- **LLM judges can sit at chance while sounding sure.** Measure them against execution labels,
  on tasks with both a correct and a wrong patch, and accept patches on tests plus a static scan
  rather than on a judge's word.
- **Only the report goes to stdout,** so `jq` in CI never parses a log line.
- **Small suites detect only large regressions.** The report says how large, every time.

## 14. Limitations and non-goals

- The demo tasks are small pure functions. Real bug fixing spans files, repositories and tests
  that need an environment; the harness's structure carries over, its sandbox does not.
- The sandbox is not a security boundary (section 8).
- The demo suite is public, so its held-out split is held out from prompt tuning, not from the
  internet: a model trained on this repository has seen it. A real deployment keeps its held-out
  file private; `--suite` can point anywhere.
- Synthetic cassettes demonstrate the mechanics, not the quality of any model.
- A static scan narrows what a patch can do unseen; it does not close it (`sast.py`, LIMITS).

## 15. Adding a feature

1. Define its JSON schema (`verdicts.JsonSchema`) and thinking policy.
2. Implement the `features.Feature` protocol: `items(suite, split)` and `score(item, response)`.
3. Add its templates to `prompts.TEMPLATES`, its entry to `prompts.CURRENT`, and the placeholders
   its templates may use to `prompts._FIELDS` (a template that uses another feature's field is
   refused).
4. Give it a primary metric in `metrics.PRIMARY` and `metrics.arm_metrics`, and a floor in
   `gate.DEFAULT_FLOORS`.
5. Register it in `features.FEATURES` and `records.FEATURES`; record cassettes; add it to the CI
   loop.

The runner, the gate, the report, the ledger and the workflow need no other change.

## 16. Repository layout

```text
pyproject.toml                  package llm_eval_harness, console script llm-eval
src/llm_eval_harness/           the harness (standard library only)
tests/                          unit tests; test_demo.py (the demo end to end); test_workflow.py (ci.yml)
tasks/demo/                     the synthetic demo suite: suite.json, tasks.jsonl
tasks/humanevalfix/             115 Python repair tasks from HumanEvalFix
tasks/swe_review_bench/         400 PR review tasks (kind: review, tasks.jsonl.gz)
tasks/swe_coder_effort/         297 multi-file patch tasks (kind: patch, tasks.jsonl.gz)
cassettes/baseline/             synthetic recordings: the current baseline (prompts v1)
cassettes/candidate/            synthetic recordings: a candidate that should promote (prompts v2)
cassettes/candidate-regressed/  synthetic recordings: a candidate the gate must reject
ledger/ledger.jsonl             the committed ledger
scripts/make_demo_cassettes.py  how the synthetic cassettes were made
.github/workflows/ci.yml        tests, the regression gate and its self-test, the live job
docs/DESIGN.md                  this document
```

## Demo results

These come from `llm-eval gate` on the committed cassettes, against the ledger's current baseline
(`cassettes/baseline`), with the default gate configuration. Every arm is **synthetic**: the
numbers show how the harness decides, not how good any model is. `tests/test_demo.py` pins them.
Intervals are the 95% paired bootstrap of the delta (2,000 cluster draws, seed 0). The held-out
split has 21 tasks in 20 clusters (`digit-sum` and `sum-of-digits` are twins and share a group).
The dev split has 13 tasks.

### Held-out split

bugfix (21 items, primary `resolved_rate`):

| Arm | Resolved | Schema | Think | Overfit rate | Block SAST | Delta [95% CI] | Gains / losses | McNemar p | Verdict | Failed gates |
|---|---|---|---|---|---|---|---|---|---|---|
| baseline (demo-model-a, bugfix/v1) | 14 = 0.667 | 1.000 | 1.000 | 0.333 | none | | | | current | |
| candidate (demo-model-a, bugfix/v2) | 17 = 0.810 | 1.000 | 1.000 | 0.190 | none | +0.143 [-0.045, +0.350] | 4 / 1 | 0.375 | PROMOTE | none |
| candidate-regressed (demo-model-small, bugfix/v2) | 10 = 0.476 | 0.762 | 0.952 | 0.333 | eval-exec in evaluate-sum | -0.190 [-0.391, +0.000] | 0 / 4 | 0.125 | REJECT | primary, schema_validity, new_sast_violations |

The regressed candidate has 5 replies that are not schema-valid. Two are leniently valid (prose
around the JSON), two lack the `explanation` key, and one is truncated at `max_tokens` inside an
unclosed `<think>`. With mde80 0.28 and `min_significant_losses` 6, its 4 losses are not
significant. It is rejected on the point estimate, the format and the new SAST violation; the
`regression_test` gate passes.

judge (42 items, primary `balanced_accuracy`):

| Arm | Balanced accuracy [95% CI] | TPR | TNR | Approve rate | Schema | Delta [95% CI] | Gains / losses | McNemar p | Verdict | Failed gates |
|---|---|---|---|---|---|---|---|---|---|---|
| baseline (demo-model-a, judge/v1) | 0.571 [0.500, 0.650] | 1.000 | 0.143 | 0.929 | 1.000 | | | | current | |
| candidate (demo-model-a, judge/v2) | 0.881 [0.775, 0.955] | 0.952 | 0.810 | 0.571 | 1.000 | +0.310 [+0.175, +0.429] | 15 / 2 | 0.0023 | PROMOTE | none |
| candidate-regressed (demo-model-small, judge/v2) | 0.429 [0.250, 0.609] | 0.571 | 0.286 | 0.595 | 0.905 | -0.143 [-0.350, +0.048] | 6 / 12 | 0.238 | REJECT | primary, schema_validity |

The warnings in the reports:

- every report: the arms are synthetic, and the suite is small (at least 6 losses with no gains
  are needed for significance);
- both judge reports: the baseline judge is a rubber stamp (it approves 0.929) and cannot be told
  apart from chance;
- the regressed judge's report: the candidate judge cannot be told apart from chance either;
- cost warnings: the judge candidate uses 5.0 times the baseline's mean completion tokens (it
  thinks; the rubber stamp does not), and the regressed arms use 2.3 times (bugfix) and 5.7
  times (judge), mostly in their truncated replies. The bugfix candidate uses 1.33 times, under
  the warning's 1.5.

With `--tie fail --require-gain` (a strict, significant improvement), the bugfix candidate is
rejected on `gain_test` (p 0.375), and the judge candidate still promotes.

### Dev split

| Feature | Baseline | Candidate | Regressed | Candidate: delta, gains / losses | Regressed: delta, gains / losses, failed gates |
|---|---|---|---|---|---|
| bugfix (13 items) | 9 = 0.692 | 10 = 0.769 (PROMOTE) | 4 = 0.308 (REJECT) | +0.077, 2 / 1 | -0.385, 0 / 5 (p 0.0625); primary, schema_validity (0.769), new_sast_violations (3 `eval()` patches) |
| judge (26 items) | 0.577 | 0.846 (PROMOTE) | 0.423 (REJECT) | +0.269, 8 / 1 (p 0.039) | -0.154, 4 / 8; primary, schema_validity (0.923) |

The verdicts are the same on both splits, by construction: the generator's rules are
written over each task's position within its own split.
