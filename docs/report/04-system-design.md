# III. System Design

The harness is a Python package (`src/llm_eval_harness/`, standard library only, Python 3.10 or later) with a command-line interface, `llm-eval`. Its full specification is `docs/DESIGN.md`; the module docstrings state the same contracts in more detail. This section summarises the design decisions on which the experiments depend.

## A. Architecture and data flow

An evaluation proceeds through the following stages, each owned by one module:

1. **Suite.** `tasks.load_suite` reads a suite directory (`suite.json`, `tasks.jsonl`) and `tasks.split_of` assigns every task to the development (dev) or held-out split. `validation.validate_suite` checks the suite before anything is scored on it.
2. **Items.** A feature (`features.py`) turns the tasks of one split into items: `bugfix:<task>` for the bug-fix feature, and `judge:<task>:reference` and `judge:<task>:wrong` for the judge.
3. **Requests.** `prompts.build_request` renders a versioned template for an item into a chat request. Its SHA-256, computed over the canonical JSON of exactly the body that the live client sends, identifies the request.
4. **Replies.** A live OpenAI-compatible client (`clients.OpenAICompatClient`, built on `urllib`) answers requests; a recording client stores each reply in a cassette; a replay client (`clients.ReplayClient`) answers from the cassette and refuses a recording whose request hash differs from the rebuilt request.
5. **Scoring.** The feature scores each reply (Section III-C) into an `ItemResult`.
6. **Metrics and comparison.** `metrics.arm_metrics` aggregates one arm; `metrics.compare` joins two arms on item identifiers.
7. **Decision.** `gate.decide` applies the gate rule (Section III-G) and produces a JSON report, the only output written to standard output.
8. **Record keeping.** `ledger.py` appends evaluation, promotion and rollback rows to a hash-chained ledger.

Only the shared records in `records.py` cross module boundaries, and `metrics` and `gate` depend on nothing but those records. `tests/test_contract.py` fails if a module imports anything outside the standard library.

## B. The features under test

**Bug fix.** The model receives a buggy Python function with its docstring and one test that fails on it, rendered as a single assertion. It must answer with an optional thinking block followed by exactly one JSON object with the keys `explanation` and `patched_code`. A product would apply `patched_code` directly, so the harness scores exactly that. The model never sees the task's hidden tests, its reference fix or its wrong fix. Templates are versioned (`bugfix/v1`, `bugfix/v2`) and never change once published; their SHA-256 values are pinned by `tests/test_prompts.py`.

**Judge.** The model receives a task and a proposed patch and answers `{"verdict": true}` or `{"verdict": false}`. Each task yields two items: its reference fix (label true) and a plausible wrong fix that passes the shown test and fails at least one hidden test (label false). Labels therefore come from execution, never from another model. Because every task contributes one item of each class, a judge that decides from the task alone, without reading the patch, scores exactly 0.5 balanced accuracy.

## C. Scoring a reply

A bug-fix reply is checked in a fixed order, and each check is recorded in the item's result:

1. **Thinking.** Thinking may arrive in a separate field of the server's response or as a `<think>` block at the start of the content. It is valid when it is closed (an answer follows it), lies within the feature's word limits (1 to 1,200 words for both features), and appears in only one place; it is optional for both features.
2. **Schema.** A reply that ended at the token limit (`finish_reason` "length") is invalid. Otherwise the text after the thinking must be exactly one JSON object with exactly the required keys, of the required types, and non-blank strings: no surrounding prose, code fence, duplicate key or trailing text. A lenient parse that strips a fence or extracts the outermost braces is recorded as a diagnostic only and never makes a reply valid.
3. **Static scan.** The patch is scanned (Section III-D) and the findings it added relative to the buggy code are recorded.
4. **Execution.** Only a schema-valid patch without an added block-class finding is executed, against the shown test and then the hidden tests (Section III-E).

An item is *resolved* when its reply is schema-valid, adds no block-class finding, and passes every shown and hidden test. An item whose shown test passed and a hidden test failed is an *overfit*. A judge reply's prediction is the verdict of a schema-valid reply; a reply without a valid verdict is wrong for either label, and no code runs.

## D. Static analysis

`sast.scan` parses the patch and resolves imports, aliases, simple assignments and `getattr(module, "name")`, so that `from os import system as run; run(cmd)` is found; when the code does not parse, the block-class rules run as regular expressions on the raw text. A finding belongs to the patch when the buggy code did not already contain it (a multiset difference on rule and snippet). Six block-class rules make an item unresolved and prevent its code from running; six flag-class rules are counted and decide nothing.

*Table: The block-class static-analysis rules (`src/llm_eval_harness/sast.py`). Flag-class rules (subprocess, dynamic import, file writes, network, introspection, `sys.exit`) are reported but decide nothing.*

| Rule | CWE | Matches |
|---|---|---|
| `eval-exec` | CWE-95 | calls of the built-in `eval` or `exec` |
| `os-command` | CWE-78 | `os.system`, `os.popen`, `os.exec*`, `os.spawn*`, `pty.spawn` |
| `shell-true` | CWE-78 | any call with `shell=True` |
| `unsafe-deserialization` | CWE-502 | `pickle`, `marshal` or `dill` loads, `shelve.open`, unsafe `yaml.load` |
| `process-control` | | `os._exit`, `os.kill`, `atexit.register`, `signal.signal`, `signal.alarm` |
| `interpreter-tamper` | | `sys.settrace`, `sys.setprofile`, writes into `sys.modules` or `builtins` |

A block-class rule is admitted only if it never fires on a correct fix: suite validation checks every reference solution against it.

## E. Execution and the sandbox

Tests are stored as data: a call expression and either the Python literal it must return, optionally with a numeric tolerance, or the exception class it must raise. The child process that runs model-written code receives only the call expressions and reports, per call, the `repr` of the returned value or the class names of the raised exception. The parent process decides the outcome (execution rule `values/1`). A test passes only when its record exists, is the only record for its index, and matches the expectation; a missing record is a failure, so an early `os._exit`, a crash or a skipped call cannot pass, and an exit status of 0 is never a pass by itself. If the sandbox itself cannot start, the evaluation stops with an infrastructure error instead of scoring the item as failed.

The sandbox runs the child as `python -I -S` in a fresh temporary directory that is its working directory, `HOME` and `TMPDIR`, with an environment built from nothing, under a deadline (10 s per item by default) after which the whole process group is killed, and with best-effort POSIX limits on CPU time, address space (1 GiB) and file size (10 MiB). It is explicitly not a security boundary: the child runs as the invoking user and can read that user's files and open network connections. In CI the only code executed is the code in committed cassettes, reviewed like any other file; untrusted recordings should be scored in a container without network access.

## F. Statistics

**Rates.** Every rate $k/n$ carries a Wilson score interval [26] with $z = 1.96$:

$$ \frac{k + z^2/2}{n + z^2} \pm \frac{z \sqrt{k(n-k)/n + z^2/4}}{n + z^2}, $$

which, unlike the normal approximation, stays inside [0, 1] and does not collapse at $k = 0$ or $k = n$ [27].

**Judge metrics.** With true positives and false negatives counted on reference items and true negatives and false positives on wrong items, and a reply without a valid verdict counted as wrong in its class, the primary metric is balanced accuracy,

$$ \mathrm{BA} = \tfrac{1}{2}\left(\frac{TP}{P} + \frac{TN}{N}\right), $$

with a bootstrap interval over tasks. A report flags a judge whose interval does not lie above 0.5.

**Paired comparison.** The arms are joined on item identifiers. For item $i$, let $x_i = c_i - b_i$ with $c_i, b_i \in \{0, 1\}$ the correctness of the candidate and of the baseline. *Gains* are the items with $x_i = +1$, *losses* those with $x_i = -1$, and $\Delta$ is the primary metric of the candidate minus that of the baseline on the joined items, $(\text{gains} - \text{losses})/n$ for the resolved rate. The 95% interval of $\Delta$ is a percentile interval from 2,000 paired bootstrap draws [23] of *clusters* with replacement: a task's two judge items, and twin tasks that share a group, are drawn together because they are not independent. The generator is seeded with the string `<seed>/<feature>/<split>`, so the interval is identical on every machine.

**McNemar's test.** With $b$ losses and $c$ gains, the exact two-sided McNemar p-value [16] is

$$ p = \min\left(1,\; 2 \sum_{k=0}^{\min(b,c)} \binom{b+c}{k} 2^{-(b+c)}\right), $$

and $p = 1$ when $b + c = 0$. With no gains, at least six losses are needed for $p < 0.05$ (since $2 \cdot 2^{-5} = 0.0625$ and $2 \cdot 2^{-6} = 0.03125$), whatever the suite size; every report states this number.

**Minimum detectable effect.** With $\widehat{\mathrm{se}}$ the standard deviation of the bootstrap deltas, the report gives $\mathrm{MDE}_{80} = (z_{0.975} + z_{0.80})\,\widehat{\mathrm{se}} \approx 2.80\,\widehat{\mathrm{se}}$: the smallest true change that a two-sided 5% test on this suite would detect with 80% power.

## G. The gate rule

Rule `noninferior/1` promotes a candidate only when every applicable gate passes. The table below lists them.

*Table: The gates of rule `noninferior/1` (`src/llm_eval_harness/gate.py`). The defaults are margin 0, ties pass, and alpha 0.05.*

| Gate | Passes when |
|---|---|
| `schema_validity` | the candidate's schema validity is at least 0.95 |
| `think_validity` | the candidate's thinking validity is at least 0.90 |
| `new_sast_violations` | no item has a block-class finding in the candidate's reply whose rule is absent from the baseline's reply to the same item (bug fix only) |
| `primary` | $\Delta \geq -\text{margin}$ (ties pass), or $\Delta > -\text{margin}$ with `--tie fail` |
| `regression_test` | not (losses > gains and McNemar $p <$ alpha) |
| `gain_test` | only with `--require-gain`: gains > losses and McNemar $p <$ alpha |

The tie policy is part of the rule because a tie is a decision. Passing ties is the right default for a regression test, which should not block a refactoring that changes nothing; replacing a baseline because a change is meant to help should demand more, `--tie fail --require-gain`. A positive margin absorbs the noise of a nondeterministic endpoint. The gate decides on the point estimate, with McNemar's test as a second guard, because on suites of a few dozen items the 95% interval of $\Delta$ is wider than any margin worth setting: requiring the interval's lower end to exceed the negative margin would fail nearly every candidate, and requiring a significant regression before rejecting would pass nearly every one. Without a baseline, the primary metric is compared with a floor (0.5 for the bug fix, 0.7 for the judge).

The gate refuses, with exit status 2 and no report, to compare arms of different features, splits or suites, arms that do not cover the same item identifiers, and recordings that do not answer the requests rebuilt today. Exit status 0 means only that the evaluation ran; CI reads the verdict from the report (`jq -e '.promote == true'`), so a rejection and a failure to run remain distinguishable. The report contains no time stamps, host names or absolute paths.

## H. The ledger

`ledger/ledger.jsonl` is committed with the code. Each row carries the SHA-256 of the previous line and a hash of its own content, so an edited, removed or reordered row is detected by `llm-eval ledger verify`, which also re-fingerprints each feature's current baseline directory. A promotion is refused unless its report is a passing held-out report of exactly the candidate being promoted against the current baseline; an override must state a reason and is recorded as one. A rollback pops the stack of promotions. Evaluation rows, written by `llm-eval gate --record-evaluation`, count the distinct candidates judged on the held-out split; once ten have been judged, every report warns that the split is spent. CI reads the ledger and never writes it.

## I. Continuous integration

*Table: Jobs of `.github/workflows/ci.yml`. The workflow has read-only repository permissions.*

| Job | Trigger | Function |
|---|---|---|
| `tests` | every push to `main`, every pull request | unit and end-to-end tests on Python 3.10, 3.11 and 3.12; suite validation; ledger verification |
| `regression-gate` | after `tests` | replays the demo cassettes, gates the candidate and a deliberately regressed candidate against the current baseline on the held-out split, writes the job summary, uploads the reports, then requires the candidate to promote and the regressed candidate to be rejected by the expected gates |
| `experiments` | after `tests` | replays every experiment's recorded real-model cassettes and fails if the current harness does not reproduce its committed reports byte for byte |
| `live-config`, `live-eval` | manual dispatch | when repository secrets configure an endpoint, record a candidate and a baseline live and gate them |

The self-test in `regression-gate` makes a harness change that would let a clearly regressed candidate through fail the build even though the real candidate still passes. `tests/test_workflow.py` parses the workflow without a YAML library, checks every `llm-eval` command line in it against the command-line parser, and runs the gate job's scripts, including their failure paths, on a copy of the repository.

## J. Report generation

The result tables of this report are generated by `scripts/make_report_tables.py` from the committed gate reports in `experiments/*/expected/*.json`, as markdown (`docs/report/tables.md` and marked blocks in Section V) and as LaTeX (`docs/paper/tables/*.tex`). The script only formats; it computes no statistic. `tests/test_report_tables.py` fails when a committed table differs from the one the reports produce, so the tables cannot drift from the data that CI reproduces.
