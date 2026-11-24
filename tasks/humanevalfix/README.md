# The HumanEvalFix suite (`tasks/humanevalfix`)

A second task suite for the harness, converted from **HumanEvalFix**: the Python subset of
HumanEvalPack (`bigcode/humanevalpack`, MIT license). Each of its tasks is a real benchmark
problem with a human-written bug. The demo suite (`tasks/demo`, 34 synthetic tasks) can show only
large changes. This suite is larger, so it can show smaller ones. Experiment C uses it to compare a
ternary-quantized model with its bf16 base.

This document covers the suite's provenance, the conversion rules, what was left out and why,
the split, the statistical power, and the threats to validity, in the order a methods section
needs them. Attribution and the license text are in [NOTICE](NOTICE). The per-task record of
every left-out task and dropped assert is in [IMPORT.json](IMPORT.json).

## 1. Summary

| Quantity | Value |
|---|---|
| Source | `bigcode/humanevalpack`, configuration `python`, split `test`, revision `9a41762f73a8`, file sha256 `ed5f15d7...8075` |
| Dataset rows | 164 (HumanEval's 164 Python problems, each with one human-written bug) |
| Tasks kept | 115 of the 164 rows (70.1%) |
| Split | 74 held-out tasks, 41 dev tasks (`heldout_pct` 60) |
| Tests | 320 shown tests (median 2 per task), 540 hidden tests (median 4, range 2 to 25) |
| Items | bugfix: one per task (74 held-out). judge: two per task, the reference and the wrong patch (148 held-out) |
| Validation | `llm-eval tasks validate --suite tasks/humanevalfix`: valid, 0 problems, 0 warnings |
| Synthetic | no: the code, docstrings, bugs and test values are the dataset's |

## 2. Provenance

HumanEvalPack (Muennighoff et al., 2023, the OctoPack paper) extends OpenAI's HumanEval (Chen et
al., 2021). Its Python split keeps HumanEval's problems, canonical solutions and tests. For each
problem it adds a `buggy_solution` with one human-written bug, labelled with a `bug_type` (missing
logic, excess logic, value misuse, operator misuse, variable misuse, function misuse) and a
`failure_symptoms` value (incorrect output, stack overflow, infinite loop). The bug-fixing task
built on these is HumanEvalFix.

`scripts/import_humanevalfix.py` reads the Python split's parquet file with the standard library
only (the script has its own small parquet reader). `--download` fetches the file of the pinned
revision from the Hugging Face hub and refuses it unless its sha256 matches. `--source` takes a
local `.parquet` or `.jsonl` copy. The source's sha256 is recorded in `IMPORT.json`.

## 3. Conversion rules

| Harness field | Built from |
|---|---|
| `id` | `task_id`, in the harness's id alphabet: `Python/12` becomes `python-12` |
| `entry_point` | `entry_point` |
| `buggy_code` | `prompt` + `buggy_solution`: imports, helper functions, signature, docstring and the buggy body. This is what the model is shown |
| `reference_code` | `prompt` + `canonical_solution` |
| `wrong_code` | `prompt` + a guard + `buggy_solution` (see below) |
| `shown_tests` | the expressible asserts of `example_test`, the docstring's examples written as asserts |
| `hidden_tests` | the expressible asserts of `test`, the full test, without those the model is shown (see below) |
| `tags` | `bug_type` and `failure_symptoms` as slugs, for example `["operator-misuse", "incorrect-output"]` |
| `group` | shared by near-duplicate tasks (section 5) |
| `synthetic` | `false` |

**Expressible asserts.** The harness stores a test as data: a call, plus the literal its value
must equal (section 4 of `docs/DESIGN.md`). An assert is converted when it is a top-level
statement of the test's `check()` function of the form `assert f(<args>) == <literal>` (either
side, with or without a message), where `f` is the function under test and every argument is a
literal or arithmetic on literals (`factorize(3 * 19)`). The harness then evaluates the call in its
sandbox and compares the value with the literal by `==`, as the dataset's assert does. Every
other assert is dropped on its own, with a reason. The dropped forms are tolerance comparisons
(`abs(f(x) - y) < 1e-6`), identity asserts (`f(x) is True`), truthiness asserts (`assert f(x)`,
`assert not f(x)`), non-literal arguments or expectations, asserts inside loops (randomised
property tests), and HumanEval's `assert True` lines. The first three could be written as call
expressions that evaluate to `True`, but they are dropped on purpose. That way every test stays
a call compared with a literal, and the shown test renders as a plain assert the model can read.

**Hidden means unseen.** A hidden assert is dropped when its call is also a shown test (HumanEval's
full tests usually repeat the examples: 247 asserts), when the call appears in the docstring
(compared as syntax trees, so `f("a")` in the docstring matches `f('a')`), or when its text
appears in any rendered prompt of any published template.

**The wrong patch.** The judge feature needs one patch per task that is wrong. The harness requires
that patch to pass every shown test and fail a hidden one (it must look plausible), and to differ
from the buggy code. The buggy solution fails a shown test by construction, so it cannot be used
unchanged. `wrong_code` is therefore the buggy solution behind a guard that returns each shown
test's expected value for exactly that test's arguments:

```python
def sort_array(array):
    """... (the dataset's docstring) ..."""
    if array == []:
        return []
    if array == [5]:
        return [5]
    ...
    return [] if len(array) == 0 else sorted(array, reverse= (array[0]+array[-1]) % 2 != 0)
```

The guard is the overfitted patch the harness's `wrong_code` stands for: special-casing the shown
inputs. It keeps the dataset's bug, so the patch fails the hidden tests the bug fails. The guard
binds every parameter, including the defaults the call leaves out, so it fires only on
arguments equal to those of a shown test.

**Verification by execution.** Every candidate task runs in the harness's own checker
(`verdicts.exec_verdict`, the sandbox described in the main README). The dataset's code runs
nowhere else. A test the reference fails under the harness's value rule would be dropped as an
inexact conversion, but no test needed it: every converted assert gives the reference the same
verdict as the dataset's assert. A task is then kept only if it meets every rule of
`llm-eval tasks validate`:

- the buggy code fails at least one shown test, so the shown tests demonstrate the bug;
- the buggy code fails at least one hidden test;
- the reference passes every test twice, with the same outcomes;
- the wrong patch passes every shown test and fails at least one hidden test;
- no code trips a block-class SAST rule;
- no prompt shows a hidden test or code it must not show;
- at least 2 hidden tests remain.

## 4. What was left out

Each skipped task carries the first rule that refused it.

| Tasks | Reason |
|---|---|
| 14 | no expressible example test: the examples are tolerance, truthiness or `is` asserts, or there are none (`decode_cyclic`, `find_zero`, `correct_bracketing`, ...) |
| 12 | the buggy code passes every example test, so the shown tests demonstrate no bug (the bug shows only in the full test) |
| 9 | the buggy code passes every expressible hidden test (it fails only asserts the model is shown, or only dropped ones) |
| 7 | a prompt contains the reference code as text, which the validator's leakage rule refuses. In all seven, the bug extends the fix's last line (`return x + y` becomes `return x + y + y + x`), so the stripped reference is a substring of the buggy code, although nothing leaks |
| 4 | 0 expressible hidden tests left (for example, the full test only repeats the examples) |
| 3 | 1 expressible hidden test left; the harness needs at least 2 |
| **49** | in total, of 164 |

Asserts dropped inside the 115 kept tasks: 247 hidden asserts whose call is a shown test, 4 whose
call appears in the docstring, 1 repeated call, 1 non-literal argument, 1 non-literal expected
value, 1 inside a loop, and 62 `assert True` lines (36 in the full tests, 26 in the examples).

Leaving tasks out changes the population. The 49 left-out tasks are not a random sample: tasks
whose bug shows only beyond the examples, and tasks with floating-point or boolean-only tests,
are under-represented. Scores on this suite are therefore not comparable with published
HumanEvalFix pass@1 numbers (section 8).

## 5. Split and near-duplicates

The split is the harness's: a task goes to the held-out split when
`int(sha256(group or id)[:8], 16) % 100 < heldout_pct`. A `heldout_pct` of 60 is the smallest
round value that puts at least 70 tasks in the held-out split: **74 held-out**, **41 dev**. The
importer refuses a value that gives fewer than 70.

Near-duplicate tasks share a `group`, so they fall on the same side. Two tasks are near
duplicates when the harness's twin detector (`tasks.find_twins`: identifiers and literals
normalised, difflib ratio at least 0.85) calls their buggy code near duplicates, or when their
docstrings are at least 0.80 similar. The detector drops docstrings, so it cannot see the same
problem solved by different code. Two groups result:

| Group | Tasks |
|---|---|
| `humanevalfix-twins-7-29` | `filter_by_substring`, `filter_by_prefix` |
| `humanevalfix-twins-46-55-63` | `fib4`, `fib`, `fibfib` |

Bug classes per split (`by_tag` in every report):

| Tag | Dev | Held-out |
|---|---|---|
| `value-misuse` | 14 | 18 |
| `missing-logic` | 8 | 15 |
| `operator-misuse` | 2 | 16 |
| `excess-logic` | 5 | 12 |
| `variable-misuse` | 10 | 8 |
| `function-misuse` | 2 | 5 |
| `incorrect-output` (symptom) | 41 | 70 |
| `infinite-loop` (symptom) | 0 | 3 |
| `stackoverflow` (symptom) | 0 | 1 |

The hash split is not stratified. For example, operator-misuse bugs are 22% of the held-out split
and 5% of dev. Per-tag numbers from dev do not transfer to the held-out split.

## 6. Statistical power

The gate compares two arms paired, on the same items. Only the items where the arms disagree (the
discordant items) carry information. For a paired difference of rates near zero, the standard
error is about `sqrt(d / n)`, where `d` is the share of discordant items. The minimum detectable
effect at 80% power and a two-sided 5% test is `mde80 = 2.8 x se`. Every report gives the bootstrap
version of this number. Approximate values, in percentage points of the resolved rate:

| Items `n` | One item | d = 2% | d = 5% | d = 10% | d = 20% |
|---|---|---|---|---|---|
| 74 (bugfix, held-out) | 1.35 | 4.6 | 7.3 | 10.3 | 14.6 |
| 115 (bugfix, all tasks) | 0.87 | 3.7 | 5.8 | 8.3 | 11.7 |
| 148 (judge, held-out; clustered by task, so somewhat less in practice) | 0.68 | 3.3 | 5.1 | 7.3 | 10.3 |

On the 74 held-out items, an effect of a few points is resolved only when the two arms disagree
on very few items (2% to 5%). That is plausible for a quantized model against its own bf16 base,
and implausible for two unrelated models. McNemar's exact test needs at least 6 losses with no
gains before it calls a regression significant, whatever the suite size. Record a no-op arm
first (experiment A shows why) to measure the floor of disagreement the serving stack alone
produces.

## 7. Using the suite

```bash
llm-eval tasks validate --suite tasks/humanevalfix       # about 40 s: runs every task's code in the sandbox
llm-eval tasks split --suite tasks/humanevalfix
llm-eval record --suite tasks/humanevalfix --feature bugfix --model my-model --out experiments/<exp>/<arm>
llm-eval gate --suite tasks/humanevalfix --feature bugfix --split heldout \
  --candidate experiments/<exp>/<candidate> --baseline experiments/<exp>/<baseline>
```

Pass `--suite tasks/humanevalfix` to every command: the default suite is the demo. Cassettes record the suite's sha256, and the gate refuses to compare arms scored on different suites.

To rebuild the suite from the dataset (it is not in the repository):

```bash
python scripts/import_humanevalfix.py --download                    # or --source <file.parquet|file.jsonl>
python scripts/import_humanevalfix.py --source <file> --check       # rebuild in memory and compare bytes
```

The import is deterministic. With the pinned file it reproduces `suite.json`, `tasks.jsonl` and
`IMPORT.json` byte for byte. A JSONL copy of the same rows reproduces `suite.json` and `tasks.jsonl`. Its
`IMPORT.json` differs only in `source_sha256`, the hash of the file actually read. `tests/test_humanevalfix_suite.py` pins the counts in this document.
It also runs every task's reference, buggy and wrong code in the checker (about 30 s). The full
validation (`llm-eval tasks validate` on the whole suite, about 40 s) runs with `LLM_EVAL_SLOW=1`, and
always in CI, where `CI=true` is set, so a broken or drifted suite fails the CI tests job.

## 8. Threats to validity

- **Contamination.** HumanEval has been public since 2021, and its solutions are widely present in
  training data. A model may know the canonical solution without reading the bug. Here, held-out
  means held out from prompt tuning, not unseen by the model. The suite measures a difference
  between two arms on the same items. It does not measure an absolute bug-fixing ability.
- **Not the HumanEvalFix protocol.** HumanEvalFix as published shows the model the buggy function
  and the full unit tests, and reports pass@k on the full test. This suite shows the docstring
  examples only, and scores against the expressible hidden asserts, on 115 of the 164 problems.
  Its numbers are not comparable with published HumanEvalFix scores.
- **Fewer hidden tests than the original.** The original full tests hold more asserts (tolerance
  checks, property loops). A patch that passes this suite's hidden tests may still fail one of
  those dropped asserts. The rate of such false resolutions is not measured.
- **An easy wrong patch.** The judge's wrong patch is the buggy code plus a visible guard. A judge
  that spots special-cased inputs can reject it without understanding the bug, so judge scores
  on this suite overstate how well a judge detects subtle bugs.
- **Selection.** The skipped tasks are not random (section 4). Bugs whose symptom shows only beyond
  the docstring's examples are left out by the rule that the shown test must fail.
- **Known dataset quirks are kept.** Some HumanEval docstrings disagree with their tests (for
  example, `python-116`'s docstring gives sorted outputs that its example asserts do not). The
  converted tests follow the dataset's asserts, not the prose.
- **The sandbox is not a security boundary** (main README). The dataset's code is trusted public
  benchmark code, and it runs only inside the harness's checker.

## 9. Citation

```bibtex
@article{muennighoff2023octopack,
  title   = {OctoPack: Instruction Tuning Code Large Language Models},
  author  = {Muennighoff, Niklas and Liu, Qian and Zebaze, Armel and Zheng, Qinkai and Hui, Binyuan and
             Zhuo, Terry Yue and Singh, Swayam and Tang, Xiangru and von Werra, Leandro and Longpre, Shayne},
  journal = {arXiv preprint arXiv:2308.07124},
  year    = {2023}
}
@article{chen2021codex,
  title   = {Evaluating Large Language Models Trained on Code},
  author  = {Chen, Mark and others},
  journal = {arXiv preprint arXiv:2107.03374},
  year    = {2021}
}
```
