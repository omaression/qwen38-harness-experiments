# The project report

This directory holds the text of the class deliverable, an IEEE-format project report on the harness and its experiments, one file per section. `docs/paper/main.tex` holds the same text as an IEEEtran conference paper, with `docs/paper/refs.bib` and the generated tables in `docs/paper/tables/`.

The text is written formally and in the third person. Every claim in it is meant to be traceable to a file, a committed gate report or a verified reference. The index below gives, for each section, the files to check a claim against.

## Index: IEEE sections and their sources

| Section | File | Source material in the repository |
|---|---|---|
| Abstract, Index Terms | [01-abstract.md](01-abstract.md) | the results of Sections V-A to V-F |
| I. Introduction: problem, motivation, numbered contributions | [02-introduction.md](02-introduction.md) | `README.md`; `docs/DESIGN.md` sections 1 and 13; the modules named in each contribution |
| II. Related Work | [03-related-work.md](03-related-work.md) | [09-references.md](09-references.md) only |
| III. System Design: architecture and data flow, the features, scoring, SAST, sandbox, statistics, the gate rule, ledger, CI, report generation | [04-system-design.md](04-system-design.md) | `docs/DESIGN.md` sections 2 to 12; `src/llm_eval_harness/{records,tasks,prompts,clients,cassettes,features,verdicts,sast,sandbox,metrics,gate,ledger}.py`; `.github/workflows/ci.yml`; `scripts/make_report_tables.py` |
| IV. Experimental Methodology: suites and splits, arms, recording, replay determinism, hypotheses H1 to H6, analysis plan | [05-methodology.md](05-methodology.md) | `tasks/demo/`; `tasks/humanevalfix/` and `scripts/import_humanevalfix.py` (Experiment C's suite); `scripts/make_demo_cassettes.py`; `experiments/exp-*/README.md` and `*/manifest.json`; the CI job `experiments` |
| V. Results | [06-results.md](06-results.md) | generated tables (below); `README.md` "The demo results" (pinned by `tests/test_demo.py`); per-item rows from `llm-eval gate --items-out` |
| VI. Discussion and Threats to Validity: internal, external, construct and conclusion validity | [07-discussion.md](07-discussion.md) | Sections V-C to V-G; `tasks/humanevalfix/README.md` section 8; `docs/DESIGN.md` section 14; `README.md` "Limitations" |
| VII. Conclusion and Future Work | [08-conclusion.md](08-conclusion.md) | Sections V and VI |
| References | [09-references.md](09-references.md) | `docs/paper/refs.bib` (same entries, same order) |
| Result tables | [tables.md](tables.md) | generated from `experiments/*/expected/*.json` |

## Where the numbers come from

| Claim | Evidence |
|---|---|
| Experiment A's gate decisions, deltas, intervals, gains and losses, McNemar p, MDE80, failed gates, per-arm metrics | `experiments/exp-a-qwen38/expected/{bugfix,judge}-{heldout,dev}.json`, through the generated tables |
| Every experiment report (4 of Experiment A, 16 of Experiment C, 12 of its demo replication) is reproduced byte for byte | the CI job `experiments` in `.github/workflows/ci.yml` |
| Experiment C's gate decisions, deltas, intervals, gains and losses, McNemar p, MDE80, token means and ratios, per-arm metrics | `experiments/exp-c-bonsai/expected/<candidate>--vs--<baseline>.<feature>-<split>.json` and `experiments/exp-c-bonsai-demo/expected/*.json`, through the generated tables |
| Experiment C's discordant items and their classification (Sections V-F, V-G) | `llm-eval gate --suite tasks/<SUITE> --feature <f> --split <s> --candidate experiments/<exp>/<arm> --baseline none --items-out DIR` for each arm, then the `correct`, `problems`, `finish_reason` and `think_words` fields of the rows, joined on `item_id` |
| Recording settings of Experiment C, and the publisher's retention figure and effort guidance | `experiments/exp-c-bonsai*/*/manifest.json`, `experiments/exp-c-bonsai/README.md`; reference [31] (model card and its `KNOWN_ISSUES.md`) |
| The seven discordant items and their classification (Section V-D) | `llm-eval gate --feature <f> --split <s> --candidate experiments/exp-a-qwen38/qwen38-adapters --baseline experiments/exp-a-qwen38/qwen38-base --items-out DIR`, then the `problems`, `finish_reason` and `think_words` fields of the rows in `DIR` |
| The synthetic arms' decisions (Section V-A) | `README.md` "The demo results" and `docs/DESIGN.md` "Demo results", pinned by `tests/test_demo.py` |
| Recording settings of Experiment A | `experiments/exp-a-qwen38/qwen38-base/manifest.json`, `experiments/exp-a-qwen38/qwen38-adapters/manifest.json`, `experiments/exp-a-qwen38/README.md` |
| Split sizes (21 held-out tasks in 20 clusters, 13 dev tasks) | `llm-eval tasks split`; `docs/DESIGN.md` section 4 |
| Constants: thinking window 1 to 1,200 words, sandbox deadline 10 s, 1 GiB address space, 10 MiB files, 2,000 bootstrap draws, held-out budget 10 | `features.py` (`BUGFIX_THINK`, `JUDGE_THINK`), `sandbox.py`, `metrics.py`, `gate.py` (`HELDOUT_BUDGET`) |

## Generated tables

Every table in Section V except the demo table of Section V-A is generated by

```bash
python scripts/make_report_tables.py          # rewrite the tables after any report changes
python scripts/make_report_tables.py --check  # what tests/test_report_tables.py runs
```

The script reads every `experiments/*/expected/*.json` and writes `docs/report/tables.md`, `docs/paper/tables/<id>-gate.tex` and `<id>-arms.tex` for each experiment (`<id>` is the directory's leading `exp-<letter>`, with `-demo` appended for a directory ending in `-demo`: `exp-a`, `exp-c` and `exp-c-demo`), and the text between `<!-- BEGIN GENERATED: <name> -->` and `<!-- END GENERATED: <name> -->` in these markdown files. It only formats numbers; it computes no statistic. Never edit a generated table by hand: `tests/test_report_tables.py` fails until the script has been rerun.

## Experiment C

Experiment C (Ternary Bonsai 2 against bf16 Qwen3.8-27B on HumanEvalFix, with a Q8_0 engine-control arm and a no-op rerun of the bf16 arm) is recorded in `experiments/exp-c-bonsai/`, and its replication on the demo suite in `experiments/exp-c-bonsai-demo/`. Each experiment directory's `SUITE` file names its suite, and each report is named `<candidate>--vs--<baseline>.<feature>-<split>.json`. The CI job `experiments` reads both from the names, and `scripts/make_report_tables.py` writes one gate row per comparison (C0 to C3, defined in its `COMPARISONS` table) with each comparison's token cost, and one arms row per arm. A new comparison of the same arms needs only a new report; a new arm needs an entry in `ARM_DESCRIPTIONS`, and a new pair an entry in `COMPARISONS`, or it is shown under its arm names.

## Building the paper

```bash
cd docs/paper
pdflatex main && bibtex main && pdflatex main && pdflatex main
```

This needs a TeX distribution with the IEEEtran class and bibliography style (`IEEEtran.cls`, `IEEEtran.bst`). A full TeX Live installation includes them, but split distribution packages often do not: install the package that carries IEEEtran (`texlive-publishers` on Debian and Ubuntu), copy the two files from CTAN (`macros/latex/contrib/IEEEtran`) into `docs/paper/`, or use Overleaf. `kpsewhich IEEEtran.cls` prints a path when the class is installed. The build has not been verified on every machine: compile once before submission. The build files (`main.aux`, `main.pdf` and others) are not meant to be committed. The text of `main.tex` mirrors the markdown sections 01 to 08: when one is edited, edit the other in the same change. The author block is a placeholder.

## Conventions

- Cite only entries of [09-references.md](09-references.md). Each was checked on 2026-11-24 against the publisher, arXiv, the ACL Anthology, PMLR, Crossref, a bibliographic index or the repository it names. A new reference is added to both `09-references.md` and `docs/paper/refs.bib` only after the same check, and the numbering follows the order of first citation.
- A number in the prose either comes from a generated table, is quoted with its source file, or is derived in the text from such numbers.
- The demo arms are synthetic and say nothing about any model; the text never presents them as model results.
- Hypotheses are stated in Section IV before any result. Section VI-E states which of them were written after their data existed.
