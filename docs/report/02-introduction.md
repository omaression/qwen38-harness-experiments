# I. Introduction

## A. Problem

Software teams increasingly ship features whose behaviour is produced by an LLM: a prompt template, a model, optional adapters and a serving configuration together determine the output. Each of these is changed routinely. A prompt is reworded, a model is replaced by a newer release, a parameter-efficient adapter is trained [1], weights are quantized to fit cheaper hardware [2], [3], or the serving engine is replaced [4], [5]. None of these changes touches the code that unit tests exercise, yet any of them can make the feature worse. The question that a regression test must answer for such a change is therefore not whether the code is correct, but whether the change makes the feature worse on inputs it was not tuned on, and how confident that answer can be.

Benchmarks for code generation answer a related but different question: how good a model is, on average, on a public task set [6], [7], [8]. A regression gate instead compares two configurations of one product feature, decides whether the second may replace the first, and must make that decision automatically, reproducibly and cheaply on every change, typically in CI, where calling a 27-billion-parameter model is neither free nor deterministic.

## B. Motivation

Three observations motivate a dedicated harness. First, the comparison is between two arms on the same items, so a paired analysis is available and much stronger than comparing two aggregate scores [9], [10]. Second, the suites a product team can afford to label with executable tests are small, tens to hundreds of items, and on such suites statistical power is low [11]; a gate that does not state how large a change it could detect invites over-interpretation of both its passes and its failures. Third, the decision rule must be explicit about ties. A rule of the form "candidate >= baseline" promotes every change that does nothing, and a rule that demands significance rejects nearly every candidate on a small suite. Non-inferiority testing, as used in clinical trials [12], makes the margin and the direction of the burden of proof explicit, and the same discipline applies here.

LLM-specific failure modes add requirements. A reply can be malformed, truncated at the token limit, or correct only on the test the model was shown. Generated code can call `eval` or spawn processes, and a judge model can approve every patch while sounding confident [13]. Serving is nondeterministic even at temperature zero [14], [15], so two recordings of the same configuration can differ.

## C. Contributions

This report makes the following contributions, each implemented and tested in the accompanying repository:

1. **A paired non-inferiority gate for LLM features** (rule `noninferior/1`, `src/llm_eval_harness/gate.py`) with an explicit margin and tie policy, hard gates on schema and thinking validity, a gate on newly introduced static-analysis violations, an exact McNemar guard [16], and an optional significant-gain requirement. Every report states its own detection limit: the minimum detectable effect at 80% power and the fewest losses McNemar's test would call significant.
2. **Execution-grounded scoring** (`verdicts.py`, `sandbox.py`, `sast.py`) in which tests are stored as data, the child process that runs model-written code returns only values, and the parent compares them, so that an exit status of 0, an early exit or a skipped test is never a pass. Shown and hidden tests separate genuine fixes from patches overfitted to the shown test.
3. **An execution-labelled judge evaluation** in which every task contributes one correct and one plausible wrong patch, so that a judge that ignores the patch scores exactly 0.5 balanced accuracy.
4. **Record and replay of model replies** (`cassettes.py`, `clients.py`): each reply is stored under the SHA-256 of the exact request that produced it, replay refuses a recording that does not answer today's request, and every report is byte-identical across replays. CI therefore needs no GPU, network or secrets, and it gates a deliberately regressed candidate in every run as a self-test of the gate.
5. **A hash-chained promotion ledger** (`ledger.py`) that records promotions and rollbacks, re-fingerprints the current baseline in CI, and counts held-out evaluations against a budget.
6. **An empirical study on a 27-billion-parameter open-weight model** with LoRA adapters (Experiment A), which shows that an identity adapter is rejected by a margin-0 gate because of serving nondeterminism, and that a held-out split at ceiling cannot confirm a development-split gain. A quantization study (Experiment C) is designed with an engine-control arm, and its tables are generated automatically once it is recorded.

## D. Organization

Section II reviews related work. Section III describes the system. Section IV states the experimental methodology and the hypotheses before Section V reports the results. Section VI discusses the results and the threats to their validity, and Section VII concludes.
