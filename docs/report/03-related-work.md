# II. Related Work

## A. Execution-based evaluation of code generation

HumanEval [6] established functional correctness, measured by executing unit tests, as the standard for evaluating generated code, and MBPP [7] provided a larger set of entry-level Python problems with test cases. HumanEvalPack [17] extended HumanEval to code repair, explanation and synthesis in six languages; its repair task, HumanEvalFix, gives the model a function with an inserted bug and is the task family closest to the `bugfix` feature evaluated here. SWE-bench [8] moved evaluation to real repositories, with issues and pull requests from popular Python projects. EvalPlus [18] showed that the original HumanEval tests miss a substantial share of wrong programs, and that adding generated tests lowers reported pass rates. The present harness adopts the same lesson: function tasks include hidden tests beyond those shown to the model so patches overfitted to shown tests are counted separately, while repository-level suites extend evaluation to real pull-request reviews and multi-file patches. Unlike general leaderboards, this harness compares two configurations of one feature to govern replacement decisions in CI.

## B. Contamination and the reuse of held-out data

Public benchmarks may appear in pretraining data. Sainz et al. [19] argue that contamination must be measured per benchmark, and LiveCodeBench [20] mitigates it using problems published after model cut-offs. A second leakage mechanism arises adaptively from repeated evaluation on a test split. Dwork et al. [21] formalise adaptive holdout reuse, and Recht et al. [22] examine accuracy shifts when rebuilding vision test sets. To prevent split drift and track reuse, our harness partitions tasks deterministically by hashing a stable group or identifier key (keeping twin or same-repository tasks on the same side), and records held-out evaluations against a ledger budget.

## C. LLMs as judges and reviewers

Zheng et al. [13] found strong LLM judges agree closely with human preferences while exhibiting position, verbosity and self-enhancement biases. For software patches, correctness can be checked against execution rather than subjective preference. Both our `judge` feature (symmetric reference vs. wrong patch pairs) and `review` feature (real PR diffs labelled by repository execution outcomes) score verdicts strictly against execution-derived ground truth so that rubber-stamping or ignoring the diff yields 0.5 balanced accuracy.

## D. Statistical comparison of models

McNemar's test [16] compares two classifiers on the same items using only discordant pairs; Dietterich [9] showed it maintains low Type I error. Paired bootstrap resampling [23] provides confidence intervals for score differences [24], and Dror et al. [25] systematize significance selection in NLP. Card et al. [11] documented widespread underpowered comparisons, and Miller [10] recommends clustered standard errors and power planning when items share clusters. The Wilson score interval [26] bounds binomial proportions within $[0, 1]$ without boundary collapse [27]. Our harness integrates Wilson intervals, cluster bootstrap confidence intervals, exact McNemar testing, explicit $\mathrm{MDE}_{80}$ reporting, and clinical non-inferiority gating [12].

## E. Testing machine-learning systems

Breck et al. [28] organize production ML readiness into 28 tests, noting that model behaviour is hard to assert statically. CheckList [29] brings structured behavioural testing to NLP models. Rather than non-deterministic live endpoints in CI, our harness adopts HTTP record-and-replay semantics akin to VCR.py [30], but binds every cassette row to the SHA-256 of the exact canonical JSON request payload so prompt, model, or setting edits refuse stale replays.

## F. Adaptation, quantization and serving

LoRA [1] adapts frozen models via low-rank updates; zero-initialized update matrices form an exact arithmetic identity control. Post-training quantization such as GPTQ [2] compresses weights to 3–4 bits, while BitNet b1.58 [3] trains native ternary $\{-1, 0, +1\}$ weights. Ternary Bonsai 2 27B [31] builds on Qwen3.8-27B [32] in GGUF format for llama.cpp [5]. vLLM [4] serves models with PagedAttention, continuous batching, and speculative decoding [33]. Yet floating-point non-associativity across dynamic batching and kernel reductions breaks determinism: Yuan et al. [14] observe up to 9% reasoning swings under greedy bfloat16 decoding, and He et al. [15] link run-to-run variation to batch-variant kernels. Our experiments quantify both serving nondeterminism and reasoning-effort truncation directly under CI regression gating.
