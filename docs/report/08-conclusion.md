# VII. Conclusion and Future Work

This report presented an evaluation harness that turns a change to an LLM feature into a paired, reproducible and explicitly specified promotion decision, enforced in CI without model calls. Scoring is grounded in execution, with hidden tests, values compared outside the process that runs the model's code, and a static scan that keeps unsafe patches from running. The gate applies a non-inferiority rule with an explicit tie policy and reports, with every decision, how large a change its suite could detect. Recorded replies make every report reproducible byte for byte, and a deliberately regressed candidate tests the gate in every CI run.

The study on a 27-billion-parameter model shows that these properties are necessary rather than ornamental. An identity adapter was rejected by a margin-0 gate on the strength of one reply per split, a dev-split gain of +0.154 balanced accuracy could not be confirmed on a held-out split at ceiling, and six of the seven disagreements between arms were failures of format or truncation rather than of reasoning. At the suite sizes a product team can afford, the decision rule, its margin and the suite's difficulty determine the verdict as much as the change under test does.

Future work follows directly from the threats in Section VI:

1. **Calibrate the margin from a no-op arm.** Record a no-op comparison for every serving stack and derive the margin, or a significance requirement on losses, from the measured discordance.
2. **Separate format from capability.** Report functional failures and format failures as separate metrics, so that a regression in output format is not read as a regression in reasoning, and vice versa.
3. **Harder, larger and private suites.** Replace the ceiling-bound demo split with a suite on which the baseline is far from 1.000, keep its held-out part private, and extend the executor to repository-level tasks with per-task environments.
4. **Several samples per item.** Record several replies per item, or replies under several batch configurations, to estimate the nondeterministic component of each comparison, and evaluate batch-invariant inference [15] as a way to remove it.
5. **Complete Experiment C.** Record the bf16, Q8_0 and ternary arms on HumanEvalFix, commit their reports, and regenerate the tables of Section V-F with `scripts/make_report_tables.py`.
