# VI. Discussion and Threats to Validity

## A. What the decisions mean

The controlled arms show that the gate does what its rule says: it promotes a candidate that is not worse, rejects one that is, and names the gates responsible (H1), and CI reproduces every decision exactly (H2). Experiment A shows what the rule means on a real model at small suite sizes. A margin of 0 rejected an adapter that changes nothing, on the evidence of one reply per split, and the significance guard correctly did not fire. Conversely, a real dev-split gain of four items was promoted but could not be confirmed, and the held-out split could not have confirmed any gain because the base model was at ceiling. Both outcomes are visible in the reports themselves (McNemar's p, $\mathrm{MDE}_{80}$ and the warning on the number of losses needed for significance), which is the purpose of reporting them with every decision.

Two practical rules follow. First, before a real comparison, a no-op arm should be recorded against the baseline to measure the serving noise floor, and the margin should be set from it, or promotion should additionally require that a loss be significant; Experiment A puts that floor at one discordant item in 13 to 21 on this suite and stack. Second, a held-out split on which the baseline scores at or near 1.000 cannot rank candidates, and a gain observed elsewhere should not be reported as confirmed until a harder or larger held-out suite is available.

## B. Internal validity

**Serving nondeterminism.** The identity adapter is exactly equal to the base model in exact arithmetic, yet it changed replies. The serving stack in Experiment A applied the adapter through LoRA kernels, batched requests and used speculative decoding (`experiments/exp-a-qwen38/README.md`); each can change floating-point reduction order, and greedy decoding amplifies a single changed token into a different reply [14], [15]. The experiment cannot attribute the change to one of these mechanisms, because it recorded one sample per item and did not vary them. Every difference between arms in this report therefore includes a nondeterministic component of unknown size beyond the one measured by the no-op arm.

**One sample per item.** Each arm holds one reply per item at temperature 0. A second recording of the same arm could differ on some items, and the paired analysis treats the recorded reply as the arm's fixed answer. The bootstrap interval reflects sampling of tasks, not sampling of replies.

**Engine confounds (Experiment C).** vLLM and llama.cpp differ in kernels, batching, sampling implementation, chat-template rendering and the parsing of reasoning output. A comparison of a ternary model under llama.cpp with a bf16 model under vLLM confounds the weights with the engine; the Q8_0 arm under llama.cpp is designed to separate the two (C1 and C2). The separation is partial: C1 also includes 8-bit rounding, and C2 compares two different weight sets, because Ternary Bonsai 2 is published as its own model built from Qwen3.8-27B [31] rather than as a rounding of the same weights, so C2 measures the published ternary model and not ternary rounding as such.

**Recording conditions.** Both arms of Experiment A were recorded on the same server and with identical settings, so differences in hardware and settings do not enter the comparison; the order and load of requests during recording were not controlled.

## C. External validity

The demo suite consists of 34 small, synthetic, single-function Python tasks, and Experiment A evaluates one model family, one base checkpoint (a community fine-tune rather than the stock release) and two adapters. Real bug fixing spans files and repositories with their own environments [8]; the harness's structure (paired items, execution labels, the gate and the ledger) carries over, but its sandbox does not, and per-task containers would be needed. HumanEvalFix, the planned suite of Experiment C, is also single-function Python. The findings about the gate's behaviour on small suites are expected to generalise; the absolute scores are not.

## D. Construct validity

**Format and capability are confounded in the primary metric.** Six of the seven discordant items in Experiment A are format or truncation failures (Section V-D), and no bug-fix reply that was valid failed a test. The resolved rate and the judge's balanced accuracy, as defined, measure whether the feature produces a usable answer, which is what a product receives, but a reader interpreting them as reasoning ability would be misled. A per-item classification of failures, as in Section V-D, should accompany any interpretation.

**The thinking window.** The 1 to 1,200 word window and the 16,384-token limit turn long reasoning into failures. They encode a product decision about latency and cost, not a property of correctness; another product could set them differently and obtain different verdicts.

**Test adequacy.** "Resolved" means that the shown and hidden tests passed. Hidden tests reduce, but do not eliminate, the acceptance of wrong patches [18].

**The judge's task.** The judge is evaluated on exactly two patches per task, the reference and one wrong fix that passes the shown test. This design isolates whether the judge reads the patch, but it does not cover other kinds of wrong patch, and its labels are only as good as the tests.

**Static analysis.** The scan resolves common aliasing but cannot see a name constructed at run time, so the absence of a block-class finding is evidence, not proof, of the absence of the construct.

## E. Conclusion validity

**Small suites and low power.** With 13 to 42 items per comparison, $\mathrm{MDE}_{80}$ lies between 0.064 and 0.210, and McNemar's test needs at least six losses with no gains to call a regression significant [11]. A promotion is therefore evidence of non-inferiority at the stated margin, not of equality, and a rejection by one item is not evidence of a regression.

**Interval and test can disagree.** On the judge dev split the bootstrap interval of $\Delta$, [+0.038, +0.269], excludes 0 while McNemar's exact p is 0.125. With four discordant items, all in one direction, the exact test cannot reach significance, whereas a percentile bootstrap over only 13 clusters, whose resamples contain no loss at all, can understate the uncertainty at such sizes. The gate decides on neither alone; the report gives both, and neither should be cited without the other.

**Ceiling effects.** When the baseline scores 1.000, $\Delta \le 0$ is certain and any change can only tie or lose. Two of the four comparisons in Experiment A have a baseline at ceiling.

**Multiple comparisons.** Experiment A comprises four decisions and Experiment C will add at least three, without correction. Each decision is reported as a separate gate verdict and not as a family-wide claim.

**Post hoc hypotheses and reuse of the held-out split.** H3 and H4 were written in their present form after Experiment A's replies existed. They are the predictions implied by the arms' construction (an exact identity and a trained adapter), but they were not registered in advance. The demo suite's held-out split has been used for the synthetic arms and for Experiment A; the ledger counts held-out evaluations and warns after ten distinct candidates, a heuristic budget rather than a guarantee of the kind studied by Dwork et al. [21].

**Contamination.** The demo suite is public in the repository, so a model trained on the repository could have seen its held-out split; its held-out status protects against prompt tuning, not against training on the internet. HumanEvalFix derives from HumanEval, published in 2021 and widely copied, and may be in the pretraining data of the models compared in Experiment C [19], [20]. Contamination raises absolute scores and could interact with quantization if memorised solutions survive weight reduction differently from reasoned ones; a paired comparison removes neither effect.
