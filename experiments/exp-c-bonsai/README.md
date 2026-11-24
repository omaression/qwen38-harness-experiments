# Experiment C: a ternary 27B model against its bf16 base, under the gate

This experiment puts a weight-format and serving-engine change through the harness. Ternary Bonsai 2 27B, a ternary model built from Qwen3.8-27B, is compared with the stock bf16 Qwen3.8-27B on the HumanEvalFix suite. An engine-control arm and a no-op rerun arm separate the effect of the weights from that of the engine and from serving noise. The cassettes in this directory are the models' replies, recorded once. CI replays them, and the `experiments` job checks that the current harness still reproduces the committed reports byte for byte.

## Setup

- **Suite:** `tasks/humanevalfix` (named in the file `SUITE`): 115 HumanEvalFix Python tasks with human-written bugs, split 74 held-out and 41 dev. The bug fix has one item per task. The judge has two items per task, the reference and the wrong patch: 148 held-out and 82 dev items. The suite's conversion, its left-out tasks and its threats to validity are in [tasks/humanevalfix/README.md](../../tasks/humanevalfix/README.md).
- **Arms:**

  | Arm | Weights | Engine |
  |---|---|---|
  | `bf16-vllm` | stock Qwen3.8-27B, bf16 | vLLM |
  | `bf16-vllm-rerun` | the same as `bf16-vllm`, recorded a second time: the **no-op arm**, which measures serving noise | vLLM |
  | `q8-llamacpp` | stock Qwen3.8-27B, converted to a Q8_0 GGUF file: the **engine control** | llama.cpp b12682 |
  | `ternary-llamacpp` | Ternary Bonsai 2 27B, its published `Q2_0` GGUF file (the packing that runs on stock llama.cpp) | llama.cpp b12682 |

- **Recording settings, the same for every arm** (`*/manifest.json`): temperature 0, seed 0, at most 16,384 completion tokens, thinking on and `reasoning_effort: "medium"` (`{"chat_template_kwargs": {"enable_thinking": true, "reasoning_effort": "medium"}}`), prompts `bugfix/v2` and `judge/v2`. Every arm was served under the model name `qwen38`, so the arm name is what tells the arms apart. Medium effort follows PrismML's known-issues page for the ternary model, which recommends it against long reasoning and loops at moderate output limits. Both engines return the thinking in a separate reasoning field.
- **Comparisons**, each one gate report per feature and split, named `expected/<candidate>--vs--<baseline>.<feature>-<split>.json`:

  | Id | Candidate | Baseline | Isolates |
  |---|---|---|---|
  | C0 | `bf16-vllm-rerun` | `bf16-vllm` | nothing: the noise floor of the serving stack |
  | C1 | `q8-llamacpp` | `bf16-vllm` | the engine change, with 8-bit rounding |
  | C2 | `ternary-llamacpp` | `q8-llamacpp` | the ternary model at a fixed engine |
  | C3 | `ternary-llamacpp` | `bf16-vllm` | the whole deployment change |

- **Gate settings:** the defaults: margin 0, ties pass, schema validity ≥ 0.95, think validity ≥ 0.90, alpha 0.05. No other margin was written down before recording, so the reports use margin 0.

## Results

| Id | Feature, split | Items | Baseline | Candidate | Δ (95% CI) | Gains / losses | McNemar p | Gate |
|---|---|---|---|---|---|---|---|---|
| C0 | bugfix, held-out | 74 | 0.986 | 0.986 | +0.000 (−0.041, +0.041) | 1 / 1 | 1.000 | promoted |
| C0 | bugfix, dev | 41 | 1.000 | 0.976 | −0.024 (−0.075, +0.000) | 0 / 1 | 1.000 | rejected (primary) |
| C0 | judge, held-out | 148 | 0.953 | 0.973 | +0.020 (+0.000, +0.047) | 3 / 0 | 0.250 | promoted |
| C0 | judge, dev | 82 | 0.988 | 0.963 | −0.024 (−0.062, +0.000) | 0 / 2 | 0.500 | rejected (primary) |
| C1 | bugfix, held-out | 74 | 0.986 | 1.000 | +0.014 (+0.000, +0.042) | 1 / 0 | 1.000 | promoted |
| C1 | bugfix, dev | 41 | 1.000 | 1.000 | +0.000 | 0 / 0 | 1.000 | promoted |
| C1 | judge, held-out | 148 | 0.953 | 0.966 | +0.014 (−0.013, +0.042) | 3 / 1 | 0.625 | promoted |
| C1 | judge, dev | 82 | 0.988 | 1.000 | +0.012 (+0.000, +0.037) | 1 / 0 | 1.000 | promoted |
| C2 | bugfix, held-out | 74 | 1.000 | 0.959 | −0.041 (−0.095, +0.000) | 0 / 3 | 0.250 | rejected (primary) |
| C2 | bugfix, dev | 41 | 1.000 | 0.976 | −0.024 (−0.075, +0.000) | 0 / 1 | 1.000 | rejected (primary) |
| C2 | judge, held-out | 148 | 0.966 | 0.973 | +0.007 (+0.000, +0.021) | 2 / 1 | 1.000 | promoted |
| C2 | judge, dev | 82 | 1.000 | 1.000 | +0.000 | 0 / 0 | 1.000 | promoted |
| C3 | bugfix, held-out | 74 | 0.986 | 0.959 | −0.027 (−0.069, +0.000) | 0 / 2 | 0.500 | rejected (primary) |
| C3 | bugfix, dev | 41 | 1.000 | 0.976 | −0.024 (−0.075, +0.000) | 0 / 1 | 1.000 | rejected (primary) |
| C3 | judge, held-out | 148 | 0.953 | 0.973 | +0.020 (−0.007, +0.051) | 4 / 1 | 0.375 | promoted |
| C3 | judge, dev | 82 | 0.988 | 1.000 | +0.012 (+0.000, +0.037) | 1 / 0 | 1.000 | promoted |

The primary metric is the resolved rate for `bugfix` and balanced accuracy for `judge`. The intervals come from a paired bootstrap that resamples tasks. Each arm on the held-out bug-fix split:

| Arm | Resolved | Truncated | Mean completion tokens | Median latency (s) |
|---|---|---|---|---|
| `bf16-vllm` | 73/74 = 0.986 | 0/74 | 768 | 6.05 |
| `bf16-vllm-rerun` | 73/74 = 0.986 | 0/74 | 911 | 6.13 |
| `q8-llamacpp` | 74/74 = 1.000 | 0/74 | 722 | 44.10 |
| `ternary-llamacpp` | 71/74 = 0.959 | 1/74 | 1,232 | 53.23 |

The full reports are in `expected/`. Every table of the project report on this experiment is generated from them (`docs/report/tables.md`, "Experiment C").

## What it shows

1. **The noise floor is as large as the effect.**
   - The no-op arm C0 changes nothing but the recording run. On the held-out bug fix it ties at 0.986 with one gain and one loss, so even the arm's single failure moved to another item.
   - The gate still rejects the no-op on two splits: bug fix dev (1 loss) and judge dev (2 losses). The serving stack alone moves one to three items per split, and its held-out bug-fix interval is ±0.041.
   - The ternary model's bug-fix discordance is the same size: 3 items over both splits against bf16 (C3), the same count as the no-op's 3. The difference is direction: the no-op lost 2 and gained 1, the ternary model lost all 3.
2. **The ternary model keeps about 97% of the bf16 resolved rate on the held-out bug fix,** 71/74 against 73/74 (C3: −0.027, 2 losses, p = 0.5). That is close to PrismML's reported 98.2% retention (a 14-benchmark average at their sampling settings, not this suite). Against the engine control (C2) the loss is −0.041 (3 losses, p = 0.25). No comparison in this experiment is significant: McNemar needs at least 6 losses with no gains on these splits.
3. **The engine change costs nothing measurable.** Q8_0 on llama.cpp (C1) loses no bug-fix item on either split and is promoted on all four splits. Its one judge loss is a format failure. So there is no sign that the engine caused the ternary model's losses in C3, and C2 shows the same losses as C3, plus one. At this suite size an engine effect of 2 to 3 items could not be detected, though, and the llama.cpp stack's noise floor was not measured.
4. **What the losses are.** Of the ternary model's three bug-fix losses against bf16, one reply ran to the 16,384-token limit without an answer (`python-132`), one answer was not valid JSON (`python-68`, dev), and one patch was valid JSON but not valid Python (`python-6`). Only the last is a wrong fix, and the reply was still well formed. C2 adds `python-93`, a wrong patch, where bf16 failed too.
5. **The ternary model costs more tokens.** On the held-out bug fix it uses 1.6 times bf16's mean completion tokens (1,232 against 768) and 1.7 times the Q8_0 arm's. On the dev bug fix the ratio is 1.25, and on the judge splits 1.18 to 1.41. It is also the only arm with replies cut at the token limit: one bug-fix and one judge reply on the held-out split.
6. **The judge comparisons say little about the weights.** Every arm rejects every wrong patch except where its reply was invalid (TNR 1.000, and 0.986 for Q8_0 held-out). The judge's mistakes are all false rejections of reference patches, or format failures. The wrong patch is the buggy code behind a visible guard that special-cases the shown inputs ([tasks/humanevalfix/README.md](../../tasks/humanevalfix/README.md), section 8), so it is easy to reject. The judge feature is an exploratory addition: the experiment was designed for the bug fix.
7. **A defensible margin is near the noise floor.** The no-op's held-out intervals reach ±0.041 (bug fix) and +0.047 (judge), and the held-out bug fix's minimum detectable effect at 80% power is 0.053. A margin below about 0.05 rejects changes the suite cannot tell from a rerun. At margin 0.05 every comparison here would pass the primary gate (the lowest Δ is −0.041), which is the honest verdict: on 74 items, a loss of 2 or 3 items is not evidence of a regression.

## Reproduce

Replaying needs no GPU and no model. Pass the suite, since the default is the demo:
```
llm-eval gate --suite tasks/humanevalfix --feature bugfix --split heldout \
  --candidate experiments/exp-c-bonsai/ternary-llamacpp --baseline experiments/exp-c-bonsai/bf16-vllm
```

`--items-out DIR` writes both arms' scored items, with the `problems`, `finish_reason` and `think_words` of each reply; the classification of the discordant items above comes from them. The CI job `experiments` replays all 16 reports.

To record new arms, serve a model on an OpenAI-compatible endpoint and run `llm-eval record --suite tasks/humanevalfix` for each arm and feature, with the settings above (see "Recording new cassettes against a live endpoint" in the main README). The ternary `Q2_0` file needs stock llama.cpp b11556 or newer: older builds load it without warning and produce garbage (PrismML's model card).
