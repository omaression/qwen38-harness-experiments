# Experiment C on the demo suite: the same arms where every arm is near ceiling

This directory replays Experiment C's arms on the repository's own synthetic suite (`tasks/demo`, named in the file `SUITE`) instead of HumanEvalFix. It shows what the gate says when the suite is too easy to separate the arms. The main experiment, with its no-op arm, is in [../exp-c-bonsai](../exp-c-bonsai/README.md). The cassettes here are the models' replies, recorded once. CI replays them, and the `experiments` job checks that the current harness still reproduces the committed reports byte for byte.

## Setup

- **Suite:** `tasks/demo`, 34 synthetic single-function Python tasks, split 21 held-out and 13 dev. The bug fix has 21 and 13 items, the judge 42 and 26.
- **Arms** (no rerun arm on this suite):

  | Arm | Weights | Engine |
  |---|---|---|
  | `bf16-vllm` | stock Qwen3.8-27B, bf16 | vLLM |
  | `q8-llamacpp` | stock Qwen3.8-27B, converted to a Q8_0 GGUF file (the engine control) | llama.cpp b12682 |
  | `ternary-llamacpp` | Ternary Bonsai 2 27B, its published `Q2_0` GGUF file | llama.cpp b12682 |

- **Recording settings,** the same as the main experiment for every arm: temperature 0, seed 0, at most 16,384 completion tokens, thinking on, `reasoning_effort: "medium"`, prompts `bugfix/v2` and `judge/v2`, served model name `qwen38`.
- **Comparisons:** C1 (`q8-llamacpp` against `bf16-vllm`), C2 (`ternary-llamacpp` against `q8-llamacpp`) and C3 (`ternary-llamacpp` against `bf16-vllm`), one report per feature and split in `expected/<candidate>--vs--<baseline>.<feature>-<split>.json`.
- **Gate settings:** the defaults: margin 0, ties pass, schema validity ≥ 0.95, think validity ≥ 0.90, alpha 0.05.

## Results

| Arm | bugfix held-out | bugfix dev | judge held-out | judge dev |
|---|---|---|---|---|
| `bf16-vllm` | 21/21 = 1.000 | 13/13 = 1.000 | 1.000 | 1.000 |
| `q8-llamacpp` | 21/21 = 1.000 | 13/13 = 1.000 | 0.976 | 1.000 |
| `ternary-llamacpp` | 21/21 = 1.000 | 13/13 = 1.000 | 1.000 | 0.962 |

Nine of the twelve comparisons have no discordant item at all, or a single gain, and promote. The other three are rejected, each on one loss:

| Id | Feature, split | Items | Baseline | Candidate | Δ (95% CI) | Gains / losses | McNemar p | Gate |
|---|---|---|---|---|---|---|---|---|
| C1 | judge, held-out | 42 | 1.000 | 0.976 | −0.024 (−0.075, +0.000) | 0 / 1 | 1.000 | rejected (primary) |
| C2 | judge, dev | 26 | 1.000 | 0.962 | −0.038 (−0.115, +0.000) | 0 / 1 | 1.000 | rejected (primary) |
| C3 | judge, dev | 26 | 1.000 | 0.962 | −0.038 (−0.115, +0.000) | 0 / 1 | 1.000 | rejected (primary) |

The full reports are in `expected/`, and every row is in the generated tables of the project report (`docs/report/tables.md`, "Experiment C (demo suite)").

## What it shows

1. **The suite is at ceiling for every arm.** The bf16 model resolves every bug-fix item and judges every patch correctly on both splits. The other two arms miss one judge item each. A comparison against a perfect baseline can only tie or lose, so this suite cannot rank these arms.
2. **Both losses are format failures, not wrong answers.** The Q8_0 arm's loss (`longest-run:wrong`) thought for 1,818 words, outside the 1 to 1,200 word window, and its answer was valid only under the lenient parse. The ternary arm's loss (`parse-int-list:reference`) thought for 4,627 words and was cut at the 16,384-token limit without a verdict.
3. **One truncated reply dominates a mean.** The ternary arm's judge dev mean is 1,227 completion tokens, 2.7 times bf16's 454. The truncated reply alone is 16,384 of its 31,907 tokens; the other 25 replies average 621.
4. **The bug fix does not separate the arms here, but it does on HumanEvalFix.** On this suite all three arms resolve every bug-fix item. On HumanEvalFix the ternary model loses 2 of 74 held-out items against bf16 ([../exp-c-bonsai](../exp-c-bonsai/README.md)). A suite at ceiling hides a difference that a harder suite shows.
5. **These are not Experiment A's base model.** Experiment A's base is a community fine-tune of Qwen3.8-27B recorded without a reasoning-effort setting. Here the stock checkpoint runs at medium effort. The two experiments' demo scores cannot be compared as one model.

## Reproduce

Replaying needs no GPU and no model:
```
llm-eval gate --feature judge --split dev \
  --candidate experiments/exp-c-bonsai-demo/ternary-llamacpp --baseline experiments/exp-c-bonsai-demo/bf16-vllm
```

The demo suite is the default, so `--suite` may be left out. `--items-out DIR` writes the scored items behind the two losses above. To record new arms, follow [../exp-c-bonsai](../exp-c-bonsai/README.md) with `--suite tasks/demo`.
