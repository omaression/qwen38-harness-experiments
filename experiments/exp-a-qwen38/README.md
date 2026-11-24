# Experiment A: model-side changes to an open-weight Qwen model, under the gate

This experiment puts real model-side changes through the harness: LoRA adapters on a 27B open-weight model. The cassettes in this directory are the model's replies, recorded once. CI replays them, and the `experiments` job checks that the current harness still reproduces the committed reports byte for byte.

## Setup

- **Base model:** Qwen3.8-27B, a community abliterated fine-tune, in bf16. It was served by vLLM on two A100 GPUs with thinking on (`enable_thinking: true`), temperature 0, seed 0 and at most 16,384 completion tokens. The thinking arrives in the server's separate reasoning field, so the replies scored here are the final answers.
- **Arms**, both on the same server:

  | Arm | `bugfix` feature | `judge` feature |
  |---|---|---|
  | `qwen38-base` | the base model | the base model |
  | `qwen38-adapters` | an **identity** LoRA adapter (rank 64, all `lora_B` weights zero, so a no-op) | a **trained** judge LoRA adapter (rank 64, trained on the base model's own rejection-sampled verdicts on public code-review data) |

- **Data:** the repository's own 34-task synthetic suite (`tasks/demo`), on both splits.
- **Gate settings:** the defaults: margin 0, ties pass, schema validity ≥ 0.95, think validity ≥ 0.90, alpha 0.05.

## Results

| Feature, split | Items | Base | Adapters | Δ (95% CI) | Gains / losses | McNemar p | Gate |
|---|---|---|---|---|---|---|---|
| bugfix, held-out | 21 | 1.000 | 0.952 | −0.048 (−0.15, 0.00) | 0 / 1 | 1.00 | rejected (primary) |
| bugfix, dev | 13 | 0.923 | 0.846 | −0.077 (−0.23, 0.00) | 0 / 1 | 1.00 | rejected (primary, schema, think) |
| judge, held-out | 42 | 1.000 | 0.976 | −0.024 (−0.08, 0.00) | 0 / 1 | 1.00 | rejected (primary) |
| judge, dev | 26 | 0.808 | **0.962** | **+0.154 (+0.04, +0.27)** | 4 / 0 | 0.125 | **promoted** |

The primary metric is the resolved rate for `bugfix` and balanced accuracy for `judge`. The intervals come from a paired bootstrap that resamples tasks. The full reports are in `expected/`.

## What it shows

1. **A no-op change is not a no-op at serving time.**
   - The identity adapter cannot change a single output in exact arithmetic.
   - Served through the LoRA kernels, alongside speculative decoding and batching, it produced one different reply on each `bugfix` split, and each of those replies failed.
   - A gate with margin 0 then rejects an adapter that changes nothing. The comparison itself says the loss is noise: McNemar p = 1.0, and the report's warning says a regression needs at least 6 losses to be significant on 21 items.
   - Before a real comparison, record a no-op arm to measure this floor. Then set `--margin` from it, or add the significance guard.
2. **A trained adapter's gain shows only where the base model has room to move.**
   - The judge adapter turns 4 dev-split mistakes of the base model into correct verdicts, with no loss, and passes every gate there.
   - On the held-out split the base model is already perfect (1.000). There the adapter can only tie or lose, and one changed reply makes it lose.
   - A held-out split at ceiling cannot show an improvement. The suite needs harder items before it can rank these two arms.
3. **The suite is small, and the report says so.** The minimum detectable effect at 80% power is 13 points on the held-out `bugfix` split and 6 points on the held-out `judge` split. Gains smaller than that are invisible here, whatever the verdict.

## Reproduce

Replaying needs no GPU and no model:
```
llm-eval gate --feature judge --split dev \
  --candidate experiments/exp-a-qwen38/qwen38-adapters --baseline experiments/exp-a-qwen38/qwen38-base
```

To record new arms, serve a model on an OpenAI-compatible endpoint and run `llm-eval record` for each arm and feature, with the settings above (see "Recording new cassettes against a live endpoint" in the main README).
