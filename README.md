<img width="128" height="128" align="right" alt="Logo" src="https://github.com/user-attachments/assets/df5f2840-2f92-4991-aa57-252747d7182e" />

# Heretic Enhanced: Multi-Token KL + False Refusal Detection

[![Discord](https://img.shields.io/discord/1447831134212984903?color=5865F2&label=discord&labelColor=black&logo=discord&logoColor=white&style=for-the-badge)](https://discord.gg/gdXc48gSyT) [![Follow us on Hugging Face](https://huggingface.co/datasets/huggingface/badges/resolve/main/follow-us-on-hf-md-dark.svg)](https://huggingface.co/Kewk)

A fork of [p-e-w/heretic](https://github.com/p-e-w/heretic) with enhanced optimization features for higher-quality abliteration. All changes are backwards-compatible — existing workflows work identically unless you opt in to the new features.

## What's New

### Multi-Token KL Divergence (`--kl-tokens N`)

The upstream Heretic measures KL divergence using only the first generated token. This is noisy — token 1 is often a generic starter like "I" or "The" where distributions barely differ between the original and abliterated model.

This fork generates **N tokens** and averages KL divergence across all positions, giving a much more robust quality signal.

```bash
heretic --model Qwen/Qwen3.5-9B --kl-tokens 3
```

- **Recommended N:** 3 (sweet spot between signal quality and speed)
- **5** is slightly better but diminishing returns
- **>5** not worth the compute cost
- KL thresholds (`kl_divergence_scale`, `kl_divergence_target`) are automatically scaled by N

### False Refusal Detection (`--detect-false-refusals`)

High KL divergence is a proxy for model damage, but doesn't directly catch the worst symptom: the model refusing **harmless** prompts it shouldn't refuse. This feature checks for over-abliteration by running the good evaluation prompts through the refusal classifier.

```bash
heretic --model Qwen/Qwen3.5-9B --detect-false-refusals --false-refusal-weight 0.5
```

- False refusals are penalized in the KL divergence component: `penalty = weight * (false_refusals / total_good_prompts)`
- **Default weight:** 0.5 — higher values (1.0+) aggressively punish any false refusal
- Adds ~10-15% overhead per trial (one extra generation pass on good prompts)

### Widened max_weight Range

The `max_weight` search range was widened from upstream's `[0.8, 1.5]` to `[0.8, 2.5]`. This was the key breakthrough that broke through the 70/100 refusal ceiling on Qwen3.5 models — stronger ablation weights allow the optimizer to fully suppress refusal behavior in hybrid DeltaNet+Attention architectures.

### Auto-Continue Mode (`--auto-continue`)

For headless or non-interactive runs, `--auto-continue` automatically resumes from the checkpoint without prompting.

### Real-Time Monitor (`monitor.py`)

A terminal dashboard that watches heretic output in real time:

```bash
# In one terminal, run heretic with logging:
heretic --model Qwen/Qwen3.5-9B ... 2>&1 | tee heretic_output.log

# In another terminal:
python monitor.py
```

Shows GPU stats, batch test progress, trial-by-trial results, and the current Pareto front.

## Published Models

All models abliterated using this fork on an RTX 4090 with `--orthogonalize-direction`:

| Model | Params | Refusals | KL Divergence | HuggingFace |
| :--- | ---: | ---: | ---: | :--- |
| **Heretical-Qwen3.5-9B** | 9.1B | 3/100 | 0.0366 | [Kewk/Heretical-Qwen3.5-9B](https://huggingface.co/Kewk/Heretical-Qwen3.5-9B) |
| **Heretical-Qwen3.5-4B** | 4.0B | 4/100 | 0.0574 | [Kewk/Heretical-Qwen3.5-4B](https://huggingface.co/Kewk/Heretical-Qwen3.5-4B) |
| **Heretical-Qwen3.5-2B** | 2.0B | 2/100 | 0.0259 | [Kewk/Heretical-Qwen3.5-2B](https://huggingface.co/Kewk/Heretical-Qwen3.5-2B) |
| **Heretical-Qwen3.5-0.8B** | 0.8B | 5/100 | 0.0224 | [Kewk/Heretical-Qwen3.5-0.8B](https://huggingface.co/Kewk/Heretical-Qwen3.5-0.8B) |

These are the **first abliterated Qwen3.5 models** — a hybrid DeltaNet+Attention architecture that required the widened max_weight range to achieve proper decensoring. All models use LoRA adapters for minimal size overhead.

## Quick Start

```bash
# Install
pip install -U heretic-llm

# Basic (same as upstream)
heretic Qwen/Qwen3.5-9B

# With all enhancements
PYTHONUTF8=1 PYTHONIOENCODING=utf-8 CUDA_VISIBLE_DEVICES=0 \
heretic \
  --model Qwen/Qwen3.5-9B \
  --quantization bnb_4bit \
  --n-trials 400 \
  --n-startup-trials 80 \
  --orthogonalize-direction \
  --kl-tokens 3 \
  --detect-false-refusals

# Evaluate an existing model against its base
heretic --model Qwen/Qwen3.5-9B --evaluate-model Kewk/Heretical-Qwen3.5-9B
```

**Note:** On Windows, always set `PYTHONUTF8=1 PYTHONIOENCODING=utf-8` to prevent Rich console crashes with cp1252 encoding.

## Recommended Configuration by Model Size

| Model Size | Quantization | Trials | KL Tokens | False Refusals | Notes |
| :--- | :--- | ---: | ---: | :--- | :--- |
| < 3B | none | 200 | 3 | yes | Fits in VRAM unquantized |
| 3B - 9B | bnb_4bit | 400 | 3 | yes | Sweet spot for RTX 4090 |
| 9B - 14B | bnb_4bit | 400 | 1-3 | optional | Longer trials, 3 may be slow |
| > 14B | bnb_4bit | 200 | 1 | no | Minimize overhead |

## New CLI Options

| Flag | Default | Description |
| :--- | :--- | :--- |
| `--kl-tokens N` | 1 | Tokens to generate for KL measurement (1 = upstream behavior) |
| `--detect-false-refusals` | off | Check if abliteration causes benign prompt refusals |
| `--false-refusal-weight F` | 0.5 | Penalty weight for false refusals (higher = stricter) |
| `--auto-continue` | off | Skip interactive prompts, resume from checkpoint |

## How It Works

This fork extends Heretic's bi-objective optimization (minimize KL divergence + minimize refusals) with:

1. **Multi-token KL:** Instead of comparing first-token logprobs, generates N tokens and computes KL divergence across all positions. The logprobs tensor is reshaped from `(prompts, vocab)` to `(prompts * N, vocab)` so `batchmean` reduction naturally averages across positions. All thresholds are scaled by N to maintain consistent optimization behavior.

2. **False refusal detection:** After computing the standard score, runs the good evaluation prompts through `count_refusals()`. If any benign prompts are refused (over-abliteration), a penalty is added to the KL divergence score component: `kld_score += weight * (false_refusals / total_good_prompts)`. This steers the optimizer away from parameter combinations that damage general capability.

3. **Widened search space:** `max_weight` range `[0.8, 2.5]` (upstream: `[0.8, 1.5]`) allows stronger ablation that some architectures (especially hybrid attention) require.

## Upstream

This fork is based on [Heretic v1.2.0](https://github.com/p-e-w/heretic) by Philipp Emanuel Weidmann. All upstream features (research plots, residual geometry, projected abliteration, etc.) are fully preserved.

For upstream documentation, research features, and technical details about how abliteration works, see the [upstream README](https://github.com/p-e-w/heretic#readme).

## License

AGPL-3.0-or-later — same as upstream. See [LICENSE](LICENSE) for details.
