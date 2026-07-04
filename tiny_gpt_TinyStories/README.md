# TinyStories Token-Level GPT Controlled Study

This folder contains a stronger language-modeling extension of the paper's Tiny Shakespeare experiment. It compares `GELU` with `ReLU`, `SiLU`/Swish, fixed-width `UELU`, trainable-width `TUELU`, and centered dynamic GELU (`DGELU`) in the feedforward blocks of a compact decoder-only Transformer trained on TinyStories.

## Quick Start

Smoke test:

```bash
python gelu_vs_uelu_tinystories_gpt.py --dry-run --activations gelu uelu
```

Single-seed protocol check:

```bash
python gelu_vs_uelu_tinystories_gpt.py --max-iters 5000 --seed 1
```

Three-seed final comparison:

```bash
python gelu_vs_uelu_tinystories_gpt.py --max-iters 10000 --seeds 1 2 3
```

Use local text files instead of downloading from Hugging Face:

```bash
python gelu_vs_uelu_tinystories_gpt.py --train-text-file train.txt --val-text-file validation.txt
```

Choose a tokenizer vocabulary:

```bash
python gelu_vs_uelu_tinystories_gpt.py --encoding bpe8k
python gelu_vs_uelu_tinystories_gpt.py --encoding bpe16k
python gelu_vs_uelu_tinystories_gpt.py --encoding gpt2
```

## Outputs

Outputs are written to `./outputs` by default:

- `eval_metrics.csv`
- `final_metrics.csv`
- `summary.json`
- best checkpoints for each activation and seed

For UELU variants, the script records the fraction of feedforward preactivations below `-beta`, inside `[-beta, beta]`, and above `beta`. Use `--activations uelu --uelu-beta VALUE` for beta sweeps; `uelu2` and `uelu3` remain as reproducibility aliases. Use `tuelu` to learn one shared positive beta initialized from `--uelu-beta`.