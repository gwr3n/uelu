# Tiny Character-Level GPT Controlled Study

This folder contains the third controlled experiment for the paper's computational study. It compares `GELU` with `ReLU`, `SiLU`/Swish, fixed-width `UELU`, trainable-width `TUELU`, and centered dynamic GELU (`DGELU`) in the feedforward blocks of a small decoder-only Transformer language model.

## Quick Start

Smoke test:

```bash
python gelu_vs_uelu_tiny_gpt.py --dry-run
```

Single-seed protocol check:

```bash
python gelu_vs_uelu_tiny_gpt.py --max-iters 5000 --seed 1
```

Three-seed final comparison:

```bash
python gelu_vs_uelu_tiny_gpt.py --max-iters 10000 --seeds 1 2 3
```

Diagnostic comparison with a chosen compact width:

```bash
python gelu_vs_uelu_tiny_gpt.py --activations gelu uelu --uelu-beta 0.5 --max-iters 10000 --seeds 1 2 3
```

Comparison including dynamic GELU:

```bash
python gelu_vs_uelu_tiny_gpt.py --activations gelu dgelu uelu --uelu-beta 0.5 --max-iters 10000 --seeds 1 2 3
```

Trainable-beta UELU:

```bash
python gelu_vs_uelu_tiny_gpt.py --activations gelu tuelu --uelu-beta 0.5 --max-iters 10000 --seeds 1 2 3
```

## Outputs

Outputs are written to `./outputs` by default:

- `eval_metrics.csv`
- `final_metrics.csv`
- `summary.json`
- best checkpoints for each activation and seed

For UELU variants, the script records the fraction of feedforward preactivations below `-beta`, inside `[-beta, beta]`, and above `beta`. Use `--activations uelu --uelu-beta VALUE` for new beta sweeps; `uelu2` and `uelu3` remain as reproducibility aliases. Use `tuelu` to learn one shared positive beta initialized from `--uelu-beta`; the learned value is written to the metrics files.
For `DGELU`, the script records the scheduled `gamma` value in `eval_metrics.csv`. By default, the centered correction `gamma * (phi(z) - phi(0))` is linearly annealed from `gamma = 1` to `gamma = 0` over training.

## Notes

This is not intended to be a modern large language model benchmark. It is a small controlled probe of the activation substitution in Transformer feedforward layers.
