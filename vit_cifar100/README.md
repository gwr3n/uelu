# Compact ViT CIFAR-100 Controlled Study

This folder contains the second controlled experiment for the paper's computational study. It compares `GELU` with `ReLU`, `SiLU`/Swish, fixed-width `UELU`, trainable-width `TUELU`, and centered dynamic GELU (`DGELU`) in a compact Vision Transformer trained on CIFAR-100.

## Quick Start

Smoke test:

```bash
python gelu_vs_uelu_vit.py --dry-run --num-workers 0
```

Single-seed paper run:

```bash
python gelu_vs_uelu_vit.py --epochs 20 --seed 1
```

Three-seed paper run:

```bash
python gelu_vs_uelu_vit.py --epochs 20 --seeds 1 2 3
```

Diagnostic comparison with a chosen compact width:

```bash
python gelu_vs_uelu_vit.py --activations gelu uelu --uelu-beta 1.0 --epochs 20 --seeds 1 2 3
```

Comparison including dynamic GELU:

```bash
python gelu_vs_uelu_vit.py --activations gelu dgelu uelu --uelu-beta 1.0 --epochs 20 --seeds 1 2 3
```

Trainable-beta UELU:

```bash
python gelu_vs_uelu_vit.py --activations gelu tuelu --uelu-beta 1.0 --epochs 20 --seeds 1 2 3
```

## Outputs

Outputs are written to `./outputs` by default:

- `epoch_metrics.csv`
- `final_metrics.csv`
- `summary.json`
- best checkpoints for each activation and seed

For UELU variants, the script records the fraction of activation inputs below `-beta`, inside `[-beta, beta]`, and above `beta`. Use `--activations uelu --uelu-beta VALUE` for new beta sweeps; `uelu2` and `uelu3` remain as reproducibility aliases. Use `tuelu` to learn one shared positive beta initialized from `--uelu-beta`; the learned value is written to the metrics files.
For `DGELU`, the script records the scheduled `gamma` value in `epoch_metrics.csv`. By default, the centered correction `gamma * (phi(z) - phi(0))` is linearly annealed from `gamma = 1` to `gamma = 0` over training.
