# MLP-Mixer CIFAR-100 Controlled Study

This folder contains the first controlled experiment for the paper's computational study.

The experiment compares standard `GELU` with `ReLU`, `SiLU`/Swish, fixed-width `UELU`, trainable-width `TUELU`, and centered dynamic GELU (`DGELU`). The generic `uelu` activation uses the CLI parameter `--uelu-beta`, so beta can be swept without changing code. The `tuelu` activation uses the same initial beta but learns one shared positive beta for the whole model. `DGELU` implements a centered dynamic correction, annealing `gamma * (phi(z) - phi(0))` from `gamma = 1` to `gamma = 0` by default.

## Quick Start

From this folder:

```bash
python gelu_vs_uelu_mixer.py --dry-run
```

A single-seed paper run:

```bash
python gelu_vs_uelu_mixer.py --epochs 20 --seed 1
```

A full paper-protocol comparison:

```bash
python gelu_vs_uelu_mixer.py --epochs 20 --seeds 1 2 3
```

Diagnostic comparison with a chosen compact width:

```bash
python gelu_vs_uelu_mixer.py --activations gelu uelu --uelu-beta 1.0 --epochs 20 --seeds 1 2 3
```

Comparison including dynamic GELU:

```bash
python gelu_vs_uelu_mixer.py --activations gelu dgelu uelu --uelu-beta 1.0 --epochs 20 --seeds 1 2 3
```

Trainable-beta UELU:

```bash
python gelu_vs_uelu_mixer.py --activations gelu tuelu --uelu-beta 1.0 --epochs 20 --seeds 1 2 3
```

On Apple Silicon, the script will use MPS when available. On NVIDIA hardware, it will use CUDA when available.

## Outputs

By default, outputs are written to `./outputs`:

- `epoch_metrics.csv`
- `final_metrics.csv`
- `summary.json`
- best-model checkpoints for each activation/seed

## Notes

The aliases `uelu2` and `uelu3` are retained for reproducibility, but new beta sweeps should use `--activations uelu --uelu-beta VALUE`.
Use `tuelu` to learn one shared positive beta initialized from `--uelu-beta`; the learned value is written to the metrics files.
Use `DGELU` to test whether retaining a centered Gaussian correction early in training improves optimization before annealing back to ordinary GELU.
