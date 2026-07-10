# GELU / UELU Computational Study

R. Rossi, "A Structural Interpretation of GELU and Threshold-Transmission Activations via the First-Order Loss Function," [arXiv:2607.03664](https://arxiv.org/abs/2607.03664) [cs.LG]

This folder contains the five experiment modules used by the manuscript's `Computational study` section and a root orchestration script.

The study compares standard `GELU` with `ReLU`, `SiLU`/Swish, fixed-width `UELU`, fixed-width `UELU` at the conventional hard-swish width (`beta=3`), trainable-width `TUELU`, and centered dynamic GELU (`DGELU`). It follows the manuscript's threshold-transmission framing: GELU is treated as the Gaussian signal-transmission term, while UELU/TUELU provide compact uniform-threshold analogues.

The five experiments are:

- MLP-Mixer on CIFAR-100 (`mixer`)
- compact Vision Transformer on CIFAR-100 (`vit`, `vit_cifar100`)
- tiny character-level GPT on Tiny Shakespeare (`gpt`, `shakespeare`, `tiny_gpt_text`)
- token-level GPT on TinyStories (`tinystories`, `tiny_gpt_TinyStories`)
- token-level GPT on WikiText-2 (`wikitext2`, `wikitext-2`, `tiny_gpt_WikiText-2`)

## One-shot study runner

Print the commands without running them:

```bash
python run_computational_study.py --print-only --seeds 1 2 3
```

Run the shorter protocol check. This uses the runner's `exploratory` profile name, but the experiments are still the controlled experiments from the paper:

```bash
python run_computational_study.py --profile exploratory --seeds 1 2 3
```

Run only selected experiments:

```bash
python run_computational_study.py --experiments vit gpt tinystories --profile exploratory --seeds 1 2 3
```

Run the final paper comparison:

```bash
python run_computational_study.py --profile final --seeds 1 2 3
```

Valid short experiment names are `mixer`, `vit`, `gpt`, `tinystories`, and `wikitext2`. The aliases `shakespeare` and `wikitext-2` also work, as do the full names `vit_cifar100`, `tiny_gpt_text`, `tiny_gpt_TinyStories`, and `tiny_gpt_WikiText-2`.

Run tiny smoke tests through all modules and generate aggregate outputs:

```bash
python run_computational_study.py --dry-run-modules --seeds 1 --output-root ./study_outputs_dry_run --num-workers 0
```

Summarize existing outputs without rerunning experiments:

```bash
python run_computational_study.py --skip-runs --output-root ./study_outputs
```

## Paper Protocol

The experimental protocol is aligned with the manuscript's computational-study section:

- Mixer / CIFAR-100: 20 epochs, beta initialized at 1.0.
- ViT / CIFAR-100: 20 epochs, beta initialized at 1.0.
- Tiny GPT / Tiny Shakespeare: 10000 iterations, beta initialized at 0.5.
- TinyStories GPT: 10000 iterations, beta initialized at 0.5.
- WikiText-2 GPT: 10000 iterations, beta initialized at 0.5.
- Activations: `relu`, `silu`, `gelu`, `dgelu`, `uelu`, `uelu3`, `tuelu`.

The `uelu3` alias is fixed-width UELU with `beta=3.0`, corresponding to the conventional hard-swish transition width. The generic `uelu` activation still uses the architecture-specific beta from the protocol above.

The runner's default `--profile exploratory` keeps the same 20-epoch vision budget but runs the GPT experiments for 5000 iterations as a shorter protocol check. Use `--profile final` for the manuscript comparison reported in the paper.

## Outputs

By default, outputs are written to `./study_outputs`:

- `aggregate_final_metrics.csv`
- `tables/*.tex`
- `figures/*.png`
- per-module output folders with raw metrics and checkpoints:
	- `mixer/`
	- `vit_cifar100/`
	- `tiny_gpt_text/`
	- `tiny_gpt_TinyStories/`
	- `tiny_gpt_WikiText-2/`

The LaTeX table fragments in `tables/` are designed to support the computational-study tables in `main.tex` after final runs are complete.
