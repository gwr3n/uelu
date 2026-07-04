"""
Run and summarize the GELU/UELU computational study.

This script orchestrates the five experiment modules used by the manuscript's
Computational study section:
1. MLP-Mixer on CIFAR-100
2. compact Vision Transformer on CIFAR-100
3. tiny character-level GPT on Tiny Shakespeare
4. token-level GPT on TinyStories
5. token-level GPT on WikiText-2

The default protocol is the exploratory/reproducible protocol described in the
paper draft: 20 epochs for the vision models and 5000 iterations for GPT. Use
--profile final to keep the same vision budget but run GPT for 10000 iterations.
The paper protocol includes the conventional hard-swish-width UELU case via
the `uelu3` activation alias.

Outputs are written under ./study_outputs by default:

    study_outputs/
        mixer/
        vit_cifar100/
        tiny_gpt_text/
        tiny_gpt_TinyStories/
        tiny_gpt_WikiText-2/
        aggregate_final_metrics.csv
        tables/
        figures/

The tables directory contains LaTeX tabular fragments that can replace the
placeholder tables in main.tex once the final runs are complete. The figures
directory contains learning curves, beta trajectories, and UELU region
occupancy charts.
"""

from __future__ import annotations

import argparse
import csv
import math
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Sequence

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parent
EXPERIMENTS = ("mixer", "vit_cifar100", "tiny_gpt_text", "tiny_gpt_TinyStories", "tiny_gpt_WikiText-2")
EXPERIMENT_ALIASES = {
    "mixer": "mixer",
    "vit": "vit_cifar100",
    "vit_cifar100": "vit_cifar100",
    "gpt": "tiny_gpt_text",
    "shakespeare": "tiny_gpt_text",
    "tiny_gpt_text": "tiny_gpt_text",
    "tinystories": "tiny_gpt_TinyStories",
    "tiny_gpt_TinyStories": "tiny_gpt_TinyStories",
    "wikitext2": "tiny_gpt_WikiText-2",
    "wikitext-2": "tiny_gpt_WikiText-2",
    "tiny_gpt_WikiText-2": "tiny_gpt_WikiText-2",
}


@dataclass(frozen=True)
class ExperimentConfig:
    name: str
    folder: Path
    script: str
    output_dir: Path
    beta: float
    activations: Sequence[str]
    metric_kind: str


@dataclass(frozen=True)
class Protocol:
    vision_epochs: int
    gpt_iters: int
    gpt_eval_interval: int


def protocol_from_profile(profile: str) -> Protocol:
    if profile == "final":
        return Protocol(vision_epochs=20, gpt_iters=10000, gpt_eval_interval=500)
    return Protocol(vision_epochs=20, gpt_iters=5000, gpt_eval_interval=500)


PLOT_STYLES = {
    "dgelu": {"marker": "D"},
    "gelu": {"marker": "s"},
    "relu": {"marker": "^"},
    "silu": {"marker": "v"},
    "tuelu": {"marker": "P"},
    "uelu": {"marker": "X"},
    "uelu3": {"marker": "*"},
}
FIGURE_DPI = 600
REGION_ACTIVATIONS = ("uelu", "tuelu", "uelu3")
REGION_ACTIVATION_LABELS = {
    "uelu": "UELU",
    "tuelu": "TUELU",
    "uelu3": "UELU3",
}


def curve_style(activation: str) -> Dict[str, object]:
    style = PLOT_STYLES.get(activation, {"marker": "o"})
    return {
        **style,
        "linestyle": "-",
        "linewidth": 1.1,
        "markersize": 3.2,
        "markeredgewidth": 0.8,
    }


def build_experiment_configs(output_root: Path, activations: Sequence[str]) -> List[ExperimentConfig]:
    return [
        ExperimentConfig(
            name="mixer",
            folder=ROOT / "mixer",
            script="gelu_vs_uelu_mixer.py",
            output_dir=output_root / "mixer",
            beta=1.0,
            activations=activations,
            metric_kind="classification",
        ),
        ExperimentConfig(
            name="vit_cifar100",
            folder=ROOT / "vit_cifar100",
            script="gelu_vs_uelu_vit.py",
            output_dir=output_root / "vit_cifar100",
            beta=1.0,
            activations=activations,
            metric_kind="classification",
        ),
        ExperimentConfig(
            name="tiny_gpt_text",
            folder=ROOT / "tiny_gpt_text",
            script="gelu_vs_uelu_tiny_gpt.py",
            output_dir=output_root / "tiny_gpt_text",
            beta=0.5,
            activations=activations,
            metric_kind="language",
        ),
        ExperimentConfig(
            name="tiny_gpt_TinyStories",
            folder=ROOT / "tiny_gpt_TinyStories",
            script="gelu_vs_uelu_tinystories_gpt.py",
            output_dir=output_root / "tiny_gpt_TinyStories",
            beta=0.5,
            activations=activations,
            metric_kind="language",
        ),
        ExperimentConfig(
            name="tiny_gpt_WikiText-2",
            folder=ROOT / "tiny_gpt_WikiText-2",
            script="gelu_vs_uelu_wikitext2_gpt.py",
            output_dir=output_root / "tiny_gpt_WikiText-2",
            beta=0.5,
            activations=activations,
            metric_kind="language",
        ),
    ]


def select_experiment_configs(configs: Sequence[ExperimentConfig], requested: Sequence[str]) -> List[ExperimentConfig]:
    selected_names = []
    for name in requested:
        if name not in EXPERIMENT_ALIASES:
            valid = ", ".join(sorted(EXPERIMENT_ALIASES))
            raise ValueError(f"unknown experiment '{name}'. Choose from: {valid}")
        selected_names.append(EXPERIMENT_ALIASES[name])
    selected = [config for config in configs if config.name in selected_names]
    if not selected:
        raise ValueError("at least one experiment must be selected")
    return selected


def run_command(command: Sequence[str], cwd: Path, dry_run: bool) -> None:
    printable = " ".join(command)
    print(f"\n[{cwd.name}] {printable}")
    if dry_run:
        return
    subprocess.run(command, cwd=cwd, check=True)


def run_experiment(config: ExperimentConfig, args: argparse.Namespace, protocol: Protocol) -> None:
    command = [
        sys.executable,
        config.script,
        "--activations",
        *config.activations,
        "--seeds",
        *[str(seed) for seed in args.seeds],
        "--uelu-beta",
        str(config.beta),
        "--output-dir",
        str(config.output_dir),
        "--device",
        args.device,
    ]

    if config.metric_kind == "classification":
        command.extend([
            "--epochs",
            str(protocol.vision_epochs),
            "--num-workers",
            str(args.num_workers),
        ])
    else:
        command.extend([
            "--max-iters",
            str(protocol.gpt_iters),
            "--eval-interval",
            str(protocol.gpt_eval_interval),
        ])

    if args.dry_run_modules:
        command.append("--dry-run")

    run_command(command, config.folder, args.print_only)


def read_csv(path: Path) -> List[Dict[str, str]]:
    if not path.exists():
        return []
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: Iterable[Dict[str, object]]) -> None:
    rows = list(rows)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("")
        return
    fieldnames = sorted({fieldname for row in rows for fieldname in row.keys()})
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def as_float(value: object) -> float | None:
    if value is None:
        return None
    text = str(value).strip()
    if text == "" or text.lower() == "nan":
        return None
    try:
        return float(text)
    except ValueError:
        return None


def mean_std(values: Iterable[float | None]) -> tuple[float | None, float | None]:
    real_values = [value for value in values if value is not None and not math.isnan(value)]
    if not real_values:
        return None, None
    return float(np.mean(real_values)), float(np.std(real_values, ddof=0))


def percent(value: float | None) -> str:
    if value is None:
        return "--"
    return f"{100.0 * value:.2f}"


def number(value: float | None, digits: int = 4) -> str:
    if value is None:
        return "--"
    return f"{value:.{digits}f}"


def summarize_experiment(config: ExperimentConfig) -> List[Dict[str, object]]:
    final_rows = [row for row in read_csv(config.output_dir / "final_metrics.csv") if row.get("activation") in config.activations]
    aggregate_rows: List[Dict[str, object]] = []
    for activation in sorted({row["activation"] for row in final_rows}):
        rows = [row for row in final_rows if row["activation"] == activation]
        if config.metric_kind == "classification":
            primary_values = [as_float(row.get("test_accuracy")) for row in rows]
            val_values = [as_float(row.get("best_val_accuracy")) for row in rows]
            loss_values = [as_float(row.get("test_loss")) for row in rows]
            primary_name = "test_accuracy"
            val_name = "best_val_accuracy"
            loss_name = "test_loss"
        else:
            primary_values = [as_float(row.get("final_val_perplexity")) for row in rows]
            val_values = [as_float(row.get("final_val_loss")) for row in rows]
            loss_values = [as_float(row.get("best_val_loss")) for row in rows]
            primary_name = "final_val_perplexity"
            val_name = "final_val_loss"
            loss_name = "best_val_loss"

        primary_mean, primary_std = mean_std(primary_values)
        val_mean, val_std = mean_std(val_values)
        loss_mean, loss_std = mean_std(loss_values)
        closed_mean, closed_std = mean_std(as_float(row.get("closed_fraction")) for row in rows)
        transition_mean, transition_std = mean_std(as_float(row.get("transition_fraction")) for row in rows)
        open_mean, open_std = mean_std(as_float(row.get("open_fraction")) for row in rows)
        beta_mean, beta_std = mean_std(as_float(row.get("trainable_beta")) for row in rows)

        aggregate_rows.append(
            {
                "experiment": config.name,
                "activation": activation,
                "initial_beta": initial_beta_for_activation(activation, config.beta),
                f"{primary_name}_mean": primary_mean,
                f"{primary_name}_std": primary_std,
                f"{val_name}_mean": val_mean,
                f"{val_name}_std": val_std,
                f"{loss_name}_mean": loss_mean,
                f"{loss_name}_std": loss_std,
                "closed_fraction_mean": closed_mean,
                "closed_fraction_std": closed_std,
                "transition_fraction_mean": transition_mean,
                "transition_fraction_std": transition_std,
                "open_fraction_mean": open_mean,
                "open_fraction_std": open_std,
                "trainable_beta_mean": beta_mean,
                "trainable_beta_std": beta_std,
            }
        )
    return aggregate_rows


def initial_beta_for_activation(activation: str, configured_beta: float) -> float | str:
    if activation in {"uelu", "tuelu"}:
        return configured_beta
    if activation == "uelu2":
        return 2.0
    if activation == "uelu3":
        return 3.0
    return ""


def display_activation_name(activation: object) -> str:
    text = str(activation)
    if text in {"uelu2", "uelu3"}:
        return "UELU"
    return text.upper()


def latex_table_for_experiment(config: ExperimentConfig, rows: List[Dict[str, object]]) -> str:
    lines: List[str] = []
    if config.metric_kind == "classification":
        lines.extend(
            [
                "\\begin{tabular}{lllllll}",
                "\\toprule",
                    "Activation & $\\beta$ & Learned $\\beta$ & Val. acc. & Test acc. & Test loss & Regions C/T/O \\\\",
                "\\midrule",
            ]
        )
        for row in rows:
            activation = display_activation_name(row["activation"])
            beta = str(row["initial_beta"]) if row["initial_beta"] != "" else "--"
            learned_beta = number(row.get("trainable_beta_mean"), 3)
            val_acc = percent(row.get("best_val_accuracy_mean"))
            test_acc = percent(row.get("test_accuracy_mean"))
            test_loss = number(row.get("test_loss_mean"), 4)
            regions = "/".join(
                [
                    percent(row.get("closed_fraction_mean")),
                    percent(row.get("transition_fraction_mean")),
                    percent(row.get("open_fraction_mean")),
                ]
            )
            lines.append(f"{activation} & {beta} & {learned_beta} & {val_acc} & {test_acc} & {test_loss} & {regions} \\\\")
    else:
        lines.extend(
            [
                "\\begin{tabular}{llllll}",
                "\\toprule",
                    "Activation & $\\beta$ & Learned $\\beta$ & Val. loss & Perplexity & Regions C/T/O \\\\",
                "\\midrule",
            ]
        )
        for row in rows:
            activation = display_activation_name(row["activation"])
            beta = str(row["initial_beta"]) if row["initial_beta"] != "" else "--"
            learned_beta = number(row.get("trainable_beta_mean"), 3)
            val_loss = number(row.get("final_val_loss_mean"), 4)
            perplexity = number(row.get("final_val_perplexity_mean"), 3)
            regions = "/".join(
                [
                    percent(row.get("closed_fraction_mean")),
                    percent(row.get("transition_fraction_mean")),
                    percent(row.get("open_fraction_mean")),
                ]
            )
            lines.append(f"{activation} & {beta} & {learned_beta} & {val_loss} & {perplexity} & {regions} \\\\")
    lines.extend(["\\bottomrule", "\\end{tabular}", ""])
    return "\n".join(lines)


def plot_classification_curves(config: ExperimentConfig, figure_dir: Path) -> None:
    rows = [row for row in read_csv(config.output_dir / "epoch_metrics.csv") if row.get("activation") in config.activations]
    if not rows:
        return
    figure_dir.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(7.0, 4.2))
    for activation in sorted({row["activation"] for row in rows}):
        subset = [row for row in rows if row["activation"] == activation]
        by_epoch: Dict[int, List[float]] = {}
        for row in subset:
            by_epoch.setdefault(int(row["epoch"]), []).append(float(row["val_accuracy"]))
        epochs = sorted(by_epoch)
        means = [np.mean(by_epoch[epoch]) for epoch in epochs]
        plt.plot(epochs, means, label=activation.upper(), **curve_style(activation))
    plt.xlabel("Epoch")
    plt.ylabel("Validation accuracy (log scale)")
    plt.yscale("log")
    plt.title(config.name.replace("_", " "))
    plt.grid(alpha=0.25)
    plt.legend()
    plt.tight_layout()
    plt.savefig(figure_dir / f"{config.name}_validation_accuracy.png", dpi=FIGURE_DPI)
    plt.close()

    beta_rows = [row for row in rows if as_float(row.get("trainable_beta")) is not None]
    if beta_rows:
        plt.figure(figsize=(7.0, 4.2))
        for activation in sorted({row["activation"] for row in beta_rows}):
            subset = [row for row in beta_rows if row["activation"] == activation]
            by_epoch: Dict[int, List[float]] = {}
            for row in subset:
                by_epoch.setdefault(int(row["epoch"]), []).append(float(row["trainable_beta"]))
            epochs = sorted(by_epoch)
            means = [np.mean(by_epoch[epoch]) for epoch in epochs]
            plt.plot(epochs, means, label=activation.upper(), **curve_style(activation))
        plt.xlabel("Epoch")
        plt.ylabel("Learned beta")
        plt.title(f"{config.name.replace('_', ' ')} learned beta")
        plt.grid(alpha=0.25)
        plt.legend()
        plt.tight_layout()
        plt.savefig(figure_dir / f"{config.name}_learned_beta.png", dpi=FIGURE_DPI)
        plt.close()


def plot_language_curves(config: ExperimentConfig, figure_dir: Path) -> None:
    rows = [row for row in read_csv(config.output_dir / "eval_metrics.csv") if row.get("activation") in config.activations]
    if not rows:
        return
    figure_dir.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(7.0, 4.2))
    plotted_values: List[float] = []
    post_initial_values: List[float] = []
    for activation in sorted({row["activation"] for row in rows}):
        subset = [row for row in rows if row["activation"] == activation]
        by_iter: Dict[int, List[float]] = {}
        for row in subset:
            by_iter.setdefault(int(row["iteration"]), []).append(float(row["val_perplexity"]))
        iterations = sorted(by_iter)
        means = [np.mean(by_iter[iteration]) for iteration in iterations]
        if len(iterations) > 1:
            iterations = iterations[1:]
            means = means[1:]
        plotted_values.extend(float(value) for value in means)
        post_initial_values.extend(float(value) for value in means)
        plt.plot(iterations, means, label=activation.upper(), **curve_style(activation))
    plt.xlabel("Iteration")
    plt.ylabel("Validation perplexity (log scale)")
    plt.yscale("log")
    if post_initial_values and plotted_values:
        top = min(max(plotted_values), max(post_initial_values) * 1.25)
        bottom = min(plotted_values) * 0.9
        if bottom > 0 and top > bottom:
            plt.ylim(bottom=bottom, top=top)
    plt.title(config.name.replace("_", " ").replace("-", "-"))
    plt.grid(alpha=0.25)
    plt.legend()
    plt.tight_layout()
    plt.savefig(figure_dir / f"{config.name}_perplexity.png", dpi=FIGURE_DPI)
    plt.close()

    beta_rows = [row for row in rows if as_float(row.get("trainable_beta")) is not None]
    if beta_rows:
        plt.figure(figsize=(7.0, 4.2))
        for activation in sorted({row["activation"] for row in beta_rows}):
            subset = [row for row in beta_rows if row["activation"] == activation]
            by_iter: Dict[int, List[float]] = {}
            for row in subset:
                by_iter.setdefault(int(row["iteration"]), []).append(float(row["trainable_beta"]))
            iterations = sorted(by_iter)
            means = [np.mean(by_iter[iteration]) for iteration in iterations]
            plt.plot(iterations, means, label=activation.upper(), **curve_style(activation))
        plt.xlabel("Iteration")
        plt.ylabel("Learned beta")
        plt.title(f"{config.name.replace('_', ' ')} learned beta")
        plt.grid(alpha=0.25)
        plt.legend()
        plt.tight_layout()
        plt.savefig(figure_dir / f"{config.name}_learned_beta.png", dpi=FIGURE_DPI)
        plt.close()


def plot_region_occupancy(configs: Sequence[ExperimentConfig], figure_dir: Path) -> None:
    grouped_rows: Dict[str, List[Dict[str, object]]] = {}
    for config in configs:
        rows = read_csv(config.output_dir / "final_metrics.csv")
        for activation in REGION_ACTIVATIONS:
            activation_rows = [row for row in rows if row.get("activation") == activation]
            closed, _ = mean_std(as_float(row.get("closed_fraction")) for row in activation_rows)
            transition, _ = mean_std(as_float(row.get("transition_fraction")) for row in activation_rows)
            open_fraction, _ = mean_std(as_float(row.get("open_fraction")) for row in activation_rows)
            if closed is None or transition is None or open_fraction is None:
                continue
            grouped_rows.setdefault(config.name, []).append(
                {
                    "label": REGION_ACTIVATION_LABELS[activation],
                    "closed": closed,
                    "transition": transition,
                    "open": open_fraction,
                }
            )
    grouped_rows = {name: rows for name, rows in grouped_rows.items() if rows}
    if not grouped_rows:
        return
    figure_dir.mkdir(parents=True, exist_ok=True)

    experiment_names = [config.name for config in configs if config.name in grouped_rows]
    figure, axes = plt.subplots(
        nrows=len(experiment_names),
        ncols=1,
        figsize=(7.2, 2.2 * len(experiment_names)),
        sharey=True,
    )
    if len(experiment_names) == 1:
        axes = [axes]

    display_names = {
        "mixer": "MLP-Mixer",
        "vit_cifar100": "ViT",
        "tiny_gpt_text": "Tiny Shakespeare GPT",
        "tiny_gpt_TinyStories": "TinyStories GPT",
        "tiny_gpt_WikiText-2": "WikiText-2 GPT",
    }

    for axis, experiment_name in zip(axes, experiment_names):
        rows = grouped_rows[experiment_name]
        labels = [str(row["label"]) for row in rows]
        closed = np.array([float(row["closed"]) for row in rows])
        transition = np.array([float(row["transition"]) for row in rows])
        open_fraction = np.array([float(row["open"]) for row in rows])
        x = np.arange(len(rows))

        axis.bar(x, closed, label="Closed")
        axis.bar(x, transition, bottom=closed, label="Transition")
        axis.bar(x, open_fraction, bottom=closed + transition, label="Open")
        axis.set_xticks(x)
        axis.set_xticklabels(labels)
        axis.set_ylim(0, 1)
        axis.set_ylabel(display_names.get(experiment_name, experiment_name))
        axis.grid(axis="y", alpha=0.2)

    axes[-1].set_xlabel("Activation")
    figure.supylabel("Fraction")
    handles, labels = axes[0].get_legend_handles_labels()
    figure.legend(handles, labels, loc="upper center", ncol=3, bbox_to_anchor=(0.5, 0.97))
    plt.tight_layout()
    plt.subplots_adjust(top=0.86)
    plt.savefig(figure_dir / "uelu_region_occupancy.png", dpi=FIGURE_DPI)
    plt.close()


def summarize_outputs(configs: Sequence[ExperimentConfig], output_root: Path) -> None:
    table_dir = output_root / "tables"
    figure_dir = output_root / "figures"
    table_dir.mkdir(parents=True, exist_ok=True)
    figure_dir.mkdir(parents=True, exist_ok=True)

    aggregate_rows: List[Dict[str, object]] = []
    for config in configs:
        rows = summarize_experiment(config)
        aggregate_rows.extend(rows)
        (table_dir / f"{config.name}_results.tex").write_text(latex_table_for_experiment(config, rows))
        if config.metric_kind == "classification":
            plot_classification_curves(config, figure_dir)
        else:
            plot_language_curves(config, figure_dir)

    write_csv(output_root / "aggregate_final_metrics.csv", aggregate_rows)
    plot_region_occupancy(configs, figure_dir)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the GELU/UELU computational study and generate paper-ready summaries.")
    parser.add_argument("--profile", choices=("exploratory", "final"), default="exploratory")
    parser.add_argument(
        "--experiments",
        nargs="+",
        default=list(EXPERIMENTS),
        help="Subset to run/summarize: mixer, vit, gpt, tinystories, wikitext2. Full experiment names also work.",
    )
    parser.add_argument("--seeds", nargs="+", type=int, default=[1, 2, 3])
    parser.add_argument("--activations", nargs="+", default=["relu", "silu", "gelu", "dgelu", "uelu", "uelu3", "tuelu"])
    parser.add_argument("--output-root", default="./study_outputs")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--skip-runs", action="store_true", help="Only summarize existing module outputs.")
    parser.add_argument("--dry-run-modules", action="store_true", help="Pass --dry-run to each experiment module.")
    parser.add_argument("--print-only", action="store_true", help="Print commands without executing them.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_root = (ROOT / args.output_root).resolve()
    protocol = protocol_from_profile(args.profile)
    configs = select_experiment_configs(build_experiment_configs(output_root, args.activations), args.experiments)

    output_root.mkdir(parents=True, exist_ok=True)
    if not args.skip_runs:
        for config in configs:
            run_experiment(config, args, protocol)

    if not args.print_only:
        summarize_outputs(configs, output_root)
        print(f"\nWrote aggregate outputs to {output_root}")
        print(f"Tables:  {output_root / 'tables'}")
        print(f"Figures: {output_root / 'figures'}")


if __name__ == "__main__":
    main()
