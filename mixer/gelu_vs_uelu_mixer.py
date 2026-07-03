"""
MLP-Mixer CIFAR-100 experiment for the GELU/UELU computational study.

Examples
--------
Smoke test:
    python gelu_vs_uelu_mixer.py --dry-run

Single-seed paper run:
    python gelu_vs_uelu_mixer.py --epochs 20 --seed 1

Three-seed paper run:
    python gelu_vs_uelu_mixer.py --epochs 20 --seeds 1 2 3
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm


CIFAR100_MEAN = (0.5071, 0.4867, 0.4408)
CIFAR100_STD = (0.2675, 0.2565, 0.2761)
ACTIVATIONS = ("gelu", "relu", "silu", "dgelu", "uelu", "tuelu", "uelu2", "uelu3")


def raw_beta_from_beta(beta: float, beta_min: float) -> float:
    value = beta - beta_min
    if value <= 0:
        raise ValueError("initial beta must be greater than beta_min")
    return math.log(math.expm1(value))


class UELU(nn.Module):
    """Uniform Error Linear Unit with compact threshold support [-beta, beta]."""

    def __init__(self, beta: float = 3.0) -> None:
        super().__init__()
        if beta <= 0:
            raise ValueError("beta must be positive")
        self.beta = float(beta)
        self.inv_2beta = 1.0 / (2.0 * self.beta)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate = (0.5 + x * self.inv_2beta).clamp(0.0, 1.0)
        return x * gate


class TrainableUELU(nn.Module):
    """UELU with a shared trainable positive beta."""

    def __init__(self, raw_beta: nn.Parameter, beta_min: float) -> None:
        super().__init__()
        object.__setattr__(self, "raw_beta", raw_beta)
        self.beta_min = float(beta_min)

    def beta_tensor(self) -> torch.Tensor:
        return self.beta_min + F.softplus(self.raw_beta)

    def beta_value(self) -> float:
        return float(self.beta_tensor().detach().cpu().item())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        beta = self.beta_tensor().to(dtype=x.dtype, device=x.device)
        gate = (0.5 + x / (2.0 * beta)).clamp(0.0, 1.0)
        return x * gate


class DynamicGELU(nn.Module):
    """GELU plus a scheduled centered Gaussian correction."""

    def __init__(self, gamma: float = 1.0) -> None:
        super().__init__()
        self.register_buffer("gamma", torch.tensor(float(gamma)))

    def set_gamma(self, gamma: float) -> None:
        self.gamma.fill_(float(gamma))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        sqrt_two = x.new_tensor(math.sqrt(2.0))
        inv_sqrt_two_pi = x.new_tensor(1.0 / math.sqrt(2.0 * math.pi))
        gelu = 0.5 * x * (1.0 + torch.erf(x / sqrt_two))
        phi = inv_sqrt_two_pi * torch.exp(-0.5 * x * x)
        correction = phi - inv_sqrt_two_pi
        return gelu + self.gamma.to(dtype=x.dtype, device=x.device) * correction


def make_activation(
    name: str,
    uelu_beta: float = 3.0,
    trainable_raw_beta: nn.Parameter | None = None,
    trainable_beta_min: float = 1e-3,
) -> nn.Module:
    if name == "gelu":
        return nn.GELU()
    if name == "relu":
        return nn.ReLU()
    if name == "silu":
        return nn.SiLU()
    if name == "dgelu":
        return DynamicGELU(gamma=1.0)
    if name == "uelu":
        return UELU(beta=uelu_beta)
    if name == "tuelu":
        if trainable_raw_beta is None:
            raise ValueError("tuelu requires a shared trainable beta parameter")
        return TrainableUELU(trainable_raw_beta, trainable_beta_min)
    if name == "uelu2":
        return UELU(beta=2.0)
    if name == "uelu3":
        return UELU(beta=3.0)
    raise ValueError(f"unknown activation: {name}")


def dynamic_gamma_for_progress(progress: float, args: argparse.Namespace) -> float:
    progress = min(1.0, max(0.0, progress))
    if args.dynamic_gamma_schedule == "constant":
        weight = 0.0
    elif args.dynamic_gamma_schedule == "cosine":
        weight = 0.5 * (1.0 - math.cos(math.pi * progress))
    else:
        weight = progress
    return args.dynamic_gamma_start + weight * (args.dynamic_gamma_end - args.dynamic_gamma_start)


def set_dynamic_gelu_gamma(model: nn.Module, gamma: float) -> None:
    for module in model.modules():
        if isinstance(module, DynamicGELU):
            module.set_gamma(gamma)


class MixerBlock(nn.Module):
    """One MLP-Mixer block with pre-normalization."""

    def __init__(
        self,
        num_patches: int,
        hidden_dim: int,
        tokens_mlp_dim: int,
        channels_mlp_dim: int,
        activation: str,
        uelu_beta: float,
        trainable_raw_beta: nn.Parameter | None,
        trainable_beta_min: float,
        dropout: float,
    ) -> None:
        super().__init__()
        self.token_norm = nn.LayerNorm(hidden_dim)
        self.token_mlp = nn.Sequential(
            nn.Linear(num_patches, tokens_mlp_dim),
            make_activation(activation, uelu_beta, trainable_raw_beta, trainable_beta_min),
            nn.Dropout(dropout),
            nn.Linear(tokens_mlp_dim, num_patches),
            nn.Dropout(dropout),
        )
        self.channel_norm = nn.LayerNorm(hidden_dim)
        self.channel_mlp = nn.Sequential(
            nn.Linear(hidden_dim, channels_mlp_dim),
            make_activation(activation, uelu_beta, trainable_raw_beta, trainable_beta_min),
            nn.Dropout(dropout),
            nn.Linear(channels_mlp_dim, hidden_dim),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        token_input = self.token_norm(x).transpose(1, 2)
        x = x + self.token_mlp(token_input).transpose(1, 2)
        x = x + self.channel_mlp(self.channel_norm(x))
        return x


class SmallMixer(nn.Module):
    """Compact MLP-Mixer for 32x32 images."""

    def __init__(
        self,
        activation: str,
        image_size: int = 32,
        patch_size: int = 4,
        num_classes: int = 100,
        hidden_dim: int = 192,
        depth: int = 6,
        tokens_mlp_dim: int = 96,
        channels_mlp_dim: int = 384,
        uelu_beta: float = 3.0,
        trainable_beta_min: float = 1e-3,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        if image_size % patch_size != 0:
            raise ValueError("image_size must be divisible by patch_size")
        num_patches = (image_size // patch_size) ** 2
        patch_dim = 3 * patch_size * patch_size
        self.trainable_beta_min = float(trainable_beta_min)
        self.raw_uelu_beta = (
            nn.Parameter(torch.tensor(raw_beta_from_beta(uelu_beta, trainable_beta_min), dtype=torch.float32))
            if activation == "tuelu"
            else None
        )
        self.patch_size = patch_size
        self.stem = nn.Linear(patch_dim, hidden_dim)
        self.blocks = nn.Sequential(
            *[
                MixerBlock(
                    num_patches=num_patches,
                    hidden_dim=hidden_dim,
                    tokens_mlp_dim=tokens_mlp_dim,
                    channels_mlp_dim=channels_mlp_dim,
                    activation=activation,
                    uelu_beta=uelu_beta,
                    trainable_raw_beta=self.raw_uelu_beta,
                    trainable_beta_min=trainable_beta_min,
                    dropout=dropout,
                )
                for _ in range(depth)
            ]
        )
        self.norm = nn.LayerNorm(hidden_dim)
        self.head = nn.Linear(hidden_dim, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        patches = self._patchify(x)
        x = self.stem(patches)
        x = self.blocks(x)
        x = self.norm(x).mean(dim=1)
        return self.head(x)

    def _patchify(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, channels, height, width = x.shape
        patch = self.patch_size
        x = x.reshape(batch_size, channels, height // patch, patch, width // patch, patch)
        x = x.permute(0, 2, 4, 1, 3, 5)
        return x.reshape(batch_size, -1, channels * patch * patch)


@dataclass
class EpochMetrics:
    activation: str
    seed: int
    epoch: int
    train_loss: float
    train_accuracy: float
    val_loss: float
    val_accuracy: float
    learning_rate: float
    dynamic_gamma: float | None
    trainable_beta: float | None
    seconds: float


@dataclass
class FinalMetrics:
    activation: str
    seed: int
    best_val_accuracy: float
    best_epoch: int
    test_loss: float
    test_accuracy: float
    total_seconds: float
    closed_fraction: float | None
    transition_fraction: float | None
    open_fraction: float | None
    trainable_beta: float | None


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def choose_device(requested: str) -> torch.device:
    if requested != "auto":
        return torch.device(requested)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def make_loaders(args: argparse.Namespace) -> Tuple[DataLoader, DataLoader, DataLoader]:
    train_transform = transforms.Compose(
        [
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(CIFAR100_MEAN, CIFAR100_STD),
        ]
    )
    eval_transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(CIFAR100_MEAN, CIFAR100_STD),
        ]
    )

    data_root = Path(args.data_dir)
    full_train = torchvision.datasets.CIFAR100(data_root, train=True, download=True, transform=train_transform)
    full_val_source = torchvision.datasets.CIFAR100(data_root, train=True, download=False, transform=eval_transform)
    test_set = torchvision.datasets.CIFAR100(data_root, train=False, download=True, transform=eval_transform)

    generator = torch.Generator().manual_seed(args.split_seed)
    shuffled_indices = torch.randperm(len(full_train), generator=generator).tolist()
    val_indices = shuffled_indices[: args.val_size]
    train_indices = shuffled_indices[args.val_size :]

    if args.dry_run:
        train_indices = train_indices[: min(args.dry_run_train_size, len(train_indices))]
        val_indices = val_indices[: min(args.dry_run_val_size, len(val_indices))]
        test_indices = list(range(min(args.dry_run_val_size, len(test_set))))
        test_set = Subset(test_set, test_indices)

    train_set = Subset(full_train, train_indices)
    val_set = Subset(full_val_source, val_indices)

    loader_kwargs = {
        "batch_size": args.batch_size,
        "num_workers": args.num_workers,
        "pin_memory": torch.cuda.is_available(),
    }
    train_loader = DataLoader(train_set, shuffle=True, drop_last=False, **loader_kwargs)
    val_loader = DataLoader(val_set, shuffle=False, drop_last=False, **loader_kwargs)
    test_loader = DataLoader(test_set, shuffle=False, drop_last=False, **loader_kwargs)
    return train_loader, val_loader, test_loader


def accuracy_from_logits(logits: torch.Tensor, targets: torch.Tensor) -> int:
    predictions = logits.argmax(dim=1)
    return int((predictions == targets).sum().item())


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: optim.Optimizer,
    device: torch.device,
    epoch: int,
) -> Tuple[float, float]:
    model.train()
    loss_sum = 0.0
    correct = 0
    total = 0
    progress = tqdm(loader, desc=f"epoch {epoch} train", leave=False)
    for inputs, targets in progress:
        inputs = inputs.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        logits = model(inputs)
        loss = criterion(logits, targets)
        loss.backward()
        optimizer.step()

        batch_size = inputs.size(0)
        loss_sum += float(loss.item()) * batch_size
        correct += accuracy_from_logits(logits.detach(), targets)
        total += batch_size
        progress.set_postfix(loss=f"{loss_sum / total:.4f}", acc=f"{100 * correct / total:.2f}")
    return loss_sum / total, correct / total


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    label: str,
) -> Tuple[float, float]:
    model.eval()
    loss_sum = 0.0
    correct = 0
    total = 0
    progress = tqdm(loader, desc=label, leave=False)
    for inputs, targets in progress:
        inputs = inputs.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        logits = model(inputs)
        loss = criterion(logits, targets)
        batch_size = inputs.size(0)
        loss_sum += float(loss.item()) * batch_size
        correct += accuracy_from_logits(logits, targets)
        total += batch_size
        progress.set_postfix(loss=f"{loss_sum / total:.4f}", acc=f"{100 * correct / total:.2f}")
    return loss_sum / total, correct / total


class ActivationRegionTracker:
    """Counts UELU preactivations in each module's closed, transition, and open regions."""

    def __init__(self, beta: float = 3.0) -> None:
        self.beta = beta
        self.closed = 0
        self.transition = 0
        self.open = 0
        self.handles: List[torch.utils.hooks.RemovableHandle] = []

    def attach(self, model: nn.Module) -> None:
        for module in model.modules():
            if isinstance(module, (UELU, TrainableUELU)):
                self.handles.append(module.register_forward_hook(self._hook))

    def detach(self) -> None:
        for handle in self.handles:
            handle.remove()
        self.handles.clear()

    def _hook(self, module: nn.Module, inputs: Tuple[torch.Tensor, ...], output: torch.Tensor) -> None:
        preactivation = inputs[0].detach()
        beta = module.beta if isinstance(module, UELU) else module.beta_value() if isinstance(module, TrainableUELU) else self.beta
        self.closed += int((preactivation < -beta).sum().item())
        self.transition += int(((preactivation >= -beta) & (preactivation <= beta)).sum().item())
        self.open += int((preactivation > beta).sum().item())

    def fractions(self) -> Tuple[float, float, float]:
        total = self.closed + self.transition + self.open
        if total == 0:
            return math.nan, math.nan, math.nan
        return self.closed / total, self.transition / total, self.open / total


@torch.no_grad()
def measure_activation_regions(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    max_batches: int,
) -> Tuple[float | None, float | None, float | None]:
    tracker = ActivationRegionTracker(beta=3.0)
    tracker.attach(model)
    if not tracker.handles:
        return None, None, None
    model.eval()
    try:
        for batch_index, (inputs, _) in enumerate(loader):
            if batch_index >= max_batches:
                break
            model(inputs.to(device, non_blocking=True))
    finally:
        tracker.detach()
    return tracker.fractions()


def get_trainable_beta(model: nn.Module) -> float | None:
    raw_beta = getattr(model, "raw_uelu_beta", None)
    if raw_beta is None:
        return None
    beta_min = getattr(model, "trainable_beta_min", 1e-3)
    beta = beta_min + F.softplus(raw_beta)
    return float(beta.detach().cpu().item())


def cosine_scheduler(optimizer: optim.Optimizer, epochs: int) -> optim.lr_scheduler.LRScheduler:
    return optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, epochs))


def run_one_setting(
    args: argparse.Namespace,
    activation: str,
    seed: int,
    train_loader: DataLoader,
    val_loader: DataLoader,
    test_loader: DataLoader,
    device: torch.device,
) -> Tuple[List[EpochMetrics], FinalMetrics]:
    seed_everything(seed)
    model = SmallMixer(
        activation=activation,
        hidden_dim=args.hidden_dim,
        depth=args.depth,
        tokens_mlp_dim=args.tokens_mlp_dim,
        channels_mlp_dim=args.channels_mlp_dim,
        uelu_beta=args.uelu_beta,
        trainable_beta_min=args.trainable_beta_min,
        dropout=args.dropout,
    ).to(device)

    criterion = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = cosine_scheduler(optimizer, args.epochs)

    output_dir = Path(args.output_dir)
    checkpoint_path = output_dir / f"best_{activation}_seed{seed}.pt"
    epoch_metrics: List[EpochMetrics] = []
    best_val_accuracy = -1.0
    best_epoch = 0
    start_time = time.perf_counter()

    print(f"\n=== activation={activation} seed={seed} device={device} ===")
    for epoch in range(1, args.epochs + 1):
        dynamic_gamma = None
        if activation == "dgelu":
            progress = (epoch - 1) / max(1, args.epochs - 1)
            dynamic_gamma = dynamic_gamma_for_progress(progress, args)
            set_dynamic_gelu_gamma(model, dynamic_gamma)
        epoch_start = time.perf_counter()
        train_loss, train_accuracy = train_one_epoch(model, train_loader, criterion, optimizer, device, epoch)
        val_loss, val_accuracy = evaluate(model, val_loader, criterion, device, label=f"epoch {epoch} val")
        scheduler.step()
        seconds = time.perf_counter() - epoch_start
        learning_rate = scheduler.get_last_lr()[0]
        trainable_beta = get_trainable_beta(model)

        epoch_metrics.append(
            EpochMetrics(
                activation=activation,
                seed=seed,
                epoch=epoch,
                train_loss=train_loss,
                train_accuracy=train_accuracy,
                val_loss=val_loss,
                val_accuracy=val_accuracy,
                learning_rate=learning_rate,
                dynamic_gamma=dynamic_gamma,
                trainable_beta=trainable_beta,
                seconds=seconds,
            )
        )
        print(
            f"epoch {epoch:03d} | train {100 * train_accuracy:5.2f}% "
            f"val {100 * val_accuracy:5.2f}% | loss {val_loss:.4f} | {seconds:.1f}s"
        )

        if val_accuracy > best_val_accuracy:
            best_val_accuracy = val_accuracy
            best_epoch = epoch
            torch.save(model.state_dict(), checkpoint_path)

    total_seconds = time.perf_counter() - start_time
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    test_loss, test_accuracy = evaluate(model, test_loader, criterion, device, label="test")
    closed, transition, open_fraction = measure_activation_regions(
        model, val_loader, device, max_batches=args.region_batches
    )
    trainable_beta = get_trainable_beta(model)

    final_metrics = FinalMetrics(
        activation=activation,
        seed=seed,
        best_val_accuracy=best_val_accuracy,
        best_epoch=best_epoch,
        test_loss=test_loss,
        test_accuracy=test_accuracy,
        total_seconds=total_seconds,
        closed_fraction=closed,
        transition_fraction=transition,
        open_fraction=open_fraction,
        trainable_beta=trainable_beta,
    )
    return epoch_metrics, final_metrics


def write_csv(path: Path, rows: Iterable[object]) -> None:
    rows = list(rows)
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(asdict(rows[0]).keys()))
        writer.writeheader()
        for row in rows:
            writer.writerow(asdict(row))


def write_summary(path: Path, args: argparse.Namespace, final_rows: List[FinalMetrics]) -> None:
    grouped: Dict[str, List[FinalMetrics]] = {activation: [] for activation in ACTIVATIONS}
    for row in final_rows:
        grouped[row.activation].append(row)

    summary = {
        "config": vars(args),
        "results": [asdict(row) for row in final_rows],
        "means": {},
    }
    for activation, rows in grouped.items():
        if not rows:
            continue
        summary["means"][activation] = {
            "test_accuracy_mean": float(np.mean([row.test_accuracy for row in rows])),
            "test_accuracy_std": float(np.std([row.test_accuracy for row in rows], ddof=0)),
            "best_val_accuracy_mean": float(np.mean([row.best_val_accuracy for row in rows])),
            "seconds_mean": float(np.mean([row.total_seconds for row in rows])),
            "closed_fraction_mean": none_safe_mean([row.closed_fraction for row in rows]),
            "transition_fraction_mean": none_safe_mean([row.transition_fraction for row in rows]),
            "open_fraction_mean": none_safe_mean([row.open_fraction for row in rows]),
            "trainable_beta_mean": none_safe_mean([row.trainable_beta for row in rows]),
        }
    path.write_text(json.dumps(summary, indent=2))


def none_safe_mean(values: Iterable[float | None]) -> float | None:
    real_values = [value for value in values if value is not None and not math.isnan(value)]
    if not real_values:
        return None
    return float(np.mean(real_values))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Controlled GELU/UELU CIFAR-100 Mixer study.")
    parser.add_argument("--activations", nargs="+", choices=ACTIVATIONS, default=list(ACTIVATIONS))
    parser.add_argument("--seeds", nargs="+", type=int, default=[1])
    parser.add_argument("--seed", type=int, default=None, help="Convenience alias for a single seed.")
    parser.add_argument("--split-seed", type=int, default=123)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--uelu-beta", type=float, default=1.0, help="Beta used by the generic 'uelu' activation.")
    parser.add_argument("--trainable-beta-min", type=float, default=1e-3, help="Positive floor for the 'tuelu' trainable beta.")
    parser.add_argument("--dynamic-gamma-start", type=float, default=1.0)
    parser.add_argument("--dynamic-gamma-end", type=float, default=0.0)
    parser.add_argument("--dynamic-gamma-schedule", choices=("linear", "cosine", "constant"), default="linear")
    parser.add_argument("--weight-decay", type=float, default=0.05)
    parser.add_argument("--label-smoothing", type=float, default=0.1)
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--hidden-dim", type=int, default=192)
    parser.add_argument("--depth", type=int, default=6)
    parser.add_argument("--tokens-mlp-dim", type=int, default=96)
    parser.add_argument("--channels-mlp-dim", type=int, default=384)
    parser.add_argument("--val-size", type=int, default=5000)
    parser.add_argument("--region-batches", type=int, default=20)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--device", default="auto", help="auto, cpu, mps, cuda, or cuda:0")
    parser.add_argument("--data-dir", default="./data")
    parser.add_argument("--output-dir", default="./outputs")
    parser.add_argument("--dry-run", action="store_true", help="Run tiny subsets for a fast correctness check.")
    parser.add_argument("--dry-run-train-size", type=int, default=512)
    parser.add_argument("--dry-run-val-size", type=int, default=256)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.seed is not None:
        args.seeds = [args.seed]
    if args.dry_run:
        args.epochs = min(args.epochs, 1)
        args.batch_size = min(args.batch_size, 64)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    device = choose_device(args.device)
    print(f"Using device: {device}")
    print("Activation comparison: " + ", ".join(args.activations))

    train_loader, val_loader, test_loader = make_loaders(args)
    all_epoch_rows: List[EpochMetrics] = []
    final_rows: List[FinalMetrics] = []

    for seed in args.seeds:
        for activation in args.activations:
            epoch_rows, final_metrics = run_one_setting(
                args=args,
                activation=activation,
                seed=seed,
                train_loader=train_loader,
                val_loader=val_loader,
                test_loader=test_loader,
                device=device,
            )
            all_epoch_rows.extend(epoch_rows)
            final_rows.append(final_metrics)
            print(
                f"final {activation} seed={seed}: test={100 * final_metrics.test_accuracy:.2f}% "
                f"best_val={100 * final_metrics.best_val_accuracy:.2f}% "
                f"best_epoch={final_metrics.best_epoch}"
            )
            if final_metrics.transition_fraction is not None:
                print(
                    f"{activation.upper()} regions: "
                    f"closed={100 * final_metrics.closed_fraction:.2f}% "
                    f"transition={100 * final_metrics.transition_fraction:.2f}% "
                    f"open={100 * final_metrics.open_fraction:.2f}%"
                )

    write_csv(output_dir / "epoch_metrics.csv", all_epoch_rows)
    write_csv(output_dir / "final_metrics.csv", final_rows)
    write_summary(output_dir / "summary.json", args, final_rows)
    print(f"Wrote results to {output_dir.resolve()}")


if __name__ == "__main__":
    main()
