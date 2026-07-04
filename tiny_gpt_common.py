"""Shared token-level GPT utilities for GELU/UELU language studies."""

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
from tqdm import tqdm


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


class CausalSelfAttention(nn.Module):
    def __init__(self, n_embd: int, n_head: int, block_size: int, dropout: float) -> None:
        super().__init__()
        if n_embd % n_head != 0:
            raise ValueError("n_embd must be divisible by n_head")
        self.n_head = n_head
        self.head_dim = n_embd // n_head
        self.query_key_value = nn.Linear(n_embd, 3 * n_embd)
        self.proj = nn.Linear(n_embd, n_embd)
        self.attn_dropout = nn.Dropout(dropout)
        self.resid_dropout = nn.Dropout(dropout)
        mask = torch.tril(torch.ones(block_size, block_size)).view(1, 1, block_size, block_size)
        self.register_buffer("mask", mask)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, n_embd = x.shape
        qkv = self.query_key_value(x)
        query, key, value = qkv.split(n_embd, dim=2)
        query = query.view(batch_size, seq_len, self.n_head, self.head_dim).transpose(1, 2)
        key = key.view(batch_size, seq_len, self.n_head, self.head_dim).transpose(1, 2)
        value = value.view(batch_size, seq_len, self.n_head, self.head_dim).transpose(1, 2)

        scores = (query @ key.transpose(-2, -1)) / math.sqrt(self.head_dim)
        scores = scores.masked_fill(self.mask[:, :, :seq_len, :seq_len] == 0, float("-inf"))
        weights = F.softmax(scores, dim=-1)
        weights = self.attn_dropout(weights)
        y = weights @ value
        y = y.transpose(1, 2).contiguous().view(batch_size, seq_len, n_embd)
        return self.resid_dropout(self.proj(y))


class FeedForward(nn.Module):
    def __init__(
        self,
        n_embd: int,
        activation: str,
        uelu_beta: float,
        trainable_raw_beta: nn.Parameter | None,
        trainable_beta_min: float,
        dropout: float,
    ) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            make_activation(activation, uelu_beta, trainable_raw_beta, trainable_beta_min),
            nn.Linear(4 * n_embd, n_embd),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class Block(nn.Module):
    def __init__(
        self,
        n_embd: int,
        n_head: int,
        block_size: int,
        activation: str,
        uelu_beta: float,
        trainable_raw_beta: nn.Parameter | None,
        trainable_beta_min: float,
        dropout: float,
    ) -> None:
        super().__init__()
        self.ln1 = nn.LayerNorm(n_embd)
        self.attn = CausalSelfAttention(n_embd, n_head, block_size, dropout)
        self.ln2 = nn.LayerNorm(n_embd)
        self.ffn = FeedForward(n_embd, activation, uelu_beta, trainable_raw_beta, trainable_beta_min, dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.ln1(x))
        x = x + self.ffn(self.ln2(x))
        return x


class TokenGPT(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        block_size: int,
        activation: str,
        n_layer: int = 6,
        n_head: int = 6,
        n_embd: int = 384,
        uelu_beta: float = 0.5,
        trainable_beta_min: float = 1e-3,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.block_size = block_size
        self.trainable_beta_min = float(trainable_beta_min)
        self.raw_uelu_beta = (
            nn.Parameter(torch.tensor(raw_beta_from_beta(uelu_beta, trainable_beta_min), dtype=torch.float32))
            if activation == "tuelu"
            else None
        )
        self.token_embedding = nn.Embedding(vocab_size, n_embd)
        self.position_embedding = nn.Embedding(block_size, n_embd)
        self.dropout = nn.Dropout(dropout)
        self.blocks = nn.Sequential(
            *[
                Block(
                    n_embd,
                    n_head,
                    block_size,
                    activation,
                    uelu_beta,
                    self.raw_uelu_beta,
                    trainable_beta_min,
                    dropout,
                )
                for _ in range(n_layer)
            ]
        )
        self.ln_f = nn.LayerNorm(n_embd)
        self.head = nn.Linear(n_embd, vocab_size, bias=False)
        self.token_embedding.weight = self.head.weight
        self.apply(self._init_weights)

    def _init_weights(self, module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx: torch.Tensor, targets: torch.Tensor | None = None):
        batch_size, seq_len = idx.shape
        if seq_len > self.block_size:
            raise ValueError("sequence length exceeds block_size")
        positions = torch.arange(0, seq_len, device=idx.device)
        x = self.token_embedding(idx) + self.position_embedding(positions)
        x = self.dropout(x)
        x = self.blocks(x)
        x = self.ln_f(x)
        logits = self.head(x)
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
        return logits, loss


class BPETokenDataset:
    def __init__(
        self,
        train_text: str,
        val_text: str,
        device: torch.device,
        encoding_name: str = "bpe8k",
        tokenizer_dir: Path | None = None,
        train_fraction: float | None = None,
    ) -> None:
        self.encoding_name = encoding_name
        self.tokenizer_dir = tokenizer_dir
        if not val_text:
            encoded = self.encode(train_text)
            split = int((train_fraction or 0.9) * len(encoded))
            train_tokens = encoded[:split]
            val_tokens = encoded[split:]
        else:
            train_tokens = self.encode(train_text)
            val_tokens = self.encode(val_text)
        self.train_data = torch.tensor(train_tokens, dtype=torch.long, device=device)
        self.val_data = torch.tensor(val_tokens, dtype=torch.long, device=device)

    @property
    def vocab_size(self) -> int:
        if self.encoding_name == "gpt2":
            return self.encoding.n_vocab
        return self.tokenizer.get_vocab_size()

    def encode(self, text: str) -> List[int]:
        if self.encoding_name == "gpt2":
            if not hasattr(self, "encoding"):
                try:
                    import tiktoken
                except ImportError as exc:
                    raise ImportError("Install tiktoken to use --encoding gpt2.") from exc
                self.encoding = tiktoken.get_encoding("gpt2")
            return self.encoding.encode(text)
        if not hasattr(self, "tokenizer"):
            self.tokenizer = self.load_or_train_bpe_tokenizer(text)
        return self.tokenizer.encode(text).ids

    def load_or_train_bpe_tokenizer(self, train_text: str):
        if self.encoding_name == "bpe8k":
            vocab_size = 8000
        elif self.encoding_name == "bpe16k":
            vocab_size = 16000
        else:
            raise ValueError("encoding must be one of: bpe8k, bpe16k, gpt2")
        try:
            from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers
        except ImportError as exc:
            raise ImportError("Install tokenizers to use --encoding bpe8k or --encoding bpe16k.") from exc

        tokenizer_dir = self.tokenizer_dir or Path("./data/tokenizers")
        tokenizer_dir.mkdir(parents=True, exist_ok=True)
        tokenizer_path = tokenizer_dir / f"byte_bpe_{vocab_size}.json"
        if tokenizer_path.exists():
            return Tokenizer.from_file(str(tokenizer_path))

        tokenizer = Tokenizer(models.BPE(unk_token="<unk>"))
        tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
        tokenizer.decoder = decoders.ByteLevel()
        trainer = trainers.BpeTrainer(
            vocab_size=vocab_size,
            min_frequency=2,
            special_tokens=["<unk>"],
            initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
        )
        tokenizer.train_from_iterator([train_text], trainer=trainer)
        tokenizer.save(str(tokenizer_path))
        return tokenizer

    def get_batch(self, split: str, batch_size: int, block_size: int) -> Tuple[torch.Tensor, torch.Tensor]:
        data = self.train_data if split == "train" else self.val_data
        if len(data) <= block_size + 1:
            raise ValueError(f"{split} split has {len(data)} tokens, but block_size={block_size}")
        starts = torch.randint(0, len(data) - block_size - 1, (batch_size,), device=data.device)
        x = torch.stack([data[start : start + block_size] for start in starts])
        y = torch.stack([data[start + 1 : start + block_size + 1] for start in starts])
        return x, y


@dataclass
class EvalMetrics:
    activation: str
    seed: int
    iteration: int
    train_loss: float
    val_loss: float
    val_perplexity: float
    learning_rate: float
    dynamic_gamma: float | None
    trainable_beta: float | None
    seconds: float


@dataclass
class FinalMetrics:
    activation: str
    seed: int
    best_val_loss: float
    best_iteration: int
    final_train_loss: float
    final_val_loss: float
    final_val_perplexity: float
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


@torch.no_grad()
def estimate_loss(model: TokenGPT, dataset: BPETokenDataset, args: argparse.Namespace) -> Tuple[float, float]:
    model.eval()
    losses: Dict[str, List[float]] = {"train": [], "val": []}
    for split in ("train", "val"):
        for _ in range(args.eval_iters):
            xb, yb = dataset.get_batch(split, args.batch_size, args.block_size)
            _, loss = model(xb, yb)
            losses[split].append(float(loss.item()))
    model.train()
    return float(np.mean(losses["train"])), float(np.mean(losses["val"]))


class ActivationRegionTracker:
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
        beta = module.beta if isinstance(module, UELU) else module.beta_value()
        self.closed += int((preactivation < -beta).sum().item())
        self.transition += int(((preactivation >= -beta) & (preactivation <= beta)).sum().item())
        self.open += int((preactivation > beta).sum().item())

    def fractions(self) -> Tuple[float, float, float]:
        total = self.closed + self.transition + self.open
        if total == 0:
            return math.nan, math.nan, math.nan
        return self.closed / total, self.transition / total, self.open / total


@torch.no_grad()
def measure_activation_regions(model: nn.Module, dataset: BPETokenDataset, args: argparse.Namespace):
    tracker = ActivationRegionTracker(beta=3.0)
    tracker.attach(model)
    if not tracker.handles:
        return None, None, None
    model.eval()
    try:
        for _ in range(args.region_batches):
            xb, _ = dataset.get_batch("val", args.batch_size, args.block_size)
            model(xb)
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


def learning_rate_for_iter(iteration: int, args: argparse.Namespace) -> float:
    if iteration < args.warmup_iters:
        return args.lr * iteration / max(1, args.warmup_iters)
    progress = (iteration - args.warmup_iters) / max(1, args.max_iters - args.warmup_iters)
    cosine = 0.5 * (1.0 + math.cos(math.pi * min(1.0, progress)))
    return args.min_lr + cosine * (args.lr - args.min_lr)


def run_one_setting(args: argparse.Namespace, activation: str, seed: int, dataset: BPETokenDataset, device: torch.device):
    seed_everything(seed)
    model = TokenGPT(
        vocab_size=dataset.vocab_size,
        block_size=args.block_size,
        activation=activation,
        n_layer=args.n_layer,
        n_head=args.n_head,
        n_embd=args.n_embd,
        uelu_beta=args.uelu_beta,
        trainable_beta_min=args.trainable_beta_min,
        dropout=args.dropout,
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    output_dir = Path(args.output_dir)
    checkpoint_path = output_dir / f"best_{activation}_seed{seed}.pt"
    eval_rows: List[EvalMetrics] = []
    best_val_loss = float("inf")
    best_iteration = 0
    final_train_loss = float("nan")
    final_val_loss = float("nan")
    start_time = time.perf_counter()

    print(f"\n=== activation={activation} seed={seed} device={device} ===")
    progress = tqdm(range(1, args.max_iters + 1), desc=f"train {activation} seed {seed}")
    for iteration in progress:
        dynamic_gamma = None
        if activation == "dgelu":
            progress_fraction = (iteration - 1) / max(1, args.max_iters - 1)
            dynamic_gamma = dynamic_gamma_for_progress(progress_fraction, args)
            set_dynamic_gelu_gamma(model, dynamic_gamma)
        lr = learning_rate_for_iter(iteration, args)
        for group in optimizer.param_groups:
            group["lr"] = lr
        xb, yb = dataset.get_batch("train", args.batch_size, args.block_size)
        _, loss = model(xb, yb)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        optimizer.step()

        if iteration == 1 or iteration % args.eval_interval == 0 or iteration == args.max_iters:
            elapsed = time.perf_counter() - start_time
            train_loss, val_loss = estimate_loss(model, dataset, args)
            trainable_beta = get_trainable_beta(model)
            final_train_loss = train_loss
            final_val_loss = val_loss
            val_perplexity = float(math.exp(min(20.0, val_loss)))
            eval_rows.append(
                EvalMetrics(
                    activation=activation,
                    seed=seed,
                    iteration=iteration,
                    train_loss=train_loss,
                    val_loss=val_loss,
                    val_perplexity=val_perplexity,
                    learning_rate=lr,
                    dynamic_gamma=dynamic_gamma,
                    trainable_beta=trainable_beta,
                    seconds=elapsed,
                )
            )
            progress.set_postfix(train=f"{train_loss:.3f}", val=f"{val_loss:.3f}")
            print(
                f"iter {iteration:05d} | train_loss {train_loss:.4f} "
                f"val_loss {val_loss:.4f} ppl {val_perplexity:.2f}"
            )
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_iteration = iteration
                torch.save(model.state_dict(), checkpoint_path)

    total_seconds = time.perf_counter() - start_time
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    final_train_loss, final_val_loss = estimate_loss(model, dataset, args)
    closed, transition, open_fraction = measure_activation_regions(model, dataset, args)
    trainable_beta = get_trainable_beta(model)
    final = FinalMetrics(
        activation=activation,
        seed=seed,
        best_val_loss=best_val_loss,
        best_iteration=best_iteration,
        final_train_loss=final_train_loss,
        final_val_loss=final_val_loss,
        final_val_perplexity=float(math.exp(min(20.0, final_val_loss))),
        total_seconds=total_seconds,
        closed_fraction=closed,
        transition_fraction=transition,
        open_fraction=open_fraction,
        trainable_beta=trainable_beta,
    )
    return eval_rows, final


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


def none_safe_mean(values: Iterable[float | None]) -> float | None:
    real_values = [value for value in values if value is not None and not math.isnan(value)]
    if not real_values:
        return None
    return float(np.mean(real_values))


def write_summary(path: Path, args: argparse.Namespace, final_rows: List[FinalMetrics]) -> None:
    summary = {"config": vars(args), "results": [asdict(row) for row in final_rows], "means": {}}
    for activation in ACTIVATIONS:
        rows = [row for row in final_rows if row.activation == activation]
        if not rows:
            continue
        summary["means"][activation] = {
            "final_val_loss_mean": float(np.mean([row.final_val_loss for row in rows])),
            "final_val_loss_std": float(np.std([row.final_val_loss for row in rows], ddof=0)),
            "final_val_perplexity_mean": float(np.mean([row.final_val_perplexity for row in rows])),
            "best_val_loss_mean": float(np.mean([row.best_val_loss for row in rows])),
            "seconds_mean": float(np.mean([row.total_seconds for row in rows])),
            "closed_fraction_mean": none_safe_mean([row.closed_fraction for row in rows]),
            "transition_fraction_mean": none_safe_mean([row.transition_fraction for row in rows]),
            "open_fraction_mean": none_safe_mean([row.open_fraction for row in rows]),
            "trainable_beta_mean": none_safe_mean([row.trainable_beta for row in rows]),
        }
    path.write_text(json.dumps(summary, indent=2))


def add_common_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--activations", nargs="+", choices=ACTIVATIONS, default=list(ACTIVATIONS))
    parser.add_argument("--seeds", nargs="+", type=int, default=[1])
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--max-iters", type=int, default=10000)
    parser.add_argument("--eval-interval", type=int, default=500)
    parser.add_argument("--eval-iters", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--block-size", type=int, default=256)
    parser.add_argument("--n-layer", type=int, default=6)
    parser.add_argument("--n-head", type=int, default=6)
    parser.add_argument("--n-embd", type=int, default=384)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--uelu-beta", type=float, default=0.5, help="Beta used by the generic 'uelu' activation.")
    parser.add_argument("--trainable-beta-min", type=float, default=1e-3, help="Positive floor for the 'tuelu' trainable beta.")
    parser.add_argument("--dynamic-gamma-start", type=float, default=1.0)
    parser.add_argument("--dynamic-gamma-end", type=float, default=0.0)
    parser.add_argument("--dynamic-gamma-schedule", choices=("linear", "cosine", "constant"), default="linear")
    parser.add_argument("--min-lr", type=float, default=3e-5)
    parser.add_argument("--warmup-iters", type=int, default=100)
    parser.add_argument("--weight-decay", type=float, default=0.1)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--train-fraction", type=float, default=0.9)
    parser.add_argument("--region-batches", type=int, default=20)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--data-dir", default="./data")
    parser.add_argument(
        "--encoding",
        choices=("bpe8k", "bpe16k", "gpt2"),
        default="bpe8k",
        help="Tokenizer vocabulary: trained byte-level BPE 8k/16k, or GPT-2's 50k vocabulary.",
    )
    parser.add_argument("--output-dir", default="./outputs")
    parser.add_argument("--dry-run", action="store_true")


def apply_dry_run_overrides(args: argparse.Namespace) -> None:
    if not args.dry_run:
        return
    args.max_iters = min(args.max_iters, 20)
    args.eval_interval = min(args.eval_interval, 10)
    args.eval_iters = min(args.eval_iters, 3)
    args.batch_size = min(args.batch_size, 8)
    args.block_size = min(args.block_size, 64)
    args.n_layer = min(args.n_layer, 2)
    args.n_head = min(args.n_head, 2)
    args.n_embd = min(args.n_embd, 64)


def run_study(args: argparse.Namespace, dataset: BPETokenDataset) -> None:
    if args.seed is not None:
        args.seeds = [args.seed]
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    device = choose_device(args.device)
    print(f"Using device: {device}")
    print("Activation comparison: " + ", ".join(args.activations))
    print(
        f"Loaded {len(dataset.train_data):,} train tokens and {len(dataset.val_data):,} validation tokens "
        f"with {args.encoding} vocabulary size {dataset.vocab_size}"
    )

    all_eval_rows: List[EvalMetrics] = []
    final_rows: List[FinalMetrics] = []
    for seed in args.seeds:
        for activation in args.activations:
            eval_rows, final = run_one_setting(args, activation, seed, dataset, device)
            all_eval_rows.extend(eval_rows)
            final_rows.append(final)
            print(
                f"final {activation} seed={seed}: val_loss={final.final_val_loss:.4f} "
                f"ppl={final.final_val_perplexity:.2f} best_iter={final.best_iteration}"
            )
            if final.transition_fraction is not None:
                print(
                    f"{activation.upper()} regions: "
                    f"closed={100 * final.closed_fraction:.2f}% "
                    f"transition={100 * final.transition_fraction:.2f}% "
                    f"open={100 * final.open_fraction:.2f}%"
                )
    write_csv(output_dir / "eval_metrics.csv", all_eval_rows)
    write_csv(output_dir / "final_metrics.csv", final_rows)
    write_summary(output_dir / "summary.json", args, final_rows)
    print(f"Wrote results to {output_dir.resolve()}")