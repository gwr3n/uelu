"""
TinyStories token-level GPT experiment for the GELU/UELU computational study.

Examples
--------
Smoke test:
    python gelu_vs_uelu_tinystories_gpt.py --dry-run --activations gelu uelu

Single-seed protocol check:
    python gelu_vs_uelu_tinystories_gpt.py --max-iters 5000 --seed 1

Three-seed final comparison:
    python gelu_vs_uelu_tinystories_gpt.py --max-iters 10000 --seeds 1 2 3
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tiny_gpt_common import BPETokenDataset, add_common_args, apply_dry_run_overrides, choose_device, run_study


def load_tinystories_text(args: argparse.Namespace) -> tuple[str, str]:
    if args.train_text_file:
        train_text = Path(args.train_text_file).read_text(encoding="utf-8")
        val_text = Path(args.val_text_file).read_text(encoding="utf-8") if args.val_text_file else ""
        return train_text, val_text
    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise ImportError("Install datasets to download TinyStories, or pass --train-text-file.") from exc

    data_dir = Path(args.data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)
    train_cache = data_dir / "tinystories_train.txt"
    val_cache = data_dir / "tinystories_validation.txt"
    if train_cache.exists() and val_cache.exists() and not args.refresh_cache:
        return train_cache.read_text(encoding="utf-8"), val_cache.read_text(encoding="utf-8")

    print("Loading TinyStories from Hugging Face datasets")
    train_split = load_dataset("roneneldan/TinyStories", split=args.train_split, cache_dir=str(data_dir / "hf_cache"))
    val_split = load_dataset("roneneldan/TinyStories", split=args.val_split, cache_dir=str(data_dir / "hf_cache"))
    if args.max_train_stories:
        train_split = train_split.select(range(min(args.max_train_stories, len(train_split))))
    if args.max_val_stories:
        val_split = val_split.select(range(min(args.max_val_stories, len(val_split))))
    train_text = "\n\n".join(example["text"] for example in train_split)
    val_text = "\n\n".join(example["text"] for example in val_split)
    train_cache.write_text(train_text, encoding="utf-8")
    val_cache.write_text(val_text, encoding="utf-8")
    return train_text, val_text


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Controlled GELU/UELU token-level GPT study on TinyStories.")
    add_common_args(parser)
    parser.set_defaults(block_size=256, n_layer=6, n_head=6, n_embd=384, batch_size=32, max_iters=10000)
    parser.add_argument("--train-split", default="train")
    parser.add_argument("--val-split", default="validation")
    parser.add_argument("--max-train-stories", type=int, default=50000)
    parser.add_argument("--max-val-stories", type=int, default=5000)
    parser.add_argument("--train-text-file", default=None)
    parser.add_argument("--val-text-file", default=None)
    parser.add_argument("--refresh-cache", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    apply_dry_run_overrides(args)
    if args.dry_run:
        args.max_train_stories = min(args.max_train_stories, 128)
        args.max_val_stories = min(args.max_val_stories, 64)
    device = choose_device(args.device)
    train_text, val_text = load_tinystories_text(args)
    dataset = BPETokenDataset(
        train_text,
        val_text,
        device=device,
        encoding_name=args.encoding,
        tokenizer_dir=Path(args.data_dir) / "tokenizers",
    )
    run_study(args, dataset)


if __name__ == "__main__":
    main()