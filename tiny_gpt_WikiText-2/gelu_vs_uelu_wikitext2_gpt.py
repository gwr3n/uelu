"""
WikiText-2 token-level GPT experiment for the GELU/UELU computational study.

Examples
--------
Smoke test:
    python gelu_vs_uelu_wikitext2_gpt.py --dry-run --activations gelu uelu

Single-seed protocol check:
    python gelu_vs_uelu_wikitext2_gpt.py --max-iters 5000 --seed 1

Three-seed final comparison:
    python gelu_vs_uelu_wikitext2_gpt.py --max-iters 10000 --seeds 1 2 3
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tiny_gpt_common import BPETokenDataset, add_common_args, apply_dry_run_overrides, choose_device, run_study


def load_wikitext2_text(args: argparse.Namespace) -> tuple[str, str]:
    if args.train_text_file:
        train_text = Path(args.train_text_file).read_text(encoding="utf-8")
        val_text = Path(args.val_text_file).read_text(encoding="utf-8") if args.val_text_file else ""
        return train_text, val_text
    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise ImportError("Install datasets to download WikiText-2, or pass --train-text-file.") from exc

    data_dir = Path(args.data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)
    train_cache = data_dir / "wikitext2_train.txt"
    val_cache = data_dir / "wikitext2_validation.txt"
    if train_cache.exists() and val_cache.exists() and not args.refresh_cache:
        return train_cache.read_text(encoding="utf-8"), val_cache.read_text(encoding="utf-8")

    print("Loading WikiText-2 from Hugging Face datasets")
    train_split = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split="train", cache_dir=str(data_dir / "hf_cache"))
    val_split = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split="validation", cache_dir=str(data_dir / "hf_cache"))
    train_text = "\n".join(example["text"] for example in train_split if example["text"].strip())
    val_text = "\n".join(example["text"] for example in val_split if example["text"].strip())
    train_cache.write_text(train_text, encoding="utf-8")
    val_cache.write_text(val_text, encoding="utf-8")
    return train_text, val_text


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Controlled GELU/UELU token-level GPT study on WikiText-2.")
    add_common_args(parser)
    parser.set_defaults(block_size=256, n_layer=4, n_head=4, n_embd=256, batch_size=32, max_iters=10000)
    parser.add_argument("--train-text-file", default=None)
    parser.add_argument("--val-text-file", default=None)
    parser.add_argument("--refresh-cache", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    apply_dry_run_overrides(args)
    device = choose_device(args.device)
    train_text, val_text = load_wikitext2_text(args)
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