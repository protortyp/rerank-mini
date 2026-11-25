"""Weighted N-way linear weight-space interpolation (Model Soup / WiSE-FT) for identical-architecture checkpoints."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer


def merge_checkpoints(ckpts: list[Path], weights: list[float], out_dir: Path) -> None:
    if len(ckpts) != len(weights) or not ckpts:
        raise ValueError("Provide matching non-empty lists of checkpoints and weights")
    total_w = float(sum(weights))
    norm_w = [float(w) / total_w for w in weights]

    base_model = AutoModelForSequenceClassification.from_pretrained(ckpts[0], dtype=torch.float32)
    acc = {
        k: (v.clone() * norm_w[0]) if v.is_floating_point() else v.clone()
        for k, v in base_model.state_dict().items()
    }
    for path, w in zip(ckpts[1:], norm_w[1:]):
        m = AutoModelForSequenceClassification.from_pretrained(path, dtype=torch.float32)
        sd = m.state_dict()
        if set(sd.keys()) != set(acc.keys()):
            raise ValueError(f"State dict mismatch for {path}")
        for k in acc:
            if acc[k].is_floating_point():
                acc[k].add_(sd[k], alpha=w)

    base_model.load_state_dict(acc)
    out_dir.mkdir(parents=True, exist_ok=True)
    base_model.save_pretrained(out_dir, safe_serialization=True)
    tokenizer = AutoTokenizer.from_pretrained(ckpts[0])
    tokenizer.save_pretrained(out_dir)
    sb_cfg = {"max_seq_length": int(getattr(base_model.config, "max_position_embeddings", 8192))}
    (out_dir / "sentence_bert_config.json").write_text(json.dumps(sb_cfg, indent=2) + "\n", encoding="utf-8")
    meta = {
        "merge": "weighted_soup",
        "checkpoints": [str(p) for p in ckpts],
        "weights": norm_w,
    }
    (out_dir / "train_meta.json").write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
    print(f"Saved {len(ckpts)}-way soup -> {out_dir} (weights={[round(w, 4) for w in norm_w]})")


def main() -> int:
    parser = argparse.ArgumentParser(description="Weighted N-way Model Soup for rerank-mini checkpoints")
    parser.add_argument("--ckpts", type=Path, nargs="+", required=True)
    parser.add_argument("--weights", type=float, nargs="+", required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()
    merge_checkpoints(args.ckpts, args.weights, args.out_dir)
    return 0


if __name__ == "__main__":
    sys.exit(main())
