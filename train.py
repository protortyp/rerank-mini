#!/usr/bin/env python3
"""Student distillation trainer for rerank-mini (32M ModernBERT, 8K context).

Distills continuous Qwen/Qwen3-Reranker-8B log-odds (yes_logit - no_logit)
onto jhu-clsp/ettin-encoder-32m (ModernBertForSequenceClassification, num_labels=1)
using configurable listwise KL divergence, pointwise MSE, and pairwise Margin-MSE,
with length-curriculum staging (512 -> 2K/4K/8K) and BF16 mixed precision.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow.parquet as pq
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from transformers import (
    AutoConfig,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    get_cosine_schedule_with_warmup,
)

from rerank_mini.env import load_dotenv

DEFAULT_BACKBONE = "jhu-clsp/ettin-encoder-32m"


class GroupedRerankDataset(Dataset):
    """Loads scored Parquet shards of grouped (query, docs, teacher_logits) rows."""

    def __init__(
        self,
        shard_paths: list[Path],
        *,
        docs_per_query: int = 8,
        max_samples: int | None = None,
        long_doc_upsample: int = 1,
        code_upsample: int = 1,
        min_doc_words: int = 0,
        seed: int = 42,
    ):
        self.docs_per_query = docs_per_query
        self.rng = random.Random(seed)
        self.rows: list[dict[str, Any]] = []

        for path in sorted(shard_paths):
            table = pq.read_table(
                path,
                columns=["query", "docs", "doc_labels", "teacher_logits", "domain", "source"],
            )
            for row in table.to_pylist():
                docs = row["docs"]
                logits = row["teacher_logits"]
                if not docs or not logits or len(docs) != len(logits) or len(docs) < 2:
                    continue
                if min_doc_words > 0:
                    max_w = max(len(str(d).split()) for d in docs)
                    if max_w < min_doc_words:
                        continue
                domain = str(row.get("domain", "web_qa"))
                repeat = 1
                if domain == "long_doc":
                    repeat = max(1, long_doc_upsample)
                elif domain in ("code", "tool_api"):
                    repeat = max(1, code_upsample)
                item = {
                    "query": str(row["query"]),
                    "docs": [str(d) for d in docs],
                    "labels": [int(x) for x in row["doc_labels"]],
                    "logits": [float(x) for x in logits],
                    "domain": domain,
                }
                for _ in range(repeat):
                    self.rows.append(item)

        self.rng.shuffle(self.rows)
        if max_samples is not None and len(self.rows) > max_samples:
            self.rows = self.rows[:max_samples]

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        row = self.rows[idx]
        docs = row["docs"]
        logits = row["logits"]
        n = len(docs)
        k = self.docs_per_query
        if n > k:
            # Always keep the highest-scoring teacher doc plus a stratified sample of the rest.
            best = int(np.argmax(logits))
            others = [i for i in range(n) if i != best]
            # Keep top hard negatives and sample remaining to preserve full score spread.
            others_sorted = sorted(others, key=lambda i: logits[i], reverse=True)
            n_hard = max(1, (k - 1) // 2)
            picked = [best] + others_sorted[:n_hard]
            rem = others_sorted[n_hard:]
            if len(picked) < k and rem:
                picked.extend(random.sample(rem, min(k - len(picked), len(rem))))
            random.shuffle(picked)
            docs = [docs[i] for i in picked]
            logits = [logits[i] for i in picked]
        elif n < k:
            # Pad short groups by repeating lowest-scoring negatives so every batch tensor is rectangular.
            pad_idx = int(np.argmin(logits))
            docs = list(docs) + [docs[pad_idx]] * (k - n)
            logits = list(logits) + [logits[pad_idx]] * (k - n)
        return {"query": row["query"], "docs": docs, "logits": logits}


def make_collate_fn(tokenizer: Any, max_length: int):
    def collate(batch: list[dict[str, Any]]) -> dict[str, torch.Tensor]:
        bsz = len(batch)
        k = len(batch[0]["docs"])
        qs: list[str] = []
        ds: list[str] = []
        teacher = np.empty((bsz, k), dtype=np.float32)
        for i, item in enumerate(batch):
            q = item["query"]
            for j, (doc, sc) in enumerate(zip(item["docs"], item["logits"])):
                qs.append(q)
                ds.append(doc)
                teacher[i, j] = sc
        enc = tokenizer(
            qs,
            ds,
            padding=True,
            truncation="longest_first",
            max_length=max_length,
            return_tensors="pt",
        )
        enc["teacher_logits"] = torch.from_numpy(teacher)
        enc["group_size"] = torch.tensor(k, dtype=torch.long)
        return enc

    return collate


def compute_distill_loss(
    student_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    *,
    loss_type: str,
    tau: float,
    mse_weight: float,
    margin_weight: float,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Compute listwise KL, pointwise MSE, and pairwise Margin-MSE over [B, K] scores."""
    # Clamp extreme teacher logits to [-16, 16] to avoid exploding MSE gradients on outlier tails.
    t_clamped = teacher_logits.clamp(-16.0, 16.0)

    # Listwise KL(softmax(t/tau) || softmax(s/tau)) scaled by tau^2.
    p_t = F.softmax(t_clamped / tau, dim=-1)
    log_p_s = F.log_softmax(student_logits / tau, dim=-1)
    loss_kl = F.kl_div(log_p_s, p_t, reduction="batchmean") * (tau * tau)

    # Pointwise MSE on raw logits preserves cross-query calibration (sigmoid probability).
    loss_mse = F.mse_loss(student_logits, t_clamped)

    # Pairwise Margin-MSE between best doc and every other doc in each group.
    best_idx = t_clamped.argmax(dim=-1, keepdim=True)
    t_best = t_clamped.gather(1, best_idx)
    s_best = student_logits.gather(1, best_idx)
    loss_margin = F.mse_loss(s_best - student_logits, t_best - t_clamped)

    if loss_type == "kl":
        total = loss_kl
    elif loss_type == "mse":
        total = loss_mse
    else:
        total = loss_kl + mse_weight * loss_mse + margin_weight * loss_margin

    return total, {
        "loss": float(total.detach().item()),
        "kl": float(loss_kl.detach().item()),
        "mse": float(loss_mse.detach().item()),
        "margin": float(loss_margin.detach().item()),
    }


def build_student_model(
    backbone_or_ckpt: str,
    *,
    pooling: str = "cls",
    max_position_embeddings: int = 8192,
    grad_checkpointing: bool = False,
) -> tuple[nn.Module, Any]:
    tokenizer = AutoTokenizer.from_pretrained(backbone_or_ckpt)
    tokenizer.model_max_length = max_position_embeddings

    cfg = AutoConfig.from_pretrained(backbone_or_ckpt)
    cfg.num_labels = 1
    cfg.problem_type = "regression"
    if hasattr(cfg, "classifier_pooling"):
        cfg.classifier_pooling = pooling
    # Ensure 8,192 max context is reflected in config for downstream runtimes (SentenceTransformers/ONNX/Transformers.js).
    if getattr(cfg, "max_position_embeddings", 0) < max_position_embeddings and getattr(cfg, "model_type", "") == "modernbert":
        cfg.max_position_embeddings = max_position_embeddings

    model = AutoModelForSequenceClassification.from_pretrained(
        backbone_or_ckpt,
        config=cfg,
        ignore_mismatched_sizes=True,
    )
    if grad_checkpointing and hasattr(model, "gradient_checkpointing_enable"):
        model.gradient_checkpointing_enable()
    return model, tokenizer


def save_cross_encoder_checkpoint(model: nn.Module, tokenizer: Any, out_dir: Path, meta: dict[str, Any]) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(out_dir, safe_serialization=True)
    tokenizer.save_pretrained(out_dir)
    # Write sentence_bert_config.json so CrossEncoder picks up max_seq_length=8192 natively.
    sb_cfg = {"max_seq_length": int(getattr(model.config, "max_position_embeddings", 8192))}
    (out_dir / "sentence_bert_config.json").write_text(json.dumps(sb_cfg, indent=2) + "\n", encoding="utf-8")
    (out_dir / "train_meta.json").write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")


def main() -> int:
    load_dotenv()
    parser = argparse.ArgumentParser(description="Distill Qwen3-Reranker-8B into 32M 8K ModernBERT student")
    parser.add_argument("--data-dirs", type=Path, nargs="+", required=True, help="Directories containing scored *.parquet shards")
    parser.add_argument("--out-dir", type=Path, required=True, help="Output checkpoint directory")
    parser.add_argument("--backbone", type=str, default=DEFAULT_BACKBONE, help="Base backbone or stage-1 checkpoint")
    parser.add_argument("--pooling", choices=["cls", "mean"], default="cls")
    parser.add_argument("--loss", choices=["kl", "mse", "hybrid"], default="hybrid")
    parser.add_argument("--tau", type=float, default=1.0, help="Distillation temperature for listwise KL")
    parser.add_argument("--mse-weight", type=float, default=0.05, help="Pointwise MSE weight in hybrid loss")
    parser.add_argument("--margin-weight", type=float, default=0.05, help="Pairwise Margin-MSE weight in hybrid loss")
    parser.add_argument("--docs-per-query", type=int, default=8)
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--batch-size", type=int, default=16, help="Query groups per microbatch")
    parser.add_argument("--grad-accum", type=int, default=2)
    parser.add_argument("--lr", type=float, default=8e-5)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--warmup-ratio", type=float, default=0.05)
    parser.add_argument("--epochs", type=float, default=1.0)
    parser.add_argument("--max-steps", type=int, default=-1, help="Stop after N optimizer steps (>0 overrides epochs)")
    parser.add_argument("--max-samples", type=int, default=None, help="Cap dataset queries (for smoke tests)")
    parser.add_argument("--long-doc-upsample", type=int, default=1)
    parser.add_argument("--code-upsample", type=int, default=1)
    parser.add_argument("--min-doc-words", type=int, default=0)
    parser.add_argument("--grad-checkpointing", action="store_true")
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--log-every", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    shards: list[Path] = []
    for d in args.data_dirs:
        shards.extend(sorted(d.glob("*.parquet")))
    if not shards:
        raise FileNotFoundError(f"No scored parquet shards found in {args.data_dirs}")

    print(f"Loading {len(shards)} scored shards...", flush=True)
    ds = GroupedRerankDataset(
        shards,
        docs_per_query=args.docs_per_query,
        max_samples=args.max_samples,
        long_doc_upsample=args.long_doc_upsample,
        code_upsample=args.code_upsample,
        min_doc_words=args.min_doc_words,
        seed=args.seed,
    )
    print(f"Dataset ready: {len(ds)} query groups (x{args.docs_per_query} docs = {len(ds)*args.docs_per_query:,} pairs)", flush=True)

    model, tokenizer = build_student_model(
        args.backbone,
        pooling=args.pooling,
        max_position_embeddings=8192,
        grad_checkpointing=args.grad_checkpointing,
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    num_params = sum(p.numel() for p in model.parameters())
    print(f"Student ready: {args.backbone} | params={num_params:,} | pooling={args.pooling} | max_len={args.max_length}", flush=True)

    loader = DataLoader(
        ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True,
        collate_fn=make_collate_fn(tokenizer, args.max_length),
    )

    steps_per_epoch = max(1, len(loader) // args.grad_accum)
    total_steps = args.max_steps if args.max_steps > 0 else max(1, int(math.ceil(steps_per_epoch * args.epochs)))
    warmup_steps = max(1, int(total_steps * args.warmup_ratio))

    no_decay = ("bias", "norm", "LayerNorm")
    param_groups = [
        {
            "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
            "weight_decay": args.weight_decay,
        },
        {
            "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
            "weight_decay": 0.0,
        },
    ]
    optimizer = torch.optim.AdamW(param_groups, lr=args.lr, fused=torch.cuda.is_available())
    scheduler = get_cosine_schedule_with_warmup(optimizer, warmup_steps, total_steps)

    model.train()
    optimizer.zero_grad(set_to_none=True)
    global_step = 0
    micro_step = 0
    t_start = time.perf_counter()
    running = {"loss": 0.0, "kl": 0.0, "mse": 0.0, "margin": 0.0}
    running_n = 0

    while global_step < total_steps:
        for batch in loader:
            teacher_logits = batch.pop("teacher_logits").to(device, non_blocking=True)
            k = int(batch.pop("group_size").item())
            model_inputs = {key: val.to(device, non_blocking=True) for key, val in batch.items()}
            bsz = teacher_logits.shape[0]

            with torch.amp.autocast("cuda", dtype=torch.bfloat16, enabled=torch.cuda.is_available()):
                out = model(**model_inputs).logits.view(bsz, k)
                loss, metrics = compute_distill_loss(
                    out.float(),
                    teacher_logits.float(),
                    loss_type=args.loss,
                    tau=args.tau,
                    mse_weight=args.mse_weight,
                    margin_weight=args.margin_weight,
                )
                scaled_loss = loss / args.grad_accum

            scaled_loss.backward()
            micro_step += 1
            for mk, mv in metrics.items():
                running[mk] += mv
            running_n += 1

            if micro_step % args.grad_accum == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
                global_step += 1

                if global_step % args.log_every == 0 or global_step == 1 or global_step == total_steps:
                    elapsed = time.perf_counter() - t_start
                    lr_now = scheduler.get_last_lr()[0]
                    avgs = {mk: round(mv / max(1, running_n), 4) for mk, mv in running.items()}
                    sps = (global_step * args.batch_size * args.grad_accum * k) / max(elapsed, 1e-6)
                    print(
                        f"step {global_step:5d}/{total_steps} | lr={lr_now:.2e} | "
                        f"loss={avgs['loss']:.4f} (kl={avgs['kl']:.4f} mse={avgs['mse']:.4f} mrg={avgs['margin']:.4f}) | "
                        f"{sps:.0f} pairs/s | {elapsed:.1f}s",
                        flush=True,
                    )
                    running = {"loss": 0.0, "kl": 0.0, "mse": 0.0, "margin": 0.0}
                    running_n = 0

                if global_step >= total_steps:
                    break

    total_sec = time.perf_counter() - t_start
    meta = {
        "backbone": args.backbone,
        "pooling": args.pooling,
        "loss": args.loss,
        "tau": args.tau,
        "mse_weight": args.mse_weight,
        "margin_weight": args.margin_weight,
        "docs_per_query": args.docs_per_query,
        "max_length": args.max_length,
        "batch_size": args.batch_size,
        "grad_accum": args.grad_accum,
        "lr": args.lr,
        "weight_decay": args.weight_decay,
        "epochs": args.epochs,
        "total_steps": global_step,
        "num_queries": len(ds),
        "num_params": num_params,
        "elapsed_sec": round(total_sec, 1),
    }
    save_cross_encoder_checkpoint(model, tokenizer, args.out_dir, meta)
    print(f"Saved checkpoint -> {args.out_dir} in {total_sec:.1f}s", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
