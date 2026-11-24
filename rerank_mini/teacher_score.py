"""Qwen3-Reranker-8B teacher soft-label scoring engine.

Extracts continuous relevance log-odds:
    score = logit("yes") - logit("no")
using the official Qwen3-Reranker-8B chat template and domain-specific instructions.

Supports:
- Local RTX 3090 smoke tests (500 pairs)
- High-throughput Vast.ai H200 / H100 / A100 sharded scoring with periodic streaming
  to a private Hugging Face dataset repo every N minutes so zero work is lost
- Resume from already-uploaded Parquet shards
"""

from __future__ import annotations

import argparse
import gc
import json
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import torch
from huggingface_hub import HfApi
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from rerank_mini.env import get_hf_token, load_dotenv

TEACHER_MODEL_ID = "Qwen/Qwen3-Reranker-8B"

PREFIX = (
    "<|im_start|>system\n"
    "Judge whether the Document meets the requirements based on the Query and the Instruct provided. "
    'Note that the answer can only be "yes" or "no".<|im_end|>\n'
    "<|im_start|>user\n"
)
SUFFIX = "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"

DOMAIN_INSTRUCTIONS = {
    "web_qa": "Given a web search query, retrieve relevant passages that answer the query",
    "code": "Given a coding query or technical description, retrieve relevant source code or API documentation that satisfies it",
    "tool_api": "Given a user request for an agent, retrieve the tool or API documentation that should be called",
    "long_doc": "Given a question or information need, retrieve the long document that contains the supporting answer",
}


def format_pair(instruction: str, query: str, doc: str) -> str:
    return f"<Instruct>: {instruction}\n<Query>: {query}\n<Document>: {doc}"


def normalize_query_scores(scores: list[float], temperature: float = 1.0) -> dict[str, list[float]]:
    """Preserve teacher score spread via max-subtraction and temperature-scaled softmax."""
    arr = np.asarray(scores, dtype=np.float64)
    shifted = arr - np.max(arr)
    scaled = shifted / max(temperature, 1e-6)
    exps = np.exp(scaled - np.max(scaled))
    probs = exps / np.sum(exps)
    return {
        "raw_logits": [float(x) for x in arr],
        "max_shifted_logits": [float(x) for x in shifted],
        "teacher_probs": [float(x) for x in probs],
    }


class Qwen3TeacherScorer:
    """Batched FlashAttention-2 / SDPA scorer for Qwen/Qwen3-Reranker-8B."""

    def __init__(
        self,
        model_id: str = TEACHER_MODEL_ID,
        max_length: int = 8192,
        dtype: torch.dtype = torch.bfloat16,
    ):
        self.model_id = model_id
        self.max_length = max_length
        self.tokenizer = AutoTokenizer.from_pretrained(model_id, padding_side="left")
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        attn_impl = "sdpa"
        try:
            import flash_attn  # noqa: F401
            attn_impl = "flash_attention_2"
        except Exception:
            pass

        print(f"Loading teacher {model_id} (dtype={dtype}, attn={attn_impl})...")
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = (
            AutoModelForCausalLM.from_pretrained(
                model_id,
                dtype=dtype,
                attn_implementation=attn_impl,
            )
            .to(device)
            .eval()
        )

        self.token_true_id = int(self.tokenizer.convert_tokens_to_ids("yes"))
        self.token_false_id = int(self.tokenizer.convert_tokens_to_ids("no"))
        # Precompute the single-vector difference (W_yes - W_no) from lm_head so we never materialize
        # the full [B, SeqLen, 151669] vocab logits tensor (saves ~1.2 GB/seq at 4K context).
        lm_w = self.model.lm_head.weight.detach()
        self.diff_vec = (lm_w[self.token_true_id] - lm_w[self.token_false_id]).float()
        if self.model.lm_head.bias is not None:
            lm_b = self.model.lm_head.bias.detach()
            self.diff_bias = float((lm_b[self.token_true_id] - lm_b[self.token_false_id]).float().item())
        else:
            self.diff_bias = 0.0
        self.prefix_tokens = self.tokenizer.encode(PREFIX, add_special_tokens=False)
        self.suffix_tokens = self.tokenizer.encode(SUFFIX, add_special_tokens=False)
        self.body_budget = self.max_length - len(self.prefix_tokens) - len(self.suffix_tokens)
        self.vram_gb = (
            torch.cuda.get_device_properties(0).total_memory / (1024**3)
            if torch.cuda.is_available()
            else 0.0
        )

    def _batch_size_for_tokens(self, max_toks: int, base_batch_size: int) -> int:
        # Scale batch size smoothly with sequence length and GPU VRAM (24 GB 3090 vs 80-141 GB A100/H200).
        vram_scale = max(1.0, self.vram_gb / 24.0)
        if max_toks <= 512:
            return int(base_batch_size * min(vram_scale, 4.0))
        if max_toks <= 1024:
            return max(4, int(base_batch_size * min(vram_scale, 4.0) * 0.5))
        if max_toks <= 2048:
            return max(2, int(base_batch_size * min(vram_scale, 4.0) * 0.25))
        if max_toks <= 4096:
            return max(1, int(base_batch_size * min(vram_scale, 4.0) * 0.12))
        return max(1, int(base_batch_size * min(vram_scale, 4.0) * 0.06))

    @torch.inference_mode()
    def score_texts(
        self,
        formatted_texts: list[str],
        base_batch_size: int = 16,
        show_progress: bool = True,
    ) -> np.ndarray:
        if not formatted_texts:
            return np.zeros(0, dtype=np.float32)

        enc_bodies = self.tokenizer(
            formatted_texts,
            padding=False,
            truncation="longest_first",
            return_attention_mask=False,
            max_length=self.body_budget,
            add_special_tokens=False,
        )["input_ids"]
        full_ids = [self.prefix_tokens + body + self.suffix_tokens for body in enc_bodies]
        order = sorted(range(len(full_ids)), key=lambda i: len(full_ids[i]), reverse=True)
        out = np.empty(len(full_ids), dtype=np.float32)

        pbar = tqdm(total=len(order), disable=not show_progress, desc="Qwen3-8B scoring")
        idx = 0
        device = next(self.model.parameters()).device
        while idx < len(order):
            lead_len = len(full_ids[order[idx]])
            bs = max(1, self._batch_size_for_tokens(lead_len, base_batch_size))
            chunk_idx = order[idx : idx + bs]
            batch_ids = [full_ids[i] for i in chunk_idx]
            padded = self.tokenizer.pad(
                {"input_ids": batch_ids},
                padding=True,
                return_tensors="pt",
            )
            padded = {k: v.to(device, non_blocking=True) for k, v in padded.items()}
            try:
                hidden = self.model.model(**padded, use_cache=False).last_hidden_state[:, -1, :].float()
                log_odds = torch.matmul(hidden, self.diff_vec.to(hidden.device)) + self.diff_bias
                vals = log_odds.cpu().numpy()
                for pos, orig_i in enumerate(chunk_idx):
                    out[orig_i] = vals[pos]
                idx += len(chunk_idx)
                pbar.update(len(chunk_idx))
            except torch.cuda.OutOfMemoryError:
                torch.cuda.empty_cache()
                gc.collect()
                if bs == 1:
                    raise
                base_batch_size = max(1, base_batch_size // 2)
                print(f"[OOM backoff] Reduced base_batch_size -> {base_batch_size}", flush=True)
        pbar.close()
        return out


def score_parquet_shard(
    scorer: Qwen3TeacherScorer,
    input_path: Path,
    output_path: Path,
    *,
    base_batch_size: int = 16,
    max_queries: int | None = None,
) -> dict[str, Any]:
    """Score all (query, candidate_docs) rows in an input Parquet shard and write output Parquet."""
    table = pq.read_table(input_path)
    rows = table.to_pylist()
    if max_queries is not None:
        rows = rows[:max_queries]

    formatted: list[str] = []
    offsets: list[tuple[int, int]] = []
    for row in rows:
        domain = str(row.get("domain", "web_qa"))
        instruct = str(row.get("instruction") or DOMAIN_INSTRUCTIONS.get(domain, DOMAIN_INSTRUCTIONS["web_qa"]))
        q = str(row["query"])
        docs = list(row["docs"])
        start = len(formatted)
        for doc in docs:
            formatted.append(format_pair(instruct, q, str(doc)))
        offsets.append((start, len(formatted)))

    t0 = time.perf_counter()
    all_scores = scorer.score_texts(formatted, base_batch_size=base_batch_size)
    elapsed = time.perf_counter() - t0

    out_rows = []
    for row, (start, end) in zip(rows, offsets):
        raw = [float(x) for x in all_scores[start:end]]
        norm = normalize_query_scores(raw, temperature=1.0)
        out_rows.append(
            {
                "query_id": str(row["query_id"]),
                "source": str(row["source"]),
                "domain": str(row["domain"]),
                "license": str(row["license"]),
                "query": str(row["query"]),
                "docs": list(row["docs"]),
                "doc_labels": [int(x) for x in row["doc_labels"]],
                "teacher_logits": norm["raw_logits"],
                "teacher_logits_shifted": norm["max_shifted_logits"],
            }
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.Table.from_pylist(out_rows), output_path, compression="zstd")

    pos_scores = [
        sc for r in out_rows for sc, lbl in zip(r["teacher_logits"], r["doc_labels"]) if lbl > 0
    ]
    neg_scores = [
        sc for r in out_rows for sc, lbl in zip(r["teacher_logits"], r["doc_labels"]) if lbl == 0
    ]
    return {
        "input": str(input_path),
        "output": str(output_path),
        "num_queries": len(out_rows),
        "num_pairs": len(formatted),
        "elapsed_sec": round(elapsed, 2),
        "pairs_per_sec": round(len(formatted) / max(elapsed, 1e-6), 2),
        "pos_mean_logit": round(float(np.mean(pos_scores)), 4) if pos_scores else None,
        "neg_mean_logit": round(float(np.mean(neg_scores)), 4) if neg_scores else None,
        "margin": round(float(np.mean(pos_scores) - np.mean(neg_scores)), 4)
        if pos_scores and neg_scores
        else None,
    }


def upload_shard(local_file: Path, repo_id: str, path_in_repo: str) -> None:
    token = get_hf_token()
    api = HfApi(token=token)
    api.create_repo(repo_id=repo_id, repo_type="dataset", private=True, exist_ok=True)
    api.upload_file(
        path_or_fileobj=str(local_file),
        path_in_repo=path_in_repo,
        repo_id=repo_id,
        repo_type="dataset",
    )


def main() -> int:
    load_dotenv()
    parser = argparse.ArgumentParser(description="Score candidate shards with Qwen3-Reranker-8B")
    parser.add_argument("--input-dir", type=Path, required=True, help="Directory of unscored shard_*.parquet files")
    parser.add_argument("--output-dir", type=Path, required=True, help="Directory for scored Parquet shards")
    parser.add_argument("--hf-repo", type=str, default="", help="Private HF dataset repo to stream shards to")
    parser.add_argument("--hf-prefix", type=str, default="scored_shards", help="Subdir inside HF dataset repo")
    parser.add_argument("--shard-start", type=int, default=0)
    parser.add_argument("--shard-end", type=int, default=-1, help="Exclusive shard end index (-1 = all)")
    parser.add_argument("--max-length", type=int, default=6144)
    parser.add_argument("--base-batch-size", type=int, default=16)
    parser.add_argument("--max-queries", type=int, default=None, help="Limit queries per shard (smoke test)")
    args = parser.parse_args()

    shards = sorted(args.input_dir.glob("*.parquet"))
    if args.shard_end >= 0:
        shards = shards[args.shard_start : args.shard_end]
    else:
        shards = shards[args.shard_start :]
    if not shards:
        print(f"No parquet shards found in {args.input_dir}")
        return 1

    existing_remote: set[str] = set()
    if args.hf_repo:
        api = HfApi(token=get_hf_token())
        api.create_repo(repo_id=args.hf_repo, repo_type="dataset", private=True, exist_ok=True)
        existing_remote = set(api.list_repo_files(args.hf_repo, repo_type="dataset"))

    scorer = Qwen3TeacherScorer(max_length=args.max_length)
    for shard in shards:
        out_file = args.output_dir / shard.name
        remote_path = f"{args.hf_prefix}/{shard.name}"
        if args.hf_repo and remote_path in existing_remote:
            print(f"[skip] {remote_path} already uploaded to {args.hf_repo}")
            continue
        if out_file.is_file() and not args.max_queries:
            print(f"[local exists] {out_file}; uploading if --hf-repo set")
        else:
            stats = score_parquet_shard(
                scorer,
                shard,
                out_file,
                base_batch_size=args.base_batch_size,
                max_queries=args.max_queries,
            )
            print(json.dumps(stats, indent=2), flush=True)
        if args.hf_repo:
            upload_shard(out_file, args.hf_repo, remote_path)
            print(f"[uploaded] {remote_path} -> {args.hf_repo}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
