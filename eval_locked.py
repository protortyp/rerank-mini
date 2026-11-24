#!/usr/bin/env python3
"""Immutable evaluation harness for rerank-mini.

Locked after Phase 1 (`chmod 444 eval_locked.py` and `suite_v1.json`).
Evaluates any PyTorch CrossEncoder, ONNX Runtime graph, or HTTP /v1/rerank server
(e.g. llama.cpp GGUF) on the frozen three-pillar benchmark:
  1. BEIR Retrieval (7 NanoBEIR datasets, fixed top-100 BM25 candidates)
  2. Code & Tech Docs Retrieval (CoSQA, CodeSearchNet-Python, StackOverflow-QA, top-100 BM25)
  3. Long-Context Retrieval >1,024 tokens (MLDR-en, LongEmbed-2WikiMQA, LongEmbed-Needle, top-30 BM25)
Plus deterministic CPU (ONNX) and GPU (PyTorch/ONNX) throughput & latency benchmarks.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import math
import os
import re
import sys
import time
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import requests
from datasets import load_dataset
from huggingface_hub import HfApi, hf_hub_download
from rank_bm25 import BM25Okapi

from rerank_mini.env import REPO_ROOT, get_data_dir, load_dotenv

SUITE_FILENAME = "suite_v1.json"
MANIFEST_PATH = REPO_ROOT / "results" / "eval_locked_manifest.json"
TOKEN_RE = re.compile(r"[a-z0-9_]+")

BEIR_TASKS = [
    ("NanoMSMARCO", "zeta-alpha-ai/NanoMSMARCO"),
    ("NanoNFCorpus", "zeta-alpha-ai/NanoNFCorpus"),
    ("NanoFiQA2018", "zeta-alpha-ai/NanoFiQA2018"),
    ("NanoSciFact", "zeta-alpha-ai/NanoSciFact"),
    ("NanoNQ", "zeta-alpha-ai/NanoNQ"),
    ("NanoHotpotQA", "zeta-alpha-ai/NanoHotpotQA"),
    ("NanoSCIDOCS", "zeta-alpha-ai/NanoSCIDOCS"),
]


def _tokenize_bm25(text: str) -> list[str]:
    return TOKEN_RE.findall(text.lower())


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _dcg_at_k(rels: list[float], k: int) -> float:
    return sum(((2.0 ** r) - 1.0) / math.log2(idx + 2.0) for idx, r in enumerate(rels[:k]) if r > 0)


def compute_ndcg_at_k(
    qrels: dict[str, dict[str, float]],
    ranked: dict[str, list[str]],
    k: int = 10,
) -> float:
    scores: list[float] = []
    for qid, rel_map in qrels.items():
        pos_rels = sorted((float(v) for v in rel_map.values() if float(v) > 0), reverse=True)
        if not pos_rels:
            continue
        idcg = _dcg_at_k(pos_rels, k)
        if idcg <= 0:
            continue
        pred_ids = ranked.get(qid, [])[:k]
        pred_rels = [float(rel_map.get(doc_id, 0.0)) for doc_id in pred_ids]
        scores.append(_dcg_at_k(pred_rels, k) / idcg)
    return float(np.mean(scores)) if scores else 0.0


def compute_recall_at_k(
    qrels: dict[str, dict[str, float]],
    ranked: dict[str, list[str]],
    k: int,
) -> float:
    hits: list[float] = []
    for qid, rel_map in qrels.items():
        pos_ids = {d for d, r in rel_map.items() if float(r) > 0}
        if not pos_ids:
            continue
        pred_ids = set(ranked.get(qid, [])[:k])
        hits.append(len(pos_ids & pred_ids) / len(pos_ids))
    return float(np.mean(hits)) if hits else 0.0


def _bm25_retrieve(
    queries: dict[str, str],
    corpus: dict[str, str],
    qrels: dict[str, dict[str, float]],
    top_k: int,
    max_queries: int | None = None,
) -> dict[str, Any]:
    doc_ids = sorted(corpus.keys())
    tokenized_corpus = [_tokenize_bm25(corpus[d]) for d in doc_ids]
    bm25 = BM25Okapi(tokenized_corpus)

    valid_qids = sorted(
        qid for qid in qrels if qid in queries and any(v > 0 for v in qrels[qid].values())
    )
    if max_queries is not None and len(valid_qids) > max_queries:
        # Deterministic evenly spaced subsample across sorted query IDs.
        step = len(valid_qids) / max_queries
        valid_qids = [valid_qids[int(i * step)] for i in range(max_queries)]

    candidates: dict[str, list[str]] = {}
    used_docs: dict[str, str] = {}
    sub_queries: dict[str, str] = {}
    sub_qrels: dict[str, dict[str, float]] = {}

    k_eff = min(top_k, len(doc_ids))
    for qid in valid_qids:
        q_text = queries[qid]
        scores = bm25.get_scores(_tokenize_bm25(q_text))
        if k_eff < len(doc_ids):
            top_idx = np.argpartition(-scores, k_eff - 1)[:k_eff]
            top_idx = top_idx[np.argsort(-scores[top_idx], kind="mergesort")]
        else:
            top_idx = np.argsort(-scores, kind="mergesort")
        cands = [doc_ids[int(i)] for i in top_idx]
        candidates[qid] = cands
        sub_queries[qid] = q_text
        sub_qrels[qid] = qrels[qid]
        for cid in cands:
            if cid not in used_docs:
                used_docs[cid] = corpus[cid]

    word_lens = sorted(len(text.split()) for text in used_docs.values())
    med_words = word_lens[len(word_lens) // 2] if word_lens else 0
    return {
        "queries": sub_queries,
        "corpus": used_docs,
        "qrels": sub_qrels,
        "candidates": candidates,
        "meta": {
            "num_queries": len(sub_queries),
            "num_unique_docs": len(used_docs),
            "top_k": k_eff,
            "median_doc_words": med_words,
            "bm25_ndcg10": round(compute_ndcg_at_k(sub_qrels, candidates, k=10), 6),
            "bm25_recall_at_k": round(compute_recall_at_k(sub_qrels, candidates, k=k_eff), 6),
        },
    }


def _build_nanobeir_task(hf_id: str) -> dict[str, Any]:
    q_ds = load_dataset(hf_id, "queries", split="train")
    c_ds = load_dataset(hf_id, "corpus", split="train")
    r_ds = load_dataset(hf_id, "qrels", split="train")
    queries = {str(row["_id"]): str(row["text"]) for row in q_ds}
    corpus = {str(row["_id"]): str(row["text"]) for row in c_ds}
    qrels: dict[str, dict[str, float]] = defaultdict(dict)
    for row in r_ds:
        qrels[str(row["query-id"])][str(row["corpus-id"])] = float(row.get("score", 1.0))
    return _bm25_retrieve(queries, corpus, dict(qrels), top_k=100)


def _build_coir_task(hf_id: str, max_queries: int = 200) -> dict[str, Any]:
    q_ds = load_dataset(hf_id, "queries", split="queries")
    c_ds = load_dataset(hf_id, "corpus", split="corpus")
    r_ds = load_dataset(hf_id, "default", split="test")
    queries = {str(row["_id"]): str(row["text"]) for row in q_ds}
    corpus = {}
    for row in c_ds:
        title = str(row.get("title") or "").strip()
        body = str(row["text"])
        corpus[str(row["_id"])] = f"{title}\n{body}" if title else body
    qrels: dict[str, dict[str, float]] = defaultdict(dict)
    for row in r_ds:
        score = float(row.get("score", 1.0))
        if score > 0:
            qrels[str(row["query-id"])][str(row["corpus-id"])] = score
    return _bm25_retrieve(queries, corpus, dict(qrels), top_k=100, max_queries=max_queries)


def _build_csn_python_task(max_queries: int = 200) -> dict[str, Any]:
    q_ds = load_dataset("mteb/CodeSearchNetRetrieval", "python-queries", split="test")
    c_ds = load_dataset("mteb/CodeSearchNetRetrieval", "python-corpus", split="test")
    r_ds = load_dataset("mteb/CodeSearchNetRetrieval", "python-qrels", split="test")
    queries = {str(row["id"]): str(row["text"]) for row in q_ds}
    corpus = {str(row["id"]): str(row["text"]) for row in c_ds}
    qrels: dict[str, dict[str, float]] = defaultdict(dict)
    for row in r_ds:
        score = float(row.get("score", 1.0))
        if score > 0:
            qrels[str(row["query-id"])][str(row["corpus-id"])] = score
    return _bm25_retrieve(queries, corpus, dict(qrels), top_k=100, max_queries=max_queries)


def _build_longembed_task(cfg: str, max_queries: int = 150, top_k: int = 30) -> dict[str, Any]:
    q_ds = load_dataset("dwzhu/LongEmbed", cfg, split="queries")
    c_ds = load_dataset("dwzhu/LongEmbed", cfg, split="corpus")
    r_ds = load_dataset("dwzhu/LongEmbed", cfg, split="qrels")
    queries = {str(row["qid"]): str(row["text"]) for row in q_ds}
    corpus = {str(row["doc_id"]): str(row["text"]) for row in c_ds}
    qrels: dict[str, dict[str, float]] = defaultdict(dict)
    for row in r_ds:
        score = float(row.get("score", 1.0))
        if score > 0:
            qrels[str(row["qid"])][str(row["doc_id"])] = score
    return _bm25_retrieve(queries, corpus, dict(qrels), top_k=top_k, max_queries=max_queries)


def _build_mldr_en_task(max_queries: int = 150, top_k: int = 30) -> dict[str, Any]:
    test_gz = hf_hub_download("Shitao/MLDR", "mldr-v1.0-en/test.jsonl.gz", repo_type="dataset")
    queries: dict[str, str] = {}
    corpus: dict[str, str] = {}
    qrels: dict[str, dict[str, float]] = defaultdict(dict)
    with gzip.open(test_gz, "rt", encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            qid = str(row["query_id"])
            pos_list = row.get("positive_passages") or []
            neg_list = row.get("negative_passages") or []
            # Keep only genuine long-context queries (>= 700 words / ~1,000+ tokens).
            long_pos = [p for p in pos_list if len(str(p["text"]).split()) >= 700]
            if not long_pos:
                continue
            queries[qid] = str(row["query"])
            for p in long_pos:
                did = str(p["docid"])
                corpus[did] = str(p["text"])
                qrels[qid][did] = 1.0
            for n in neg_list:
                did = str(n["docid"])
                if did not in corpus:
                    corpus[did] = str(n["text"])
    return _bm25_retrieve(queries, corpus, dict(qrels), top_k=top_k, max_queries=max_queries)


def build_suite(suite_path: Path) -> None:
    load_dotenv()
    suite_path.parent.mkdir(parents=True, exist_ok=True)
    MANIFEST_PATH.parent.mkdir(parents=True, exist_ok=True)

    pillars: dict[str, dict[str, Any]] = {"beir": {}, "code": {}, "long_context": {}}

    print("Building BEIR pillar (7 NanoBEIR datasets)...")
    for name, hf_id in BEIR_TASKS:
        print(f"  -> {name}")
        pillars["beir"][name] = _build_nanobeir_task(hf_id)

    print("Building Code & Tech Docs pillar (3 datasets)...")
    print("  -> CoSQA")
    pillars["code"]["CoSQA"] = _build_coir_task("CoIR-Retrieval/cosqa", max_queries=200)
    print("  -> CodeSearchNet-Python")
    pillars["code"]["CodeSearchNet-Python"] = _build_csn_python_task(max_queries=200)
    print("  -> StackOverflow-QA")
    pillars["code"]["StackOverflow-QA"] = _build_coir_task("CoIR-Retrieval/stackoverflow-qa", max_queries=200)

    print("Building Long-Context (>1,024 tok) pillar (3 datasets)...")
    print("  -> MLDR-en")
    pillars["long_context"]["MLDR-en"] = _build_mldr_en_task(max_queries=150, top_k=30)
    print("  -> LongEmbed-2WikiMQA")
    pillars["long_context"]["LongEmbed-2WikiMQA"] = _build_longembed_task("2wikimqa", max_queries=150, top_k=30)
    print("  -> LongEmbed-Needle")
    pillars["long_context"]["LongEmbed-Needle"] = _build_longembed_task("needle", max_queries=150, top_k=30)

    # Build deterministic latency benchmark batches directly from the frozen suite.
    short_pairs: list[list[str]] = []
    ms_task = pillars["beir"]["NanoMSMARCO"]
    for qid in sorted(ms_task["candidates"].keys()):
        q = ms_task["queries"][qid]
        for cid in ms_task["candidates"][qid][:5]:
            short_pairs.append([q, ms_task["corpus"][cid]])
            if len(short_pairs) >= 100:
                break
        if len(short_pairs) >= 100:
            break

    long_pairs: list[list[str]] = []
    mldr_task = pillars["long_context"]["MLDR-en"]
    for qid in sorted(mldr_task["candidates"].keys()):
        q = mldr_task["queries"][qid]
        for cid in mldr_task["candidates"][qid][:2]:
            doc = mldr_task["corpus"][cid]
            words = doc.split()
            if len(words) >= 1400:
                long_pairs.append([q, " ".join(words[:1600])])
            if len(long_pairs) >= 32:
                break
        if len(long_pairs) >= 32:
            break

    payload = {
        "version": "v1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "pillars": pillars,
        "latency_batches": {
            "short_100": short_pairs,
            "long_32": long_pairs,
        },
    }
    suite_path.write_text(json.dumps(payload), encoding="utf-8")
    digest = _sha256_file(suite_path)

    manifest: dict[str, Any] = {
        "version": "v1",
        "suite_path": str(suite_path),
        "suite_sha256": digest,
        "created_utc": payload["created_utc"],
        "pillars": {},
    }
    pillar_bm25 = {}
    for pname, tasks in pillars.items():
        manifest["pillars"][pname] = {tname: t["meta"] for tname, t in tasks.items()}
        pillar_bm25[pname] = round(float(np.mean([t["meta"]["bm25_ndcg10"] for t in tasks.values()])), 6)
    manifest["bm25_summary"] = {
        **pillar_bm25,
        "overall_ndcg10": round(float(np.mean(list(pillar_bm25.values()))), 6),
    }
    MANIFEST_PATH.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print("\nBuilt locked eval suite successfully!")
    print(json.dumps(manifest, indent=2))


def load_verified_suite(suite_path: Path) -> dict[str, Any]:
    if not suite_path.is_file():
        raise FileNotFoundError(f"Locked suite missing at {suite_path}. Run --build-suite first.")
    if MANIFEST_PATH.is_file():
        manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
        expected = manifest.get("suite_sha256")
        actual = _sha256_file(suite_path)
        if expected and actual != expected:
            raise RuntimeError(f"Locked suite checksum mismatch! expected={expected} actual={actual}")
    return json.loads(suite_path.read_text(encoding="utf-8"))


class BaseScorer:
    def score_pairs(self, pairs: list[tuple[str, str]]) -> np.ndarray:
        raise NotImplementedError


class PyTorchCrossEncoderScorer(BaseScorer):
    def __init__(
        self,
        model_id: str,
        device: str = "cuda",
        dtype_name: str = "bf16",
        max_seq_length: int | None = None,
    ):
        import torch
        from sentence_transformers import CrossEncoder

        dtype_map = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}
        dtype = dtype_map[dtype_name]
        self.torch = torch
        self.device = device
        self.model = CrossEncoder(
            model_id,
            device=device,
            trust_remote_code=False,
            model_kwargs={"dtype": dtype},
        )
        native_max = getattr(self.model, "max_seq_length", None) or getattr(self.model, "max_length", 512)
        if max_seq_length is not None:
            self.max_seq_length = min(int(max_seq_length), int(native_max))
        else:
            self.max_seq_length = min(int(native_max), 8192)
        self.model.max_seq_length = self.max_seq_length
        self.num_params = sum(p.numel() for p in self.model.model.parameters())
        self.is_large = self.num_params > 100_000_000

    def _batch_size_for_chars(self, max_chars: int) -> int:
        if self.is_large:
            if max_chars <= 1500:
                return 64
            if max_chars <= 5000:
                return 16
            if max_chars <= 14000:
                return 4
            return 2
        if max_chars <= 1500:
            return 256
        if max_chars <= 5000:
            return 64
        if max_chars <= 14000:
            return 16
        return 8

    def score_pairs(self, pairs: list[tuple[str, str]]) -> np.ndarray:
        if not pairs:
            return np.zeros(0, dtype=np.float32)
        # Sort by total character length for minimal padding overhead; restore exact order after.
        order = sorted(range(len(pairs)), key=lambda i: len(pairs[i][0]) + len(pairs[i][1]), reverse=True)
        out = np.empty(len(pairs), dtype=np.float32)
        idx = 0
        with self.torch.inference_mode():
            while idx < len(order):
                lead_i = order[idx]
                max_chars = len(pairs[lead_i][0]) + len(pairs[lead_i][1])
                bs = self._batch_size_for_chars(max_chars)
                chunk_idx = order[idx : idx + bs]
                chunk_pairs = [pairs[i] for i in chunk_idx]
                preds = self.model.predict(
                    chunk_pairs,
                    batch_size=len(chunk_pairs),
                    show_progress_bar=False,
                    convert_to_numpy=True,
                )
                preds = np.asarray(preds, dtype=np.float32).reshape(-1)
                for pos, orig_i in enumerate(chunk_idx):
                    out[orig_i] = preds[pos]
                idx += len(chunk_idx)
        return out


class OnnxCrossEncoderScorer(BaseScorer):
    def __init__(
        self,
        onnx_path: str | Path,
        tokenizer_id: str,
        provider: str = "cpu",
        num_threads: int = 4,
        max_seq_length: int = 8192,
    ):
        import onnxruntime as ort
        from transformers import AutoConfig, AutoTokenizer

        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_id)
        try:
            cfg = AutoConfig.from_pretrained(tokenizer_id)
            native_max = int(getattr(cfg, "max_position_embeddings", max_seq_length))
        except Exception:
            native_max = max_seq_length
        self.max_seq_length = min(max_seq_length, native_max, 8192)

        sess_opts = ort.SessionOptions()
        sess_opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        if provider == "cpu" and num_threads > 0:
            sess_opts.intra_op_num_threads = num_threads
            sess_opts.inter_op_num_threads = 1
        providers = ["CUDAExecutionProvider", "CPUExecutionProvider"] if provider == "cuda" else ["CPUExecutionProvider"]
        self.provider = provider
        self.session = ort.InferenceSession(str(onnx_path), sess_options=sess_opts, providers=providers)
        self.input_names = {inp.name for inp in self.session.get_inputs()}

    def _batch_size_for_chars(self, max_chars: int) -> int:
        if self.provider == "cpu":
            if max_chars <= 1500:
                return 32
            if max_chars <= 5000:
                return 8
            return 2
        if max_chars <= 1500:
            return 128
        if max_chars <= 5000:
            return 32
        return 8

    def score_pairs(self, pairs: list[tuple[str, str]]) -> np.ndarray:
        if not pairs:
            return np.zeros(0, dtype=np.float32)
        order = sorted(range(len(pairs)), key=lambda i: len(pairs[i][0]) + len(pairs[i][1]), reverse=True)
        out = np.empty(len(pairs), dtype=np.float32)
        idx = 0
        while idx < len(order):
            lead_i = order[idx]
            max_chars = len(pairs[lead_i][0]) + len(pairs[lead_i][1])
            bs = self._batch_size_for_chars(max_chars)
            chunk_idx = order[idx : idx + bs]
            qs = [pairs[i][0] for i in chunk_idx]
            ds = [pairs[i][1] for i in chunk_idx]
            enc = self.tokenizer(
                qs,
                ds,
                padding=True,
                truncation=True,
                max_length=self.max_seq_length,
                return_tensors="np",
            )
            feed = {k: v.astype(np.int64) for k, v in enc.items() if k in self.input_names}
            logits = self.session.run(None, feed)[0]
            logits = np.asarray(logits, dtype=np.float32)
            if logits.ndim == 2 and logits.shape[1] == 2:
                scores = logits[:, 1] - logits[:, 0]
            else:
                scores = logits.reshape(-1)
            for pos, orig_i in enumerate(chunk_idx):
                out[orig_i] = scores[pos]
            idx += len(chunk_idx)
        return out


class HttpRerankScorer(BaseScorer):
    """Evaluates any OpenAI/Jina/llama-server compatible /v1/rerank endpoint (e.g., GGUF)."""

    def __init__(self, endpoint: str, model_name: str = "rerank", timeout: int = 120):
        self.endpoint = endpoint
        self.model_name = model_name
        self.timeout = timeout
        self.session = requests.Session()

    def score_pairs(self, pairs: list[tuple[str, str]]) -> np.ndarray:
        # Group contiguous pairs by query to batch /v1/rerank calls cleanly.
        out = np.empty(len(pairs), dtype=np.float32)
        grouped: dict[str, list[tuple[int, str]]] = defaultdict(list)
        for i, (q, d) in enumerate(pairs):
            grouped[q].append((i, d))
        for q, items in grouped.items():
            docs = [d for _, d in items]
            resp = self.session.post(
                self.endpoint,
                json={"model": self.model_name, "query": q, "documents": docs, "top_n": len(docs)},
                timeout=self.timeout,
            )
            resp.raise_for_status()
            results = resp.json().get("results", [])
            for item in results:
                orig_idx = items[int(item["index"])][0]
                out[orig_idx] = float(item["relevance_score"])
        return out


def evaluate_suite(scorer: BaseScorer, suite: dict[str, Any]) -> dict[str, Any]:
    pillars_out: dict[str, Any] = {}
    pillar_avgs: dict[str, float] = {}

    for pname, tasks in suite["pillars"].items():
        pillars_out[pname] = {}
        task_ndcgs: list[float] = []
        for tname, task in tasks.items():
            t0 = time.perf_counter()
            pairs: list[tuple[str, str]] = []
            pair_index: list[tuple[str, str]] = []
            for qid, cand_ids in task["candidates"].items():
                q_text = task["queries"][qid]
                for cid in cand_ids:
                    pairs.append((q_text, task["corpus"][cid]))
                    pair_index.append((qid, cid))
            scores = scorer.score_pairs(pairs)
            by_q: dict[str, list[tuple[float, str]]] = defaultdict(list)
            for (qid, cid), sc in zip(pair_index, scores.tolist()):
                by_q[qid].append((float(sc), cid))
            ranked = {
                qid: [cid for _, cid in sorted(items, key=lambda x: (-x[0], x[1]))]
                for qid, items in by_q.items()
            }
            ndcg10 = compute_ndcg_at_k(task["qrels"], ranked, k=10)
            elapsed = time.perf_counter() - t0
            pillars_out[pname][tname] = {
                "ndcg10": round(ndcg10, 6),
                "num_pairs": len(pairs),
                "elapsed_sec": round(elapsed, 2),
                "pairs_per_sec": round(len(pairs) / max(elapsed, 1e-6), 1),
            }
            task_ndcgs.append(ndcg10)
            print(f"  [{pname}/{tname}] NDCG@10={ndcg10:.4f} ({len(pairs)} pairs in {elapsed:.1f}s)")
        pillar_avg = float(np.mean(task_ndcgs))
        pillar_avgs[pname] = round(pillar_avg, 6)
        print(f"==> Pillar {pname} mean NDCG@10 = {pillar_avg:.4f}")

    overall = float(np.mean(list(pillar_avgs.values())))
    print(f"==> OVERALL macro NDCG@10 = {overall:.4f}")
    return {
        "pillars": pillars_out,
        "summary": {
            "beir_ndcg10": pillar_avgs["beir"],
            "code_ndcg10": pillar_avgs["code"],
            "long_context_ndcg10": pillar_avgs["long_context"],
            "overall_ndcg10": round(overall, 6),
        },
    }


def benchmark_latency(
    scorer: BaseScorer,
    suite: dict[str, Any],
    *,
    is_cuda: bool,
    repeats: int = 3,
) -> dict[str, Any]:
    short_pairs = [(q, d) for q, d in suite["latency_batches"]["short_100"]]
    long_pairs = [(q, d) for q, d in suite["latency_batches"]["long_32"]]

    def _time(pairs: list[tuple[str, str]], reps: int) -> dict[str, float]:
        # Warmup
        _ = scorer.score_pairs(pairs[: min(16, len(pairs))])
        if is_cuda:
            import torch
            torch.cuda.synchronize()
        times = []
        for _ in range(reps):
            t0 = time.perf_counter()
            _ = scorer.score_pairs(pairs)
            if is_cuda:
                import torch
                torch.cuda.synchronize()
            times.append(time.perf_counter() - t0)
        med_sec = float(np.median(times))
        pps = len(pairs) / max(med_sec, 1e-9)
        ms_per_100 = (med_sec / len(pairs)) * 100.0 * 1000.0
        return {
            "num_pairs": len(pairs),
            "median_sec": round(med_sec, 4),
            "pairs_per_sec": round(pps, 1),
            "ms_per_100_pairs": round(ms_per_100, 2),
        }

    return {
        "short_100": _time(short_pairs, repeats),
        "long_32": _time(long_pairs, max(1, repeats - 1)),
    }


def _resolve_hub_onnx(model_id: str) -> Path | None:
    local = Path(model_id)
    if local.is_dir():
        cand = local / "onnx" / "model.onnx"
        return cand if cand.is_file() else None
    try:
        files = HfApi().list_repo_files(model_id)
        if "onnx/model.onnx" in files:
            return Path(hf_hub_download(model_id, "onnx/model.onnx"))
    except Exception:
        return None
    return None


def main() -> int:
    load_dotenv()
    parser = argparse.ArgumentParser(description="Locked evaluation harness for rerank-mini")
    parser.add_argument("--suite-path", type=Path, default=get_data_dir() / "eval_locked" / SUITE_FILENAME)
    parser.add_argument("--build-suite", action="store_true", help="Build the immutable v1 eval suite and manifest")
    parser.add_argument("--model", type=str, help="HF model ID or local checkpoint directory")
    parser.add_argument("--backend", choices=["pytorch", "onnx", "http"], default="pytorch")
    parser.add_argument("--onnx-path", type=Path, help="Path to ONNX model (for --backend onnx or extra CPU latency)")
    parser.add_argument("--tokenizer", type=str, help="Tokenizer path/ID (defaults to --model)")
    parser.add_argument("--provider", choices=["cpu", "cuda"], default="cpu", help="ONNX Runtime EP")
    parser.add_argument("--cpu-threads", type=int, default=4, help="CPU intra-op threads for ONNX Runtime")
    parser.add_argument("--dtype", choices=["bf16", "fp16", "fp32"], default="bf16")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--max-seq-length", type=int, default=None)
    parser.add_argument("--endpoint", type=str, help="HTTP /v1/rerank URL for --backend http")
    parser.add_argument("--skip-latency", action="store_true")
    parser.add_argument("--skip-cpu-onnx-latency", action="store_true")
    parser.add_argument("--out", type=Path, help="Output JSON report path")
    parser.add_argument("--log-experiment", action="store_true", help="Append summary row to experiments.jsonl")
    parser.add_argument("--tag", type=str, default="")
    parser.add_argument("--notes", type=str, default="")
    args = parser.parse_args()

    if args.build_suite:
        build_suite(args.suite_path)
        return 0

    if not args.model and args.backend != "http":
        parser.error("--model is required unless --build-suite or --backend http is used")

    suite = load_verified_suite(args.suite_path)
    tokenizer_id = args.tokenizer or args.model

    print(f"Evaluating model={args.model} backend={args.backend} provider={args.provider}")
    if args.backend == "pytorch":
        scorer: BaseScorer = PyTorchCrossEncoderScorer(
            args.model, device=args.device, dtype_name=args.dtype, max_seq_length=args.max_seq_length
        )
        max_len_used = scorer.max_seq_length
        num_params = scorer.num_params
    elif args.backend == "onnx":
        if not args.onnx_path:
            parser.error("--onnx-path is required for --backend onnx")
        scorer = OnnxCrossEncoderScorer(
            args.onnx_path,
            tokenizer_id=tokenizer_id,
            provider=args.provider,
            num_threads=args.cpu_threads,
            max_seq_length=args.max_seq_length or 8192,
        )
        max_len_used = scorer.max_seq_length
        num_params = None
    else:
        if not args.endpoint:
            parser.error("--endpoint is required for --backend http")
        scorer = HttpRerankScorer(args.endpoint, model_name=args.model or "gguf")
        max_len_used = args.max_seq_length or 8192
        num_params = None

    eval_res = evaluate_suite(scorer, suite)
    latency: dict[str, Any] = {}
    if not args.skip_latency:
        is_cuda = (args.backend == "pytorch" and args.device.startswith("cuda")) or (
            args.backend == "onnx" and args.provider == "cuda"
        )
        key = f"{args.backend}_{'cuda' if is_cuda else 'cpu'}"
        print(f"Benchmarking latency ({key})...")
        latency[key] = benchmark_latency(scorer, suite, is_cuda=is_cuda, repeats=3)
        print(f"  short_100: {latency[key]['short_100']}")
        print(f"  long_32:   {latency[key]['long_32']}")

        if args.backend == "pytorch" and not args.skip_cpu_onnx_latency:
            onnx_file = args.onnx_path or _resolve_hub_onnx(args.model)
            if onnx_file and onnx_file.is_file():
                print(f"Benchmarking CPU ONNX latency ({onnx_file.name}, {args.cpu_threads} threads)...")
                cpu_scorer = OnnxCrossEncoderScorer(
                    onnx_file,
                    tokenizer_id=tokenizer_id,
                    provider="cpu",
                    num_threads=args.cpu_threads,
                    max_seq_length=max_len_used,
                )
                latency["onnx_cpu"] = benchmark_latency(cpu_scorer, suite, is_cuda=False, repeats=2)
                print(f"  onnx_cpu short_100: {latency['onnx_cpu']['short_100']}")
                print(f"  onnx_cpu long_32:   {latency['onnx_cpu']['long_32']}")

    report = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "tag": args.tag or args.model,
        "model": args.model,
        "backend": args.backend,
        "provider": args.provider if args.backend == "onnx" else args.device,
        "dtype": args.dtype if args.backend == "pytorch" else None,
        "max_seq_length": max_len_used,
        "num_params": num_params,
        "notes": args.notes,
        "summary": eval_res["summary"],
        "pillars": eval_res["pillars"],
        "latency": latency,
    }

    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        print(f"Saved report -> {args.out}")

    if args.log_experiment:
        exp_path = REPO_ROOT / "experiments.jsonl"
        row = {
            "timestamp_utc": report["timestamp_utc"],
            "tag": report["tag"],
            "model": report["model"],
            "backend": report["backend"],
            **report["summary"],
            "gpu_short_pps": latency.get("pytorch_cuda", latency.get("onnx_cuda", {})).get("short_100", {}).get("pairs_per_sec"),
            "cpu_onnx_short_pps": latency.get("onnx_cpu", {}).get("short_100", {}).get("pairs_per_sec"),
            "notes": args.notes,
        }
        with exp_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(row) + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
