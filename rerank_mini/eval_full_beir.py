"""Evaluate rerankers on full standard BEIR test splits (top-100 BM25 candidates):
  - SciFact (300 test queries)
  - NFCorpus (323 test queries)
  - FiQA2018 (648 test queries)
  - TREC-COVID (50 test queries)
  - SCIDOCS (1,000 test queries)
Total: 2,321 standard BEIR test queries (232,100 query-document pairs).

Complements immutable eval_locked.py without modifying it.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
from datasets import load_dataset
from scipy import sparse
from sklearn.feature_extraction.text import CountVectorizer

from eval_locked import PyTorchCrossEncoderScorer, compute_ndcg_at_k
from rerank_mini.env import get_data_dir, load_dotenv

FULL_BEIR_TASKS = [
    ("SciFact", "mteb/scifact"),
    ("NFCorpus", "mteb/nfcorpus"),
    ("FiQA2018", "mteb/fiqa"),
    ("TREC-COVID", "mteb/trec-covid"),
    ("SCIDOCS", "mteb/scidocs"),
]


def _bm25_sparse_topk(
    corpus_texts: list[str],
    query_texts: list[str],
    top_k: int,
    k1: float = 1.5,
    b: float = 0.75,
) -> list[list[int]]:
    """Exact Okapi BM25 top-k retrieval vectorized via scipy sparse matrices."""
    vec = CountVectorizer(token_pattern=r"(?u)\b[a-z0-9_]+\b", lowercase=True, dtype=np.float32)
    X = vec.fit_transform(corpus_texts).tocsr()  # [N_docs, V]
    Q = vec.transform(query_texts).tocsr()       # [N_q, V]

    N = X.shape[0]
    df = np.bincount(X.indices, minlength=X.shape[1]).astype(np.float32)
    idf = np.log((N - df + 0.5) / (df + 0.5) + 1.0).astype(np.float32)
    doc_lens = np.asarray(X.sum(axis=1)).ravel().astype(np.float32)
    avgdl = float(doc_lens.mean()) if N else 1.0

    # Transform doc term frequencies into BM25 term weights: idf * tf*(k1+1) / (tf + k1*(1 - b + b*dl/avgdl))
    X_bm25 = X.copy().astype(np.float32)
    row_idx = np.repeat(np.arange(N, dtype=np.int32), np.diff(X_bm25.indptr))
    denom = X_bm25.data + k1 * (1.0 - b + b * (doc_lens[row_idx] / max(avgdl, 1e-6)))
    X_bm25.data = (X_bm25.data * (k1 + 1.0) / denom) * idf[X_bm25.indices]

    # Binary query term indicator (standard BM25) multiplied by X_bm25^T in chunks
    Q_bin = Q.copy().astype(np.float32)
    Q_bin.data[:] = 1.0
    Xt = X_bm25.T.tocsr()

    k_eff = min(top_k, N)
    out: list[list[int]] = []
    chunk = 128
    for start in range(0, Q_bin.shape[0], chunk):
        scores = (Q_bin[start : start + chunk] @ Xt).toarray()
        for row in scores:
            if k_eff < N:
                idx = np.argpartition(-row, k_eff - 1)[:k_eff]
                idx = idx[np.argsort(-row[idx], kind="mergesort")]
            else:
                idx = np.argsort(-row, kind="mergesort")
            out.append([int(i) for i in idx])
    return out


def build_or_load_full_beir(cache_path: Path, top_k: int = 100) -> dict[str, Any]:
    if cache_path.is_file():
        return json.loads(cache_path.read_text(encoding="utf-8"))

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    suite: dict[str, Any] = {"top_k": top_k, "tasks": {}}
    for name, hf_id in FULL_BEIR_TASKS:
        t0 = time.perf_counter()
        q_ds = load_dataset(hf_id, "queries", split="queries")
        c_ds = load_dataset(hf_id, "corpus", split="corpus")
        r_ds = load_dataset(hf_id, "default", split="test")

        queries = {str(row["_id"]): str(row["text"]) for row in q_ds}
        corpus: dict[str, str] = {}
        for row in c_ds:
            title = str(row.get("title") or "").strip()
            body = str(row["text"])
            corpus[str(row["_id"])] = f"{title}\n{body}" if title else body

        qrels: dict[str, dict[str, float]] = defaultdict(dict)
        for row in r_ds:
            score = float(row.get("score", 1.0))
            if score > 0:
                qrels[str(row["query-id"])][str(row["corpus-id"])] = score

        valid_qids = sorted(qid for qid in qrels if qid in queries)
        doc_ids = sorted(corpus.keys())
        topk_idx = _bm25_sparse_topk(
            [corpus[d] for d in doc_ids],
            [queries[q] for q in valid_qids],
            top_k=top_k,
        )

        candidates: dict[str, list[str]] = {}
        used_docs: dict[str, str] = {}
        sub_queries: dict[str, str] = {}
        sub_qrels: dict[str, dict[str, float]] = {}
        for qid, idxs in zip(valid_qids, topk_idx):
            cands = [doc_ids[i] for i in idxs]
            candidates[qid] = cands
            sub_queries[qid] = queries[qid]
            sub_qrels[qid] = qrels[qid]
            for cid in cands:
                if cid not in used_docs:
                    used_docs[cid] = corpus[cid]

        bm25_ndcg = compute_ndcg_at_k(sub_qrels, candidates, k=10)
        suite["tasks"][name] = {
            "queries": sub_queries,
            "corpus": used_docs,
            "qrels": sub_qrels,
            "candidates": candidates,
            "num_queries": len(sub_queries),
            "bm25_ndcg10": round(bm25_ndcg, 6),
        }
        print(f"Built full BEIR {name}: {len(sub_queries)} queries, BM25 NDCG@10={bm25_ndcg:.4f} ({time.perf_counter()-t0:.1f}s)")

    cache_path.write_text(json.dumps(suite), encoding="utf-8")
    return suite


def eval_models(suite: dict[str, Any], models: list[tuple[str, str]]) -> dict[str, Any]:
    results: dict[str, Any] = {
        "tasks": {k: {"num_queries": v["num_queries"], "bm25_ndcg10": v["bm25_ndcg10"]} for k, v in suite["tasks"].items()},
        "bm25_mean": round(float(np.mean([v["bm25_ndcg10"] for v in suite["tasks"].values()])), 6),
        "models": {},
    }
    for tag, model_id in models:
        print(f"\n=== Full BEIR eval: {tag} ({model_id}) ===")
        scorer = PyTorchCrossEncoderScorer(model_id, device="cuda", dtype_name="bf16")
        per_task: dict[str, float] = {}
        for tname, task in suite["tasks"].items():
            t0 = time.perf_counter()
            pairs: list[tuple[str, str]] = []
            pair_index: list[tuple[str, str]] = []
            for qid, cands in task["candidates"].items():
                q_text = task["queries"][qid]
                for cid in cands:
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
            per_task[tname] = round(ndcg10, 6)
            print(f"  [{tname:10s}] NDCG@10={ndcg10:.4f} ({len(pairs)} pairs in {time.perf_counter()-t0:.1f}s)")
        mean_ndcg = round(float(np.mean(list(per_task.values()))), 6)
        print(f"==> {tag} Full BEIR (5 tasks, 2,321q) mean NDCG@10 = {mean_ndcg:.4f}")
        results["models"][tag] = {
            "model": model_id,
            "num_params": scorer.num_params,
            "max_seq_length": scorer.max_seq_length,
            "mean_ndcg10": mean_ndcg,
            "per_task": per_task,
        }
        del scorer
    return results


def main() -> int:
    load_dotenv()
    parser = argparse.ArgumentParser(description="Evaluate rerankers on 5 full standard BEIR test splits (2,321 queries)")
    parser.add_argument("--cache", type=Path, default=get_data_dir() / "eval_full_beir" / "full_beir_5.json")
    parser.add_argument("--build-only", action="store_true")
    parser.add_argument("--out", type=Path, required=False)
    args = parser.parse_args()

    suite = build_or_load_full_beir(args.cache, top_k=100)
    if args.build_only:
        return 0
    if not args.out:
        parser.error("--out is required unless --build-only is set")
    models = [
        ("ms-marco-MiniLM-L6-v2", "cross-encoder/ms-marco-MiniLM-L6-v2"),
        ("ettin-reranker-32m-v1", "cross-encoder/ettin-reranker-32m-v1"),
        ("mxbai-rerank-xsmall-v1", "mixedbread-ai/mxbai-rerank-xsmall-v1"),
        ("gte-reranker-modernbert-base", "Alibaba-NLP/gte-reranker-modernbert-base"),
        ("rerank-mini-32m", "/mnt/nvme1/chris/rerank-mini/release/rerank-mini-32m"),
    ]
    report = eval_models(suite, models)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"\nSaved full BEIR report -> {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
