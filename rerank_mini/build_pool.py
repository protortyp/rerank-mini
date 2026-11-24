"""Assemble the 100% Apache-2.0 / MIT training pool with strict eval deduplication.

Produces two tracks under <RERANK_DATA_DIR>/pool/:
  - track_a_prescored/: Curated Apache-2.0/MIT English subsets from
    KaLM-Embedding/KaLM-reranker-training-data (MIT), already scored with
    Qwen/Qwen3-Reranker-8B (yes_logit - no_logit), stratified to 12 docs/query.
  - track_b_unscored/: Freshly mined Code, Agent Tool/API, and Long-Context
    (1K-6K token) query-candidate shards (10 docs/query via BM25 hard-negative
    mining) ready for Qwen/Qwen3-Reranker-8B scoring on Vast.ai H200 or Colab.

Every query and positive document is strictly deduplicated against the locked
evaluation suite (`suite_v1.json`) via normalized SHA-256 hashes.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import random
import re
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from datasets import load_dataset
from huggingface_hub import HfApi, hf_hub_download
from sklearn.feature_extraction.text import TfidfVectorizer
from tqdm import tqdm

from rerank_mini.env import REPO_ROOT, get_data_dir, get_hf_token, load_dotenv
from rerank_mini.teacher_score import DOMAIN_INSTRUCTIONS, normalize_query_scores

TOKEN_RE = re.compile(r"[a-z0-9_]+")
WS_RE = re.compile(r"\s+")

# Strictly verified Apache-2.0 / MIT English subsets of KaLM-reranker-training-data.
KALM_ALLOWED_SUBSETS: list[tuple[str, str, str, int]] = [
    # (subdir, domain, spdx_license, max_queries)
    ("gooaq", "web_qa", "apache-2.0", 45000),
    ("codefeedback", "code", "apache-2.0", 45000),
    ("pubmedqa", "web_qa", "mit", 40000),
    ("openorca", "web_qa", "mit", 35000),
    ("esci", "web_qa", "apache-2.0", 25000),
    ("rag-dataset-12000", "web_qa", "apache-2.0", 9500),
    ("paq_pairs", "web_qa", "mit", 40000),
    ("triviaqa", "web_qa", "apache-2.0", 40000),
]


def _norm_hash(text: str) -> str:
    norm = WS_RE.sub(" ", text.strip().lower())
    return hashlib.sha256(norm.encode("utf-8")).hexdigest()[:24]


def _tokenize_bm25(text: str) -> list[str]:
    return TOKEN_RE.findall(text.lower())


def load_eval_blocklist(suite_path: Path) -> tuple[set[str], set[str]]:
    """Collect normalized hashes of all locked eval queries and documents."""
    suite = json.loads(suite_path.read_text(encoding="utf-8"))
    q_hashes: set[str] = set()
    d_hashes: set[str] = set()
    for tasks in suite["pillars"].values():
        for task in tasks.values():
            for q in task["queries"].values():
                q_hashes.add(_norm_hash(q))
            for d in task["corpus"].values():
                d_hashes.add(_norm_hash(d))
    return q_hashes, d_hashes


def _stratified_sample_indices(scores: list[float], k: int, rng: random.Random) -> list[int]:
    """Select k negatives spanning hard (top), medium, and easy (low) teacher scores."""
    n = len(scores)
    if n <= k:
        return list(range(n))
    ranked = sorted(range(n), key=lambda i: scores[i], reverse=True)
    n_hard = max(1, int(round(k * 0.55)))
    n_med = max(1, int(round(k * 0.30)))
    n_easy = max(0, k - n_hard - n_med)
    hard_pool = ranked[: max(n_hard, n // 3)]
    med_pool = ranked[len(hard_pool) : max(len(hard_pool) + n_med, (2 * n) // 3)]
    easy_pool = ranked[len(hard_pool) + len(med_pool) :]

    chosen: list[int] = []
    chosen.extend(hard_pool[:n_hard])
    if med_pool and n_med > 0:
        chosen.extend(rng.sample(med_pool, min(n_med, len(med_pool))))
    if easy_pool and n_easy > 0:
        chosen.extend(rng.sample(easy_pool, min(n_easy, len(easy_pool))))
    # Top up from remaining hardest if any bucket ran short.
    seen = set(chosen)
    for idx in ranked:
        if len(chosen) >= k:
            break
        if idx not in seen:
            chosen.append(idx)
            seen.add(idx)
    return chosen[:k]


def build_track_a(
    out_dir: Path,
    eval_q_hashes: set[str],
    eval_d_hashes: set[str],
    *,
    docs_per_query: int = 12,
    shard_size: int = 10000,
    seed: int = 42,
) -> dict[str, Any]:
    """Extract and deduplicate pre-scored Apache-2.0/MIT English subsets from KaLM."""
    out_dir.mkdir(parents=True, exist_ok=True)
    rng = random.Random(seed)
    stats: dict[str, Any] = {"subsets": {}, "total_queries": 0, "total_pairs": 0, "dropped_eval_overlap": 0}
    buffer: list[dict[str, Any]] = []
    shard_idx = 0

    def _flush() -> None:
        nonlocal shard_idx, buffer
        if not buffer:
            return
        path = out_dir / f"track_a_{shard_idx:04d}.parquet"
        pq.write_table(pa.Table.from_pylist(buffer), path, compression="zstd")
        shard_idx += 1
        buffer = []

    for subdir, domain, spdx, max_q in KALM_ALLOWED_SUBSETS:
        print(f"[Track A] Loading KaLM/{subdir} ({spdx}, cap={max_q})...", flush=True)
        ds = load_dataset("KaLM-Embedding/KaLM-reranker-training-data", data_dir=subdir, split="train")
        indices = list(range(len(ds)))
        rng.shuffle(indices)
        kept = 0
        dropped = 0
        seen_q: set[str] = set()

        for i in indices:
            if kept >= max_q:
                break
            row = ds[i]
            query = str(row["query"]).strip()
            if not query or len(query) < 4:
                continue
            qh = _norm_hash(query)
            if qh in eval_q_hashes or qh in seen_q:
                dropped += 1
                continue
            pos_list = list(row["pos"] or [])
            neg_list = list(row["neg"] or [])
            pos_scores = [float(x) for x in (row["pos_scores"] or [])]
            neg_scores = [float(x) for x in (row["neg_scores"] or [])]
            if not pos_list or not neg_list or len(pos_list) != len(pos_scores) or len(neg_list) != len(neg_scores):
                continue

            # Pick the highest-scoring positive and check eval doc overlap.
            best_p_idx = int(np.argmax(pos_scores))
            pos_doc = str(pos_list[best_p_idx]).strip()
            pos_sc = float(pos_scores[best_p_idx])
            if not pos_doc or _norm_hash(pos_doc) in eval_d_hashes:
                dropped += 1
                continue

            # Filter clean negatives and stratify across hard/med/easy teacher scores.
            clean_negs: list[tuple[str, float]] = []
            seen_d = {_norm_hash(pos_doc)}
            for nd, ns in zip(neg_list, neg_scores):
                nd_s = str(nd).strip()
                if not nd_s:
                    continue
                nh = _norm_hash(nd_s)
                if nh in seen_d or nh in eval_d_hashes:
                    continue
                seen_d.add(nh)
                clean_negs.append((nd_s, float(ns)))
            if len(clean_negs) < 4:
                continue

            k_neg = min(docs_per_query - 1, len(clean_negs))
            pick = _stratified_sample_indices([x[1] for x in clean_negs], k_neg, rng)
            docs = [pos_doc] + [clean_negs[j][0] for j in pick]
            labels = [1] + [0] * len(pick)
            raw_logits = [pos_sc] + [clean_negs[j][1] for j in pick]

            # Shuffle candidate order deterministically so position 0 is not always positive.
            perm = list(range(len(docs)))
            rng.shuffle(perm)
            docs = [docs[p] for p in perm]
            labels = [labels[p] for p in perm]
            raw_logits = [raw_logits[p] for p in perm]
            norm = normalize_query_scores(raw_logits, temperature=1.0)

            seen_q.add(qh)
            buffer.append(
                {
                    "query_id": f"kalm_{subdir}_{kept}",
                    "source": f"kalm:{subdir}",
                    "domain": domain,
                    "license": spdx,
                    "query": query,
                    "docs": docs,
                    "doc_labels": labels,
                    "teacher_logits": norm["raw_logits"],
                    "teacher_logits_shifted": norm["max_shifted_logits"],
                }
            )
            kept += 1
            stats["total_queries"] += 1
            stats["total_pairs"] += len(docs)
            if len(buffer) >= shard_size:
                _flush()

        stats["subsets"][subdir] = {"kept_queries": kept, "dropped": dropped, "license": spdx, "domain": domain}
        stats["dropped_eval_overlap"] += dropped
        print(f"  -> kept={kept}, dropped={dropped}", flush=True)

    _flush()
    stats["num_shards"] = shard_idx
    return stats


def _mine_bm25_hard_negatives(
    pairs: list[tuple[str, str]],
    *,
    source: str,
    domain: str,
    spdx: str,
    docs_per_query: int,
    eval_q_hashes: set[str],
    eval_d_hashes: set[str],
    rng: random.Random,
) -> list[dict[str, Any]]:
    """Mine stratified lexical hard + medium + random negatives via vectorized sublinear TF-IDF."""
    clean: list[tuple[str, str]] = []
    seen_q: set[str] = set()
    for q, d in pairs:
        q_s = q.strip()
        d_s = d.strip()
        if len(q_s) < 8 or len(d_s) < 20:
            continue
        qh = _norm_hash(q_s)
        dh = _norm_hash(d_s)
        if qh in eval_q_hashes or dh in eval_d_hashes or qh in seen_q:
            continue
        seen_q.add(qh)
        clean.append((q_s, d_s))
    if len(clean) < docs_per_query + 5:
        return []

    queries = [q for q, _ in clean]
    corpus_docs = [d for _, d in clean]
    vec = TfidfVectorizer(
        sublinear_tf=True,
        max_features=120000,
        token_pattern=r"[a-zA-Z0-9_]{2,}",
        dtype=np.float32,
    )
    doc_mat = vec.fit_transform([d[:12000] for d in corpus_docs])
    q_mat = vec.transform(queries)

    n_docs = len(corpus_docs)
    k_neg = docs_per_query - 1
    k_hard = max(1, int(round(k_neg * 0.55)))
    k_med = max(1, int(round(k_neg * 0.25)))
    pool_top = min(n_docs, 60)

    out_rows: list[dict[str, Any]] = []
    instruct = DOMAIN_INSTRUCTIONS[domain]
    chunk_sz = 512
    for start in tqdm(range(0, n_docs, chunk_sz), desc=f"Vectorized mining {source}"):
        end = min(n_docs, start + chunk_sz)
        sim = (q_mat[start:end] @ doc_mat.T).toarray()
        for local_i in range(end - start):
            idx = start + local_i
            q, pos_doc = clean[idx]
            scores = sim[local_i]
            top_idx = np.argpartition(-scores, pool_top - 1)[:pool_top]
            top_idx = top_idx[np.argsort(-scores[top_idx], kind="mergesort")]

            pos_h = _norm_hash(pos_doc)
            negs: list[str] = []
            seen_h = {pos_h}

            ranked_cand = [int(i) for i in top_idx if int(i) != idx]
            for ci in ranked_cand[: k_hard * 2]:
                if len(negs) >= k_hard:
                    break
                cand = corpus_docs[ci]
                ch = _norm_hash(cand)
                if ch not in seen_h:
                    seen_h.add(ch)
                    negs.append(cand)

            med_slice = ranked_cand[max(k_hard, 12) :]
            rng.shuffle(med_slice)
            for ci in med_slice:
                if len(negs) >= k_hard + k_med:
                    break
                cand = corpus_docs[ci]
                ch = _norm_hash(cand)
                if ch not in seen_h:
                    seen_h.add(ch)
                    negs.append(cand)

            attempts = 0
            while len(negs) < k_neg and attempts < 50:
                attempts += 1
                ci = rng.randrange(n_docs)
                if ci == idx:
                    continue
                cand = corpus_docs[ci]
                ch = _norm_hash(cand)
                if ch not in seen_h:
                    seen_h.add(ch)
                    negs.append(cand)

            if len(negs) < 4:
                continue
            docs = [pos_doc, *negs]
            labels = [1] + [0] * len(negs)
            perm = list(range(len(docs)))
            rng.shuffle(perm)
            out_rows.append(
                {
                    "query_id": f"{source}_{idx}",
                    "source": source,
                    "domain": domain,
                    "license": spdx,
                    "instruction": instruct,
                    "query": q,
                    "docs": [docs[p] for p in perm],
                    "doc_labels": [labels[p] for p in perm],
                }
            )
    return out_rows


def _build_mldr_train_rows(
    eval_q_hashes: set[str],
    eval_d_hashes: set[str],
    *,
    docs_per_query: int = 10,
    max_queries: int = 10000,
    rng: random.Random,
) -> list[dict[str, Any]]:
    train_gz = hf_hub_download("Shitao/MLDR", "mldr-v1.0-en/train.jsonl.gz", repo_type="dataset")
    rows: list[dict[str, Any]] = []
    instruct = DOMAIN_INSTRUCTIONS["long_doc"]
    with gzip.open(train_gz, "rt", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if len(rows) >= max_queries:
                break
            rec = json.loads(line)
            q = str(rec["query"]).strip()
            if not q or _norm_hash(q) in eval_q_hashes:
                continue
            pos_list = rec.get("positive_passages") or []
            neg_list = rec.get("negative_passages") or []
            if not pos_list or len(neg_list) < 4:
                continue
            pos_doc = str(pos_list[0]["text"]).strip()
            if not pos_doc or _norm_hash(pos_doc) in eval_d_hashes:
                continue
            # Truncate extremely long docs to ~5,000 words (~6,500 tok) so they fit cleanly in 8K.
            pos_doc = " ".join(pos_doc.split()[:4800])
            negs: list[str] = []
            seen = {_norm_hash(pos_doc)}
            k_neg = docs_per_query - 1
            # Keep first half hardest pre-mined negatives + random slice from remaining negatives.
            n_hard = k_neg // 2 + 1
            cand_indices = list(range(min(n_hard, len(neg_list))))
            rem = list(range(len(cand_indices), len(neg_list)))
            rng.shuffle(rem)
            cand_indices.extend(rem)
            for ni in cand_indices:
                if len(negs) >= k_neg:
                    break
                nt = " ".join(str(neg_list[ni]["text"]).strip().split()[:4800])
                nh = _norm_hash(nt)
                if nt and nh not in seen and nh not in eval_d_hashes:
                    seen.add(nh)
                    negs.append(nt)
            if len(negs) < 4:
                continue
            docs = [pos_doc, *negs]
            labels = [1] + [0] * len(negs)
            perm = list(range(len(docs)))
            rng.shuffle(perm)
            rows.append(
                {
                    "query_id": f"mldr_en_{i}",
                    "source": "mldr_en_train",
                    "domain": "long_doc",
                    "license": "mit",
                    "instruction": instruct,
                    "query": q,
                    "docs": [docs[p] for p in perm],
                    "doc_labels": [labels[p] for p in perm],
                }
            )
    return rows


def _extract_glaive_tool_pairs(max_pairs: int, rng: random.Random) -> list[tuple[str, str]]:
    """Extract (user_request, tool_definition_doc) pairs from glaive-function-calling-v2 (Apache-2.0)."""
    ds = load_dataset("glaiveai/glaive-function-calling-v2", split="train")
    indices = list(range(len(ds)))
    rng.shuffle(indices)
    pairs: list[tuple[str, str]] = []
    for i in indices:
        if len(pairs) >= max_pairs:
            break
        row = ds[i]
        sys_txt = str(row.get("system") or "")
        chat_txt = str(row.get("chat") or "")
        if "functioncall" not in chat_txt.lower() and '"name"' not in sys_txt:
            continue
        m_user = re.search(r"USER:\s*(.+?)(?:\nASSISTANT:|$)", chat_txt, flags=re.DOTALL)
        m_fn = re.search(r"(\{.*\})", sys_txt, flags=re.DOTALL)
        if not m_user or not m_fn:
            continue
        q = m_user.group(1).strip()[:600]
        tool_doc = m_fn.group(1).strip()[:3000]
        if len(q) >= 12 and len(tool_doc) >= 40:
            pairs.append((q, tool_doc))
    return pairs


def _extract_narrativeqa_long_pairs(max_pairs: int, rng: random.Random) -> list[tuple[str, str]]:
    """Extract (question, 2K-4K word document window) pairs from deepmind/narrativeqa (Apache-2.0)."""
    ds = load_dataset("deepmind/narrativeqa", split="train")
    indices = list(range(len(ds)))
    rng.shuffle(indices)
    pairs: list[tuple[str, str]] = []
    for i in indices:
        if len(pairs) >= max_pairs:
            break
        row = ds[i]
        q = str((row.get("question") or {}).get("text") or "").strip()
        doc_obj = row.get("document") or {}
        summary = str((doc_obj.get("summary") or {}).get("text") or "").strip()
        full_text = str(doc_obj.get("text") or "").strip()
        words = full_text.split()
        if len(q) < 10 or len(words) < 1200:
            continue
        # Anchor window on the best answer keyword hit inside the long text so the chunk is answerable.
        ans_list = [str(a.get("text", "")).strip() for a in (row.get("answers") or []) if a.get("text")]
        window_words = 2600
        start_w = 0
        if ans_list:
            needle = ans_list[0].lower()
            pos_char = full_text.lower().find(needle)
            if pos_char >= 0:
                approx_w = len(full_text[:pos_char].split())
                offset = rng.randint(200, 1800)
                start_w = max(0, min(len(words) - window_words, approx_w - offset))
        chunk = " ".join(words[start_w : start_w + window_words])
        if summary and len(chunk.split()) < 1500:
            chunk = f"{summary}\n\n{chunk}"
        pairs.append((q, chunk))
    return pairs


def build_track_b(
    out_dir: Path,
    eval_q_hashes: set[str],
    eval_d_hashes: set[str],
    *,
    docs_per_query: int = 10,
    shard_size: int = 5000,
    seed: int = 42,
) -> dict[str, Any]:
    """Build freshly mined Code, Tool/API, and Long-Context shards for Qwen3-Reranker-8B scoring."""
    out_dir.mkdir(parents=True, exist_ok=True)
    rng = random.Random(seed)
    all_rows: list[dict[str, Any]] = []
    source_stats: dict[str, Any] = {}

    # 1. MLDR-en long-context (MIT) — uses Shitao/MLDR's own pre-mined hard negatives!
    print("[Track B 1/5] Building MLDR-en long-context train (MIT)...", flush=True)
    mldr_rows = _build_mldr_train_rows(eval_q_hashes, eval_d_hashes, docs_per_query=docs_per_query, max_queries=10000, rng=rng)
    all_rows.extend(mldr_rows)
    source_stats["mldr_en_train"] = {"queries": len(mldr_rows), "license": "mit", "domain": "long_doc"}
    print(f"  -> MLDR-en train queries: {len(mldr_rows)}", flush=True)

    # 2. NarrativeQA long-context chunks (Apache-2.0)
    print("[Track B 2/5] Mining NarrativeQA long-context chunks (Apache-2.0)...", flush=True)
    nqa_pairs = _extract_narrativeqa_long_pairs(max_pairs=10000, rng=rng)
    nqa_rows = _mine_bm25_hard_negatives(
        nqa_pairs,
        source="narrativeqa_long",
        domain="long_doc",
        spdx="apache-2.0",
        docs_per_query=docs_per_query,
        eval_q_hashes=eval_q_hashes,
        eval_d_hashes=eval_d_hashes,
        rng=rng,
    )
    all_rows.extend(nqa_rows)
    source_stats["narrativeqa_long"] = {"queries": len(nqa_rows), "license": "apache-2.0", "domain": "long_doc"}
    print(f"  -> NarrativeQA long queries: {len(nqa_rows)}", flush=True)

    # 3. CodeSearchNet-Python (Apache-2.0) — strictly deduped against eval!
    print("[Track B 3/5] Mining Nan-Do/code-search-net-python (Apache-2.0)...", flush=True)
    csn = load_dataset("Nan-Do/code-search-net-python", split="train")
    csn_idx = list(range(len(csn)))
    rng.shuffle(csn_idx)
    csn_pairs: list[tuple[str, str]] = []
    for i in csn_idx:
        if len(csn_pairs) >= 25000:
            break
        r = csn[i]
        docstr = str(r.get("docstring") or "").strip()
        code = str(r.get("code") or "").strip()
        if len(docstr) >= 15 and 40 <= len(code) <= 12000:
            csn_pairs.append((docstr[:600], code))
    csn_rows = _mine_bm25_hard_negatives(
        csn_pairs,
        source="codesearchnet_python",
        domain="code",
        spdx="apache-2.0",
        docs_per_query=docs_per_query,
        eval_q_hashes=eval_q_hashes,
        eval_d_hashes=eval_d_hashes,
        rng=rng,
    )
    all_rows.extend(csn_rows)
    source_stats["codesearchnet_python"] = {"queries": len(csn_rows), "license": "apache-2.0", "domain": "code"}
    print(f"  -> CodeSearchNet-Python queries: {len(csn_rows)}", flush=True)

    # 4. Magicoder-OSS-Instruct-75K multi-language code retrieval (MIT)
    print("[Track B 4/6] Mining ise-uiuc/Magicoder-OSS-Instruct-75K (MIT)...", flush=True)
    mc = load_dataset("ise-uiuc/Magicoder-OSS-Instruct-75K", split="train")
    mc_idx = list(range(len(mc)))
    rng.shuffle(mc_idx)
    mc_pairs: list[tuple[str, str]] = []
    for i in mc_idx:
        if len(mc_pairs) >= 25000:
            break
        r = mc[i]
        prob = str(r.get("problem") or "").strip()
        sol = str(r.get("solution") or "").strip()
        if len(prob) >= 15 and 60 <= len(sol) <= 12000:
            mc_pairs.append((prob[:700], sol))
    mc_rows = _mine_bm25_hard_negatives(
        mc_pairs,
        source="magicoder_oss_75k",
        domain="code",
        spdx="mit",
        docs_per_query=docs_per_query,
        eval_q_hashes=eval_q_hashes,
        eval_d_hashes=eval_d_hashes,
        rng=rng,
    )
    all_rows.extend(mc_rows)
    source_stats["magicoder_oss_75k"] = {"queries": len(mc_rows), "license": "mit", "domain": "code"}
    print(f"  -> Magicoder-OSS queries: {len(mc_rows)}", flush=True)

    # 5. Glaive Code Assistant v2 technical Q&A (Apache-2.0)
    print("[Track B 5/6] Mining glaiveai/glaive-code-assistant-v2 (Apache-2.0)...", flush=True)
    gc = load_dataset("glaiveai/glaive-code-assistant-v2", split="train", streaming=True)
    gc_pairs: list[tuple[str, str]] = []
    for r in gc:
        if len(gc_pairs) >= 20000:
            break
        q = str(r.get("question") or "").strip()
        a = str(r.get("answer") or "").strip()
        if len(q) >= 15 and 60 <= len(a) <= 12000:
            gc_pairs.append((q[:700], a))
    gc_rows = _mine_bm25_hard_negatives(
        gc_pairs,
        source="glaive_code_qa",
        domain="code",
        spdx="apache-2.0",
        docs_per_query=docs_per_query,
        eval_q_hashes=eval_q_hashes,
        eval_d_hashes=eval_d_hashes,
        rng=rng,
    )
    all_rows.extend(gc_rows)
    source_stats["glaive_code_qa"] = {"queries": len(gc_rows), "license": "apache-2.0", "domain": "code"}
    print(f"  -> Glaive Code QA queries: {len(gc_rows)}", flush=True)

    # 6. Glaive Function Calling v2 tool/API retrieval (Apache-2.0)
    print("[Track B 6/6] Mining glaive-function-calling-v2 tool/API docs (Apache-2.0)...", flush=True)
    tool_pairs = _extract_glaive_tool_pairs(max_pairs=15000, rng=rng)
    tool_rows = _mine_bm25_hard_negatives(
        tool_pairs,
        source="glaive_tool_api",
        domain="tool_api",
        spdx="apache-2.0",
        docs_per_query=docs_per_query,
        eval_q_hashes=eval_q_hashes,
        eval_d_hashes=eval_d_hashes,
        rng=rng,
    )
    all_rows.extend(tool_rows)
    source_stats["glaive_tool_api"] = {"queries": len(tool_rows), "license": "apache-2.0", "domain": "tool_api"}
    print(f"  -> Glaive Tool/API queries: {len(tool_rows)}", flush=True)

    # Shuffle across all Track B sources and write evenly sized Parquet shards.
    rng.shuffle(all_rows)
    shard_idx = 0
    for start in range(0, len(all_rows), shard_size):
        chunk = all_rows[start : start + shard_size]
        path = out_dir / f"track_b_{shard_idx:04d}.parquet"
        pq.write_table(pa.Table.from_pylist(chunk), path, compression="zstd")
        shard_idx += 1

    total_pairs = sum(len(r["docs"]) for r in all_rows)
    return {
        "sources": source_stats,
        "total_queries": len(all_rows),
        "total_pairs": total_pairs,
        "num_shards": shard_idx,
    }


def upload_directory_to_hf(local_dir: Path, repo_id: str, path_in_repo: str) -> None:
    api = HfApi(token=get_hf_token())
    api.create_repo(repo_id=repo_id, repo_type="dataset", private=True, exist_ok=True)
    api.upload_folder(
        folder_path=str(local_dir),
        path_in_repo=path_in_repo,
        repo_id=repo_id,
        repo_type="dataset",
    )


def main() -> int:
    load_dotenv()
    parser = argparse.ArgumentParser(description="Build Apache-2.0/MIT training pool shards")
    parser.add_argument("--suite-path", type=Path, default=get_data_dir() / "eval_locked" / "suite_v1.json")
    parser.add_argument("--out-dir", type=Path, default=get_data_dir() / "pool")
    parser.add_argument("--track", choices=["a", "b", "both"], default="both")
    parser.add_argument("--upload-hf-repo", type=str, default="", help="Optional private HF dataset repo to upload shards")
    args = parser.parse_args()

    eval_q_hashes, eval_d_hashes = load_eval_blocklist(args.suite_path)
    print(f"Loaded eval blocklist: {len(eval_q_hashes)} queries, {len(eval_d_hashes)} docs", flush=True)

    manifest: dict[str, Any] = {}
    manifest_file = REPO_ROOT / "results" / "training_pool_manifest.json"
    if manifest_file.is_file():
        manifest = json.loads(manifest_file.read_text(encoding="utf-8"))

    if args.track in ("a", "both"):
        manifest["track_a_prescored"] = build_track_a(
            args.out_dir / "track_a_prescored", eval_q_hashes, eval_d_hashes
        )
        if args.upload_hf_repo:
            upload_directory_to_hf(args.out_dir / "track_a_prescored", args.upload_hf_repo, "track_a_prescored")

    if args.track in ("b", "both"):
        manifest["track_b_unscored"] = build_track_b(
            args.out_dir / "track_b_unscored", eval_q_hashes, eval_d_hashes
        )
        if args.upload_hf_repo:
            upload_directory_to_hf(args.out_dir / "track_b_unscored", args.upload_hf_repo, "track_b_unscored")

    manifest_file.parent.mkdir(parents=True, exist_ok=True)
    manifest_file.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print("\nTraining pool build complete:")
    print(json.dumps(manifest, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
