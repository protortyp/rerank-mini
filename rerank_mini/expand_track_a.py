"""Extract remaining unused Apache-2.0 / MIT web-search & QA queries from KaLM-reranker-training-data
(triviaqa, gooaq, paq_pairs) that were capped in the initial Track A pass, strictly deduplicated
against both the locked eval suite and existing Track A shards.
"""

from __future__ import annotations

import json
import random
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from datasets import load_dataset
from huggingface_hub import HfApi

from rerank_mini.build_pool import _norm_hash, _stratified_sample_indices, load_eval_blocklist
from rerank_mini.env import get_data_dir, get_hf_token, load_dotenv
from rerank_mini.teacher_score import normalize_query_scores

EXTRA_SUBSETS = [
    ("triviaqa", "web_qa", "apache-2.0"),
    ("gooaq", "web_qa", "apache-2.0"),
    ("paq_pairs", "web_qa", "mit"),
]


def main() -> None:
    load_dotenv()
    data_dir = get_data_dir()
    track_a_dir = data_dir / "pool" / "track_a_prescored"
    eval_q, eval_d = load_eval_blocklist(data_dir / "eval_locked" / "suite_v1.json")

    seen_q = set(eval_q)
    existing = sorted(track_a_dir.glob("track_a_*.parquet"))
    for p in existing:
        tbl = pq.read_table(p, columns=["query"])
        for q in tbl["query"].to_pylist():
            seen_q.add(_norm_hash(str(q)))
    next_idx = len(existing)
    print(f"Loaded {len(existing)} existing Track A shards ({len(seen_q)} seen query hashes)", flush=True)

    rng = random.Random(1337)
    buf = []
    added_by_sub = {}
    new_files: list[Path] = []

    def _flush() -> None:
        nonlocal next_idx, buf
        if not buf:
            return
        path = track_a_dir / f"track_a_{next_idx:04d}.parquet"
        pq.write_table(pa.Table.from_pylist(buf), path, compression="zstd")
        new_files.append(path)
        next_idx += 1
        buf = []

    for subdir, domain, spdx in EXTRA_SUBSETS:
        ds = load_dataset("KaLM-Embedding/KaLM-reranker-training-data", data_dir=subdir, split="train")
        kept = 0
        for row in ds:
            query = str(row["query"]).strip()
            if len(query) < 4:
                continue
            qh = _norm_hash(query)
            if qh in seen_q:
                continue
            pos_list = list(row["pos"] or [])
            neg_list = list(row["neg"] or [])
            pos_scores = [float(x) for x in (row["pos_scores"] or [])]
            neg_scores = [float(x) for x in (row["neg_scores"] or [])]
            if not pos_list or not neg_list or len(pos_list) != len(pos_scores) or len(neg_list) != len(neg_scores):
                continue
            best_p = int(np.argmax(pos_scores))
            pos_doc = str(pos_list[best_p]).strip()
            pos_sc = float(pos_scores[best_p])
            if not pos_doc or _norm_hash(pos_doc) in eval_d:
                continue
            clean_negs = []
            seen_d = {_norm_hash(pos_doc)}
            for nd, ns in zip(neg_list, neg_scores):
                nd_s = str(nd).strip()
                if not nd_s:
                    continue
                nh = _norm_hash(nd_s)
                if nh in seen_d or nh in eval_d:
                    continue
                seen_d.add(nh)
                clean_negs.append((nd_s, float(ns)))
            if len(clean_negs) < 4:
                continue
            k_neg = min(11, len(clean_negs))
            pick = _stratified_sample_indices([x[1] for x in clean_negs], k_neg, rng)
            docs = [pos_doc] + [clean_negs[j][0] for j in pick]
            labels = [1] + [0] * len(pick)
            raw_logits = [pos_sc] + [clean_negs[j][1] for j in pick]
            perm = list(range(len(docs)))
            rng.shuffle(perm)
            docs = [docs[p] for p in perm]
            labels = [labels[p] for p in perm]
            raw_logits = [raw_logits[p] for p in perm]
            norm = normalize_query_scores(raw_logits, temperature=1.0)
            seen_q.add(qh)
            buf.append(
                {
                    "query_id": f"kalm_{subdir}_extra_{kept}",
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
            if len(buf) >= 10000:
                _flush()
        added_by_sub[subdir] = kept
        print(f"[extra {subdir}] added {kept} new queries", flush=True)

    _flush()
    api = HfApi(token=get_hf_token())
    for f in new_files:
        api.upload_file(
            path_or_fileobj=str(f),
            path_in_repo=f"track_a_prescored/{f.name}",
            repo_id="christianengel/rerank-mini-distill-shards",
            repo_type="dataset",
        )
        print(f"Uploaded {f.name}", flush=True)
    print("Done:", json.dumps(added_by_sub, indent=2))


if __name__ == "__main__":
    main()
