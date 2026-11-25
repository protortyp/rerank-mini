"""Build an Apache-2.0 / MIT short-context web-search & Wikipedia QA boost pool (Track C)
from unused allenai/gooaq (Apache-2.0) and facebookresearch/PAQ (MIT) queries, strictly
deduplicated against both the locked eval suite and Track A shards.
"""

from __future__ import annotations

import json
import random
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
from datasets import load_dataset
from huggingface_hub import HfApi

from rerank_mini.build_pool import _mine_bm25_hard_negatives, _norm_hash, load_eval_blocklist
from rerank_mini.env import REPO_ROOT, get_data_dir, get_hf_token, load_dotenv


def main() -> None:
    load_dotenv()
    data_dir = get_data_dir()
    eval_q, eval_d = load_eval_blocklist(data_dir / "eval_locked" / "suite_v1.json")

    # Exclude all existing Track A queries too so Track C is 100% net-new.
    seen_q = set(eval_q)
    for p in sorted((data_dir / "pool" / "track_a_prescored").glob("*.parquet")):
        tbl = pq.read_table(p, columns=["query"])
        for q in tbl["query"].to_pylist():
            seen_q.add(_norm_hash(str(q)))
    print(f"Dedup blocklist: {len(seen_q)} queries, {len(eval_d)} eval docs", flush=True)

    rng = random.Random(2026)
    all_rows = []
    stats = {}

    # 1. 40,000 net-new GooAQ web search QA pairs (Apache-2.0)
    print("[Track C 1/2] Streaming 40K net-new sentence-transformers/gooaq (Apache-2.0)...", flush=True)
    gooaq_ds = load_dataset("sentence-transformers/gooaq", split="train", streaming=True)
    gooaq_pairs = []
    for i, row in enumerate(gooaq_ds):
        if i < 60000:
            continue
        if len(gooaq_pairs) >= 40000:
            break
        q = str(row.get("question") or "").strip()
        a = str(row.get("answer") or "").strip()
        if len(q) >= 10 and 30 <= len(a) <= 4000 and _norm_hash(q) not in seen_q:
            seen_q.add(_norm_hash(q))
            gooaq_pairs.append((q, a))
    gooaq_rows = _mine_bm25_hard_negatives(
        gooaq_pairs,
        source="gooaq_extra_40k",
        domain="web_qa",
        spdx="apache-2.0",
        docs_per_query=10,
        eval_q_hashes=eval_q,
        eval_d_hashes=eval_d,
        rng=rng,
    )
    all_rows.extend(gooaq_rows)
    stats["gooaq_extra_40k"] = {"queries": len(gooaq_rows), "license": "apache-2.0", "domain": "web_qa"}
    print(f"  -> GooAQ extra queries: {len(gooaq_rows)}", flush=True)

    # 2. 20,000 net-new PAQ Wikipedia QA pairs (MIT)
    print("[Track C 2/2] Streaming 20K net-new sentence-transformers/paq (MIT)...", flush=True)
    paq_ds = load_dataset("sentence-transformers/paq", split="train", streaming=True)
    paq_pairs = []
    for i, row in enumerate(paq_ds):
        if i < 60000:
            continue
        if len(paq_pairs) >= 20000:
            break
        q = str(row.get("query") or row.get("question") or "").strip()
        a = str(row.get("answer") or row.get("passage") or "").strip()
        if len(q) >= 10 and 30 <= len(a) <= 4000 and _norm_hash(q) not in seen_q:
            seen_q.add(_norm_hash(q))
            paq_pairs.append((q, a))
    paq_rows = _mine_bm25_hard_negatives(
        paq_pairs,
        source="paq_extra_20k",
        domain="web_qa",
        spdx="mit",
        docs_per_query=10,
        eval_q_hashes=eval_q,
        eval_d_hashes=eval_d,
        rng=rng,
    )
    all_rows.extend(paq_rows)
    stats["paq_extra_20k"] = {"queries": len(paq_rows), "license": "mit", "domain": "web_qa"}
    print(f"  -> PAQ extra queries: {len(paq_rows)}", flush=True)

    rng.shuffle(all_rows)
    out_dir = data_dir / "pool" / "track_c_unscored"
    out_dir.mkdir(parents=True, exist_ok=True)
    shard_size = 10000
    for idx, start in enumerate(range(0, len(all_rows), shard_size)):
        path = out_dir / f"track_c_{idx:04d}.parquet"
        pq.write_table(pa.Table.from_pylist(all_rows[start : start + shard_size]), path, compression="zstd")

    api = HfApi(token=get_hf_token())
    api.upload_folder(
        folder_path=str(out_dir),
        path_in_repo="track_c_unscored",
        repo_id="christianengel/rerank-mini-distill-shards",
        repo_type="dataset",
    )
    print("Uploaded track_c_unscored ->", json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()
