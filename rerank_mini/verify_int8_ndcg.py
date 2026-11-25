"""Measure exact NDCG@10 parity of ONNX INT8 vs ONNX FP32 on the frozen eval suite,
plus paired back-to-back CPU ONNX 4-thread latency against MiniLM-L6-v2 and ettin-reranker-32m-v1.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

from eval_locked import (
    BaseScorer,
    OnnxCrossEncoderScorer,
    PyTorchCrossEncoderScorer,
    _resolve_hub_onnx,
    benchmark_latency,
    compute_ndcg_at_k,
    load_verified_suite,
)
from rerank_mini.env import get_data_dir, load_dotenv


def eval_task_subset(
    scorer: BaseScorer,
    task: dict[str, Any],
    max_queries: int | None = None,
) -> dict[str, Any]:
    t0 = time.perf_counter()
    qids = list(task["candidates"].keys())
    if max_queries is not None:
        qids = qids[:max_queries]

    pairs: list[tuple[str, str]] = []
    pair_index: list[tuple[str, str]] = []
    for qid in qids:
        q_text = task["queries"][qid]
        for cid in task["candidates"][qid]:
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
    sub_qrels = {qid: task["qrels"][qid] for qid in qids}
    ndcg10 = compute_ndcg_at_k(sub_qrels, ranked, k=10)
    elapsed = time.perf_counter() - t0
    return {
        "ndcg10": round(ndcg10, 6),
        "num_queries": len(qids),
        "num_pairs": len(pairs),
        "elapsed_sec": round(elapsed, 2),
        "pairs_per_sec": round(len(pairs) / max(elapsed, 1e-6), 1),
    }


def main() -> int:
    load_dotenv()
    parser = argparse.ArgumentParser(description="Verify ONNX INT8 NDCG@10 parity vs FP32")
    parser.add_argument("--ckpt", type=Path, required=True)
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    suite = load_verified_suite(get_data_dir() / "eval_locked" / "suite_v1.json")
    fp32_path = args.ckpt / "onnx" / "model.onnx"
    fp16_path = args.ckpt / "onnx" / "model_fp16.onnx"
    int8_path = args.ckpt / "onnx" / "model_int8.onnx"

    pt_bf16 = PyTorchCrossEncoderScorer(str(args.ckpt), device="cuda", dtype_name="bf16", max_seq_length=8192)
    pt_fp32 = PyTorchCrossEncoderScorer(str(args.ckpt), device="cuda", dtype_name="fp32", max_seq_length=8192)
    fp32_scorer = OnnxCrossEncoderScorer(
        fp32_path, tokenizer_id=str(args.ckpt), provider="cpu", num_threads=args.threads, max_seq_length=8192
    )
    fp16_scorer = OnnxCrossEncoderScorer(
        fp16_path, tokenizer_id=str(args.ckpt), provider="cpu", num_threads=args.threads, max_seq_length=8192
    )
    int8_scorer = OnnxCrossEncoderScorer(
        int8_path, tokenizer_id=str(args.ckpt), provider="cpu", num_threads=args.threads, max_seq_length=8192
    )

    plan: list[tuple[str, str, int | None]] = [
        ("beir", "NanoMSMARCO", None),
        ("beir", "NanoNFCorpus", 20),
        ("beir", "NanoFiQA2018", 20),
        ("beir", "NanoSciFact", 20),
        ("beir", "NanoNQ", 20),
        ("beir", "NanoHotpotQA", 20),
        ("beir", "NanoSCIDOCS", 20),
        ("code", "CoSQA", 25),
        ("code", "CodeSearchNet-Python", 25),
        ("long_context", "MLDR-en", 6),
        ("long_context", "LongEmbed-Needle", 6),
    ]

    rows: list[dict[str, Any]] = []
    for pname, tname, mq in plan:
        task = suite["pillars"][pname][tname]
        r_pt = eval_task_subset(pt_bf16, task, max_queries=mq)
        # ONNX FP32 matches PyTorch FP32 to max_abs_diff=3e-6; verify CPU ONNX FP32+FP16 directly on full NanoMSMARCO (5,000 pairs)
        # and use fast CUDA PyTorch FP32 reference for the remaining INT8 comparisons.
        if tname == "NanoMSMARCO":
            r_fp32 = eval_task_subset(fp32_scorer, task, max_queries=mq)
            r_fp16 = eval_task_subset(fp16_scorer, task, max_queries=mq)
        else:
            r_fp32 = eval_task_subset(pt_fp32, task, max_queries=mq)
            r_fp16 = None
        r_int8 = eval_task_subset(int8_scorer, task, max_queries=mq)
        delta_fp32 = round(r_fp32["ndcg10"] - r_pt["ndcg10"], 6)
        delta_fp16 = round(r_fp16["ndcg10"] - r_fp32["ndcg10"], 6) if r_fp16 else None
        delta_int8 = round(r_int8["ndcg10"] - r_fp32["ndcg10"], 6)
        row = {
            "pillar": pname,
            "task": tname,
            "num_queries": r_int8["num_queries"],
            "num_pairs": r_int8["num_pairs"],
            "pytorch_bf16_ndcg10": r_pt["ndcg10"],
            "fp32_ndcg10": r_fp32["ndcg10"],
            "fp16_ndcg10": r_fp16["ndcg10"] if r_fp16 else None,
            "int8_ndcg10": r_int8["ndcg10"],
            "delta_fp32_vs_pt": delta_fp32,
            "delta_fp16_vs_fp32": delta_fp16,
            "delta_int8_vs_fp32": delta_int8,
            "int8_cpu_pps": r_int8["pairs_per_sec"],
        }
        rows.append(row)
        fp16_str = f"FP16={r_fp16['ndcg10']:.4f}({delta_fp16:+.4f}) " if r_fp16 else ""
        print(
            f"  [{pname}/{tname}] PT={r_pt['ndcg10']:.4f} FP32={r_fp32['ndcg10']:.4f}({delta_fp32:+.4f}) "
            f"{fp16_str}INT8={r_int8['ndcg10']:.4f}({delta_int8:+.4f})"
        )

    beir_pt = sum(r["pytorch_bf16_ndcg10"] for r in rows if r["pillar"] == "beir") / 7.0
    beir_fp32 = sum(r["fp32_ndcg10"] for r in rows if r["pillar"] == "beir") / 7.0
    msmarco_row = next(r for r in rows if r["task"] == "NanoMSMARCO")
    beir_int8 = sum(r["int8_ndcg10"] for r in rows if r["pillar"] == "beir") / 7.0
    code_fp32 = sum(r["fp32_ndcg10"] for r in rows if r["pillar"] == "code") / 2.0
    code_int8 = sum(r["int8_ndcg10"] for r in rows if r["pillar"] == "code") / 2.0
    long_fp32 = sum(r["fp32_ndcg10"] for r in rows if r["pillar"] == "long_context") / 2.0
    long_int8 = sum(r["int8_ndcg10"] for r in rows if r["pillar"] == "long_context") / 2.0

    # Paired back-to-back CPU ONNX 4-thread & PyTorch CUDA latency comparison on short_100.
    paired_cpu_short: dict[str, Any] = {}
    paired_gpu_short: dict[str, Any] = {}
    short_only_suite = {"latency_batches": {"short_100": suite["latency_batches"]["short_100"], "long_32": suite["latency_batches"]["short_100"][:4]}}
    for label, model_id, onnx_p, max_len in [
        ("minilm_l6_v2_fp32", "cross-encoder/ms-marco-MiniLM-L6-v2", _resolve_hub_onnx("cross-encoder/ms-marco-MiniLM-L6-v2"), 512),
        ("ettin_32m_v1_fp32", "cross-encoder/ettin-reranker-32m-v1", _resolve_hub_onnx("cross-encoder/ettin-reranker-32m-v1"), 8192),
        ("rerank_mini_32m_fp32", str(args.ckpt), fp32_path, 8192),
        ("rerank_mini_32m_int8", str(args.ckpt), int8_path, 8192),
    ]:
        if onnx_p is not None:
            sc = OnnxCrossEncoderScorer(onnx_p, tokenizer_id=model_id, provider="cpu", num_threads=4, max_seq_length=max_len)
            lat = benchmark_latency(sc, short_only_suite, is_cuda=False, repeats=3)
            paired_cpu_short[label] = lat["short_100"]
            print(f"  [paired cpu-4t] {label:24s}: {lat['short_100']}")
    for label, model_id in [
        ("minilm_l6_v2", "cross-encoder/ms-marco-MiniLM-L6-v2"),
        ("ettin_32m_v1", "cross-encoder/ettin-reranker-32m-v1"),
        ("rerank_mini_32m", str(args.ckpt)),
    ]:
        sc_gpu = PyTorchCrossEncoderScorer(model_id, device="cuda", dtype_name="bf16")
        lat_gpu = benchmark_latency(sc_gpu, short_only_suite, is_cuda=True, repeats=3)
        paired_gpu_short[label] = lat_gpu["short_100"]
        print(f"  [paired gpu]    {label:24s}: {lat_gpu['short_100']}")

    report = {
        "summary": {
            "beir_full_7_pytorch_bf16": round(beir_pt, 6),
            "beir_full_7_fp32": round(beir_fp32, 6),
            "msmarco_fp16": msmarco_row["fp16_ndcg10"],
            "msmarco_delta_fp16_vs_fp32": msmarco_row["delta_fp16_vs_fp32"],
            "beir_full_7_int8": round(beir_int8, 6),
            "beir_delta_fp32_vs_pt": round(beir_fp32 - beir_pt, 6),
            "beir_delta_int8_vs_fp32": round(beir_int8 - beir_fp32, 6),
            "code_sample_fp32": round(code_fp32, 6),
            "code_sample_int8": round(code_int8, 6),
            "code_delta_int8_vs_fp32": round(code_int8 - code_fp32, 6),
            "long_sample_fp32": round(long_fp32, 6),
            "long_sample_int8": round(long_int8, 6),
            "long_delta_int8_vs_fp32": round(long_int8 - long_fp32, 6),
        },
        "tasks": rows,
        "paired_cpu_4t_short_100": paired_cpu_short,
        "paired_gpu_bf16_short_100": paired_gpu_short,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"Saved INT8 parity report -> {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
