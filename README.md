# rerank-mini

Apache-2.0, 32M-parameter, **8,192-token** cross-encoder reranker distilled from `Qwen/Qwen3-Reranker-8B` onto `jhu-clsp/ettin-encoder-32m` (ModernBERT, CLS pooling). Built as a 1-line drop-in replacement for `cross-encoder/ms-marco-MiniLM-L6-v2`.

- **Model (Safetensors + ONNX + GGUF):** [christianengel/rerank-mini-32m](https://huggingface.co/christianengel/rerank-mini-32m)
- **Browser WebGPU Demo:** [christianengel/rerank-mini-32m-demo](https://huggingface.co/spaces/christianengel/rerank-mini-32m-demo)
- **Training Shards (5.0M Apache-2.0/MIT pairs):** [christianengel/rerank-mini-distill-shards](https://huggingface.co/datasets/christianengel/rerank-mini-distill-shards)

## Results

### Three-Pillar Locked Suite (`eval_locked.py`, 13 datasets, NDCG@10)

| Model | Params | Ctx | BEIR (7) | Code (3) | Long >1K (3) | **Overall** | RTX 3090 (p/s) | CPU ONNX 4t (p/s) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| BM25 | — | — | 0.4881 | 0.4603 | 0.7189 | 0.5558 | — | — |
| `mxbai-rerank-xsmall-v1` | 70.8M | 512 | 0.5634 | 0.4290 | 0.5221 | 0.5048 | 1,968 | — |
| `ms-marco-MiniLM-L6-v2` | 22.7M | 512 | 0.5914 | 0.5055 | 0.5392 | 0.5453 | 4,874 | 128.5 |
| `ettin-reranker-32m-v1` | 32.0M | 8K | 0.6012 | 0.6347 | 0.5334 | 0.5898 | 2,876 | 85.4 |
| `BAAI/bge-reranker-v2-m3` | 568M | 8K | 0.6189 | 0.5267 | 0.7094 | 0.6183 | 769 | — |
| `gte-reranker-modernbert-base` | 149M | 8K | **0.6257** | **0.6515** | **0.7275** | **0.6682** | 569 | — |
| **`rerank-mini-32m`** | **32.0M** | **8K** | **0.6084** | **0.6486** | **0.7212** | **0.6594** | **3,072** | **93.3** |

### Full Standard BEIR Test Splits (5 tasks, 2,321 queries, 232,100 pairs, NDCG@10)

| Model | Params | SciFact | NFCorpus | FiQA2018 | TREC-COVID | SCIDOCS | **Mean** | Time (3090) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| BM25 | — | 0.6646 | 0.3116 | 0.2322 | 0.5662 | 0.1521 | 0.3853 | — |
| `ms-marco-MiniLM-L6-v2` | 22.7M | 0.6784 | **0.3430** | 0.3375 | 0.6865 | 0.1648 | 0.4420 | 109.8s |
| `mxbai-rerank-xsmall-v1` | 70.8M | 0.6650 | 0.3205 | 0.3320 | **0.7793** | 0.1698 | 0.4533 | 331.5s |
| `ettin-reranker-32m-v1` | 32.0M | 0.7248 | 0.3271 | 0.3716 | 0.7171 | 0.1571 | 0.4595 | 224.0s |
| `gte-reranker-modernbert-base` | 149M | 0.7119 | 0.2906 | **0.4037** | 0.7711 | **0.1908** | **0.4736** | 667.0s |
| **`rerank-mini-32m`** | **32.0M** | **0.7412** | 0.3368 | 0.3467 | 0.7441 | 0.1747 | **0.4687** | **191.0s** |

Runtime parity (against FP32):
- **ONNX FP32 (`122.5 MB`):** `r = 1.0000`, `max_diff = 3e-6`, `0.0000` drop.
- **ONNX FP16 (`61.5 MB`):** `r = 1.0000`, `max_diff = 0.002`, `0.0000` drop (`NanoMSMARCO 0.6462`).
- **GGUF `Q8_0` (`34.2 MB`):** `r > 0.9999`, **`-0.0001` BEIR drop** (`0.6083` vs `0.6084` across all 7 BEIR tasks via `llama-server --reranking --pooling rank`).
- **ONNX INT8 (`86.9 MB`):** `r = 0.9963`, **`-0.0285` BEIR drop** (prefer GGUF `Q8_0` for compact 8-bit serving).

## Use

```python
from sentence_transformers import CrossEncoder

model = CrossEncoder("christianengel/rerank-mini-32m")
scores = model.predict([("query", "passage 1"), ("query", "passage 2")])
```

```bash
# Or serve the 34.2 MB GGUF Q8_0 directly with llama.cpp:
llama-server -hf christianengel/rerank-mini-32m:Q8_0 --reranking --pooling rank -c 8192 -ub 8192 -b 8192 --port 8080
```

## Reproduce

```bash
uv sync --extra cu128 --extra onnx

# 1. Build immutable eval suite + run baselines
uv run --extra cu128 --extra onnx python eval_locked.py --build-suite
uv run --extra cu128 --extra onnx python eval_locked.py --model cross-encoder/ettin-reranker-32m-v1 --out results/baseline_ettin_32m_v1.json

# 2. Run 2,321-query full standard BEIR comparison
uv run --extra cu128 --extra onnx python -m rerank_mini.eval_full_beir --out results/full_beir_5_comparison.json

# 3. Export ONNX (FP32/FP16/INT8) + GGUF (F16/Q8_0) and verify parity
uv run --extra cu128 --extra onnx python -m rerank_mini.export_runtimes --ckpt <ckpt> --try-gguf
uv run --extra cu128 --extra onnx python -m rerank_mini.verify_int8_ndcg --ckpt <ckpt> --out results/final_onnx_parity_and_latency.json
```

## License

Code, weights, and distilled dataset shards: [Apache-2.0](LICENSE) (see [NOTICE](NOTICE) for Apache-2.0 / MIT upstream attributions).
