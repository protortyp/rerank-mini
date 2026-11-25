# rerank-mini

Apache-2.0, 32M-parameter, **8,192-token** cross-encoder reranker distilled from `Qwen/Qwen3-Reranker-8B` onto `jhu-clsp/ettin-encoder-32m` (ModernBERT, CLS pooling). Built as a drop-in replacement for `cross-encoder/ms-marco-MiniLM-L6-v2`.

- **Model (Safetensors + ONNX + GGUF):** [christianengel/rerank-mini-32m](https://huggingface.co/christianengel/rerank-mini-32m)
- **Browser WebGPU Demo:** [christianengel/rerank-mini-32m-demo](https://huggingface.co/spaces/christianengel/rerank-mini-32m-demo)

## Results

Frozen 13-dataset evaluation (`eval_locked.py`, NDCG@10):

| Model | Params | Ctx | BEIR (7) | Code (3) | Long >1K (3) | **Overall** | RTX 3090 (p/s) | CPU ONNX 4t (p/s) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| BM25 | — | — | 0.4881 | 0.4603 | 0.7189 | 0.5558 | — | — |
| `ms-marco-MiniLM-L6-v2` | 22.7M | 512 | 0.5914 | 0.5055 | 0.5392 | 0.5453 | 4,874 | 128.5 |
| `ettin-reranker-32m-v1` | 32.0M | 8K | 0.6012 | 0.6347 | 0.5334 | 0.5898 | 2,876 | 85.4 |
| `BAAI/bge-reranker-v2-m3` | 568M | 8K | **0.6189** | 0.5267 | 0.7094 | 0.6183 | 769 | — |
| **`rerank-mini-32m`** | **32.0M** | **8K** | **0.6084** | **0.6486** | **0.7212** | **0.6594** | **3,072** | **93.3** |

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

## Reproduce

```bash
uv sync --extra cu128 --extra onnx

# 1. Build immutable eval suite + run baselines
uv run --extra cu128 --extra onnx python eval_locked.py --build-suite
uv run --extra cu128 --extra onnx python eval_locked.py --model cross-encoder/ettin-reranker-32m-v1 --out results/baseline_ettin_32m_v1.json

# 2. Build Apache-2.0/MIT training shards & score with Qwen3-Reranker-8B (local or Vast.ai / Colab)
uv run --extra cu128 --extra onnx python -m rerank_mini.build_pool --help
uv run --extra cu128 --extra onnx python infra/run_vast_teacher.py --help

# 3. Distill student & merge 5-way model soup
uv run --extra cu128 --extra onnx python train.py --help
uv run --extra cu128 --extra onnx python -m rerank_mini.merge_soup --help

# 4. Export ONNX (FP32/FP16/INT8) + GGUF (F16/Q8_0) and verify parity
uv run --extra cu128 --extra onnx python -m rerank_mini.export_runtimes --ckpt <ckpt> --try-gguf
uv run --extra cu128 --extra onnx python -m rerank_mini.verify_int8_ndcg --ckpt <ckpt> --out results/final_onnx_parity_and_latency.json
```

## Repository Layout

- `eval_locked.py` — immutable 3-pillar evaluation harness (`chmod 444`)
- `teacher_score.py` — batched FlashAttention/BF16 `Qwen/Qwen3-Reranker-8B` log-odds scorer
- `train.py` — listwise KL + hybrid MSE/Margin ModernBERT distillation trainer with 2-stage long-context curriculum
- `rerank_mini/` — dataset pool builders, model soup merger, ONNX/GGUF exporter, INT8 parity verifier, HF staging/publisher
- `infra/` — verified-only Vast.ai orchestrator with `try/finally` auto-destroy watchdog & `budget_ledger.json`
- `colab/` — optional resumable Colab notebooks (`01_teacher_scoring.ipynb`, `02_student_train.ipynb`)
- `card/README.md` — Hugging Face model card
- `space/` — client-side WebGPU Transformers.js comparison Space
- `results/` & `experiments.jsonl` — all locked benchmark outputs and ablation logs

## License

Code and model weights: [Apache-2.0](LICENSE) (see [NOTICE](NOTICE) for Apache-2.0 / MIT upstream attributions).
