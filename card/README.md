---
license: apache-2.0
language:
  - en
library_name: sentence-transformers
pipeline_tag: text-ranking
base_model: jhu-clsp/ettin-encoder-32m
tags:
  - reranker
  - cross-encoder
  - onnx
  - gguf
  - transformers.js
  - modernbert
  - rag
---

# rerank-mini-32m

Apache-2.0, 32M-parameter, **8,192-token** cross-encoder reranker built as a 1-line drop-in replacement for `cross-encoder/ms-marco-MiniLM-L6-v2` (22.7M, 512 tokens). Distilled from `Qwen/Qwen3-Reranker-8B` (Apache-2.0) onto `jhu-clsp/ettin-encoder-32m` (MIT, ModernBERT architecture, CLS pooling) over 5.0M strictly Apache-2.0/MIT `(query, passage)` pairs.

Ships in **PyTorch / `sentence-transformers`**, **ONNX (`FP32`, `FP16`, `INT8`)**, **Transformers.js v4 (WebGPU/WASM)**, and **GGUF (`F16`, `Q8_0` at 34.2 MB)**.

- **Code & Eval Harness:** [github.com/protortyp/rerank-mini](https://github.com/protortyp/rerank-mini)
- **Live Browser WebGPU Demo:** [huggingface.co/spaces/christianengel/rerank-mini-32m-demo](https://huggingface.co/spaces/christianengel/rerank-mini-32m-demo)
- **Training Shards (5.0M Apache-2.0/MIT pairs):** [huggingface.co/datasets/christianengel/rerank-mini-distill-shards](https://huggingface.co/datasets/christianengel/rerank-mini-distill-shards)

## Head-to-Head Benchmarks

### 1. Three-Pillar Locked Suite (`eval_locked.py`, 13 datasets, NDCG@10)

Fixed first-stage BM25 candidates across 7 NanoBEIR datasets (top-100), 3 Code/Tech datasets (`CoSQA`, `CodeSearchNet-Python`, `StackOverflow-QA`, top-100), and 3 Long-Context datasets >1,024 tokens (`MLDR-en`, `LongEmbed-2WikiMQA`, `LongEmbed-Needle`, top-30).

| Model | License | Params | Ctx | BEIR (7) | Code (3) | Long >1K (3) | **Overall** | RTX 3090 (p/s) | CPU ONNX 4t (p/s) |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| BM25 (first stage) | — | — | — | 0.4881 | 0.4603 | 0.7189 | 0.5558 | — | — |
| `mixedbread-ai/mxbai-rerank-xsmall-v1` | Apache-2.0 | 70.8M | 512 | 0.5634 | 0.4290 | 0.5221 | 0.5048 | 1,968 | — |
| `cross-encoder/ms-marco-MiniLM-L6-v2` | Apache-2.0 | 22.7M | 512 | 0.5914 | 0.5055 | 0.5392 | 0.5453 | 4,874 | 128.5 |
| `cross-encoder/ettin-reranker-32m-v1` | MIT | 32.0M | 8,192 | 0.6012 | 0.6347 | 0.5334 | 0.5898 | 2,876 | 85.4 |
| `BAAI/bge-reranker-v2-m3` | MIT | 568M | 8,192 | 0.6189 | 0.5267 | 0.7094 | 0.6183 | 769 | — |
| `Alibaba-NLP/gte-reranker-modernbert-base` | Apache-2.0 | 149M | 8,192 | **0.6257** | **0.6515** | **0.7275** | **0.6682** | 569 | — |
| **`christianengel/rerank-mini-32m`** | **Apache-2.0** | **32.0M** | **8,192** | **0.6084** | **0.6486** | **0.7212** | **0.6594** | **3,072** | **93.3** |

- Captures **98.7% of 4.65x larger 149M `gte-reranker-modernbert-base`** at **5.4x higher GPU throughput** (`3,072` vs `569` pairs/s on `short_100`, `129` vs `26` pairs/s on `long_32`).
- Sweeps all three pillars against `MiniLM-L6-v2` (`+11.41`), `mxbai-rerank-xsmall-v1` (`+15.46`), and `ettin-reranker-32m-v1` (`+6.96`, `+6.8%` faster GPU / `+9.3%` faster CPU ONNX), and beats 18x larger 568M `bge-reranker-v2-m3` by `+4.11` overall at 4.0x GPU speed.

### 2. Full Standard BEIR Test Splits (5 tasks, 2,321 test queries, 232,100 pairs, NDCG@10)

Reranking top-100 Okapi BM25 candidates on full standard BEIR test splits (`rerank_mini/eval_full_beir.py`):

| Model | Params | Ctx | SciFact (300q) | NFCorpus (323q) | FiQA2018 (648q) | TREC-COVID (50q) | SCIDOCS (1,000q) | **Full BEIR Mean** | Wall Time (3090) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| BM25 | — | — | 0.6646 | 0.3116 | 0.2322 | 0.5662 | 0.1521 | 0.3853 | — |
| `ms-marco-MiniLM-L6-v2` | 22.7M | 512 | 0.6784 | **0.3430** | 0.3375 | 0.6865 | 0.1648 | 0.4420 | 109.8s |
| `mxbai-rerank-xsmall-v1` | 70.8M | 512 | 0.6650 | 0.3205 | 0.3320 | **0.7793** | 0.1698 | 0.4533 | 331.5s |
| `ettin-reranker-32m-v1` | 32.0M | 8K | 0.7248 | 0.3271 | 0.3716 | 0.7171 | 0.1571 | 0.4595 | 224.0s |
| `gte-reranker-modernbert-base` | 149M | 8K | 0.7119 | 0.2906 | **0.4037** | 0.7711 | **0.1908** | **0.4736** | 667.0s |
| **`rerank-mini-32m` (ours)** | **32.0M** | **8K** | **0.7412** | 0.3368 | 0.3467 | 0.7441 | 0.1747 | **0.4687** | **191.0s** |

On full BEIR, `rerank-mini-32m` beats `MiniLM-L6-v2` (`+2.67`), 70.8M `mxbai-rerank-xsmall-v1` (`+1.54`, 1.74x faster), and `ettin-reranker-32m-v1` (`+0.92`, 17% faster), reaching **99.0% of 149M `gte-reranker-modernbert-base` at 3.5x speed** (and beating it on `SciFact` by `+2.93` and `NFCorpus` by `+4.62`).

### 3. Per-Dataset Breakdown on Locked Suite (NDCG@10)

| Pillar | Dataset | BM25 | MiniLM | ettin-32m | bge-v2-m3 | gte-149m | **rerank-mini-32m** |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| BEIR | NanoMSMARCO | 0.4732 | 0.6619 | **0.6936** | 0.6866 | 0.6897 | 0.6392 |
| BEIR | NanoNFCorpus | 0.3280 | **0.3505** | 0.3179 | 0.3239 | 0.3447 | 0.3441 |
| BEIR | NanoFiQA2018 | 0.3317 | 0.4669 | 0.4896 | 0.5250 | **0.5300** | 0.4774 |
| BEIR | NanoSciFact | 0.7197 | 0.7083 | 0.7232 | 0.7530 | 0.7834 | **0.7892** |
| BEIR | NanoNQ | 0.4511 | 0.7451 | **0.7539** | 0.7496 | 0.7324 | 0.7452 |
| BEIR | NanoHotpotQA | 0.7804 | 0.8679 | 0.9087 | **0.9268** | 0.9105 | 0.8891 |
| BEIR | NanoSCIDOCS | 0.3327 | 0.3391 | 0.3213 | 0.3672 | **0.3889** | 0.3748 |
| Code | CoSQA | 0.1494 | 0.1650 | 0.2472 | 0.2029 | **0.2658** | 0.2460 |
| Code | CodeSearchNet-Python | 0.5766 | 0.7794 | 0.8575 | 0.7302 | **0.8905** | 0.8746 |
| Code | StackOverflow-QA | 0.6550 | 0.5720 | 0.7993 | 0.6469 | 0.7981 | **0.8252** |
| Long | MLDR-en (med 5.0K words) | 0.7618 | 0.5715 | 0.6209 | 0.7565 | **0.8157** | 0.7876 |
| Long | LongEmbed-2WikiMQA (med 5.3K words) | **0.9755** | 0.7635 | 0.7010 | 0.9554 | 0.9482 | 0.9572 |
| Long | LongEmbed-Needle (med 3.1K words) | **0.4194** | 0.2825 | 0.2784 | 0.4164 | 0.4187 | 0.4189 |

*Note on Long-Context: On `MLDR-en`, `rerank-mini-32m` (`0.7876`) improves clearly over both BM25 (`0.7618`) and 568M `bge-v2-m3` (`0.7565`). On exact-keyword `2WikiMQA` and `Needle`, BM25 is already near ceiling (`0.9755` / `0.4194`); `rerank-mini-32m` preserves full-document fidelity (`0.9572` / `0.4189`) without collapsing past 512 tokens like `MiniLM-L6-v2` (`0.7635` / `0.2825`) or `ettin-reranker-32m-v1` (`0.7010` / `0.2784`).*

## Multi-Runtime Files & Quantization Parity

| Format | File | Size | Logit Pearson `r` vs FP32 | Ranking Parity vs FP32 | Recommended For |
| --- | --- | ---: | ---: | --- | --- |
| PyTorch Safetensors | `model.safetensors` | 122.2 MB | 1.0000 | reference | `sentence-transformers` / GPU servers |
| ONNX FP32 | `onnx/model.onnx` | 122.5 MB | 1.0000 (`max_diff=3e-6`) | `0.0000` (`NanoMSMARCO 0.6462`) | CPU servers / WASM fallback (exact) |
| ONNX FP16 | `onnx/model_fp16.onnx` | 61.5 MB | 1.0000 (`max_diff=0.002`) | `0.0000` (`NanoMSMARCO 0.6462`) | ONNX CUDA & Transformers.js WebGPU |
| **GGUF `Q8_0`** | `gguf/rerank-mini-32m-q8_0.gguf` | **34.2 MB** | 0.9999+ | **`-0.0001`** (all 7 BEIR / 35K pairs: `0.6083` vs `0.6084`) | Edge & local CPU/GPU via `llama.cpp` |
| GGUF `F16` | `gguf/rerank-mini-32m-f16.gguf` | 62.8 MB | 1.0000 | `~0.0000` | `llama.cpp` full-precision reference |
| ONNX INT8 (dynamic) | `onnx/model_int8.onnx` | 86.9 MB | 0.9963 | **`-0.0285`** BEIR (`-0.0500` MSMARCO, `-0.0582` Needle) | Short-text latency-only CPU (see warning) |

> **INT8 Warning:** Dynamic ONNX INT8 quantization (`onnx/model_int8.onnx`, per-channel QUInt8 with FP32 head) costs **~2.8 NDCG@10 on BEIR** (up to -5.0 on `NanoMSMARCO` and -5.8 on `LongEmbed-Needle`) due to activation quantization sensitivity in ModernBERT GeGLU layers. For a compact 8-bit model with **zero ranking loss**, use **`gguf/rerank-mini-32m-q8_0.gguf` (34.2 MB, `-0.0001` BEIR drop)** instead, or `onnx/model_fp16.onnx` (61.5 MB) on GPU/WebGPU and `onnx/model.onnx` on CPU.

## Quick Start

### 1. `sentence-transformers` (1-line drop-in swap for MiniLM)

```python
from sentence_transformers import CrossEncoder

# Drop-in replacement for "cross-encoder/ms-marco-MiniLM-L6-v2" (now supports 8,192 tokens)
model = CrossEncoder("christianengel/rerank-mini-32m")

query = "python async context manager that retries HTTP 429"
passages = [
    "class AsyncRetryClient: async def __aenter__(self): ... if r.status == 429: await asyncio.sleep(...)",
    "HTTP 429 Too Many Requests is defined in RFC 6585.",
    "def parse_csv(path): return list(csv.DictReader(open(path)))",
]

scores = model.predict([(query, p) for p in passages])
for score, doc in sorted(zip(scores, passages), reverse=True):
    print(f"{score:.4f}  {doc[:80]}")
```

### 2. ONNX Runtime (CPU or CUDA)

```python
import numpy as np
import onnxruntime as ort
from huggingface_hub import hf_hub_download
from transformers import AutoConfig, AutoTokenizer

repo_id = "christianengel/rerank-mini-32m"
_ = AutoConfig.from_pretrained(repo_id)
tokenizer = AutoTokenizer.from_pretrained(repo_id)

# Use "onnx/model_fp16.onnx" on CUDAExecutionProvider, or "onnx/model.onnx" on CPUExecutionProvider.
onnx_path = hf_hub_download(repo_id, "onnx/model.onnx")
session = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])

query = "Why does FlashAttention reduce GPU memory usage?"
docs = [
    "FlashAttention tiles Q, K, V in SRAM with online softmax, cutting memory from O(N^2) to O(N).",
    "Rotary Position Embeddings rotate query and key pairs in 2D subspaces.",
]

enc = tokenizer([query] * len(docs), docs, padding=True, truncation=True, max_length=8192, return_tensors="np")
logits = session.run(None, {
    "input_ids": enc["input_ids"].astype(np.int64),
    "attention_mask": enc["attention_mask"].astype(np.int64),
})[0].reshape(-1)
probs = 1.0 / (1.0 + np.exp(-logits))
print("logits:", logits, "probs:", probs)
```

### 3. Transformers.js v4 (Browser WebGPU / WASM)

```js
import { AutoTokenizer, AutoModelForSequenceClassification } from "@huggingface/transformers";

const modelId = "christianengel/rerank-mini-32m";
const hasWebGPU = Boolean(navigator.gpu && (await navigator.gpu.requestAdapter()));
const tokenizer = await AutoTokenizer.from_pretrained(modelId);
const model = await AutoModelForSequenceClassification.from_pretrained(modelId, {
  device: hasWebGPU ? "webgpu" : "wasm",
  dtype: hasWebGPU ? "fp16" : "fp32",
});

const query = "How do I read a Parquet file in Python?";
const docs = [
  "Use pyarrow.parquet.read_table(path) or pandas.read_parquet(path).",
  "Bananas are tropical fruits rich in potassium.",
];

const inputs = tokenizer(docs.map(() => query), {
  text_pair: docs,
  padding: true,
  truncation: true,
  max_length: 8192,
});
const { logits } = await model(inputs);
console.log(Array.from(logits.data));
```

### 4. `llama.cpp` GGUF (`Q8_0`, 34.2 MB)

```bash
llama-server \
  -hf christianengel/rerank-mini-32m:Q8_0 \
  --reranking --pooling rank -c 8192 -ub 8192 -b 8192 --port 8080

curl http://127.0.0.1:8080/v1/rerank \
  -H "Content-Type: application/json" \
  -d '{"model":"rerank","query":"What is FlashAttention?","documents":["FlashAttention tiles QKV in SRAM...","Bananas are yellow."]}'
```

*(You can also pin the path explicitly with `--hf-file gguf/rerank-mini-32m-q8_0.gguf`. Note that `llama-server` rejects requests exceeding 8,192 tokens with HTTP 400 rather than silently truncating; cap documents to ~7,500 tokens client-side when serving via `llama.cpp`.)*

## Training & Data Provenance

- **Backbone:** `jhu-clsp/ettin-encoder-32m` (MIT, ModernBERT 10-layer encoder, 32,031,361 params, CLS pooling, 8,192 max tokens).
- **Teacher:** `Qwen/Qwen3-Reranker-8B` (Apache-2.0), continuous log-odds (`logit("yes") - logit("no")`).
- **Distillation Objective:** Hybrid listwise KL divergence (`KL(softmax(t/τ) || softmax(s/τ))`) + light cross-query MSE & margin anchoring across 8–16 hard negatives per query, trained with a 2-stage curriculum (Stage 1: 512 tokens -> Stage 2: 4,096 tokens with long-doc upsampling) and combined via a 5-way same-init model soup across diverse temperature/curriculum trajectories (`τ ∈ {0.5, 1.0, 1.2, 2.0}`).
- **Data (100% Apache-2.0 / MIT, 435,816 queries / 4,989,720 pairs, SHA-256 deduplicated against all eval sets):** published at [`christianengel/rerank-mini-distill-shards`](https://huggingface.co/datasets/christianengel/rerank-mini-distill-shards). Zero MS MARCO non-commercial or CC-BY-SA data.

## Limitations

- **English-focused:** Training and evaluation data are English + programming languages. Use a multilingual reranker (e.g. `bge-reranker-v2-m3`) for non-English corpora.
- **MS MARCO head-to-head:** MS MARCO training pairs were excluded to keep strict Apache-2.0/MIT provenance, so `NanoMSMARCO` (`0.6392` BF16 / `0.6462` FP32) trails MS-MARCO-trained models like `ettin-reranker-32m-v1` (`0.6936`), even while beating `ettin-32m-v1` on both 7-task NanoBEIR (`0.6084` vs `0.6012`) and 2,321-query Full BEIR (`0.4687` vs `0.4595`).
- **Needle-in-a-haystack ceiling:** On synthetic 3K-word `LongEmbed-Needle` (`0.4189`), the model matches 568M `bge-reranker-v2-m3` (`0.4164`) and 149M `gte-reranker-modernbert-base` (`0.4187`), but small encoders remain bounded when a single unindexed 10-word fact is buried in pure distractor text.
- **ONNX Dynamic INT8:** Loses ~2.8 NDCG@10 on BEIR; prefer GGUF `Q8_0` (34.2 MB) or ONNX `FP16`/`FP32`.

## Citation

```bibtex
@misc{engel2026rerankmini32m,
  title  = {rerank-mini-32m: An Apache-2.0 32M-Parameter 8K-Context Cross-Encoder Reranker Distilled from Qwen3-Reranker-8B},
  author = {Christian Engel},
  year   = {2026},
  url    = {https://huggingface.co/christianengel/rerank-mini-32m}
}
```
