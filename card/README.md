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

Ships ready to run in **PyTorch / `sentence-transformers`**, **ONNX (`FP32`, `FP16`, `INT8`)**, **Transformers.js v4 (WebGPU/WASM)**, and **GGUF (`F16`, `Q8_0` at 34.2 MB)**.

- **Code, eval harness & training scripts:** [github.com/protortyp/rerank-mini](https://github.com/protortyp/rerank-mini)
- **Live Browser WebGPU Demo:** [huggingface.co/spaces/christianengel/rerank-mini-32m-demo](https://huggingface.co/spaces/christianengel/rerank-mini-32m-demo)

## Head-to-Head Benchmarks

Evaluated on our frozen 13-dataset benchmark (`eval_locked.py`, fixed first-stage BM25 candidates): 7 BEIR datasets (top-100), 3 Code/Tech datasets (`CoSQA`, `CodeSearchNet-Python`, `StackOverflow-QA`, top-100), and 3 Long-Context datasets >1,024 tokens (`MLDR-en`, `LongEmbed-2WikiMQA`, `LongEmbed-Needle`, top-30). Metric: **NDCG@10**.

| Model | License | Params | Context | BEIR (7) | Code (3) | Long >1K (3) | **Overall** | RTX 3090 (pairs/s) | CPU ONNX 4t (pairs/s) |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| BM25 (first stage) | — | — | — | 0.4881 | 0.4603 | 0.7189 | 0.5558 | — | — |
| `cross-encoder/ms-marco-MiniLM-L6-v2` | Apache-2.0 | 22.7M | 512 | 0.5914 | 0.5055 | 0.5392 | 0.5453 | 4,874 | 128.5 |
| `cross-encoder/ettin-reranker-32m-v1` | MIT | 32.0M | 8,192 | 0.6012 | 0.6347 | 0.5334 | 0.5898 | 2,876 | 85.4 |
| `BAAI/bge-reranker-v2-m3` (ceiling) | MIT | 568M | 8,192 | **0.6189** | 0.5267 | 0.7094 | 0.6183 | 769 | — |
| **`christianengel/rerank-mini-32m`** | **Apache-2.0** | **32.0M** | **8,192** | **0.6084** | **0.6486** | **0.7212** | **0.6594** | **3,072** | **93.3** |

*Paired GPU throughput measured back-to-back in PyTorch BF16 on an RTX 3090 (`short_100` batch); quiet-host CPU ONNX FP32 measured at 4 threads on Ryzen 9 5950X (`rerank-mini-32m` runs +6.8% faster on GPU and +9.3% faster on CPU than `ettin-reranker-32m-v1`, and 4.0x faster on GPU than 568M `bge-reranker-v2-m3`).*

### Per-Dataset Breakdown (NDCG@10)

| Pillar | Dataset | BM25 | MiniLM-L6-v2 | ettin-32m-v1 | bge-v2-m3 (568M) | **rerank-mini-32m** |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| BEIR | NanoMSMARCO | 0.4732 | 0.6619 | **0.6936** | 0.6866 | 0.6392 |
| BEIR | NanoNFCorpus | 0.3280 | **0.3505** | 0.3179 | 0.3239 | 0.3441 |
| BEIR | NanoFiQA2018 | 0.3317 | 0.4669 | 0.4896 | **0.5250** | 0.4774 |
| BEIR | NanoSciFact | 0.7197 | 0.7083 | 0.7232 | 0.7530 | **0.7892** |
| BEIR | NanoNQ | 0.4511 | 0.7451 | **0.7539** | 0.7496 | 0.7452 |
| BEIR | NanoHotpotQA | 0.7804 | 0.8679 | 0.9087 | **0.9268** | 0.8891 |
| BEIR | NanoSCIDOCS | 0.3327 | 0.3391 | 0.3213 | 0.3672 | **0.3748** |
| Code | CoSQA | 0.1494 | 0.1650 | **0.2472** | 0.2029 | 0.2460 |
| Code | CodeSearchNet-Python | 0.5766 | 0.7794 | 0.8575 | 0.7302 | **0.8746** |
| Code | StackOverflow-QA | 0.6550 | 0.5720 | 0.7993 | 0.6469 | **0.8252** |
| Long | MLDR-en (med 5.0K words) | 0.7618 | 0.5715 | 0.6209 | 0.7565 | **0.7876** |
| Long | LongEmbed-2WikiMQA (med 5.3K words) | **0.9755** | 0.7635 | 0.7010 | 0.9554 | 0.9572 |
| Long | LongEmbed-Needle (med 3.1K words) | **0.4194** | 0.2825 | 0.2784 | 0.4164 | 0.4189 |

## Multi-Runtime Files & Quantization Parity

| Format | File | Size | Logit Pearson `r` vs FP32 | NDCG@10 Delta vs FP32 | Recommended For |
| --- | --- | ---: | ---: | ---: | --- |
| PyTorch Safetensors | `model.safetensors` | 122.2 MB | 1.0000 | reference | `sentence-transformers` / GPU servers |
| ONNX FP32 | `onnx/model.onnx` | 122.5 MB | 1.0000 (`max_diff=3e-6`) | `0.0000` (`NanoMSMARCO 0.6462`) | CPU servers (exact ranking) |
| ONNX FP16 | `onnx/model_fp16.onnx` | 61.5 MB | 1.0000 (`max_diff=0.002`) | `0.0000` (`NanoMSMARCO 0.6462`) | ONNX CUDA & Transformers.js WebGPU |
| **GGUF `Q8_0`** | `gguf/rerank-mini-32m-q8_0.gguf` | **34.2 MB** | 0.9999+ | **`-0.0001`** (full 7-task BEIR: `0.6083` vs `0.6084`) | Edge, local CPU/GPU via `llama.cpp` |
| GGUF `F16` | `gguf/rerank-mini-32m-f16.gguf` | 62.8 MB | 1.0000 | `~0.0000` | `llama.cpp` full-precision reference |
| ONNX INT8 (dynamic) | `onnx/model_int8.onnx` | 86.9 MB | 0.9963 | **`-0.0285`** BEIR (`-0.0500` MSMARCO, `-0.0582` Needle) | Latency-critical short-text CPU only (see note below) |

> **INT8 Warning:** Dynamic ONNX INT8 quantization (`onnx/model_int8.onnx`, per-channel QUInt8 with FP32 head) speeds up short-sequence CPU inference by ~1.3–1.9x, but costs **~2.8 NDCG@10 points on BEIR** (up to -5.0 on `NanoMSMARCO` and -5.8 on `LongEmbed-Needle`) due to activation quantization sensitivity in ModernBERT GeGLU layers. If you want a compact 8-bit model with **zero ranking loss**, use **`gguf/rerank-mini-32m-q8_0.gguf` (34.2 MB, `-0.0001` BEIR drop)** instead, or use `onnx/model_fp16.onnx` (61.5 MB) on GPU/WebGPU and `onnx/model.onnx` on CPU.

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

### 3. Transformers.js v4 (Browser WebGPU / Node.js)

```js
import { AutoTokenizer, AutoModelForSequenceClassification } from "@huggingface/transformers";

const modelId = "christianengel/rerank-mini-32m";
const tokenizer = await AutoTokenizer.from_pretrained(modelId);
const model = await AutoModelForSequenceClassification.from_pretrained(modelId, {
  device: "webgpu", // or "wasm"
  dtype: "fp16",    // loads onnx/model_fp16.onnx (61.5 MB)
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

*Note: `llama-server` rejects requests exceeding 8,192 tokens with HTTP 400 rather than silently truncating; cap input documents to ~7,500 tokens client-side when serving via `llama.cpp`.*

## Training & Data Provenance

- **Backbone:** `jhu-clsp/ettin-encoder-32m` (MIT, ModernBERT 10-layer encoder, 32,031,361 params, CLS pooling, 8,192 max tokens).
- **Teacher:** `Qwen/Qwen3-Reranker-8B` (Apache-2.0), continuous log-odds (`logit("yes") - logit("no")`).
- **Distillation Objective:** Hybrid listwise KL divergence (`KL(softmax(t/τ) || softmax(s/τ))`) + light cross-query MSE & margin anchoring across 8–16 hard negatives per query, trained with a 2-stage curriculum (Stage 1: 512 tokens -> Stage 2: 4,096 tokens with long-doc upsampling) and combined via a 5-way same-init model soup across diverse temperature/curriculum trajectories (`τ ∈ {0.5, 1.0, 1.2, 2.0}`).
- **Data (100% Apache-2.0 / MIT, 5.0M pairs, SHA-256 deduplicated against all 13 eval sets):**
  - Pre-scored `Qwen3-Reranker-8B` Apache-2.0/MIT English splits from `KaLM-Embedding/KaLM-reranker-training-data` (`gooaq`, `codefeedback`, `pubmedqa`, `openorca`, `esci`, `rag-dataset-12000`, `paq_pairs`, `triviaqa`).
  - 1.2M freshly scored `Qwen3-Reranker-8B` pairs covering long documents (`Shitao/MLDR`, `deepmind/narrativeqa`), code & tool APIs (`Nan-Do/code-search-net-python`, `ise-uiuc/Magicoder-OSS-Instruct-75K`, `glaiveai/glaive-code-assistant-v2`, `glaiveai/glaive-function-calling-v2`), and web QA (`sentence-transformers/gooaq`, `sentence-transformers/paq`).
  - Zero MS MARCO non-commercial or CC-BY-SA training data.

## Limitations

- **English-focused:** Training and evaluation data are English + programming languages. Multilingual retrieval is untested; use a multilingual reranker (such as `bge-reranker-v2-m3`) for non-English corpora.
- **MS MARCO head-to-head:** Because MS MARCO training pairs were intentionally excluded to maintain strict Apache-2.0/MIT provenance, `NanoMSMARCO` (`0.6392` BF16 / `0.6462` FP32) trails MS-MARCO-trained models like `ettin-reranker-32m-v1` (`0.6936`), even while winning BEIR overall (`0.6084` vs `0.6012`) through stronger zero-shot generalization on `SciFact`, `SCIDOCS`, and `NFCorpus`.
- **Needle-in-a-haystack ceiling:** On synthetic 3K-word `LongEmbed-Needle` (`0.4189`), the model matches 568M `bge-reranker-v2-m3` (`0.4164`) and beats `ettin-32m-v1` (`0.2784`) by +14 points, but very small encoders still struggle when the relevant fact is a single unindexed 10-word needle buried in pure distractor prose.
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
