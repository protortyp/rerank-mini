---
license: apache-2.0
language:
  - en
task_categories:
  - text-ranking
tags:
  - reranker
  - distillation
  - qwen3-reranker-8b
  - rag
  - code-retrieval
  - long-context
configs:
  - config_name: default
    data_files:
      - split: train
        path:
          - track_a_prescored/*.parquet
          - track_b_scored/*.parquet
          - track_c_scored/*.parquet
  - config_name: track_a_prescored
    data_files:
      - split: train
        path: track_a_prescored/*.parquet
  - config_name: track_b_long_code_tool
    data_files:
      - split: train
        path: track_b_scored/*.parquet
  - config_name: track_c_web_qa
    data_files:
      - split: train
        path: track_c_scored/*.parquet
---

# rerank-mini-distill-shards

**435,816 queries / 4,989,720 `(query, passage)` pairs** scored with **`Qwen/Qwen3-Reranker-8B`** (continuous log-odds `logit("yes") - logit("no")`), strictly curated from **Apache-2.0 / MIT** English, code, tool/API, and long-context datasets.

Used to distill [`christianengel/rerank-mini-32m`](https://huggingface.co/christianengel/rerank-mini-32m). All queries and passages are SHA-256 deduplicated against the 13-dataset evaluation suite in [`protortyp/rerank-mini`](https://github.com/protortyp/rerank-mini).

## Composition

| Track | Shards | Queries | Pairs | Domains | Teacher Scoring |
| --- | ---: | ---: | ---: | --- | --- |
| `track_a_prescored` | 33 | 315,817 | 3,789,730 | General QA, Code, Medical, RAG (`gooaq`, `codefeedback`, `pubmedqa`, `openorca`, `esci`, `rag-dataset-12000`, `paq_pairs`, `triviaqa`) | Pre-scored `Qwen3-Reranker-8B` (strictly Apache-2.0/MIT subset of `KaLM-reranker-training-data`) |
| `track_b_scored` | 12 | 60,000 | 600,000 | Long Docs (`MLDR`, `NarrativeQA`), Code (`CodeSearchNet-Python`, `Magicoder-OSS-75K`, `Glaive Code`), Tool APIs (`Glaive Function Calling v2`) | Fresh `Qwen3-Reranker-8B` (`max_length=5120`, BF16) |
| `track_c_scored` | 6 | 59,999 | 599,990 | Web QA (`sentence-transformers/gooaq`, `sentence-transformers/paq`) | Fresh `Qwen3-Reranker-8B` (`max_length=1024`, BF16) |
| **Total** | **51** | **435,816** | **4,989,720** | **100% Apache-2.0 / MIT** | **`Qwen/Qwen3-Reranker-8B`** |

## Schema

Every Parquet shard shares an identical 9-column schema:

- `query_id` (`string`): unique deterministic query identifier
- `source` (`string`): upstream dataset slug
- `domain` (`string`): `web_qa`, `code`, `tool_api`, `long_doc`, `medical`, `ecommerce`, etc.
- `license` (`string`): verified SPDX identifier (`apache-2.0` or `mit`)
- `query` (`string`): query text
- `docs` (`list<string>`): 8–16 candidate passages (positives + hard negatives)
- `doc_labels` (`list<int>`): binary relevance labels (`1` positive, `0` negative)
- `teacher_logits` (`list<float>`): raw `Qwen/Qwen3-Reranker-8B` log-odds (`yes_logit - no_logit`)
- `teacher_logits_shifted` (`list<float>`): max-subtracted per-query log-odds (`logits - max(logits)`) preserving relative margin spread

## Usage

```python
from datasets import load_dataset

ds = load_dataset("christianengel/rerank-mini-distill-shards", split="train", streaming=True)
row = next(iter(ds))
print(row["source"], row["license"], len(row["docs"]), row["teacher_logits"][:3])
```

## License & Provenance

Released under **Apache-2.0**. Every upstream source is Apache-2.0 or MIT (zero MS MARCO non-commercial or CC-BY-SA splits):
- `Qwen/Qwen3-Reranker-8B` (`Apache-2.0`)
- `KaLM-Embedding/KaLM-reranker-training-data` (`MIT`, strictly restricted to Apache-2.0/MIT subsets)
- `Shitao/MLDR` (`MIT`), `deepmind/narrativeqa` (`Apache-2.0`)
- `Nan-Do/code-search-net-python` (`Apache-2.0`), `ise-uiuc/Magicoder-OSS-Instruct-75K` (`MIT`)
- `glaiveai/glaive-code-assistant-v2` (`Apache-2.0`), `glaiveai/glaive-function-calling-v2` (`Apache-2.0`)
- `sentence-transformers/gooaq` (`Apache-2.0`), `sentence-transformers/paq` (`MIT`)
