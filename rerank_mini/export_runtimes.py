"""Export a trained ModernBertForSequenceClassification reranker to ONNX (FP32, FP16, INT8)
and verify numerical parity against PyTorch.

Uses attn_implementation='eager' during export so the graph uses standard ONNX ops
compatible with ONNX Runtime (CPU/CUDA), ONNX Runtime Web (WebGPU/WASM), and Transformers.js v4.
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np
import onnx
import onnxruntime as ort
import torch
from onnxruntime.quantization import QuantType, quantize_dynamic
from onnxruntime.transformers import float16 as ort_fp16
from transformers import AutoModelForSequenceClassification, AutoTokenizer


class _LogitsWrapper(torch.nn.Module):
    def __init__(self, model: torch.nn.Module):
        super().__init__()
        self.model = model

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        return self.model(input_ids=input_ids, attention_mask=attention_mask).logits


def export_fp32(ckpt_dir: Path, out_onnx: Path) -> None:
    out_onnx.parent.mkdir(parents=True, exist_ok=True)
    tokenizer = AutoTokenizer.from_pretrained(ckpt_dir)
    model = AutoModelForSequenceClassification.from_pretrained(
        ckpt_dir,
        attn_implementation="eager",
        dtype=torch.float32,
    ).eval()
    wrapper = _LogitsWrapper(model).eval()

    dummy = tokenizer(
        ["What is ModernBERT?", "How to parse JSON in Python?"],
        ["ModernBERT is an 8K-context encoder.", "Use json.loads(text) from the standard library."],
        padding=True,
        truncation=True,
        max_length=128,
        return_tensors="pt",
    )
    torch.onnx.export(
        wrapper,
        (dummy["input_ids"], dummy["attention_mask"]),
        str(out_onnx),
        input_names=["input_ids", "attention_mask"],
        output_names=["logits"],
        dynamic_axes={
            "input_ids": {0: "batch_size", 1: "sequence_length"},
            "attention_mask": {0: "batch_size", 1: "sequence_length"},
            "logits": {0: "batch_size"},
        },
        opset_version=17,
        do_constant_folding=True,
        dynamo=False,
    )
    # Inline external data if torch created small sidecar files so single-file ONNX stays clean.
    proto = onnx.load(str(out_onnx), load_external_data=True)
    onnx.checker.check_model(proto)
    onnx.save_model(proto, str(out_onnx), save_as_external_data=False)


def export_fp16(fp32_onnx: Path, fp16_onnx: Path) -> None:
    model_fp32 = onnx.load(str(fp32_onnx))
    model_fp16 = ort_fp16.convert_float_to_float16(
        model_fp32,
        keep_io_types=True,
        op_block_list=[
            "ArrayFeatureExtractor", "Binarizer", "CastMap", "CategoryMapper",
            "DictVectorizer", "FeatureVectorizer", "Imputer", "LabelEncoder",
            "LinearClassifier", "LinearRegressor", "Normalizer", "OneHotEncoder",
            "RandomUniformLike", "SVMClassifier", "SVMRegressor", "Scaler",
            "TreeEnsembleClassifier", "TreeEnsembleRegressor", "ZipMap",
            "NonMaxSuppression", "TopK", "RoiAlign", "Resize", "Range",
            "CumSum", "Min", "Max", "LayerNormalization", "SimplifiedLayerNormalization",
            "Softmax", "ReduceMean", "Pow", "Sqrt",
        ],
    )
    onnx.save_model(model_fp16, str(fp16_onnx), save_as_external_data=False)


def export_int8(fp32_onnx: Path, int8_onnx: Path) -> None:
    quantize_dynamic(
        model_input=str(fp32_onnx),
        model_output=str(int8_onnx),
        weight_type=QuantType.QInt8,
        op_types_to_quantize=["MatMul", "Gemm"],
    )


def check_numerical_parity(ckpt_dir: Path, onnx_files: dict[str, Path]) -> dict[str, Any]:
    tokenizer = AutoTokenizer.from_pretrained(ckpt_dir)
    model = AutoModelForSequenceClassification.from_pretrained(
        ckpt_dir, attn_implementation="eager", dtype=torch.float32
    ).eval()

    samples = [
        ("How do I read a Parquet file in Python?", "Use pyarrow.parquet.read_table(path) or pandas.read_parquet(path)."),
        ("How do I read a Parquet file in Python?", "Bananas are tropical fruits rich in potassium."),
        ("What is FlashAttention?", "FlashAttention computes exact attention using IO-aware tiling in SRAM."),
        ("What is FlashAttention?", "def add(a, b): return a + b"),
        ("Find the secret code in the document.", ("filler text " * 400) + "The secret code is ALPHA-7721."),
        ("Find the secret code in the document.", ("filler text " * 400) + "Weather forecast for Tuesday is sunny."),
    ]
    enc = tokenizer(
        [s[0] for s in samples],
        [s[1] for s in samples],
        padding=True,
        truncation=True,
        max_length=2048,
        return_tensors="pt",
    )
    with torch.inference_mode():
        pt_logits = model(input_ids=enc["input_ids"], attention_mask=enc["attention_mask"]).logits.reshape(-1).numpy()

    results: dict[str, Any] = {"pytorch_fp32_logits": [round(float(x), 5) for x in pt_logits], "variants": {}}
    feed = {
        "input_ids": enc["input_ids"].numpy().astype(np.int64),
        "attention_mask": enc["attention_mask"].numpy().astype(np.int64),
    }
    for name, path in onnx_files.items():
        sess = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
        out = sess.run(None, feed)[0].reshape(-1).astype(np.float32)
        diff = np.abs(out - pt_logits)
        corr = float(np.corrcoef(out, pt_logits)[0, 1]) if len(out) > 1 else 1.0
        results["variants"][name] = {
            "path": str(path),
            "size_mb": round(path.stat().st_size / (1024 * 1024), 2),
            "max_abs_diff": round(float(diff.max()), 6),
            "mean_abs_diff": round(float(diff.mean()), 6),
            "pearson_r": round(corr, 6),
            "logits": [round(float(x), 5) for x in out],
        }
    return results


def maybe_export_gguf(ckpt_dir: Path, gguf_dir: Path) -> dict[str, Any]:
    """Attempt GGUF F16 and Q8_0 export if llama.cpp converter & quantizer are installed locally."""
    conv = shutil.which("convert_hf_to_gguf.py") or shutil.which("llama-convert-hf-to-gguf")
    quant = shutil.which("llama-quantize")
    # Also check common local build locations.
    for cand in [
        Path.home() / "llama.cpp" / "convert_hf_to_gguf.py",
        Path("/usr/local/bin/convert_hf_to_gguf.py"),
    ]:
        if not conv and cand.is_file():
            conv = str(cand)
    if not conv:
        return {"status": "skipped", "reason": "convert_hf_to_gguf.py not found in PATH"}

    gguf_dir.mkdir(parents=True, exist_ok=True)
    f16_path = gguf_dir / "rerank-mini-32m-f16.gguf"
    q8_path = gguf_dir / "rerank-mini-32m-q8_0.gguf"
    try:
        cmd = [sys.executable, conv, str(ckpt_dir), "--outfile", str(f16_path), "--outtype", "f16"]
        subprocess.run(cmd, check=True, capture_output=True, text=True)
        info: dict[str, Any] = {
            "status": "ok",
            "f16_path": str(f16_path),
            "f16_size_mb": round(f16_path.stat().st_size / (1024 * 1024), 2),
        }
        if quant:
            subprocess.run([quant, str(f16_path), str(q8_path), "Q8_0"], check=True, capture_output=True, text=True)
            info["q8_0_path"] = str(q8_path)
            info["q8_0_size_mb"] = round(q8_path.stat().st_size / (1024 * 1024), 2)
        return info
    except subprocess.CalledProcessError as exc:
        return {"status": "unsupported", "stderr": (exc.stderr or "")[-600:]}


def main() -> int:
    parser = argparse.ArgumentParser(description="Export rerank-mini checkpoint to ONNX (FP32/FP16/INT8) + GGUF")
    parser.add_argument("--ckpt", type=Path, required=True, help="Input HuggingFace/CrossEncoder checkpoint directory")
    parser.add_argument("--out-dir", type=Path, help="Output directory (defaults to <ckpt>/onnx)")
    parser.add_argument("--try-gguf", action="store_true")
    parser.add_argument("--report", type=Path, help="Optional JSON parity report path")
    args = parser.parse_args()

    onnx_dir = args.out_dir or (args.ckpt / "onnx")
    onnx_dir.mkdir(parents=True, exist_ok=True)

    fp32 = onnx_dir / "model.onnx"
    fp16 = onnx_dir / "model_fp16.onnx"
    int8 = onnx_dir / "model_int8.onnx"

    print(f"Exporting FP32 ONNX -> {fp32}")
    export_fp32(args.ckpt, fp32)
    print(f"Converting FP16 ONNX -> {fp16}")
    export_fp16(fp32, fp16)
    print(f"Quantizing INT8 ONNX -> {int8}")
    export_int8(fp32, int8)

    parity = check_numerical_parity(args.ckpt, {"fp32": fp32, "fp16": fp16, "int8": int8})
    if args.try_gguf:
        parity["gguf"] = maybe_export_gguf(args.ckpt, args.ckpt / "gguf")

    print(json.dumps(parity, indent=2))
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(parity, indent=2) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
