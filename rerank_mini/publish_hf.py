"""Stage a verified rerank-mini release directory and upload to Hugging Face (private by default)."""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

from huggingface_hub import HfApi

from rerank_mini.env import REPO_ROOT, load_dotenv

REQUIRED_RELEASE_FILES = [
    "config.json",
    "model.safetensors",
    "tokenizer.json",
    "tokenizer_config.json",
    "sentence_bert_config.json",
    "README.md",
    "LICENSE",
    "onnx/model.onnx",
    "onnx/model_fp16.onnx",
    "onnx/model_int8.onnx",
]


def verify_release_dir(release_dir: Path) -> list[str]:
    missing = [rel for rel in REQUIRED_RELEASE_FILES if not (release_dir / rel).is_file()]
    if missing:
        raise FileNotFoundError(f"Release directory {release_dir} missing required files: {missing}")

    cfg = json.loads((release_dir / "config.json").read_text(encoding="utf-8"))
    if cfg.get("num_labels", len(cfg.get("id2label", {}))) != 1:
        raise ValueError(f"Expected 1-label cross-encoder config, got id2label={cfg.get('id2label')}")

    return sorted(str(p.relative_to(release_dir)) for p in release_dir.rglob("*") if p.is_file())


def stage_release(
    ckpt_dir: Path,
    release_dir: Path,
    card_path: Path,
    notice_path: Path | None = None,
) -> list[str]:
    release_dir.mkdir(parents=True, exist_ok=True)
    for name in [
        "config.json",
        "model.safetensors",
        "tokenizer.json",
        "tokenizer_config.json",
        "sentence_bert_config.json",
    ]:
        src = ckpt_dir / name
        if not src.is_file():
            raise FileNotFoundError(f"Missing checkpoint file: {src}")
        shutil.copy2(src, release_dir / name)

    for subdir in ("onnx", "gguf"):
        src_sub = ckpt_dir / subdir
        if src_sub.is_dir():
            dst_sub = release_dir / subdir
            if dst_sub.exists():
                shutil.rmtree(dst_sub)
            shutil.copytree(src_sub, dst_sub)

    shutil.copy2(card_path, release_dir / "README.md")
    shutil.copy2(REPO_ROOT / "LICENSE", release_dir / "LICENSE")
    if notice_path and notice_path.is_file():
        shutil.copy2(notice_path, release_dir / "NOTICE")

    return verify_release_dir(release_dir)


def upload_model(release_dir: Path, repo_id: str, *, private: bool = True) -> str:
    load_dotenv()
    files = verify_release_dir(release_dir)
    api = HfApi()
    api.create_repo(repo_id=repo_id, repo_type="model", private=private, exist_ok=True)
    api.upload_folder(
        repo_id=repo_id,
        repo_type="model",
        folder_path=str(release_dir),
        commit_message="Upload rerank-mini-32m release (Safetensors, ONNX FP32/FP16/INT8, GGUF F16/Q8_0, model card)",
    )
    print(f"Uploaded {len(files)} files -> https://huggingface.co/{repo_id} (private={private})")
    return f"https://huggingface.co/{repo_id}"


def upload_space(space_dir: Path, repo_id: str, *, private: bool = True) -> str:
    load_dotenv()
    api = HfApi()
    api.create_repo(
        repo_id=repo_id,
        repo_type="space",
        space_sdk="static",
        private=private,
        exist_ok=True,
    )
    api.upload_folder(
        repo_id=repo_id,
        repo_type="space",
        folder_path=str(space_dir),
        commit_message="Upload client-side WebGPU comparison Space (rerank-mini-32m vs MiniLM-L6-v2)",
    )
    print(f"Uploaded Space -> https://huggingface.co/spaces/{repo_id} (private={private})")
    return f"https://huggingface.co/spaces/{repo_id}"


def main() -> int:
    parser = argparse.ArgumentParser(description="Stage and publish rerank-mini release to Hugging Face")
    parser.add_argument("--ckpt", type=Path, help="Source checkpoint directory (with onnx/ and gguf/)")
    parser.add_argument("--release-dir", type=Path, required=True, help="Target clean release directory")
    parser.add_argument("--card", type=Path, default=REPO_ROOT / "card" / "README.md")
    parser.add_argument("--notice", type=Path, default=REPO_ROOT / "NOTICE")
    parser.add_argument("--model-repo", type=str, default="christianengel/rerank-mini-32m")
    parser.add_argument("--space-repo", type=str, default="christianengel/rerank-mini-32m-demo")
    parser.add_argument("--space-dir", type=Path, default=REPO_ROOT / "space")
    parser.add_argument("--upload", action="store_true", help="Upload staged release and Space to Hugging Face")
    parser.add_argument("--public", action="store_true", help="Create repos as public (default: private)")
    args = parser.parse_args()

    if args.ckpt:
        files = stage_release(args.ckpt, args.release_dir, args.card, args.notice)
        print(f"Staged {len(files)} files in {args.release_dir}:")
        for f in files:
            sz = (args.release_dir / f).stat().st_size / (1024 * 1024)
            print(f"  {f:35s} ({sz:6.2f} MB)")

    if args.upload:
        upload_model(args.release_dir, args.model_repo, private=not args.public)
        upload_space(args.space_dir, args.space_repo, private=not args.public)

    return 0


if __name__ == "__main__":
    sys.exit(main())
