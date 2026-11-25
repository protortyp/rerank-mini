"""Download a trained checkpoint subfolder from the private HF staging repo and evaluate it
with immutable eval_locked.py on the local RTX 3090.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

from huggingface_hub import snapshot_download

from rerank_mini.env import REPO_ROOT, get_data_dir, get_hf_token, load_dotenv


def main() -> int:
    load_dotenv()
    parser = argparse.ArgumentParser(description="Pull staging checkpoint from HF and run eval_locked.py")
    parser.add_argument("--run-name", type=str, required=True, help="Subfolder name inside staging repo")
    parser.add_argument("--hf-ckpt-repo", type=str, default="christianengel/rerank-mini-32m-staging")
    parser.add_argument("--notes", type=str, default="")
    parser.add_argument("--skip-cpu-onnx-latency", action="store_true", default=True)
    args = parser.parse_args()

    local_root = get_data_dir() / "ckpts_staging"
    local_root.mkdir(parents=True, exist_ok=True)
    snapshot_download(
        repo_id=args.hf_ckpt_repo,
        repo_type="model",
        token=get_hf_token(),
        local_dir=str(local_root),
        allow_patterns=[f"{args.run_name}/*"],
    )
    ckpt_dir = local_root / args.run_name
    if not (ckpt_dir / "config.json").is_file():
        raise FileNotFoundError(f"Checkpoint config.json missing under {ckpt_dir}")

    out_json = REPO_ROOT / "results" / f"{args.run_name}.json"
    cmd = [
        sys.executable,
        str(REPO_ROOT / "eval_locked.py"),
        "--model",
        str(ckpt_dir),
        "--tag",
        args.run_name,
        "--out",
        str(out_json),
        "--log-experiment",
        "--notes",
        args.notes or f"Vast sweep checkpoint {args.run_name}",
    ]
    if args.skip_cpu_onnx_latency:
        cmd.append("--skip-cpu-onnx-latency")
    subprocess.run(cmd, check=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
