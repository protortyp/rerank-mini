"""Autonomous Vast.ai H200/A100 teacher scoring dispatcher.

Provisions a verified Vast.ai GPU (preferring H200 NVL 141 GB, fallback H100/A100 80 GB),
streams unscored Track B shards from private HF dataset `christianengel/rerank-mini-distill-shards`,
scores them with Qwen/Qwen3-Reranker-8B, uploads every completed shard immediately to HF,
and guarantees instance destruction + budget ledger recording in try/finally.
"""

from __future__ import annotations

import argparse
import sys
import tarfile
import tempfile
from pathlib import Path

from infra.vast_runner import VastClient, scp_upload, ssh_exec
from rerank_mini.env import REPO_ROOT, get_hf_token, load_dotenv


def _make_code_bundle() -> Path:
    tmp = Path(tempfile.mkdtemp(prefix="rerank_bundle_")) / "bundle.tar.gz"
    with tarfile.open(tmp, "w:gz") as tar:
        for rel in ["rerank_mini", "train.py", "pyproject.toml"]:
            p = REPO_ROOT / rel
            if p.exists():
                tar.add(p, arcname=rel)
    return tmp


def main() -> int:
    load_dotenv()
    parser = argparse.ArgumentParser(description="Dispatch Qwen3-Reranker-8B shard scoring to Vast.ai")
    parser.add_argument("--hf-repo", type=str, default="christianengel/rerank-mini-distill-shards")
    parser.add_argument("--max-length", type=int, default=4096)
    parser.add_argument("--base-batch-size", type=int, default=32)
    parser.add_argument("--max-dph", type=float, default=4.5)
    parser.add_argument("--max-hours", type=float, default=4.0)
    parser.add_argument("--gpu", action="append", default=None, help="Preferred GPU names")
    args = parser.parse_args()

    hf_token = get_hf_token()
    gpu_names = args.gpu or ["H200 NVL", "H200", "H200 SXM", "H100 SXM", "H100 NVL", "A100 SXM4"]

    bundle = _make_code_bundle()
    client = VastClient()
    print(f"Searching verified Vast.ai offers for {gpu_names} (max ${args.max_dph}/hr)...", flush=True)

    with client.provision(
        gpu_names=gpu_names,
        min_gpu_ram_gb=75.0,
        min_disk_gb=120.0,
        max_dph=args.max_dph,
        disk_gb=120,
        label="rerank-mini-teacher-8b",
        max_hours=args.max_hours,
    ) as inst:
        print(
            f"Provisioned Vast instance {inst.instance_id}: {inst.gpu_name} "
            f"(${inst.dph_total:.3f}/hr) at {inst.ssh_host}:{inst.ssh_port}",
            flush=True,
        )
        scp_upload(inst, bundle, "/workspace/bundle.tar.gz")

        remote_script = f"""
set -euo pipefail
cd /workspace
tar -xzf bundle.tar.gz
pip install -q -U "transformers>=4.51" "huggingface_hub>=0.30" pyarrow tqdm accelerate

python3 - <<'PY'
import os
from pathlib import Path
from huggingface_hub import snapshot_download
token = os.environ["HF_TOKEN"]
snapshot_download(
    repo_id="{args.hf_repo}",
    repo_type="dataset",
    token=token,
    local_dir="/workspace/shards",
    allow_patterns=["track_b_unscored/*.parquet"],
)
print("Downloaded unscored shards:", len(list(Path("/workspace/shards/track_b_unscored").glob("*.parquet"))))
PY

PYTHONPATH=/workspace python3 -m rerank_mini.teacher_score \\
    --input-dir /workspace/shards/track_b_unscored \\
    --output-dir /workspace/shards/track_b_scored \\
    --hf-repo "{args.hf_repo}" \\
    --hf-prefix "track_b_scored" \\
    --max-length {args.max_length} \\
    --base-batch-size {args.base_batch_size}
"""
        ssh_exec(inst, remote_script, env={"HF_TOKEN": hf_token, "HUGGINGFACE_TOKEN": hf_token})

    print("Teacher scoring job finished and Vast.ai instance destroyed cleanly.", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
