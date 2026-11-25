"""Autonomous Vast.ai student distillation sweep runner.

Provisions a verified Vast.ai GPU (RTX 5090 / RTX 4090 / A100 / H200), pulls all
scored shards (`track_a_prescored/` + `track_b_scored/`) from the private HF dataset,
runs one or more structured autoresearch distillation configs in sequence on the SAME
warm instance (maximizing GPU efficiency), uploads every checkpoint to a private HF
staging repo, and guarantees instance destruction + budget ledger tracking in try/finally.
"""

from __future__ import annotations

import argparse
import json
import sys
import tarfile
import tempfile
from pathlib import Path

from infra.vast_runner import VastClient, scp_upload, ssh_exec
from rerank_mini.env import REPO_ROOT, get_hf_token, load_dotenv


def _make_code_bundle() -> Path:
    tmp = Path(tempfile.mkdtemp(prefix="rerank_train_")) / "bundle.tar.gz"
    with tarfile.open(tmp, "w:gz") as tar:
        for rel in ["rerank_mini", "train.py", "pyproject.toml"]:
            p = REPO_ROOT / rel
            if p.exists():
                tar.add(p, arcname=rel)
    return tmp


def main() -> int:
    load_dotenv()
    parser = argparse.ArgumentParser(description="Dispatch student training sweep to Vast.ai")
    parser.add_argument("--sweep-spec", type=Path, required=True, help="JSON file listing training runs")
    parser.add_argument("--hf-data-repo", type=str, default="christianengel/rerank-mini-distill-shards")
    parser.add_argument("--hf-ckpt-repo", type=str, default="christianengel/rerank-mini-32m-staging")
    parser.add_argument("--max-dph", type=float, default=1.6)
    parser.add_argument("--max-hours", type=float, default=5.0)
    parser.add_argument("--min-vram", type=float, default=40.0)
    parser.add_argument("--gpu", action="append", default=None)
    args = parser.parse_args()

    hf_token = get_hf_token()
    spec = json.loads(args.sweep_spec.read_text(encoding="utf-8"))
    runs = spec["runs"]
    gpu_names = args.gpu or ["A100 SXM4", "A100X", "RTX 5090", "RTX 4090", "H100 SXM", "H200 NVL"]

    bundle = _make_code_bundle()
    client = VastClient()

    with client.provision(
        gpu_names=gpu_names,
        min_gpu_ram_gb=args.min_vram,
        min_disk_gb=120.0,
        max_dph=args.max_dph,
        disk_gb=120,
        label=f"rerank-mini-sweep-{spec.get('name', 'v1')}",
        max_hours=args.max_hours,
    ) as inst:
        print(
            f"Provisioned Vast instance {inst.instance_id}: {inst.gpu_name} "
            f"(${inst.dph_total:.3f}/hr) at {inst.ssh_host}:{inst.ssh_port}",
            flush=True,
        )
        scp_upload(inst, bundle, "/workspace/bundle.tar.gz")
        scp_upload(inst, args.sweep_spec, "/workspace/sweep_spec.json")

        remote_script = f"""
set -euo pipefail
cd /workspace
tar -xzf bundle.tar.gz
pip install -q -U "transformers>=4.51,<5.0" "sentence-transformers>=5.0" "huggingface_hub>=0.30,<1.0" pyarrow tqdm brotli

python3 - <<'PY'
import json, os, subprocess, sys
from pathlib import Path
from huggingface_hub import HfApi, snapshot_download

token = os.environ["HF_TOKEN"]
api = HfApi(token=token)
api.create_repo(repo_id="{args.hf_ckpt_repo}", repo_type="model", private=True, exist_ok=True)

data_root = Path("/workspace/distill_shards")
snapshot_download(
    repo_id="{args.hf_data_repo}",
    repo_type="dataset",
    token=token,
    local_dir=str(data_root),
    allow_patterns=["track_a_prescored/*.parquet", "track_b_scored/*.parquet", "track_c_scored/*.parquet"],
)
data_dirs = [str(p) for p in [data_root / "track_a_prescored", data_root / "track_b_scored", data_root / "track_c_scored"] if p.is_dir() and list(p.glob("*.parquet"))]
print("Using data dirs:", data_dirs, flush=True)

spec = json.loads(Path("/workspace/sweep_spec.json").read_text())
for run in spec["runs"]:
    name = run["name"]
    out_dir = Path("/workspace/ckpts") / name
    cmd = [sys.executable, "/workspace/train.py", "--data-dirs", *data_dirs, "--out-dir", str(out_dir)]
    for k, v in run["args"].items():
        if k == "backbone" and isinstance(v, str) and v.startswith("staging:"):
            sub = v.split(":", 1)[1]
            local_bb = Path("/workspace/ckpts_in")
            snapshot_download(
                repo_id="{args.hf_ckpt_repo}",
                repo_type="model",
                token=token,
                local_dir=str(local_bb),
                allow_patterns=[f"{{sub}}/*"],
            )
            v = str(local_bb / sub)
        if isinstance(v, bool):
            if v:
                cmd.append(f"--{{k}}")
        elif isinstance(v, list):
            cmd.extend([f"--{{k}}", *[str(x) for x in v]])
        else:
            cmd.extend([f"--{{k}}", str(v)])
    print(f"\\n========== RUNNING {{name}} ==========", flush=True)
    print("CMD:", " ".join(cmd), flush=True)
    env = os.environ.copy()
    env["PYTHONPATH"] = "/workspace"
    subprocess.run(cmd, check=True, env=env)
    api.upload_folder(
        folder_path=str(out_dir),
        path_in_repo=name,
        repo_id="{args.hf_ckpt_repo}",
        repo_type="model",
    )
    print(f"[uploaded] {{name}} -> {args.hf_ckpt_repo}/{{name}}", flush=True)
PY
"""
        ssh_exec(inst, remote_script, env={"HF_TOKEN": hf_token, "HUGGINGFACE_TOKEN": hf_token})

    print("Sweep finished and Vast.ai instance destroyed cleanly.", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
