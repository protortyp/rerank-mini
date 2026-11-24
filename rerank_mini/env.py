"""Environment and path resolution helpers."""

from __future__ import annotations

import os
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_DATA_DIR = Path("/mnt/nvme1/chris/rerank-mini")


def load_dotenv(search_paths: list[Path] | None = None) -> dict[str, str]:
    """Load key=value pairs from ../.env (or local .env) without printing secrets."""
    paths = search_paths or [REPO_ROOT.parent / ".env", REPO_ROOT / ".env"]
    loaded: dict[str, str] = {}
    for path in paths:
        if not path.is_file():
            continue
        for raw_line in path.read_text(encoding="utf-8").splitlines():
            line = raw_line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, val = line.split("=", 1)
            key = key.strip()
            val = val.strip().strip('"').strip("'")
            if key and key not in loaded:
                loaded[key] = val
    for key, val in loaded.items():
        os.environ.setdefault(key, val)
    # Normalize token aliases so downstream tools work regardless of naming convention.
    hf = os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_TOKEN")
    if hf:
        os.environ["HF_TOKEN"] = hf
        os.environ["HUGGINGFACE_TOKEN"] = hf
    vast = os.environ.get("VAST_API_KEY") or os.environ.get("VAST_AI_TOKEN")
    if vast:
        os.environ["VAST_API_KEY"] = vast
        os.environ["VAST_AI_TOKEN"] = vast
    return loaded


def get_hf_token() -> str:
    load_dotenv()
    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_TOKEN")
    if not token:
        raise RuntimeError("Missing HF_TOKEN / HUGGINGFACE_TOKEN in environment or ../.env")
    return token


def get_vast_token() -> str:
    load_dotenv()
    token = os.environ.get("VAST_API_KEY") or os.environ.get("VAST_AI_TOKEN")
    if not token:
        raise RuntimeError("Missing VAST_API_KEY / VAST_AI_TOKEN in environment or ../.env")
    return token


def get_data_dir() -> Path:
    env_dir = os.environ.get("RERANK_DATA_DIR")
    if env_dir:
        path = Path(env_dir)
    elif DEFAULT_DATA_DIR.parent.exists():
        path = DEFAULT_DATA_DIR
    else:
        path = REPO_ROOT / "data"
    path.mkdir(parents=True, exist_ok=True)
    return path
