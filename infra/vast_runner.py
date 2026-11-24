"""Verified-host Vast.ai provisioning, SSH execution, watchdog teardown, and spend tracking.

Rules enforced by this module:
- Always filters verified=true, reliability > 0.99, inet_down > 500.
- Hard budget cap ($400.00 USD) tracked in budget_ledger.json.
- Auto-destroy in try/finally (never leaves instances stopped).
- Never modifies or destroys pre-existing instances not created by this runner.
"""

from __future__ import annotations

import argparse
import json
import os
import shlex
import subprocess
import sys
import time
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator

import requests

from rerank_mini.env import REPO_ROOT, get_vast_token

VAST_API_BASE = "https://console.vast.ai/api/v0"
DEFAULT_LEDGER_PATH = REPO_ROOT / "budget_ledger.json"
HARD_BUDGET_CAP_USD = 400.0

SSH_BASE_OPTS = [
    "-o", "StrictHostKeyChecking=no",
    "-o", "UserKnownHostsFile=/dev/null",
    "-o", "LogLevel=ERROR",
    "-o", "ConnectTimeout=15",
    "-o", "ServerAliveInterval=30",
    "-o", "ServerAliveCountMax=6",
]


@dataclass
class VastOffer:
    offer_id: int
    gpu_name: str
    num_gpus: int
    gpu_ram_mb: int
    dph_total: float
    reliability: float
    inet_down: float
    inet_up: float
    disk_space: float
    geolocation: str
    cuda_max_good: float


@dataclass
class ActiveInstance:
    instance_id: int
    offer_id: int
    gpu_name: str
    num_gpus: int
    dph_total: float
    label: str
    ssh_host: str
    ssh_port: int
    created_at: float
    max_hours: float


class BudgetExceededError(RuntimeError):
    """Raised when launching a job would exceed the project budget cap or account credit."""


class BudgetLedger:
    """Append-only spend ledger persisted to budget_ledger.json."""

    def __init__(self, path: Path = DEFAULT_LEDGER_PATH, cap_usd: float = HARD_BUDGET_CAP_USD):
        self.path = path
        self.cap_usd = cap_usd
        if not self.path.exists():
            self._write({
                "project": "rerank-mini",
                "budget_cap_usd": cap_usd,
                "total_spent_usd": 0.0,
                "entries": [],
            })

    def _read(self) -> dict[str, Any]:
        return json.loads(self.path.read_text(encoding="utf-8"))

    def _write(self, data: dict[str, Any]) -> None:
        tmp = self.path.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")
        tmp.replace(self.path)

    def total_spent(self) -> float:
        return float(self._read().get("total_spent_usd", 0.0))

    def remaining(self) -> float:
        return max(0.0, self.cap_usd - self.total_spent())

    def check_can_afford(self, estimated_usd: float, account_credit_usd: float | None = None) -> None:
        spent = self.total_spent()
        if spent + estimated_usd > self.cap_usd:
            raise BudgetExceededError(
                f"Estimated spend ${estimated_usd:.2f} + spent ${spent:.2f} exceeds "
                f"hard cap ${self.cap_usd:.2f}"
            )
        if account_credit_usd is not None and estimated_usd > account_credit_usd:
            raise BudgetExceededError(
                f"Estimated spend ${estimated_usd:.2f} exceeds live Vast.ai credit "
                f"${account_credit_usd:.2f}"
            )

    def record(
        self,
        *,
        instance_id: int,
        gpu_name: str,
        num_gpus: int,
        label: str,
        dph_total: float,
        started_at: float,
        ended_at: float,
        status: str,
        notes: str = "",
    ) -> dict[str, Any]:
        duration_sec = max(0.0, ended_at - started_at)
        cost_usd = round((duration_sec / 3600.0) * dph_total, 4)
        data = self._read()
        entry = {
            "timestamp_utc": datetime.fromtimestamp(ended_at, tz=timezone.utc).isoformat(),
            "instance_id": instance_id,
            "gpu_name": gpu_name,
            "num_gpus": num_gpus,
            "label": label,
            "dph_total": round(dph_total, 4),
            "duration_sec": round(duration_sec, 1),
            "cost_usd": cost_usd,
            "status": status,
            "notes": notes,
        }
        data["entries"].append(entry)
        data["total_spent_usd"] = round(sum(e["cost_usd"] for e in data["entries"]), 4)
        self._write(data)
        return entry


class VastClient:
    """Thin REST client for Vast.ai v0 API."""

    def __init__(self, api_key: str | None = None, ledger: BudgetLedger | None = None):
        self.api_key = api_key or get_vast_token()
        self.ledger = ledger or BudgetLedger()
        self.session = requests.Session()
        self.session.headers.update({"Authorization": f"Bearer {self.api_key}"})

    def _req(self, method: str, endpoint: str, **kwargs: Any) -> Any:
        url = f"{VAST_API_BASE}{endpoint}"
        resp = self.session.request(method, url, timeout=kwargs.pop("timeout", 30), **kwargs)
        if not resp.ok:
            raise RuntimeError(f"Vast API {method} {endpoint} failed ({resp.status_code}): {resp.text[:400]}")
        return resp.json()

    def get_credit_usd(self) -> float:
        data = self._req("GET", "/users/current/")
        return float(data.get("credit", 0.0))

    def list_instances(self) -> list[dict[str, Any]]:
        data = self._req("GET", "/instances/")
        return list(data.get("instances", []))

    def get_instance(self, instance_id: int) -> dict[str, Any] | None:
        for inst in self.list_instances():
            if int(inst.get("id", -1)) == instance_id:
                return inst
        return None

    def search_offers(
        self,
        *,
        gpu_names: list[str] | None = None,
        num_gpus: int = 1,
        min_gpu_ram_gb: float = 24.0,
        min_disk_gb: float = 80.0,
        max_dph: float = 4.5,
        min_cuda: float = 12.1,
        min_reliability: float = 0.99,
        min_inet_down: float = 500.0,
        limit: int = 20,
    ) -> list[VastOffer]:
        query: dict[str, Any] = {
            "verified": {"eq": True},
            "external": {"eq": False},
            "rentable": {"eq": True},
            "rented": {"eq": False},
            "reliability": {"gt": min_reliability},
            "inet_down": {"gt": min_inet_down},
            "num_gpus": {"eq": num_gpus},
            "gpu_ram": {"gte": int(min_gpu_ram_gb * 1024)},
            "disk_space": {"gte": min_disk_gb},
            "dph_total": {"lte": max_dph},
            "cuda_max_good": {"gte": min_cuda},
            "type": "on-demand",
            "order": [["dph_total", "asc"]],
            "limit": limit,
        }
        if gpu_names:
            query["gpu_name"] = {"in": gpu_names}

        data = self._req("POST", "/bundles/", json=query)
        offers: list[VastOffer] = []
        for row in data.get("offers", []):
            offers.append(
                VastOffer(
                    offer_id=int(row["id"]),
                    gpu_name=str(row.get("gpu_name", "unknown")),
                    num_gpus=int(row.get("num_gpus", num_gpus)),
                    gpu_ram_mb=int(row.get("gpu_ram", 0)),
                    dph_total=float(row.get("dph_total", 999.0)),
                    reliability=float(row.get("reliability", 0.0)),
                    inet_down=float(row.get("inet_down", 0.0)),
                    inet_up=float(row.get("inet_up", 0.0)),
                    disk_space=float(row.get("disk_space", 0.0)),
                    geolocation=str(row.get("geolocation", "")),
                    cuda_max_good=float(row.get("cuda_max_good", 0.0)),
                )
            )
        offers.sort(key=lambda o: o.dph_total)
        return offers

    def create_instance(
        self,
        offer: VastOffer,
        *,
        image: str = "pytorch/pytorch:2.6.0-cuda12.6-cudnn9-devel",
        disk_gb: int = 100,
        label: str = "rerank-mini",
        onstart: str = "",
        max_hours: float = 4.0,
        wait_ready_sec: int = 420,
    ) -> ActiveInstance:
        credit = self.get_credit_usd()
        est_cost = offer.dph_total * max_hours
        self.ledger.check_can_afford(est_cost, account_credit_usd=credit)

        payload: dict[str, Any] = {
            "client_id": "me",
            "image": image,
            "disk": disk_gb,
            "label": label,
            "runtype": "ssh",
            "ssh": True,
            "direct": True,
        }
        if onstart:
            payload["onstart"] = onstart

        resp = self._req("PUT", f"/asks/{offer.offer_id}/", json=payload)
        instance_id = int(resp["new_contract"])
        created_at = time.time()

        try:
            ssh_host, ssh_port = self._wait_ssh_ready(instance_id, timeout_sec=wait_ready_sec)
        except Exception:
            self.destroy_instance(instance_id)
            self.ledger.record(
                instance_id=instance_id,
                gpu_name=offer.gpu_name,
                num_gpus=offer.num_gpus,
                label=label,
                dph_total=offer.dph_total,
                started_at=created_at,
                ended_at=time.time(),
                status="boot_failed_destroyed",
                notes="Destroyed automatically because SSH never became ready.",
            )
            raise

        return ActiveInstance(
            instance_id=instance_id,
            offer_id=offer.offer_id,
            gpu_name=offer.gpu_name,
            num_gpus=offer.num_gpus,
            dph_total=offer.dph_total,
            label=label,
            ssh_host=ssh_host,
            ssh_port=ssh_port,
            created_at=created_at,
            max_hours=max_hours,
        )

    def _wait_ssh_ready(self, instance_id: int, timeout_sec: int = 420) -> tuple[str, int]:
        deadline = time.time() + timeout_sec
        ssh_host = ""
        ssh_port = 0
        while time.time() < deadline:
            inst = self.get_instance(instance_id)
            if inst is None:
                raise RuntimeError(f"Instance {instance_id} disappeared during boot")
            status = str(inst.get("actual_status") or "")
            if status == "running":
                # Prefer direct port mapping when available, else Vast SSH proxy.
                ports = inst.get("ports") or {}
                direct = ports.get("22/tcp")
                if isinstance(direct, list) and direct and inst.get("public_ipaddr"):
                    ssh_host = str(inst["public_ipaddr"])
                    ssh_port = int(direct[0]["HostPort"])
                elif inst.get("ssh_host") and inst.get("ssh_port"):
                    ssh_host = str(inst["ssh_host"])
                    ssh_port = int(inst["ssh_port"])
                if ssh_host and ssh_port:
                    probe = subprocess.run(
                        ["ssh", *SSH_BASE_OPTS, "-p", str(ssh_port), f"root@{ssh_host}", "echo READY"],
                        capture_output=True,
                        text=True,
                        timeout=20,
                    )
                    if probe.returncode == 0 and "READY" in probe.stdout:
                        return ssh_host, ssh_port
            time.sleep(8)
        raise TimeoutError(f"Instance {instance_id} did not become SSH-ready within {timeout_sec}s")

    def destroy_instance(self, instance_id: int) -> None:
        for attempt in range(5):
            try:
                self._req("DELETE", f"/instances/{instance_id}/")
                return
            except Exception as exc:
                if attempt == 4:
                    raise RuntimeError(f"Failed to destroy Vast instance {instance_id}: {exc}") from exc
                time.sleep(3 * (attempt + 1))

    @contextmanager
    def provision(
        self,
        *,
        gpu_names: list[str] | None = None,
        num_gpus: int = 1,
        min_gpu_ram_gb: float = 24.0,
        min_disk_gb: float = 100.0,
        max_dph: float = 4.5,
        image: str = "pytorch/pytorch:2.6.0-cuda12.6-cudnn9-devel",
        disk_gb: int = 100,
        label: str = "rerank-mini",
        max_hours: float = 4.0,
        max_retries: int = 3,
    ) -> Iterator[ActiveInstance]:
        """Provision a verified GPU instance and guarantee destruction on exit."""
        offers = self.search_offers(
            gpu_names=gpu_names,
            num_gpus=num_gpus,
            min_gpu_ram_gb=min_gpu_ram_gb,
            min_disk_gb=min_disk_gb,
            max_dph=max_dph,
        )
        if not offers:
            raise RuntimeError(f"No verified Vast.ai offers matched filters (gpus={gpu_names}, max_dph={max_dph})")

        inst: ActiveInstance | None = None
        last_err: Exception | None = None
        for offer in offers[:max_retries]:
            try:
                inst = self.create_instance(
                    offer,
                    image=image,
                    disk_gb=disk_gb,
                    label=label,
                    max_hours=max_hours,
                )
                break
            except Exception as exc:
                last_err = exc
        if inst is None:
            raise RuntimeError(f"All {max_retries} Vast.ai offers failed to boot: {last_err}")

        status = "completed"
        notes = ""
        try:
            yield inst
        except BaseException as exc:
            status = "failed"
            notes = f"{type(exc).__name__}: {exc}"[:300]
            raise
        finally:
            try:
                self.destroy_instance(inst.instance_id)
            finally:
                self.ledger.record(
                    instance_id=inst.instance_id,
                    gpu_name=inst.gpu_name,
                    num_gpus=inst.num_gpus,
                    label=inst.label,
                    dph_total=inst.dph_total,
                    started_at=inst.created_at,
                    ended_at=time.time(),
                    status=status,
                    notes=notes,
                )


def ssh_exec(
    inst: ActiveInstance,
    cmd: str,
    *,
    env: dict[str, str] | None = None,
    timeout: int | None = None,
    stream: bool = True,
) -> subprocess.CompletedProcess[str]:
    """Execute a shell command on an ActiveInstance without echoing secret env vars."""
    prefix_parts = []
    if env:
        for k, v in env.items():
            prefix_parts.append(f"export {k}={shlex.quote(v)}")
    full_cmd = "\n".join([*prefix_parts, cmd])
    ssh_cmd = [
        "ssh",
        *SSH_BASE_OPTS,
        "-p",
        str(inst.ssh_port),
        f"root@{inst.ssh_host}",
        "bash -s",
    ]
    if stream:
        proc = subprocess.run(ssh_cmd, input=full_cmd, text=True, timeout=timeout)
    else:
        proc = subprocess.run(ssh_cmd, input=full_cmd, capture_output=True, text=True, timeout=timeout)
    if proc.returncode != 0:
        raise subprocess.CalledProcessError(proc.returncode, "ssh_exec", proc.stdout, proc.stderr)
    return proc


def scp_upload(inst: ActiveInstance, local_path: Path, remote_path: str) -> None:
    subprocess.run(
        [
            "scp",
            *SSH_BASE_OPTS,
            "-P",
            str(inst.ssh_port),
            "-r",
            str(local_path),
            f"root@{inst.ssh_host}:{remote_path}",
        ],
        check=True,
    )


def scp_download(inst: ActiveInstance, remote_path: str, local_path: Path) -> None:
    local_path.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        [
            "scp",
            *SSH_BASE_OPTS,
            "-P",
            str(inst.ssh_port),
            "-r",
            f"root@{inst.ssh_host}:{remote_path}",
            str(local_path),
        ],
        check=True,
    )


def _cli() -> int:
    parser = argparse.ArgumentParser(description="Vast.ai verified-host search and status CLI")
    sub = parser.add_subparsers(dest="cmd", required=True)

    sub.add_parser("status", help="Show account credit, budget ledger, and running instances")

    p_search = sub.add_parser("search", help="Search verified Vast.ai GPU offers")
    p_search.add_argument("--gpu", action="append", help="Filter GPU name (repeatable), e.g. 'H200' 'RTX 5090'")
    p_search.add_argument("--num-gpus", type=int, default=1)
    p_search.add_argument("--min-vram", type=float, default=24.0)
    p_search.add_argument("--min-disk", type=float, default=100.0)
    p_search.add_argument("--max-dph", type=float, default=4.5)
    p_search.add_argument("--limit", type=int, default=10)

    args = parser.parse_args()
    client = VastClient()

    if args.cmd == "status":
        credit = client.get_credit_usd()
        insts = client.list_instances()
        print(f"Vast.ai live credit: ${credit:.2f}")
        print(
            f"Project budget ledger: spent=${client.ledger.total_spent():.2f} / "
            f"cap=${client.ledger.cap_usd:.2f} (remaining=${client.ledger.remaining():.2f})"
        )
        print(f"Running/existing instances ({len(insts)}):")
        for i in insts:
            print(
                f"  id={i.get('id')} gpu={i.get('gpu_name')} x{i.get('num_gpus')} "
                f"status={i.get('actual_status')} dph=${float(i.get('dph_total', 0)):.3f}/hr "
                f"label={i.get('label')}"
            )
        return 0

    if args.cmd == "search":
        offers = client.search_offers(
            gpu_names=args.gpu,
            num_gpus=args.num_gpus,
            min_gpu_ram_gb=args.min_vram,
            min_disk_gb=args.min_disk,
            max_dph=args.max_dph,
            limit=args.limit,
        )
        print(json.dumps([asdict(o) for o in offers], indent=2))
        return 0

    return 1


if __name__ == "__main__":
    sys.exit(_cli())
