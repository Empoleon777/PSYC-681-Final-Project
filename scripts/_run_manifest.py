"""Reproducibility manifest helpers.

Each model run should emit a JSON sidecar describing exactly what data, seed,
and code version produced its outputs. Reviewers and a future "run B6 on real
data with 3 seeds" sweep both depend on this.
"""

from __future__ import annotations

import csv
import hashlib
import json
import platform
import socket
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional


def _read_bytes(path: Path, chunk: int = 1 << 20) -> Iterable[bytes]:
    with open(path, "rb") as f:
        while True:
            data = f.read(chunk)
            if not data:
                return
            yield data


def hash_file(path: Path) -> Optional[Dict[str, Any]]:
    """Return {sha256, size_bytes, n_rows?} for a file, or None if missing.

    n_rows is filled in for .csv files (line count minus header).
    """
    if path is None or not Path(path).exists():
        return None
    h = hashlib.sha256()
    size = 0
    for chunk in _read_bytes(Path(path)):
        h.update(chunk)
        size += len(chunk)
    payload: Dict[str, Any] = {
        "path": str(path),
        "sha256": h.hexdigest(),
        "size_bytes": size,
    }
    if str(path).lower().endswith(".csv"):
        try:
            with open(path, "r", encoding="utf-8", newline="") as f:
                # Subtract header.
                payload["n_rows"] = sum(1 for _ in csv.reader(f)) - 1
        except OSError:
            pass
    return payload


def _git_value(args: List[str]) -> Optional[str]:
    try:
        out = subprocess.run(
            ["git"] + args,
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
        if out.returncode == 0:
            return out.stdout.strip() or None
    except (OSError, subprocess.SubprocessError):
        return None
    return None


def git_info() -> Dict[str, Any]:
    return {
        "commit": _git_value(["rev-parse", "HEAD"]),
        "branch": _git_value(["rev-parse", "--abbrev-ref", "HEAD"]),
        "dirty": _git_value(["status", "--porcelain"]) not in (None, ""),
    }


def env_info() -> Dict[str, Any]:
    info: Dict[str, Any] = {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "hostname": socket.gethostname(),
    }
    try:
        import torch  # type: ignore

        info["torch"] = torch.__version__
        info["cuda_available"] = bool(torch.cuda.is_available())
        if torch.cuda.is_available():
            info["cuda_device"] = torch.cuda.get_device_name(0)
    except ImportError:
        pass
    try:
        import transformers  # type: ignore

        info["transformers"] = transformers.__version__
    except ImportError:
        pass
    try:
        import sklearn  # type: ignore

        info["sklearn"] = sklearn.__version__
    except ImportError:
        pass
    return info


def build_manifest(
    *,
    run_name: str,
    seed: Optional[int] = None,
    inputs: Optional[Dict[str, Any]] = None,
    outputs: Optional[Dict[str, Any]] = None,
    config: Optional[Dict[str, Any]] = None,
    extra: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Return a manifest dict; pair with `write_manifest` to persist it.

    `inputs` should be a {name: Path} mapping; each path is hashed.
    `outputs` may be a free-form dict (paths or scalars).
    """
    hashed_inputs: Dict[str, Any] = {}
    for name, value in (inputs or {}).items():
        if isinstance(value, (str, Path)):
            hashed_inputs[name] = hash_file(Path(value))
        else:
            hashed_inputs[name] = value
    return {
        "run_name": run_name,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "seed": seed,
        "git": git_info(),
        "environment": env_info(),
        "config": config or {},
        "inputs": hashed_inputs,
        "outputs": outputs or {},
        "extra": extra or {},
    }


def write_manifest(path: Path, manifest: Dict[str, Any]) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(manifest, indent=2, ensure_ascii=True), encoding="utf-8")
