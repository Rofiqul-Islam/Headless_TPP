"""Configuration, seeding, logging and small I/O helpers."""
from __future__ import annotations

import copy
import json
import logging
import os
import random
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import yaml


# ----------------------------------------------------------------------------
# configuration
# ----------------------------------------------------------------------------

def _deep_update(base: dict, upd: dict) -> dict:
    for k, v in upd.items():
        if isinstance(v, dict) and isinstance(base.get(k), dict):
            _deep_update(base[k], v)
        else:
            base[k] = copy.deepcopy(v)
    return base


def _parse_value(s: str) -> Any:
    try:
        return yaml.safe_load(s)
    except yaml.YAMLError:
        return s


def load_config(paths: list[str] | str, overrides: list[str] | None = None) -> dict:
    """Load one or more YAML files (later ones override earlier ones; a file may name a
    parent with `inherit: other.yaml`) and apply `key.sub=value` overrides."""
    if isinstance(paths, str):
        paths = [paths]
    cfg: dict = {}
    for p in paths:
        p = Path(p)
        with open(p) as f:
            c = yaml.safe_load(f) or {}
        parent = c.pop("inherit", None)
        if parent:
            cfg = _deep_update(cfg, load_config(str(p.parent / parent)))
        cfg = _deep_update(cfg, c)
    for o in overrides or []:
        if "=" not in o:
            raise ValueError(f"override must be key=value, got {o!r}")
        key, val = o.split("=", 1)
        node = cfg
        parts = key.split(".")
        for part in parts[:-1]:
            node = node.setdefault(part, {})
        node[parts[-1]] = _parse_value(val)
    return cfg


def save_json(obj: Any, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, indent=2, default=_json_default)


def _json_default(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    return str(o)


def load_json(path: str | Path) -> Any:
    with open(path) as f:
        return json.load(f)


def read_jsonl(path: str | Path) -> list[dict]:
    out = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out


def write_jsonl(rows: list[dict], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for r in rows:
            f.write(json.dumps(r, default=_json_default) + "\n")


# ----------------------------------------------------------------------------
# seeding and logging
# ----------------------------------------------------------------------------

def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    except ImportError:  # pragma: no cover
        pass


def get_logger(out_dir: str | Path | None = None, name: str = "htpp") -> logging.Logger:
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    fmt = logging.Formatter("%(asctime)s %(levelname)s %(message)s", "%H:%M:%S")
    h = logging.StreamHandler(sys.stdout)
    h.setFormatter(fmt)
    logger.addHandler(h)
    if out_dir is not None:
        Path(out_dir).mkdir(parents=True, exist_ok=True)
        fh = logging.FileHandler(Path(out_dir) / "log.txt")
        fh.setFormatter(fmt)
        logger.addHandler(fh)
    logger.propagate = False
    return logger


class Timer:
    def __init__(self):
        self.t0 = time.time()

    def __call__(self) -> float:
        return time.time() - self.t0


def stable_hash(*parts) -> int:
    """Deterministic 63-bit hash (Python's hash() is salted per process)."""
    import hashlib
    h = hashlib.blake2b(digest_size=8)
    for p in parts:
        h.update(str(p).encode())
        h.update(b"\x1f")
    return int.from_bytes(h.digest(), "little") & ((1 << 63) - 1)


def env_info() -> dict:
    info = {"python": sys.version.split()[0], "cwd": os.getcwd()}
    try:
        import torch
        info["torch"] = torch.__version__
        info["cuda"] = torch.cuda.is_available()
        if torch.cuda.is_available():
            info["gpu"] = torch.cuda.get_device_name(0)
    except ImportError:  # pragma: no cover
        pass
    try:
        import transformers
        info["transformers"] = transformers.__version__
    except ImportError:  # pragma: no cover
        pass
    try:
        import peft
        info["peft"] = peft.__version__
    except ImportError:  # pragma: no cover
        pass
    return info
