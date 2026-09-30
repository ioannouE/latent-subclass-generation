import hashlib
import json
import logging
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
log = logging.getLogger("lsgen")


def setup_logging(level=logging.INFO):
    logging.basicConfig(level=level, format="%(asctime)s %(levelname)s %(name)s: %(message)s")


def load_config(path):
    path = Path(path)
    cfg = yaml.safe_load(path.read_text())
    cfg["_config_path"] = str(path)
    for k in ("data_root", "reports_root"):
        if k in cfg and not Path(cfg[k]).is_absolute():
            cfg[k] = str(REPO_ROOT / cfg[k])
    return cfg


def sha256_file(path, chunk=1 << 20):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while b := f.read(chunk):
            h.update(b)
    return h.hexdigest()


def sha256_bytes(b):
    return hashlib.sha256(b).hexdigest()


def git_hash():
    try:
        rev = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, text=True).strip()
        dirty = subprocess.check_output(["git", "status", "--porcelain", "--untracked-files=no"], cwd=REPO_ROOT, text=True).strip()
        return rev + ("-dirty" if dirty else "")
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "unknown"


def provenance(cfg, seed=None, dataset_manifest_hash=None):
    """Header recorded in every output file: git hash, config, seed, dataset manifest hash."""
    return {
        "git_hash": git_hash(),
        "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "config": {k: v for k, v in cfg.items() if not k.startswith("_")},
        "config_path": cfg.get("_config_path"),
        "seed": seed,
        "dataset_manifest_hash": dataset_manifest_hash,
    }


def write_json(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=1, sort_keys=False, default=str) + "\n")


def manifest_hash(files: dict) -> str:
    """Hash of a {relative_path: sha256} mapping, order-independent."""
    return sha256_bytes("\n".join(f"{k} {files[k]}" for k in sorted(files)).encode())
