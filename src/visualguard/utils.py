from __future__ import annotations

import hashlib
import json
import os
import platform
import re
import tempfile
from datetime import UTC, datetime
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path, PurePosixPath

from .errors import VisualGuardError


def utc_now() -> str:
    return datetime.now(UTC).isoformat()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_hash(value) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode()).hexdigest()


def safe_name(value: str) -> str:
    if not re.fullmatch(r"[a-zA-Z0-9][a-zA-Z0-9_.-]{0,79}", value):
        raise VisualGuardError(
            "Identifier must contain only letters, numbers, dot, dash or underscore."
        )
    return value


def resolve_inside(root: Path, relative: str) -> Path:
    pure = PurePosixPath(relative)
    if (
        not relative
        or pure.is_absolute()
        or ".." in pure.parts
        or "\\" in relative
        or ":" in relative
    ):
        raise VisualGuardError(f"Unsafe relative path: {relative}")
    result = (root / relative).resolve()
    if not result.is_relative_to(root.resolve()):
        raise VisualGuardError(f"Path escapes root: {relative}")
    return result


def write_json(path: Path, value, *, overwrite: bool = False) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and not overwrite:
        raise VisualGuardError(f"Output exists; use a new run/output path: {path}")
    data = json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n"
    # Exclusive create prevents concurrent runs from replacing one another.
    if not overwrite:
        with path.open("x", encoding="utf-8") as handle:
            handle.write(data)
    else:
        with tempfile.NamedTemporaryFile(
            mode="w", dir=path.parent, delete=False, encoding="utf-8"
        ) as handle:
            handle.write(data)
            temp_path = Path(handle.name)
        os.replace(temp_path, path)


def environment() -> dict:
    names = [
        "visualguard-ai",
        "numpy",
        "pillow",
        "scipy",
        "scikit-learn",
        "torch",
        "torchvision",
        "anomalib",
        "timm",
        "safetensors",
        "streamlit",
    ]
    packages = {}
    for name in names:
        try:
            packages[name] = version(name)
        except PackageNotFoundError:
            packages[name] = None
    import psutil

    return {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "processor": platform.processor(),
        "logical_cpus": os.cpu_count(),
        "ram_bytes": psutil.virtual_memory().total,
        "packages": packages,
    }
