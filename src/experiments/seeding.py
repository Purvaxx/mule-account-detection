"""
Layer 6 - Seed Control
=======================

One place that sets every random source the framework can reach, and one place
that reports how far reproducibility actually extends.

``set_global_seeds`` is called by the runner once per fold with a
fold-derived seed, so re-running fold 2 alone reproduces fold 2 from a full run.

Honesty about determinism
-------------------------
Seeding does not make a run bit-identical.  Floating-point reduction order on
CPU threads, BLAS versions, and GPU kernel selection all move the last digits.
``determinism_report`` records what was pinned and what was not, and that block
goes into the experiment artifact rather than an unqualified "reproducible"
claim.
"""

from __future__ import annotations

import os
import platform
import random
import sys
from typing import Any

import numpy as np

__all__ = ["set_global_seeds", "derive_seed", "determinism_report", "environment_report"]


def derive_seed(base_seed: int, *parts: int | str) -> int:
    """Derive a stable child seed from a base seed and some identifying parts.

    Uses a plain mixing function rather than ``hash()``, which is salted per
    process for ``str`` and would make runs irreproducible across invocations.
    """
    value = int(base_seed) & 0xFFFFFFFF
    for part in parts:
        if isinstance(part, str):
            for ch in part:
                value = (value * 1099511628211 + ord(ch)) & 0xFFFFFFFF
        else:
            value = (value * 1099511628211 + int(part)) & 0xFFFFFFFF
    return int(value)


def set_global_seeds(seed: int, *, torch_deterministic: bool = False) -> dict[str, Any]:
    """Seed ``random``, ``numpy`` and (when importable) ``torch``.

    Parameters
    ----------
    torch_deterministic:
        Additionally ask torch for deterministic algorithms.  Off by default
        because some operators have no deterministic implementation and would
        raise; turn it on when a run must be bit-reproducible and you are
        willing to trade throughput and coverage for it.

    Returns
    -------
    A report of what was actually seeded.
    """
    report: dict[str, Any] = {"seed": int(seed), "python_random": True, "numpy": True}
    random.seed(seed)
    np.random.seed(seed % (2**32))
    os.environ["PYTHONHASHSEED"] = str(seed)

    try:
        import torch
    except ImportError:
        report["torch"] = False
        return report

    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        report["cuda_seeded"] = True
    report["torch"] = True

    if torch_deterministic:
        try:
            torch.use_deterministic_algorithms(True)
            report["torch_deterministic_algorithms"] = True
        except Exception as exc:  # pragma: no cover - depends on op coverage
            report["torch_deterministic_algorithms"] = False
            report["torch_deterministic_error"] = str(exc)
    else:
        report["torch_deterministic_algorithms"] = False
    return report


def determinism_report() -> dict[str, Any]:
    """What reproducibility can and cannot be claimed for this process."""
    report: dict[str, Any] = {
        "seeded_sources": ["python_random", "numpy", "torch (if installed)"],
        "not_controlled": [
            "floating-point reduction order across CPU threads",
            "BLAS / MKL version differences",
            "GPU kernel selection when CUDA is used",
        ],
    }
    try:
        import torch

        report["torch_version"] = torch.__version__
        report["torch_num_threads"] = int(torch.get_num_threads())
        report["cuda_available"] = bool(torch.cuda.is_available())
    except ImportError:
        report["torch_version"] = None
    return report


def environment_report() -> dict[str, Any]:
    """Versions and platform, recorded with every experiment artifact."""
    versions: dict[str, Any] = {}
    for module_name in ("numpy", "pandas", "networkx", "sklearn", "torch", "torch_geometric",
                        "xgboost", "lightgbm"):
        try:
            module = __import__(module_name)
            versions[module_name] = getattr(module, "__version__", "unknown")
        except ImportError:
            versions[module_name] = None
    return {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "processor": platform.processor(),
        "packages": versions,
    }
