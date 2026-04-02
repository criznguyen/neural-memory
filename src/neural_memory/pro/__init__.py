"""Neural Memory Pro — Advanced features bundled in the main package.

Pro features require optional dependencies: numpy, hnswlib, msgpack.
Install with: pip install neural-memory[pro]

All features are gated behind config.is_pro() — license key required.
"""

from __future__ import annotations

PRO_VERSION = "0.3.0"


def _check_deps() -> tuple[bool, list[str]]:
    """Check if Pro optional dependencies are available."""
    missing: list[str] = []
    try:
        import numpy as _np  # noqa: F401
    except ImportError:
        missing.append("numpy")
    try:
        import hnswlib as _hnswlib  # noqa: F401
    except ImportError:
        missing.append("hnswlib")
    try:
        import msgpack as _msgpack  # noqa: F401
    except ImportError:
        missing.append("msgpack")
    return len(missing) == 0, missing


def is_pro_deps_installed() -> bool:
    """Return True if all Pro optional dependencies are installed."""
    available, _ = _check_deps()
    return available


def get_missing_deps() -> list[str]:
    """Return list of missing Pro dependencies."""
    _, missing = _check_deps()
    return missing


PRO_INSTALL_HINT = 'pip install "neural-memory[pro]"'

__all__ = [
    "PRO_VERSION",
    "is_pro_deps_installed",
    "get_missing_deps",
    "PRO_INSTALL_HINT",
]
