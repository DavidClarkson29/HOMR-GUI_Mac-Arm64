from __future__ import annotations

import atexit
import os
import sys
import tempfile
from pathlib import Path

APP_NAME = "HOMR"
_SESSION: tempfile.TemporaryDirectory[str] | None = None


def _configured_path(environment_variable: str) -> Path | None:
    value = os.environ.get(environment_variable)
    if not value:
        return None
    return Path(value).expanduser().resolve()


def project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def is_frozen_app() -> bool:
    return getattr(sys, "_MEIPASS", None) is not None


def session_dir() -> Path:
    """A disposable per-process workspace for screenshots and generated files."""
    global _SESSION  # noqa: PLW0603
    if _SESSION is None:
        _SESSION = tempfile.TemporaryDirectory(prefix="HOMR-")
    return Path(_SESSION.name)


def cleanup_session() -> None:
    global _SESSION  # noqa: PLW0603
    if _SESSION is not None:
        _SESSION.cleanup()
        _SESSION = None


atexit.register(cleanup_session)


def data_dir() -> Path:
    configured = _configured_path("HOMR_DATA_DIR")
    if configured is not None:
        return configured
    if is_frozen_app():
        return session_dir() / "data"
    return project_root() / "work" / "runtime" / "data"


def cache_dir() -> Path:
    configured = _configured_path("HOMR_CACHE_DIR")
    if configured is not None:
        return configured
    return session_dir() / "cache"


def bundled_resource_path(relative_path: str | Path) -> Path:
    """Resolve a resource in a PyInstaller bundle or the project checkout."""
    bundle_root = getattr(sys, "_MEIPASS", None)
    if bundle_root is not None:
        return Path(bundle_root) / relative_path
    return project_root() / relative_path


def model_dir() -> Path:
    configured = _configured_path("HOMR_MODEL_DIR")
    if configured is not None:
        return configured

    bundled_models = bundled_resource_path(Path("homr") / "models")
    if is_frozen_app() and bundled_models.is_dir():
        return bundled_models
    return project_root() / "work" / "runtime" / "models"


def ensure_runtime_directories() -> None:
    for directory in (data_dir(), cache_dir()):
        directory.mkdir(parents=True, exist_ok=True)
    models = model_dir()
    if not (is_frozen_app() and models.is_dir()):
        models.mkdir(parents=True, exist_ok=True)
