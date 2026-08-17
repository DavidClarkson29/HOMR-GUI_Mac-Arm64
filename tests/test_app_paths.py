# ruff: noqa: S101

from pathlib import Path

from homr import app_paths


def test_runtime_path_overrides(monkeypatch, tmp_path: Path) -> None:
    data = tmp_path / "data"
    cache = tmp_path / "cache"
    models = tmp_path / "models"
    monkeypatch.setenv("HOMR_DATA_DIR", str(data))
    monkeypatch.setenv("HOMR_CACHE_DIR", str(cache))
    monkeypatch.setenv("HOMR_MODEL_DIR", str(models))

    app_paths.ensure_runtime_directories()

    assert app_paths.data_dir() == data
    assert app_paths.cache_dir() == cache
    assert app_paths.model_dir() == models
    assert data.is_dir()
    assert cache.is_dir()
    assert models.is_dir()


def test_models_default_to_project_runtime(monkeypatch) -> None:
    monkeypatch.delenv("HOMR_MODEL_DIR", raising=False)
    monkeypatch.delattr(app_paths.sys, "_MEIPASS", raising=False)

    assert app_paths.model_dir() == app_paths.project_root() / "work" / "runtime" / "models"


def test_cache_defaults_to_disposable_session(monkeypatch) -> None:
    monkeypatch.delenv("HOMR_CACHE_DIR", raising=False)

    cache = app_paths.cache_dir()

    assert cache.parent == app_paths.session_dir()
    assert "HOMR-" in str(cache)


def test_frozen_app_prefers_bundled_models(monkeypatch, tmp_path: Path) -> None:
    bundle = tmp_path / "bundle"
    bundled_models = bundle / "homr" / "models"
    bundled_models.mkdir(parents=True)
    monkeypatch.setattr(app_paths.sys, "_MEIPASS", str(bundle), raising=False)
    monkeypatch.delenv("HOMR_MODEL_DIR", raising=False)

    assert app_paths.model_dir() == bundled_models
