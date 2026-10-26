import json

import pytest

from app.config import SettingsStore


def test_model_settings_never_serialize_api_keys(monkeypatch) -> None:
    secret = "sk-contract-test-never-expose"
    monkeypatch.setenv("OCR_BASE_URL", "https://ocr.example/v1")
    monkeypatch.setenv("OCR_MODEL", "ocr-test")
    monkeypatch.setenv("OCR_API_KEY", secret)
    monkeypatch.setenv("VLM_API_KEY", secret)
    monkeypatch.setenv("RETRIEVAL_API_KEY", secret)
    monkeypatch.setenv("RERANKER_API_KEY", secret)

    store = SettingsStore()
    public = store.public()

    assert secret not in repr(public)
    assert public["ocr"]["api_key_set"] is True
    assert set(public) >= {"ocr", "vlm", "retrieval", "reranker"}
    for kind in ("ocr", "vlm", "retrieval", "reranker"):
        assert public[kind]["api_key_set"] is True
        assert "api_key" not in public[kind]


def test_empty_api_key_update_preserves_secret_and_clear_is_explicit(monkeypatch) -> None:
    for prefix in ("OCR", "VLM", "RETRIEVAL", "RERANKER"):
        monkeypatch.setenv(f"{prefix}_API_KEY", f"{prefix.lower()}-existing-secret")
    store = SettingsStore()

    store.update({kind: {"api_key": ""} for kind in ("ocr", "vlm", "retrieval", "reranker")})
    for kind in ("ocr", "vlm", "retrieval", "reranker"):
        assert store.endpoint(kind).api_key == f"{kind}-existing-secret"

    store.update({kind: {"clear_api_key": True} for kind in ("ocr", "vlm", "retrieval", "reranker")})
    for kind in ("ocr", "vlm", "retrieval", "reranker"):
        assert store.endpoint(kind).api_key == ""


def test_model_settings_persist_and_reload_without_public_secret(tmp_path, monkeypatch) -> None:
    for name in (
        "DEMO_MODE",
        "OCR_BASE_URL",
        "OCR_API_KEY",
        "OCR_MODEL",
        "VLM_BASE_URL",
        "VLM_API_KEY",
        "VLM_MODEL",
        "RETRIEVAL_BASE_URL",
        "RETRIEVAL_API_KEY",
        "RETRIEVAL_MODEL",
        "RERANKER_BASE_URL",
        "RERANKER_API_KEY",
        "RERANKER_MODEL",
    ):
        monkeypatch.delenv(name, raising=False)
    path = tmp_path / "runtime" / "model-settings.json"
    secret = "sk-persisted-secret"
    store = SettingsStore(path)

    public = store.update(
        {
            "demo_mode": False,
            "ocr": {
                "base_url": "https://ocr.example/v1",
                "api_key": secret,
                "model": "ocr-model",
                "timeout": 90,
            },
            "vlm": {
                "base_url": "https://vlm.example/v1",
                "api_key": secret,
                "model": "vlm-model",
                "temperature": 0.2,
            },
            "retrieval": {
                "base_url": "https://retrieval.example/v1",
                "api_key": secret,
                "model": "embedding-model",
                "timeout": 45,
            },
            "reranker": {
                "base_url": "https://reranker.example/v1",
                "api_key": secret,
                "model": "reranker-model",
                "timeout": 60,
            },
        }
    )

    assert path.is_file()
    assert secret not in repr(public)
    assert public["persistence"]["enabled"] is True
    reloaded = SettingsStore(path)
    assert reloaded.settings.demo_mode is False
    assert reloaded.endpoint("ocr").base_url == "https://ocr.example/v1"
    assert reloaded.endpoint("ocr").api_key == secret
    assert reloaded.endpoint("ocr").model == "ocr-model"
    assert reloaded.endpoint("vlm").api_key == secret
    assert reloaded.endpoint("retrieval").model == "embedding-model"
    assert reloaded.endpoint("reranker").base_url == "https://reranker.example/v1"
    assert all(reloaded.public()[kind]["api_key_set"] for kind in ("ocr", "vlm", "retrieval", "reranker"))
    assert secret not in repr(reloaded.public())


def test_clear_api_key_is_persisted(tmp_path, monkeypatch) -> None:
    monkeypatch.delenv("OCR_API_KEY", raising=False)
    path = tmp_path / "model-settings.json"
    store = SettingsStore(path)
    store.update({"ocr": {"base_url": "https://ocr.example/v1", "model": "ocr", "api_key": "secret"}})
    store.update({"ocr": {"clear_api_key": True}})

    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["ocr"]["api_key"] == ""
    assert SettingsStore(path).endpoint("ocr").api_key == ""


def test_environment_values_override_persisted_file(tmp_path, monkeypatch) -> None:
    path = tmp_path / "model-settings.json"
    path.write_text(
        json.dumps(
            {
                "version": 1,
                "demo_mode": True,
                "vlm": {"base_url": "https://file.example/v1", "api_key": "file-secret", "model": "file-model"},
                "retrieval": {"base_url": "https://file-retrieval.example/v1", "api_key": "file-secret", "model": "file-embedding"},
                "reranker": {"base_url": "https://file-reranker.example/v1", "api_key": "file-secret", "model": "file-reranker"},
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setenv("DEMO_MODE", "false")
    monkeypatch.setenv("VLM_BASE_URL", "https://env.example/v1")
    monkeypatch.setenv("VLM_API_KEY", "env-secret")
    monkeypatch.setenv("VLM_MODEL", "env-model")
    monkeypatch.setenv("RETRIEVAL_BASE_URL", "https://env-retrieval.example/v1")
    monkeypatch.setenv("RETRIEVAL_API_KEY", "env-retrieval-secret")
    monkeypatch.setenv("RETRIEVAL_MODEL", "env-embedding")
    monkeypatch.setenv("RERANKER_BASE_URL", "https://env-reranker.example/v1")
    monkeypatch.setenv("RERANKER_API_KEY", "env-reranker-secret")
    monkeypatch.setenv("RERANKER_MODEL", "env-reranker")

    store = SettingsStore(path)

    assert store.settings.demo_mode is False
    assert store.endpoint("vlm").base_url == "https://env.example/v1"
    assert store.endpoint("vlm").api_key == "env-secret"
    assert store.endpoint("vlm").model == "env-model"
    assert store.endpoint("retrieval").base_url == "https://env-retrieval.example/v1"
    assert store.endpoint("retrieval").api_key == "env-retrieval-secret"
    assert store.endpoint("retrieval").model == "env-embedding"
    assert store.endpoint("reranker").base_url == "https://env-reranker.example/v1"
    assert store.endpoint("reranker").api_key == "env-reranker-secret"
    assert store.endpoint("reranker").model == "env-reranker"


def test_failed_atomic_write_does_not_mutate_live_settings(tmp_path, monkeypatch) -> None:
    path = tmp_path / "model-settings.json"
    store = SettingsStore(path)
    store.update({"ocr": {"base_url": "https://before.example/v1", "model": "before", "api_key": "secret"}})

    def fail_replace(source, destination):
        del source, destination
        raise OSError("disk unavailable")

    monkeypatch.setattr("app.config.os.replace", fail_replace)
    with pytest.raises(OSError, match="disk unavailable"):
        store.update({"ocr": {"base_url": "https://after.example/v1"}})

    assert store.endpoint("ocr").base_url == "https://before.example/v1"
