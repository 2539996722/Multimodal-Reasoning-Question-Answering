from __future__ import annotations

import json
import os
from copy import deepcopy
from dataclasses import asdict, dataclass, field
from pathlib import Path
from threading import RLock
from typing import Any, Literal


ROOT_DIR = Path(__file__).resolve().parent.parent
MODEL_SETTINGS_PATH = ROOT_DIR / "runtime" / "model-settings.json"


def _env_bool(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _env_float(name: str, default: float) -> float:
    try:
        return float(os.getenv(name, default))
    except (TypeError, ValueError):
        return default


@dataclass(slots=True)
class ModelEndpointConfig:
    base_url: str = ""
    api_key: str = ""
    model: str = ""
    timeout: float = 120.0
    temperature: float = 0.0

    @property
    def configured(self) -> bool:
        return bool(self.base_url.strip() and self.api_key.strip() and self.model.strip())

    def public_dict(self) -> dict[str, Any]:
        return {
            "base_url": self.base_url,
            "model": self.model,
            "timeout": self.timeout,
            "temperature": self.temperature,
            "api_key_set": bool(self.api_key),
            "configured": self.configured,
        }


@dataclass(slots=True)
class AppSettings:
    demo_mode: bool = True
    ocr: ModelEndpointConfig = field(default_factory=ModelEndpointConfig)
    vlm: ModelEndpointConfig = field(default_factory=ModelEndpointConfig)
    retrieval: ModelEndpointConfig = field(default_factory=ModelEndpointConfig)
    reranker: ModelEndpointConfig = field(default_factory=ModelEndpointConfig)
    max_batch_size: int = 500
    max_concurrency: int = 4
    version: str = "1.0.0"


class SettingsStore:
    """Thread-safe settings with optional local persistence; public responses never include API keys."""

    def __init__(self, persist_path: Path | None = None) -> None:
        common_timeout = _env_float("MODEL_TIMEOUT_SECONDS", 120.0)
        self._settings = AppSettings(
            demo_mode=True,
            ocr=ModelEndpointConfig(
                timeout=common_timeout,
            ),
            vlm=ModelEndpointConfig(
                timeout=common_timeout,
            ),
            retrieval=ModelEndpointConfig(timeout=common_timeout),
            reranker=ModelEndpointConfig(timeout=common_timeout),
        )
        self._lock = RLock()
        self.persist_path = persist_path
        self._load_persisted()
        self._apply_environment_overrides(common_timeout)

    @property
    def settings(self) -> AppSettings:
        return self._settings

    def public(self) -> dict[str, Any]:
        with self._lock:
            return {
                "demo_mode": self._settings.demo_mode,
                "effective_mode": "demo" if self._settings.demo_mode else "real",
                "ocr": self._settings.ocr.public_dict(),
                "vlm": self._settings.vlm.public_dict(),
                "retrieval": self._settings.retrieval.public_dict(),
                "reranker": self._settings.reranker.public_dict(),
                "persistence": {
                    "enabled": self.persist_path is not None,
                    "file": "runtime/model-settings.json" if self.persist_path is not None else None,
                },
            }

    def update(self, data: dict[str, Any]) -> dict[str, Any]:
        with self._lock:
            proposed = deepcopy(self._settings)
            self._apply_update(proposed, data)
            self._persist(proposed)
            self._copy_into(self._settings, proposed)
            return self.public()

    def endpoint(self, kind: Literal["ocr", "vlm", "retrieval", "reranker"]) -> ModelEndpointConfig:
        return getattr(self._settings, kind)

    @staticmethod
    def _apply_update(settings: AppSettings, data: dict[str, Any]) -> None:
        if "demo_mode" in data and data["demo_mode"] is not None:
            settings.demo_mode = bool(data["demo_mode"])
        for kind in ("ocr", "vlm", "retrieval", "reranker"):
            incoming = data.get(kind)
            if not isinstance(incoming, dict):
                continue
            endpoint = getattr(settings, kind)
            for key in ("base_url", "model", "timeout", "temperature"):
                value = incoming.get(key)
                if value is not None:
                    setattr(endpoint, key, value)
            api_key = incoming.get("api_key")
            if api_key is not None and api_key != "":
                endpoint.api_key = api_key
            if incoming.get("clear_api_key"):
                endpoint.api_key = ""

    @staticmethod
    def _copy_into(target: AppSettings, source: AppSettings) -> None:
        target.demo_mode = source.demo_mode
        for kind in ("ocr", "vlm", "retrieval", "reranker"):
            target_endpoint = getattr(target, kind)
            source_endpoint = getattr(source, kind)
            for key in ("base_url", "api_key", "model", "timeout", "temperature"):
                setattr(target_endpoint, key, getattr(source_endpoint, key))

    def _load_persisted(self) -> None:
        if self.persist_path is None or not self.persist_path.is_file():
            return
        try:
            payload = json.loads(self.persist_path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return
        if not isinstance(payload, dict):
            return
        sanitized: dict[str, Any] = {}
        if isinstance(payload.get("demo_mode"), bool):
            sanitized["demo_mode"] = payload["demo_mode"]
        for kind in ("ocr", "vlm", "retrieval", "reranker"):
            incoming = payload.get(kind)
            if not isinstance(incoming, dict):
                continue
            endpoint: dict[str, Any] = {}
            for key in ("base_url", "api_key", "model"):
                if isinstance(incoming.get(key), str):
                    endpoint[key] = incoming[key]
            for key in ("timeout", "temperature"):
                if isinstance(incoming.get(key), (int, float)):
                    endpoint[key] = float(incoming[key])
            sanitized[kind] = endpoint
        self._apply_update(self._settings, sanitized)

    def _apply_environment_overrides(self, common_timeout: float) -> None:
        data: dict[str, Any] = {}
        if os.getenv("DEMO_MODE") is not None:
            data["demo_mode"] = _env_bool("DEMO_MODE", self._settings.demo_mode)
        for kind, prefix in (
            ("ocr", "OCR"),
            ("vlm", "VLM"),
            ("retrieval", "RETRIEVAL"),
            ("reranker", "RERANKER"),
        ):
            endpoint: dict[str, Any] = {}
            for key, suffix in (("base_url", "BASE_URL"), ("api_key", "API_KEY"), ("model", "MODEL")):
                value = os.getenv(f"{prefix}_{suffix}")
                if value is not None:
                    endpoint[key] = value
            if os.getenv(f"{prefix}_TIMEOUT_SECONDS") is not None:
                endpoint["timeout"] = _env_float(f"{prefix}_TIMEOUT_SECONDS", common_timeout)
            if os.getenv(f"{prefix}_TEMPERATURE") is not None:
                endpoint["temperature"] = _env_float(f"{prefix}_TEMPERATURE", 0.0)
            if endpoint:
                data[kind] = endpoint
        self._apply_update(self._settings, data)

    def _persist(self, settings: AppSettings) -> None:
        if self.persist_path is None:
            return
        payload = {
            "version": 1,
            "demo_mode": settings.demo_mode,
            "ocr": asdict(settings.ocr),
            "vlm": asdict(settings.vlm),
            "retrieval": asdict(settings.retrieval),
            "reranker": asdict(settings.reranker),
        }
        self.persist_path.parent.mkdir(parents=True, exist_ok=True)
        temporary = self.persist_path.with_name(f"{self.persist_path.name}.tmp")
        try:
            temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
            os.replace(temporary, self.persist_path)
            try:
                os.chmod(self.persist_path, 0o600)
            except OSError:
                pass
        finally:
            if temporary.exists():
                temporary.unlink(missing_ok=True)


settings_store = SettingsStore(MODEL_SETTINGS_PATH)
