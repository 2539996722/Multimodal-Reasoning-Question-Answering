import asyncio
import base64
from typing import Any

import httpx
import pytest

from app.config import ModelEndpointConfig
from app.errors import AppError
from app.services import openai_client as client_module
from app.services.openai_client import OpenAICompatibleClient


class _FakeResponse:
    status_code = 200
    text = ""

    @staticmethod
    def json() -> dict[str, Any]:
        return {
            "model": "MiniMax-M3",
            "choices": [{"message": {"content": "OK"}}],
            "usage": {"total_tokens": 5},
        }


class _CapturingAsyncClient:
    calls: list[dict[str, Any]] = []

    def __init__(self, timeout: float) -> None:
        self.timeout = timeout

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, traceback) -> None:
        del exc_type, exc, traceback

    async def post(self, url: str, *, headers: dict[str, str], json: dict[str, Any]):
        self.calls.append({"url": url, "headers": headers, "json": json, "timeout": self.timeout})
        return _FakeResponse()


def _config() -> ModelEndpointConfig:
    return ModelEndpointConfig(
        base_url="https://api.minimax.chat/v1",
        api_key="test-secret",
        model="MiniMax-M3",
        timeout=30,
        temperature=0,
    )


def test_multimodal_request_matches_minimax_image_url_detail_contract(monkeypatch) -> None:
    _CapturingAsyncClient.calls = []
    monkeypatch.setattr(client_module.httpx, "AsyncClient", _CapturingAsyncClient)
    image = b"png-binary-content"

    result = asyncio.run(
        OpenAICompatibleClient(_config(), "vlm").chat(
            "识别目标页",
            "系统提示",
            image_png=image,
            max_tokens=256,
            json_mode=True,
        )
    )

    assert result["content"] == "OK"
    assert len(_CapturingAsyncClient.calls) == 1
    request = _CapturingAsyncClient.calls[0]
    assert request["url"] == "https://api.minimax.chat/v1/chat/completions"
    assert request["headers"]["Authorization"] == "Bearer test-secret"
    body = request["json"]
    assert body["response_format"] == {"type": "json_object"}
    assert body["messages"][0] == {"role": "system", "content": "系统提示"}
    assert body["messages"][1]["content"] == [
        {"type": "text", "text": "识别目标页"},
        {
            "type": "image_url",
            "image_url": {
                "url": f"data:image/png;base64,{base64.b64encode(image).decode('ascii')}",
            },
        },
    ]


def test_image_detail_hint_is_only_sent_when_requested(monkeypatch) -> None:
    _CapturingAsyncClient.calls = []
    monkeypatch.setattr(client_module.httpx, "AsyncClient", _CapturingAsyncClient)
    monkeypatch.delenv("MODEL_IMAGE_DETAIL", raising=False)

    asyncio.run(
        OpenAICompatibleClient(_config(), "vlm").chat(
            "识别目标页", "系统提示", image_png=b"png", max_tokens=256, image_detail="high"
        )
    )

    image_url = _CapturingAsyncClient.calls[0]["json"]["messages"][1]["content"][1]["image_url"]
    assert image_url["detail"] == "high"


def test_image_detail_hint_is_absent_by_default(monkeypatch) -> None:
    _CapturingAsyncClient.calls = []
    monkeypatch.setattr(client_module.httpx, "AsyncClient", _CapturingAsyncClient)
    monkeypatch.delenv("MODEL_IMAGE_DETAIL", raising=False)

    asyncio.run(
        OpenAICompatibleClient(_config(), "vlm").chat(
            "识别目标页", "系统提示", image_png=b"png", max_tokens=256
        )
    )

    image_url = _CapturingAsyncClient.calls[0]["json"]["messages"][1]["content"][1]["image_url"]
    assert "detail" not in image_url


def test_text_only_request_does_not_add_image_detail(monkeypatch) -> None:
    _CapturingAsyncClient.calls = []
    monkeypatch.setattr(client_module.httpx, "AsyncClient", _CapturingAsyncClient)

    asyncio.run(OpenAICompatibleClient(_config(), "ocr").chat("ping", "system"))

    user_content = _CapturingAsyncClient.calls[0]["json"]["messages"][1]["content"]
    assert user_content == [{"type": "text", "text": "ping"}]
    assert "image_url" not in str(user_content)


def test_minimax_ocr_requests_reasoning_split(monkeypatch) -> None:
    _CapturingAsyncClient.calls = []
    monkeypatch.setattr(client_module.httpx, "AsyncClient", _CapturingAsyncClient)

    asyncio.run(OpenAICompatibleClient(_config(), "ocr").chat("识别", "system"))

    assert _CapturingAsyncClient.calls[0]["json"]["reasoning_split"] is True


def test_minimax_vlm_requests_reasoning_split(monkeypatch) -> None:
    _CapturingAsyncClient.calls = []
    monkeypatch.setattr(client_module.httpx, "AsyncClient", _CapturingAsyncClient)

    asyncio.run(OpenAICompatibleClient(_config(), "vlm").chat("识别", "system"))

    assert _CapturingAsyncClient.calls[0]["json"]["reasoning_split"] is True


def test_connection_gives_reasoning_model_enough_output_budget(monkeypatch) -> None:
    _CapturingAsyncClient.calls = []
    monkeypatch.setattr(client_module.httpx, "AsyncClient", _CapturingAsyncClient)

    result = asyncio.run(OpenAICompatibleClient(_config(), "vlm").test_connection())

    assert result["status"] == "connected"
    request = _CapturingAsyncClient.calls[0]["json"]
    assert request["max_tokens"] == 256
    assert request["reasoning_split"] is True


class _RejectedResponse:
    def __init__(self, status_code: int, text: str) -> None:
        self.status_code = status_code
        self.text = text

    def json(self) -> dict[str, Any]:
        return {}


class _RejectingAsyncClient:
    status_code = 402
    text = '{"type":"error","error":{"type":"insufficient_balance_error"}}'

    def __init__(self, timeout: float) -> None:
        self.timeout = timeout

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, traceback) -> None:
        del exc_type, exc, traceback

    async def post(self, url: str, *, headers: dict[str, str], json: dict[str, Any]):
        del url, headers, json
        return _RejectedResponse(402, self.text)


def test_insufficient_balance_is_reported_as_its_own_error_code(monkeypatch) -> None:
    monkeypatch.setattr(client_module.httpx, "AsyncClient", _RejectingAsyncClient)

    with pytest.raises(AppError) as excinfo:
        asyncio.run(OpenAICompatibleClient(_config(), "vlm").chat("ping", "system"))

    assert excinfo.value.code == "MODEL_BALANCE_EXHAUSTED"
    assert excinfo.value.retryable is False
    assert "余额" in excinfo.value.message
