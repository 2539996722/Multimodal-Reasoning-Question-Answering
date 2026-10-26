from __future__ import annotations

import base64
import os
import re
import time
from typing import Any

import httpx

from app.config import ModelEndpointConfig
from app.errors import AppError


def chat_completions_url(base_url: str) -> str:
    value = base_url.strip().rstrip("/")
    if not value:
        return ""
    if value.endswith("/chat/completions"):
        return value
    if value.endswith("/v1"):
        return f"{value}/chat/completions"
    return f"{value}/v1/chat/completions"


def extract_message_content(payload: dict[str, Any]) -> str:
    try:
        content = payload["choices"][0]["message"]["content"]
    except (KeyError, IndexError, TypeError) as exc:
        raise AppError("MODEL_RESPONSE_INVALID", "模型响应缺少消息内容", 502, True) from exc
    if isinstance(content, str):
        text = content.strip()
    elif isinstance(content, list):
        values: list[str] = []
        for part in content:
            if isinstance(part, dict) and isinstance(part.get("text"), str):
                values.append(part["text"])
        text = "\n".join(values).strip()
    else:
        text = ""
    # Some reasoning-model compatible endpoints put private reasoning in XML-like
    # blocks inside `content`. It must never be displayed or fed into the answer
    # parser, otherwise letters from the reasoning/template can be mistaken for
    # a multiple-choice answer.
    text = re.sub(r"<(?:think|analysis)\b[^>]*>[\s\S]*?</(?:think|analysis)>", "", text, flags=re.IGNORECASE).strip()
    if not text:
        raise AppError("MODEL_EMPTY_RESPONSE", "模型返回了空内容", 502, True)
    return text


def _configured_image_detail() -> str | None:
    """Optional OpenAI image detail hint, enabled only through MODEL_IMAGE_DETAIL."""
    value = os.getenv("MODEL_IMAGE_DETAIL", "").strip()
    return value or None


class OpenAICompatibleClient:
    def __init__(self, config: ModelEndpointConfig, service: str) -> None:
        self.config = config
        self.service = service

    async def chat(
        self,
        prompt: str,
        system_prompt: str,
        image_png: bytes | None = None,
        max_tokens: int = 1600,
        json_mode: bool = False,
        image_detail: str | None = None,
    ) -> dict[str, Any]:
        if not self.config.configured:
            raise AppError(
                "MODEL_NOT_CONFIGURED",
                f"{self.service.upper()} 模型配置不完整",
                409,
                False,
            )
        content: list[dict[str, Any]] = [{"type": "text", "text": prompt}]
        if image_png is not None:
            encoded = base64.b64encode(image_png).decode("ascii")
            # The optional OpenAI `detail` hint is rejected by several
            # OpenAI-compatible gateways, so it is only sent when the
            # operator explicitly asks for it.
            image_url: dict[str, Any] = {"url": f"data:image/png;base64,{encoded}"}
            image_detail = image_detail if image_detail is not None else _configured_image_detail()
            if image_detail:
                image_url["detail"] = image_detail
            content.append({"type": "image_url", "image_url": image_url})
        body = {
            "model": self.config.model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": content},
            ],
            "temperature": self.config.temperature,
            "max_tokens": max_tokens,
        }
        if json_mode:
            body["response_format"] = {"type": "json_object"}
        # MiniMax reasoning models support the OpenAI-compatible
        # `reasoning_split` extension.  Keep private reasoning in
        # reasoning_details so message.content contains only the final answer.
        if self.config.model.casefold().startswith("minimax"):
            body["reasoning_split"] = True
        started = time.perf_counter()
        try:
            async with httpx.AsyncClient(timeout=self.config.timeout) as client:
                response = await client.post(
                    chat_completions_url(self.config.base_url),
                    headers={"Authorization": f"Bearer {self.config.api_key}", "Content-Type": "application/json"},
                    json=body,
                )
                if response.status_code == 400 and json_mode:
                    # Not every OpenAI-compatible gateway implements
                    # response_format. Retry once without it while preserving the
                    # JSON-only prompt contract.
                    error_text = response.text.lower()
                    if "response_format" in error_text or "json_object" in error_text:
                        body.pop("response_format", None)
                        response = await client.post(
                            chat_completions_url(self.config.base_url),
                            headers={"Authorization": f"Bearer {self.config.api_key}", "Content-Type": "application/json"},
                            json=body,
                        )
                    elif "reasoning_split" in error_text:
                        body.pop("reasoning_split", None)
                        response = await client.post(
                            chat_completions_url(self.config.base_url),
                            headers={"Authorization": f"Bearer {self.config.api_key}", "Content-Type": "application/json"},
                            json=body,
                        )
        except httpx.TimeoutException as exc:
            raise AppError("MODEL_TIMEOUT", f"{self.service.upper()} 模型请求超时", 504, True) from exc
        except httpx.RequestError as exc:
            raise AppError("MODEL_UNREACHABLE", f"无法连接 {self.service.upper()} 模型服务", 502, True) from exc

        if response.status_code in (401, 403):
            raise AppError("MODEL_AUTH_FAILED", f"{self.service.upper()} 模型鉴权失败", 502, False)
        if response.status_code == 402:
            raise AppError("MODEL_BALANCE_EXHAUSTED", f"{self.service.upper()} 上游账户余额不足或配额已耗尽", 502, False)
        if response.status_code == 429:
            raise AppError("MODEL_RATE_LIMITED", f"{self.service.upper()} 模型服务限流", 503, True)
        if response.status_code >= 500:
            raise AppError("MODEL_UPSTREAM_ERROR", f"{self.service.upper()} 模型服务异常", 502, True)
        if response.status_code >= 400:
            raise AppError("MODEL_REQUEST_REJECTED", f"{self.service.upper()} 模型拒绝了请求", 502, False)
        try:
            payload = response.json()
        except ValueError as exc:
            raise AppError("MODEL_RESPONSE_INVALID", "模型返回了非 JSON 响应", 502, True) from exc
        return {
            "content": extract_message_content(payload),
            "model": str(payload.get("model") or self.config.model),
            "usage": payload.get("usage"),
            "duration_ms": round((time.perf_counter() - started) * 1000, 1),
        }

    async def test_connection(self) -> dict[str, Any]:
        result = await self.chat(
            "Reply with only OK.",
            "你正在执行模型服务连通性测试，不要输出敏感配置。",
            max_tokens=256,
        )
        return {
            "service": self.service,
            "status": "connected",
            "model": result["model"],
            "duration_ms": result["duration_ms"],
        }
