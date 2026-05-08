from __future__ import annotations

import os
from typing import Any

import httpx

from .model_ids import normalize_model_id

DEEPSEEK_MODEL_PREFIX = "deepseek-web/"
DEEPSEEK_DEFAULT_MODEL = "deepseek-v4-pro"
DS2API_DEFAULT_BASE_URL = "http://127.0.0.1:9331/v1"

DEEPSEEK_WEB_CATALOG_MODELS = (
    "deepseek-v4-pro",
)

DEEPSEEK_WEB_MODEL_ALIASES = {
    "deepseek-v4-pro[1m]": "deepseek-v4-pro",
    "deepseek-v4-pro": "deepseek-v4-pro",
    "deepseek-v4-flash": "deepseek-v4-pro",
    "deepseek-chat": "deepseek-v4-pro",
    "deepseek-reasoner": "deepseek-v4-pro",
}

DEEPSEEK_WEB_ACCEPTED_MODELS = {
    *DEEPSEEK_WEB_CATALOG_MODELS,
    *DEEPSEEK_WEB_MODEL_ALIASES,
}


class DeepSeekDs2apiError(RuntimeError):
    def __init__(self, status_code: int, message: str) -> None:
        super().__init__(message)
        self.status_code = status_code


def is_deepseek_web_model(model: Any) -> bool:
    normalized = normalize_model_id(model)
    return bool(
        normalized
        and (
            normalized.startswith(DEEPSEEK_MODEL_PREFIX)
            or normalized in DEEPSEEK_WEB_ACCEPTED_MODELS
        )
    )


def normalize_deepseek_model(model: str) -> str:
    model = normalize_model_id(model)
    short_model = model.removeprefix(DEEPSEEK_MODEL_PREFIX)
    return DEEPSEEK_WEB_MODEL_ALIASES.get(short_model, short_model or DEEPSEEK_DEFAULT_MODEL)


class DeepSeekWebClient:
    """OpenAI-compatible DeepSeek Web client backed by a local ds2api sidecar."""

    def __init__(
        self,
        credential: dict[str, Any],
        http_client: httpx.Client | None = None,
        *,
        ds2api_base_url: str | None = None,
        request_timeout_seconds: float = 300,
        prompt_max_chars: int | None = None,
    ) -> None:
        self.credential = credential
        self.ds2api_base_url = (
            ds2api_base_url
            or os.environ.get("WEBAI_DEEPSEEK_DS2API_BASE_URL")
            or DS2API_DEFAULT_BASE_URL
        ).rstrip("/")
        self.request_timeout_seconds = max(30.0, float(request_timeout_seconds or 300))
        self.prompt_max_chars = max(4000, int(prompt_max_chars or 32000))
        self.last_diagnostic: dict[str, Any] = {}
        self.http_client = http_client or httpx.Client(timeout=self.request_timeout_seconds, trust_env=False)

    def chat_completions(self, payload: dict[str, Any]) -> dict[str, Any]:
        token = str(self.credential.get("bearer") or "").strip()
        if not token:
            raise RuntimeError("DeepSeek Web 授权缺少 bearer token，请重新完成网页登录授权")

        ds2api_payload = self._payload_for_ds2api(payload)
        self.last_diagnostic = {
            "model": ds2api_payload.get("model"),
            "ds2api_base_url": self.ds2api_base_url,
            "message_count": len(ds2api_payload.get("messages")) if isinstance(ds2api_payload.get("messages"), list) else 0,
        }
        response = self.http_client.post(
            f"{self.ds2api_base_url}/chat/completions",
            json=ds2api_payload,
            headers={
                "Authorization": f"Bearer {token}",
                "Content-Type": "application/json",
                "Accept": "application/json",
            },
            timeout=self.request_timeout_seconds,
        )
        try:
            response.raise_for_status()
        except httpx.HTTPStatusError as exc:
            raise DeepSeekDs2apiError(response.status_code, _format_ds2api_error(response)) from exc

        content_type = (response.headers.get("content-type") or "").lower()
        if "text/event-stream" in content_type:
            return {
                "_webai_stream": True,
                "_webai_sse_body": response.text,
                "_webai_content_type": content_type,
            }

        data = response.json()
        if not isinstance(data, dict):
            raise RuntimeError("ds2api 返回内容不是 JSON 对象")
        return data

    def _payload_for_ds2api(self, payload: dict[str, Any]) -> dict[str, Any]:
        ds2api_payload = {
            key: value
            for key, value in payload.items()
            if not key.startswith("_webai_")
        }
        ds2api_payload["model"] = normalize_deepseek_model(str(payload.get("model") or DEEPSEEK_DEFAULT_MODEL))
        return ds2api_payload


def _format_ds2api_error(response: httpx.Response) -> str:
    text = response.text.strip()
    if len(text) > 800:
        text = text[:800] + "..."
    detail = text or response.reason_phrase
    return f"ds2api DeepSeek 调用失败：HTTP {response.status_code} {detail}"
