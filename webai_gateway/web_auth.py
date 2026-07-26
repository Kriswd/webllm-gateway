from __future__ import annotations

import asyncio
import json
import os
import socket
import subprocess
import threading
import time
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable
from urllib.error import HTTPError, URLError
from urllib.parse import parse_qsl, quote, unquote, urlparse
from urllib.request import urlopen

from webai_gateway.deepseek_web import DEEPSEEK_WEB_CATALOG_MODELS


DEFAULT_CDP_URL = "http://127.0.0.1:9222"
_LOOPBACK_CDP_HOSTS = frozenset({"127.0.0.1", "localhost", "::1"})
# A first browser launch on Windows can take noticeably longer than a warm
# launch, especially while the browser hands off an existing instance.  Keep
# the UI synchronous long enough to distinguish that normal handoff from a
# genuine CDP failure.
_CDP_READY_TIMEOUT_SECONDS = 15.0
_CDP_READY_POLL_SECONDS = 0.15
DEEPSEEK_WEB_AVAILABLE_REASON = (
    "DeepSeek Web 现通过本地 ds2api sidecar 接入。完成网页登录授权并检测通过后，可使用 `deepseek-v4-pro`；"
    "如果检测失败，请按页面提示检查本地 ds2api、网页账号状态或重新授权。"
)
QWEN_LOGIN_COOKIE_NAMES = frozenset({"token", "qwen_session", "sessiontoken", "session_token", "qwensession"})
QWEN_DIRECT_PROVIDER_IDS = {"qwen", "qwen-coder"}
QWEN_CREDENTIAL_ORIGINS: dict[str, tuple[str, ...]] = {
    "qwen": ("https://chat.qwen.ai", "https://qwen.ai"),
    "qwen-coder": ("https://coder.qwen.ai", "https://qwen.ai"),
}
_EMPTY_CREDENTIAL_VALUES = {"null", "undefined", "none", "nil"}


def _normalized_credential_secret(value: Any) -> str:
    if not isinstance(value, str):
        return ""
    normalized = value.strip()
    if not normalized or normalized.lower() in _EMPTY_CREDENTIAL_VALUES:
        return ""
    return normalized


def _normalized_bearer(value: Any) -> str:
    token = _normalized_credential_secret(value)
    if not token:
        return ""
    parts = token.split(None, 1)
    if parts[0].lower() != "bearer":
        return token
    if len(parts) == 1:
        return ""
    return _normalized_credential_secret(parts[1])


def _normalized_session_token(value: Any) -> str:
    token = _normalized_credential_secret(value)
    if not token:
        return ""
    if token.lower() == "bearer" or token.lower().startswith("bearer "):
        return _normalized_bearer(token)
    return token


def _normalized_cookie_header(value: Any) -> str:
    return _normalized_credential_secret(value)


def _cookie_header_has_usable_value(value: Any) -> bool:
    cookie = _normalized_cookie_header(value)
    if not cookie:
        return False
    saw_pair = False
    for part in cookie.split(";"):
        name, separator, item_value = part.strip().partition("=")
        if not separator:
            continue
        saw_pair = True
        if name.strip() and _normalized_credential_secret(item_value):
            return True
    return not saw_pair


def _normalized_credential_metadata(value: Any) -> dict[str, Any]:
    metadata = dict(value) if isinstance(value, dict) else {}
    if "sessionToken" in metadata:
        metadata["sessionToken"] = _normalized_session_token(metadata.get("sessionToken"))
    return metadata


def _normalize_loaded_credential(value: Any) -> dict[str, Any] | None:
    if not isinstance(value, dict):
        return None
    normalized = dict(value)
    normalized["cookie"] = _normalized_cookie_header(value.get("cookie"))
    normalized["bearer"] = _normalized_bearer(value.get("bearer"))
    normalized["metadata"] = _normalized_credential_metadata(value.get("metadata"))
    return normalized


def default_cdp_url() -> str:
    return os.environ.get("WEBAI_DEFAULT_CDP_URL", "").strip() or DEFAULT_CDP_URL


REMOTE_AUTH_EMPTY_URL_DETAIL = (
    "这个授权 URL 里没有可直接使用的 cookie/bearer/session token。"
    "Qwen/DeepSeek 普通网页登录通常不会把登录态放在地址栏里；"
    "如果 Gateway 部署在 NAS 或远程机器上，请改用远程 CDP：在电脑上用 Chrome/Edge 启动远程调试端口，"
    "把 CDP 地址填为 http://电脑IP:9222 后再点“打开授权浏览器/重新检测登录态”。"
)
REMOTE_AUTH_CODE_ONLY_DETAIL = (
    "这个授权 URL 里只有一次性的 OAuth code，Gateway 没有这次授权的 verifier，不能直接换取网页登录态。"
    "请改用远程 CDP：在电脑上用 Chrome/Edge 启动远程调试端口，把 CDP 地址填为 http://电脑IP:9222 后再检测。"
)
REMOTE_AUTH_TOKEN_KEYS = ("bearer", "access_token", "token", "id_token", "auth_token", "jwt")
REMOTE_AUTH_COOKIE_KEYS = ("cookie", "cookies")
REMOTE_AUTH_SESSION_KEYS = ("session_token", "sessionToken", "qwen_session", "qwenSession")
REMOTE_AUTH_CODE_KEYS = ("code", "auth_code", "authorization_code")
REMOTE_AUTH_USER_AGENT_KEYS = ("user_agent", "userAgent", "ua")


@dataclass(frozen=True)
class WebAuthProvider:
    id: str
    name: str
    login_url: str
    status: str
    description: str
    models: tuple[str, ...]
    capabilities: dict[str, bool]
    adapters: tuple[str, ...]
    route: str = "webai2api"
    credential_required: bool = False
    tool_bridge: str = "strict"
    supports_native_tools: bool = False
    preferred_protocol: str = "openai"
    available_models: tuple[str, ...] | None = None
    advertise_models: bool = True
    availability_message: str = ""


WEBAI2API_MODELS: dict[str, tuple[str, ...]] = {
    "chatgpt": ("gpt-image-2", "gpt-image-1.5"),
    "chatgpt_text": ("gpt-instant", "gpt-thinking", "gpt-pro"),
    "deepseek_text": (
        "deepseek",
        "deepseek-thinking",
        "deepseek-search",
        "deepseek-thinking-search",
        "deepseek-expert",
        "deepseek-thinking-expert",
        "deepseek-search-expert",
        "deepseek-thinking-search-expert",
    ),
    "qwen_web": (
        "qwen3.7-max-preview",
        "qwen3.7-max",
        "qwen3.7-plus-preview",
        "qwen3.6-max",
        "qwen3.6-plus",
        "qwen3.6-flash",
        "qwen3.5-plus",
        "qwen3.5-turbo",
        "qwen3-max",
        "qwen-max",
        "qwen-plus",
        "qwen-turbo",
    ),
    "qwen_cn_web": (
        "Qwen3.6-Max",
        "Qwen3.6-Plus",
        "Qwen3.5-Plus",
        "Qwen3.5-Turbo",
        "qwen-max",
        "qwen-plus",
        "qwen-turbo",
    ),
    "doubao": ("seedream-4.5", "seedream-4.0", "seedream-5.0-lite"),
    "doubao_text": ("seed", "seed-thinking", "seed-pro"),
    "gemini": ("gemini-3-pro-image-preview", "veo-3.1-generate-preview"),
    "gemini_biz": ("gemini-3-pro-image-preview", "veo-3.1-generate-preview"),
    "gemini_biz_text": (
        "gemini-3-pro",
        "gemini-2.5-pro",
        "gemini-3-flash",
        "gemini-2.5-flash",
        "gemini-3-pro-grounding",
        "gemini-2.5-pro-grounding",
        "gemini-2.5-flash-grounding",
        "gemini-3-flash-grounding",
    ),
    "gemini_text": ("gemini-3.1-flash", "gemini-3.1-flash-thinking", "gemini-3.1-pro"),
    "google_flow": (
        "gemini-3-pro-image-preview",
        "gemini-3.1-flash-image-preview",
        "gemini-2.5-flash-image-preview",
        "imagen-4",
        "gemini-3-pro-image-preview-landspace",
        "gemini-3-pro-image-preview-landscape",
        "gemini-3-pro-image-preview-portrait",
        "gemini-3.1-flash-image-preview-landscape",
        "gemini-3.1-flash-image-landscape",
        "gemini-3.1-flash-image-preview-portrait",
        "gemini-3.1-flash-image-portrait",
        "gemini-2.5-flash-image-preview-landspace",
        "gemini-2.5-flash-image-preview-landscape",
        "gemini-2.5-flash-image-preview-portrait",
        "imagen-4-landspace",
        "imagen-4-landscape",
        "imagen-4-portrait",
    ),
    "lmarena": (
        "gemini-3.1-flash-image-preview",
        "gpt-image-1.5-high-fidelity",
        "gemini-3-pro-image-preview-2k",
        "mai-image-2",
        "reve-v1.5",
        "flux-2-max",
        "flux-2-flex",
        "flux-2-pro",
        "hunyuan-image-3.0",
        "flux-2-dev",
        "seedream-4.5",
        "qwen-image-2512",
        "imagen-4.0-generate-001",
        "wan2.5-t2i-preview",
        "gpt-image-1",
        "seedream-5.0-lite",
        "seedream-4-high-res-fal",
        "gpt-image-1-mini",
        "recraft-v4",
        "seedream-3",
        "flux-2-klein-9b",
        "qwen-image-prompt-extend",
        "flux-1-kontext-pro",
        "imagen-3.0-generate-002",
        "ideogram-v3-quality",
        "photon",
        "p-image",
        "flux-2-klein-4b",
        "recraft-v3",
        "runway-gen4",
        "lucid-origin",
        "dall-e-3",
        "flux-1-kontext-dev",
        "imagen-4.0-ultra-generate-001",
        "p-image-edit",
        "hunyuan-image-2.1",
        "reve-v1.1",
        "vidu-q2-image",
        "imagen-4.0-fast-generate-001",
        "qwen-image-2.0",
        "qwen-image-2.0-pro",
        "reve-v1.1-fast",
        "kling-image-o1",
        "chatgpt-image-latest-high-fidelity",
        "hunyuan-image-3.0-instruct",
        "wan2.7-image",
        "grok-imagine-image-pro",
        "grok-imagine-image",
        "wan2.7-image-pro",
        "qwen-image-edit-2511",
        "gemini-2.5-flash-image-preview",
        "wan2.5-i2i-preview",
        "qwen-image-edit",
        "wan2.6-image",
        "seededit-3.0",
        "wan2.6-t2i",
    ),
    "lmarena_text": (
        "claude-sonnet-4-5-20250929",
        "gemini-2.5-pro",
        "claude-haiku-4-5-20251001",
        "gemini-3-flash",
        "gpt-5.2-high",
        "gpt-5.1",
        "gpt-5.2",
        "grok-4.20-beta-0309-reasoning",
        "gpt-5.2-chat-latest",
        "deepseek-v3.2",
        "deepseek-v3.2-thinking",
        "gemini-2.5-flash",
        "grok-4-0709",
        "o4-mini-2025-04-16",
        "gpt-4.1-mini-2025-04-14",
        "claude-3-7-sonnet-20250219",
        "gemini-2.0-flash-001",
        "o3-mini",
    ),
    "nanobananafree_ai": ("gemini-2.5-flash-image",),
    "sora": ("sora-2",),
    "zai_is": (
        "gemini-3-pro-image-preview",
        "gemini-3-pro-image-preview-2k",
        "gemini-3-pro-image-preview-4k",
        "gemini-2.5-flash-image",
    ),
    "zai_is_text": (
        "glm-4.6",
        "gemini-3-pro-preview",
        "gemini-2.5-pro",
        "gemini-3-flash-preview",
        "claude-sonnet-4.5",
        "claude-sonnet-4",
        "claude-haiku-4.5",
        "gpt-5.1",
        "gpt-5",
        "gpt-4.1",
        "gpt-5.2",
        "o3-high",
        "o3-mini",
        "o4-mini",
        "grok-4.1-fast",
        "grok-4",
        "kimi-k2-thinking",
    ),
    "zenmux_ai_text": (
        "gemini-3-flash-preview",
        "mimo-v2-flash",
        "glm-4.6v-flash",
        "mistral-large-2512",
        "deepseek-v3.2",
        "deepseek-v3.2-thinking",
        "grok-4.1-fast",
        "gpt-5.1-codex-mini",
        "doubao-seed-code",
        "kimi-k2-thinking",
        "glm-4.6",
        "claude-sonnet-4.5",
        "qwen3-max",
        "gpt-5-mini",
        "gpt-5-nano",
        "gemini-2.5-flash-lite",
        "gemini-2.5-flash",
        "deepseek-r1-0528",
        "claude-sonnet-4",
        "o4-mini",
        "gpt-4.1-mini",
        "gpt-4.1-nano",
        "gemini-2.0-flash-lite",
        "claude-3.7-sonnet",
        "gemini-2.0-flash",
        "claude-3.5-sonnet",
    ),
}


def _models_for(*adapter_ids: str) -> tuple[str, ...]:
    seen: set[str] = set()
    models: list[str] = []
    for adapter_id in adapter_ids:
        for model_id in WEBAI2API_MODELS.get(adapter_id, ()):
            if model_id not in seen:
                seen.add(model_id)
                models.append(model_id)
    return tuple(models)


PROVIDERS: dict[str, WebAuthProvider] = {
    "lmarena": WebAuthProvider(
        id="lmarena",
        name="LMArena",
        login_url="https://lmarena.ai/",
        status="upstream",
        description="通过 WebAI2API 的 lmarena / lmarena_text 适配器支持文本与图片生成。",
        models=_models_for("lmarena_text", "lmarena"),
        capabilities={"text": True, "image": True, "video": False},
        adapters=("lmarena_text", "lmarena"),
    ),
    "gemini-biz": WebAuthProvider(
        id="gemini-biz",
        name="Gemini Enterprise Business",
        login_url="https://business.gemini.google/",
        status="upstream",
        description="通过 WebAI2API 的 gemini_biz / gemini_biz_text 适配器支持文本、图片和视频。",
        models=_models_for("gemini_biz_text", "gemini_biz"),
        capabilities={"text": True, "image": True, "video": True},
        adapters=("gemini_biz_text", "gemini_biz"),
        advertise_models=False,
    ),
    "nano-banana-free": WebAuthProvider(
        id="nano-banana-free",
        name="Nano Banana Free",
        login_url="https://nanobananafree.ai/",
        status="upstream",
        description="通过 WebAI2API 的 nanobananafree_ai 适配器支持图片生成。",
        models=_models_for("nanobananafree_ai"),
        capabilities={"text": False, "image": True, "video": False},
        adapters=("nanobananafree_ai",),
    ),
    "zai": WebAuthProvider(
        id="zai",
        name="zAI",
        login_url="https://zai.is/",
        status="upstream",
        description="通过 WebAI2API 的 zai_is / zai_is_text 适配器支持文本与图片生成。",
        models=_models_for("zai_is_text", "zai_is"),
        capabilities={"text": True, "image": True, "video": False},
        adapters=("zai_is_text", "zai_is"),
    ),
    "gemini": WebAuthProvider(
        id="gemini",
        name="Google Gemini",
        login_url="https://gemini.google.com/",
        status="upstream",
        description="通过 WebAI2API 的 gemini / gemini_text 适配器支持文本、图片和视频。",
        models=_models_for("gemini_text", "gemini"),
        capabilities={"text": True, "image": True, "video": True},
        adapters=("gemini_text", "gemini"),
        advertise_models=False,
    ),
    "zenmux": WebAuthProvider(
        id="zenmux",
        name="ZenMux",
        login_url="https://zenmux.ai/",
        status="upstream",
        description="通过 WebAI2API 的 zenmux_ai_text 适配器支持文本生成。",
        models=_models_for("zenmux_ai_text"),
        capabilities={"text": True, "image": False, "video": False},
        adapters=("zenmux_ai_text",),
    ),
    "chatgpt": WebAuthProvider(
        id="chatgpt",
        name="ChatGPT",
        login_url="https://chatgpt.com/",
        status="upstream",
        description="通过 WebAI2API 的 chatgpt / chatgpt_text 适配器支持文本与图片生成。",
        models=_models_for("chatgpt_text", "chatgpt"),
        capabilities={"text": True, "image": True, "video": False},
        adapters=("chatgpt_text", "chatgpt"),
    ),
    "qwen": WebAuthProvider(
        id="qwen",
        name="Qwen / 通义千问国际版",
        login_url="https://chat.qwen.ai/",
        status="available",
        description="Qwen Web 直连使用当前网页模型目录；可在本机授权浏览器登录 Qwen，并直接使用 qwen-web/ 前缀模型。",
        models=(
            "qwen-web/qwen3.8-max-preview",
            "qwen-web/qwen3.7-max",
            "qwen-web/qwen3.7-plus",
            "qwen-web/qwen3.6-plus",
        ),
        capabilities={"text": True, "image": False, "video": False},
        adapters=("qwen_web", "qwen-web"),
        route="direct",
        credential_required=True,
    ),
    "qwen-coder": WebAuthProvider(
        id="qwen-coder",
        name="Qwen Coder / 通义千问编程版",
        login_url="https://coder.qwen.ai/",
        status="available",
        description="Qwen Coder 专用编程助手，支持代码生成和调试；Gateway API 默认关闭上游 artifacts 工件流以保持协议稳定。使用 qwen-coder/ 前缀模型。",
        models=(
            "qwen-coder/qwen3-coder-plus",
            "qwen-coder/qwen-coder-plus",
        ),
        capabilities={"text": True, "image": False, "video": False, "artifacts": False},
        adapters=("qwen_coder", "qwen-coder"),
        route="direct",
        credential_required=True,
    ),
    "qwen-cn": WebAuthProvider(
        id="qwen-cn",
        name="通义千问（国内版）",
        login_url="https://www.qianwen.com/",
        status="upstream",
        description="对标 OpenClaw Zero Token 的 qwen-cn-web；当前走 WebAI2API 适配器目录，后续可接入国内版直连。",
        models=_models_for("qwen_cn_web"),
        capabilities={"text": True, "image": False, "video": False},
        adapters=("qwen_cn_web", "qwen-cn-web"),
    ),
    "deepseek-web": WebAuthProvider(
        id="deepseek-web",
        name="DeepSeek Web",
        login_url="https://chat.deepseek.com/",
        status="available",
        description=(
            "网关已内置本地网页登录授权；调用链路通过本地 ds2api sidecar 转到 DeepSeek Web。"
            "是否可用取决于本地 ds2api 是否运行、网页账号状态和当次模型检测结果。"
        ),
        models=DEEPSEEK_WEB_CATALOG_MODELS,
        capabilities={"text": True, "image": False, "video": False},
        adapters=("deepseek_text",),
        route="direct",
        credential_required=True,
        available_models=DEEPSEEK_WEB_CATALOG_MODELS,
        advertise_models=True,
        availability_message=DEEPSEEK_WEB_AVAILABLE_REASON,
    ),
    "sora": WebAuthProvider(
        id="sora",
        name="Sora",
        login_url="https://sora.chatgpt.com/",
        status="upstream",
        description="通过 WebAI2API 的 sora 适配器支持视频生成。",
        models=_models_for("sora"),
        capabilities={"text": False, "image": False, "video": True},
        adapters=("sora",),
        advertise_models=False,
    ),
    "google-flow": WebAuthProvider(
        id="google-flow",
        name="Google Flow",
        login_url="https://labs.google/fx/zh/tools/flow",
        status="upstream",
        description="通过 WebAI2API 的 google_flow 适配器支持图片生成。",
        models=_models_for("google_flow"),
        capabilities={"text": False, "image": True, "video": False},
        adapters=("google_flow",),
        advertise_models=False,
    ),
    "doubao": WebAuthProvider(
        id="doubao",
        name="豆包",
        login_url="https://www.doubao.com/",
        status="upstream",
        description="通过 WebAI2API 的 doubao / doubao_text 适配器支持文本与图片生成。",
        models=_models_for("doubao_text", "doubao"),
        capabilities={"text": True, "image": True, "video": False},
        adapters=("doubao_text", "doubao"),
    ),
}


ProgressCallback = Callable[[str], None]


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def get_provider(provider_id: str) -> WebAuthProvider:
    provider = PROVIDERS.get(provider_id)
    if not provider:
        raise ValueError(f"不支持的网页模型提供商：{provider_id}")
    return provider


def catalog_model_payloads(*, include_webai2api: bool = True) -> list[dict[str, Any]]:
    seen: set[str] = set()
    payloads: list[dict[str, Any]] = []
    for provider in PROVIDERS.values():
        if not provider.advertise_models:
            continue
        if not include_webai2api and provider.route != "direct":
            continue
        for model_id in provider.models:
            if model_id in seen:
                continue
            seen.add(model_id)
            payloads.append(
                {
                    "id": model_id,
                    "object": "model",
                    "owned_by": provider.id,
                    "capabilities": {
                        "tool_bridge": provider.tool_bridge != "off",
                        "supports_native_tools": provider.supports_native_tools,
                        "preferred_protocol": provider.preferred_protocol,
                    },
                }
            )
    return payloads


class CredentialStore:
    def __init__(self, root: str | Path = "credentials") -> None:
        self.root = Path(root)
        self._lock = threading.RLock()
        self._revisions: dict[str, int] = {}

    def _path(self, provider_id: str) -> Path:
        return self.root / f"{provider_id}.json"

    def _profile_path(self, provider_id: str, profile_id: str) -> Path:
        safe_profile = quote(str(profile_id or "default"), safe="")
        return self.root / "direct" / provider_id / f"{safe_profile}.json"

    def save(self, provider_id: str, credential: dict[str, Any]) -> dict[str, Any]:
        with self._lock:
            data = self._save_locked(provider_id, credential)
            self._advance_revision_locked(provider_id)
            return data

    def revision(self, provider_id: str) -> int:
        get_provider(provider_id)
        with self._lock:
            return self._revisions.get(provider_id, 0)

    def save_if_revision(
        self,
        provider_id: str,
        credential: dict[str, Any],
        expected_revision: int,
    ) -> dict[str, Any] | None:
        """Save a captured credential only while its authorization generation is current.

        Browser capture can take minutes. A credential clear or a newer explicit
        save that happens meanwhile advances the generation, so the stale capture
        cannot recreate or overwrite the user's current credential.
        """

        get_provider(provider_id)
        with self._lock:
            if self._revisions.get(provider_id, 0) != expected_revision:
                return None
            data = self._save_locked(provider_id, credential)
            self._advance_revision_locked(provider_id)
            return data

    def _advance_revision_locked(self, provider_id: str) -> None:
        self._revisions[provider_id] = self._revisions.get(provider_id, 0) + 1

    def _save_locked(self, provider_id: str, credential: dict[str, Any]) -> dict[str, Any]:
        get_provider(provider_id)
        existing = self.get(provider_id) or {}
        cookie = _normalized_cookie_header(credential.get("cookie"))
        bearer = _normalized_bearer(credential.get("bearer"))
        user_agent = _normalized_credential_secret(credential.get("userAgent") or credential.get("user_agent"))
        metadata = _normalized_credential_metadata(credential.get("metadata"))
        if not _cookie_header_has_usable_value(cookie) and not bearer:
            raise ValueError("没有捕获到可用的 cookie 或 bearer，网页登录可能尚未完成")
        now = utc_now()
        data = {
            "provider": provider_id,
            "cookie": cookie,
            "bearer": bearer,
            "userAgent": user_agent,
            "metadata": metadata,
            "createdAt": str(existing.get("createdAt") or now),
            "updatedAt": now,
        }
        if not is_credential_authorized(provider_id, data):
            if provider_id == "qwen":
                raise ValueError("没有捕获到可用的 Qwen 登录态，请先完成 chat.qwen.ai 登录")
            if provider_id == "deepseek-web":
                raise ValueError("没有捕获到可用的 DeepSeek bearer token，请重新完成 chat.deepseek.com 登录授权")
            raise ValueError("没有捕获到可用的 cookie 或 bearer，网页登录可能尚未完成")
        path = self._path(provider_id)
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(path.suffix + ".tmp")
        tmp.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        tmp.replace(path)
        if provider_id in QWEN_DIRECT_PROVIDER_IDS or provider_id == "deepseek-web":
            self._save_profile_locked(provider_id, "default", credential)
        return data

    def save_profile(self, provider_id: str, profile_id: str, credential: dict[str, Any]) -> dict[str, Any]:
        with self._lock:
            data = self._save_profile_locked(provider_id, profile_id, credential)
            self._advance_revision_locked(provider_id)
            return data

    def _save_profile_locked(self, provider_id: str, profile_id: str, credential: dict[str, Any]) -> dict[str, Any]:
        get_provider(provider_id)
        profile = str(profile_id or "default").strip() or "default"
        existing = self.get_profile(provider_id, profile) or {}
        cookie = _normalized_cookie_header(credential.get("cookie"))
        bearer = _normalized_bearer(credential.get("bearer"))
        user_agent = _normalized_credential_secret(credential.get("userAgent") or credential.get("user_agent"))
        metadata = _normalized_credential_metadata(credential.get("metadata"))
        if not _cookie_header_has_usable_value(cookie) and not bearer:
            raise ValueError("没有捕获到可用的网页登录凭据，请重新完成授权")
        now = utc_now()
        data = {
            "provider": provider_id,
            "profileId": profile,
            "cookie": cookie,
            "bearer": bearer,
            "userAgent": user_agent,
            "metadata": metadata,
            "createdAt": str(existing.get("createdAt") or now),
            "updatedAt": now,
        }
        if not is_credential_authorized(provider_id, data):
            raise ValueError("没有捕获到可用的网页登录态，请重新完成授权")
        path = self._profile_path(provider_id, profile)
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(path.suffix + ".tmp")
        tmp.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        tmp.replace(path)
        return data

    def get(self, provider_id: str) -> dict[str, Any] | None:
        with self._lock:
            path = self._path(provider_id)
            if not path.exists():
                return None
            try:
                data = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, UnicodeError, json.JSONDecodeError):
                return None
            return _normalize_loaded_credential(data)

    def get_profile(self, provider_id: str, profile_id: str) -> dict[str, Any] | None:
        with self._lock:
            profile = str(profile_id or "default").strip() or "default"
            path = self._profile_path(provider_id, profile)
            if path.exists():
                try:
                    data = json.loads(path.read_text(encoding="utf-8"))
                except (OSError, UnicodeError, json.JSONDecodeError):
                    data = None
                normalized = _normalize_loaded_credential(data)
                if normalized is not None:
                    return normalized
            if profile == "default":
                return self.get(provider_id)
            return None

    def list_profile_ids(self, provider_id: str) -> list[str]:
        with self._lock:
            ids: list[str] = []
            root = self.root / "direct" / provider_id
            if root.exists():
                for path in sorted(root.glob("*.json")):
                    ids.append(unquote(path.stem))
            if self.get(provider_id) is not None and "default" not in ids:
                ids.insert(0, "default")
            return ids

    def delete(self, provider_id: str) -> None:
        get_provider(provider_id)
        with self._lock:
            # Advance even when no file exists: clearing credentials is an
            # explicit invalidation barrier for every capture already running.
            self._advance_revision_locked(provider_id)
            path = self._path(provider_id)
            if path.exists():
                path.unlink()
            profile_root = self.root / "direct" / provider_id
            is_junction = getattr(profile_root, "is_junction", lambda: False)
            if profile_root.is_dir() and not profile_root.is_symlink() and not is_junction():
                for profile_path in profile_root.glob("*.json"):
                    if profile_path.is_file():
                        profile_path.unlink()
                try:
                    profile_root.rmdir()
                except OSError:
                    # Preserve unrelated files instead of recursively deleting a
                    # credential directory that may contain user-managed data.
                    pass

    def summary(self, provider_id: str) -> dict[str, Any]:
        credential = self.get(provider_id)
        return credential_summary(provider_id, credential)

    def list_summaries(self) -> list[dict[str, Any]]:
        return [self.summary(provider_id) for provider_id in PROVIDERS]


def credential_summary(provider_id: str, credential: dict[str, Any] | None) -> dict[str, Any]:
    return {
        "provider": provider_id,
        "authorized": is_credential_authorized(provider_id, credential),
        "updatedAt": credential.get("updatedAt") if credential else None,
        "fields": {
            "cookie": bool(credential and _cookie_header_has_usable_value(credential.get("cookie"))),
            "bearer": bool(credential and _normalized_bearer(credential.get("bearer"))),
            "userAgent": bool(credential and _normalized_credential_secret(credential.get("userAgent"))),
        },
    }


def is_credential_authorized(provider_id: str, credential: dict[str, Any] | None) -> bool:
    if not credential:
        return False
    bearer = _normalized_bearer(credential.get("bearer"))
    if provider_id == "qwen":
        # Qwen Web's v2 chat endpoints require the web API bearer token. A
        # browser session cookie may prove that the user is signed in to the
        # page, but cannot by itself invoke the API.
        return bool(bearer)
    if provider_id == "qwen-coder":
        metadata = _normalized_credential_metadata(credential.get("metadata"))
        session_token = _normalized_session_token(metadata.get("sessionToken"))
        cookie_session = _qwen_session_from_cookie_header(_normalized_cookie_header(credential.get("cookie")))
        return bool(bearer or (session_token and cookie_session and session_token == cookie_session))
    if provider_id == "deepseek-web":
        return bool(bearer)
    return bool(_cookie_header_has_usable_value(credential.get("cookie")) or bearer)


def credential_from_remote_auth_url(provider_id: str, auth_url: str, user_agent: str = "") -> dict[str, Any]:
    provider = get_provider(provider_id)
    url = str(auth_url or "").strip()
    parsed = urlparse(url)
    if not parsed.scheme or not parsed.netloc:
        raise ValueError("请粘贴完整的授权 URL。")

    params = _remote_auth_url_params(parsed)
    bearer = _strip_bearer_prefix(_first_remote_auth_param(params, REMOTE_AUTH_TOKEN_KEYS))
    cookie = _normalized_cookie_header(_first_remote_auth_param(params, REMOTE_AUTH_COOKIE_KEYS))
    raw_session_token = _first_remote_auth_param(params, REMOTE_AUTH_SESSION_KEYS)
    session_token = _normalized_session_token(raw_session_token)
    code = _first_remote_auth_param(params, REMOTE_AUTH_CODE_KEYS)
    parsed_user_agent = _first_remote_auth_param(params, REMOTE_AUTH_USER_AGENT_KEYS)

    if not cookie and provider.id in QWEN_DIRECT_PROVIDER_IDS and session_token:
        cookie = f"qwen_session={session_token}"
    # A remote callback can explicitly carry a Bearer value in its Qwen
    # session field. Preserve that explicit API credential instead of treating
    # every session-shaped value as a cookie-only browser login.
    if (
        provider.id == "qwen"
        and not bearer
        and str(raw_session_token or "").strip().lower().startswith("bearer ")
    ):
        bearer = _normalized_bearer(raw_session_token)
    if not session_token and provider.id in QWEN_DIRECT_PROVIDER_IDS and cookie:
        session_token = _qwen_session_from_cookie_header(cookie)

    if code and not any([bearer, cookie, session_token]):
        raise ValueError(REMOTE_AUTH_CODE_ONLY_DETAIL)

    metadata: dict[str, Any] = {}
    if session_token:
        metadata["sessionToken"] = session_token

    credential = {
        "cookie": cookie,
        "bearer": bearer,
        "userAgent": str(user_agent or parsed_user_agent or ""),
        "metadata": metadata,
    }
    if not is_credential_authorized(provider.id, credential):
        raise ValueError(REMOTE_AUTH_EMPTY_URL_DETAIL)
    return credential


def _remote_auth_url_params(parsed_url: Any) -> dict[str, str]:
    params: dict[str, str] = {}
    for raw in (parsed_url.query, parsed_url.fragment):
        for key, value in parse_qsl(raw, keep_blank_values=True):
            clean_key = str(key or "").strip()
            if clean_key and value and clean_key not in params:
                params[clean_key] = str(value).strip()
    return params


def _first_remote_auth_param(params: dict[str, str], names: tuple[str, ...]) -> str:
    for name in names:
        value = params.get(name)
        if value:
            return value
    lowered = {key.lower(): value for key, value in params.items() if value}
    for name in names:
        value = lowered.get(name.lower())
        if value:
            return value
    return ""


def _strip_bearer_prefix(value: str) -> str:
    return _normalized_bearer(value)


def _qwen_session_from_cookie_header(cookie: str) -> str:
    for part in str(cookie or "").split(";"):
        name, _, value = part.strip().partition("=")
        lowered = name.lower()
        if value and lowered in QWEN_LOGIN_COOKIE_NAMES:
            return _normalized_session_token(value)
    return ""


def provider_payload(store: CredentialStore) -> dict[str, Any]:
    providers = []
    for provider in PROVIDERS.values():
        available_models = provider.models if provider.available_models is None else provider.available_models
        providers.append(
            {
                "id": provider.id,
                "name": provider.name,
                "loginUrl": provider.login_url,
                "status": provider.status,
                "description": provider.description,
                "models": list(provider.models),
                "availableModels": list(available_models),
                "modelCount": len(available_models),
                "advertiseModels": provider.advertise_models,
                "availabilityMessage": provider.availability_message,
                "credential": store.summary(provider.id),
                "capabilities": provider.capabilities,
                "adapters": list(provider.adapters),
                "route": provider.route,
                "credentialRequired": provider.credential_required,
                "toolBridge": provider.tool_bridge,
                "supportsNativeTools": provider.supports_native_tools,
                "preferredProtocol": provider.preferred_protocol,
            }
        )
    return {"providers": providers, "defaultCdpUrl": default_cdp_url()}


class BrowserLauncher:
    def __init__(self, profile_dir: str | Path = ".webai-gateway/chrome-auth-profile") -> None:
        self.profile_dir = Path(profile_dir)
        self._active_cdp_urls: dict[str, str] = {}
        self._lock = threading.RLock()

    def start(self, provider_id: str, cdp_url: str | None = None) -> dict[str, Any]:
        provider = get_provider(provider_id)
        requested_cdp_url = str(cdp_url or default_cdp_url()).strip()
        if not _is_loopback_cdp_url(requested_cdp_url):
            cdp_ready = _wait_for_cdp_ready(requested_cdp_url)
            return {
                "provider": provider.id,
                "cdpUrl": requested_cdp_url,
                "loginUrl": provider.login_url,
                "started": False,
                "cdpReady": cdp_ready,
                "message": (
                    "已检测到远程 CDP 授权浏览器，可以开始检测登录态。"
                    if cdp_ready
                    else "当前 CDP 地址不是本机浏览器，Gateway 不会在本机另开浏览器。"
                    "请先在该设备启动 Chrome/Edge 远程调试并确认地址可访问，然后重新检测登录态。"
                ),
            }

        try:
            requested_port = _port_from_cdp_url(requested_cdp_url)
        except ValueError:
            return {
                "provider": provider.id,
                "cdpUrl": requested_cdp_url,
                "loginUrl": provider.login_url,
                "started": False,
                "cdpReady": False,
                "message": "CDP 地址中的端口格式无效，请检查后重试。",
            }
        port = requested_port
        port_reassigned = _loopback_port_in_use(port)
        if port_reassigned:
            try:
                port = _find_available_loopback_port()
            except RuntimeError:
                return {
                    "provider": provider.id,
                    "cdpUrl": requested_cdp_url,
                    "loginUrl": provider.login_url,
                    "started": False,
                    "cdpReady": False,
                    "message": "未能为授权浏览器分配本机调试端口，请稍后重试。",
                }
        effective_cdp_url = _loopback_cdp_url(port)

        with self._lock:
            active_cdp_url = self._active_cdp_urls.get(provider.id, "")
        if active_cdp_url and _cdp_endpoint_ready(active_cdp_url):
            return {
                "provider": provider.id,
                "cdpUrl": active_cdp_url,
                "loginUrl": provider.login_url,
                "started": False,
                "cdpReady": True,
                "message": "已检测到本次授权窗口，正在继续使用，不会重复打开浏览器。",
            }
        if active_cdp_url:
            with self._lock:
                self._active_cdp_urls.pop(provider.id, None)

        browsers = find_browser_executables()
        if not browsers:
            cdp_ready = _wait_for_cdp_ready(effective_cdp_url)
            return {
                "provider": provider.id,
                "cdpUrl": effective_cdp_url,
                "loginUrl": provider.login_url,
                "started": False,
                "cdpReady": cdp_ready,
                "message": (
                    "已检测到现有授权浏览器，可以开始检测登录态。"
                    if cdp_ready
                    else "没有找到 Chrome 或 Edge，也没有检测到可用的授权浏览器。"
                    "请先安装浏览器，或手动用调试端口启动后再点“重新检测登录态”。"
                ),
            }

        # A browser may display a window even when its CDP endpoint is not
        # reachable.  Starting more fallbacks in that state creates several
        # indistinguishable authorization windows, so one user action starts
        # exactly one fresh, isolated browser profile.  A later explicit retry
        # can choose a new session without touching normal browser profiles.
        primary_name, primary_browser = browsers[0]
        profile_dir = self.profile_dir / provider.id / f"session-{uuid.uuid4().hex[:10]}"
        profile_dir.mkdir(parents=True, exist_ok=True)
        command = [
            primary_browser,
            "--remote-debugging-address=127.0.0.1",
            f"--remote-debugging-port={port}",
            f"--user-data-dir={str(profile_dir.resolve())}",
            "--no-first-run",
            "--disable-default-apps",
            "--new-window",
            provider.login_url,
        ]
        try:
            process = subprocess.Popen(command, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        except OSError:
            return {
                "provider": provider.id,
                "cdpUrl": effective_cdp_url,
                "loginUrl": provider.login_url,
                "started": False,
                "cdpReady": False,
                "message": "授权浏览器未能启动。请检查 Chrome/Edge 是否可正常打开，然后重试。",
            }

        cdp_ready = _wait_for_cdp_ready(effective_cdp_url)
        process_exited = bool(getattr(process, "poll", lambda: None)() is not None)
        attempts = [{"browser": primary_name, "started": True, "cdpReady": cdp_ready, "launcherExited": process_exited}]
        if not cdp_ready:
            return {
                "provider": provider.id,
                "cdpUrl": effective_cdp_url,
                "loginUrl": provider.login_url,
                "started": False,
                "cdpReady": False,
                "pid": process.pid,
                "attempts": attempts,
                "message": (
                    "授权窗口已启动，但未能建立本机调试连接；为避免重复窗口，本次不会自动再打开浏览器。"
                    "请改用“远程/NAS 授权”，或确认该窗口关闭后再重试。"
                ),
            }

        with self._lock:
            self._active_cdp_urls[provider.id] = effective_cdp_url
        return {
            "provider": provider.id,
            "cdpUrl": effective_cdp_url,
            "loginUrl": provider.login_url,
            "started": True,
            "cdpReady": True,
            "pid": process.pid,
            "attempts": attempts,
            "message": (
                "默认调试端口已被其他浏览器占用，已为本次授权打开独立浏览器窗口。"
                if port_reassigned
                else "授权浏览器已启动，请在弹出的窗口里完成登录。"
            ),
        }


def find_browser_executables() -> list[tuple[str, str]]:
    candidates = [
        ("Chrome", os.environ.get("CHROME_PATH")),
        ("Edge", os.environ.get("EDGE_PATH")),
        ("Chrome", Path(os.environ.get("PROGRAMFILES", "")) / "Google/Chrome/Application/chrome.exe"),
        ("Chrome", Path(os.environ.get("PROGRAMFILES(X86)", "")) / "Google/Chrome/Application/chrome.exe"),
        ("Chrome", Path(os.environ.get("LOCALAPPDATA", "")) / "Google/Chrome/Application/chrome.exe"),
        ("Edge", Path(os.environ.get("PROGRAMFILES", "")) / "Microsoft/Edge/Application/msedge.exe"),
        ("Edge", Path(os.environ.get("PROGRAMFILES(X86)", "")) / "Microsoft/Edge/Application/msedge.exe"),
    ]
    found: list[tuple[str, str]] = []
    seen: set[str] = set()
    for name, candidate in candidates:
        if not candidate:
            continue
        path = Path(candidate)
        key = str(path).casefold()
        if path.exists() and key not in seen:
            seen.add(key)
            found.append((name, str(path)))
    return found


def find_browser_executable() -> str | None:
    browsers = find_browser_executables()
    return browsers[0][1] if browsers else None


def _port_from_cdp_url(cdp_url: str) -> int:
    parsed = urlparse(cdp_url)
    if parsed.port:
        return parsed.port
    return 9222


def _is_loopback_cdp_url(cdp_url: str) -> bool:
    try:
        parsed = urlparse(cdp_url)
        return parsed.scheme in {"http", "https"} and (parsed.hostname or "").lower() in _LOOPBACK_CDP_HOSTS
    except ValueError:
        return False


def _loopback_cdp_url(port: int) -> str:
    return f"http://127.0.0.1:{port}"


def _loopback_port_in_use(port: int) -> bool:
    for family, address in ((socket.AF_INET, "127.0.0.1"), (socket.AF_INET6, "::1")):
        try:
            with socket.socket(family, socket.SOCK_STREAM) as probe:
                probe.settimeout(0.15)
                probe.connect((address, port))
                return True
        except OSError:
            continue
    return False


def _find_available_loopback_port() -> int:
    for _ in range(32):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as reservation:
            reservation.bind(("127.0.0.1", 0))
            port = int(reservation.getsockname()[1])
        if not _loopback_port_in_use(port):
            return port
    raise RuntimeError("无法分配本机授权浏览器调试端口")


def _cdp_endpoint_ready(cdp_url: str) -> bool:
    endpoint = f"{cdp_url.rstrip('/')}/json/version"
    try:
        with urlopen(endpoint, timeout=0.5) as response:  # nosec B310 - user-authorized local/remote CDP endpoint
            if int(getattr(response, "status", 200) or 200) != 200:
                return False
            payload = json.loads(response.read().decode("utf-8"))
    except (HTTPError, OSError, URLError, UnicodeError, ValueError, json.JSONDecodeError):
        return False
    return isinstance(payload, dict) and bool(payload.get("webSocketDebuggerUrl"))


def _wait_for_cdp_ready(cdp_url: str, timeout_seconds: float = _CDP_READY_TIMEOUT_SECONDS) -> bool:
    deadline = time.monotonic() + max(0.0, timeout_seconds)
    while True:
        if _cdp_endpoint_ready(cdp_url):
            return True
        if time.monotonic() >= deadline:
            return False
        time.sleep(_CDP_READY_POLL_SECONDS)


class DeepSeekWebAuthService:
    async def capture(
        self,
        provider_id: str = "deepseek-web",
        cdp_url: str | None = None,
        progress: ProgressCallback | None = None,
        timeout_seconds: int = 180,
    ) -> dict[str, Any]:
        provider = get_provider(provider_id)
        cdp_url = cdp_url or default_cdp_url()
        if provider.id not in {"deepseek-web", *QWEN_DIRECT_PROVIDER_IDS}:
            raise ValueError(f"{provider.name} 暂未实现本地自动捕获；请在 WebAI2API 管理台完成该站点登录。")
        try:
            from playwright.async_api import async_playwright
        except Exception as exc:  # pragma: no cover - environment dependent
            raise RuntimeError("缺少 Playwright，请先运行 pip install playwright") from exc

        def notify(message: str) -> None:
            if progress:
                progress(message)

        bearer_seen = ""
        cookie_only_notice_sent = False
        qwen_token_notice_sent = False
        started_at = time.monotonic()
        notify("正在连接授权浏览器")
        async with async_playwright() as playwright:
            browser = await playwright.chromium.connect_over_cdp(cdp_url)
            context = browser.contexts[0] if browser.contexts else await browser.new_context()
            # This browser may be a user-managed remote CDP instance.  Create
            # an authorization page instead of navigating one of the user's
            # existing tabs, then let Playwright disconnect with the manager.
            # Calling browser.close() here would close the user's browser.
            page = await context.new_page()

            def on_request(request: Any) -> None:
                nonlocal bearer_seen
                try:
                    auth = str(request.headers.get("authorization") or "")
                except Exception:
                    auth = ""
                url = str(request.url)
                if provider.id == "deepseek-web" and "/api/v0/" in url and auth.lower().startswith("bearer "):
                    bearer_seen = auth.split(" ", 1)[1].strip()
                if provider.id in QWEN_DIRECT_PROVIDER_IDS and "qwen.ai" in url and auth.lower().startswith("bearer "):
                    bearer_seen = auth.split(" ", 1)[1].strip()

            page.on("request", on_request)
            await page.goto(provider.login_url)
            if provider.id in QWEN_DIRECT_PROVIDER_IDS:
                notify(f"等待 {provider.name} 登录态：请在 {provider.login_url} 完成登录，检测到登录态后才会完成")
            else:
                notify(f"请在弹出的浏览器里完成 {provider.name} 登录")

            while time.monotonic() - started_at < timeout_seconds:
                if provider.id in QWEN_DIRECT_PROVIDER_IDS:
                    credential = await _read_qwen_credential(
                        context,
                        page,
                        bearer_seen,
                        origins=QWEN_CREDENTIAL_ORIGINS[provider.id],
                    )
                else:
                    credential = await _read_deepseek_credential(context, page, bearer_seen)
                if credential and is_credential_authorized(provider.id, credential):
                    if provider.id in QWEN_DIRECT_PROVIDER_IDS:
                        notify(f"已检测到 {provider.name} 登录态")
                    else:
                        notify(f"已捕获 {provider.name} 登录态")
                    return credential
                if (
                    provider.id == "deepseek-web"
                    and credential
                    and credential.get("cookie")
                    and not credential.get("bearer")
                    and not cookie_only_notice_sent
                ):
                    notify("已看到 DeepSeek 登录 cookie，正在继续等待 bearer token；请保持聊天页打开，必要时刷新 chat.deepseek.com。")
                    cookie_only_notice_sent = True
                if (
                    provider.id == "qwen"
                    and credential
                    and credential.get("cookie")
                    and not credential.get("bearer")
                    and not qwen_token_notice_sent
                ):
                    notify(
                        "已检测到 Qwen 网页登录，但还没有捕获可调用的 API token。"
                        "请在授权浏览器中打开任一对话并发送一条消息，保持窗口打开；Gateway 将自动继续检测。"
                    )
                    qwen_token_notice_sent = True
                await asyncio.sleep(2)
        if provider.id in QWEN_DIRECT_PROVIDER_IDS:
            raise TimeoutError(f"等待 {provider.name} 登录态超时，请确认已经在 {provider.login_url} 登录成功后重试")
        if provider.id == "deepseek-web":
            raise TimeoutError(
                "等待 DeepSeek Web bearer token 超时；请确认 chat.deepseek.com 已进入聊天页，"
                "保持授权浏览器打开，然后点击“重新检测登录态”。"
            )
        raise TimeoutError(f"等待 {provider.name} 登录超时，请确认网页已经登录成功后重试")


async def _read_deepseek_credential(context: Any, page: Any, bearer_seen: str = "") -> dict[str, Any] | None:
    cookies = await context.cookies("https://chat.deepseek.com")
    cookie_text = "; ".join(f"{item.get('name')}={item.get('value')}" for item in cookies if item.get("name") and item.get("value"))
    user_agent = await page.evaluate("() => navigator.userAgent")
    bearer = bearer_seen
    try:
        response = await context.request.get(
            "https://chat.deepseek.com/api/v0/users/current",
            headers={"Cookie": cookie_text, "User-Agent": user_agent},
        )
        if response.ok:
            data = await response.json()
            bearer = _deep_get(data, ("data", "biz_data", "token")) or _deep_get(data, ("data", "token")) or bearer
    except Exception:
        pass
    has_session_cookie = any(token in cookie_text for token in ("ds_session_id=", "HWSID=", "d_id=", "uuid="))
    if not bearer and not has_session_cookie:
        return None
    return {"cookie": cookie_text, "bearer": bearer, "userAgent": user_agent}


async def _read_qwen_credential(
    context: Any,
    page: Any,
    bearer_seen: str = "",
    *,
    origins: tuple[str, ...] = QWEN_CREDENTIAL_ORIGINS["qwen"],
) -> dict[str, Any] | None:
    # Ask Playwright for cookies applicable to the provider origin. Parent
    # domain cookies are included automatically, while host-only cookies from
    # the sibling Qwen application cannot overwrite one another here.
    cookies = await context.cookies(origins[0])
    cookie_text = "; ".join(f"{item.get('name')}={item.get('value')}" for item in cookies if item.get("name") and item.get("value"))
    try:
        browser_state = await page.evaluate(
            "() => ({ userAgent: navigator.userAgent, token: window.localStorage?.getItem('token') || '' })"
        )
    except Exception:
        try:
            browser_state = {"userAgent": await page.evaluate("() => navigator.userAgent"), "token": ""}
        except Exception:
            browser_state = {}
    if isinstance(browser_state, dict):
        user_agent = str(browser_state.get("userAgent") or "")
        local_storage_token = _normalized_session_token(browser_state.get("token"))
    else:
        # Kept for lightweight alternate Playwright adapters and test doubles.
        user_agent = str(browser_state or "")
        local_storage_token = ""
    captured_bearer = _normalized_bearer(bearer_seen)
    cookie_session_token = _qwen_session_token_from_cookies(cookies)
    session_token = captured_bearer or local_storage_token or cookie_session_token
    if not session_token:
        return None
    bearer = captured_bearer or local_storage_token or _qwen_bearer_token_from_cookies(cookies)
    if not await _verify_qwen_login_state(
        context,
        origin=origins[0],
        cookie=cookie_text,
        user_agent=user_agent,
        bearer=bearer,
    ):
        return None
    return {
        "cookie": cookie_text,
        "bearer": bearer,
        "userAgent": user_agent,
        "metadata": {"sessionToken": session_token},
    }


def _qwen_session_token_from_cookies(cookies: list[dict[str, Any]]) -> str:
    for item in cookies:
        name = str(item.get("name") or "").lower()
        if name in QWEN_LOGIN_COOKIE_NAMES:
            value = _normalized_session_token(item.get("value"))
            if value:
                return value
    return ""


def _qwen_bearer_token_from_cookies(cookies: list[dict[str, Any]]) -> str:
    for item in cookies:
        if str(item.get("name") or "").lower() in {"token", "tongyi_sso_ticket"}:
            return _normalized_bearer(item.get("value"))
    return ""


async def _verify_qwen_login_state(
    context: Any,
    *,
    origin: str,
    cookie: str,
    user_agent: str,
    bearer: str,
) -> bool:
    request_context = getattr(context, "request", None)
    request_get = getattr(request_context, "get", None)
    if not callable(request_get):
        return False
    headers = {
        "Cookie": cookie,
        "User-Agent": user_agent,
        "Accept": "application/json",
    }
    if bearer:
        headers["Authorization"] = f"Bearer {bearer}"
    try:
        auth_url = f"{str(origin).rstrip('/')}/api/v1/auths/"
        response = await request_get(auth_url, headers=headers)
    except Exception:
        return False
    if not bool(getattr(response, "ok", False)):
        return False
    response_url = str(getattr(response, "url", "") or "")
    if response_url and not _same_qwen_auth_endpoint(auth_url, response_url):
        return False
    response_headers = getattr(response, "headers", {})
    content_type = ""
    if isinstance(response_headers, dict):
        content_type = str(response_headers.get("content-type") or response_headers.get("Content-Type") or "")
    if content_type and "json" not in content_type.lower():
        return False
    response_json = getattr(response, "json", None)
    if not callable(response_json):
        return False
    try:
        payload = await response_json()
    except Exception:
        return False
    return _qwen_auth_payload_has_authenticated_user(payload)


def _same_qwen_auth_endpoint(expected_url: str, response_url: str) -> bool:
    expected = urlparse(expected_url)
    actual = urlparse(response_url)
    return (
        actual.scheme.lower() == expected.scheme.lower()
        and actual.netloc.lower() == expected.netloc.lower()
        and actual.path.rstrip("/") == expected.path.rstrip("/")
    )


def _qwen_auth_payload_has_authenticated_user(payload: Any) -> bool:
    if not isinstance(payload, dict):
        return False

    candidates: list[tuple[dict[str, Any], bool]] = [(payload, True)]
    has_root_envelope = False
    for key in ("user", "account", "data"):
        nested = payload.get(key)
        if not isinstance(nested, dict):
            continue
        has_root_envelope = True
        candidates.append((nested, False))
    data = payload.get("data")
    if isinstance(data, dict):
        for key in ("user", "account"):
            nested = data.get(key)
            if isinstance(nested, dict):
                candidates.append((nested, False))

    # Reject a guest marker from any supported identity envelope before
    # considering positive identity evidence. Qwen has returned both JSON
    # booleans and stringified booleans for this field.
    for candidate, _is_root in candidates:
        is_guest = _normalized_qwen_bool(candidate.get("is_guest")) or _normalized_qwen_bool(
            candidate.get("isGuest")
        )
        role = _normalized_credential_secret(candidate.get("role")).lower()
        email = _normalized_credential_secret(candidate.get("email")).lower()
        if is_guest or role in {"guest", "visitor"} or email.endswith("@guest.com"):
            return False

    strong_identity_fields = (
        "email",
        "profile_image_url",
        "profileImageUrl",
        "avatar_url",
        "avatarUrl",
        "picture",
        "phone",
    )
    for candidate, is_root in candidates:
        explicit_user_id = candidate.get("user_id") or candidate.get("userId") or candidate.get("uid")
        if _normalized_qwen_identity_id(explicit_user_id):
            return True

        # A generic `id` is ambiguous: auth envelopes also use it as a
        # request/response identifier. It is accepted only alongside strong
        # identity evidence on the same object, and never from a root envelope
        # that already contains a user/account/data object.
        if is_root and has_root_envelope:
            continue
        user_id = _normalized_qwen_identity_id(candidate.get("id"))
        has_strong_identity = any(
            _normalized_credential_secret(candidate.get(key)) for key in strong_identity_fields
        )
        if user_id and has_strong_identity:
            return True
    return False


def _normalized_qwen_bool(value: Any) -> bool:
    if value is True:
        return True
    if isinstance(value, str):
        return value.strip().lower() in {"true", "1", "yes", "y", "on"}
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return value == 1
    return False


def _normalized_qwen_identity_id(value: Any) -> str:
    if isinstance(value, bool) or value is None:
        return ""
    if isinstance(value, (int, float)):
        return str(value)
    return _normalized_credential_secret(value)


def _deep_get(data: Any, path: tuple[str, ...]) -> Any:
    current = data
    for key in path:
        if not isinstance(current, dict):
            return None
        current = current.get(key)
    return current


def create_job(provider_id: str) -> dict[str, Any]:
    return {
        "id": "auth_" + uuid.uuid4().hex,
        "provider": provider_id,
        "status": "running",
        "message": "正在等待网页登录授权",
        "createdAt": utc_now(),
        "updatedAt": utc_now(),
    }
