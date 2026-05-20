from __future__ import annotations

import base64
from collections import OrderedDict, deque
from datetime import datetime, timezone
import inspect
import json
import os
import re
import secrets
import shutil
import subprocess
import threading
import time
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit, urlunsplit

import httpx
from fastapi import FastAPI, Header, HTTPException, Request
from fastapi.responses import FileResponse, JSONResponse, Response
from fastapi.staticfiles import StaticFiles
from starlette.concurrency import run_in_threadpool

from webai_gateway.accounts import (
    AccountRegistry,
    direct_account_id,
    parse_account_id,
    stable_json_hash,
    webai2api_account_id,
)
from webai_gateway.auto_research import build_auto_research_status
from webai_gateway.config import GatewayConfig, config_to_admin, config_to_public, load_config, save_config, update_config
from webai_gateway.deepseek_web import (
    DEEPSEEK_DEFAULT_MODEL,
    DeepSeekDs2apiError,
    DeepSeekWebClient,
    is_deepseek_web_model,
)
from webai_gateway.ds2api_oracle import DS2API_ORACLE_COMMIT, DS2API_ORACLE_VERSION
from webai_gateway.model_ids import normalize_model_body, normalize_model_id
from webai_gateway.runtime_supervisor import collect_supervisor_status
from webai_gateway.openai_api import (
    bridge_error_headers,
    build_incomplete_response_retry_payload,
    build_missing_required_tool_input_recovery_payload,
    build_native_web_search_retry_payload,
    build_off_task_question_recovery_payload,
    build_preflight_chat_response,
    build_repair_payload,
    build_required_tool_choice_recovery_payload,
    build_skill_loader_preflight_chat_response,
    build_tool_refusal_recovery_payload,
    build_unknown_tool_recovery_payload,
    build_virtual_loader_tool_recovery_payload,
    build_openai_tool_calls_sse,
    build_tool_call_sse,
    build_upstream_payload,
    parse_chat_response,
    parse_sse_text,
    post_upstream,
    should_retry_incomplete_response,
    should_retry_native_web_search_response,
    should_retry_virtual_loader_tool_recovery,
    _with_response_language_instruction,
    upstream_headers,
)
from webai_gateway.tool_controller import RetryState, classify_bridge_result, decision_to_openai_chat_response
from webai_gateway.anthropic_api import (
    _anthropic_tool_use_id,
    anthropic_body_to_openai,
    anthropic_count_tokens,
    anthropic_response_to_sse,
    openai_response_to_anthropic,
)
from webai_gateway.qwen_web import QwenWebClient, is_qwen_web_model
from webai_gateway.qwen_coder import QwenCoderClient, is_qwen_coder_model
from webai_gateway.semantic_judge import judge_bridge_semantics
from webai_gateway.tool_bridge import (
    BridgeError,
    BridgeResult,
    build_context,
    parse_tool_response,
    prefer_local_tools_for_local_agent_task,
)
from webai_gateway.web_auth import (
    DEFAULT_CDP_URL,
    BrowserLauncher,
    CredentialStore,
    DeepSeekWebAuthService,
    PROVIDERS,
    catalog_model_payloads,
    create_job,
    get_provider,
    is_credential_authorized,
    provider_payload,
    utc_now,
)


LOCAL_ADMIN_HOSTS = {"127.0.0.1", "::1", "localhost", "testclient"}
CHATGPT_AUTH_COOKIE_HINTS = (
    "__secure-next-auth.session-token",
    "oai-client-auth-info",
    "__secure-oai-is",
)
GOOGLE_AUTH_COOKIE_HINTS = (
    "__secure-1psid",
    "__secure-3psid",
    "sid",
    "hsid",
    "ssid",
    "apisid",
    "sapisid",
    "lsid",
    "osid",
)
WEBAI2API_AUTH_COOKIE_HINTS: dict[str, tuple[str, ...]] = {
    "chatgpt": CHATGPT_AUTH_COOKIE_HINTS,
    "sora": CHATGPT_AUTH_COOKIE_HINTS,
    "gemini": GOOGLE_AUTH_COOKIE_HINTS,
    "gemini-biz": GOOGLE_AUTH_COOKIE_HINTS,
    "google-flow": GOOGLE_AUTH_COOKIE_HINTS,
}
WEBAI2API_AUTH_DOMAIN_HINTS: dict[str, tuple[str, ...]] = {
    "sora": ("sora.chatgpt.com", "chatgpt.com"),
    "gemini": ("gemini.google.com", "accounts.google.com", "google.com"),
    "gemini-biz": ("gemini.google.com", "accounts.google.com", "google.com"),
    "google-flow": ("labs.google", "accounts.google.com", "google.com"),
}
HOMEPAGE_SUPPORTED_PROVIDER_IDS = {
    "deepseek-web",
    "qwen",
    "qwen-coder",
    "chatgpt",
    "google-flow",
    "sora",
    "gemini",
}
TOOL_BRIDGE_EVENT_LIMIT = 200
MEDIA_GENERATION_CACHE_LIMIT = 100
MEDIA_GENERATION_TTL_SECONDS = 60 * 60
DEFAULT_IMAGE_GENERATION_MODEL = "gpt-image-2"
DEFAULT_VIDEO_GENERATION_MODEL = "sora-2"
WEBAI2API_BROWSER_READY_RETRY_DELAYS_SECONDS = (2.0, 5.0, 10.0, 20.0)
WEBAI2API_BROWSER_NOT_READY_MARKERS = (
    "页面加载超时",
    "页面加载失败",
    "未检测到可选模型",
    "未检测到可用模型",
    "当前页面可用模型：未检测到可选模型",
    "page load timeout",
    "page.goto:",
    "ns_binding_aborted",
    "frame was detached",
    "no selectable model",
    "no selectable models",
    "model menu is empty",
)
WEBAI2API_MODEL_UNAVAILABLE_MARKERS = (
    "无法选择 ChatGPT 模型",
    "未检测到 Thinking",
    "账号不支持该模型",
    "当前页面可用模型：未检测到可选模型",
    "does not support this model",
    "no selectable model",
    "no selectable models",
)
REQUEST_DIAGNOSTIC_LIMIT = 200
QWEN_LIVE_MODEL_CATALOG_CACHE_SECONDS = 120
TOOL_CALL_REGISTRY_LIMIT = 512
OFF_TASK_QUESTION_ERROR_KINDS = {
    "off_task_environment_configuration_question",
    "off_task_scope_escalation_question",
    "optional_scope_question_without_need",
    "ask_user_question_budget_exceeded",
}
_SENSITIVE_BEARER_RE = re.compile(r"(?i)(bearer\s+)[A-Za-z0-9._~+/=-]+")
_SENSITIVE_ASSIGNMENT_RE = re.compile(
    r"(?i)((?:api[_-]?key|authorization|bearer|cookie|session|token|secret|qwen_session|ds_session_id)"
    r"[\w.-]*\s*[\"']?\s*[:=]\s*[\"']?)([^\"'\s,;}]+)"
)
_SMOKE_SENSITIVE_FIELD_RE = re.compile(r"(?i)\b(?:qwen_session|ds_session_id|hwsid|sessiontoken)\s*=\s*\[redacted\]")
_GPT_THINKING_MODEL_IDS = {"gpt-thinking", "chatgpt_text/gpt-thinking"}


class _ProviderRateLimitGate:
    def __init__(self, *, max_inflight: int, cooldown_seconds: float, max_queue: int | None = None) -> None:
        self.max_inflight = max(1, int(max_inflight))
        self.cooldown_seconds = max(0.0, float(cooldown_seconds))
        self.max_queue = None if max_queue is None else max(0, int(max_queue))
        self._condition = threading.Condition()
        self._inflight = 0
        self._waiting = 0
        self._cooldown_until = 0.0

    def acquire(self) -> bool:
        with self._condition:
            queued = False
            while True:
                now = time.monotonic()
                cooldown_wait = max(0.0, self._cooldown_until - now)
                if self._inflight < self.max_inflight and cooldown_wait <= 0:
                    self._inflight += 1
                    if queued:
                        self._waiting = max(0, self._waiting - 1)
                    return True
                if self.max_queue is not None and not queued:
                    if self._waiting >= self.max_queue:
                        return False
                    self._waiting += 1
                    queued = True
                self._condition.wait(timeout=cooldown_wait if cooldown_wait > 0 else 0.1)

    def release(self) -> None:
        with self._condition:
            self._inflight = max(0, self._inflight - 1)
            self._condition.notify_all()

    def note_rate_limit(self) -> None:
        if self.cooldown_seconds <= 0:
            return
        with self._condition:
            self._cooldown_until = max(self._cooldown_until, time.monotonic() + self.cooldown_seconds)
            self._condition.notify_all()

    def snapshot(self) -> dict[str, int]:
        with self._condition:
            return {"inflight": self._inflight, "waiting": self._waiting}


class _DirectProviderPoolGate:
    def __init__(self, *, max_inflight_per_account: int, max_queue: int, cooldown_seconds: float) -> None:
        self.max_inflight_per_account = max(1, int(max_inflight_per_account))
        self.max_queue = max(0, int(max_queue))
        self.cooldown_seconds = max(0.0, float(cooldown_seconds))
        self._condition = threading.Condition()
        self._queue: list[str] = []
        self._in_use: dict[str, int] = {}
        self._waiting = 0
        self._cooldown_until = 0.0

    def acquire(self, account_ids: list[str], *, target: str = "") -> str:
        normalized = [str(account_id) for account_id in account_ids if str(account_id).strip()]
        if not normalized:
            return ""
        queued = False
        with self._condition:
            self._sync_accounts_locked(normalized)
            while True:
                now = time.monotonic()
                cooldown_wait = max(0.0, self._cooldown_until - now)
                if cooldown_wait <= 0:
                    for account_id in self._candidate_accounts_locked(normalized, target=target):
                        if self._in_use.get(account_id, 0) < self.max_inflight_per_account:
                            self._in_use[account_id] = self._in_use.get(account_id, 0) + 1
                            self._bump_queue_locked(account_id)
                            if queued:
                                self._waiting = max(0, self._waiting - 1)
                            return account_id
                if not queued:
                    if self._waiting >= self.max_queue:
                        return ""
                    self._waiting += 1
                    queued = True
                self._condition.wait(timeout=cooldown_wait if cooldown_wait > 0 else 0.1)

    def release(self, account_id: str) -> None:
        if not account_id:
            return
        with self._condition:
            count = self._in_use.get(account_id, 0)
            if count <= 1:
                self._in_use.pop(account_id, None)
            else:
                self._in_use[account_id] = count - 1
            self._condition.notify_all()

    def note_rate_limit(self) -> None:
        if self.cooldown_seconds <= 0:
            return
        with self._condition:
            self._cooldown_until = max(self._cooldown_until, time.monotonic() + self.cooldown_seconds)
            self._condition.notify_all()

    def snapshot(self, account_ids: list[str]) -> dict[str, Any]:
        normalized = [str(account_id) for account_id in account_ids if str(account_id).strip()]
        with self._condition:
            self._sync_accounts_locked(normalized)
            in_use_accounts = sorted(account_id for account_id, count in self._in_use.items() if count > 0)
            available_accounts = [
                account_id
                for account_id in self._queue
                if self._in_use.get(account_id, 0) < self.max_inflight_per_account
            ]
            in_use_slots = sum(self._in_use.get(account_id, 0) for account_id in normalized)
            recommended = len(normalized) * self.max_inflight_per_account
            return {
                "available": len(available_accounts),
                "in_use": in_use_slots,
                "total": len(normalized),
                "available_accounts": available_accounts,
                "in_use_accounts": in_use_accounts,
                "max_inflight_per_account": self.max_inflight_per_account,
                "global_max_inflight": recommended,
                "recommended_concurrency": recommended,
                "waiting": self._waiting,
                "max_queue_size": self.max_queue,
            }

    def _sync_accounts_locked(self, account_ids: list[str]) -> None:
        current = set(account_ids)
        self._queue = [account_id for account_id in self._queue if account_id in current]
        for account_id in account_ids:
            if account_id not in self._queue:
                self._queue.append(account_id)
        for account_id in list(self._in_use):
            if account_id not in current:
                self._in_use.pop(account_id, None)

    def _candidate_accounts_locked(self, account_ids: list[str], *, target: str) -> list[str]:
        if target:
            return [target] if target in account_ids else []
        return [account_id for account_id in self._queue if account_id in account_ids]

    def _bump_queue_locked(self, account_id: str) -> None:
        if account_id not in self._queue:
            return
        self._queue = [item for item in self._queue if item != account_id]
        self._queue.append(account_id)


def create_app(
    *,
    config: GatewayConfig | None = None,
    config_path: str | Path = "config.json",
    http_client: httpx.Client | None = None,
    credential_store: CredentialStore | None = None,
    web_auth_service: Any | None = None,
    browser_launcher: BrowserLauncher | None = None,
    deepseek_client_factory: Any | None = None,
    qwen_client_factory: Any | None = None,
    qwen_coder_client_factory: Any | None = None,
    native_ui_dir: str | Path | None = None,
    auto_research_fixture_dir: str | Path | None = None,
    webai2api_sidecar_starter: Any | None = None,
    run_auth_jobs_inline: bool = False,
) -> FastAPI:
    config_file = Path(config_path)
    cfg = config or load_config(config_file)
    client = http_client or httpx.Client(timeout=120, trust_env=False)
    app = FastAPI(title="WebAI Gateway", version="0.1.0")
    app.state.config = cfg
    app.state.credential_store = credential_store or CredentialStore(config_file.parent / "credentials")
    app.state.account_registry = AccountRegistry(config_file.parent / ".webai-gateway" / "accounts.json")
    app.state.web_auth_service = web_auth_service or DeepSeekWebAuthService()
    app.state.browser_launcher = browser_launcher or BrowserLauncher(config_file.parent / ".webai-gateway" / "chrome-auth-profile")
    app.state.deepseek_client_factory = deepseek_client_factory or DeepSeekWebClient
    app.state.qwen_client_factory = qwen_client_factory or QwenWebClient
    app.state.qwen_coder_client_factory = qwen_coder_client_factory or QwenCoderClient
    app.state.web_auth_jobs = {}
    app.state.tool_bridge_events = deque(maxlen=TOOL_BRIDGE_EVENT_LIMIT)
    app.state.request_diagnostics = deque(maxlen=REQUEST_DIAGNOSTIC_LIMIT)
    app.state.request_diagnostic_sequence = 0
    app.state.request_diagnostic_lock = threading.Lock()
    app.state.qwen_live_model_catalog_cache = {}
    app.state.tool_call_registry = OrderedDict()
    app.state.media_generations = OrderedDict()
    app.state.auto_research_fixture_dir = Path(auto_research_fixture_dir) if auto_research_fixture_dir is not None else _default_auto_research_fixture_dir()
    app.state.gateway_root = config_file.parent.resolve()
    app.state.webai2api_sidecar_starter = webai2api_sidecar_starter
    app.state.runtime_started_at = utc_now()
    app.state.runtime_started_epoch = datetime.now(timezone.utc).timestamp()
    app.state.deepseek_ds2api_bearer_gate = None
    app.state.deepseek_ds2api_bearer_gate_config = None
    app.state.deepseek_ds2api_bearer_gate_lock = threading.Lock()

    static_dir = Path(__file__).with_name("static")
    webai2api_ui_dir = Path(native_ui_dir) if native_ui_dir is not None else _default_native_ui_dir()
    if static_dir.exists():
        app.mount("/static", StaticFiles(directory=static_dir), name="static")

    def current_config() -> GatewayConfig:
        return app.state.config

    def load_gateway_models_with_direct_catalog(cfg: GatewayConfig) -> dict[str, Any]:
        return _append_qwen_live_model_catalog(app, client, _load_gateway_models(client, cfg), cfg)

    def auth_value_matches(value: str | None, expected: str) -> bool:
        if value is None:
            return False
        candidate = str(value).strip()
        if not candidate:
            return False
        for _ in range(3):
            if secrets.compare_digest(candidate, expected):
                return True
            parts = candidate.split(None, 1)
            if len(parts) != 2 or parts[0].lower() != "bearer":
                return False
            candidate = parts[1].strip()
        return secrets.compare_digest(candidate, expected)

    def require_auth(authorization: str | None, x_api_key: str | None = None, api_key: str | None = None) -> None:
        cfg = current_config()
        if not cfg.server.api_key:
            return
        expected = cfg.server.api_key
        if (
            auth_value_matches(authorization, expected)
            or auth_value_matches(x_api_key, expected)
            or auth_value_matches(api_key, expected)
        ):
            return
        raise HTTPException(status_code=401, detail="Unauthorized")

    def require_local_admin(request: Request) -> None:
        host = request.client.host if request.client else ""
        if host not in LOCAL_ADMIN_HOSTS:
            raise HTTPException(status_code=403, detail="Admin UI is only available from localhost")

    @app.get("/", include_in_schema=False)
    def admin_ui(request: Request) -> FileResponse:
        require_local_admin(request)
        index_path = _native_index_path(webai2api_ui_dir) or (static_dir / "index.html")
        if not index_path.exists():
            raise HTTPException(status_code=404, detail="Admin UI assets are missing")
        return _native_file_response(index_path)

    @app.get("/favicon.png", include_in_schema=False)
    def native_favicon(request: Request) -> FileResponse:
        require_local_admin(request)
        return _safe_native_file(webai2api_ui_dir, "favicon.png")

    @app.get("/assets/{asset_path:path}", include_in_schema=False)
    def native_assets(asset_path: str, request: Request) -> FileResponse:
        require_local_admin(request)
        return _safe_native_file(webai2api_ui_dir / "assets", asset_path)

    @app.get("/health")
    def health() -> dict[str, Any]:
        cfg = current_config()
        return {"ok": True, "config": config_to_public(cfg), "runtime": _runtime_status(app)}

    @app.get("/api/admin/config")
    def admin_config(request: Request) -> dict[str, Any]:
        require_local_admin(request)
        return config_to_admin(current_config())

    @app.get("/api/admin/tool-bridge-events")
    @app.get("/api/admin/tool-bridge/events")
    def admin_tool_bridge_events(request: Request) -> dict[str, Any]:
        require_local_admin(request)
        return {"events": list(app.state.tool_bridge_events)}

    @app.get("/api/admin/request-diagnostics")
    def admin_request_diagnostics(request: Request) -> dict[str, Any]:
        require_local_admin(request)
        return {"events": list(app.state.request_diagnostics)}

    @app.get("/api/admin/auto-research/status")
    def admin_auto_research_status(request: Request) -> dict[str, Any]:
        require_local_admin(request)
        return build_auto_research_status(app.state.auto_research_fixture_dir)

    @app.get("/api/admin/auto-research/candidates")
    def admin_auto_research_candidates(request: Request) -> dict[str, Any]:
        require_local_admin(request)
        return _auto_research_candidate_report(app)

    @app.get("/api/admin/onboarding")
    def admin_onboarding(request: Request) -> dict[str, Any]:
        require_local_admin(request)
        cfg = current_config()
        models_payload = load_gateway_models_with_direct_catalog(cfg)
        models = models_payload.get("data") if isinstance(models_payload.get("data"), list) else []
        provider_data = provider_payload(app.state.credential_store)["providers"]
        webai2api_instances = _load_webai2api_instances(client, cfg)
        models = _with_onboarding_provider_catalog_models(models, provider_data, webai2api_instances)
        model_ids = {item.get("id") for item in models if isinstance(item, dict)}
        webai2api_auth = _load_webai2api_auth_states(client, cfg, provider_data, webai2api_instances)
        providers: list[dict[str, Any]] = []
        for provider in provider_data:
            provider_models = _available_models_for_provider(provider, models, model_ids)
            account_state = _provider_account_state(provider, provider_models, webai2api_instances)
            provider_models = account_state["availableModels"]
            is_direct = provider.get("route") == "direct"
            provider_auth = webai2api_auth.get(str(provider.get("id") or ""), {"checked": False, "authorized": False})
            credential = provider.get("credential") if isinstance(provider.get("credential"), dict) else {}
            if provider_auth.get("authorized"):
                credential = {
                    **credential,
                    "authorized": True,
                    "fields": {**(credential.get("fields") if isinstance(credential.get("fields"), dict) else {}), "cookie": True},
                }
            providers.append(
                {
                    **provider,
                    "credential": credential,
                    "webAI2APIAuth": provider_auth,
                    "availableModels": provider_models,
                    "modelCount": len(provider_models),
                    "accounts": account_state["accounts"],
                    "currentAccountId": account_state["currentAccountId"],
                    "modelAvailability": account_state["modelAvailability"],
                    "loginKind": "direct" if is_direct else "webai2api",
                }
            )
        include_candidates = _include_onboarding_candidates(request)
        visible_providers = providers if include_candidates else [
            provider for provider in providers if _onboarding_provider_visible_on_homepage(provider)
        ]
        visible_model_ids = _onboarding_visible_model_ids(visible_providers)
        visible_models = models if include_candidates else _filter_onboarding_models(models, visible_model_ids)
        connection_profiles = _onboarding_connection_profiles(visible_providers, cfg)
        return {
            "gateway": {
                "baseUrl": "/v1",
                "apiKey": cfg.server.api_key,
                "defaultModel": cfg.upstream.model,
                "toolMode": cfg.upstream.tool_mode,
                "toolBridge": {
                    "mode": cfg.tool_bridge.mode,
                    "maxToolsInPrompt": cfg.tool_bridge.max_tools_in_prompt,
                    "exposurePolicy": cfg.tool_bridge.exposure_policy,
                    "statusText": "工具适配层：严格模式已启用" if cfg.tool_bridge.mode == "strict" else "工具适配层：兼容模式",
                },
                "upstreamBaseUrl": cfg.upstream.base_url,
            },
            "summary": {
                "providers": len(visible_providers),
                "models": len(visible_models),
                "authorizedProviders": _onboarding_authorized_provider_count(visible_providers),
                "authorizedDirectProviders": _onboarding_authorized_direct_provider_count(visible_providers),
                "webAI2APIProviders": _onboarding_webai2api_provider_count(visible_providers),
                "candidateProviders": len(providers),
                "candidateModels": len(models),
                "hiddenCandidateProviders": max(len(providers) - len(visible_providers), 0),
            },
            "providers": visible_providers,
            "models": visible_models,
            "connectionProfiles": connection_profiles,
            "recommendedConnectionProfile": _recommended_connection_profile(connection_profiles),
        }


    def _include_onboarding_candidates(request: Request) -> bool:
        value = str(
            request.query_params.get("includeCandidates")
            or request.query_params.get("include_candidates")
            or request.query_params.get("scope")
            or ""
        ).strip().lower()
        return value in {"1", "true", "yes", "all", "candidates"}

    def _onboarding_provider_visible_on_homepage(provider: dict[str, Any]) -> bool:
        provider_id = str(provider.get("id") or "")
        if provider_id in HOMEPAGE_SUPPORTED_PROVIDER_IDS:
            return True
        credential = provider.get("credential") if isinstance(provider.get("credential"), dict) else {}
        accounts = provider.get("accounts") if isinstance(provider.get("accounts"), list) else []
        has_authorized_account = bool(credential.get("authorized")) or any(
            isinstance(account, dict) and bool(account.get("authorized")) for account in accounts
        )
        return has_authorized_account and bool(_onboarding_visible_model_ids([provider]))

    def _onboarding_visible_model_ids(providers: list[dict[str, Any]]) -> set[str]:
        model_ids: set[str] = set()
        for provider in providers:
            available_models = provider.get("availableModels") if isinstance(provider.get("availableModels"), list) else []
            model_availability = provider.get("modelAvailability") if isinstance(provider.get("modelAvailability"), dict) else {}
            for model_id in available_models:
                if isinstance(model_id, str) and model_id.strip():
                    model_ids.add(model_id)
            for model_id, availability in model_availability.items():
                if not isinstance(model_id, str) or not isinstance(availability, dict):
                    continue
                if availability.get("status") == "available":
                    model_ids.add(model_id)
        return model_ids

    def _filter_onboarding_models(models: list[Any], visible_model_ids: set[str]) -> list[Any]:
        if not visible_model_ids:
            return []
        return [
            item
            for item in models
            if isinstance(item, dict) and isinstance(item.get("id"), str) and item.get("id") in visible_model_ids
        ]

    def _onboarding_authorized_provider_count(providers: list[dict[str, Any]]) -> int:
        return sum(
            1
            for provider in providers
            if isinstance(provider.get("credential"), dict) and bool(provider["credential"].get("authorized"))
        )

    def _onboarding_authorized_direct_provider_count(providers: list[dict[str, Any]]) -> int:
        return sum(
            1
            for provider in providers
            if provider.get("route") == "direct"
            and isinstance(provider.get("credential"), dict)
            and bool(provider["credential"].get("authorized"))
        )

    def _onboarding_webai2api_provider_count(providers: list[dict[str, Any]]) -> int:
        return sum(1 for provider in providers if provider.get("route") != "direct")


    def _with_onboarding_provider_catalog_models(
        models: list[Any],
        providers: list[dict[str, Any]],
        webai2api_instances: list[Any],
    ) -> list[Any]:
        out = [dict(item) if isinstance(item, dict) else item for item in models]
        seen = {item.get("id") for item in out if isinstance(item, dict)}
        for provider in providers:
            if not _should_add_provider_catalog_models(provider, webai2api_instances):
                continue
            for model_id in _provider_catalog_model_ids_for_onboarding(provider, seen):
                if model_id in seen:
                    continue
                seen.add(model_id)
                out.append(_provider_catalog_model_payload(provider, model_id))
        return out

    def _should_add_provider_catalog_models(provider: dict[str, Any], webai2api_instances: list[Any]) -> bool:
        provider_id = str(provider.get("id") or "")
        if provider_id == "chatgpt":
            return bool(_webai2api_provider_instances(provider, webai2api_instances))
        if provider.get("route") != "direct":
            return bool(provider.get("advertiseModels")) and bool(_webai2api_provider_instances(provider, webai2api_instances))
        credential = provider.get("credential") if isinstance(provider.get("credential"), dict) else {}
        return (
            provider.get("route") == "direct"
            and bool(provider.get("advertiseModels"))
            and bool(credential.get("authorized"))
        )

    def _provider_catalog_model_ids_for_onboarding(provider: dict[str, Any], seen: set[Any]) -> list[str]:
        provider_id = str(provider.get("id") or "")
        declared_models = provider.get("models") if isinstance(provider.get("models"), list) else []
        if provider_id == "chatgpt":
            adapter_ids = [str(adapter) for adapter in provider.get("adapters", []) if isinstance(adapter, str)]
            return [
                model_id
                for model_id in declared_models
                if isinstance(model_id, str) and model_id in {"gpt-instant", "gpt-thinking", "gpt-pro"}
                and not any(f"{adapter}/{model_id}" in seen for adapter in adapter_ids)
            ]
        if provider.get("route") != "direct":
            adapter_ids = [str(adapter) for adapter in provider.get("adapters", []) if isinstance(adapter, str)]
            return [
                model_id
                for model_id in declared_models
                if isinstance(model_id, str) and not any(f"{adapter}/{model_id}" in seen for adapter in adapter_ids)
            ]
        if provider.get("route") == "direct":
            return [model_id for model_id in declared_models if isinstance(model_id, str)]
        return []

    def _provider_catalog_model_payload(provider: dict[str, Any], model_id: str) -> dict[str, Any]:
        provider_id = str(provider.get("id") or "")
        model_type = _infer_model_type(model_id, provider_id, None)
        is_media_model = model_type in {"image", "video"}
        payload = {
            "id": model_id,
            "object": "model",
            "owned_by": provider_id,
            "capabilities": {
                "tool_bridge": False
                if is_media_model
                else str(provider.get("toolBridge") or "strict").strip().lower() != "off",
                "supports_native_tools": False if is_media_model else bool(provider.get("supportsNativeTools")),
                "preferred_protocol": str(provider.get("preferredProtocol") or "openai"),
            },
            "availability_source": "provider_catalog",
        }
        if model_type:
            payload["type"] = model_type
        return _enrich_model_payload(payload)

    def _available_models_for_provider(
        provider: dict[str, Any],
        models: list[Any],
        model_ids: set[Any],
    ) -> list[str]:
        declared_models = provider.get("availableModels")
        if not isinstance(declared_models, list):
            declared_models = provider.get("models", [])
        declared_model_ids = [model_id for model_id in declared_models if isinstance(model_id, str)]
        provider_models: list[str] = []
        seen: set[str] = set()

        def add_model(model_id: str) -> None:
            if model_id not in seen:
                seen.add(model_id)
                provider_models.append(model_id)

        for model_id in declared_model_ids:
            if model_id in model_ids:
                add_model(model_id)

        if provider.get("route") == "direct":
            provider_id = str(provider.get("id") or "")
            for item in models:
                if not isinstance(item, dict):
                    continue
                model_id = item.get("id")
                if not isinstance(model_id, str):
                    continue
                owner = str(item.get("owned_by") or "")
                if _direct_catalog_model_belongs_to_provider(provider_id, model_id, owner):
                    add_model(model_id)
            return provider_models

        provider_id = str(provider.get("id") or "")
        adapter_ids = {str(adapter) for adapter in provider.get("adapters", []) if isinstance(adapter, str)}
        adapter_prefixes = tuple(f"{adapter}/" for adapter in adapter_ids)
        for item in models:
            if not isinstance(item, dict):
                continue
            model_id = item.get("id")
            if not isinstance(model_id, str):
                continue
            owner = str(item.get("owned_by") or "")
            if owner == provider_id or owner in adapter_ids or model_id.startswith(adapter_prefixes):
                add_model(model_id)
        return provider_models

    def _load_webai2api_instances(http_client: httpx.Client, cfg: GatewayConfig) -> list[Any]:
        try:
            response = http_client.get(
                f"{_sidecar_root_from_base_url(cfg.upstream.base_url)}/admin/config/instances",
                headers=upstream_headers(cfg),
            )
            if response.status_code != 200:
                return []
            instances = response.json()
        except Exception:
            return []
        return instances if isinstance(instances, list) else []

    def _load_webai2api_auth_states(
        http_client: httpx.Client,
        cfg: GatewayConfig,
        providers: list[dict[str, Any]],
        instances: list[Any] | None = None,
    ) -> dict[str, dict[str, Any]]:
        if instances is None:
            instances = _load_webai2api_instances(http_client, cfg)
        if not isinstance(instances, list):
            return {}
        states: dict[str, dict[str, Any]] = {}
        for provider in providers:
            if provider.get("route") == "direct":
                continue
            provider_id = str(provider.get("id") or "")
            matching_instances = _webai2api_provider_instances(provider, instances)
            if not matching_instances:
                continue
            cookie_count = 0
            authorized = False
            for instance_name in matching_instances:
                for domain in _webai2api_auth_domains(provider):
                    cookie_names = _load_webai2api_cookie_names(http_client, cfg, instance_name, domain)
                    if cookie_names is None:
                        continue
                    cookie_count += len(cookie_names)
                    if _webai2api_cookies_authorized(provider_id, cookie_names):
                        authorized = True
            if not authorized and cookie_count > 0 and provider_id not in WEBAI2API_AUTH_COOKIE_HINTS:
                authorized = True
            states[provider_id] = {
                "checked": True,
                "authorized": authorized,
                "cookieCount": cookie_count,
                "instances": matching_instances,
            }
        return states

    def _webai2api_provider_instances(provider: dict[str, Any], instances: list[Any]) -> list[str]:
        adapter_ids = {str(adapter) for adapter in provider.get("adapters", []) if isinstance(adapter, str)}
        matching: list[str] = []
        for instance in instances:
            if not isinstance(instance, dict):
                continue
            instance_name = str(instance.get("name") or "")
            if not instance_name:
                continue
            workers = instance.get("workers") if isinstance(instance.get("workers"), list) else []
            for worker in workers:
                if not isinstance(worker, dict):
                    continue
                worker_type = str(worker.get("type") or "")
                merge_types = {str(item) for item in worker.get("mergeTypes", []) if isinstance(item, str)}
                if worker_type in adapter_ids or adapter_ids.intersection(merge_types):
                    matching.append(instance_name)
                    break
        return list(dict.fromkeys(matching))

    def _webai2api_auth_domains(provider: dict[str, Any]) -> list[str]:
        provider_id = str(provider.get("id") or "")
        login_url = str(provider.get("loginUrl") or "")
        parsed = urlsplit(login_url)
        host = parsed.netloc or parsed.path
        domains = [host.split("@")[-1].split(":")[0]] if host else []
        domains.extend(WEBAI2API_AUTH_DOMAIN_HINTS.get(provider_id, ()))
        return [domain for domain in dict.fromkeys(domains) if domain]

    def _load_webai2api_cookie_names(
        http_client: httpx.Client,
        cfg: GatewayConfig,
        instance_name: str,
        domain: str,
    ) -> list[str] | None:
        try:
            response = http_client.get(
                f"{cfg.upstream.base_url.rstrip('/')}/cookies",
                params={"name": instance_name, "domain": domain},
                headers=upstream_headers(cfg),
            )
            if response.status_code != 200:
                return None
            payload = response.json()
        except Exception:
            return None
        cookies = payload.get("cookies") if isinstance(payload, dict) else None
        if not isinstance(cookies, list):
            return []
        names: list[str] = []
        for item in cookies:
            if isinstance(item, dict) and item.get("name"):
                names.append(str(item["name"]))
        return names

    def _webai2api_cookies_authorized(provider_id: str, cookie_names: list[str]) -> bool:
        hints = WEBAI2API_AUTH_COOKIE_HINTS.get(provider_id)
        if not hints:
            return False
        lowered = [name.lower() for name in cookie_names]
        return any(name == hint or name.startswith(f"{hint}.") for name in lowered for hint in hints)

    def _provider_account_state(
        provider: dict[str, Any],
        provider_models: list[str],
        webai2api_instances: list[Any],
    ) -> dict[str, Any]:
        provider_id = str(provider.get("id") or "")
        accounts = _provider_accounts(provider, provider_models, webai2api_instances)
        current = app.state.account_registry.current_account_id(provider_id)
        account_ids = {account["id"] for account in accounts}
        if current not in account_ids:
            current = next((account["id"] for account in accounts if account.get("authorized")), accounts[0]["id"] if accounts else "")
        model_availability = _model_availability_for_account(current, provider_models)
        current_validation = _public_account_validation(app.state.account_registry.validation_for(current)) if current else {}
        has_current_validation = any(model_id in current_validation for model_id in provider_models)
        available_models = [
            model_id
            for model_id in provider_models
            if model_availability.get(model_id, {}).get("status") != "unavailable"
            and (not has_current_validation or model_availability.get(model_id, {}).get("status") == "available")
        ]
        for account in accounts:
            account["current"] = account["id"] == current
            account["availableModelCount"] = _available_model_count_for_account(account["id"], provider_models)
        return {
            "accounts": accounts,
            "currentAccountId": current,
            "availableModels": available_models,
            "modelAvailability": model_availability,
        }

    def _provider_accounts(
        provider: dict[str, Any],
        provider_models: list[str],
        webai2api_instances: list[Any],
    ) -> list[dict[str, Any]]:
        if provider.get("route") == "direct":
            return _direct_provider_accounts(provider, provider_models)
        return _webai2api_provider_accounts(provider, provider_models, webai2api_instances)

    def _direct_provider_accounts(provider: dict[str, Any], provider_models: list[str]) -> list[dict[str, Any]]:
        provider_id = str(provider.get("id") or "")
        accounts: list[dict[str, Any]] = []
        for account_id in _direct_account_ids_for_provider(app, provider_id):
            credential = _direct_credential_for_account(app, provider_id, account_id)
            metadata = app.state.account_registry.metadata(account_id)
            validation = _public_account_validation(metadata.get("validation"))
            authorized = is_credential_authorized(provider_id, credential)
            if not authorized and not metadata:
                continue
            profile_id = _direct_profile_id_from_account_id(provider_id, account_id) or "default"
            accounts.append(
                {
                    "id": account_id,
                    "providerId": provider_id,
                    "source": "direct-profile",
                    "displayName": str(metadata.get("displayName") or ("默认账号" if profile_id == "default" else profile_id)),
                    "planType": str(metadata.get("planType") or "unknown"),
                    "authorized": authorized,
                    "current": False,
                    "instanceName": None,
                    "workerName": None,
                    "workerType": None,
                    "availableModelCount": _available_model_count_from_validation(provider_models, validation),
                    "lastValidatedAt": metadata.get("lastValidatedAt") if isinstance(metadata.get("lastValidatedAt"), str) else None,
                    "validation": validation,
                }
            )
        return accounts

    def _webai2api_provider_accounts(
        provider: dict[str, Any],
        provider_models: list[str],
        instances: list[Any],
    ) -> list[dict[str, Any]]:
        provider_id = str(provider.get("id") or "")
        accounts: list[dict[str, Any]] = []
        for instance in instances:
            if not isinstance(instance, dict):
                continue
            instance_name = str(instance.get("name") or "")
            if not instance_name:
                continue
            user_data_mark = str(instance.get("userDataMark") or "").strip()
            authorized = _webai2api_instance_authorized(provider, instance_name)
            workers = instance.get("workers") if isinstance(instance.get("workers"), list) else []
            for worker in workers:
                if not isinstance(worker, dict) or not _worker_supports_provider(worker, provider):
                    continue
                worker_name = str(worker.get("name") or "")
                if not worker_name:
                    continue
                account_id = webai2api_account_id(provider_id, instance_name, worker_name)
                metadata = app.state.account_registry.metadata(account_id)
                validation = _public_account_validation(metadata.get("validation"))
                fallback_name = f"{user_data_mark or instance_name} / {worker_name}"
                accounts.append(
                    {
                        "id": account_id,
                        "providerId": provider_id,
                        "source": "webai2api-worker",
                        "displayName": str(metadata.get("displayName") or fallback_name),
                        "planType": str(metadata.get("planType") or "unknown"),
                        "authorized": authorized,
                        "current": False,
                        "instanceName": instance_name,
                        "workerName": worker_name,
                        "workerType": str(worker.get("type") or ""),
                        "availableModelCount": _available_model_count_from_validation(provider_models, validation),
                        "lastValidatedAt": metadata.get("lastValidatedAt") if isinstance(metadata.get("lastValidatedAt"), str) else None,
                        "validation": validation,
                    }
                )
        return accounts

    def _webai2api_instance_authorized(provider: dict[str, Any], instance_name: str) -> bool:
        provider_id = str(provider.get("id") or "")
        cookie_count = 0
        for domain in _webai2api_auth_domains(provider):
            cookie_names = _load_webai2api_cookie_names(client, current_config(), instance_name, domain)
            if cookie_names is None:
                continue
            cookie_count += len(cookie_names)
            if _webai2api_cookies_authorized(provider_id, cookie_names):
                return True
        return cookie_count > 0 and provider_id not in WEBAI2API_AUTH_COOKIE_HINTS

    def _worker_supports_provider(worker: dict[str, Any], provider: dict[str, Any]) -> bool:
        adapter_ids = {str(adapter) for adapter in provider.get("adapters", []) if isinstance(adapter, str)}
        worker_type = str(worker.get("type") or "")
        merge_types = {str(item) for item in worker.get("mergeTypes", []) if isinstance(item, str)}
        return worker_type in adapter_ids or bool(adapter_ids.intersection(merge_types))

    def _public_account_validation(raw: Any) -> dict[str, Any]:
        if not isinstance(raw, dict):
            return {}
        out: dict[str, Any] = {}
        for model_id, item in raw.items():
            if not isinstance(item, dict):
                continue
            status = str(item.get("status") or "pending")
            if status not in {"available", "unavailable", "pending"}:
                status = "pending"
            entry: dict[str, Any] = {
                "status": status,
                "ok": status == "available",
            }
            if isinstance(item.get("message"), str):
                entry["message"] = item["message"]
            if isinstance(item.get("checkedAt"), str):
                entry["checkedAt"] = item["checkedAt"]
            out[str(model_id)] = entry
        return out

    def _model_availability_for_account(account_id: str, provider_models: list[str]) -> dict[str, dict[str, Any]]:
        validation = app.state.account_registry.validation_for(account_id) if account_id else {}
        availability: dict[str, dict[str, Any]] = {}
        for model_id in provider_models:
            item = validation.get(model_id) if isinstance(validation, dict) else None
            if isinstance(item, dict):
                public = _public_account_validation({model_id: item}).get(model_id)
                availability[model_id] = public or {"status": "pending", "ok": False}
            else:
                availability[model_id] = {"status": "pending", "ok": False}
        return availability

    def _available_model_count_for_account(account_id: str, provider_models: list[str]) -> int:
        validation = _public_account_validation(app.state.account_registry.validation_for(account_id))
        return _available_model_count_from_validation(provider_models, validation)

    def _available_model_count_from_validation(provider_models: list[str], validation: dict[str, Any]) -> int:
        return sum(1 for model_id in provider_models if validation.get(model_id, {}).get("status") != "unavailable")

    @app.get("/api/admin/accounts")
    def admin_accounts(request: Request, providerId: str | None = None) -> dict[str, Any]:
        require_local_admin(request)
        cfg = current_config()
        provider_data = provider_payload(app.state.credential_store)["providers"]
        models_payload = load_gateway_models_with_direct_catalog(cfg)
        models = models_payload.get("data") if isinstance(models_payload.get("data"), list) else []
        model_ids = {item.get("id") for item in models if isinstance(item, dict)}
        instances = _load_webai2api_instances(client, cfg)
        providers = []
        for provider in provider_data:
            if providerId and str(provider.get("id") or "") != providerId:
                continue
            provider_models = _available_models_for_provider(provider, models, model_ids)
            state = _provider_account_state(provider, provider_models, instances)
            providers.append(
                {
                    "providerId": str(provider.get("id") or ""),
                    "name": provider.get("name"),
                    **state,
                }
            )
        if providerId:
            if not providers:
                raise HTTPException(status_code=404, detail="Provider 不存在")
            return providers[0]
        return {"providers": providers}

    @app.get("/api/admin/providers/{provider_id}/queue/status")
    def admin_provider_queue_status(provider_id: str, request: Request) -> dict[str, Any]:
        require_local_admin(request)
        provider = get_provider(provider_id)
        if provider.route != "direct":
            raise HTTPException(status_code=400, detail="该 Provider 不使用 Gateway direct 队列")
        cfg = current_config()
        accounts = _authorized_direct_account_ids(app, provider_id)
        route = "qwen-coder" if provider_id == "qwen-coder" else "qwen-web" if provider_id == "qwen" else provider_id
        gate = _qwen_direct_gate(app, cfg, route)
        return gate.snapshot(accounts)

    @app.post("/api/admin/accounts/select")
    async def admin_accounts_select(request: Request) -> dict[str, Any]:
        require_local_admin(request)
        body = await _json_body(request)
        provider_id = str(body.get("providerId") or body.get("provider_id") or "").strip()
        account_id = str(body.get("accountId") or body.get("account_id") or "").strip()
        provider = get_provider(provider_id)
        parsed = parse_account_id(account_id)
        if not parsed or parsed.provider_id != provider_id:
            raise HTTPException(status_code=400, detail="账号不属于当前 Provider")
        if provider.route != "direct":
            _sync_webai2api_selected_account(provider, parsed)
        app.state.account_registry.set_current(provider_id, account_id)
        return admin_accounts(request, providerId=provider_id)

    @app.post("/api/admin/accounts/validate")
    async def admin_accounts_validate(request: Request) -> dict[str, Any]:
        require_local_admin(request)
        body = await _json_body(request)
        provider_id = str(body.get("providerId") or body.get("provider_id") or "").strip()
        account_id = str(body.get("accountId") or body.get("account_id") or "").strip()
        provider = get_provider(provider_id)
        parsed = parse_account_id(account_id)
        if not parsed or parsed.provider_id != provider_id:
            raise HTTPException(status_code=400, detail="账号不属于当前 Provider")
        model_ids = body.get("modelIds") if isinstance(body.get("modelIds"), list) else body.get("model_ids")
        if not isinstance(model_ids, list):
            model_ids = []
        force_validation = bool(body.get("force") or body.get("refresh") or body.get("ignoreCache"))
        cfg = current_config()
        instances = _load_webai2api_instances(client, cfg)
        if provider.route != "direct":
            if not _webai2api_selected_account_already_active(provider, parsed, instances):
                _sync_webai2api_selected_account(provider, parsed)
                if not _wait_for_webai2api_api_mode(client, cfg):
                    raise HTTPException(status_code=502, detail="WebAI2API 正在重启，请稍后再检测模型")
                instances = _load_webai2api_instances(client, cfg)
            app.state.account_registry.set_current(provider.id, account_id)
        models_payload = load_gateway_models_with_direct_catalog(cfg)
        models = models_payload.get("data") if isinstance(models_payload.get("data"), list) else []
        provider_payloads = provider_payload(app.state.credential_store)["providers"]
        models = _with_onboarding_provider_catalog_models(models, provider_payloads, instances)
        model_ids_known = {item.get("id") for item in models if isinstance(item, dict)}
        provider_dict = next((item for item in provider_payloads if item.get("id") == provider_id), None)
        if not provider_dict:
            raise HTTPException(status_code=404, detail="Provider 不存在")
        provider_models = _available_models_for_provider(provider_dict, models, model_ids_known)
        requested = [str(model_id) for model_id in model_ids if isinstance(model_id, str) and model_id in provider_models]
        if not requested:
            requested = provider_models
        validation: dict[str, Any] = {}
        for model_id in requested:
            if not force_validation:
                cached = app.state.account_registry.fresh_validation(account_id, model_id)
                if cached is not None:
                    validation[model_id] = _public_account_validation({model_id: cached})[model_id]
                    continue
            result = await run_in_threadpool(_validate_account_model, cfg, provider_id, model_id, provider_dict, parsed)
            saved = app.state.account_registry.save_validation(account_id, model_id, result)
            validation[model_id] = _public_account_validation({model_id: saved})[model_id]
        return {
            "providerId": provider.id,
            "accountId": account_id,
            "validation": validation,
        }

    @app.patch("/api/admin/accounts/{account_id:path}")
    async def admin_accounts_update(account_id: str, request: Request) -> dict[str, Any]:
        require_local_admin(request)
        body = await _json_body(request)
        provider_id = str(body.get("providerId") or body.get("provider_id") or "").strip()
        parsed = parse_account_id(account_id)
        if not parsed or (provider_id and parsed.provider_id != provider_id):
            raise HTTPException(status_code=400, detail="账号不属于当前 Provider")
        app.state.account_registry.update_metadata(account_id, body)
        return admin_accounts(request, providerId=parsed.provider_id)

    async def _start_webai2api_login_mode(provider_id: str, body: dict[str, Any]) -> dict[str, Any]:
        provider = get_provider(provider_id)
        if provider.route == "direct":
            raise HTTPException(status_code=400, detail="该 Provider 使用 Gateway 直连授权，不需要 WebAI2API 登录模式")
        cfg = current_config()
        sidecar_ready = await run_in_threadpool(_ensure_webai2api_sidecar_available, app, client, cfg)
        if not sidecar_ready.get("available"):
            raise HTTPException(
                status_code=502,
                detail=sidecar_ready.get("message") or "WebAI2API sidecar 未启动或无法连接",
            )
        instances = _load_webai2api_instances(client, cfg)
        if not instances:
            raise HTTPException(status_code=502, detail="WebAI2API 工作池为空或不可读取，无法启动登录模式")
        requested_worker = str(body.get("workerName") or body.get("worker_name") or "").strip()
        create_account_raw = body.get("newAccount", body.get("new_account", None))
        create_account = bool(create_account_raw) if create_account_raw is not None else not requested_worker
        existing_worker = "" if requested_worker else _first_webai2api_worker_for_provider(
            provider,
            instances,
            require_dedicated_profile=True,
        )
        if not create_account and not requested_worker and not existing_worker:
            create_account = True
        if create_account:
            instances, instance_name, worker_name = _webai2api_instances_with_new_login_account(provider, instances)
            response = client.post(
                f"{_sidecar_root_from_base_url(cfg.upstream.base_url)}/admin/config/instances",
                json=instances,
                headers=upstream_headers(cfg),
            )
            if response.status_code >= 400:
                raise HTTPException(status_code=502, detail=_upstream_http_error_message(response))
            account_id = webai2api_account_id(provider.id, instance_name, worker_name)
            app.state.account_registry.update_metadata(
                account_id,
                {
                    "providerId": provider.id,
                    "displayName": f"{provider.name} 新账号",
                    "planType": "unknown",
                },
            )
        else:
            worker_name = requested_worker or existing_worker
            if not worker_name:
                raise HTTPException(status_code=400, detail="未找到可用于该 Provider 的 WebAI2API Worker")
            instance_name = _instance_name_for_worker(instances, worker_name)
            account_id = webai2api_account_id(provider.id, instance_name, worker_name) if instance_name else ""

        restart_response = client.post(
            f"{_sidecar_root_from_base_url(cfg.upstream.base_url)}/admin/restart",
            json={"loginMode": True, "workerName": worker_name},
            headers=upstream_headers(cfg),
        )
        if restart_response.status_code >= 400:
            raise HTTPException(status_code=502, detail=_upstream_http_error_message(restart_response))
        app.state.webai2api_login_mode_started_at = time.time()
        return {
            "success": True,
            "providerId": provider.id,
            "instanceName": instance_name,
            "workerName": worker_name,
            "accountId": account_id,
            "newAccount": create_account,
            "sidecarStarted": bool(sidecar_ready.get("started")),
            "sidecarPid": sidecar_ready.get("pid"),
            "loginKind": "webai2api",
            "openMode": "runtime_browser",
            "browserLabel": "Camoufox",
            "displayUrl": None,
            "actionLabel": "打开网页登录授权",
            "message": (
                f"已为 {provider.name} 创建独立浏览器 Profile 并进入登录模式"
                if create_account
                else f"已进入 {worker_name} 的登录模式"
            ),
        }

    async def _finish_webai2api_login_mode() -> dict[str, Any]:
        cfg = current_config()
        sidecar_ready = await run_in_threadpool(_ensure_webai2api_sidecar_available, app, client, cfg)
        if not sidecar_ready.get("available"):
            raise HTTPException(
                status_code=502,
                detail=sidecar_ready.get("message") or "WebAI2API sidecar 未启动或无法连接",
            )
        restarted = _restart_webai2api_api_mode(client, cfg)
        if not restarted:
            raise HTTPException(status_code=502, detail="WebAI2API 未能恢复普通 API 模式，请在缓存与重启页面手动重启")
        app.state.webai2api_login_mode_started_at = None
        return {"success": True, "message": "WebAI2API 已恢复普通 API 模式，可以继续调用模型"}

    @app.post("/api/admin/onboarding/providers/{provider_id}/login")
    async def admin_onboarding_provider_login(provider_id: str, request: Request) -> dict[str, Any]:
        require_local_admin(request)
        body = await _json_body(request)
        provider = get_provider(provider_id)
        if provider.route == "direct":
            cdp_url = str(body.get("cdpUrl") or body.get("cdp_url") or DEFAULT_CDP_URL)
            result = app.state.browser_launcher.start(provider_id, cdp_url)
            return {
                **result,
                "success": True,
                "providerId": provider_id,
                "loginKind": "direct",
                "actionLabel": "打开网页登录授权",
            }
        return await _start_webai2api_login_mode(provider_id, body)

    @app.post("/api/admin/onboarding/login/finish")
    async def admin_onboarding_login_finish(request: Request) -> dict[str, Any]:
        require_local_admin(request)
        return await _finish_webai2api_login_mode()

    @app.post("/api/admin/webai2api/login/start")
    async def admin_webai2api_login_start(request: Request) -> dict[str, Any]:
        require_local_admin(request)
        body = await _json_body(request)
        provider_id = str(body.get("providerId") or body.get("provider_id") or "").strip()
        return await _start_webai2api_login_mode(provider_id, body)

    @app.post("/api/admin/webai2api/login/finish")
    async def admin_webai2api_login_finish(request: Request) -> dict[str, Any]:
        require_local_admin(request)
        return await _finish_webai2api_login_mode()

    def _webai2api_instances_with_new_login_account(
        provider: Any,
        instances: list[Any],
    ) -> tuple[list[Any], str, str]:
        adapters = [str(adapter) for adapter in getattr(provider, "adapters", ()) if str(adapter).strip()]
        if not adapters:
            raise HTTPException(status_code=400, detail="Provider 没有可用于 WebAI2API 的 adapter")
        existing_instance_names = {str(item.get("name") or "") for item in instances if isinstance(item, dict)}
        existing_worker_names = {
            str(worker.get("name") or "")
            for item in instances
            if isinstance(item, dict)
            for worker in (item.get("workers") if isinstance(item.get("workers"), list) else [])
            if isinstance(worker, dict)
        }
        provider_slug = _safe_webai2api_name_part(provider.id)
        for _ in range(20):
            suffix = f"{datetime.now(timezone.utc).strftime('%Y%m%d%H%M%S')}_{secrets.token_hex(3)}"
            instance_name = f"gateway_{provider_slug}_{suffix}"
            worker_name = f"gateway_{provider_slug}_{suffix}"
            user_data_mark = f"gateway_{provider_slug}_{suffix}"
            if instance_name not in existing_instance_names and worker_name not in existing_worker_names:
                break
        else:
            raise HTTPException(status_code=500, detail="无法生成唯一的 WebAI2API 账号名称")
        if len(adapters) == 1:
            worker = {"name": worker_name, "type": adapters[0]}
        else:
            worker = {
                "name": worker_name,
                "type": "merge",
                "mergeTypes": adapters,
                "mergeMonitor": adapters[0],
            }
        new_instance = {
            "name": instance_name,
            "userDataMark": user_data_mark,
            "workers": [worker],
        }
        return [*instances, new_instance], instance_name, worker_name

    def _first_webai2api_worker_for_provider(
        provider: Any,
        instances: list[Any],
        *,
        require_dedicated_profile: bool = False,
    ) -> str:
        provider_dict = {"adapters": list(getattr(provider, "adapters", ()))}
        for instance in instances:
            if not isinstance(instance, dict):
                continue
            if require_dedicated_profile and not _instance_dedicated_to_provider(instance, provider_dict):
                continue
            workers = instance.get("workers") if isinstance(instance.get("workers"), list) else []
            for worker in workers:
                if isinstance(worker, dict) and _worker_supports_provider(worker, provider_dict):
                    worker_name = str(worker.get("name") or "")
                    if worker_name:
                        return worker_name
        return ""

    def _instance_dedicated_to_provider(instance: dict[str, Any], provider: dict[str, Any]) -> bool:
        provider_adapters = {str(adapter) for adapter in provider.get("adapters", []) if isinstance(adapter, str)}
        workers = instance.get("workers") if isinstance(instance.get("workers"), list) else []
        dict_workers = [worker for worker in workers if isinstance(worker, dict)]
        if not dict_workers:
            return False
        worker_adapter_sets = [_worker_adapter_ids(worker) for worker in dict_workers]
        return all(adapter_ids and adapter_ids.issubset(provider_adapters) for adapter_ids in worker_adapter_sets)

    def _worker_adapter_ids(worker: dict[str, Any]) -> set[str]:
        worker_type = str(worker.get("type") or "")
        merge_types = {str(item) for item in worker.get("mergeTypes", []) if isinstance(item, str)}
        if worker_type == "merge":
            return merge_types
        return {worker_type, *merge_types} if worker_type else merge_types

    def _instance_name_for_worker(instances: list[Any], worker_name: str) -> str:
        for instance in instances:
            if not isinstance(instance, dict):
                continue
            workers = instance.get("workers") if isinstance(instance.get("workers"), list) else []
            for worker in workers:
                if isinstance(worker, dict) and str(worker.get("name") or "") == worker_name:
                    return str(instance.get("name") or "")
        return ""

    def _webai2api_selected_account_already_active(provider: Any, parsed_account: Any, instances: list[Any]) -> bool:
        if not parsed_account.instance_name or not parsed_account.worker_name:
            return False
        active_provider_workers: list[tuple[str, str]] = []
        provider_dict = {
            "id": getattr(provider, "id", ""),
            "loginUrl": getattr(provider, "login_url", ""),
            "adapters": list(getattr(provider, "adapters", ())),
        }
        for instance in instances:
            if not isinstance(instance, dict):
                continue
            instance_name = str(instance.get("name") or "")
            workers = instance.get("workers") if isinstance(instance.get("workers"), list) else []
            for worker in workers:
                if not isinstance(worker, dict) or not _worker_supports_provider(worker, provider_dict):
                    continue
                worker_name = str(worker.get("name") or "")
                if instance_name and worker_name:
                    active_provider_workers.append((instance_name, worker_name))
        if active_provider_workers != [(parsed_account.instance_name, parsed_account.worker_name)]:
            return False
        return _webai2api_runtime_instance_loaded(provider_dict, parsed_account.instance_name)

    def _webai2api_runtime_instance_loaded(provider: dict[str, Any], instance_name: str) -> bool:
        domains = _webai2api_auth_domains(provider)
        if not domains:
            return True
        cookie_names = _load_webai2api_cookie_names(client, current_config(), instance_name, domains[0])
        return cookie_names is not None

    def _webai2api_wait_for_runtime_instance(provider: dict[str, Any], instance_name: str) -> bool:
        for _ in range(20):
            if _webai2api_runtime_instance_loaded(provider, instance_name):
                return True
            time.sleep(0.5)
        return False

    def _sync_webai2api_selected_account(provider: Any, parsed_account: Any) -> None:
        if not parsed_account.instance_name or not parsed_account.worker_name:
            raise HTTPException(status_code=400, detail="WebAI2API 账号缺少 instance/worker")
        cfg = current_config()
        instances = _load_webai2api_instances(client, cfg)
        current_hash = stable_json_hash(instances)
        state = app.state.account_registry.sync_state(provider.id)
        base_instances = instances
        if state.get("managedHash"):
            if current_hash == state.get("managedHash"):
                base_instances = _restore_provider_workers_snapshot(instances, provider, state.get("baseProviderWorkers"))
            elif current_hash == state.get("baseHash"):
                base_instances = instances
            else:
                raise HTTPException(status_code=409, detail="WebAI2API 工作池配置已被手动修改，未自动覆盖。请刷新账号列表后再选择。")
        synced = _webai2api_instances_for_selected_account(base_instances, provider, parsed_account.instance_name, parsed_account.worker_name)
        response = client.post(
            f"{_sidecar_root_from_base_url(cfg.upstream.base_url)}/admin/config/instances",
            json=synced,
            headers=upstream_headers(cfg),
        )
        if response.status_code >= 400:
            raise HTTPException(status_code=502, detail=_upstream_http_error_message(response))
        restart_response = client.post(
            f"{_sidecar_root_from_base_url(cfg.upstream.base_url)}/admin/restart",
            json={"loginMode": False},
            headers=upstream_headers(cfg),
        )
        if restart_response.status_code >= 400:
            raise HTTPException(status_code=502, detail=_upstream_http_error_message(restart_response))
        app.state.account_registry.save_sync_state(
            provider.id,
            {
                "baseHash": stable_json_hash(base_instances),
                "baseProviderWorkers": _provider_workers_snapshot(base_instances, provider),
                "managedHash": stable_json_hash(synced),
                "selectedAccountId": webai2api_account_id(provider.id, parsed_account.instance_name, parsed_account.worker_name),
                "updatedAt": utc_now(),
            },
        )

    def _webai2api_instances_for_selected_account(
        instances: list[Any],
        provider: Any,
        instance_name: str,
        worker_name: str,
    ) -> list[Any]:
        out: list[Any] = []
        for instance in instances:
            if not isinstance(instance, dict):
                continue
            copied = {**instance}
            workers = instance.get("workers") if isinstance(instance.get("workers"), list) else []
            kept_workers: list[Any] = []
            for worker in workers:
                if not isinstance(worker, dict):
                    continue
                if not _worker_supports_provider(worker, {"adapters": list(provider.adapters)}):
                    kept_workers.append(worker)
                    continue
                if str(instance.get("name") or "") == instance_name and str(worker.get("name") or "") == worker_name:
                    kept_workers.append(worker)
                    continue
                trimmed = _worker_without_provider_adapters(worker, set(provider.adapters))
                if trimmed is not None:
                    kept_workers.append(trimmed)
            if kept_workers:
                copied["workers"] = kept_workers
                out.append(copied)
        return out

    def _provider_workers_snapshot(instances: list[Any], provider: Any) -> list[dict[str, Any]]:
        snapshot: list[dict[str, Any]] = []
        for instance in instances:
            if not isinstance(instance, dict):
                continue
            workers = instance.get("workers") if isinstance(instance.get("workers"), list) else []
            provider_workers = [
                worker
                for worker in workers
                if isinstance(worker, dict) and _worker_supports_provider(worker, {"adapters": list(provider.adapters)})
            ]
            if provider_workers:
                snapshot.append({"name": str(instance.get("name") or ""), "workers": provider_workers})
        return snapshot

    def _restore_provider_workers_snapshot(instances: list[Any], provider: Any, snapshot: Any) -> list[Any]:
        if not isinstance(snapshot, list):
            return instances
        snapshot_by_instance = {
            str(item.get("name") or ""): item.get("workers")
            for item in snapshot
            if isinstance(item, dict) and isinstance(item.get("workers"), list)
        }
        out: list[Any] = []
        seen_instances: set[str] = set()
        for instance in instances:
            if not isinstance(instance, dict):
                continue
            instance_name = str(instance.get("name") or "")
            seen_instances.add(instance_name)
            workers = instance.get("workers") if isinstance(instance.get("workers"), list) else []
            non_provider_workers = [
                worker
                for worker in workers
                if isinstance(worker, dict) and not _worker_supports_provider(worker, {"adapters": list(provider.adapters)})
            ]
            restored_workers = snapshot_by_instance.get(instance_name, [])
            out.append({**instance, "workers": [*restored_workers, *non_provider_workers]})
        for instance_name, workers in snapshot_by_instance.items():
            if instance_name and instance_name not in seen_instances:
                out.append({"name": instance_name, "workers": workers})
        return out

    def _worker_without_provider_adapters(worker: dict[str, Any], adapter_ids: set[str]) -> dict[str, Any] | None:
        worker_type = str(worker.get("type") or "")
        merge_types = [str(item) for item in worker.get("mergeTypes", []) if isinstance(item, str)]
        remaining_merge = [item for item in merge_types if item not in adapter_ids]
        if worker_type not in adapter_ids:
            return {**worker, "mergeTypes": remaining_merge}
        if not remaining_merge:
            return None
        return {**worker, "type": remaining_merge[0], "mergeTypes": remaining_merge[1:]}

    def _validate_account_model(
        cfg: GatewayConfig,
        provider_id: str,
        model_id: str,
        provider: dict[str, Any] | None = None,
        parsed_account: Any | None = None,
    ) -> dict[str, Any]:
        payload = {
            "model": model_id,
            "messages": [{"role": "user", "content": "请只回复 OK。"}],
            "stream": False,
            "max_tokens": 8,
        }
        direct_validator = {
            "deepseek-web": _deepseek_web_chat,
            "qwen": _qwen_web_chat,
            "qwen-coder": _qwen_coder_chat,
        }.get(provider_id)
        if direct_validator is not None:
            return _validate_direct_account_model(direct_validator, cfg, provider_id, model_id, payload)
        if provider is not None and _is_webai2api_media_validation_model(provider, model_id):
            if (
                parsed_account is not None
                and getattr(parsed_account, "source", "") == "webai2api"
                and getattr(parsed_account, "instance_name", None)
                and _webai2api_wait_for_runtime_instance(provider, str(getattr(parsed_account, "instance_name")))
                and _webai2api_instance_authorized(provider, str(getattr(parsed_account, "instance_name")))
            ):
                return {"status": "available", "message": "网页登录态已检测，媒体模型由 WebAI2API 原生适配器调用"}
            return _unavailable_account_model_result(
                provider_id,
                model_id,
                "未检测到有效网页登录态，请重新打开授权窗口登录后再检测模型。",
            )
        try:
            response = client.post(
                cfg.upstream.base_url.rstrip("/") + "/chat/completions",
                json=payload,
                headers=upstream_headers(cfg),
                timeout=max(30, int(cfg.provider_runtime.request_timeout_seconds or 300)),
            )
        except Exception as exc:
            return _unavailable_account_model_result(provider_id, model_id, str(exc))
        if response.status_code < 400:
            return {"status": "available", "message": "验证通过"}
        return _unavailable_account_model_result(provider_id, model_id, _upstream_http_error_message(response))

    def _is_webai2api_media_validation_model(provider: dict[str, Any], model_id: str) -> bool:
        if provider.get("route") == "direct":
            return False
        capabilities = provider.get("capabilities") if isinstance(provider.get("capabilities"), dict) else {}
        owner = str(provider.get("id") or "")
        model_type = _infer_model_type(model_id, owner, None)
        if model_type in {"image", "video"}:
            return True
        return bool(capabilities.get("image") or capabilities.get("video")) and not bool(capabilities.get("text"))

    def _validate_direct_account_model(
        validator: Any,
        cfg: GatewayConfig,
        provider_id: str,
        model_id: str,
        payload: dict[str, Any],
    ) -> dict[str, Any]:
        try:
            response = validator(app, client, payload, cfg)
        except HTTPException as exc:
            return _unavailable_account_model_result(provider_id, model_id, str(exc.detail or exc))
        except Exception as exc:
            return _unavailable_account_model_result(provider_id, model_id, str(exc))
        status_code = getattr(response, "status_code", 200)
        if status_code < 400:
            return {"status": "available", "message": "验证通过"}
        body = getattr(response, "body", b"")
        if isinstance(body, bytes):
            body = body.decode("utf-8", errors="replace")
        return _unavailable_account_model_result(provider_id, model_id, str(body or status_code))

    def _unavailable_account_model_result(provider_id: str, model_id: str, message: str) -> dict[str, Any]:
        safe_message = _preview_text(_redact_sensitive_text(str(message or "")), max_chars=500)
        return {
            "status": "unavailable",
            "message": _friendly_account_model_validation_message(provider_id, model_id, safe_message),
        }

    def _friendly_account_model_validation_message(provider_id: str, model_id: str, message: str) -> str:
        lower = message.lower()
        if provider_id == "deepseek-web" and (
            "invalid token" in lower
            or "http 401" in lower
            or "unauthorized" in lower
            or "config.keys" in lower
            or "网页登录授权已过期" in message
        ):
            return (
                "DeepSeek Web 的网页登录授权已过期或失效。请点击“打开授权浏览器”重新登录 DeepSeek，"
                "完成后回到这里点“刷新模型”，再点“检测模型”。"
            )
        if (
            "model invalid" in lower
            or "模型无效" in message
            or "backend pool" in lower
            or "后端 pool" in message
            or "不支持" in message
        ):
            provider_hint = "DeepSeek/ds2api 通道" if provider_id == "deepseek-web" else "对应通道"
            return (
                f"当前通道没有启用 {model_id}，检测请求没有找到可用后端。"
                f"请先点击“刷新模型”；如果仍失败，进入“通道设置”确认{provider_hint}已启动、账号可用，"
                "并且后端池支持这个模型。重新网页登录不一定能解决。"
            )
        return message or "模型检测失败，请刷新模型后重试。"


    @app.post("/api/admin/provider-smoke/{provider_id}")
    async def admin_provider_smoke(provider_id: str, request: Request) -> dict[str, Any]:
        require_local_admin(request)
        cfg = current_config()
        return await run_in_threadpool(_run_provider_smoke_test, app, client, cfg, provider_id)

    @app.put("/api/admin/config")
    async def update_admin_config(request: Request) -> dict[str, Any]:
        require_local_admin(request)
        body = await _json_body(request)
        cfg = update_config(current_config(), body)
        app.state.config = cfg
        save_config(cfg, config_file)
        return config_to_admin(cfg)

    @app.post("/api/admin/token/rotate")
    def rotate_admin_token(request: Request) -> dict[str, Any]:
        require_local_admin(request)
        token = "wg_" + secrets.token_urlsafe(24)
        cfg = update_config(current_config(), {"server": {"apiKey": token}})
        app.state.config = cfg
        save_config(cfg, config_file)
        return config_to_admin(cfg)

    @app.get("/api/admin/web-auth/providers")
    def web_auth_providers(request: Request) -> dict[str, Any]:
        require_local_admin(request)
        return provider_payload(app.state.credential_store)

    @app.get("/api/admin/web-auth/credentials")
    def web_auth_credentials(request: Request) -> dict[str, Any]:
        require_local_admin(request)
        return {"credentials": app.state.credential_store.list_summaries()}

    @app.delete("/api/admin/web-auth/credentials/{provider_id}")
    def delete_web_auth_credential(provider_id: str, request: Request) -> dict[str, Any]:
        require_local_admin(request)
        get_provider(provider_id)
        app.state.credential_store.delete(provider_id)
        return {"credential": app.state.credential_store.summary(provider_id)}

    @app.post("/api/admin/web-auth/browser/start")
    async def start_web_auth_browser(request: Request) -> dict[str, Any]:
        require_local_admin(request)
        body = await _json_body(request)
        provider_id = str(body.get("provider") or "deepseek-web")
        cdp_url = str(body.get("cdpUrl") or DEFAULT_CDP_URL)
        get_provider(provider_id)
        return app.state.browser_launcher.start(provider_id, cdp_url)

    @app.post("/api/admin/web-auth/jobs")
    async def start_web_auth_job(request: Request) -> dict[str, Any]:
        require_local_admin(request)
        body = await _json_body(request)
        provider_id = str(body.get("provider") or "deepseek-web")
        cdp_url = str(body.get("cdpUrl") or DEFAULT_CDP_URL)
        provider = get_provider(provider_id)
        if provider.status != "available":
            raise HTTPException(status_code=400, detail=f"{provider.name} 暂未开放自动授权")
        job = create_job(provider_id)
        app.state.web_auth_jobs[job["id"]] = job

        async def run_job() -> None:
            try:
                def progress(message: str) -> None:
                    job["message"] = message
                    job["updatedAt"] = utc_now()

                credential = await app.state.web_auth_service.capture(provider_id, cdp_url, progress)
                app.state.credential_store.save(provider_id, credential)
                job.update(
                    {
                        "status": "succeeded",
                        "message": "网页登录状态已保存，可以直接使用对应网页模型。",
                        "credential": app.state.credential_store.summary(provider_id),
                        "updatedAt": utc_now(),
                    }
                )
            except Exception as exc:
                job.update({"status": "failed", "message": str(exc), "updatedAt": utc_now()})

        if run_auth_jobs_inline:
            await run_job()
        else:
            import asyncio

            asyncio.create_task(run_job())
        return job

    @app.get("/api/admin/web-auth/jobs/{job_id}")
    def get_web_auth_job(job_id: str, request: Request) -> dict[str, Any]:
        require_local_admin(request)
        job = app.state.web_auth_jobs.get(job_id)
        if not job:
            raise HTTPException(status_code=404, detail="授权任务不存在")
        return job

    @app.api_route("/admin", methods=["GET", "POST", "PUT", "PATCH", "DELETE", "OPTIONS"], include_in_schema=False)
    @app.api_route("/admin/{admin_path:path}", methods=["GET", "POST", "PUT", "PATCH", "DELETE", "OPTIONS"], include_in_schema=False)
    async def proxy_webai2api_admin(request: Request, admin_path: str = "") -> Response:
        require_local_admin(request)
        cfg = current_config()
        sidecar_authorization = f"Bearer {cfg.upstream.api_key}" if cfg.upstream.api_key else None
        gateway_authorization = f"Bearer {cfg.server.api_key}" if cfg.server.api_key else None
        return await _proxy_to_sidecar(
            client,
            request,
            _sidecar_root_from_base_url(cfg.upstream.base_url),
            "admin",
            admin_path,
            sidecar_authorization=sidecar_authorization,
            gateway_authorization=gateway_authorization,
        )

    @app.get("/v1/models")
    def models(
        authorization: str | None = Header(default=None),
        x_api_key: str | None = Header(default=None, alias="x-api-key"),
        api_key: str | None = Header(default=None, alias="api-key"),
    ) -> JSONResponse:
        require_auth(authorization, x_api_key, api_key)
        cfg = current_config()
        data: dict[str, Any] | None = None
        try:
            response = client.get(cfg.upstream.base_url.rstrip("/") + "/models", headers=upstream_headers(cfg))
            if response.status_code == 200:
                upstream_data = response.json()
                if isinstance(upstream_data, dict) and isinstance(upstream_data.get("data"), list):
                    data = _append_web_models(upstream_data)
        except Exception:
            pass
        if data is None:
            data = _append_web_models({"object": "list", "data": [{"id": cfg.upstream.model, "object": "model"}]})
        return JSONResponse(_append_qwen_live_model_catalog(app, client, data, cfg))

    @app.post("/v1/images/generations")
    async def image_generations(
        request: Request,
        authorization: str | None = Header(default=None),
        x_api_key: str | None = Header(default=None, alias="x-api-key"),
        api_key: str | None = Header(default=None, alias="api-key"),
    ) -> JSONResponse:
        require_auth(authorization, x_api_key, api_key)
        cfg = current_config()
        body = await _json_body(request)
        result = await run_in_threadpool(_create_image_generation_response, app, client, cfg, body)
        return JSONResponse(result)

    @app.post("/v1/videos")
    async def video_generations(
        request: Request,
        authorization: str | None = Header(default=None),
        x_api_key: str | None = Header(default=None, alias="x-api-key"),
        api_key: str | None = Header(default=None, alias="api-key"),
    ) -> JSONResponse:
        require_auth(authorization, x_api_key, api_key)
        cfg = current_config()
        body = await _json_body(request)
        result = await run_in_threadpool(_create_video_generation_response, app, client, cfg, body)
        return JSONResponse(result)

    @app.get("/v1/videos/{video_id}")
    def get_video_generation(
        video_id: str,
        authorization: str | None = Header(default=None),
        x_api_key: str | None = Header(default=None, alias="x-api-key"),
        api_key: str | None = Header(default=None, alias="api-key"),
    ) -> JSONResponse:
        require_auth(authorization, x_api_key, api_key)
        item = _get_cached_media_generation(app, video_id)
        if item is None:
            raise HTTPException(status_code=404, detail="Video generation not found")
        return JSONResponse(_video_generation_public_payload(item))

    @app.get("/v1/videos/{video_id}/content")
    def get_video_generation_content(
        video_id: str,
        authorization: str | None = Header(default=None),
        x_api_key: str | None = Header(default=None, alias="x-api-key"),
        api_key: str | None = Header(default=None, alias="api-key"),
    ) -> Response:
        require_auth(authorization, x_api_key, api_key)
        item = _get_cached_media_generation(app, video_id)
        if item is None:
            raise HTTPException(status_code=404, detail="Video generation not found")
        data = _decode_data_uri(str(item.get("data_uri") or ""), expected_kind="video")
        if data is None:
            raise HTTPException(status_code=502, detail="Cached video content is invalid")
        mime_type, content = data
        return Response(content, media_type=mime_type)

    @app.post("/v1/chat/completions")
    async def chat(
        request: Request,
        authorization: str | None = Header(default=None),
        x_api_key: str | None = Header(default=None, alias="x-api-key"),
        api_key: str | None = Header(default=None, alias="api-key"),
    ) -> Response:
        require_auth(authorization, x_api_key, api_key)
        cfg = current_config()
        body = normalize_model_body(await _json_body(request), default_model=cfg.upstream.model)
        target_account = str(request.headers.get("X-Ds2-Target-Account") or request.headers.get("X-WebAI-Target-Account") or "").strip()
        if target_account:
            body["_webai_target_account_id"] = target_account
        if is_deepseek_web_model(body.get("model")):
            return await run_in_threadpool(_deepseek_web_chat, app, client, body, cfg)
        if is_qwen_web_model(body.get("model")):
            if _qwen_web_uses_deepseek_ds2api(cfg):
                return await run_in_threadpool(
                    _deepseek_web_chat,
                    app,
                    client,
                    _qwen_web_body_for_deepseek_ds2api(body),
                    cfg,
                )
            return await run_in_threadpool(_qwen_web_chat, app, client, body, cfg)
        if is_qwen_coder_model(body.get("model")):
            return await run_in_threadpool(_qwen_coder_chat, app, client, body, cfg)
        if _gpt_thinking_uses_deepseek_ds2api(cfg) and _is_gpt_thinking_model(body.get("model")):
            try:
                return await run_in_threadpool(
                    _deepseek_web_chat,
                    app,
                    client,
                    _gpt_thinking_body_for_deepseek_ds2api(body),
                    cfg,
                )
            except HTTPException as exc:
                if exc.status_code != 424:
                    raise
                _record_request_diagnostic(
                    app,
                    "gpt_thinking_ds2api_fallback",
                    endpoint="/v1/chat/completions",
                    route="deepseek-web",
                    model=str(body.get("model") or ""),
                    reason=_preview_text(_redact_sensitive_text(str(exc.detail)), max_chars=500),
                )
        payload, bridge, allowed_tools, bridge_context = build_upstream_payload(body, cfg)
        model = str(payload.get("model") or cfg.upstream.model)
        preflight = _local_preflight_response(app, body, str(payload.get("model") or cfg.upstream.model), bridge_context)
        if preflight is not None:
            return preflight
        _record_completion_started_diagnostic(
            app,
            endpoint="/v1/chat/completions",
            route="upstream",
            model=model,
            body=body,
            stream=bool(body.get("stream")),
            bridge=bridge,
            bridge_context=bridge_context,
        )
        response = await run_in_threadpool(
            _post_upstream_with_login_mode_recovery,
            client,
            cfg,
            payload,
            _webai2api_login_mode_recovery_allowed(app),
        )
        if response.status_code >= 400:
            return _openai_upstream_http_error_response(
                app,
                endpoint="/v1/chat/completions",
                body=body,
                route="upstream",
                model=model,
                stream=bool(body.get("stream")),
                bridge=bridge,
                response=response,
                bridge_context=bridge_context,
            )
        if bool(body.get("stream")):
            if bridge:
                content, _finish_reason = parse_sse_text(response.text)
                stream_data = _openai_response_from_stream_buffer(content, finish_reason=_finish_reason, model=model)

                def retry_upstream_stream(retry_payload: dict[str, Any]) -> dict[str, Any] | None:
                    retry_response = _post_upstream_with_login_mode_recovery(
                        client,
                        cfg,
                        retry_payload,
                        _webai2api_login_mode_recovery_allowed(app),
                    )
                    if retry_response.status_code >= 400:
                        message = _upstream_http_error_message(retry_response)
                        gateway_status_code = _gateway_status_code_for_upstream_http_error(retry_response)
                        _record_completion_error_diagnostic(
                            app,
                            endpoint="/v1/chat/completions",
                            route="upstream",
                            model=model,
                            body=body,
                            stream=True,
                            bridge=bridge,
                            status_code=gateway_status_code,
                            error_kind="upstream_http_error",
                            error=RuntimeError(message),
                            bridge_context=bridge_context,
                        )
                        raise HTTPException(
                            status_code=gateway_status_code,
                            detail=_preview_text(_redact_sensitive_text(message), max_chars=800),
                        )
                    retry_data = retry_response.json()
                    return retry_data if isinstance(retry_data, dict) else None

                parsed, bridge_result = await run_in_threadpool(
                    _parse_bridge_chat_data,
                    stream_data,
                    app=app,
                    payload=payload,
                    bridge=bridge,
                    allowed_tools=allowed_tools,
                    bridge_context=bridge_context,
                    model=model,
                    retry_chat=retry_upstream_stream,
                )
                tool_calls = _response_tool_calls(parsed)
                if tool_calls:
                    sse_body = build_openai_tool_calls_sse(tool_calls, model=model)
                else:
                    text = _response_message_text(parsed)
                    sse_body = build_tool_call_sse(
                        text,
                        allowed_tools=allowed_tools,
                        model=model,
                        bridge_context=bridge_context,
                    )
                return _openai_stream_response(
                    app,
                    body=body,
                    route="upstream",
                    model=model,
                    bridge=bridge,
                    sse_body=sse_body,
                    bridge_result=bridge_result,
                    bridge_context=bridge_context,
                )
            stream_error = _upstream_sse_error_message(response.text)
            if stream_error:
                return _openai_completion_error_response(
                    app,
                    endpoint="/v1/chat/completions",
                    body=body,
                    route="upstream",
                    model=str(payload.get("model") or cfg.upstream.model),
                    stream=True,
                    bridge=bridge,
                    status_code=502,
                    diagnostic_status_code=response.status_code,
                    error_kind="upstream_stream_error",
                    message=f"WebAI2API upstream stream error: {stream_error}",
                    bridge_context=bridge_context,
                )
            _record_completion_diagnostic(
                app,
                endpoint="/v1/chat/completions",
                route="upstream",
                model=str(payload.get("model") or cfg.upstream.model),
                body=body,
                stream=True,
                bridge=bridge,
                passThrough=True,
                statusCode=response.status_code,
                **_openai_sse_response_fields(response.text),
            )
            return Response(response.content, media_type=response.headers.get("content-type") or "text/event-stream")
        data = response.json()
        if not isinstance(data, dict):
            raise HTTPException(status_code=502, detail="Upstream response must be a JSON object")
        def retry_upstream(retry_payload: dict[str, Any]) -> dict[str, Any] | None:
            retry_response = _post_upstream_with_login_mode_recovery(
                client,
                cfg,
                retry_payload,
                _webai2api_login_mode_recovery_allowed(app),
            )
            if retry_response.status_code >= 400:
                message = _upstream_http_error_message(retry_response)
                gateway_status_code = _gateway_status_code_for_upstream_http_error(retry_response)
                _record_completion_error_diagnostic(
                    app,
                    endpoint="/v1/chat/completions",
                    route="upstream",
                    model=model,
                    body=body,
                    stream=bool(body.get("stream")),
                    bridge=bridge,
                    status_code=gateway_status_code,
                    error_kind="upstream_http_error",
                    error=RuntimeError(message),
                    bridge_context=bridge_context,
                )
                raise HTTPException(
                    status_code=gateway_status_code,
                    detail=_preview_text(_redact_sensitive_text(message), max_chars=800),
                )
            retry_data = retry_response.json()
            return retry_data if isinstance(retry_data, dict) else None

        parsed, bridge_result = await run_in_threadpool(
            _parse_bridge_chat_data,
            data,
            app=app,
            payload=payload,
            bridge=bridge,
            allowed_tools=allowed_tools,
            bridge_context=bridge_context,
            model=model,
            retry_chat=retry_upstream,
        )
        return _openai_json_response(
            app,
            body=body,
            route="upstream",
            model=model,
            bridge=bridge,
            parsed=parsed,
            bridge_result=bridge_result,
            bridge_context=bridge_context,
        )

    @app.post("/messages/count_tokens")
    @app.post("/v1/messages/count_tokens")
    async def anthropic_count_tokens_route(
        request: Request,
        authorization: str | None = Header(default=None),
        x_api_key: str | None = Header(default=None, alias="x-api-key"),
        api_key: str | None = Header(default=None, alias="api-key"),
    ) -> JSONResponse:
        require_auth(authorization, x_api_key, api_key)
        body = await _json_body(request)
        return JSONResponse(anthropic_count_tokens(body))

    @app.post("/messages")
    @app.post("/v1/messages")
    async def anthropic_messages(
        request: Request,
        authorization: str | None = Header(default=None),
        x_api_key: str | None = Header(default=None, alias="x-api-key"),
        api_key: str | None = Header(default=None, alias="api-key"),
    ) -> Response:
        require_auth(authorization, x_api_key, api_key)
        cfg = current_config()
        body = normalize_model_body(await _json_body(request), default_model=cfg.upstream.model)
        try:
            openai_body = anthropic_body_to_openai(
                body,
                tool_call_registry=getattr(app.state, "tool_call_registry", None),
            )
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        target_account = str(request.headers.get("X-Ds2-Target-Account") or request.headers.get("X-WebAI-Target-Account") or "").strip()
        if target_account:
            openai_body["_webai_target_account_id"] = target_account
        bridge_result = None
        direct_bridge_headers: dict[str, str] = {}
        if is_deepseek_web_model(openai_body.get("model")):
            parsed, direct_bridge_headers = await run_in_threadpool(
                _direct_provider_chat_payload_with_headers,
                _deepseek_web_chat,
                app,
                client,
                openai_body,
                cfg,
            )
        elif is_qwen_web_model(openai_body.get("model")):
            if _qwen_web_uses_deepseek_ds2api(cfg):
                parsed, direct_bridge_headers = await run_in_threadpool(
                    _direct_provider_chat_payload_with_headers,
                    _deepseek_web_chat,
                    app,
                    client,
                    _qwen_web_body_for_deepseek_ds2api(openai_body),
                    cfg,
                )
            else:
                parsed, direct_bridge_headers = await run_in_threadpool(
                    _direct_provider_chat_payload_with_headers,
                    _qwen_web_chat,
                    app,
                    client,
                    openai_body,
                    cfg,
                )
        elif is_qwen_coder_model(openai_body.get("model")):
            parsed, direct_bridge_headers = await run_in_threadpool(
                _direct_provider_chat_payload_with_headers,
                _qwen_coder_chat,
                app,
                client,
                openai_body,
                cfg,
            )
        elif _gpt_thinking_uses_deepseek_ds2api(cfg) and _is_gpt_thinking_model(openai_body.get("model")):
            parsed, direct_bridge_headers = await run_in_threadpool(
                _direct_provider_chat_payload_with_headers,
                _deepseek_web_chat,
                app,
                client,
                _gpt_thinking_body_for_deepseek_ds2api(openai_body),
                cfg,
            )
        else:
            payload, bridge, allowed_tools, bridge_context = build_upstream_payload(openai_body, cfg)
            model = str(openai_body.get("model") or cfg.upstream.model)
            _record_completion_started_diagnostic(
                app,
                endpoint="/v1/messages",
                route="upstream",
                model=model,
                body=body,
                stream=bool(body.get("stream")),
                bridge=bridge,
                bridge_context=bridge_context,
            )
            preflight_data = build_preflight_chat_response(model, bridge_context)
            if preflight_data is not None:
                _record_tool_bridge_event(app, "local_repo_preflight", model=model, **_tool_call_event_fields(preflight_data))
                parsed = preflight_data
            else:
                response = await run_in_threadpool(
                    _post_upstream_with_login_mode_recovery,
                    client,
                    cfg,
                    payload,
                    _webai2api_login_mode_recovery_allowed(app),
                )
                if response.status_code >= 400:
                    message = _upstream_http_error_message(response)
                    gateway_status_code = _gateway_status_code_for_upstream_http_error(response)
                    _record_completion_error_diagnostic(
                        app,
                        endpoint="/v1/messages",
                        route="upstream",
                        model=model,
                        body=openai_body,
                        stream=bool(body.get("stream")),
                        bridge=bridge,
                        status_code=gateway_status_code,
                        error_kind="upstream_http_error",
                        error=RuntimeError(message),
                        bridge_context=bridge_context,
                    )
                    raise HTTPException(
                        status_code=gateway_status_code,
                        detail=_preview_text(_redact_sensitive_text(message), max_chars=800),
                    )
                data = response.json()
                if not isinstance(data, dict):
                    raise HTTPException(status_code=502, detail="Upstream response must be a JSON object")

                def retry_upstream(retry_payload: dict[str, Any]) -> dict[str, Any] | None:
                    retry_response = _post_upstream_with_login_mode_recovery(
                        client,
                        cfg,
                        retry_payload,
                        _webai2api_login_mode_recovery_allowed(app),
                    )
                    if retry_response.status_code >= 400:
                        message = _upstream_http_error_message(retry_response)
                        gateway_status_code = _gateway_status_code_for_upstream_http_error(retry_response)
                        _record_completion_error_diagnostic(
                            app,
                            endpoint="/v1/messages",
                            route="upstream",
                            model=model,
                            body=openai_body,
                            stream=bool(body.get("stream")),
                            bridge=bridge,
                            status_code=gateway_status_code,
                            error_kind="upstream_http_error",
                            error=RuntimeError(message),
                            bridge_context=bridge_context,
                        )
                        raise HTTPException(
                            status_code=gateway_status_code,
                            detail=_preview_text(_redact_sensitive_text(message), max_chars=800),
                        )
                    retry_data = retry_response.json()
                    return retry_data if isinstance(retry_data, dict) else None

                parsed, bridge_result = await run_in_threadpool(
                    _parse_bridge_chat_data,
                    data,
                    app=app,
                    payload=payload,
                    bridge=bridge,
                    allowed_tools=allowed_tools,
                    model=model,
                    bridge_context=bridge_context,
                    retry_chat=retry_upstream,
                )
        _remember_openai_tool_calls(app, parsed)
        anthropic_message = openai_response_to_anthropic(
            parsed,
            original_model=normalize_model_id(body.get("model") or openai_body.get("model"), cfg.upstream.model),
        )
        response_headers = bridge_error_headers(bridge_result)
        response_headers.update(direct_bridge_headers)
        if bool(body.get("stream")):
            sse_body = anthropic_response_to_sse(anthropic_message)
            _record_completion_diagnostic(
                app,
                endpoint="/v1/messages",
                route=_provider_route(normalize_model_id(openai_body.get("model"), cfg.upstream.model), cfg),
                model=normalize_model_id(body.get("model") or openai_body.get("model"), cfg.upstream.model),
                body=body,
                stream=True,
                bridge=bool(bridge_result) or bool(direct_bridge_headers),
                **_anthropic_message_response_fields(anthropic_message, response_kind="sse"),
                **_tool_bridge_completion_fields(bridge_result),
            )
            return Response(
                sse_body,
                media_type="text/event-stream",
                headers=response_headers,
            )
        _record_completion_diagnostic(
            app,
            endpoint="/v1/messages",
            route=_provider_route(normalize_model_id(openai_body.get("model"), cfg.upstream.model), cfg),
            model=normalize_model_id(body.get("model") or openai_body.get("model"), cfg.upstream.model),
            body=body,
            stream=False,
            bridge=bool(bridge_result) or bool(direct_bridge_headers),
            **_anthropic_message_response_fields(anthropic_message, response_kind="json"),
            **_tool_bridge_completion_fields(bridge_result),
        )
        return JSONResponse(anthropic_message, headers=response_headers)

    @app.get("/{spa_path:path}", include_in_schema=False)
    def native_spa_fallback(spa_path: str, request: Request) -> FileResponse:
        require_local_admin(request)
        if spa_path.startswith(("api/", "v1/", "admin/", "static/", "assets/")):
            raise HTTPException(status_code=404, detail="Not found")
        index_path = _native_index_path(webai2api_ui_dir)
        if not index_path:
            raise HTTPException(status_code=404, detail="Native WebAI2API UI assets are missing")
        return _native_file_response(index_path)

    return app


def _default_native_ui_dir() -> Path:
    return Path(__file__).resolve().parents[1] / "webui" / "dist"


def _default_auto_research_fixture_dir() -> Path:
    return Path(__file__).resolve().parents[1] / "tests" / "fixtures" / "tool_bridge_replays"


def _runtime_status(app: FastAPI) -> dict[str, Any]:
    latest = _latest_gateway_source_mtime()
    started_epoch = float(getattr(app.state, "runtime_started_epoch", 0) or 0)
    latest_epoch = float(latest.get("mtimeEpoch") or 0)
    stale = latest_epoch > started_epoch + 0.5
    ds2api_status = {
        "oracleCommit": DS2API_ORACLE_COMMIT,
        "oracleVersion": DS2API_ORACLE_VERSION,
        "sidecarVersion": "unknown",
        "sidecarCommit": "unknown",
        "latestAlignmentClaimAllowed": False,
    }
    return {
        "startedAt": getattr(app.state, "runtime_started_at", ""),
        "sourceFresh": not stale,
        "sourceStale": stale,
        "latestSource": latest,
        "ds2api": ds2api_status,
        "supervisor": collect_supervisor_status(
            getattr(app.state, "config", GatewayConfig()),
            Path(getattr(app.state, "gateway_root", Path.cwd())).resolve(),
        ),
        "statusText": "运行代码是最新的" if not stale else "源码已更新，请重启 Gateway 让补丁生效",
    }


def _run_provider_smoke_test(
    app: FastAPI,
    client: httpx.Client,
    config: GatewayConfig,
    provider_id: str,
) -> dict[str, Any]:
    try:
        provider = get_provider(provider_id)
    except ValueError:
        provider = None
    if provider is not None and provider.route != "direct":
        model = _webai2api_provider_smoke_model(provider, client, config)
        if not model:
            return {
                "provider": provider_id,
                "authorized": True,
                "ok": False,
                "passed": 0,
                "total": 1,
                "results": [{"id": "models", "ok": False, "message": "WebAI2API 未返回该 provider 的可用文本模型"}],
            }
        results = _run_webai2api_provider_smoke(app, client, config, provider_id=provider_id, model=model)
        passed = sum(1 for item in results if item.get("ok"))
        return {
            "provider": provider_id,
            "authorized": True,
            "model": model,
            "ok": passed == len(results),
            "passed": passed,
            "total": len(results),
            "results": results,
        }

    credential = app.state.credential_store.get(provider_id)
    if not is_credential_authorized(provider_id, credential):
        return {
            "provider": provider_id,
            "authorized": False,
            "ok": False,
            "passed": 0,
            "total": 1,
            "results": [{"id": "auth", "ok": False, "message": "Provider 未授权"}],
        }
    if provider_id == "qwen":
        results = _run_qwen_provider_smoke(app, client, config)
    elif provider_id == "deepseek-web":
        results = _run_deepseek_provider_smoke(app, client, config)
    else:
        return {
            "provider": provider_id,
            "authorized": True,
            "ok": False,
            "passed": 0,
            "total": 1,
            "results": [{"id": "unsupported", "ok": False, "message": "当前只支持 qwen 和 deepseek-web 自测"}],
        }
    passed = sum(1 for item in results if item.get("ok"))
    return {
        "provider": provider_id,
        "authorized": True,
        "ok": passed == len(results),
        "passed": passed,
        "total": len(results),
        "results": results,
    }


def _run_qwen_provider_smoke(app: FastAPI, client: httpx.Client, config: GatewayConfig) -> list[dict[str, Any]]:
    return _run_direct_provider_smoke(
        app,
        client,
        config,
        provider_id="qwen",
        model="qwen-web/qwen3.6-plus",
        chat_func=_qwen_web_chat_payload,
    )


def _run_deepseek_provider_smoke(app: FastAPI, client: httpx.Client, config: GatewayConfig) -> list[dict[str, Any]]:
    return _run_direct_provider_smoke(
        app,
        client,
        config,
        provider_id="deepseek-web",
        model="deepseek-v4-pro",
        chat_func=_deepseek_web_chat_payload,
    )


def _run_webai2api_provider_smoke(
    app: FastAPI,
    client: httpx.Client,
    config: GatewayConfig,
    *,
    provider_id: str,
    model: str,
) -> list[dict[str, Any]]:
    return _run_direct_provider_smoke(
        app,
        client,
        config,
        provider_id=provider_id,
        model=model,
        chat_func=_webai2api_upstream_chat_payload,
    )


def _webai2api_provider_smoke_model(provider: Any, client: httpx.Client, config: GatewayConfig) -> str:
    if not bool(getattr(provider, "capabilities", {}).get("text")):
        return ""
    models_payload = _load_gateway_models(client, config)
    models = models_payload.get("data") if isinstance(models_payload.get("data"), list) else []
    model_ids = {item.get("id") for item in models if isinstance(item, dict)}
    for model_id in getattr(provider, "models", ()):
        if model_id in model_ids:
            return str(model_id)
    adapter_ids = {str(adapter) for adapter in getattr(provider, "adapters", ())}
    adapter_prefixes = tuple(f"{adapter}/" for adapter in adapter_ids)
    for item in models:
        if not isinstance(item, dict):
            continue
        model_id = item.get("id")
        if not isinstance(model_id, str):
            continue
        owner = str(item.get("owned_by") or "")
        if owner in adapter_ids or model_id.startswith(adapter_prefixes):
            return model_id
    first = next((str(model_id) for model_id in getattr(provider, "models", ()) if str(model_id)), "")
    return first


def _webai2api_upstream_chat_payload(
    app: FastAPI,
    client: httpx.Client,
    body: dict[str, Any],
    config: GatewayConfig,
) -> dict[str, Any]:
    payload, bridge, allowed_tools, bridge_context = build_upstream_payload(body, config)
    model = str(payload.get("model") or config.upstream.model)
    preflight = _local_preflight_response(app, body, model, bridge_context)
    if preflight is not None:
        return json_loads_response_from_any(preflight)
    response = _post_upstream_with_login_mode_recovery(
        client,
        config,
        payload,
        _webai2api_login_mode_recovery_allowed(app),
    )
    if response.status_code >= 400:
        raise RuntimeError(_upstream_http_error_message(response))
    data = response.json()
    if not isinstance(data, dict):
        raise RuntimeError("WebAI2API upstream response must be a JSON object")

    def retry_chat(retry_payload: dict[str, Any]) -> dict[str, Any] | None:
        retry_response = _post_upstream_with_login_mode_recovery(
            client,
            config,
            retry_payload,
            _webai2api_login_mode_recovery_allowed(app),
        )
        if retry_response.status_code >= 400:
            raise RuntimeError(_upstream_http_error_message(retry_response))
        retry_data = retry_response.json()
        return retry_data if isinstance(retry_data, dict) else None

    parsed, _bridge_result = _parse_bridge_chat_data(
        data,
        app=app,
        payload=payload,
        bridge=bridge,
        allowed_tools=allowed_tools,
        bridge_context=bridge_context,
        model=model,
        retry_chat=retry_chat,
    )
    return parsed


def _run_direct_provider_smoke(
    app: FastAPI,
    client: httpx.Client,
    config: GatewayConfig,
    *,
    provider_id: str,
    model: str,
    chat_func: Any,
) -> list[dict[str, Any]]:
    results: list[dict[str, Any]] = []

    def run_step(step_id: str, fn: Any) -> None:
        try:
            results.append(fn())
        except Exception as exc:
            results.append(_smoke_error_result(step_id, exc))

    run_step("models", lambda: _provider_smoke_models_result(client, config, model))
    run_step("openai_text", lambda: _provider_smoke_openai_text_result(app, client, config, model, chat_func))
    run_step("openai_tool_use", lambda: _provider_smoke_openai_tool_use_result(app, client, config, model, chat_func))
    run_step("anthropic_tool_use", lambda: _provider_smoke_anthropic_tool_use_result(app, client, config, model, chat_func))
    run_step("anthropic_tool_result", lambda: _provider_smoke_anthropic_tool_result_result(app, client, config, model, chat_func))
    return results


def _provider_smoke_models_result(client: httpx.Client, config: GatewayConfig, model: str) -> dict[str, Any]:
    models_payload = _load_gateway_models(client, config)
    model_ids = {item.get("id") for item in models_payload.get("data", []) if isinstance(item, dict)}
    return _smoke_result("models", model in model_ids, detail={"model": model})


def _provider_smoke_openai_text_result(
    app: FastAPI,
    client: httpx.Client,
    config: GatewayConfig,
    model: str,
    chat_func: Any,
) -> dict[str, Any]:
    text_payload = chat_func(
        app,
        client,
        {"model": model, "messages": [{"role": "user", "content": "只回复 PROVIDER_SMOKE_OK，不要解释。"}]},
        config,
    )
    text = _response_message_text(json_loads_response_from_any(text_payload))
    return _smoke_result(
        "openai_text",
        _smoke_text_matches(text, "PROVIDER_SMOKE_OK"),
        detail={"text": _smoke_detail_text(text)},
    )


def _provider_smoke_openai_tool_use_result(
    app: FastAPI,
    client: httpx.Client,
    config: GatewayConfig,
    model: str,
    chat_func: Any,
) -> dict[str, Any]:
    tool_payload = chat_func(app, client, _openai_weather_tool_body(model), config)
    tool_data = json_loads_response_from_any(tool_payload)
    return _openai_tool_use_smoke_result("openai_tool_use", tool_data)


def _provider_smoke_anthropic_tool_use_result(
    app: FastAPI,
    client: httpx.Client,
    config: GatewayConfig,
    model: str,
    chat_func: Any,
) -> dict[str, Any]:
    body = {
        "model": model,
        "max_tokens": 256,
        "messages": [{"role": "user", "content": "请调用 get_weather，city 必须是 Beijing，不要直接回答。"}],
        "tools": [_anthropic_weather_tool()],
        "tool_choice": {"type": "tool", "name": "get_weather"},
    }
    openai_body = anthropic_body_to_openai(body, tool_call_registry=getattr(app.state, "tool_call_registry", None))
    tool_payload = chat_func(app, client, openai_body, config)
    anthropic_message = openai_response_to_anthropic(json_loads_response_from_any(tool_payload), original_model=model)
    tool_use = _first_anthropic_tool_use(anthropic_message)
    input_value = tool_use.get("input") if isinstance(tool_use.get("input"), dict) else {}
    return _smoke_result(
        "anthropic_tool_use",
        bool(tool_use) and str(tool_use.get("name") or "") == "get_weather" and input_value == {"city": "Beijing"},
        detail={"name": str(tool_use.get("name") or ""), "input": input_value},
    )


def _provider_smoke_anthropic_tool_result_result(
    app: FastAPI,
    client: httpx.Client,
    config: GatewayConfig,
    model: str,
    chat_func: Any,
) -> dict[str, Any]:
    body = {
        "model": model,
        "max_tokens": 256,
        "messages": [
            {"role": "user", "content": "查询北京天气，然后用一句话回答。"},
            {
                "role": "assistant",
                "content": [
                    {
                        "type": "tool_use",
                        "id": "call_weather_1",
                        "name": "get_weather",
                        "input": {"city": "Beijing"},
                    }
                ],
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "tool_result",
                        "tool_use_id": "call_weather_1",
                        "content": [{"type": "text", "text": "北京今天晴，气温 22°C。"}],
                    }
                ],
            },
        ],
        "tools": [_anthropic_weather_tool()],
    }
    openai_body = anthropic_body_to_openai(body, tool_call_registry=getattr(app.state, "tool_call_registry", None))
    payload = chat_func(app, client, openai_body, config)
    anthropic_message = openai_response_to_anthropic(json_loads_response_from_any(payload), original_model=model)
    text = _anthropic_message_text(anthropic_message)
    return _smoke_result(
        "anthropic_tool_result",
        bool(text.strip()) and not _first_anthropic_tool_use(anthropic_message),
        detail={"text": _smoke_detail_text(text)},
    )


def _openai_weather_tool_body(model: str) -> dict[str, Any]:
    return {
        "model": model,
        "messages": [{"role": "user", "content": "请调用 get_weather，city 必须是 Beijing，不要直接回答。"}],
        "tools": [
            {
                "type": "function",
                "function": {
                    "name": "get_weather",
                    "description": "查询天气",
                    "parameters": {
                        "type": "object",
                        "properties": {"city": {"type": "string"}},
                        "required": ["city"],
                    },
                },
            }
        ],
        "tool_choice": {"type": "function", "function": {"name": "get_weather"}},
    }


def _anthropic_weather_tool() -> dict[str, Any]:
    return {
        "name": "get_weather",
        "description": "查询天气",
        "input_schema": {
            "type": "object",
            "properties": {"city": {"type": "string"}},
            "required": ["city"],
        },
    }


def _openai_tool_use_smoke_result(step_id: str, tool_data: dict[str, Any]) -> dict[str, Any]:
    tool_calls = _response_tool_calls(tool_data)
    first_tool = tool_calls[0] if tool_calls else {}
    first_function = first_tool.get("function") if isinstance(first_tool, dict) and isinstance(first_tool.get("function"), dict) else {}
    first_input = {}
    try:
        first_input = json.loads(str(first_function.get("arguments") or "{}"))
    except Exception:
        first_input = {}
    return _smoke_result(
        step_id,
        bool(tool_calls) and str(first_function.get("name") or "") == "get_weather" and first_input == {"city": "Beijing"},
        detail={"name": str(first_function.get("name") or ""), "input": first_input},
    )


def _first_anthropic_tool_use(message: dict[str, Any]) -> dict[str, Any]:
    content = message.get("content") if isinstance(message.get("content"), list) else []
    for block in content:
        if isinstance(block, dict) and block.get("type") == "tool_use":
            return block
    return {}


def _anthropic_message_text(message: dict[str, Any]) -> str:
    content = message.get("content") if isinstance(message.get("content"), list) else []
    return "".join(str(block.get("text") or "") for block in content if isinstance(block, dict) and block.get("type") == "text")


def _smoke_result(step_id: str, ok: bool, *, detail: dict[str, Any] | None = None, message: str = "") -> dict[str, Any]:
    result: dict[str, Any] = {"id": step_id, "ok": bool(ok)}
    if message:
        result["message"] = _smoke_detail_text(message)
    if detail is not None:
        result["detail"] = detail
    return result


def _smoke_error_result(step_id: str, exc: Exception) -> dict[str, Any]:
    return _smoke_result(step_id, False, message=_smoke_exception_message(exc))


def _smoke_exception_message(exc: Exception) -> str:
    if isinstance(exc, HTTPException):
        detail = exc.detail if isinstance(exc.detail, str) else json.dumps(exc.detail, ensure_ascii=False)
        return f"HTTP {exc.status_code}: {detail}"
    return str(exc)


def _smoke_text_matches(text: str, expected: str) -> bool:
    normalized = (text or "").strip().strip("`\"'“”‘’")
    return normalized == expected


def _smoke_detail_text(text: str) -> str:
    redacted = _redact_sensitive_text(str(text or ""))
    redacted = _SMOKE_SENSITIVE_FIELD_RE.sub("[redacted]", redacted)
    return _preview_text(redacted, max_chars=320)


def json_loads_response_from_any(response: Any) -> dict[str, Any]:
    if isinstance(response, JSONResponse):
        return json_loads_response(response)
    if isinstance(response, dict):
        return response
    raise HTTPException(status_code=502, detail="Smoke response must be JSON serializable")


def _response_message_text(data: dict[str, Any]) -> str:
    choices = data.get("choices") if isinstance(data.get("choices"), list) else []
    message = choices[0].get("message") if choices and isinstance(choices[0], dict) and isinstance(choices[0].get("message"), dict) else {}
    return str(message.get("content") or "")


def _response_tool_calls(data: dict[str, Any]) -> list[dict[str, Any]]:
    choices = data.get("choices") if isinstance(data.get("choices"), list) else []
    message = choices[0].get("message") if choices and isinstance(choices[0], dict) and isinstance(choices[0].get("message"), dict) else {}
    tool_calls = message.get("tool_calls")
    return tool_calls if isinstance(tool_calls, list) else []


def _openai_response_from_stream_buffer(content: str, *, finish_reason: str, model: str) -> dict[str, Any]:
    return {
        "id": "upstream-stream-buffer",
        "object": "chat.completion",
        "choices": [
            {
                "index": 0,
                "finish_reason": finish_reason or "stop",
                "message": {"role": "assistant", "content": content or ""},
            }
        ],
        "model": model,
    }


def _latest_gateway_source_mtime() -> dict[str, Any]:
    tool_bridge_source = inspect.getsourcefile(parse_tool_response)
    source_files = [
        Path(__file__),
        Path(__file__).with_name("openai_api.py"),
        Path(__file__).with_name("anthropic_api.py"),
        Path(__file__).with_name("qwen_web.py"),
        Path(__file__).with_name("qwen_coder.py"),
        Path(__file__).with_name("prompt_compaction.py"),
        Path(__file__).with_name("config.py"),
    ]
    if tool_bridge_source:
        source_files.append(Path(tool_bridge_source))
    existing = [path.resolve() for path in source_files if str(path) and path.exists()]
    if not existing:
        return {"path": "", "mtime": "", "mtimeEpoch": 0}
    latest_path = max(existing, key=lambda path: path.stat().st_mtime)
    latest_epoch = latest_path.stat().st_mtime
    return {
        "path": latest_path.name,
        "mtime": datetime.fromtimestamp(latest_epoch, timezone.utc).isoformat(),
        "mtimeEpoch": latest_epoch,
    }


def _native_index_path(native_ui_dir: Path) -> Path | None:
    index_path = native_ui_dir / "index.html"
    return index_path if index_path.exists() else None


def _onboarding_connection_profiles(providers: list[dict[str, Any]], cfg: GatewayConfig) -> list[dict[str, Any]]:
    profiles: list[dict[str, Any]] = []
    for provider in providers:
        provider_id = str(provider.get("id") or "")
        provider_name = str(provider.get("name") or provider_id or "网页模型")
        route = str(provider.get("route") or "")
        credential = provider.get("credential") if isinstance(provider.get("credential"), dict) else {}
        model_availability = provider.get("modelAvailability") if isinstance(provider.get("modelAvailability"), dict) else {}
        authorized = bool(credential.get("authorized"))
        available_models = provider.get("availableModels") if isinstance(provider.get("availableModels"), list) else []
        profile_model_ids: list[str] = []
        seen_model_ids: set[str] = set()
        for candidate in [*available_models, *model_availability.keys()]:
            if isinstance(candidate, str) and candidate.strip() and candidate not in seen_model_ids:
                seen_model_ids.add(candidate)
                profile_model_ids.append(candidate)

        for model_id in profile_model_ids:
            if not isinstance(model_id, str) or not model_id.strip():
                continue
            availability = model_availability.get(model_id) if isinstance(model_availability.get(model_id), dict) else {}
            status = str(availability.get("status") or ("available" if authorized else "pending"))
            message = str(availability.get("message") or provider.get("availabilityMessage") or "")
            backend_kind, backend_base_url = _onboarding_model_backend_metadata(provider, model_id, cfg)
            profiles.append(
                {
                    "id": f"{provider_id}:{model_id}",
                    "providerId": provider_id,
                    "providerName": provider_name,
                    "displayName": f"{provider_name} / {model_id}",
                    "modelId": model_id,
                    "route": route,
                    "loginKind": provider.get("loginKind") or ("direct" if route == "direct" else "webai2api"),
                    "authorized": authorized,
                    "available": status != "unavailable",
                    "availabilityStatus": status,
                    "availabilityMessage": message,
                    "clientBaseUrl": "/v1",
                    "openaiBaseUrl": "/v1",
                    "anthropicBaseUrl": "/v1",
                    "openaiEndpoint": "/v1/chat/completions",
                    "anthropicEndpoint": "/v1/messages",
                    "backendKind": backend_kind,
                    "backendBaseUrl": backend_base_url,
                }
            )
    return profiles


def _onboarding_model_backend_metadata(provider: dict[str, Any], model_id: str, cfg: GatewayConfig) -> tuple[str, str]:
    if _is_gpt_thinking_model(model_id) and _gpt_thinking_uses_deepseek_ds2api(cfg):
        return "ds2api-sidecar", cfg.provider_runtime.deepseek_ds2api_base_url
    return _onboarding_backend_metadata(provider, cfg)


def _onboarding_backend_metadata(provider: dict[str, Any], cfg: GatewayConfig) -> tuple[str, str]:
    provider_id = str(provider.get("id") or "")
    route = str(provider.get("route") or "")
    if provider_id == "deepseek-web":
        return "ds2api-sidecar", cfg.provider_runtime.deepseek_ds2api_base_url
    if route == "webai2api":
        return "webai2api-sidecar", cfg.upstream.base_url
    return "direct-http", str(provider.get("loginUrl") or "").rstrip("/")


def _recommended_connection_profile(profiles: list[dict[str, Any]]) -> dict[str, Any] | None:
    for profile in profiles:
        if profile.get("authorized") and profile.get("available"):
            return profile
    for profile in profiles:
        if profile.get("available"):
            return profile
    return profiles[0] if profiles else None


def _safe_native_file(base_dir: Path, relative_path: str) -> FileResponse:
    base = base_dir.resolve()
    target = (base / relative_path).resolve()
    if base != target and base not in target.parents:
        raise HTTPException(status_code=403, detail="Forbidden")
    if not target.exists() or not target.is_file():
        raise HTTPException(status_code=404, detail="Native UI asset is missing")
    return _native_file_response(target)


def _native_file_response(path: Path) -> FileResponse:
    return FileResponse(path, headers={"Cache-Control": "no-store, max-age=0", "Pragma": "no-cache"})


def _sidecar_root_from_base_url(base_url: str) -> str:
    parsed = urlsplit(base_url)
    path = parsed.path.rstrip("/")
    if path.endswith("/v1"):
        path = path[:-3]
    root = urlunsplit((parsed.scheme, parsed.netloc, path.rstrip("/"), "", ""))
    return root.rstrip("/")


def _wait_for_webai2api_config_instances(
    client: httpx.Client,
    cfg: GatewayConfig,
    *,
    timeout_seconds: float = 0.0,
) -> bool:
    url = f"{_sidecar_root_from_base_url(cfg.upstream.base_url)}/admin/config/instances"
    deadline = time.monotonic() + max(0.0, timeout_seconds)
    while True:
        try:
            response = client.get(url, headers=upstream_headers(cfg))
            if response.status_code == 200:
                return True
            if response.status_code not in {502, 503, 504}:
                return False
        except httpx.HTTPError:
            pass
        if time.monotonic() >= deadline:
            return False
        time.sleep(0.5)


def _default_webai2api_sidecar_dir(gateway_root: Path) -> Path:
    configured = os.environ.get("WEBAI2API_SIDECAR_DIR")
    if configured and configured.strip():
        return Path(configured).expanduser()
    return gateway_root.parent / "WebAI2API-sidecar"


def _start_webai2api_sidecar_process(gateway_root: Path) -> dict[str, Any]:
    sidecar_dir = _default_webai2api_sidecar_dir(gateway_root).resolve()
    if not (sidecar_dir / "package.json").exists():
        raise RuntimeError(f"未找到 WebAI2API sidecar：{sidecar_dir}")
    corepack = shutil.which("corepack.cmd") or shutil.which("corepack")
    if not corepack:
        raise RuntimeError("未找到 corepack，请先安装 Node.js/Corepack 后再启动 WebAI2API sidecar")
    log_dir = gateway_root / ".webai-gateway" / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    creationflags = getattr(subprocess, "CREATE_NO_WINDOW", 0)
    with (log_dir / "webai2api-out.log").open("ab") as stdout, (log_dir / "webai2api-err.log").open("ab") as stderr:
        process = subprocess.Popen(
            [corepack, "pnpm", "start"],
            cwd=sidecar_dir,
            stdout=stdout,
            stderr=stderr,
            stdin=subprocess.DEVNULL,
            creationflags=creationflags,
        )
    return {"started": True, "pid": process.pid, "sidecarDir": str(sidecar_dir)}


def _ensure_webai2api_sidecar_available(app: FastAPI, client: httpx.Client, cfg: GatewayConfig) -> dict[str, Any]:
    if _wait_for_webai2api_config_instances(client, cfg, timeout_seconds=0.0):
        return {"available": True, "started": False}
    gateway_root = Path(getattr(app.state, "gateway_root", Path.cwd())).resolve()
    starter = getattr(app.state, "webai2api_sidecar_starter", None) or _start_webai2api_sidecar_process
    try:
        start_result = starter(gateway_root)
    except Exception as exc:
        return {"available": False, "started": False, "message": str(exc)}
    if not isinstance(start_result, dict):
        start_result = {"started": bool(start_result)}
    if _wait_for_webai2api_config_instances(client, cfg, timeout_seconds=45.0):
        return {"available": True, **start_result}
    message = start_result.get("message")
    if not message:
        message = "WebAI2API sidecar 已尝试启动，但 45 秒内仍无法读取工作池；请检查 .webai-gateway/logs/webai2api-err.log"
    return {"available": False, **start_result, "message": message}


def _safe_webai2api_name_part(value: str) -> str:
    safe = re.sub(r"[^a-zA-Z0-9_-]+", "_", (value or "").strip()).strip("_-")
    return (safe or "account")[:40]


def _restart_webai2api_api_mode(client: httpx.Client, cfg: GatewayConfig) -> bool:
    root = _sidecar_root_from_base_url(cfg.upstream.base_url)
    try:
        response = client.post(f"{root}/admin/restart", json={"loginMode": False}, headers=upstream_headers(cfg))
    except httpx.HTTPError:
        return False
    if response.status_code >= 400:
        return False
    return _wait_for_webai2api_api_mode(client, cfg)


def _post_upstream_with_login_mode_recovery(
    client: httpx.Client,
    cfg: GatewayConfig,
    payload: dict[str, Any],
    allow_recovery: bool = True,
) -> httpx.Response:
    response = post_upstream(client, cfg, payload)
    if allow_recovery and _is_webai2api_login_mode_response(response) and _restart_webai2api_api_mode(client, cfg):
        response = post_upstream(client, cfg, payload)
    for delay_seconds in WEBAI2API_BROWSER_READY_RETRY_DELAYS_SECONDS:
        if not _is_webai2api_transient_browser_not_ready_response(response):
            break
        time.sleep(delay_seconds)
        response = post_upstream(client, cfg, payload)
    return response


def _is_webai2api_transient_browser_not_ready_response(response: httpx.Response) -> bool:
    if response.status_code not in {500, 502, 503, 504}:
        return False
    preview = _extract_upstream_error_preview(response)
    lowered = preview.lower()
    has_browser_not_ready_marker = any(
        marker in preview or marker in lowered for marker in WEBAI2API_BROWSER_NOT_READY_MARKERS
    )
    if has_browser_not_ready_marker:
        return True
    return False


def _webai2api_login_mode_recovery_allowed(app: FastAPI) -> bool:
    started_at = getattr(app.state, "webai2api_login_mode_started_at", None)
    if not isinstance(started_at, (int, float)):
        return True
    return time.time() - started_at > 30 * 60


def _wait_for_webai2api_api_mode(client: httpx.Client, cfg: GatewayConfig) -> bool:
    models_url = f"{cfg.upstream.base_url.rstrip('/')}/models"
    for _ in range(24):
        try:
            response = client.get(models_url, headers=upstream_headers(cfg))
            if response.status_code == 200:
                return True
            if response.status_code != 503 or not _is_webai2api_login_mode_response(response):
                return True
        except httpx.HTTPError:
            pass
        time.sleep(0.25)
    return False


def _is_webai2api_login_mode_response(response: httpx.Response) -> bool:
    if response.status_code != 503:
        return False
    preview = _extract_upstream_error_preview(response)
    return "登录模式" in preview and "OpenAI API" in preview and "不可用" in preview


async def _proxy_to_sidecar(
    client: httpx.Client,
    request: Request,
    root_url: str,
    prefix: str,
    path: str,
    *,
    sidecar_authorization: str | None = None,
    gateway_authorization: str | None = None,
) -> Response:
    suffix = f"{prefix}/{path}".rstrip("/")
    target = f"{root_url}/{suffix}"
    if request.url.query:
        target = f"{target}?{request.url.query}"
    body = await request.body()
    headers = {
        key: value
        for key, value in request.headers.items()
        if key.lower() not in {"host", "content-length", "connection", "transfer-encoding"}
    }
    if sidecar_authorization:
        auth_key = next((key for key in headers if key.lower() == "authorization"), "authorization")
        existing_authorization = headers.get(auth_key)
        if not existing_authorization or existing_authorization == gateway_authorization:
            if auth_key in headers:
                headers.pop(auth_key, None)
            headers["authorization"] = sidecar_authorization
    try:
        upstream_response = await run_in_threadpool(client.request, request.method, target, content=body, headers=headers)
    except httpx.HTTPError:
        return JSONResponse(
            {
                "detail": {
                    "code": "webai2api_sidecar_unavailable",
                    "message": "WebAI2API sidecar 未启动或无法连接，请先启动 sidecar 后再刷新状态。",
                    "upstream": target,
                }
            },
            status_code=502,
        )
    response_headers = {
        key: value
        for key, value in upstream_response.headers.items()
        if key.lower() not in {"content-length", "connection", "transfer-encoding"}
    }
    return Response(content=upstream_response.content, status_code=upstream_response.status_code, headers=response_headers)


def _record_tool_bridge_event(app: FastAPI, kind: str, **fields: Any) -> None:
    events = getattr(app.state, "tool_bridge_events", None)
    if events is None:
        return
    event = {"at": utc_now(), "kind": kind}
    for key, value in fields.items():
        if value in (None, "", [], {}):
            continue
        event[key] = value
    events.append(event)


def _record_request_diagnostic(app: FastAPI, kind: str, **fields: Any) -> None:
    events = getattr(app.state, "request_diagnostics", None)
    if events is None:
        return
    sequence = 0
    lock = getattr(app.state, "request_diagnostic_lock", None)
    if lock is not None:
        with lock:
            sequence = int(getattr(app.state, "request_diagnostic_sequence", 0) or 0) + 1
            app.state.request_diagnostic_sequence = sequence
    else:
        sequence = int(getattr(app.state, "request_diagnostic_sequence", 0) or 0) + 1
        app.state.request_diagnostic_sequence = sequence
    event = {"at": utc_now(), "kind": kind, "diagnosticSeq": sequence}
    for key, value in fields.items():
        if value in (None, "", [], {}):
            continue
        event[key] = value
    events.append(event)


def _auto_research_candidate_report(app: FastAPI, *, limit: int = 20) -> dict[str, Any]:
    candidates: list[dict[str, Any]] = []
    seen: set[tuple[str, str, str]] = set()

    for event in reversed(list(getattr(app.state, "tool_bridge_events", []) or [])):
        error = str(event.get("errorKind") or "")
        if not error:
            continue
        preview = _candidate_preview(event, ("rawPreview", "commandPreview", "errorMessage", "warning"))
        key = ("tool_bridge_event", error, preview)
        if key in seen:
            continue
        seen.add(key)
        candidates.append(
            {
                "at": event.get("at"),
                "source": "tool_bridge_event",
                "kind": event.get("kind"),
                "stage": event.get("stage"),
                "model": event.get("model"),
                "error": error,
                "repairable": bool(event.get("repairable")),
                "preview": preview,
            }
        )

    for event in reversed(list(getattr(app.state, "request_diagnostics", []) or [])):
        error = str(event.get("toolBridgeError") or "")
        if not error:
            continue
        preview = _candidate_preview(event, ("responseContentPreview", "toolBridgeMessage", "toolBridgeWarning"))
        key = ("request_diagnostic", error, preview)
        if key in seen:
            continue
        seen.add(key)
        candidates.append(
            {
                "at": event.get("at"),
                "source": "request_diagnostic",
                "kind": event.get("kind"),
                "route": event.get("route"),
                "model": event.get("model"),
                "error": error,
                "repairable": bool(event.get("toolBridgeRepairable")),
                "controllerState": event.get("toolBridgeControllerState"),
                "controllerReason": event.get("toolBridgeControllerReason"),
                "semanticVerdict": event.get("semanticFinalJudgeVerdict"),
                "preview": preview,
            }
        )

    return {
        "total": len(candidates),
        "candidates": candidates[:limit],
        "message": "这些候选来自最近运行时诊断，确认后可沉淀为 replay fixture。",
    }


def _candidate_preview(event: dict[str, Any], keys: tuple[str, ...]) -> str:
    for key in keys:
        value = event.get(key)
        if isinstance(value, str) and value.strip():
            return _preview_text(_redact_sensitive_text(value), max_chars=500)
    return ""


def _record_completion_diagnostic(
    app: FastAPI,
    *,
    endpoint: str,
    route: str,
    model: str,
    body: dict[str, Any],
    stream: bool,
    bridge: bool,
    **response_fields: Any,
) -> None:
    _record_request_diagnostic(
        app,
        "completion_response",
        endpoint=endpoint,
        route=route,
        model=model,
        stream=stream,
        bridge=bridge,
        requestFingerprint=_completion_request_fingerprint(
            endpoint=endpoint,
            route=route,
            model=model,
            body=body,
            stream=stream,
            bridge=bridge,
        ),
        **_request_body_diagnostic_fields(body),
        **response_fields,
    )


def _record_completion_started_diagnostic(
    app: FastAPI,
    *,
    endpoint: str,
    route: str,
    model: str,
    body: dict[str, Any],
    stream: bool,
    bridge: bool,
    bridge_context: Any | None = None,
) -> None:
    _record_request_diagnostic(
        app,
        "completion_request_started",
        endpoint=endpoint,
        route=route,
        model=model,
        stream=stream,
        bridge=bridge,
        requestFingerprint=_completion_request_fingerprint(
            endpoint=endpoint,
            route=route,
            model=model,
            body=body,
            stream=stream,
            bridge=bridge,
        ),
        **_request_body_diagnostic_fields(body),
        **_tool_bridge_context_diagnostic_fields(bridge_context),
    )


def _record_completion_error_diagnostic(
    app: FastAPI,
    *,
    endpoint: str,
    route: str,
    model: str,
    body: dict[str, Any],
    stream: bool,
    bridge: bool,
    status_code: int,
    error_kind: str,
    error: BaseException,
    provider_diagnostic: Any = None,
    bridge_context: Any | None = None,
) -> None:
    _record_request_diagnostic(
        app,
        "completion_error",
        endpoint=endpoint,
        route=route,
        model=model,
        stream=stream,
        bridge=bridge,
        requestFingerprint=_completion_request_fingerprint(
            endpoint=endpoint,
            route=route,
            model=model,
            body=body,
            stream=stream,
            bridge=bridge,
        ),
        statusCode=status_code,
        errorKind=error_kind,
        errorPreview=_preview_text(_redact_sensitive_text(str(error)), max_chars=500),
        **_request_body_diagnostic_fields(body),
        **_tool_bridge_context_diagnostic_fields(bridge_context),
        **_provider_diagnostic_fields(provider_diagnostic),
    )


def _openai_completion_error_response(
    app: FastAPI,
    *,
    endpoint: str,
    body: dict[str, Any],
    route: str,
    model: str,
    stream: bool,
    bridge: bool,
    status_code: int,
    diagnostic_status_code: int,
    error_kind: str,
    message: str,
    bridge_context: Any | None = None,
) -> JSONResponse:
    sanitized = _preview_text(_redact_sensitive_text(message), max_chars=800)
    _record_completion_error_diagnostic(
        app,
        endpoint=endpoint,
        route=route,
        model=model,
        body=body,
        stream=stream,
        bridge=bridge,
        status_code=diagnostic_status_code,
        error_kind=error_kind,
        error=RuntimeError(sanitized),
        bridge_context=bridge_context,
    )
    return JSONResponse(
        {
            "error": {
                "message": sanitized,
                "type": _openai_error_type(status_code),
                "code": error_kind,
                "param": None,
            }
        },
        status_code=status_code,
    )


def _openai_upstream_http_error_response(
    app: FastAPI,
    *,
    endpoint: str,
    body: dict[str, Any],
    route: str,
    model: str,
    stream: bool,
    bridge: bool,
    response: httpx.Response,
    bridge_context: Any | None = None,
) -> JSONResponse:
    status_code = _gateway_status_code_for_upstream_http_error(response)
    return _openai_completion_error_response(
        app,
        endpoint=endpoint,
        body=body,
        route=route,
        model=model,
        stream=stream,
        bridge=bridge,
        status_code=status_code,
        diagnostic_status_code=response.status_code,
        error_kind="upstream_http_error",
        message=_upstream_http_error_message(response),
        bridge_context=bridge_context,
    )


def _openai_error_type(status_code: int) -> str:
    if status_code == 400:
        return "invalid_request_error"
    if status_code == 401:
        return "authentication_error"
    if status_code == 403:
        return "permission_error"
    if status_code == 429:
        return "rate_limit_error"
    if status_code == 503:
        return "service_unavailable_error"
    if status_code >= 500:
        return "api_error"
    return "invalid_request_error"


def _gateway_status_code_for_upstream_http_error(response: httpx.Response) -> int:
    if _is_webai2api_model_unavailable_response(response):
        return 400
    return 502


def _is_webai2api_model_unavailable_response(response: httpx.Response) -> bool:
    if response.status_code not in {400, 403, 500, 502, 503, 504}:
        return False
    preview = _extract_upstream_error_preview(response)
    lowered = preview.lower()
    return any(marker in preview or marker in lowered for marker in WEBAI2API_MODEL_UNAVAILABLE_MARKERS)


def _upstream_http_error_message(response: httpx.Response) -> str:
    preview = _extract_upstream_error_preview(response)
    if not preview:
        preview = response.reason_phrase or "empty response body"
    return f"WebAI2API upstream returned HTTP {response.status_code}: {preview}"


def _extract_upstream_error_preview(response: httpx.Response) -> str:
    try:
        data = response.json()
    except ValueError:
        data = None
    if isinstance(data, dict):
        extracted = _extract_error_message_from_payload(data)
        if extracted:
            return _preview_text(_redact_sensitive_text(extracted), max_chars=700)
        return _preview_text(_redact_sensitive_text(json.dumps(data, ensure_ascii=False)), max_chars=700)
    text = _strip_html_for_error_preview(response.text or "")
    return _preview_text(_redact_sensitive_text(text), max_chars=700)


def _extract_error_message_from_payload(payload: dict[str, Any]) -> str:
    error = payload.get("error")
    if isinstance(error, dict):
        message = error.get("message") or error.get("detail") or error.get("code")
        if isinstance(message, str):
            return message
        return json.dumps(error, ensure_ascii=False)
    if isinstance(error, str):
        return error
    detail = payload.get("detail")
    if isinstance(detail, str):
        return detail
    message = payload.get("message")
    if isinstance(message, str):
        return message
    return ""


def _strip_html_for_error_preview(text: str) -> str:
    stripped = re.sub(r"(?is)<(?:script|style)\b[^>]*>.*?</(?:script|style)>", " ", text or "")
    stripped = re.sub(r"(?s)<[^>]+>", " ", stripped)
    return stripped


def _upstream_sse_error_message(text: str) -> str:
    for raw_line in (text or "").splitlines():
        line = raw_line.strip()
        if not line.startswith("data:"):
            continue
        data_text = line[5:].strip()
        if not data_text or data_text == "[DONE]":
            continue
        try:
            payload = json.loads(data_text)
        except ValueError:
            continue
        if isinstance(payload, dict):
            message = _extract_error_message_from_payload(payload)
            if message:
                return message
    return ""


def _openai_json_response(
    app: FastAPI,
    *,
    body: dict[str, Any],
    route: str,
    model: str,
    bridge: bool,
    parsed: dict[str, Any],
    bridge_result: Any,
    provider_diagnostic: Any = None,
    bridge_context: Any | None = None,
) -> JSONResponse:
    _remember_openai_tool_calls(app, parsed)
    _record_completion_diagnostic(
        app,
        endpoint="/v1/chat/completions",
        route=route,
        model=model,
        body=body,
        stream=False,
        bridge=bridge,
        **_openai_message_response_fields(parsed, response_kind="json"),
        **_provider_diagnostic_fields(provider_diagnostic),
        **_tool_bridge_completion_fields(bridge_result),
        **_tool_bridge_context_diagnostic_fields(bridge_context),
    )
    return JSONResponse(parsed, headers=bridge_error_headers(bridge_result))


def _remember_openai_tool_calls(app: FastAPI, parsed: dict[str, Any]) -> None:
    registry = getattr(app.state, "tool_call_registry", None)
    if not isinstance(registry, OrderedDict):
        registry = OrderedDict()
        app.state.tool_call_registry = registry
    for call in _openai_tool_calls_from_response(parsed):
        call_id = str(call.get("id") or "").strip()
        function = call.get("function") if isinstance(call.get("function"), dict) else {}
        name = str(function.get("name") or "").strip()
        if not call_id or not name:
            continue
        arguments = function.get("arguments")
        if not isinstance(arguments, str):
            arguments = json.dumps(arguments if isinstance(arguments, dict) else {}, ensure_ascii=False)
        for registry_id in _tool_call_registry_ids(call_id):
            registry[registry_id] = {
                "id": registry_id,
                "type": "function",
                "function": {
                    "name": name,
                    "arguments": arguments,
                },
            }
            registry.move_to_end(registry_id)
    while len(registry) > TOOL_CALL_REGISTRY_LIMIT:
        registry.popitem(last=False)


def _tool_call_registry_ids(call_id: str) -> tuple[str, ...]:
    raw = str(call_id or "").strip()
    if not raw:
        return ()
    anthropic_id = _anthropic_tool_use_id(raw, 0)
    if anthropic_id == raw:
        return (raw,)
    return (raw, anthropic_id)


def _openai_tool_calls_from_response(parsed: dict[str, Any]) -> list[dict[str, Any]]:
    if not isinstance(parsed, dict):
        return []
    choices = parsed.get("choices") if isinstance(parsed.get("choices"), list) else []
    out: list[dict[str, Any]] = []
    for choice in choices:
        if not isinstance(choice, dict):
            continue
        message = choice.get("message") if isinstance(choice.get("message"), dict) else {}
        tool_calls = message.get("tool_calls") if isinstance(message.get("tool_calls"), list) else []
        out.extend(call for call in tool_calls if isinstance(call, dict))
    return out


def _openai_stream_response(
    app: FastAPI,
    *,
    body: dict[str, Any],
    route: str,
    model: str,
    bridge: bool,
    sse_body: str,
    provider_diagnostic: Any = None,
    bridge_result: Any = None,
    bridge_context: Any | None = None,
    content_type: str = "text/event-stream",
) -> Response:
    _record_completion_diagnostic(
        app,
        endpoint="/v1/chat/completions",
        route=route,
        model=model,
        body=body,
        stream=True,
        bridge=bridge,
        **_openai_sse_response_fields(sse_body),
        **_provider_diagnostic_fields(provider_diagnostic),
        **_tool_bridge_completion_fields(bridge_result),
        **_tool_bridge_context_diagnostic_fields(bridge_context),
    )
    return Response(sse_body, media_type=content_type or "text/event-stream", headers=bridge_error_headers(bridge_result))


def _provider_diagnostic_fields(diagnostic: Any) -> dict[str, Any]:
    if not isinstance(diagnostic, dict):
        return {}
    mapping = {
        "prompt_chars": "providerPromptChars",
        "prompt_max_chars": "providerPromptMaxChars",
        "prompt_compacted": "providerPromptCompacted",
        "prompt_task_state_preserved": "providerPromptTaskStatePreserved",
        "prompt_task_state_chars": "providerPromptTaskStateChars",
        "prompt_task_count": "providerPromptTaskCount",
        "prompt_recent_tool_call_count": "providerRecentToolCallCount",
        "prompt_compaction_strategy": "providerPromptCompactionStrategy",
        "prompt_history_entry_count": "providerPromptHistoryEntryCount",
        "prompt_latest_entry_count": "providerPromptLatestEntryCount",
        "current_task_anchor_chars": "providerCurrentTaskAnchorChars",
        "message_count": "providerMessageCount",
        "artifacts_enabled": "providerArtifactsEnabled",
        "mcp_enabled": "providerMcpEnabled",
        "thinking_enabled": "providerThinkingEnabled",
        "stream_events": "providerStreamEvents",
        "json_events": "providerJsonEvents",
        "output_chars": "providerOutputChars",
        "think_chars": "providerThinkChars",
        "artifact_chars": "providerArtifactChars",
        "metadata_only_response": "providerMetadataOnlyResponse",
        "metadata_retry_count": "providerMetadataRetryCount",
        "metadata_retry_succeeded": "providerMetadataRetrySucceeded",
        "model": "providerModel",
        "ds2api_base_url": "providerBaseUrl",
    }
    fields: dict[str, Any] = {}
    for source_key, target_key in mapping.items():
        if source_key not in diagnostic:
            continue
        value = diagnostic[source_key]
        if isinstance(value, (bool, int, float, str)):
            fields[target_key] = value
    output_preview = diagnostic.get("output_preview") or diagnostic.get("response_preview")
    if isinstance(output_preview, str) and output_preview.strip():
        fields["providerOutputPreview"] = _preview_text(_redact_sensitive_text(output_preview), max_chars=220)
    return fields


def _completion_request_fingerprint(
    *,
    endpoint: str,
    route: str,
    model: str,
    body: dict[str, Any],
    stream: bool,
    bridge: bool,
) -> str:
    material = {
        "endpoint": endpoint,
        "route": route,
        "model": model,
        "stream": bool(stream),
        "bridge": bool(bridge),
        "request": _request_body_diagnostic_fields(body),
    }
    return stable_json_hash(material)[:16]


def _tool_bridge_completion_fields(bridge_result: Any) -> dict[str, Any]:
    if bridge_result is None:
        return {}
    fields: dict[str, Any] = {}
    error = getattr(bridge_result, "error", None)
    if error is not None:
        fields["toolBridgeError"] = str(getattr(error, "kind", "") or "")
        fields["toolBridgeRepairable"] = bool(getattr(error, "repairable", False))
        fields["toolBridgeMessage"] = _preview_text(str(getattr(error, "message", "") or ""), max_chars=360)
        raw_content = str(getattr(bridge_result, "raw_content", "") or "")
        if raw_content:
            fields["toolBridgeRawContentPreview"] = _preview_text(
                _redact_sensitive_text(raw_content),
                max_chars=360,
            )
    warning = getattr(bridge_result, "warning", None)
    if warning:
        fields["toolBridgeWarning"] = _preview_text(str(warning), max_chars=360)
    tool_calls = getattr(bridge_result, "tool_calls", None)
    if isinstance(tool_calls, list) and tool_calls:
        fields["toolBridgeToolCount"] = len(tool_calls)
        fields["toolBridgeTools"] = [str(getattr(call, "name", "") or "") for call in tool_calls[:16]]
    controller_state = str(getattr(bridge_result, "controller_state", "") or "")
    if controller_state:
        fields["toolBridgeControllerState"] = controller_state
    controller_reason = str(getattr(bridge_result, "controller_reason", "") or "")
    if controller_reason:
        fields["toolBridgeControllerReason"] = controller_reason
    retry_budget = str(getattr(bridge_result, "controller_retry_budget", "") or "")
    if retry_budget:
        fields["toolBridgeRetryBudget"] = retry_budget
    judge_mode = str(getattr(bridge_result, "semantic_final_judge_mode", "") or "")
    if judge_mode:
        fields["semanticFinalJudgeMode"] = judge_mode
        fields["semanticFinalJudgeVerdict"] = str(getattr(bridge_result, "semantic_final_judge_verdict", "") or "")
        confidence = getattr(bridge_result, "semantic_final_judge_confidence", None)
        if isinstance(confidence, (int, float)):
            fields["semanticFinalJudgeConfidence"] = float(confidence)
        fields["semanticFinalJudgeReason"] = str(getattr(bridge_result, "semantic_final_judge_reason", "") or "")
    return fields


def _request_body_diagnostic_fields(body: dict[str, Any]) -> dict[str, Any]:
    fields: dict[str, Any] = {}
    messages = body.get("messages")
    if isinstance(messages, list):
        fields["requestMessageCount"] = len(messages)
        shapes = _request_message_shapes(messages)
        if shapes:
            fields["requestMessageShapes"] = shapes
        tool_result_count = _request_tool_result_count(messages)
        if tool_result_count:
            fields["requestToolResultCount"] = tool_result_count
        tool_error_count, tool_error_preview = _request_tool_result_error_summary(messages)
        if tool_error_count:
            fields["requestToolErrorCount"] = tool_error_count
            fields["requestLatestToolErrorPreview"] = _preview_text(
                _redact_sensitive_text(tool_error_preview),
                max_chars=220,
            )
        latest_role, latest_text = _latest_message_text(messages)
        if latest_text:
            fields["requestLatestRole"] = latest_role
            fields["requestLatestTextChars"] = len(latest_text)
            fields["requestLatestTextPreview"] = _preview_text(_redact_sensitive_text(latest_text), max_chars=220)
        latest_user_text = _latest_user_message_text(messages)
        if latest_user_text:
            fields["requestLatestUserChars"] = len(latest_user_text)
            fields["requestLatestUserPreview"] = _preview_text(_redact_sensitive_text(latest_user_text), max_chars=220)
    tools = body.get("tools")
    if isinstance(tools, list):
        fields["requestToolCount"] = len(tools)
        names = _openai_tool_names(tools)
        if names:
            fields["requestToolNames"] = names[:16]
            if len(names) > 16:
                fields["requestToolNamesTruncated"] = len(names) - 16
        schema_summaries = _openai_tool_schema_summaries(tools)
        if schema_summaries:
            fields["requestToolSchemas"] = schema_summaries[:16]
    tool_choice = body.get("tool_choice")
    if isinstance(tool_choice, str):
        fields["requestToolChoice"] = tool_choice
    elif isinstance(tool_choice, dict):
        fields["requestToolChoice"] = _preview_text(json.dumps(tool_choice, ensure_ascii=False), max_chars=160)
    max_tokens = body.get("max_tokens")
    if isinstance(max_tokens, int):
        fields["requestMaxTokens"] = max_tokens
    return fields


def _latest_message_text(messages: list[Any]) -> tuple[str, str]:
    for message in reversed(messages):
        if not isinstance(message, dict):
            continue
        text = _diagnostic_text(message.get("content")).strip()
        if text:
            return str(message.get("role") or ""), text
    return "", ""


def _latest_user_message_text(messages: list[Any]) -> str:
    for message in reversed(messages):
        if not isinstance(message, dict) or str(message.get("role") or "") != "user":
            continue
        text = _diagnostic_text(message.get("content")).strip()
        if text:
            return text
    return ""


def _request_tool_result_count(messages: list[Any]) -> int:
    count = 0
    for message in messages:
        if not isinstance(message, dict):
            continue
        if str(message.get("role") or "") == "tool":
            count += 1
            continue
        content = message.get("content")
        if isinstance(content, list) and any(
            isinstance(block, dict) and block.get("type") == "tool_result" for block in content
        ):
            count += 1
    return count


def _request_tool_result_error_summary(messages: list[Any]) -> tuple[int, str]:
    count = 0
    latest = ""
    for message in messages:
        if not isinstance(message, dict):
            continue
        if str(message.get("role") or "") == "tool":
            content = _diagnostic_text(message.get("content"))
            is_error = bool(message.get("is_error")) or _looks_like_tool_result_error(content)
            if is_error:
                count += 1
                latest = content or latest
            continue
        content = message.get("content")
        if not isinstance(content, list):
            continue
        for block in content:
            if not isinstance(block, dict) or block.get("type") != "tool_result":
                continue
            block_text = _diagnostic_text(block.get("content"))
            is_error = bool(block.get("is_error")) or _looks_like_tool_result_error(block_text)
            if is_error:
                count += 1
                latest = block_text or latest
    return count, latest


def _looks_like_tool_result_error(text: str) -> bool:
    lowered = (text or "").lower()
    if not lowered:
        return False
    markers = (
        "invalid tool parameters",
        "tool error",
        "is_error: true",
        "validationerror",
        "validation error",
        "missing required",
        "missing required parameter",
        "参数无效",
        "工具参数",
        "缺少必需参数",
        "校验失败",
    )
    return any(marker in lowered for marker in markers)


def _tool_bridge_context_diagnostic_fields(bridge_context: Any | None) -> dict[str, Any]:
    if bridge_context is None:
        return {}
    fields: dict[str, Any] = {}
    tools = getattr(bridge_context, "tools", None)
    if isinstance(tools, list):
        names = [str(getattr(tool, "name", "") or "") for tool in tools if str(getattr(tool, "name", "") or "")]
        fields["toolBridgeAllowedToolCount"] = len(names)
        if names:
            fields["toolBridgeAllowedTools"] = names[:32]
            if len(names) > 32:
                fields["toolBridgeAllowedToolsTruncated"] = len(names) - 32
    fields["toolBridgeHasToolLoop"] = bool(getattr(bridge_context, "has_tool_loop", False))
    recent_names = getattr(bridge_context, "recent_tool_call_names", ())
    if isinstance(recent_names, tuple) and recent_names:
        fields["toolBridgeRecentToolCalls"] = [str(name) for name in recent_names[-8:]]
    task_text = str(getattr(bridge_context, "task_text", "") or "")
    if task_text:
        fields["toolBridgeTaskChars"] = len(task_text)
        fields["toolBridgeTaskPreview"] = _preview_text(_redact_sensitive_text(task_text), max_chars=220)
    options = getattr(bridge_context, "options", None)
    exposure_policy = str(getattr(options, "exposure_policy", "") or "")
    if exposure_policy:
        fields["toolBridgeExposurePolicy"] = exposure_policy
    tool_profile = str(getattr(options, "tool_profile", "") or "")
    if tool_profile:
        fields["toolBridgeToolProfile"] = tool_profile
    return fields


def _request_message_shapes(messages: list[Any]) -> list[dict[str, Any]]:
    shapes: list[dict[str, Any]] = []
    for message in messages[-12:]:
        if not isinstance(message, dict):
            shapes.append({"type": type(message).__name__})
            continue
        shape: dict[str, Any] = {"role": str(message.get("role") or "")}
        content = message.get("content")
        if isinstance(content, str):
            shape["contentType"] = "text"
            shape["contentChars"] = len(content)
        elif isinstance(content, list):
            shape["contentType"] = "blocks"
            block_types = [
                str(block.get("type") or "")
                for block in content
                if isinstance(block, dict) and str(block.get("type") or "")
            ]
            if block_types:
                shape["blockTypes"] = block_types[:8]
        elif isinstance(content, dict):
            shape["contentType"] = "object"
            block_type = str(content.get("type") or "")
            if block_type:
                shape["blockTypes"] = [block_type]
        elif content is None:
            shape["contentType"] = "none"
        else:
            shape["contentType"] = type(content).__name__
        tool_calls = message.get("tool_calls")
        if isinstance(tool_calls, list) and tool_calls:
            shape["toolCallCount"] = len(tool_calls)
            names: list[str] = []
            for call in tool_calls[:8]:
                if not isinstance(call, dict):
                    continue
                function = call.get("function") if isinstance(call.get("function"), dict) else {}
                name = function.get("name") or call.get("name")
                if isinstance(name, str) and name:
                    names.append(name)
            if names:
                shape["toolCallNames"] = names
        tool_call_id = None
        if str(message.get("role") or "") == "tool":
            name = message.get("name")
            if isinstance(name, str) and name:
                shape["toolName"] = name
            tool_call_id = message.get("tool_call_id")
            if isinstance(tool_call_id, str) and tool_call_id:
                shape["hasToolCallId"] = True
            if message.get("is_error") is True:
                shape["isError"] = True
        if isinstance(content, list) and any(
            isinstance(block, dict) and block.get("type") == "tool_result" and block.get("is_error") is True
            for block in content
        ):
            shape["hasToolError"] = True
        shapes.append(shape)
    return shapes


def _openai_tool_names(tools: list[Any]) -> list[str]:
    names: list[str] = []
    for tool in tools:
        if not isinstance(tool, dict):
            continue
        function = tool.get("function") if isinstance(tool.get("function"), dict) else {}
        name = function.get("name") or tool.get("name")
        if isinstance(name, str) and name:
            names.append(name)
    return names


def _openai_tool_schema_summaries(tools: list[Any]) -> list[dict[str, Any]]:
    summaries: list[dict[str, Any]] = []
    for tool in tools[:16]:
        if not isinstance(tool, dict):
            continue
        function = tool.get("function") if isinstance(tool.get("function"), dict) else {}
        name = function.get("name") or tool.get("name")
        if not isinstance(name, str) or not name:
            continue
        schema = (
            function.get("parameters")
            if isinstance(function.get("parameters"), dict)
            else tool.get("input_schema")
            if isinstance(tool.get("input_schema"), dict)
            else {}
        )
        summary: dict[str, Any] = {"name": name}
        if isinstance(schema, dict):
            required = schema.get("required")
            if isinstance(required, list):
                required_names = [str(item) for item in required if isinstance(item, str)]
                if required_names:
                    summary["required"] = required_names[:12]
            properties = schema.get("properties")
            if isinstance(properties, dict):
                summary["properties"] = [str(key) for key in list(properties)[:16]]
            if schema.get("additionalProperties") is False:
                summary["additionalProperties"] = False
        summaries.append(summary)
    return summaries


def _openai_message_response_fields(data: dict[str, Any], *, response_kind: str) -> dict[str, Any]:
    choices = data.get("choices") if isinstance(data.get("choices"), list) else []
    choice0 = choices[0] if choices and isinstance(choices[0], dict) else {}
    msg = choice0.get("message") if isinstance(choice0.get("message"), dict) else {}
    content = _diagnostic_text(msg.get("content"))
    tool_calls = msg.get("tool_calls") if isinstance(msg.get("tool_calls"), list) else []
    fields: dict[str, Any] = {
        "responseKind": response_kind,
        "responseChoiceCount": len(choices),
        "responseContentChars": len(content),
        "responseToolCallCount": len(tool_calls),
    }
    finish_reason = choice0.get("finish_reason")
    if isinstance(finish_reason, str):
        fields["finishReason"] = finish_reason
    if content:
        fields["responseContentPreview"] = _preview_text(_redact_sensitive_text(content), max_chars=220)
    if tool_calls:
        fields["responseToolNames"] = _tool_call_names(tool_calls)
    if not content and not tool_calls:
        fields["responseEmpty"] = True
        fields["responseWarning"] = "empty_assistant_response"
    return fields


def _openai_sse_response_fields(text: str) -> dict[str, Any]:
    payload_count = 0
    content_parts: list[str] = []
    tool_call_chunks = 0
    tool_call_names: list[str] = []
    finish_reason = ""
    done = False
    for block in (text or "").split("\n\n"):
        for line in block.splitlines():
            line = line.strip()
            if not line.startswith("data:"):
                continue
            raw = line[5:].strip()
            if raw == "[DONE]":
                done = True
                continue
            try:
                data = json.loads(raw)
            except Exception:
                continue
            payload_count += 1
            choices = data.get("choices") if isinstance(data.get("choices"), list) else []
            choice0 = choices[0] if choices and isinstance(choices[0], dict) else {}
            if isinstance(choice0.get("finish_reason"), str):
                finish_reason = choice0["finish_reason"]
            delta = choice0.get("delta") if isinstance(choice0.get("delta"), dict) else {}
            content = _diagnostic_text(delta.get("content"))
            if content:
                content_parts.append(content)
            tool_calls = delta.get("tool_calls") if isinstance(delta.get("tool_calls"), list) else []
            tool_call_chunks += len(tool_calls)
            if tool_calls:
                for call in tool_calls:
                    if not isinstance(call, dict):
                        continue
                    function = call.get("function") if isinstance(call.get("function"), dict) else {}
                    name = function.get("name")
                    if isinstance(name, str) and name:
                        tool_call_names.append(name)
    content_text = "".join(content_parts)
    fields: dict[str, Any] = {
        "responseKind": "sse",
        "ssePayloadCount": payload_count,
        "sseDone": done,
        "responseContentChars": len(content_text),
        "responseToolCallChunkCount": tool_call_chunks,
    }
    if finish_reason:
        fields["finishReason"] = finish_reason
    if content_text:
        fields["responseContentPreview"] = _preview_text(_redact_sensitive_text(content_text), max_chars=220)
    if tool_call_names:
        fields["responseToolNames"] = tool_call_names[:16]
    if not content_text and tool_call_chunks == 0:
        fields["responseEmpty"] = True
        fields["responseWarning"] = "empty_stream_response"
    return fields


def _anthropic_message_response_fields(message: dict[str, Any], *, response_kind: str) -> dict[str, Any]:
    content_blocks = message.get("content") if isinstance(message.get("content"), list) else []
    text_parts: list[str] = []
    tool_uses = 0
    for block in content_blocks:
        if not isinstance(block, dict):
            continue
        if block.get("type") == "text":
            text_parts.append(_diagnostic_text(block.get("text")))
        elif block.get("type") == "tool_use":
            tool_uses += 1
    text = "".join(text_parts)
    fields: dict[str, Any] = {
        "responseKind": response_kind,
        "responseContentBlockCount": len(content_blocks),
        "responseContentChars": len(text),
        "responseToolUseCount": tool_uses,
    }
    stop_reason = message.get("stop_reason")
    if isinstance(stop_reason, str):
        fields["stopReason"] = stop_reason
    if text:
        fields["responseContentPreview"] = _preview_text(_redact_sensitive_text(text), max_chars=220)
    return fields


def _tool_call_names(tool_calls: list[Any]) -> list[str]:
    names: list[str] = []
    for call in tool_calls:
        if not isinstance(call, dict):
            continue
        function = call.get("function") if isinstance(call.get("function"), dict) else {}
        name = function.get("name")
        if isinstance(name, str) and name:
            names.append(name)
    return names


def _diagnostic_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    if isinstance(value, list):
        parts: list[str] = []
        for item in value:
            if isinstance(item, dict):
                if item.get("type") == "text":
                    parts.append(str(item.get("text") or ""))
                elif isinstance(item.get("text"), str):
                    parts.append(item["text"])
            elif isinstance(item, str):
                parts.append(item)
        return "".join(parts)
    return str(value)


def _tool_bridge_phase_fields(result: Any, bridge_context: Any | None = None) -> dict[str, Any]:
    phases = getattr(result, "phases", None)
    if not phases and bridge_context is not None:
        raw_content = str(getattr(result, "raw_content", "") or "")
        if raw_content:
            try:
                phases = parse_tool_response(raw_content, bridge_context).phases
            except Exception:
                phases = None
    if not phases:
        return {}
    safe_phases: list[dict[str, str]] = []
    for phase in phases[:12]:
        if isinstance(phase, dict):
            name = phase.get("name", "")
            status = phase.get("status", "")
            detail = phase.get("detail", "")
        else:
            name = getattr(phase, "name", "")
            status = getattr(phase, "status", "")
            detail = getattr(phase, "detail", "")
        safe_phase = {
            "name": _preview_text(_redact_sensitive_text(_diagnostic_text(name)), max_chars=80),
            "status": _preview_text(_redact_sensitive_text(_diagnostic_text(status)), max_chars=40),
            "detail": _preview_text(_redact_sensitive_text(_diagnostic_text(detail)), max_chars=160),
        }
        if safe_phase["name"] or safe_phase["status"] or safe_phase["detail"]:
            safe_phases.append(safe_phase)
    return {"phases": safe_phases} if safe_phases else {}


def _provider_route(model: str, config: GatewayConfig | None = None) -> str:
    model_id = str(model or "")
    if config is not None:
        if _is_gpt_thinking_model(model_id) and _gpt_thinking_uses_deepseek_ds2api(config):
            return "deepseek-web"
        if is_qwen_web_model(model_id) and _qwen_web_uses_deepseek_ds2api(config):
            return "deepseek-web"
    if is_deepseek_web_model(model_id):
        return "deepseek-web"
    if is_qwen_web_model(model_id):
        return "qwen-web"
    if is_qwen_coder_model(model_id):
        return "qwen-coder"
    return "upstream"


def _tool_call_event_fields(data: dict[str, Any]) -> dict[str, Any]:
    choices = data.get("choices") if isinstance(data.get("choices"), list) else []
    choice0 = choices[0] if choices and isinstance(choices[0], dict) else {}
    message = choice0.get("message") if isinstance(choice0.get("message"), dict) else {}
    tool_calls = message.get("tool_calls") if isinstance(message.get("tool_calls"), list) else []
    if not tool_calls or not isinstance(tool_calls[0], dict):
        return {}
    call = tool_calls[0]
    function = call.get("function") if isinstance(call.get("function"), dict) else {}
    fields: dict[str, Any] = {
        "tool": str(function.get("name") or ""),
        "finishReason": str(choice0.get("finish_reason") or ""),
    }
    raw_args = function.get("arguments")
    try:
        args = json.loads(raw_args) if isinstance(raw_args, str) and raw_args.strip() else {}
    except Exception:
        args = {}
    if isinstance(args, dict):
        command = args.get("command") if isinstance(args.get("command"), str) else args.get("cmd")
        if isinstance(command, str):
            fields["commandPreview"] = _preview_text(command)
    return fields


def _bridge_result_event_fields(
    bridge_result: Any,
    *,
    model: str,
    allowed_tools: set[str],
    bridge_context: Any | None = None,
) -> dict[str, Any]:
    fields: dict[str, Any] = {
        "model": model,
        "allowedToolCount": len(allowed_tools),
    }
    fields.update(_tool_bridge_phase_fields(bridge_result, bridge_context=bridge_context))
    if bridge_context is not None:
        task_text = str(getattr(bridge_context, "task_text", "") or "")
        if task_text:
            fields["hasTaskText"] = True
            fields["taskChars"] = len(task_text)
        options = getattr(bridge_context, "options", None)
        exposure_policy = str(getattr(options, "exposure_policy", "") or "")
        if exposure_policy:
            fields["exposurePolicy"] = exposure_policy
    if allowed_tools and len(allowed_tools) <= 32:
        fields["allowedTools"] = sorted(allowed_tools)
    elif allowed_tools:
        fields["allowedTools"] = sorted(allowed_tools)[:32]
        fields["allowedToolsTruncated"] = len(allowed_tools) - 32
    error = getattr(bridge_result, "error", None)
    if error is not None:
        fields["errorKind"] = str(getattr(error, "kind", "") or "")
        fields["errorMessage"] = _preview_text(str(getattr(error, "message", "") or ""), max_chars=360)
        fields["repairable"] = bool(getattr(error, "repairable", False))
    warning = getattr(bridge_result, "warning", None)
    if warning:
        fields["warning"] = _preview_text(str(warning), max_chars=360)
    tool_calls = getattr(bridge_result, "tool_calls", None)
    if isinstance(tool_calls, list) and tool_calls:
        fields["tools"] = [str(getattr(call, "name", "") or "") for call in tool_calls]
        command = _first_tool_command(tool_calls)
        if command:
            fields["commandPreview"] = _preview_text(command)
    raw_content = str(getattr(bridge_result, "raw_content", "") or "")
    if raw_content:
        fields["rawPreview"] = _preview_text(_redact_sensitive_text(raw_content), max_chars=500)
    return fields


def _record_bridge_parse_event(
    app: FastAPI,
    bridge_result: Any,
    *,
    model: str,
    allowed_tools: set[str],
    stage: str,
    bridge_context: Any | None = None,
) -> None:
    if getattr(bridge_result, "error", None) is not None:
        kind = "tool_bridge_error"
    elif getattr(bridge_result, "tool_calls", None):
        kind = "tool_bridge_tool_calls"
    elif getattr(bridge_result, "warning", None):
        kind = "tool_bridge_warning"
    else:
        return
    _record_tool_bridge_event(
        app,
        kind,
        stage=stage,
        **_bridge_result_event_fields(
            bridge_result,
            model=model,
            allowed_tools=allowed_tools,
            bridge_context=bridge_context,
        ),
    )


def _record_bridge_rejection_event(
    app: FastAPI,
    bridge_result: Any,
    *,
    model: str,
    allowed_tools: set[str],
    bridge_context: Any | None = None,
) -> None:
    if getattr(bridge_result, "error", None) is None:
        return
    _record_tool_bridge_event(
        app,
        "tool_bridge_rejection",
        **_bridge_result_event_fields(
            bridge_result,
            model=model,
            allowed_tools=allowed_tools,
            bridge_context=bridge_context,
        ),
    )


def _first_tool_command(tool_calls: list[Any]) -> str:
    for call in tool_calls:
        input_value = getattr(call, "input", None)
        if not isinstance(input_value, dict):
            continue
        command = input_value.get("command") if isinstance(input_value.get("command"), str) else input_value.get("cmd")
        if isinstance(command, str):
            return command
    return ""


def _redact_sensitive_text(value: str) -> str:
    redacted = _SENSITIVE_BEARER_RE.sub(r"\1[redacted]", value or "")
    return _SENSITIVE_ASSIGNMENT_RE.sub(r"\1[redacted]", redacted)


def _redact_known_sensitive_values(value: str, sensitive_values: Any) -> str:
    redacted = value or ""
    for item in sensitive_values or ():
        text = str(item or "").strip()
        if len(text) < 6:
            continue
        redacted = redacted.replace(text, "[redacted]")
        if "=" not in text and ";" not in text:
            continue
        for part in text.split(";"):
            if "=" not in part:
                continue
            cookie_value = part.split("=", 1)[1].strip()
            if len(cookie_value) >= 6:
                redacted = redacted.replace(cookie_value, "[redacted]")
    return redacted


def _preview_text(value: str, *, max_chars: int = 240) -> str:
    text = " ".join((value or "").split())
    if len(text) <= max_chars:
        return text
    return text[: max_chars - 3] + "..."


async def _json_body(request: Request) -> dict[str, Any]:
    try:
        body = await request.json()
    except ValueError as exc:
        raise HTTPException(status_code=400, detail="Request body must be valid JSON") from exc
    if not isinstance(body, dict):
        raise HTTPException(status_code=400, detail="Request body must be a JSON object")
    return body


def _create_image_generation_response(
    app: FastAPI,
    client: httpx.Client,
    cfg: GatewayConfig,
    body: dict[str, Any],
) -> dict[str, Any]:
    prompt = str(body.get("prompt") or "").strip()
    if not prompt:
        raise HTTPException(status_code=400, detail="prompt is required")
    n = int(body.get("n") or 1)
    if n != 1:
        raise HTTPException(status_code=400, detail="Only n=1 is supported for web image generation")
    response_format = str(body.get("response_format") or "url").strip().lower()
    if response_format not in {"url", "b64_json"}:
        raise HTTPException(status_code=400, detail="response_format must be url or b64_json")
    model = normalize_model_id(body.get("model"), DEFAULT_IMAGE_GENERATION_MODEL)
    input_images = _media_input_images_from_body(body)
    data_uri = _run_webai2api_media_generation(
        client,
        cfg,
        model=model,
        prompt=prompt,
        kind="image",
        input_images=input_images,
    )
    decoded = _decode_data_uri(data_uri, expected_kind="image")
    if decoded is None:
        raise HTTPException(status_code=502, detail="Upstream image response did not contain valid image data")
    _mime_type, content = decoded
    item = {"url": data_uri} if response_format == "url" else {"b64_json": base64.b64encode(content).decode("ascii")}
    return {"created": int(time.time()), "object": "list", "data": [item]}


def _create_video_generation_response(
    app: FastAPI,
    client: httpx.Client,
    cfg: GatewayConfig,
    body: dict[str, Any],
) -> dict[str, Any]:
    prompt = str(body.get("prompt") or "").strip()
    if not prompt:
        raise HTTPException(status_code=400, detail="prompt is required")
    model = normalize_model_id(body.get("model"), DEFAULT_VIDEO_GENERATION_MODEL)
    input_images = _media_input_images_from_body(body)
    data_uri = _run_webai2api_media_generation(
        client,
        cfg,
        model=model,
        prompt=prompt,
        kind="video",
        input_images=input_images,
    )
    if _decode_data_uri(data_uri, expected_kind="video") is None:
        raise HTTPException(status_code=502, detail="Upstream video response did not contain valid video data")
    item = _store_media_generation(app, model=model, data_uri=data_uri, kind="video")
    return _video_generation_public_payload(item)


def _run_webai2api_media_generation(
    client: httpx.Client,
    cfg: GatewayConfig,
    *,
    model: str,
    prompt: str,
    kind: str,
    input_images: list[str] | None = None,
) -> str:
    payload = {
        "model": model,
        "messages": [_media_user_message(prompt, input_images or [])],
        "stream": True,
    }
    response = _post_upstream_with_login_mode_recovery(client, cfg, payload, True)
    if response.status_code >= 400:
        message = _upstream_http_error_message(response)
        raise HTTPException(
            status_code=_gateway_status_code_for_upstream_http_error(response),
            detail=_preview_text(_redact_sensitive_text(message), max_chars=800),
        )
    text = response.text.strip()
    if text.startswith("data:") or "text/event-stream" in response.headers.get("content-type", ""):
        content, _finish_reason = parse_sse_text(response.text)
        upstream_error = _extract_sse_error_message(response.text)
    else:
        upstream_error = ""
        try:
            data = response.json()
        except ValueError as exc:
            raise HTTPException(status_code=502, detail="Upstream media response must be JSON or SSE") from exc
        if not isinstance(data, dict):
            raise HTTPException(status_code=502, detail="Upstream media response must be a JSON object")
        content = _openai_response_content(data)
    data_uri = _extract_media_data_uri(content, expected_kind=kind)
    if not data_uri:
        if upstream_error:
            raise HTTPException(
                status_code=_gateway_status_code_for_upstream_error_message(upstream_error),
                detail=_preview_text(_redact_sensitive_text(upstream_error), max_chars=800),
            )
        raise HTTPException(status_code=502, detail=f"Upstream {kind} response did not contain media data")
    return data_uri


def _extract_sse_error_message(text: str) -> str:
    for block in (text or "").split("\n\n"):
        for line in block.splitlines():
            line = line.strip()
            if not line.startswith("data:"):
                continue
            raw = line[5:].strip()
            if raw == "[DONE]":
                continue
            try:
                data = json.loads(raw)
            except Exception:
                continue
            if not isinstance(data, dict):
                continue
            error = data.get("error")
            if not isinstance(error, dict):
                continue
            message = error.get("message") or error.get("detail") or ""
            code = error.get("code") or ""
            if message and code:
                return f"{code}: {message}"
            if message:
                return str(message)
            if code:
                return str(code)
    return ""


def _gateway_status_code_for_upstream_error_message(message: str) -> int:
    lowered = str(message or "").lower()
    if any(marker in lowered for marker in ("invalid_model", "invalid model", "model invalid", "does not support", "不支持")):
        return 400
    return 502


def _media_user_message(prompt: str, input_images: list[str]) -> dict[str, Any]:
    if not input_images:
        return {"role": "user", "content": prompt}
    content: list[dict[str, Any]] = [{"type": "text", "text": prompt}]
    for image in input_images:
        content.append({"type": "image_url", "image_url": {"url": image}})
    return {"role": "user", "content": content}


def _media_input_images_from_body(body: dict[str, Any]) -> list[str]:
    value = (
        body.get("input_reference")
        if "input_reference" in body
        else body.get("input_image", body.get("image", body.get("reference_image")))
    )
    values = value if isinstance(value, list) else [value]
    out: list[str] = []
    for item in values:
        if isinstance(item, dict):
            item = item.get("url") or item.get("image_url") or item.get("data")
        if isinstance(item, str) and item.startswith("data:image/"):
            out.append(item)
    return out


def _extract_media_data_uri(content: str, *, expected_kind: str) -> str:
    pattern = rf"data:{re.escape(expected_kind)}/[A-Za-z0-9.+-]+;base64,[A-Za-z0-9+/=\r\n]+"
    match = re.search(pattern, str(content or ""))
    return re.sub(r"\s+", "", match.group(0)) if match else ""


def _decode_data_uri(data_uri: str, *, expected_kind: str) -> tuple[str, bytes] | None:
    match = re.match(r"^data:([^;]+);base64,(.+)$", str(data_uri or ""), re.DOTALL)
    if not match:
        return None
    mime_type = match.group(1).strip().lower()
    if not mime_type.startswith(f"{expected_kind}/"):
        return None
    try:
        return mime_type, base64.b64decode(re.sub(r"\s+", "", match.group(2)), validate=True)
    except ValueError:
        return None


def _store_media_generation(app: FastAPI, *, model: str, data_uri: str, kind: str) -> dict[str, Any]:
    store = _media_generation_store(app)
    _prune_media_generations(store)
    media_id = f"{kind}_{secrets.token_hex(12)}"
    created_at = int(time.time())
    item = {
        "id": media_id,
        "object": kind,
        "created_at": created_at,
        "expires_at": created_at + MEDIA_GENERATION_TTL_SECONDS,
        "status": "completed",
        "model": model,
        "data_uri": data_uri,
    }
    store[media_id] = item
    store.move_to_end(media_id)
    while len(store) > MEDIA_GENERATION_CACHE_LIMIT:
        store.popitem(last=False)
    return item


def _get_cached_media_generation(app: FastAPI, media_id: str) -> dict[str, Any] | None:
    store = _media_generation_store(app)
    _prune_media_generations(store)
    item = store.get(media_id)
    if not isinstance(item, dict):
        return None
    store.move_to_end(media_id)
    return item


def _media_generation_store(app: FastAPI) -> OrderedDict[str, dict[str, Any]]:
    store = getattr(app.state, "media_generations", None)
    if not isinstance(store, OrderedDict):
        store = OrderedDict()
        app.state.media_generations = store
    return store


def _prune_media_generations(store: OrderedDict[str, dict[str, Any]]) -> None:
    now = int(time.time())
    expired = [key for key, item in store.items() if int(item.get("expires_at") or 0) <= now]
    for key in expired:
        store.pop(key, None)


def _video_generation_public_payload(item: dict[str, Any]) -> dict[str, Any]:
    return {
        "id": item.get("id"),
        "object": "video",
        "created_at": item.get("created_at"),
        "expires_at": item.get("expires_at"),
        "status": item.get("status") or "completed",
        "model": item.get("model"),
    }


def _append_web_models(data: dict[str, Any], *, include_webai2api_catalog: bool = False) -> dict[str, Any]:
    items = data.get("data") if isinstance(data.get("data"), list) else []
    normalized_items: list[Any] = []
    seen: set[Any] = set()
    for item in items:
        if not isinstance(item, dict):
            normalized_items.append(item)
            continue
        model_id = normalize_model_id(item.get("id"))
        if not model_id or model_id in seen:
            continue
        seen.add(model_id)
        normalized_items.append(_enrich_model_payload({**item, "id": model_id}))
    items = normalized_items
    for item in catalog_model_payloads(include_webai2api=include_webai2api_catalog):
        model_id = item["id"]
        if model_id not in seen:
            seen.add(model_id)
            items.append(_enrich_model_payload(dict(item)))
    return {**data, "object": data.get("object") or "list", "data": items}


def _enrich_model_payload(item: dict[str, Any]) -> dict[str, Any]:
    model_id = str(item.get("id") or "")
    owner = str(item.get("owned_by") or "")
    model_type = _infer_model_type(model_id, owner, item.get("type"))
    capabilities = item.get("capabilities") if isinstance(item.get("capabilities"), dict) else {}
    out = dict(item)
    if model_type:
        out["type"] = model_type
    enriched_capabilities = dict(capabilities)
    if model_type in {"image", "video", "text"}:
        enriched_capabilities.setdefault("text", model_type == "text")
        enriched_capabilities.setdefault("image", model_type == "image")
        enriched_capabilities.setdefault("video", model_type == "video")
    elif owner:
        provider_caps = _provider_capabilities_for_owner(owner)
        for key in ("text", "image", "video"):
            if key in provider_caps:
                enriched_capabilities.setdefault(key, bool(provider_caps.get(key)))
    if model_type in {"image", "video"}:
        enriched_capabilities.setdefault("tool_bridge", False)
        enriched_capabilities.setdefault("supports_native_tools", False)
        enriched_capabilities.setdefault("preferred_protocol", "openai")
    if enriched_capabilities:
        out["capabilities"] = enriched_capabilities
    return out


def _infer_model_type(model_id: str, owner: str, raw_type: Any) -> str:
    model = str(model_id or "").strip().lower()
    owned_by = str(owner or "").strip().lower()
    explicit = str(raw_type or "").strip().lower()
    if owned_by == "sora" or model.startswith("sora") or "veo" in model or "video" in model:
        return "video"
    if explicit == "video":
        return "video"
    if explicit == "text":
        return "text"
    if explicit == "image":
        return "image"
    if any(marker in model for marker in ("gpt-image", "image", "imagen", "dall-e", "seedream", "flux", "recraft")):
        return "image"
    provider_caps = _provider_capabilities_for_owner(owned_by)
    if provider_caps.get("video") and not provider_caps.get("text") and not provider_caps.get("image"):
        return "video"
    if provider_caps.get("image") and not provider_caps.get("text"):
        return "image"
    return explicit


def _provider_capabilities_for_owner(owner: str) -> dict[str, Any]:
    owner = str(owner or "").strip()
    if not owner:
        return {}
    provider = PROVIDERS.get(owner)
    if provider is not None:
        return dict(provider.capabilities)
    for provider in PROVIDERS.values():
        if owner in provider.adapters:
            return dict(provider.capabilities)
    return {}


def _load_gateway_models(client: httpx.Client, cfg: GatewayConfig) -> dict[str, Any]:
    try:
        response = client.get(cfg.upstream.base_url.rstrip("/") + "/models", headers=upstream_headers(cfg))
        if response.status_code == 200:
            data = response.json()
            if isinstance(data, dict) and isinstance(data.get("data"), list):
                return _append_web_models(data)
    except Exception:
        pass
    return _append_web_models({"object": "list", "data": [{"id": cfg.upstream.model, "object": "model"}]})


def _append_qwen_live_model_catalog(
    app: FastAPI,
    client: httpx.Client,
    data: dict[str, Any],
    cfg: GatewayConfig,
) -> dict[str, Any]:
    return _append_extra_model_payloads(data, _qwen_live_model_payloads(app, client, cfg))


def _append_extra_model_payloads(data: dict[str, Any], payloads: list[dict[str, Any]]) -> dict[str, Any]:
    if not payloads:
        return data
    items = data.get("data") if isinstance(data.get("data"), list) else []
    out: list[Any] = []
    seen: set[Any] = set()
    for item in items:
        if not isinstance(item, dict):
            out.append(item)
            continue
        model_id = normalize_model_id(item.get("id"))
        if not model_id or model_id in seen:
            continue
        seen.add(model_id)
        out.append(_enrich_model_payload({**item, "id": model_id}))
    for item in payloads:
        if not isinstance(item, dict):
            continue
        model_id = normalize_model_id(item.get("id"))
        if not model_id or model_id in seen:
            continue
        seen.add(model_id)
        out.append(_enrich_model_payload({**item, "id": model_id}))
    return {**data, "object": data.get("object") or "list", "data": out}


def _qwen_live_model_payloads(app: FastAPI, client: httpx.Client, cfg: GatewayConfig) -> list[dict[str, Any]]:
    if app.state.qwen_client_factory is QwenWebClient and _http_client_uses_mock_transport(client):
        return []
    cache_key = _qwen_live_model_catalog_cache_key(app)
    if not cache_key:
        return []
    now = time.time()
    cache = getattr(app.state, "qwen_live_model_catalog_cache", {})
    if (
        isinstance(cache, dict)
        and cache.get("key") == cache_key
        and float(cache.get("expiresAt") or 0) > now
        and isinstance(cache.get("models"), list)
    ):
        return [dict(item) for item in cache["models"] if isinstance(item, dict)]

    last_error = ""
    for account_id in _authorized_direct_account_ids(app, "qwen"):
        credential = _direct_credential_for_account(app, "qwen", account_id)
        if not isinstance(credential, dict) or not is_credential_authorized("qwen", credential):
            continue
        try:
            web_client = _build_qwen_web_client(app.state.qwen_client_factory, credential, client, cfg)
            if not hasattr(web_client, "list_models"):
                continue
            models = web_client.list_models()
            if not isinstance(models, list):
                continue
            payloads = [dict(item) for item in models if isinstance(item, dict) and normalize_model_id(item.get("id"))]
            app.state.qwen_live_model_catalog_cache = {
                "key": cache_key,
                "expiresAt": now + QWEN_LIVE_MODEL_CATALOG_CACHE_SECONDS,
                "models": payloads,
                "accountId": account_id,
            }
            app.state.qwen_live_model_catalog_last_error = ""
            return payloads
        except Exception as exc:
            last_error = _preview_text(_redact_sensitive_text(str(exc)), max_chars=240)
            continue
    if last_error:
        app.state.qwen_live_model_catalog_last_error = last_error
    return []


def _http_client_uses_mock_transport(client: httpx.Client) -> bool:
    return isinstance(getattr(client, "_transport", None), httpx.MockTransport)


def _qwen_live_model_catalog_cache_key(app: FastAPI) -> tuple[str, ...]:
    parts: list[str] = []
    for account_id in _authorized_direct_account_ids(app, "qwen"):
        credential = _direct_credential_for_account(app, "qwen", account_id)
        if not isinstance(credential, dict) or not is_credential_authorized("qwen", credential):
            continue
        metadata = credential.get("metadata") if isinstance(credential.get("metadata"), dict) else {}
        parts.append(
            ":".join(
                (
                    account_id,
                    str(credential.get("updatedAt") or credential.get("createdAt") or ""),
                    "bearer" if credential.get("bearer") else "",
                    "session" if metadata.get("sessionToken") else "",
                )
            )
        )
    return tuple(parts)


def _direct_catalog_model_belongs_to_provider(provider_id: str, model_id: str, owner: str = "") -> bool:
    model = normalize_model_id(model_id)
    provider = str(provider_id or "").strip()
    owned_by = str(owner or "").strip()
    if provider == "qwen":
        return owned_by == "qwen" or model.startswith("qwen-web/")
    if provider == "qwen-coder":
        return owned_by == "qwen-coder" or model.startswith("qwen-coder/")
    return False


def _build_direct_payload(
    body: dict[str, Any],
    config: GatewayConfig,
    *,
    default_model: str,
    provider_native_web_search: bool = False,
) -> tuple:
    direct_body = normalize_model_body(body, default_model=default_model)
    payload, bridge, allowed_tools, bridge_context = build_upstream_payload(
        direct_body,
        config,
        provider_native_web_search=provider_native_web_search,
    )
    if bridge_context.task_text:
        payload["_webai_current_task_text"] = bridge_context.task_text
    if not bridge:
        payload.pop("tools", None)
        payload.pop("tool_choice", None)
    return payload, bridge, allowed_tools, bridge_context


def _local_preflight_response(app: FastAPI, body: dict[str, Any], model: str, bridge_context: Any) -> Response | None:
    preflight_data = build_preflight_chat_response(model, bridge_context)
    if preflight_data is None:
        return None
    _record_tool_bridge_event(app, "local_repo_preflight", model=model, **_tool_call_event_fields(preflight_data))
    if bool(body.get("stream")):
        choices = preflight_data.get("choices") if isinstance(preflight_data.get("choices"), list) else []
        msg = choices[0].get("message") if choices and isinstance(choices[0], dict) else {}
        tool_calls = msg.get("tool_calls") if isinstance(msg, dict) and isinstance(msg.get("tool_calls"), list) else []
        return Response(
            build_openai_tool_calls_sse(tool_calls, model=model),
            media_type="text/event-stream",
        )
    return JSONResponse(preflight_data)


def _skill_loader_preflight_response(
    app: FastAPI,
    model: str,
    bridge_context: Any,
    *,
    require_namespaced_slash: bool = False,
) -> Response | None:
    if require_namespaced_slash and not _looks_like_namespaced_skill_slash(getattr(bridge_context, "task_text", "")):
        return None
    preflight_data = build_skill_loader_preflight_chat_response(model, bridge_context)
    if preflight_data is None:
        return None
    _record_tool_bridge_event(app, "skill_loader_preflight", model=model, **_tool_call_event_fields(preflight_data))
    return JSONResponse(preflight_data)


def _looks_like_namespaced_skill_slash(text: str) -> bool:
    first_line = next((line.strip() for line in (text or "").splitlines() if line.strip()), "")
    if not first_line.startswith("/"):
        return False
    command = first_line.split(None, 1)[0]
    return ":" in command


def _parse_bridge_chat_data(
    data: dict[str, Any],
    *,
    app: FastAPI,
    payload: dict[str, Any],
    bridge: bool,
    allowed_tools: set[str],
    bridge_context: Any,
    model: str,
    retry_chat: Any,
    native_web_search: bool = False,
) -> tuple[dict[str, Any], Any]:
    retry_state = RetryState()
    if native_web_search and not bridge and should_retry_native_web_search_response(data):
        retry_data = retry_chat(build_native_web_search_retry_payload(payload, _openai_response_content(data)))
        if isinstance(retry_data, dict):
            data = retry_data
    parsed, bridge_result = parse_chat_response(
        data,
        bridge=bridge,
        allowed_tools=allowed_tools,
        model=model,
        bridge_context=bridge_context,
        return_bridge_result=True,
    )
    parsed, bridge_result, retry_state = _apply_controller_decision(
        parsed,
        bridge_result,
        bridge_context=bridge_context,
        model=model,
        retry_state=retry_state,
    )
    if bridge:
        _record_bridge_parse_event(
            app,
            bridge_result,
            model=model,
            allowed_tools=allowed_tools,
            stage="initial",
            bridge_context=bridge_context,
        )
    if _should_retry_required_tool_choice_recovery(bridge_result):
        _record_tool_bridge_event(
            app,
            "tool_bridge_retry",
            stage="required_tool_choice_recovery",
            model=model,
            errorKind=bridge_result.error.kind,
            errorMessage=_preview_text(bridge_result.error.message, max_chars=360),
        )
        recovery_data = retry_chat(
            build_required_tool_choice_recovery_payload(
                payload,
                bridge_result,
                allowed_tools=allowed_tools,
                bridge_context=bridge_context,
            )
        )
        if isinstance(recovery_data, dict):
            parsed, bridge_result = parse_chat_response(
                recovery_data,
                bridge=bridge,
                allowed_tools=allowed_tools,
                model=model,
                bridge_context=bridge_context,
                return_bridge_result=True,
            )
            parsed, bridge_result, retry_state = _apply_controller_decision(
                parsed,
                bridge_result,
                bridge_context=bridge_context,
                model=model,
                retry_state=retry_state,
            )
            if bridge:
                _record_bridge_parse_event(
                    app,
                    bridge_result,
                    model=model,
                    allowed_tools=allowed_tools,
                    stage="required_tool_choice_recovery",
                    bridge_context=bridge_context,
                )
    if bridge_result.error and bridge_result.error.repairable:
        _record_tool_bridge_event(
            app,
            "tool_bridge_retry",
            stage="repair",
            model=model,
            errorKind=bridge_result.error.kind,
            errorMessage=_preview_text(bridge_result.error.message, max_chars=360),
        )
        repair_data = retry_chat(build_repair_payload(payload, bridge_result, allowed_tools=allowed_tools))
        if isinstance(repair_data, dict):
            parsed, bridge_result = parse_chat_response(
                repair_data,
                bridge=bridge,
                allowed_tools=allowed_tools,
                model=model,
                bridge_context=bridge_context,
                return_bridge_result=True,
            )
            parsed, bridge_result, retry_state = _apply_controller_decision(
                parsed,
                bridge_result,
                bridge_context=bridge_context,
                model=model,
                retry_state=retry_state,
            )
            if bridge:
                _record_bridge_parse_event(
                    app,
                    bridge_result,
                    model=model,
                    allowed_tools=allowed_tools,
                    stage="repair",
                    bridge_context=bridge_context,
                )
        if bridge_result.error and bridge_result.error.kind == "unknown_tool":
            virtual_loader_recovery = should_retry_virtual_loader_tool_recovery(
                payload, bridge_result, allowed_tools=allowed_tools
            )
            recovery_stage = "virtual_loader_tool_recovery" if virtual_loader_recovery else "unknown_tool_recovery"
            _record_tool_bridge_event(
                app,
                "tool_bridge_retry",
                stage=recovery_stage,
                model=model,
                errorKind=bridge_result.error.kind,
                errorMessage=_preview_text(bridge_result.error.message, max_chars=360),
            )
            recovery_payload = (
                build_virtual_loader_tool_recovery_payload(payload, bridge_result, allowed_tools=allowed_tools)
                if virtual_loader_recovery
                else build_unknown_tool_recovery_payload(payload, bridge_result, allowed_tools=allowed_tools)
            )
            recovery_data = retry_chat(recovery_payload)
            if isinstance(recovery_data, dict):
                parsed, bridge_result = parse_chat_response(
                    recovery_data,
                    bridge=bridge,
                    allowed_tools=allowed_tools,
                    model=model,
                    bridge_context=bridge_context,
                return_bridge_result=True,
            )
                parsed, bridge_result, retry_state = _apply_controller_decision(
                    parsed,
                    bridge_result,
                    bridge_context=bridge_context,
                    model=model,
                    retry_state=retry_state,
                )
                if bridge:
                    _record_bridge_parse_event(
                        app,
                        bridge_result,
                        model=model,
                        allowed_tools=allowed_tools,
                        stage=recovery_stage,
                        bridge_context=bridge_context,
                    )
        elif _should_retry_off_task_question_recovery(bridge_result):
            _record_tool_bridge_event(
                app,
                "tool_bridge_retry",
                stage="off_task_question_recovery",
                model=model,
                errorKind=bridge_result.error.kind,
                errorMessage=_preview_text(bridge_result.error.message, max_chars=360),
            )
            recovery_data = retry_chat(
                build_off_task_question_recovery_payload(payload, bridge_result, allowed_tools=allowed_tools)
            )
            if isinstance(recovery_data, dict):
                parsed, bridge_result = parse_chat_response(
                    recovery_data,
                    bridge=bridge,
                    allowed_tools=allowed_tools,
                    model=model,
                    bridge_context=bridge_context,
                    return_bridge_result=True,
                )
                parsed, bridge_result, retry_state = _apply_controller_decision(
                    parsed,
                    bridge_result,
                    bridge_context=bridge_context,
                    model=model,
                    retry_state=retry_state,
                )
                if bridge:
                    _record_bridge_parse_event(
                        app,
                        bridge_result,
                        model=model,
                        allowed_tools=allowed_tools,
                        stage="off_task_question_recovery",
                        bridge_context=bridge_context,
                    )
        elif _should_retry_missing_required_tool_input_recovery(bridge_result):
            _record_tool_bridge_event(
                app,
                "tool_bridge_retry",
                stage="missing_required_tool_input_recovery",
                model=model,
                errorKind=bridge_result.error.kind,
                errorMessage=_preview_text(bridge_result.error.message, max_chars=360),
            )
            recovery_data = retry_chat(
                build_missing_required_tool_input_recovery_payload(
                    payload,
                    bridge_result,
                    allowed_tools=allowed_tools,
                )
            )
            if isinstance(recovery_data, dict):
                parsed, bridge_result = parse_chat_response(
                    recovery_data,
                    bridge=bridge,
                    allowed_tools=allowed_tools,
                    model=model,
                    bridge_context=bridge_context,
                    return_bridge_result=True,
                )
                parsed, bridge_result, retry_state = _apply_controller_decision(
                    parsed,
                    bridge_result,
                    bridge_context=bridge_context,
                    model=model,
                    retry_state=retry_state,
                )
                if bridge:
                    _record_bridge_parse_event(
                        app,
                        bridge_result,
                        model=model,
                        allowed_tools=allowed_tools,
                        stage="missing_required_tool_input_recovery",
                        bridge_context=bridge_context,
                    )
        elif _should_retry_tool_refusal_recovery(bridge_result):
            _record_tool_bridge_event(
                app,
                "tool_bridge_retry",
                stage="tool_refusal_recovery",
                model=model,
                errorKind=bridge_result.error.kind,
                errorMessage=_preview_text(bridge_result.error.message, max_chars=360),
            )
            recovery_data = retry_chat(
                build_tool_refusal_recovery_payload(payload, bridge_result, allowed_tools=allowed_tools)
            )
            if isinstance(recovery_data, dict):
                parsed, bridge_result = parse_chat_response(
                    recovery_data,
                    bridge=bridge,
                    allowed_tools=allowed_tools,
                    model=model,
                    bridge_context=bridge_context,
                return_bridge_result=True,
            )
                parsed, bridge_result, retry_state = _apply_controller_decision(
                    parsed,
                    bridge_result,
                    bridge_context=bridge_context,
                    model=model,
                    retry_state=retry_state,
                )
                if bridge:
                    _record_bridge_parse_event(
                        app,
                        bridge_result,
                        model=model,
                        allowed_tools=allowed_tools,
                        stage="tool_refusal_recovery",
                        bridge_context=bridge_context,
                    )
        elif _should_retry_controller_final_repair(bridge_result):
            parsed, bridge_result, retry_state = _retry_repairable_bridge_error(
                parsed,
                bridge_result,
                retry_state,
                app=app,
                payload=payload,
                bridge=bridge,
                allowed_tools=allowed_tools,
                bridge_context=bridge_context,
                model=model,
                retry_chat=retry_chat,
                stage="post_repair_controller_final_repair",
            )
        elif _should_retry_malformed_tool_format_repair(bridge_result):
            parsed, bridge_result, retry_state = _retry_repairable_bridge_error(
                parsed,
                bridge_result,
                retry_state,
                app=app,
                payload=payload,
                bridge=bridge,
                allowed_tools=allowed_tools,
                bridge_context=bridge_context,
                model=model,
                retry_chat=retry_chat,
                stage="post_repair_malformed_tool_repair",
            )
    elif should_retry_incomplete_response(data):
        _record_tool_bridge_event(app, "tool_bridge_retry", stage="incomplete_response", model=model)
        retry_data = retry_chat(
            build_incomplete_response_retry_payload(
                payload,
                _openai_response_content(data),
                bridge=bridge,
                allowed_tools=allowed_tools,
            )
        )
        if isinstance(retry_data, dict):
            parsed, bridge_result = parse_chat_response(
                retry_data,
                bridge=bridge,
                allowed_tools=allowed_tools,
                model=model,
                bridge_context=bridge_context,
                return_bridge_result=True,
            )
            parsed, bridge_result, retry_state = _apply_controller_decision(
                parsed,
                bridge_result,
                bridge_context=bridge_context,
                model=model,
                retry_state=retry_state,
            )
            if bridge:
                _record_bridge_parse_event(
                    app,
                    bridge_result,
                    model=model,
                    allowed_tools=allowed_tools,
                    stage="incomplete_response",
                    bridge_context=bridge_context,
                )
            if bridge_result.error and bridge_result.error.repairable:
                _record_tool_bridge_event(
                    app,
                    "tool_bridge_retry",
                    stage="incomplete_repair",
                    model=model,
                    errorKind=bridge_result.error.kind,
                    errorMessage=_preview_text(bridge_result.error.message, max_chars=360),
                )
                repair_data = retry_chat(build_repair_payload(payload, bridge_result, allowed_tools=allowed_tools))
                if isinstance(repair_data, dict):
                    parsed, bridge_result = parse_chat_response(
                        repair_data,
                        bridge=bridge,
                        allowed_tools=allowed_tools,
                        model=model,
                        bridge_context=bridge_context,
                    return_bridge_result=True,
                )
                    parsed, bridge_result, retry_state = _apply_controller_decision(
                        parsed,
                        bridge_result,
                        bridge_context=bridge_context,
                        model=model,
                        retry_state=retry_state,
                    )
                    if bridge:
                        _record_bridge_parse_event(
                            app,
                            bridge_result,
                            model=model,
                            allowed_tools=allowed_tools,
                            stage="incomplete_repair",
                            bridge_context=bridge_context,
                        )
                if _should_retry_tool_refusal_recovery(bridge_result):
                    _record_tool_bridge_event(
                        app,
                        "tool_bridge_retry",
                        stage="incomplete_tool_refusal_recovery",
                        model=model,
                        errorKind=bridge_result.error.kind,
                        errorMessage=_preview_text(bridge_result.error.message, max_chars=360),
                    )
                    recovery_data = retry_chat(
                        build_tool_refusal_recovery_payload(payload, bridge_result, allowed_tools=allowed_tools)
                    )
                    if isinstance(recovery_data, dict):
                        parsed, bridge_result = parse_chat_response(
                            recovery_data,
                            bridge=bridge,
                            allowed_tools=allowed_tools,
                            model=model,
                            bridge_context=bridge_context,
                        return_bridge_result=True,
                    )
                        parsed, bridge_result, retry_state = _apply_controller_decision(
                            parsed,
                            bridge_result,
                            bridge_context=bridge_context,
                            model=model,
                            retry_state=retry_state,
                            )
                        if bridge:
                            _record_bridge_parse_event(
                                app,
                                bridge_result,
                                model=model,
                                allowed_tools=allowed_tools,
                                stage="incomplete_tool_refusal_recovery",
                                bridge_context=bridge_context,
                            )
                elif _should_retry_off_task_question_recovery(bridge_result):
                    _record_tool_bridge_event(
                        app,
                        "tool_bridge_retry",
                        stage="incomplete_off_task_question_recovery",
                        model=model,
                        errorKind=bridge_result.error.kind,
                        errorMessage=_preview_text(bridge_result.error.message, max_chars=360),
                    )
                    recovery_data = retry_chat(
                        build_off_task_question_recovery_payload(payload, bridge_result, allowed_tools=allowed_tools)
                    )
                    if isinstance(recovery_data, dict):
                        parsed, bridge_result = parse_chat_response(
                            recovery_data,
                            bridge=bridge,
                            allowed_tools=allowed_tools,
                            model=model,
                            bridge_context=bridge_context,
                            return_bridge_result=True,
                        )
                        parsed, bridge_result, retry_state = _apply_controller_decision(
                            parsed,
                            bridge_result,
                            bridge_context=bridge_context,
                            model=model,
                            retry_state=retry_state,
                        )
                        if bridge:
                            _record_bridge_parse_event(
                                app,
                                bridge_result,
                                model=model,
                                allowed_tools=allowed_tools,
                                stage="incomplete_off_task_question_recovery",
                                bridge_context=bridge_context,
                            )
                elif _should_retry_missing_required_tool_input_recovery(bridge_result):
                    _record_tool_bridge_event(
                        app,
                        "tool_bridge_retry",
                        stage="incomplete_missing_required_tool_input_recovery",
                        model=model,
                        errorKind=bridge_result.error.kind,
                        errorMessage=_preview_text(bridge_result.error.message, max_chars=360),
                    )
                    recovery_data = retry_chat(
                        build_missing_required_tool_input_recovery_payload(
                            payload,
                            bridge_result,
                            allowed_tools=allowed_tools,
                        )
                    )
                    if isinstance(recovery_data, dict):
                        parsed, bridge_result = parse_chat_response(
                            recovery_data,
                            bridge=bridge,
                            allowed_tools=allowed_tools,
                            model=model,
                            bridge_context=bridge_context,
                            return_bridge_result=True,
                        )
                        parsed, bridge_result, retry_state = _apply_controller_decision(
                            parsed,
                            bridge_result,
                            bridge_context=bridge_context,
                            model=model,
                            retry_state=retry_state,
                        )
                        if bridge:
                            _record_bridge_parse_event(
                                app,
                                bridge_result,
                                model=model,
                                allowed_tools=allowed_tools,
                                stage="incomplete_missing_required_tool_input_recovery",
                                bridge_context=bridge_context,
                            )
                elif _should_retry_controller_final_repair(bridge_result):
                    parsed, bridge_result, retry_state = _retry_repairable_bridge_error(
                        parsed,
                        bridge_result,
                        retry_state,
                        app=app,
                        payload=payload,
                        bridge=bridge,
                        allowed_tools=allowed_tools,
                        bridge_context=bridge_context,
                        model=model,
                        retry_chat=retry_chat,
                        stage="incomplete_post_repair_controller_final_repair",
                    )
                elif _should_retry_malformed_tool_format_repair(bridge_result):
                    parsed, bridge_result, retry_state = _retry_repairable_bridge_error(
                        parsed,
                        bridge_result,
                        retry_state,
                        app=app,
                        payload=payload,
                        bridge=bridge,
                        allowed_tools=allowed_tools,
                        bridge_context=bridge_context,
                        model=model,
                        retry_chat=retry_chat,
                        stage="incomplete_post_repair_malformed_tool_repair",
                    )
    if bridge:
        if _should_retry_no_progress_escalation(bridge_result):
            _record_tool_bridge_event(
                app,
                "tool_bridge_retry",
                stage="no_progress_escalation",
                model=model,
                errorKind=bridge_result.error.kind,
                errorMessage=_preview_text(bridge_result.error.message, max_chars=360),
            )
            escalation_data = retry_chat(
                build_tool_refusal_recovery_payload(
                    payload,
                    bridge_result,
                    allowed_tools=allowed_tools,
                    escalation=True,
                )
            )
            if isinstance(escalation_data, dict):
                parsed, bridge_result = parse_chat_response(
                    escalation_data,
                    bridge=bridge,
                    allowed_tools=allowed_tools,
                    model=model,
                    bridge_context=bridge_context,
                    return_bridge_result=True,
                )
                parsed, bridge_result, retry_state = _apply_controller_decision(
                    parsed,
                    bridge_result,
                    bridge_context=bridge_context,
                    model=model,
                    retry_state=retry_state,
                )
                _record_bridge_parse_event(
                    app,
                    bridge_result,
                    model=model,
                    allowed_tools=allowed_tools,
                    stage="no_progress_escalation",
                    bridge_context=bridge_context,
                )
        _record_bridge_rejection_event(
            app,
            bridge_result,
            model=model,
            allowed_tools=allowed_tools,
            bridge_context=bridge_context,
        )
        parsed, bridge_result = _finalize_retry_exhausted_bridge_error(
            parsed,
            bridge_result,
            bridge_context=bridge_context,
            model=model,
        )
    return parsed, bridge_result


def _retry_repairable_bridge_error(
    parsed: dict[str, Any],
    bridge_result: Any,
    retry_state: RetryState,
    *,
    app: FastAPI,
    payload: dict[str, Any],
    bridge: bool,
    allowed_tools: set[str],
    bridge_context: Any,
    model: str,
    retry_chat: Any,
    stage: str,
) -> tuple[dict[str, Any], Any, RetryState]:
    _record_tool_bridge_event(
        app,
        "tool_bridge_retry",
        stage=stage,
        model=model,
        errorKind=bridge_result.error.kind,
        errorMessage=_preview_text(bridge_result.error.message, max_chars=360),
    )
    repair_data = retry_chat(build_repair_payload(payload, bridge_result, allowed_tools=allowed_tools))
    if not isinstance(repair_data, dict):
        return parsed, bridge_result, retry_state
    parsed, bridge_result = parse_chat_response(
        repair_data,
        bridge=bridge,
        allowed_tools=allowed_tools,
        model=model,
        bridge_context=bridge_context,
        return_bridge_result=True,
    )
    parsed, bridge_result, retry_state = _apply_controller_decision(
        parsed,
        bridge_result,
        bridge_context=bridge_context,
        model=model,
        retry_state=retry_state,
    )
    if bridge:
        _record_bridge_parse_event(
            app,
            bridge_result,
            model=model,
            allowed_tools=allowed_tools,
            stage=stage,
            bridge_context=bridge_context,
        )
    return parsed, bridge_result, retry_state


def _apply_controller_decision(
    parsed: dict[str, Any],
    bridge_result: Any,
    *,
    bridge_context: Any,
    model: str,
    retry_state: RetryState,
) -> tuple[dict[str, Any], Any, RetryState]:
    decision = classify_bridge_result(bridge_result, bridge_context, retry_state)
    next_state = decision.retry_state or retry_state
    if decision.state == "ASK_USER":
        ask_user_response = decision_to_openai_chat_response(decision, model=model)
        if ask_user_response is not None:
            metadata = _bridge_result_controller_metadata(
                bridge_result,
                bridge_context=bridge_context,
                controller_state=decision.state,
                controller_reason=decision.reason,
                retry_state=next_state,
            )
            return (
                ask_user_response,
                BridgeResult(
                    content="",
                    tool_calls=decision.tool_calls or [],
                    raw_content=str(getattr(bridge_result, "raw_content", "") or ""),
                    **metadata,
                ),
                next_state,
            )
    if decision.state == "RETRY" and getattr(bridge_result, "error", None) is None:
        content = str(getattr(bridge_result, "content", "") or "")
        error_kind = decision.reason or decision.retry_kind or "controller_retry_required"
        metadata = _bridge_result_controller_metadata(
            bridge_result,
            bridge_context=bridge_context,
            controller_state=decision.state,
            controller_reason=decision.reason,
            retry_state=next_state,
        )
        return (
            parsed,
            BridgeResult(
                content=content,
                tool_calls=[],
                error=BridgeError(
                    error_kind,
                    _controller_retry_error_message(error_kind),
                    repairable=True,
                ),
                warning=getattr(bridge_result, "warning", None),
                raw_content=str(getattr(bridge_result, "raw_content", "") or content),
                **metadata,
            ),
            next_state,
        )
    if decision.state == "FINAL" and decision.reason == "retry_budget_exhausted" and decision.retry_kind:
        error_kind = decision.retry_kind
        original_error = getattr(bridge_result, "error", None)
        original_message = str(getattr(original_error, "message", "") or "").strip()
        metadata = _bridge_result_controller_metadata(
            bridge_result,
            bridge_context=bridge_context,
            controller_state=decision.state,
            controller_reason=decision.reason,
            retry_state=next_state,
        )
        return (
            parsed,
            BridgeResult(
                content=str(getattr(bridge_result, "content", "") or ""),
                tool_calls=[],
                error=BridgeError(
                    error_kind,
                    original_message or _controller_retry_error_message(error_kind),
                    repairable=True,
                ),
                warning=getattr(bridge_result, "warning", None),
                raw_content=str(getattr(bridge_result, "raw_content", "") or getattr(bridge_result, "content", "") or ""),
                **metadata,
            ),
            next_state,
        )
    metadata = _bridge_result_controller_metadata(
        bridge_result,
        bridge_context=bridge_context,
        controller_state=decision.state,
        controller_reason=decision.reason,
        retry_state=next_state,
    )
    return parsed, _copy_bridge_result_with_metadata(bridge_result, **metadata), next_state


def _controller_retry_error_message(error_kind: str) -> str:
    if error_kind == "insufficient_final_evidence":
        return (
            "The model returned a final answer for local-agent work without enough prior tool evidence. "
            "Request an allowed discovery/read/search tool before finalizing."
        )
    if error_kind == "off_task_environment_configuration_final":
        return (
            "The model returned agent/environment configuration advice unrelated to the current local-agent task. "
            "Continue the original task with allowed tools or answer from gathered evidence. Do not ask the user "
            "to configure Claude/Codex/statusLine/settings unless the latest user task explicitly requested that."
        )
    if error_kind == "status_only_final_without_task_answer":
        return (
            "The model returned only a tool execution status instead of answering the user's local-agent task. "
            "Continue with an allowed discovery/read/search tool or produce a substantive final review, plan, or implementation summary."
        )
    if error_kind == "history_summary_final_without_task_answer":
        return (
            "The model summarized DS2API_HISTORY.txt or the current state instead of answering the latest local-agent task. "
            "Use DS2API_HISTORY only as context, then answer the current user request directly from gathered evidence."
        )
    if error_kind == "unknown_project_structure_final_without_task_answer":
        return (
            "The model stopped after broad or truncated discovery and asked the user for project structure, paths, "
            "file lists, language, or source code instead of continuing the local-agent task. Use the current workspace "
            "context and request a narrower allowed tool such as Glob, Grep, Read, LS, or Bash when available. "
            "Do not ask the user to provide repository details that the downstream tools can inspect."
        )
    if error_kind == "review_next_step_menu_final_without_task_answer":
        return (
            "The model returned a document next-step menu or asked the user to choose an operation instead of answering "
            "the current code review/audit task. Do not summarize README/CONFIGURATION next-step suggestions as the final "
            "answer. Continue inspecting relevant files with allowed tools, or provide concrete review findings, risks, "
            "and improvement recommendations grounded in the files already read."
        )
    return (
        "The controller rejected this response as incomplete for local-agent work. "
        "Request an allowed tool if more evidence is needed, otherwise provide a substantive final answer."
    )


def _bridge_result_controller_metadata(
    bridge_result: Any,
    *,
    bridge_context: Any,
    controller_state: str,
    controller_reason: str,
    retry_state: RetryState,
) -> dict[str, Any]:
    metadata: dict[str, Any] = {
        "controller_state": controller_state,
        "controller_reason": controller_reason,
        "controller_retry_budget": (
            f"repair={retry_state.repair_attempts};"
            f"recovery={retry_state.recovery_attempts};"
            f"ask_user={retry_state.ask_user_attempts}"
        ),
    }
    mode = str(getattr(getattr(bridge_context, "options", None), "semantic_final_judge", "") or "off").strip().lower()
    if mode in {"shadow", "enforce"}:
        judge = judge_bridge_semantics(bridge_result, bridge_context, controller_state=controller_state)
        metadata.update(
            {
                "semantic_final_judge_mode": mode,
                "semantic_final_judge_verdict": judge.verdict,
                "semantic_final_judge_confidence": judge.confidence,
                "semantic_final_judge_reason": judge.reason,
            }
        )
    return metadata


def _copy_bridge_result_with_metadata(bridge_result: Any, **metadata: Any) -> Any:
    if not isinstance(bridge_result, BridgeResult):
        return bridge_result
    return BridgeResult(
        content=bridge_result.content,
        tool_calls=bridge_result.tool_calls,
        error=bridge_result.error,
        warning=bridge_result.warning,
        raw_content=bridge_result.raw_content,
        phases=bridge_result.phases,
        **metadata,
    )


def _finalize_retry_exhausted_bridge_error(
    parsed: dict[str, Any],
    bridge_result: Any,
    *,
    bridge_context: Any,
    model: str,
) -> tuple[dict[str, Any], Any]:
    error = getattr(bridge_result, "error", None)
    if not (
        error
        and getattr(error, "repairable", False)
        and str(getattr(bridge_result, "controller_reason", "") or "") == "retry_budget_exhausted"
    ):
        return parsed, bridge_result
    final_error = BridgeError(str(error.kind), str(error.message or _controller_retry_error_message(error.kind)), repairable=False)
    content = _controller_retry_exhausted_text(final_error, bridge_context=bridge_context)
    finalized = BridgeResult(
        content=content,
        tool_calls=[],
        error=final_error,
        warning=getattr(bridge_result, "warning", None),
        raw_content=str(getattr(bridge_result, "raw_content", "") or getattr(bridge_result, "content", "") or ""),
        phases=getattr(bridge_result, "phases", []),
        controller_state=str(getattr(bridge_result, "controller_state", "") or ""),
        controller_reason=str(getattr(bridge_result, "controller_reason", "") or ""),
        controller_retry_budget=str(getattr(bridge_result, "controller_retry_budget", "") or ""),
        semantic_final_judge_mode=str(getattr(bridge_result, "semantic_final_judge_mode", "") or ""),
        semantic_final_judge_verdict=str(getattr(bridge_result, "semantic_final_judge_verdict", "") or ""),
        semantic_final_judge_confidence=getattr(bridge_result, "semantic_final_judge_confidence", None),
        semantic_final_judge_reason=str(getattr(bridge_result, "semantic_final_judge_reason", "") or ""),
    )
    response = _openai_response_from_stream_buffer(content, finish_reason="stop", model=model)
    response["choices"][0]["message"]["webai_tool_bridge"] = {"error": final_error.kind, "message": final_error.message}
    return response, finalized


def _controller_retry_exhausted_text(error: BridgeError, *, bridge_context: Any) -> str:
    allowed = sorted(str(tool.name) for tool in getattr(bridge_context, "tools", []) if str(getattr(tool, "name", "") or ""))
    parts = [
        "上游网页模型连续返回低信息或不完整的最终回复，Gateway 已拒绝把它当成成功结果。",
        f"错误码：{error.kind}",
    ]
    if error.message:
        parts.append(f"原因：{_preview_text(error.message, max_chars=260)}")
    if allowed:
        suffix = " ..." if len(allowed) > 16 else ""
        parts.append(f"当前允许工具：{', '.join(allowed[:16])}{suffix}")
    parts.append("请重试；如果仍然出现，建议切换更强的网页登录模型或把这段诊断发给 Gateway 适配层排查。")
    return "\n".join(parts)


def _should_retry_tool_refusal_recovery(bridge_result: Any) -> bool:
    error = getattr(bridge_result, "error", None)
    return bool(
        error
        and getattr(error, "kind", "")
        in {
            "tool_denial_without_call",
            "deferred_tool_action_without_call",
            "deferred_code_change_without_call",
            "unverified_code_change_completion",
            "write_after_failed_read_without_discovery",
            "write_after_failed_path_without_discovery",
            "unsafe_local_shell_command",
            "repeat_discovery_call_without_progress",
            "repeat_read_call_without_progress",
            "repeat_unchanged_read_without_progress",
            "repeat_shell_housekeeping_without_progress",
            "repeat_same_skill_without_progress",
        }
    )


def _should_retry_required_tool_choice_recovery(bridge_result: Any) -> bool:
    error = getattr(bridge_result, "error", None)
    return bool(error and getattr(error, "kind", "") == "tool_choice_violation")


def _should_retry_missing_required_tool_input_recovery(bridge_result: Any) -> bool:
    error = getattr(bridge_result, "error", None)
    return bool(error and getattr(error, "kind", "") == "missing_required_tool_input")


def _should_retry_malformed_tool_format_repair(bridge_result: Any) -> bool:
    error = getattr(bridge_result, "error", None)
    return bool(
        error
        and getattr(error, "repairable", False)
        and getattr(error, "kind", "")
        in {"malformed_json", "empty_tool_call", "invalid_tool_call", "invalid_input"}
    )


def _should_retry_controller_final_repair(bridge_result: Any) -> bool:
    error = getattr(bridge_result, "error", None)
    return bool(
        error
        and getattr(error, "repairable", False)
        and getattr(error, "kind", "")
        in {
            "insufficient_final_evidence",
            "status_only_final_without_task_answer",
            "history_summary_final_without_task_answer",
            "unknown_project_structure_final_without_task_answer",
            "review_next_step_menu_final_without_task_answer",
            "off_task_environment_configuration_final",
        }
    )


def _should_retry_no_progress_escalation(bridge_result: Any) -> bool:
    error = getattr(bridge_result, "error", None)
    return bool(
        error
        and getattr(error, "kind", "")
        in {
            "repeat_read_call_without_progress",
            "repeat_unchanged_read_without_progress",
            "repeat_shell_housekeeping_without_progress",
        }
    )


def _should_retry_off_task_question_recovery(bridge_result: Any) -> bool:
    error = getattr(bridge_result, "error", None)
    return bool(error and getattr(error, "kind", "") in OFF_TASK_QUESTION_ERROR_KINDS)


def _qwen_web_backend(config: GatewayConfig) -> str:
    backend = (config.provider_runtime.qwen_web_backend or "direct").strip().lower()
    return backend.replace("_", "-")


def _qwen_web_uses_deepseek_ds2api(config: GatewayConfig) -> bool:
    return _qwen_web_backend(config) in {"deepseek-ds2api", "ds2api", "deepseek"}


def _qwen_web_body_for_deepseek_ds2api(body: dict[str, Any]) -> dict[str, Any]:
    routed = dict(body)
    routed["model"] = "deepseek-v4-pro"
    return routed


def _gpt_thinking_backend(config: GatewayConfig) -> str:
    backend = (config.provider_runtime.gpt_thinking_backend or "webai2api").strip().lower()
    return backend.replace("_", "-")


def _gpt_thinking_uses_deepseek_ds2api(config: GatewayConfig) -> bool:
    return _gpt_thinking_backend(config) in {"deepseek-ds2api", "ds2api", "deepseek"}


def _is_gpt_thinking_model(model: Any) -> bool:
    return normalize_model_id(model) in _GPT_THINKING_MODEL_IDS


def _gpt_thinking_body_for_deepseek_ds2api(body: dict[str, Any]) -> dict[str, Any]:
    routed = dict(body)
    routed["model"] = "deepseek-v4-pro"
    return routed


def _deepseek_ds2api_retry_delays(app: FastAPI) -> tuple[float, ...]:
    raw = getattr(app.state, "deepseek_ds2api_retry_delays", (1.0, 3.0, 8.0))
    if raw is None:
        return ()
    try:
        return tuple(max(0.0, float(item)) for item in raw)
    except (TypeError, ValueError):
        return ()


def _deepseek_ds2api_bearer_gate(app: FastAPI, config: GatewayConfig) -> _ProviderRateLimitGate:
    gate_config = (
        max(1, int(config.provider_runtime.deepseek_ds2api_bearer_max_inflight)),
        max(0.0, float(config.provider_runtime.deepseek_ds2api_rate_limit_cooldown_seconds)),
    )
    lock = getattr(app.state, "deepseek_ds2api_bearer_gate_lock", None)
    if lock is None:
        lock = threading.Lock()
        app.state.deepseek_ds2api_bearer_gate_lock = lock
    with lock:
        if (
            getattr(app.state, "deepseek_ds2api_bearer_gate", None) is None
            or getattr(app.state, "deepseek_ds2api_bearer_gate_config", None) != gate_config
        ):
            app.state.deepseek_ds2api_bearer_gate = _ProviderRateLimitGate(
                max_inflight=gate_config[0],
                cooldown_seconds=gate_config[1],
            )
        app.state.deepseek_ds2api_bearer_gate_config = gate_config
        return app.state.deepseek_ds2api_bearer_gate


def _qwen_direct_gate(app: FastAPI, config: GatewayConfig, route: str) -> _DirectProviderPoolGate:
    gate_config = (
        route,
        max(1, int(config.provider_runtime.qwen_direct_max_inflight)),
        max(0, int(config.provider_runtime.qwen_direct_max_queue)),
        max(0.0, float(config.provider_runtime.qwen_direct_rate_limit_cooldown_seconds)),
    )
    lock = getattr(app.state, "qwen_direct_gate_lock", None)
    if lock is None:
        lock = threading.Lock()
        app.state.qwen_direct_gate_lock = lock
    with lock:
        gates = getattr(app.state, "qwen_direct_gates", None)
        gate_configs = getattr(app.state, "qwen_direct_gate_configs", None)
        if not isinstance(gates, dict):
            gates = {}
            app.state.qwen_direct_gates = gates
        if not isinstance(gate_configs, dict):
            gate_configs = {}
            app.state.qwen_direct_gate_configs = gate_configs
        if route not in gates or gate_configs.get(route) != gate_config:
            gates[route] = _DirectProviderPoolGate(
                max_inflight_per_account=gate_config[1],
                max_queue=gate_config[2],
                cooldown_seconds=gate_config[3],
            )
            gate_configs[route] = gate_config
        return gates[route]


def _direct_profile_id_from_account_id(provider_id: str, account_id: str) -> str:
    parsed = parse_account_id(account_id)
    if not parsed or parsed.source != "direct" or parsed.provider_id != provider_id:
        return ""
    return str(parsed.profile_id or "default")


def _direct_account_ids_for_provider(app: FastAPI, provider_id: str) -> list[str]:
    ids: list[str] = []
    for profile_id in getattr(app.state.credential_store, "list_profile_ids")(provider_id):
        account_id = direct_account_id(provider_id, profile_id)
        if account_id not in ids:
            ids.append(account_id)
    registry_data = app.state.account_registry.load()
    accounts = registry_data.get("accounts") if isinstance(registry_data.get("accounts"), dict) else {}
    for account_id in accounts:
        parsed = parse_account_id(str(account_id))
        if parsed and parsed.source == "direct" and parsed.provider_id == provider_id and str(account_id) not in ids:
            ids.append(str(account_id))
    ids.sort(key=lambda value: (0 if value.endswith(":default") else 1, value))
    return ids


def _direct_credential_for_account(app: FastAPI, provider_id: str, account_id: str) -> dict[str, Any] | None:
    profile_id = _direct_profile_id_from_account_id(provider_id, account_id)
    if not profile_id:
        return None
    return app.state.credential_store.get_profile(provider_id, profile_id)


def _authorized_direct_account_ids(app: FastAPI, provider_id: str) -> list[str]:
    out: list[str] = []
    for account_id in _direct_account_ids_for_provider(app, provider_id):
        credential = _direct_credential_for_account(app, provider_id, account_id)
        if is_credential_authorized(provider_id, credential):
            out.append(account_id)
    return out


def _selected_direct_account_id(app: FastAPI, provider_id: str, body: dict[str, Any]) -> str:
    target = str(body.get("_webai_target_account_id") or "").strip()
    if target:
        parsed = parse_account_id(target)
        if not parsed or parsed.source != "direct" or parsed.provider_id != provider_id:
            raise HTTPException(status_code=400, detail="目标账号不属于当前 Provider")
        return target
    current = app.state.account_registry.current_account_id(provider_id)
    if current and current in _direct_account_ids_for_provider(app, provider_id):
        return current
    authorized = _authorized_direct_account_ids(app, provider_id)
    return authorized[0] if authorized else direct_account_id(provider_id)


def _alternate_direct_account_id(app: FastAPI, provider_id: str, account_id: str) -> str:
    for candidate in _authorized_direct_account_ids(app, provider_id):
        if candidate != account_id:
            return candidate
    return ""


def _is_qwen_direct_rate_limit_exception(exc: BaseException) -> bool:
    if isinstance(exc, httpx.HTTPStatusError) and exc.response.status_code == 429:
        return True
    text = str(exc).lower()
    markers = (
        "429",
        "rate limit",
        "rate-limit",
        "too many requests",
        "too many request",
        "quota",
        "限流",
        "频率",
        "配额",
        "请求过多",
    )
    return any(marker in text for marker in markers)


def _call_qwen_direct_with_gate(
    app: FastAPI,
    *,
    provider_id: str,
    route: str,
    account_id: str,
    web_client: Any,
    payload: dict[str, Any],
    config: GatewayConfig,
) -> dict[str, Any]:
    gate = _qwen_direct_gate(app, config, route)
    acquired_account_id = gate.acquire(_authorized_direct_account_ids(app, provider_id), target=account_id)
    if not acquired_account_id:
        raise HTTPException(status_code=429, detail="Qwen direct 请求过多：当前登录账号的等待队列已满，请稍后重试或降低客户端并发。")
    try:
        return web_client.chat_completions(payload)
    except Exception as exc:
        if _is_qwen_direct_rate_limit_exception(exc):
            gate.note_rate_limit()
        raise
    finally:
        gate.release(acquired_account_id)


def _is_deepseek_ds2api_upstream_empty_output_rate_limit(exc: DeepSeekDs2apiError) -> bool:
    text = str(exc).lower()
    return exc.status_code == 429 and ("upstream_empty_output" in text or "empty output" in text)


def _provider_auth_required_detail(provider_label: str) -> str:
    return (
        f"{provider_label} 还没有可用的网页登录授权。"
        "请打开 WebAI Gateway 首页，选择对应平台，点击“打开授权浏览器”完成网页登录；"
        "登录完成后回到首页刷新模型，再重试 API 调用。"
    )


def _provider_auth_expired_detail(provider_label: str) -> str:
    return (
        f"{provider_label} 的网页登录授权已过期或失效。"
        "请打开 WebAI Gateway 首页，选择对应平台，点击“打开授权浏览器”重新登录；"
        "登录完成后点击“刷新模型”或“验证接入”，再重试 API 调用。"
    )


def _is_provider_auth_failure(status_code: int | None, text: Any) -> bool:
    lowered = str(text or "").lower()
    if status_code in {401, 403}:
        return True
    markers = (
        "invalid token",
        "authentication_failed",
        "unauthorized",
        "token has expired",
        "token expired",
        "session expired",
        "login again",
        "not logged in",
        "please log in",
        "account token is invalid",
        "failed to get pow",
        "登录已过期",
        "登录过期",
        "未登录",
        "重新登录",
    )
    return any(marker in lowered for marker in markers)


def _call_deepseek_ds2api_with_retry(
    app: FastAPI,
    web_client: Any,
    payload: dict[str, Any],
    config: GatewayConfig,
) -> dict[str, Any]:
    delays = _deepseek_ds2api_retry_delays(app)
    gate = _deepseek_ds2api_bearer_gate(app, config)
    for attempt in range(len(delays) + 1):
        gate.acquire()
        try:
            return web_client.chat_completions(payload)
        except DeepSeekDs2apiError as exc:
            if _is_deepseek_ds2api_upstream_empty_output_rate_limit(exc):
                gate.note_rate_limit()
            if exc.status_code != 429 or attempt >= len(delays):
                raise
            delay = delays[attempt]
        finally:
            gate.release()
        if attempt < len(delays):
            if delay > 0:
                time.sleep(delay)
    raise RuntimeError("unreachable ds2api retry state")


def _deepseek_web_chat(app: FastAPI, client: httpx.Client, body: dict[str, Any], config: GatewayConfig) -> Response:
    credential = app.state.credential_store.get("deepseek-web")
    if not is_credential_authorized("deepseek-web", credential):
        raise HTTPException(status_code=424, detail=_provider_auth_required_detail("DeepSeek Web"))
    payload, bridge, allowed_tools, bridge_context = _build_deepseek_direct_payload(body, config)
    model = str(payload.get("model") or DEEPSEEK_DEFAULT_MODEL)
    skill_preflight = _skill_loader_preflight_response(app, model, bridge_context, require_namespaced_slash=True)
    if skill_preflight is not None:
        return skill_preflight
    _record_completion_started_diagnostic(
        app,
        endpoint="/v1/chat/completions",
        route="deepseek-web",
        model=model,
        body=body,
        stream=bool(body.get("stream")),
        bridge=bridge,
        bridge_context=bridge_context,
    )
    web_client = None
    try:
        web_client = _build_deepseek_web_client(app.state.deepseek_client_factory, credential, client, config)
        data = _call_deepseek_ds2api_with_retry(app, web_client, payload, config)
    except DeepSeekDs2apiError as exc:
        status_code = _provider_dependency_status_code(exc.status_code)
        _record_completion_error_diagnostic(
            app,
            endpoint="/v1/chat/completions",
            route="deepseek-web",
            model=model,
            body=body,
            stream=bool(body.get("stream")),
            bridge=bridge,
            status_code=status_code,
            error_kind="ds2api_http_error",
            error=exc,
            provider_diagnostic=getattr(web_client, "last_diagnostic", None),
            bridge_context=bridge_context,
        )
        detail = _provider_auth_expired_detail("DeepSeek Web") if _is_provider_auth_failure(exc.status_code, exc) else str(exc)
        raise HTTPException(status_code=status_code, detail=detail) from exc
    except HTTPException:
        raise
    except Exception as exc:
        _record_completion_error_diagnostic(
            app,
            endpoint="/v1/chat/completions",
            route="deepseek-web",
            model=model,
            body=body,
            stream=bool(body.get("stream")),
            bridge=bridge,
            status_code=502,
            error_kind="provider_call_failed",
            error=exc,
            provider_diagnostic=getattr(web_client, "last_diagnostic", None),
            bridge_context=bridge_context,
        )
        raise HTTPException(status_code=502, detail=f"DeepSeek Web 调用失败：{exc}") from exc
    if not isinstance(data, dict):
        raise HTTPException(status_code=502, detail="DeepSeek Web 响应必须是 JSON 对象")
    if data.get("_webai_stream") and bool(body.get("stream")):
        return _openai_stream_response(
            app,
            body=body,
            route="deepseek-web",
            model=model,
            bridge=bridge,
            sse_body=str(data.get("_webai_sse_body") or ""),
            provider_diagnostic=getattr(web_client, "last_diagnostic", None),
            bridge_context=bridge_context,
            content_type=str(data.get("_webai_content_type") or "text/event-stream"),
        )
    try:
        parsed, bridge_result = _parse_bridge_chat_data(
            data,
            app=app,
            payload=payload,
            bridge=bridge,
            allowed_tools=allowed_tools,
            bridge_context=bridge_context,
            model=model,
            retry_chat=lambda retry_payload: _call_deepseek_ds2api_with_retry(app, web_client, retry_payload, config),
            native_web_search=bool(payload.get("_webai_native_web_search")),
        )
    except HTTPException:
        raise
    except DeepSeekDs2apiError as exc:
        status_code = _provider_dependency_status_code(exc.status_code)
        _record_completion_error_diagnostic(
            app,
            endpoint="/v1/chat/completions",
            route="deepseek-web",
            model=model,
            body=body,
            stream=bool(body.get("stream")),
            bridge=bridge,
            status_code=status_code,
            error_kind="ds2api_http_error",
            error=exc,
            provider_diagnostic=getattr(web_client, "last_diagnostic", None),
            bridge_context=bridge_context,
        )
        detail = _provider_auth_expired_detail("DeepSeek Web") if _is_provider_auth_failure(exc.status_code, exc) else str(exc)
        raise HTTPException(status_code=status_code, detail=detail) from exc
    except (TimeoutError, httpx.TimeoutException) as exc:
        provider_diagnostic = _merge_provider_diagnostics(
            getattr(web_client, "last_diagnostic", None), getattr(exc, "diagnostic", None)
        )
        _record_completion_error_diagnostic(
            app,
            endpoint="/v1/chat/completions",
            route="deepseek-web",
            model=model,
            body=body,
            stream=bool(body.get("stream")),
            bridge=bridge,
            status_code=504,
            error_kind="timeout",
            error=exc,
            provider_diagnostic=provider_diagnostic,
            bridge_context=bridge_context,
        )
        diagnostic = _format_qwen_timeout_diagnostic(provider_diagnostic, None)
        raise HTTPException(
            status_code=504,
            detail=f"Qwen Web 响应超时：{exc}{diagnostic}",
            headers={"x-should-retry": "false"},
        ) from exc
    if bool(body.get("stream")):
        content = ""
        choices = parsed.get("choices") if isinstance(parsed.get("choices"), list) else []
        if choices and isinstance(choices[0], dict):
            msg = choices[0].get("message") if isinstance(choices[0].get("message"), dict) else {}
            if isinstance(msg.get("tool_calls"), list) and msg.get("tool_calls"):
                return _openai_stream_response(
                    app,
                    body=body,
                    route="deepseek-web",
                    model=model,
                    bridge=bridge,
                    sse_body=build_openai_tool_calls_sse(msg["tool_calls"], model=model),
                    provider_diagnostic=getattr(web_client, "last_diagnostic", None),
                    bridge_context=bridge_context,
                )
            content = str(msg.get("content") or "")
        return _openai_stream_response(
            app,
            body=body,
            route="deepseek-web",
            model=model,
            bridge=bridge,
            sse_body=build_tool_call_sse(content, allowed_tools=allowed_tools, model=model, bridge_context=bridge_context),
            provider_diagnostic=getattr(web_client, "last_diagnostic", None),
            bridge_context=bridge_context,
        )
    return _openai_json_response(
        app,
        body=body,
        route="deepseek-web",
        model=model,
        bridge=bridge,
        parsed=parsed,
        bridge_result=bridge_result,
        bridge_context=bridge_context,
    )


def _deepseek_web_chat_payload(app: FastAPI, client: httpx.Client, body: dict[str, Any], config: GatewayConfig) -> dict[str, Any]:
    parsed, _headers = _direct_provider_chat_payload_with_headers(_deepseek_web_chat, app, client, body, config)
    return parsed


def _qwen_web_chat(app: FastAPI, client: httpx.Client, body: dict[str, Any], config: GatewayConfig) -> Response:
    account_id = _selected_direct_account_id(app, "qwen", body)
    credential = _direct_credential_for_account(app, "qwen", account_id)
    if not is_credential_authorized("qwen", credential):
        raise HTTPException(status_code=424, detail=_provider_auth_required_detail("Qwen Web"))
    payload, bridge, allowed_tools, bridge_context = _build_direct_payload(
        body,
        config,
        default_model="qwen-web/qwen3.5-plus",
        provider_native_web_search=True,
    )
    model = str(payload.get("model") or "qwen-web/qwen3.5-plus")
    skill_preflight = _skill_loader_preflight_response(app, model, bridge_context, require_namespaced_slash=True)
    if skill_preflight is not None:
        return skill_preflight
    preflight = _local_preflight_response(app, body, model, bridge_context)
    if preflight is not None:
        return preflight
    _record_completion_started_diagnostic(
        app,
        endpoint="/v1/chat/completions",
        route="qwen-web",
        model=model,
        body=body,
        stream=bool(body.get("stream")),
        bridge=bridge,
        bridge_context=bridge_context,
    )
    try:
        web_client = None
        target_pinned = bool(str(body.get("_webai_target_account_id") or "").strip())
        web_client = _build_qwen_web_client(app.state.qwen_client_factory, credential, client, config)
        try:
            data = _call_qwen_direct_with_gate(
                app,
                provider_id="qwen",
                route="qwen-web",
                account_id=account_id,
                web_client=web_client,
                payload=payload,
                config=config,
            )
        except httpx.HTTPStatusError as exc:
            alternate_account_id = "" if target_pinned or not _is_qwen_direct_rate_limit_exception(exc) else _alternate_direct_account_id(app, "qwen", account_id)
            if not alternate_account_id:
                raise
            alternate_credential = _direct_credential_for_account(app, "qwen", alternate_account_id)
            if not is_credential_authorized("qwen", alternate_credential):
                raise
            account_id = alternate_account_id
            credential = alternate_credential
            web_client = _build_qwen_web_client(app.state.qwen_client_factory, credential, client, config)
            data = _call_qwen_direct_with_gate(
                app,
                provider_id="qwen",
                route="qwen-web",
                account_id=account_id,
                web_client=web_client,
                payload=payload,
                config=config,
            )
    except HTTPException:
        raise
    except httpx.HTTPStatusError as exc:
        _raise_direct_provider_http_status_error(
            app,
            exc,
            endpoint="/v1/chat/completions",
            route="qwen-web",
            provider_label="Qwen Web",
            model=model,
            body=body,
            stream=bool(body.get("stream")),
            bridge=bridge,
            bridge_context=bridge_context,
            provider_diagnostic=getattr(web_client, "last_diagnostic", None),
            sensitive_values=(credential.get("bearer"), credential.get("cookie")),
        )
    except (TimeoutError, httpx.TimeoutException) as exc:
        provider_diagnostic = _merge_provider_diagnostics(
            getattr(web_client, "last_diagnostic", None), getattr(exc, "diagnostic", None)
        )
        _record_completion_error_diagnostic(
            app,
            endpoint="/v1/chat/completions",
            route="qwen-web",
            model=model,
            body=body,
            stream=bool(body.get("stream")),
            bridge=bridge,
            status_code=504,
            error_kind="timeout",
            error=exc,
            provider_diagnostic=provider_diagnostic,
            bridge_context=bridge_context,
        )
        diagnostic = _format_qwen_timeout_diagnostic(provider_diagnostic, None)
        raise HTTPException(
            status_code=504,
            detail=f"Qwen Web 响应超时：{exc}{diagnostic}",
            headers={"x-should-retry": "false"},
        ) from exc
    except Exception as exc:
        if _is_provider_auth_failure(None, exc):
            _record_completion_error_diagnostic(
                app,
                endpoint="/v1/chat/completions",
                route="qwen-web",
                model=model,
                body=body,
                stream=bool(body.get("stream")),
                bridge=bridge,
                status_code=424,
                error_kind="provider_auth_expired",
                error=RuntimeError(_provider_auth_expired_detail("Qwen Web")),
                provider_diagnostic=getattr(web_client, "last_diagnostic", None),
                bridge_context=bridge_context,
            )
            raise HTTPException(status_code=424, detail=_provider_auth_expired_detail("Qwen Web")) from exc
        _record_completion_error_diagnostic(
            app,
            endpoint="/v1/chat/completions",
            route="qwen-web",
            model=model,
            body=body,
            stream=bool(body.get("stream")),
            bridge=bridge,
            status_code=502,
            error_kind="provider_call_failed",
            error=exc,
            provider_diagnostic=getattr(web_client, "last_diagnostic", None),
            bridge_context=bridge_context,
        )
        raise HTTPException(status_code=502, detail=f"Qwen Web 调用失败：{exc}") from exc
    if not isinstance(data, dict):
        raise HTTPException(status_code=502, detail="Qwen Web 响应必须是 JSON 对象")
    try:
        parsed, bridge_result = _parse_bridge_chat_data(
            data,
            app=app,
            payload=payload,
            bridge=bridge,
            allowed_tools=allowed_tools,
            bridge_context=bridge_context,
            model=model,
            retry_chat=lambda retry_payload: _call_qwen_direct_with_gate(
                app,
                provider_id="qwen",
                route="qwen-web",
                account_id=account_id,
                web_client=web_client,
                payload=retry_payload,
                config=config,
            ),
            native_web_search=bool(payload.get("_webai_native_web_search")),
        )
    except HTTPException:
        raise
    except httpx.HTTPStatusError as exc:
        _raise_direct_provider_http_status_error(
            app,
            exc,
            endpoint="/v1/chat/completions",
            route="qwen-web",
            provider_label="Qwen Web",
            model=model,
            body=body,
            stream=bool(body.get("stream")),
            bridge=bridge,
            bridge_context=bridge_context,
            provider_diagnostic=getattr(web_client, "last_diagnostic", None),
            sensitive_values=(credential.get("bearer"), credential.get("cookie")),
        )
    except (TimeoutError, httpx.TimeoutException) as exc:
        provider_diagnostic = _merge_provider_diagnostics(
            getattr(web_client, "last_diagnostic", None), getattr(exc, "diagnostic", None)
        )
        _record_completion_error_diagnostic(
            app,
            endpoint="/v1/chat/completions",
            route="qwen-web",
            model=model,
            body=body,
            stream=bool(body.get("stream")),
            bridge=bridge,
            status_code=504,
            error_kind="timeout",
            error=exc,
            provider_diagnostic=provider_diagnostic,
            bridge_context=bridge_context,
        )
        diagnostic = _format_qwen_timeout_diagnostic(provider_diagnostic, None)
        raise HTTPException(
            status_code=504,
            detail=f"Qwen Web 响应超时：{exc}{diagnostic}",
            headers={"x-should-retry": "false"},
        ) from exc
    if bool(body.get("stream")):
        content = ""
        choices = parsed.get("choices") if isinstance(parsed.get("choices"), list) else []
        if choices and isinstance(choices[0], dict):
            msg = choices[0].get("message") if isinstance(choices[0].get("message"), dict) else {}
            if isinstance(msg.get("tool_calls"), list) and msg.get("tool_calls"):
                return _openai_stream_response(
                    app,
                    body=body,
                    route="qwen-web",
                    model=model,
                    bridge=bridge,
                    sse_body=build_openai_tool_calls_sse(msg["tool_calls"], model=model),
                    provider_diagnostic=getattr(web_client, "last_diagnostic", None),
                    bridge_context=bridge_context,
                )
            content = str(msg.get("content") or "")
        return _openai_stream_response(
            app,
            body=body,
            route="qwen-web",
            model=model,
            bridge=bridge,
            sse_body=build_tool_call_sse(content, allowed_tools=allowed_tools, model=model, bridge_context=bridge_context),
            provider_diagnostic=getattr(web_client, "last_diagnostic", None),
            bridge_context=bridge_context,
        )
    return _openai_json_response(
        app,
        body=body,
        route="qwen-web",
        model=model,
        bridge=bridge,
        parsed=parsed,
        bridge_result=bridge_result,
        provider_diagnostic=getattr(web_client, "last_diagnostic", None),
        bridge_context=bridge_context,
    )


def _qwen_web_chat_payload(app: FastAPI, client: httpx.Client, body: dict[str, Any], config: GatewayConfig) -> dict[str, Any]:
    parsed, _headers = _direct_provider_chat_payload_with_headers(_qwen_web_chat, app, client, body, config)
    return parsed


def _qwen_coder_chat_payload(app: FastAPI, client: httpx.Client, body: dict[str, Any], config: GatewayConfig) -> dict[str, Any]:
    parsed, _headers = _direct_provider_chat_payload_with_headers(_qwen_coder_chat, app, client, body, config)
    return parsed


def _qwen_coder_chat(app: FastAPI, client: httpx.Client, body: dict[str, Any], config: GatewayConfig) -> Response:
    account_id = _selected_direct_account_id(app, "qwen-coder", body)
    credential = _direct_credential_for_account(app, "qwen-coder", account_id)
    if not is_credential_authorized("qwen-coder", credential):
        raise HTTPException(status_code=424, detail=_provider_auth_required_detail("Qwen Coder Web"))
    payload, bridge, allowed_tools, bridge_context = _build_direct_payload(
        body,
        config,
        default_model="qwen-coder/qwen-coder-plus",
        provider_native_web_search=True,
    )
    model = str(payload.get("model") or "qwen-coder/qwen-coder-plus")
    skill_preflight = _skill_loader_preflight_response(app, model, bridge_context, require_namespaced_slash=True)
    if skill_preflight is not None:
        return skill_preflight
    preflight = _local_preflight_response(app, body, model, bridge_context)
    if preflight is not None:
        return preflight
    _record_completion_started_diagnostic(
        app,
        endpoint="/v1/chat/completions",
        route="qwen-coder",
        model=model,
        body=body,
        stream=bool(body.get("stream")),
        bridge=bridge,
        bridge_context=bridge_context,
    )
    try:
        web_client = None
        target_pinned = bool(str(body.get("_webai_target_account_id") or "").strip())
        web_client = _build_qwen_coder_client(app.state.qwen_coder_client_factory, credential, client, config)
        try:
            data = _call_qwen_direct_with_gate(
                app,
                provider_id="qwen-coder",
                route="qwen-coder",
                account_id=account_id,
                web_client=web_client,
                payload=payload,
                config=config,
            )
        except httpx.HTTPStatusError as exc:
            alternate_account_id = "" if target_pinned or not _is_qwen_direct_rate_limit_exception(exc) else _alternate_direct_account_id(app, "qwen-coder", account_id)
            if not alternate_account_id:
                raise
            alternate_credential = _direct_credential_for_account(app, "qwen-coder", alternate_account_id)
            if not is_credential_authorized("qwen-coder", alternate_credential):
                raise
            account_id = alternate_account_id
            credential = alternate_credential
            web_client = _build_qwen_coder_client(app.state.qwen_coder_client_factory, credential, client, config)
            data = _call_qwen_direct_with_gate(
                app,
                provider_id="qwen-coder",
                route="qwen-coder",
                account_id=account_id,
                web_client=web_client,
                payload=payload,
                config=config,
            )
    except HTTPException:
        raise
    except httpx.HTTPStatusError as exc:
        _raise_direct_provider_http_status_error(
            app,
            exc,
            endpoint="/v1/chat/completions",
            route="qwen-coder",
            provider_label="Qwen Coder Web",
            model=model,
            body=body,
            stream=bool(body.get("stream")),
            bridge=bridge,
            bridge_context=bridge_context,
            provider_diagnostic=getattr(web_client, "last_diagnostic", None),
            sensitive_values=(credential.get("bearer"), credential.get("cookie")),
        )
    except (TimeoutError, httpx.TimeoutException) as exc:
        provider_diagnostic = _merge_provider_diagnostics(
            getattr(web_client, "last_diagnostic", None), getattr(exc, "diagnostic", None)
        )
        _record_completion_error_diagnostic(
            app,
            endpoint="/v1/chat/completions",
            route="qwen-coder",
            model=model,
            body=body,
            stream=bool(body.get("stream")),
            bridge=bridge,
            status_code=504,
            error_kind="timeout",
            error=exc,
            provider_diagnostic=provider_diagnostic,
            bridge_context=bridge_context,
        )
        diagnostic = _format_qwen_coder_timeout_diagnostic(provider_diagnostic, None)
        raise HTTPException(
            status_code=504,
            detail=f"Qwen Coder Web 响应超时：{exc}{diagnostic}",
            headers={"x-should-retry": "false"},
        ) from exc
    except Exception as exc:
        if _is_provider_auth_failure(None, exc):
            _record_completion_error_diagnostic(
                app,
                endpoint="/v1/chat/completions",
                route="qwen-coder",
                model=model,
                body=body,
                stream=bool(body.get("stream")),
                bridge=bridge,
                status_code=424,
                error_kind="provider_auth_expired",
                error=RuntimeError(_provider_auth_expired_detail("Qwen Coder Web")),
                provider_diagnostic=getattr(web_client, "last_diagnostic", None),
                bridge_context=bridge_context,
            )
            raise HTTPException(status_code=424, detail=_provider_auth_expired_detail("Qwen Coder Web")) from exc
        _record_completion_error_diagnostic(
            app,
            endpoint="/v1/chat/completions",
            route="qwen-coder",
            model=model,
            body=body,
            stream=bool(body.get("stream")),
            bridge=bridge,
            status_code=502,
            error_kind="provider_call_failed",
            error=exc,
            provider_diagnostic=getattr(web_client, "last_diagnostic", None),
            bridge_context=bridge_context,
        )
        raise HTTPException(status_code=502, detail=f"Qwen Coder Web 调用失败：{exc}") from exc
    if not isinstance(data, dict):
        raise HTTPException(status_code=502, detail="Qwen Coder Web 响应必须是 JSON 对象")
    try:
        parsed, bridge_result = _parse_bridge_chat_data(
            data,
            app=app,
            payload=payload,
            bridge=bridge,
            allowed_tools=allowed_tools,
            bridge_context=bridge_context,
            model=model,
            retry_chat=lambda retry_payload: _call_qwen_direct_with_gate(
                app,
                provider_id="qwen-coder",
                route="qwen-coder",
                account_id=account_id,
                web_client=web_client,
                payload=retry_payload,
                config=config,
            ),
            native_web_search=bool(payload.get("_webai_native_web_search")),
        )
    except HTTPException:
        raise
    except httpx.HTTPStatusError as exc:
        _raise_direct_provider_http_status_error(
            app,
            exc,
            endpoint="/v1/chat/completions",
            route="qwen-coder",
            provider_label="Qwen Coder Web",
            model=model,
            body=body,
            stream=bool(body.get("stream")),
            bridge=bridge,
            bridge_context=bridge_context,
            provider_diagnostic=getattr(web_client, "last_diagnostic", None),
            sensitive_values=(credential.get("bearer"), credential.get("cookie")),
        )
    except (TimeoutError, httpx.TimeoutException) as exc:
        provider_diagnostic = _merge_provider_diagnostics(
            getattr(web_client, "last_diagnostic", None), getattr(exc, "diagnostic", None)
        )
        _record_completion_error_diagnostic(
            app,
            endpoint="/v1/chat/completions",
            route="qwen-coder",
            model=model,
            body=body,
            stream=bool(body.get("stream")),
            bridge=bridge,
            status_code=504,
            error_kind="timeout",
            error=exc,
            provider_diagnostic=provider_diagnostic,
            bridge_context=bridge_context,
        )
        diagnostic = _format_qwen_coder_timeout_diagnostic(provider_diagnostic, None)
        raise HTTPException(
            status_code=504,
            detail=f"Qwen Coder Web 响应超时：{exc}{diagnostic}",
            headers={"x-should-retry": "false"},
        ) from exc
    if bool(body.get("stream")):
        content = ""
        choices = parsed.get("choices") if isinstance(parsed.get("choices"), list) else []
        if choices and isinstance(choices[0], dict):
            msg = choices[0].get("message") if isinstance(choices[0].get("message"), dict) else {}
            if isinstance(msg.get("tool_calls"), list) and msg.get("tool_calls"):
                return _openai_stream_response(
                    app,
                    body=body,
                    route="qwen-coder",
                    model=model,
                    bridge=bridge,
                    sse_body=build_openai_tool_calls_sse(msg["tool_calls"], model=model),
                    provider_diagnostic=getattr(web_client, "last_diagnostic", None),
                    bridge_context=bridge_context,
                )
            content = str(msg.get("content") or "")
        return _openai_stream_response(
            app,
            body=body,
            route="qwen-coder",
            model=model,
            bridge=bridge,
            sse_body=build_tool_call_sse(content, allowed_tools=allowed_tools, model=model, bridge_context=bridge_context),
            provider_diagnostic=getattr(web_client, "last_diagnostic", None),
            bridge_context=bridge_context,
        )
    return _openai_json_response(
        app,
        body=body,
        route="qwen-coder",
        model=model,
        bridge=bridge,
        parsed=parsed,
        bridge_result=bridge_result,
        provider_diagnostic=getattr(web_client, "last_diagnostic", None),
        bridge_context=bridge_context,
    )


def _build_qwen_coder_client(factory: Any, credential: dict[str, Any], client: httpx.Client, config: GatewayConfig) -> Any:
    timeout = config.provider_runtime.request_timeout_seconds
    prompt_max_chars = config.provider_runtime.prompt_max_chars
    try:
        signature = inspect.signature(factory)
    except (TypeError, ValueError):
        accepted_params: set[str] | None = None
        accepts_kwargs = True
    else:
        accepted_params = set(signature.parameters)
        accepts_kwargs = any(
            param.kind == inspect.Parameter.VAR_KEYWORD for param in signature.parameters.values()
        )
    kwargs: dict[str, Any] = {"http_client": client}
    if accepts_kwargs or (accepted_params and "request_timeout_seconds" in accepted_params):
        kwargs["request_timeout_seconds"] = timeout
    if accepts_kwargs or (accepted_params and "prompt_max_chars" in accepted_params):
        kwargs["prompt_max_chars"] = prompt_max_chars
    return factory(credential, **kwargs)


def _format_qwen_coder_timeout_diagnostic(client_diagnostic: Any, exception_diagnostic: Any) -> str:
    merged = _merge_provider_diagnostics(client_diagnostic, exception_diagnostic)
    if not merged:
        return ""
    allowed_keys = (
        "prompt_chars",
        "prompt_max_chars",
        "prompt_compacted",
        "prompt_task_state_preserved",
        "prompt_task_state_chars",
        "prompt_task_count",
        "prompt_recent_tool_call_count",
        "prompt_compaction_strategy",
        "prompt_history_entry_count",
        "prompt_latest_entry_count",
        "message_count",
        "stream_events",
        "json_events",
        "output_chars",
        "think_chars",
        "artifact_chars",
    )
    parts = [f"{key}={merged[key]}" for key in allowed_keys if key in merged]
    return f"锛沝iagnostic: {', '.join(parts)}" if parts else ""


def _openai_response_content(data: dict[str, Any]) -> str:
    choices = data.get("choices") if isinstance(data.get("choices"), list) else []
    choice0 = choices[0] if choices and isinstance(choices[0], dict) else {}
    msg = choice0.get("message") if isinstance(choice0.get("message"), dict) else {}
    return str(msg.get("content") or "")


def _build_qwen_web_client(factory: Any, credential: dict[str, Any], client: httpx.Client, config: GatewayConfig) -> Any:
    timeout = config.provider_runtime.request_timeout_seconds
    prompt_max_chars = config.provider_runtime.prompt_max_chars
    try:
        signature = inspect.signature(factory)
    except (TypeError, ValueError):
        accepted_params: set[str] | None = None
        accepts_kwargs = True
    else:
        accepted_params = set(signature.parameters)
        accepts_kwargs = any(
            param.kind == inspect.Parameter.VAR_KEYWORD for param in signature.parameters.values()
        )
    kwargs: dict[str, Any] = {"http_client": client}
    if accepts_kwargs or (accepted_params and "request_timeout_seconds" in accepted_params):
        kwargs["request_timeout_seconds"] = timeout
    if accepts_kwargs or (accepted_params and "prompt_max_chars" in accepted_params):
        kwargs["prompt_max_chars"] = prompt_max_chars
    return factory(credential, **kwargs)


def _build_deepseek_direct_payload(
    body: dict[str, Any],
    config: GatewayConfig,
) -> tuple[dict[str, Any], bool, set[str], Any]:
    model = str(body.get("model") or DEEPSEEK_DEFAULT_MODEL)
    payload = {**body, "model": model}
    payload["messages"] = _with_response_language_instruction(
        payload.get("messages"),
        config.provider_runtime.response_language,
    )
    bridge_mode = (config.tool_bridge.mode or config.upstream.tool_mode or "strict").strip().lower()
    bridge_context = build_context(
        payload.get("tools"),
        config.tool_bridge,
        mode=bridge_mode,
        model=model,
        tool_choice=payload.get("tool_choice"),
    )
    bridge_context = prefer_local_tools_for_local_agent_task(
        bridge_context,
        payload.get("messages"),
        tool_choice=payload.get("tool_choice"),
    )
    return payload, False, set(), bridge_context


def _build_deepseek_web_client(factory: Any, credential: dict[str, Any], client: httpx.Client, config: GatewayConfig) -> Any:
    timeout = config.provider_runtime.request_timeout_seconds
    prompt_max_chars = config.provider_runtime.prompt_max_chars
    ds2api_base_url = config.provider_runtime.deepseek_ds2api_base_url
    try:
        signature = inspect.signature(factory)
    except (TypeError, ValueError):
        accepted_params: set[str] | None = None
        accepts_kwargs = True
    else:
        accepted_params = set(signature.parameters)
        accepts_kwargs = any(
            param.kind == inspect.Parameter.VAR_KEYWORD for param in signature.parameters.values()
        )
    kwargs: dict[str, Any] = {"http_client": client}
    if accepts_kwargs or (accepted_params and "request_timeout_seconds" in accepted_params):
        kwargs["request_timeout_seconds"] = timeout
    if accepts_kwargs or (accepted_params and "prompt_max_chars" in accepted_params):
        kwargs["prompt_max_chars"] = prompt_max_chars
    if accepts_kwargs or (accepted_params and "ds2api_base_url" in accepted_params):
        kwargs["ds2api_base_url"] = ds2api_base_url
    return factory(credential, **kwargs)


def _format_qwen_timeout_diagnostic(client_diagnostic: Any, exception_diagnostic: Any) -> str:
    merged = _merge_provider_diagnostics(client_diagnostic, exception_diagnostic)
    if not merged:
        return ""
    allowed_keys = (
        "prompt_chars",
        "prompt_max_chars",
        "prompt_compacted",
        "prompt_task_state_preserved",
        "prompt_task_state_chars",
        "prompt_task_count",
        "prompt_recent_tool_call_count",
        "prompt_compaction_strategy",
        "prompt_history_entry_count",
        "prompt_latest_entry_count",
        "message_count",
        "stream_events",
        "json_events",
        "output_chars",
        "think_chars",
    )
    parts = [f"{key}={merged[key]}" for key in allowed_keys if key in merged]
    return f"锛沝iagnostic: {', '.join(parts)}" if parts else ""


def _merge_provider_diagnostics(*items: Any) -> dict[str, Any]:
    merged: dict[str, Any] = {}
    for item in items:
        if isinstance(item, dict):
            merged.update(item)
    return merged


def _provider_dependency_status_code(status_code: int) -> int:
    if status_code in {401, 403}:
        return 424
    if 400 <= status_code <= 599:
        return status_code
    return 502


def _raise_direct_provider_http_status_error(
    app: FastAPI,
    exc: httpx.HTTPStatusError,
    *,
    endpoint: str,
    route: str,
    provider_label: str,
    model: str,
    body: dict[str, Any],
    stream: bool,
    bridge: bool,
    bridge_context: Any,
    provider_diagnostic: Any = None,
    sensitive_values: Any = None,
) -> None:
    response = exc.response
    status_code = _provider_dependency_status_code(response.status_code)
    preview = _extract_upstream_error_preview(response) or response.reason_phrase or "empty response body"
    message = f"{provider_label} upstream returned HTTP {response.status_code}: {preview}"
    if _is_provider_auth_failure(response.status_code, preview):
        sanitized = _provider_auth_expired_detail(provider_label)
        error_kind = "provider_auth_expired"
    else:
        sanitized = _preview_text(
            _redact_known_sensitive_values(_redact_sensitive_text(message), sensitive_values),
            max_chars=800,
        )
        error_kind = "provider_http_error"
    _record_completion_error_diagnostic(
        app,
        endpoint=endpoint,
        route=route,
        model=model,
        body=body,
        stream=stream,
        bridge=bridge,
        status_code=status_code,
        error_kind=error_kind,
        error=RuntimeError(sanitized),
        provider_diagnostic=provider_diagnostic,
        bridge_context=bridge_context,
    )
    raise HTTPException(status_code=status_code, detail=sanitized) from exc


def _direct_provider_chat_payload_with_headers(
    chat_func: Any,
    app: FastAPI,
    client: httpx.Client,
    body: dict[str, Any],
    config: GatewayConfig,
) -> tuple[dict[str, Any], dict[str, str]]:
    response = chat_func(app, client, body, config)
    if isinstance(response, JSONResponse):
        return json_loads_response(response), _tool_bridge_headers_from_response(response)
    raise HTTPException(status_code=400, detail="Anthropic 兼容接口暂不支持 direct provider 流式响应")


def _tool_bridge_headers_from_response(response: Response) -> dict[str, str]:
    return {
        key: value
        for key, value in response.headers.items()
        if key.lower().startswith("x-webai-tool-bridge-")
    }


def json_loads_response(response: JSONResponse) -> dict[str, Any]:
    data = json.loads(bytes(response.body).decode("utf-8"))
    if not isinstance(data, dict):
        raise HTTPException(status_code=502, detail="Gateway response must be a JSON object")
    return data


app = create_app()
