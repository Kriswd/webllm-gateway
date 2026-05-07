from __future__ import annotations

from collections import OrderedDict, deque
from datetime import datetime, timezone
import inspect
import json
import re
import secrets
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
from webai_gateway.deepseek_web import DEEPSEEK_DEFAULT_MODEL, DeepSeekWebClient, is_deepseek_web_model
from webai_gateway.ds2api_oracle import DS2API_ORACLE_COMMIT, DS2API_ORACLE_VERSION
from webai_gateway.openai_api import (
    bridge_error_headers,
    build_incomplete_response_retry_payload,
    build_missing_required_tool_input_recovery_payload,
    build_native_web_search_retry_payload,
    build_off_task_question_recovery_payload,
    build_preflight_chat_response,
    build_repair_payload,
    build_required_tool_choice_recovery_payload,
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
from webai_gateway.tool_bridge import BridgeError, BridgeResult, build_context, parse_tool_response
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
WEBAI2API_AUTH_COOKIE_HINTS: dict[str, tuple[str, ...]] = {
    "chatgpt": (
        "__secure-next-auth.session-token",
        "oai-client-auth-info",
        "__secure-oai-is",
    ),
}
TOOL_BRIDGE_EVENT_LIMIT = 200
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
    app.state.tool_call_registry = OrderedDict()
    app.state.auto_research_fixture_dir = Path(auto_research_fixture_dir) if auto_research_fixture_dir is not None else _default_auto_research_fixture_dir()
    app.state.runtime_started_at = utc_now()
    app.state.runtime_started_epoch = datetime.now(timezone.utc).timestamp()

    static_dir = Path(__file__).with_name("static")
    webai2api_ui_dir = Path(native_ui_dir) if native_ui_dir is not None else _default_native_ui_dir()
    if static_dir.exists():
        app.mount("/static", StaticFiles(directory=static_dir), name="static")

    def current_config() -> GatewayConfig:
        return app.state.config

    def require_auth(authorization: str | None, x_api_key: str | None = None) -> None:
        cfg = current_config()
        if not cfg.server.api_key:
            return
        expected = f"Bearer {cfg.server.api_key}"
        if authorization != expected and x_api_key != cfg.server.api_key:
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
        models_payload = _load_gateway_models(client, cfg)
        models = models_payload.get("data") if isinstance(models_payload.get("data"), list) else []
        provider_data = provider_payload(app.state.credential_store)["providers"]
        webai2api_instances = _load_webai2api_instances(client, cfg)
        models = _with_onboarding_provider_catalog_models(models, provider_data, webai2api_instances)
        model_ids = {item.get("id") for item in models if isinstance(item, dict)}
        webai2api_auth = _load_webai2api_auth_states(client, cfg, provider_data, webai2api_instances)
        providers: list[dict[str, Any]] = []
        authorized_count = 0
        authorized_direct = 0
        webai2api_count = 0
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
            is_authorized = bool(credential.get("authorized"))
            if is_authorized:
                authorized_count += 1
            if is_direct and is_authorized:
                authorized_direct += 1
            if not is_direct:
                webai2api_count += 1
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
        connection_profiles = _onboarding_connection_profiles(providers, cfg)
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
                "providers": len(providers),
                "models": len(models),
                "authorizedProviders": authorized_count,
                "authorizedDirectProviders": authorized_direct,
                "webAI2APIProviders": webai2api_count,
            },
            "providers": providers,
            "models": models,
            "connectionProfiles": connection_profiles,
            "recommendedConnectionProfile": _recommended_connection_profile(connection_profiles),
        }


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
        if provider_id != "chatgpt":
            return False
        return bool(_webai2api_provider_instances(provider, webai2api_instances))

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
        return []

    def _provider_catalog_model_payload(provider: dict[str, Any], model_id: str) -> dict[str, Any]:
        provider_id = str(provider.get("id") or "")
        return {
            "id": model_id,
            "object": "model",
            "owned_by": provider_id,
            "type": "text",
            "capabilities": {
                "tool_bridge": str(provider.get("toolBridge") or "strict").strip().lower() != "off",
                "supports_native_tools": bool(provider.get("supportsNativeTools")),
                "preferred_protocol": str(provider.get("preferredProtocol") or "openai"),
            },
            "availability_source": "provider_catalog",
        }

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
        login_url = str(provider.get("loginUrl") or "")
        parsed = urlsplit(login_url)
        host = parsed.netloc or parsed.path
        return [host.split("@")[-1].split(":")[0]] if host else []

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
        credential = app.state.credential_store.get(provider_id)
        if not is_credential_authorized(provider_id, credential):
            return []
        account_id = direct_account_id(provider_id)
        metadata = app.state.account_registry.metadata(account_id)
        validation = _public_account_validation(metadata.get("validation"))
        return [
            {
                "id": account_id,
                "providerId": provider_id,
                "source": "direct-profile",
                "displayName": str(metadata.get("displayName") or "默认账号"),
                "planType": str(metadata.get("planType") or "unknown"),
                "authorized": True,
                "current": False,
                "instanceName": None,
                "workerName": None,
                "workerType": None,
                "availableModelCount": _available_model_count_from_validation(provider_models, validation),
                "lastValidatedAt": metadata.get("lastValidatedAt") if isinstance(metadata.get("lastValidatedAt"), str) else None,
                "validation": validation,
            }
        ]

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
        models_payload = _load_gateway_models(client, cfg)
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
        models_payload = _load_gateway_models(client, cfg)
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
            result = await run_in_threadpool(_validate_account_model, cfg, model_id)
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

    @app.post("/api/admin/webai2api/login/start")
    async def admin_webai2api_login_start(request: Request) -> dict[str, Any]:
        require_local_admin(request)
        body = await _json_body(request)
        provider_id = str(body.get("providerId") or body.get("provider_id") or "").strip()
        provider = get_provider(provider_id)
        if provider.route == "direct":
            raise HTTPException(status_code=400, detail="该 Provider 使用 Gateway 直连授权，不需要 WebAI2API 登录模式")
        cfg = current_config()
        instances = _load_webai2api_instances(client, cfg)
        if not instances:
            raise HTTPException(status_code=502, detail="WebAI2API 工作池为空或不可读取，无法启动登录模式")
        requested_worker = str(body.get("workerName") or body.get("worker_name") or "").strip()
        create_account = bool(body.get("newAccount", body.get("new_account", not requested_worker)))
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
            worker_name = requested_worker or _first_webai2api_worker_for_provider(provider, instances)
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
            "message": (
                f"已为 {provider.name} 创建独立浏览器 Profile 并进入登录模式"
                if create_account
                else f"已进入 {worker_name} 的登录模式"
            ),
        }

    @app.post("/api/admin/webai2api/login/finish")
    async def admin_webai2api_login_finish(request: Request) -> dict[str, Any]:
        require_local_admin(request)
        cfg = current_config()
        restarted = _restart_webai2api_api_mode(client, cfg)
        if not restarted:
            raise HTTPException(status_code=502, detail="WebAI2API 未能恢复普通 API 模式，请在缓存与重启页面手动重启")
        app.state.webai2api_login_mode_started_at = None
        return {"success": True, "message": "WebAI2API 已恢复普通 API 模式，可以继续调用模型"}

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

    def _first_webai2api_worker_for_provider(provider: Any, instances: list[Any]) -> str:
        provider_dict = {"adapters": list(getattr(provider, "adapters", ()))}
        for instance in instances:
            if not isinstance(instance, dict):
                continue
            workers = instance.get("workers") if isinstance(instance.get("workers"), list) else []
            for worker in workers:
                if isinstance(worker, dict) and _worker_supports_provider(worker, provider_dict):
                    worker_name = str(worker.get("name") or "")
                    if worker_name:
                        return worker_name
        return ""

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
        provider_dict = {"adapters": list(getattr(provider, "adapters", ()))}
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
        return active_provider_workers == [(parsed_account.instance_name, parsed_account.worker_name)]

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

    def _validate_account_model(cfg: GatewayConfig, model_id: str) -> dict[str, Any]:
        payload = {
            "model": model_id,
            "messages": [{"role": "user", "content": "请只回复 OK。"}],
            "stream": False,
            "max_tokens": 8,
        }
        try:
            response = client.post(
                cfg.upstream.base_url.rstrip("/") + "/chat/completions",
                json=payload,
                headers=upstream_headers(cfg),
                timeout=max(30, int(cfg.provider_runtime.request_timeout_seconds or 300)),
            )
        except Exception as exc:
            return {"status": "unavailable", "message": _preview_text(_redact_sensitive_text(str(exc)), max_chars=500)}
        if response.status_code < 400:
            return {"status": "available", "message": "验证通过"}
        return {
            "status": "unavailable",
            "message": _preview_text(_redact_sensitive_text(_upstream_http_error_message(response)), max_chars=500),
        }


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
    def models(authorization: str | None = Header(default=None), x_api_key: str | None = Header(default=None, alias="x-api-key")) -> JSONResponse:
        require_auth(authorization, x_api_key)
        cfg = current_config()
        try:
            response = client.get(cfg.upstream.base_url.rstrip("/") + "/models", headers=upstream_headers(cfg))
            if response.status_code == 200:
                data = response.json()
                if isinstance(data, dict) and isinstance(data.get("data"), list):
                    return JSONResponse(_append_web_models(data))
        except Exception:
            pass
        return JSONResponse(_append_web_models({"object": "list", "data": [{"id": cfg.upstream.model, "object": "model"}]}))

    @app.post("/v1/chat/completions")
    async def chat(
        request: Request,
        authorization: str | None = Header(default=None),
        x_api_key: str | None = Header(default=None, alias="x-api-key"),
    ) -> Response:
        require_auth(authorization, x_api_key)
        cfg = current_config()
        body = await _json_body(request)
        if is_deepseek_web_model(body.get("model")):
            return await run_in_threadpool(_deepseek_web_chat, app, client, body, cfg)
        if is_qwen_web_model(body.get("model")):
            return await run_in_threadpool(_qwen_web_chat, app, client, body, cfg)
        if is_qwen_coder_model(body.get("model")):
            return await run_in_threadpool(_qwen_coder_chat, app, client, body, cfg)
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
    ) -> JSONResponse:
        require_auth(authorization, x_api_key)
        body = await _json_body(request)
        return JSONResponse(anthropic_count_tokens(body))

    @app.post("/messages")
    @app.post("/v1/messages")
    async def anthropic_messages(
        request: Request,
        authorization: str | None = Header(default=None),
        x_api_key: str | None = Header(default=None, alias="x-api-key"),
    ) -> Response:
        require_auth(authorization, x_api_key)
        cfg = current_config()
        body = await _json_body(request)
        try:
            openai_body = anthropic_body_to_openai(
                body,
                tool_call_registry=getattr(app.state, "tool_call_registry", None),
            )
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
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
            original_model=str(body.get("model") or openai_body.get("model") or cfg.upstream.model),
        )
        response_headers = bridge_error_headers(bridge_result)
        response_headers.update(direct_bridge_headers)
        if bool(body.get("stream")):
            sse_body = anthropic_response_to_sse(anthropic_message)
            _record_completion_diagnostic(
                app,
                endpoint="/v1/messages",
                route=_provider_route(str(openai_body.get("model") or cfg.upstream.model)),
                model=str(body.get("model") or openai_body.get("model") or cfg.upstream.model),
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
            route=_provider_route(str(openai_body.get("model") or cfg.upstream.model)),
            model=str(body.get("model") or openai_body.get("model") or cfg.upstream.model),
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
        backend_kind, backend_base_url = _onboarding_backend_metadata(provider, cfg)
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
    event = {"at": utc_now(), "kind": kind}
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
    }
    fields: dict[str, Any] = {}
    for source_key, target_key in mapping.items():
        if source_key not in diagnostic:
            continue
        value = diagnostic[source_key]
        if isinstance(value, (bool, int, float, str)):
            fields[target_key] = value
    return fields


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
    tools = body.get("tools")
    if isinstance(tools, list):
        fields["requestToolCount"] = len(tools)
        names = _openai_tool_names(tools)
        if names:
            fields["requestToolNames"] = names[:16]
            if len(names) > 16:
                fields["requestToolNamesTruncated"] = len(names) - 16
    tool_choice = body.get("tool_choice")
    if isinstance(tool_choice, str):
        fields["requestToolChoice"] = tool_choice
    elif isinstance(tool_choice, dict):
        fields["requestToolChoice"] = _preview_text(json.dumps(tool_choice, ensure_ascii=False), max_chars=160)
    max_tokens = body.get("max_tokens")
    if isinstance(max_tokens, int):
        fields["requestMaxTokens"] = max_tokens
    return fields


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
    options = getattr(bridge_context, "options", None)
    exposure_policy = str(getattr(options, "exposure_policy", "") or "")
    if exposure_policy:
        fields["toolBridgeExposurePolicy"] = exposure_policy
    tool_profile = str(getattr(options, "tool_profile", "") or "")
    if tool_profile:
        fields["toolBridgeToolProfile"] = tool_profile
    return fields


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
    return fields


def _openai_sse_response_fields(text: str) -> dict[str, Any]:
    payload_count = 0
    content_parts: list[str] = []
    tool_call_chunks = 0
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


def _provider_route(model: str) -> str:
    if is_deepseek_web_model(model):
        return "deepseek-web"
    if is_qwen_web_model(model):
        return "qwen-web"
    if is_qwen_coder_model(model):
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


def _append_web_models(data: dict[str, Any], *, include_webai2api_catalog: bool = False) -> dict[str, Any]:
    items = data.get("data") if isinstance(data.get("data"), list) else []
    seen = {item.get("id") for item in items if isinstance(item, dict)}
    for item in catalog_model_payloads(include_webai2api=include_webai2api_catalog):
        model_id = item["id"]
        if model_id not in seen:
            items.append(item)
    return {**data, "object": data.get("object") or "list", "data": items}


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


def _build_direct_payload(
    body: dict[str, Any],
    config: GatewayConfig,
    *,
    default_model: str,
    provider_native_web_search: bool = False,
) -> tuple:
    direct_body = {**body, "model": str(body.get("model") or default_model)}
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
        parsed, bridge_result = _finalize_retry_exhausted_format_error(
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
        error = BridgeError(error_kind, _controller_retry_error_message(error_kind), repairable=False)
        metadata = _bridge_result_controller_metadata(
            bridge_result,
            bridge_context=bridge_context,
            controller_state=decision.state,
            controller_reason=decision.reason,
            retry_state=next_state,
        )
        content = _controller_retry_exhausted_text(error, bridge_context=bridge_context)
        return (
            _openai_response_from_stream_buffer(content, finish_reason="stop", model=model),
            BridgeResult(
                content=content,
                tool_calls=[],
                error=error,
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


_RETRY_EXHAUSTED_FORMAT_ERROR_KINDS = {"malformed_json", "empty_tool_call", "invalid_tool_call", "invalid_input"}


def _finalize_retry_exhausted_format_error(
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
        and str(getattr(error, "kind", "") or "") in _RETRY_EXHAUSTED_FORMAT_ERROR_KINDS
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
    return _openai_response_from_stream_buffer(content, finish_reason="stop", model=model), finalized


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


def _deepseek_web_chat(app: FastAPI, client: httpx.Client, body: dict[str, Any], config: GatewayConfig) -> Response:
    credential = app.state.credential_store.get("deepseek-web")
    if not is_credential_authorized("deepseek-web", credential):
        raise HTTPException(status_code=401, detail="请先在控制台重新完成 DeepSeek 网页登录授权，确保已捕获 bearer token")
    payload, bridge, allowed_tools, bridge_context = _build_deepseek_direct_payload(body, config)
    model = str(payload.get("model") or DEEPSEEK_DEFAULT_MODEL)
    preflight = _local_preflight_response(app, body, model, bridge_context)
    if preflight is not None:
        return preflight
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
    try:
        web_client = _build_deepseek_web_client(app.state.deepseek_client_factory, credential, client, config)
        data = web_client.chat_completions(payload)
    except HTTPException:
        raise
    except Exception as exc:
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
            retry_chat=web_client.chat_completions,
            native_web_search=bool(payload.get("_webai_native_web_search")),
        )
    except HTTPException:
        raise
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
                )
            content = str(msg.get("content") or "")
        return _openai_stream_response(
            app,
            body=body,
            route="deepseek-web",
            model=model,
            bridge=bridge,
            sse_body=build_tool_call_sse(content, allowed_tools=allowed_tools, model=model, bridge_context=bridge_context),
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
    credential = app.state.credential_store.get("qwen")
    if not is_credential_authorized("qwen", credential):
        raise HTTPException(status_code=401, detail="请先在控制台完成 Qwen 网页登录授权")
    payload, bridge, allowed_tools, bridge_context = _build_direct_payload(
        body,
        config,
        default_model="qwen-web/qwen3.5-plus",
        provider_native_web_search=True,
    )
    model = str(payload.get("model") or "qwen-web/qwen3.5-plus")
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
        web_client = _build_qwen_web_client(app.state.qwen_client_factory, credential, client, config)
        data = web_client.chat_completions(payload)
    except HTTPException:
        raise
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
            retry_chat=web_client.chat_completions,
            native_web_search=bool(payload.get("_webai_native_web_search")),
        )
    except HTTPException:
        raise
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
    credential = app.state.credential_store.get("qwen-coder")
    if not is_credential_authorized("qwen-coder", credential):
        raise HTTPException(status_code=401, detail="请先在控制台完成 Qwen Coder 网页登录授权")
    payload, bridge, allowed_tools, bridge_context = _build_direct_payload(
        body,
        config,
        default_model="qwen-coder/qwen-coder-plus",
        provider_native_web_search=True,
    )
    model = str(payload.get("model") or "qwen-coder/qwen-coder-plus")
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
        web_client = _build_qwen_coder_client(app.state.qwen_coder_client_factory, credential, client, config)
        data = web_client.chat_completions(payload)
    except HTTPException:
        raise
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
            retry_chat=web_client.chat_completions,
            native_web_search=bool(payload.get("_webai_native_web_search")),
        )
    except HTTPException:
        raise
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
    bridge_context = build_context([], config.tool_bridge, mode=bridge_mode, model=model)
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
