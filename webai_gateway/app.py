from __future__ import annotations

from collections import OrderedDict, deque
from datetime import datetime, timezone
import inspect
import json
import re
import secrets
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit, urlunsplit

import httpx
from fastapi import FastAPI, Header, HTTPException, Request
from fastapi.responses import FileResponse, JSONResponse, Response
from fastapi.staticfiles import StaticFiles
from starlette.concurrency import run_in_threadpool

from webai_gateway.auto_research import build_auto_research_status
from webai_gateway.config import GatewayConfig, config_to_admin, config_to_public, load_config, save_config, update_config
from webai_gateway.deepseek_web import DEEPSEEK_DEFAULT_MODEL, DeepSeekWebClient, is_deepseek_web_model
from webai_gateway.openai_api import (
    bridge_error_headers,
    build_incomplete_response_retry_payload,
    build_native_web_search_retry_payload,
    build_off_task_question_recovery_payload,
    build_preflight_chat_response,
    build_repair_payload,
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
TOOL_BRIDGE_EVENT_LIMIT = 200
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
        model_ids = {item.get("id") for item in models if isinstance(item, dict)}
        provider_data = provider_payload(app.state.credential_store)["providers"]
        providers: list[dict[str, Any]] = []
        authorized_direct = 0
        webai2api_count = 0
        for provider in provider_data:
            declared_models = provider.get("availableModels")
            if not isinstance(declared_models, list):
                declared_models = provider.get("models", [])
            provider_models = [model_id for model_id in declared_models if model_id in model_ids]
            is_direct = provider.get("route") == "direct"
            is_authorized = bool(provider.get("credential", {}).get("authorized"))
            if is_direct and is_authorized:
                authorized_direct += 1
            if not is_direct:
                webai2api_count += 1
            providers.append(
                {
                    **provider,
                    "availableModels": provider_models,
                    "modelCount": len(provider_models),
                    "loginKind": "direct" if is_direct else "webai2api",
                }
            )
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
                "authorizedDirectProviders": authorized_direct,
                "webAI2APIProviders": webai2api_count,
            },
            "providers": providers,
            "models": models,
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
        return await _proxy_to_sidecar(client, request, _sidecar_root_from_base_url(cfg.upstream.base_url), "admin", admin_path)

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
        response = await run_in_threadpool(post_upstream, client, cfg, payload)
        response.raise_for_status()
        if bool(body.get("stream")):
            if bridge:
                content, _finish_reason = parse_sse_text(response.text)
                sse_body = build_tool_call_sse(
                    content,
                    allowed_tools=allowed_tools,
                    model=cfg.upstream.model,
                    bridge_context=bridge_context,
                )
                return _openai_stream_response(
                    app,
                    body=body,
                    route="upstream",
                    model=str(payload.get("model") or cfg.upstream.model),
                    bridge=bridge,
                    sse_body=sse_body,
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
            retry_response = post_upstream(client, cfg, retry_payload)
            retry_response.raise_for_status()
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
        return _openai_json_response(app, body=body, route="upstream", model=model, bridge=bridge, parsed=parsed, bridge_result=bridge_result)

    @app.post("/v1/messages/count_tokens")
    async def anthropic_count_tokens_route(
        request: Request,
        authorization: str | None = Header(default=None),
        x_api_key: str | None = Header(default=None, alias="x-api-key"),
    ) -> JSONResponse:
        require_auth(authorization, x_api_key)
        body = await _json_body(request)
        return JSONResponse(anthropic_count_tokens(body))

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
            preflight_data = build_preflight_chat_response(model, bridge_context)
            if preflight_data is not None:
                _record_tool_bridge_event(app, "local_repo_preflight", model=model, **_tool_call_event_fields(preflight_data))
                parsed = preflight_data
            else:
                response = await run_in_threadpool(post_upstream, client, cfg, payload)
                response.raise_for_status()
                data = response.json()
                if not isinstance(data, dict):
                    raise HTTPException(status_code=502, detail="Upstream response must be a JSON object")

                def retry_upstream(retry_payload: dict[str, Any]) -> dict[str, Any] | None:
                    retry_response = post_upstream(client, cfg, retry_payload)
                    retry_response.raise_for_status()
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
    return {
        "startedAt": getattr(app.state, "runtime_started_at", ""),
        "sourceFresh": not stale,
        "sourceStale": stale,
        "latestSource": latest,
        "statusText": "运行代码是最新的" if not stale else "源码已更新，请重启 Gateway 让补丁生效",
    }


def _run_provider_smoke_test(
    app: FastAPI,
    client: httpx.Client,
    config: GatewayConfig,
    provider_id: str,
) -> dict[str, Any]:
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


async def _proxy_to_sidecar(client: httpx.Client, request: Request, root_url: str, prefix: str, path: str) -> Response:
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
    )
    return Response(sse_body, media_type="text/event-stream")


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


def _append_web_models(data: dict[str, Any]) -> dict[str, Any]:
    items = data.get("data") if isinstance(data.get("data"), list) else []
    seen = {item.get("id") for item in items if isinstance(item, dict)}
    for item in catalog_model_payloads():
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
    if bridge_result.error and bridge_result.error.repairable:
        _record_tool_bridge_event(
            app,
            "tool_bridge_retry",
            stage="repair",
            model=model,
            errorKind=bridge_result.error.kind,
            errorMessage=_preview_text(bridge_result.error.message, max_chars=360),
        )
        repair_data = retry_chat(build_repair_payload(payload, bridge_result))
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
    elif should_retry_incomplete_response(data):
        _record_tool_bridge_event(app, "tool_bridge_retry", stage="incomplete_response", model=model)
        retry_data = retry_chat(
            build_incomplete_response_retry_payload(payload, _openai_response_content(data), bridge=bridge)
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
                repair_data = retry_chat(build_repair_payload(payload, bridge_result))
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
    if bridge:
        _record_bridge_rejection_event(
            app,
            bridge_result,
            model=model,
            allowed_tools=allowed_tools,
            bridge_context=bridge_context,
        )
    return parsed, bridge_result


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
            "repeat_shell_housekeeping_without_progress",
            "repeat_same_skill_without_progress",
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
    return _openai_json_response(app, body=body, route="deepseek-web", model=model, bridge=bridge, parsed=parsed, bridge_result=bridge_result)


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
