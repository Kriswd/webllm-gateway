from __future__ import annotations

import inspect
import json
import secrets
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit, urlunsplit

import httpx
from fastapi import FastAPI, Header, HTTPException, Request
from fastapi.responses import FileResponse, JSONResponse, Response
from fastapi.staticfiles import StaticFiles
from starlette.concurrency import run_in_threadpool

from webai_gateway.config import GatewayConfig, config_to_admin, config_to_public, load_config, save_config, update_config
from webai_gateway.deepseek_web import DeepSeekWebClient, is_deepseek_web_model
from webai_gateway.openai_api import (
    bridge_error_headers,
    build_incomplete_response_retry_payload,
    build_native_web_search_retry_payload,
    build_repair_payload,
    build_unknown_tool_recovery_payload,
    build_openai_tool_calls_sse,
    build_tool_call_sse,
    build_upstream_payload,
    parse_chat_response,
    parse_sse_text,
    post_upstream,
    should_retry_incomplete_response,
    should_retry_native_web_search_response,
    upstream_headers,
)
from webai_gateway.anthropic_api import anthropic_body_to_openai, anthropic_count_tokens, anthropic_response_to_sse, openai_response_to_anthropic
from webai_gateway.qwen_web import QwenWebClient, is_qwen_web_model
from webai_gateway.qwen_coder import QwenCoderClient, is_qwen_coder_model
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
        return {"ok": True, "config": config_to_public(cfg)}

    @app.get("/api/admin/config")
    def admin_config(request: Request) -> dict[str, Any]:
        require_local_admin(request)
        return config_to_admin(current_config())

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
            provider_models = [model_id for model_id in provider.get("models", []) if model_id in model_ids]
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
                        "message": "网页登录态已保存，可以直接使用对应网页模型。",
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
        response = await run_in_threadpool(post_upstream, client, cfg, payload)
        response.raise_for_status()
        if bool(body.get("stream")):
            if bridge:
                content, _finish_reason = parse_sse_text(response.text)
                return Response(
                    build_tool_call_sse(content, allowed_tools=allowed_tools, model=cfg.upstream.model, bridge_context=bridge_context),
                    media_type="text/event-stream",
                )
            return Response(response.content, media_type=response.headers.get("content-type") or "text/event-stream")
        data = response.json()
        if not isinstance(data, dict):
            raise HTTPException(status_code=502, detail="Upstream response must be a JSON object")
        parsed, bridge_result = parse_chat_response(
            data,
            bridge=bridge,
            allowed_tools=allowed_tools,
            model=cfg.upstream.model,
            bridge_context=bridge_context,
            return_bridge_result=True,
        )
        if bridge_result.error and bridge_result.error.repairable:
            repair_response = await run_in_threadpool(post_upstream, client, cfg, build_repair_payload(payload, bridge_result))
            repair_response.raise_for_status()
            repair_data = repair_response.json()
            if isinstance(repair_data, dict):
                parsed, bridge_result = parse_chat_response(
                    repair_data,
                    bridge=bridge,
                    allowed_tools=allowed_tools,
                    model=cfg.upstream.model,
                    bridge_context=bridge_context,
                    return_bridge_result=True,
                )
        return JSONResponse(parsed, headers=bridge_error_headers(bridge_result))

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
            openai_body = anthropic_body_to_openai(body)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        bridge_result = None
        if is_deepseek_web_model(openai_body.get("model")):
            parsed = await run_in_threadpool(_deepseek_web_chat_payload, app, client, openai_body, cfg)
        elif is_qwen_web_model(openai_body.get("model")):
            parsed = await run_in_threadpool(_qwen_web_chat_payload, app, client, openai_body, cfg)
        else:
            payload, bridge, allowed_tools, bridge_context = build_upstream_payload(openai_body, cfg)
            response = await run_in_threadpool(post_upstream, client, cfg, payload)
            response.raise_for_status()
            data = response.json()
            if not isinstance(data, dict):
                raise HTTPException(status_code=502, detail="Upstream response must be a JSON object")
            parsed, bridge_result = parse_chat_response(
                data,
                bridge=bridge,
                allowed_tools=allowed_tools,
                model=str(openai_body.get("model") or cfg.upstream.model),
                bridge_context=bridge_context,
                return_bridge_result=True,
            )
            if bridge_result.error and bridge_result.error.repairable:
                repair_response = await run_in_threadpool(post_upstream, client, cfg, build_repair_payload(payload, bridge_result))
                repair_response.raise_for_status()
                repair_data = repair_response.json()
                if isinstance(repair_data, dict):
                    parsed, bridge_result = parse_chat_response(
                        repair_data,
                        bridge=bridge,
                        allowed_tools=allowed_tools,
                        model=str(openai_body.get("model") or cfg.upstream.model),
                        bridge_context=bridge_context,
                        return_bridge_result=True,
                    )
        anthropic_message = openai_response_to_anthropic(
            parsed,
            original_model=str(body.get("model") or openai_body.get("model") or cfg.upstream.model),
        )
        if bool(body.get("stream")):
            return Response(
                anthropic_response_to_sse(anthropic_message),
                media_type="text/event-stream",
                headers=bridge_error_headers(bridge_result),
            )
        return JSONResponse(anthropic_message, headers=bridge_error_headers(bridge_result))

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
    upstream_response = await run_in_threadpool(client.request, request.method, target, content=body, headers=headers)
    response_headers = {
        key: value
        for key, value in upstream_response.headers.items()
        if key.lower() not in {"content-length", "connection", "transfer-encoding"}
    }
    return Response(content=upstream_response.content, status_code=upstream_response.status_code, headers=response_headers)


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


def _parse_bridge_chat_data(
    data: dict[str, Any],
    *,
    payload: dict[str, Any],
    bridge: bool,
    allowed_tools: set[str],
    bridge_context: Any,
    model: str,
    retry_chat: Any,
    native_web_search: bool = False,
) -> tuple[dict[str, Any], Any]:
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
    if bridge_result.error and bridge_result.error.repairable:
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
        if bridge_result.error and bridge_result.error.kind == "unknown_tool":
            recovery_data = retry_chat(
                build_unknown_tool_recovery_payload(payload, bridge_result, allowed_tools=allowed_tools)
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
    elif should_retry_incomplete_response(data):
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
            if bridge_result.error and bridge_result.error.repairable:
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
    return parsed, bridge_result


def _deepseek_web_chat(app: FastAPI, client: httpx.Client, body: dict[str, Any], config: GatewayConfig) -> Response:
    credential = app.state.credential_store.get("deepseek-web")
    if not credential:
        raise HTTPException(status_code=401, detail="请先在控制台完成 DeepSeek 网页登录授权")
    payload, bridge, allowed_tools, bridge_context = _build_direct_payload(
        body,
        config,
        default_model="deepseek-web/deepseek-chat",
    )
    model = str(payload.get("model") or "deepseek-web/deepseek-chat")
    try:
        web_client = app.state.deepseek_client_factory(credential, http_client=client)
        data = web_client.chat_completions(payload)
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=502, detail=f"DeepSeek Web 调用失败：{exc}") from exc
    if not isinstance(data, dict):
        raise HTTPException(status_code=502, detail="DeepSeek Web 响应必须是 JSON 对象")
    parsed, bridge_result = _parse_bridge_chat_data(
        data,
        payload=payload,
        bridge=bridge,
        allowed_tools=allowed_tools,
        bridge_context=bridge_context,
        model=model,
        retry_chat=web_client.chat_completions,
        native_web_search=bool(payload.get("_webai_native_web_search")),
    )
    if bool(body.get("stream")):
        content = ""
        choices = parsed.get("choices") if isinstance(parsed.get("choices"), list) else []
        if choices and isinstance(choices[0], dict):
            msg = choices[0].get("message") if isinstance(choices[0].get("message"), dict) else {}
            if isinstance(msg.get("tool_calls"), list) and msg.get("tool_calls"):
                return Response(
                    build_openai_tool_calls_sse(msg["tool_calls"], model=model),
                    media_type="text/event-stream",
                )
            content = str(msg.get("content") or "")
        return Response(build_tool_call_sse(content, allowed_tools=allowed_tools, model=model, bridge_context=bridge_context), media_type="text/event-stream")
    return JSONResponse(parsed, headers=bridge_error_headers(bridge_result))


def _deepseek_web_chat_payload(app: FastAPI, client: httpx.Client, body: dict[str, Any], config: GatewayConfig) -> dict[str, Any]:
    response = _deepseek_web_chat(app, client, body, config)
    if isinstance(response, JSONResponse):
        return json_loads_response(response)
    raise HTTPException(status_code=400, detail="Anthropic 兼容接口暂不支持 direct provider 流式响应")


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
    try:
        web_client = None
        web_client = _build_qwen_web_client(app.state.qwen_client_factory, credential, client, config)
        data = web_client.chat_completions(payload)
    except HTTPException:
        raise
    except (TimeoutError, httpx.TimeoutException) as exc:
        diagnostic = _format_qwen_timeout_diagnostic(getattr(web_client, "last_diagnostic", None), getattr(exc, "diagnostic", None))
        raise HTTPException(
            status_code=504,
            detail=f"Qwen Web 响应超时：{exc}{diagnostic}",
            headers={"x-should-retry": "false"},
        ) from exc
    except Exception as exc:
        raise HTTPException(status_code=502, detail=f"Qwen Web 调用失败：{exc}") from exc
    if not isinstance(data, dict):
        raise HTTPException(status_code=502, detail="Qwen Web 响应必须是 JSON 对象")
    parsed, bridge_result = _parse_bridge_chat_data(
        data,
        payload=payload,
        bridge=bridge,
        allowed_tools=allowed_tools,
        bridge_context=bridge_context,
        model=model,
        retry_chat=web_client.chat_completions,
        native_web_search=bool(payload.get("_webai_native_web_search")),
    )
    if bool(body.get("stream")):
        content = ""
        choices = parsed.get("choices") if isinstance(parsed.get("choices"), list) else []
        if choices and isinstance(choices[0], dict):
            msg = choices[0].get("message") if isinstance(choices[0].get("message"), dict) else {}
            if isinstance(msg.get("tool_calls"), list) and msg.get("tool_calls"):
                return Response(
                    build_openai_tool_calls_sse(msg["tool_calls"], model=model),
                    media_type="text/event-stream",
                )
            content = str(msg.get("content") or "")
        return Response(build_tool_call_sse(content, allowed_tools=allowed_tools, model=model, bridge_context=bridge_context), media_type="text/event-stream")
    return JSONResponse(parsed, headers=bridge_error_headers(bridge_result))


def _qwen_web_chat_payload(app: FastAPI, client: httpx.Client, body: dict[str, Any], config: GatewayConfig) -> dict[str, Any]:
    response = _qwen_web_chat(app, client, body, config)
    if isinstance(response, JSONResponse):
        return json_loads_response(response)
    raise HTTPException(status_code=400, detail="Anthropic 兼容接口暂不支持 direct provider 流式响应")


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
    try:
        web_client = None
        web_client = _build_qwen_coder_client(app.state.qwen_coder_client_factory, credential, client, config)
        data = web_client.chat_completions(payload)
    except HTTPException:
        raise
    except (TimeoutError, httpx.TimeoutException) as exc:
        diagnostic = _format_qwen_coder_timeout_diagnostic(getattr(web_client, "last_diagnostic", None), getattr(exc, "diagnostic", None))
        raise HTTPException(
            status_code=504,
            detail=f"Qwen Coder Web 响应超时：{exc}{diagnostic}",
            headers={"x-should-retry": "false"},
        ) from exc
    except Exception as exc:
        raise HTTPException(status_code=502, detail=f"Qwen Coder Web 调用失败：{exc}") from exc
    if not isinstance(data, dict):
        raise HTTPException(status_code=502, detail="Qwen Coder Web 响应必须是 JSON 对象")
    parsed, bridge_result = _parse_bridge_chat_data(
        data,
        payload=payload,
        bridge=bridge,
        allowed_tools=allowed_tools,
        bridge_context=bridge_context,
        model=model,
        retry_chat=web_client.chat_completions,
        native_web_search=bool(payload.get("_webai_native_web_search")),
    )
    if bool(body.get("stream")):
        content = ""
        choices = parsed.get("choices") if isinstance(parsed.get("choices"), list) else []
        if choices and isinstance(choices[0], dict):
            msg = choices[0].get("message") if isinstance(choices[0].get("message"), dict) else {}
            if isinstance(msg.get("tool_calls"), list) and msg.get("tool_calls"):
                return Response(
                    build_openai_tool_calls_sse(msg["tool_calls"], model=model),
                    media_type="text/event-stream",
                )
            content = str(msg.get("content") or "")
        return Response(build_tool_call_sse(content, allowed_tools=allowed_tools, model=model, bridge_context=bridge_context), media_type="text/event-stream")
    return JSONResponse(parsed, headers=bridge_error_headers(bridge_result))


def _build_qwen_coder_client(factory: Any, credential: dict[str, Any], client: httpx.Client, config: GatewayConfig) -> Any:
    timeout = max(config.provider_runtime.request_timeout_seconds, 300)  # Coder 需要更长超时
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
    merged: dict[str, Any] = {}
    if isinstance(client_diagnostic, dict):
        merged.update(client_diagnostic)
    if isinstance(exception_diagnostic, dict):
        merged.update(exception_diagnostic)
    if not merged:
        return ""
    allowed_keys = (
        "prompt_chars",
        "prompt_max_chars",
        "prompt_compacted",
        "message_count",
        "stream_events",
        "json_events",
        "output_chars",
        "think_chars",
        "artifact_chars",
    )
    parts = [f"{key}={merged[key]}" for key in allowed_keys if key in merged]
    return f"；diagnostic: {', '.join(parts)}" if parts else ""


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


def _format_qwen_timeout_diagnostic(client_diagnostic: Any, exception_diagnostic: Any) -> str:
    merged: dict[str, Any] = {}
    if isinstance(client_diagnostic, dict):
        merged.update(client_diagnostic)
    if isinstance(exception_diagnostic, dict):
        merged.update(exception_diagnostic)
    if not merged:
        return ""
    allowed_keys = (
        "prompt_chars",
        "prompt_max_chars",
        "prompt_compacted",
        "message_count",
        "stream_events",
        "json_events",
        "output_chars",
        "think_chars",
    )
    parts = [f"{key}={merged[key]}" for key in allowed_keys if key in merged]
    return f"；diagnostic: {', '.join(parts)}" if parts else ""


def json_loads_response(response: JSONResponse) -> dict[str, Any]:
    data = json.loads(bytes(response.body).decode("utf-8"))
    if not isinstance(data, dict):
        raise HTTPException(status_code=502, detail="Gateway response must be a JSON object")
    return data


app = create_app()
