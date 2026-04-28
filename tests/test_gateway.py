from __future__ import annotations

import asyncio
import json
import sys
import types
from pathlib import Path
from typing import Any

import httpx
import pytest
from fastapi.testclient import TestClient

from webai_gateway.app import create_app
from webai_gateway.config import (
    GatewayConfig,
    ObservationPolicyConfig,
    ProviderRuntimeConfig,
    ServerConfig,
    ToolBridgeConfig,
    UpstreamConfig,
)
from webai_gateway.qwen_web import (
    QwenWebClient,
    _collect_qwen_stream_lines,
    normalize_qwen_model,
    parse_qwen_stream_text,
    qwen_messages_to_prompt_and_files,
)
from webai_gateway.qwen_coder import QwenCoderClient, normalize_qwen_coder_model
from webai_gateway.tool_bridge import build_context, parse_tool_response, compress_observation
from webai_gateway.web_auth import CredentialStore, DeepSeekWebAuthService, PROVIDERS, _read_qwen_credential, credential_summary


def _config() -> GatewayConfig:
    return GatewayConfig(
        server=ServerConfig(api_key="local-dev-key"),
        upstream=UpstreamConfig(base_url="http://upstream.test/v1", model="web-model"),
    )


def _openai_response(content: str) -> dict[str, Any]:
    return {
        "id": "upstream-chatcmpl",
        "object": "chat.completion",
        "choices": [{"index": 0, "finish_reason": "stop", "message": {"role": "assistant", "content": content}}],
    }


def _client(handler) -> TestClient:
    transport = httpx.MockTransport(handler)
    return TestClient(create_app(config=_config(), http_client=httpx.Client(transport=transport)))


def _headers() -> dict[str, str]:
    return {"Authorization": "Bearer local-dev-key"}


def _credential_store(tmp_path: Path) -> CredentialStore:
    store = CredentialStore(tmp_path / "credentials")
    store.save(
        "qwen",
        {
            "cookie": "qwen_session=session-secret",
            "bearer": "bearer-secret",
            "userAgent": "Chrome Test",
        },
    )
    return store


def _qwen_coder_credential_store(tmp_path: Path) -> CredentialStore:
    store = CredentialStore(tmp_path / "credentials")
    store.save(
        "qwen-coder",
        {
            "cookie": "qwen_session=session-secret",
            "bearer": "bearer-secret",
            "userAgent": "Chrome Test",
        },
    )
    return store


def _not_found_client() -> httpx.Client:
    return httpx.Client(transport=httpx.MockTransport(lambda request: httpx.Response(404, request=request)))


def test_gitignore_protects_local_credentials_and_runtime_state() -> None:
    gitignore = Path(".gitignore").read_text(encoding="utf-8")

    assert "```" not in gitignore
    for pattern in (
        "config.json",
        "credentials/",
        ".webai-gateway/",
        ".codex-logs/",
        ".learnings/",
        ".pytest_cache/",
        "webui/node_modules/",
    ):
        assert pattern in gitignore


def _sse_events(body: str) -> list[dict[str, Any]]:
    events: list[dict[str, Any]] = []
    for block in (body or "").split("\n\n"):
        event_name = ""
        data_lines: list[str] = []
        for line in block.splitlines():
            if line.startswith("event:"):
                event_name = line[len("event:") :].strip()
            elif line.startswith("data:"):
                data_lines.append(line[len("data:") :].strip())
        if not data_lines:
            continue
        raw = "\n".join(data_lines)
        item = json.loads(raw)
        item["_event"] = event_name
        events.append(item)
    return events


def test_models_returns_configured_model_when_upstream_models_unavailable() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(404, request=request)

    client = _client(handler)

    response = client.get("/v1/models", headers=_headers())

    assert response.status_code == 200
    body = response.json()
    assert body["object"] == "list"
    assert body["data"][0]["id"] == "web-model"
    model_ids = {item["id"] for item in body["data"]}
    assert "deepseek-web/deepseek-chat" in model_ids
    assert "gpt-instant" in model_ids
    assert "gemini-3-pro" in model_ids
    assert "qwen-web/qwen3.6-max-preview" in model_ids
    assert "qwen-web/qwen3.6-plus" in model_ids
    assert "qwen-web/qwen3-max" in model_ids
    assert "qwen-web/qwen3.6-max" not in model_ids
    qwen_preview = next(item for item in body["data"] if item["id"] == "qwen-web/qwen3.6-max-preview")
    assert qwen_preview["capabilities"]["tool_bridge"] is True
    assert qwen_preview["capabilities"]["supports_native_tools"] is False
    assert "seed" in model_ids
    assert "sora-2" in model_ids


def test_requires_bearer_api_key() -> None:
    client = _client(lambda request: httpx.Response(200, json={}, request=request))

    response = client.get("/v1/models")

    assert response.status_code == 401


def test_chat_rejects_invalid_json_with_400() -> None:
    client = _client(lambda request: httpx.Response(200, json={}, request=request))

    response = client.post(
        "/v1/chat/completions",
        headers={**_headers(), "Content-Type": "application/json"},
        content="{bad json",
    )

    assert response.status_code == 400
    assert "Request body must be valid JSON" in response.json()["detail"]


def test_forwards_normal_non_streaming_chat_without_tool_prompt() -> None:
    seen: dict[str, Any] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["path"] = request.url.path
        seen["body"] = json.loads(request.content.decode("utf-8"))
        return httpx.Response(200, json=_openai_response("plain reply"), request=request)

    client = _client(handler)

    response = client.post(
        "/v1/chat/completions",
        headers=_headers(),
        json={"model": "web-model", "messages": [{"role": "user", "content": "hello"}]},
    )

    assert response.status_code == 200
    assert response.json()["choices"][0]["message"]["content"] == "plain reply"
    assert seen["path"] == "/v1/chat/completions"
    assert "tools" not in seen["body"]
    assert seen["body"]["messages"] == [{"role": "user", "content": "hello"}]


def test_preserves_webai2api_catalog_model_when_forwarding() -> None:
    seen: dict[str, Any] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["body"] = json.loads(request.content.decode("utf-8"))
        return httpx.Response(200, json=_openai_response("webai2api reply"), request=request)

    client = _client(handler)

    response = client.post(
        "/v1/chat/completions",
        headers=_headers(),
        json={"model": "gpt-instant", "messages": [{"role": "user", "content": "hello"}]},
    )

    assert response.status_code == 200
    assert seen["body"]["model"] == "gpt-instant"


def test_injects_tools_and_returns_openai_tool_calls() -> None:
    seen: dict[str, Any] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["body"] = json.loads(request.content.decode("utf-8"))
        content = '```tool_json\n{"name":"read_file","args":{"path":"README.md"}}\n```'
        return httpx.Response(200, json=_openai_response(content), request=request)

    client = _client(handler)

    response = client.post(
        "/v1/chat/completions",
        headers=_headers(),
        json={
            "model": "web-model",
            "messages": [{"role": "user", "content": "read README"}],
            "tools": [
                {
                    "type": "function",
                    "function": {
                        "name": "read_file",
                        "description": "Read a local file",
                        "parameters": {"type": "object", "properties": {"path": {"type": "string"}}},
                    },
                }
            ],
        },
    )

    assert response.status_code == 200
    assert "tools" not in seen["body"]
    assert "tool_choice" not in seen["body"]
    system_messages = [m["content"] for m in seen["body"]["messages"] if m["role"] == "system"]
    assert system_messages
    assert "```tool_json" in system_messages[0]
    assert '"name": "read_file"' in system_messages[0]
    choice = response.json()["choices"][0]
    assert choice["finish_reason"] == "tool_calls"
    assert choice["message"]["content"] == ""
    tool_call = choice["message"]["tool_calls"][0]
    assert tool_call["type"] == "function"
    assert tool_call["function"]["name"] == "read_file"
    assert json.loads(tool_call["function"]["arguments"]) == {"path": "README.md"}


def test_strict_tool_bridge_accepts_calls_array_and_strips_content() -> None:
    seen: dict[str, Any] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["body"] = json.loads(request.content.decode("utf-8"))
        content = '```tool_json\n{"calls":[{"id":"call_read","name":"read_file","input":{"path":"README.md"}}]}\n```'
        return httpx.Response(200, json=_openai_response(content), request=request)

    client = _client(handler)

    response = client.post(
        "/v1/chat/completions",
        headers=_headers(),
        json={
            "model": "web-model",
            "messages": [{"role": "user", "content": "read README"}],
            "tools": [
                {
                    "type": "function",
                    "function": {
                        "name": "read_file",
                        "description": "Read a local file",
                        "parameters": {"type": "object", "properties": {"path": {"type": "string"}}},
                    },
                }
            ],
        },
    )

    assert response.status_code == 200
    assert "tools" not in seen["body"]
    assert "tool_choice" not in seen["body"]
    prompt = "\n".join(str(m["content"]) for m in seen["body"]["messages"] if m["role"] == "system")
    assert '"calls"' in prompt
    assert "no natural language outside it" in prompt
    choice = response.json()["choices"][0]
    assert choice["finish_reason"] == "tool_calls"
    assert choice["message"]["content"] == ""
    tool_call = choice["message"]["tool_calls"][0]
    assert tool_call["id"] == "call_read"
    assert tool_call["function"]["name"] == "read_file"
    assert json.loads(tool_call["function"]["arguments"]) == {"path": "README.md"}


def test_prompt_tool_bridge_hides_runtime_tools_for_any_client() -> None:
    seen: dict[str, Any] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["body"] = json.loads(request.content.decode("utf-8"))
        content = '```tool_json\n{"name":"fetch_url","args":{"url":"https://github.com/Einsia/OpenChronicle"}}\n```'
        return httpx.Response(200, json=_openai_response(content), request=request)

    client = _client(handler)

    response = client.post(
        "/v1/chat/completions",
        headers=_headers(),
        json={
            "model": "web-model",
            "messages": [{"role": "user", "content": "inspect a GitHub project"}],
            "tools": [
                {"type": "function", "function": {"name": "terminal", "parameters": {"type": "object"}}},
                {"type": "function", "function": {"name": "write_file", "parameters": {"type": "object"}}},
                {"type": "function", "function": {"name": "fetch_url", "parameters": {"type": "object"}}},
            ],
        },
    )

    assert response.status_code == 200
    prompt = "\n".join(str(message.get("content", "")) for message in seen["body"]["messages"] if message.get("role") == "system")
    assert '"name": "fetch_url"' in prompt
    assert '"name": "terminal"' not in prompt
    assert '"name": "write_file"' not in prompt
    choice = response.json()["choices"][0]
    assert choice["finish_reason"] == "tool_calls"
    assert choice["message"]["tool_calls"][0]["function"]["name"] == "fetch_url"


def test_strict_tool_prompt_guides_followup_and_machine_readable_urls() -> None:
    seen: dict[str, Any] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["body"] = json.loads(request.content.decode("utf-8"))
        return httpx.Response(200, json=_openai_response("plain reply"), request=request)

    client = _client(handler)

    response = client.post(
        "/v1/chat/completions",
        headers=_headers(),
        json={
            "model": "web-model",
            "messages": [{"role": "user", "content": "inspect a GitHub project"}],
            "tools": [
                {"type": "function", "function": {"name": "fetch_url", "parameters": {"type": "object"}}},
            ],
        },
    )

    assert response.status_code == 200
    prompt = "\n".join(str(message.get("content", "")) for message in seen["body"]["messages"] if message.get("role") == "system")
    assert "After a Tool result message" in prompt
    assert "machine-readable endpoints" in prompt
    assert "https://api.github.com/repos/<owner>/<repo>" in prompt
    assert "raw.githubusercontent.com" in prompt


def test_strict_tool_bridge_repairs_malformed_tool_json_once() -> None:
    requests: list[dict[str, Any]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content.decode("utf-8"))
        requests.append(body)
        if len(requests) == 1:
            return httpx.Response(200, json=_openai_response('```tool_json\n{"calls":[{"name":"read_file","input":\n```'), request=request)
        content = '```tool_json\n{"calls":[{"name":"read_file","input":{"path":"README.md"}}]}\n```'
        return httpx.Response(200, json=_openai_response(content), request=request)

    client = _client(handler)

    response = client.post(
        "/v1/chat/completions",
        headers=_headers(),
        json={
            "model": "web-model",
            "messages": [{"role": "user", "content": "read README"}],
            "tools": [{"type": "function", "function": {"name": "read_file", "parameters": {"type": "object"}}}],
        },
    )

    assert response.status_code == 200
    assert len(requests) == 2
    repair_text = "\n".join(str(m.get("content", "")) for m in requests[1]["messages"])
    assert "Previous tool JSON was invalid" in repair_text
    choice = response.json()["choices"][0]
    assert choice["finish_reason"] == "tool_calls"
    assert choice["message"]["tool_calls"][0]["function"]["name"] == "read_file"


def test_strict_tool_bridge_repairs_allowed_tool_denial_text() -> None:
    requests: list[dict[str, Any]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content.decode("utf-8"))
        requests.append(body)
        if len(requests) == 1:
            return httpx.Response(
                200,
                json=_openai_response(
                    "Tool Glob does not exists.Tool Bash does not exist.Tool Read does not exist."
                    "I cannot directly access the filesystem or execute tools in this environment."
                ),
                request=request,
            )
        return httpx.Response(
            200,
            json=_openai_response('```tool_json\n{"calls":[{"id":"toolu_read","name":"Read","input":{"file_path":"README.md"}}]}\n```'),
            request=request,
        )

    config = GatewayConfig(
        server=ServerConfig(api_key="local-dev-key"),
        upstream=UpstreamConfig(base_url="http://upstream.test/v1", model="web-model"),
        tool_bridge=ToolBridgeConfig(exposure_policy="all"),
    )
    client = TestClient(create_app(config=config, http_client=httpx.Client(transport=httpx.MockTransport(handler))))

    response = client.post(
        "/v1/messages?beta=true",
        headers={**_headers(), "anthropic-version": "2023-06-01"},
        json={
            "model": "web-model",
            "messages": [{"role": "user", "content": "/init"}],
            "tools": [
                {"name": "Glob", "input_schema": {"type": "object"}},
                {"name": "Bash", "input_schema": {"type": "object"}},
                {"name": "Read", "input_schema": {"type": "object"}},
            ],
            "max_tokens": 1024,
        },
    )

    assert response.status_code == 200
    assert len(requests) == 2
    repair_text = "\n".join(str(m.get("content", "")) for m in requests[1]["messages"])
    assert "The listed tools are available through the downstream client" in repair_text
    body = response.json()
    assert body["stop_reason"] == "tool_use"
    assert body["content"] == [{"type": "tool_use", "id": "toolu_read", "name": "Read", "input": {"file_path": "README.md"}}]


def test_strict_tool_bridge_repairs_chinese_permission_denial_text() -> None:
    requests: list[dict[str, Any]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content.decode("utf-8"))
        requests.append(body)
        if len(requests) == 1:
            return httpx.Response(
                200,
                json=_openai_response(
                    "很抱歉，我无法直接帮您更新项目。作为AI助手，我没有权限直接操作您的文件系统或执行Git命令。"
                    "系统明确限制了我不能使用任何文件系统操作工具（如Bash、Git等），这是出于安全考虑。"
                ),
                request=request,
            )
        return httpx.Response(
            200,
            json=_openai_response(
                '```tool_json\n{"calls":[{"id":"toolu_bash","name":"Bash","input":{"command":"git status --short"}}]}\n```'
            ),
            request=request,
        )

    config = GatewayConfig(
        server=ServerConfig(api_key="local-dev-key"),
        upstream=UpstreamConfig(base_url="http://upstream.test/v1", model="web-model"),
        tool_bridge=ToolBridgeConfig(exposure_policy="all"),
    )
    client = TestClient(create_app(config=config, http_client=httpx.Client(transport=httpx.MockTransport(handler))))

    response = client.post(
        "/v1/messages?beta=true",
        headers={**_headers(), "anthropic-version": "2023-06-01"},
        json={
            "model": "web-model",
            "messages": [{"role": "user", "content": "更新项目并提交"}],
            "tools": [
                {"name": "Bash", "description": "Run shell commands", "input_schema": {"type": "object"}},
                {"name": "Read", "description": "Read files", "input_schema": {"type": "object"}},
            ],
            "max_tokens": 1024,
        },
    )

    assert response.status_code == 200
    assert len(requests) == 2
    repair_text = "\n".join(str(m.get("content", "")) for m in requests[1]["messages"])
    assert "The listed tools are available through the downstream client" in repair_text
    body = response.json()
    assert body["stop_reason"] == "tool_use"
    assert body["content"] == [
        {"type": "tool_use", "id": "toolu_bash", "name": "Bash", "input": {"command": "git status --short"}}
    ]


def test_strict_tool_bridge_reports_repair_failure_without_tool_call() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=_openai_response('```tool_json\n{"calls":[{"name":"missing_tool","input":{}}]}\n```'), request=request)

    client = _client(handler)

    response = client.post(
        "/v1/chat/completions",
        headers=_headers(),
        json={
            "model": "web-model",
            "messages": [{"role": "user", "content": "use a tool"}],
            "tools": [{"type": "function", "function": {"name": "read_file", "parameters": {"type": "object"}}}],
        },
    )

    assert response.status_code == 200
    assert response.headers["x-webai-tool-bridge-error"] == "unknown_tool"
    choice = response.json()["choices"][0]
    assert choice["finish_reason"] == "stop"
    assert "tool_calls" not in choice["message"]
    assert choice["message"]["webai_tool_bridge"]["error"] == "unknown_tool"


def test_converts_tool_messages_for_web_upstream() -> None:
    seen: dict[str, Any] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["body"] = json.loads(request.content.decode("utf-8"))
        return httpx.Response(200, json=_openai_response("continued"), request=request)

    client = _client(handler)

    response = client.post(
        "/v1/chat/completions",
        headers=_headers(),
        json={
            "model": "web-model",
            "messages": [
                {"role": "assistant", "content": "", "tool_calls": [{"id": "call_1", "type": "function", "function": {"name": "read_file", "arguments": "{\"path\":\"README.md\"}"}}]},
                {"role": "tool", "tool_call_id": "call_1", "name": "read_file", "content": "file text"},
            ],
            "tools": [{"type": "function", "function": {"name": "read_file", "parameters": {"type": "object"}}}],
        },
    )

    assert response.status_code == 200
    assert all(m["role"] != "tool" for m in seen["body"]["messages"])
    joined = "\n".join(str(m["content"]) for m in seen["body"]["messages"])
    assert "Tool result for read_file" in joined
    assert "call_1" in joined
    assert "file text" in joined


def test_converts_failed_tool_result_as_error_observation() -> None:
    seen: dict[str, Any] = {}
    failed_result = json.dumps(
        {
            "url": "https://raw.githubusercontent.com/Einsia/OpenChronicle/main/README.md",
            "error": "network_error",
            "message": "Request failed after 3 attempts",
        },
        ensure_ascii=False,
    )

    def handler(request: httpx.Request) -> httpx.Response:
        seen["body"] = json.loads(request.content.decode("utf-8"))
        return httpx.Response(200, json=_openai_response("continued"), request=request)

    client = _client(handler)

    response = client.post(
        "/v1/chat/completions",
        headers=_headers(),
        json={
            "model": "web-model",
            "messages": [
                {"role": "assistant", "content": "", "tool_calls": [{"id": "call_1", "type": "function", "function": {"name": "fetch_url", "arguments": "{\"url\":\"https://raw.githubusercontent.com/Einsia/OpenChronicle/main/README.md\"}"}}]},
                {"role": "tool", "tool_call_id": "call_1", "name": "fetch_url", "content": failed_result},
            ],
            "tools": [{"type": "function", "function": {"name": "fetch_url", "parameters": {"type": "object"}}}],
        },
    )

    assert response.status_code == 200
    joined = "\n".join(str(m["content"]) for m in seen["body"]["messages"])
    assert "Tool result for fetch_url" in joined
    assert "is_error: true" in joined
    assert "is_error: false" not in joined
    assert "Request failed after 3 attempts" in joined
    assert "different allowed tool or input" in joined


def test_converts_tool_result_with_observation_compression() -> None:
    seen: dict[str, Any] = {}
    long_result = "A" * 7000

    def handler(request: httpx.Request) -> httpx.Response:
        seen["body"] = json.loads(request.content.decode("utf-8"))
        return httpx.Response(200, json=_openai_response("continued"), request=request)

    client = _client(handler)

    response = client.post(
        "/v1/chat/completions",
        headers=_headers(),
        json={
            "model": "web-model",
            "messages": [
                {"role": "assistant", "content": "", "tool_calls": [{"id": "call_1", "type": "function", "function": {"name": "read_file", "arguments": "{\"path\":\"README.md\"}"}}]},
                {"role": "tool", "tool_call_id": "call_1", "name": "read_file", "content": long_result},
            ],
            "tools": [{"type": "function", "function": {"name": "read_file", "parameters": {"type": "object"}}}],
        },
    )

    assert response.status_code == 200
    joined = "\n".join(str(m["content"]) for m in seen["body"]["messages"])
    assert "Tool result was too long and has been compressed" in joined
    assert "Original length" in joined
    assert len(joined) < 6500


def test_compress_observation_filters_dependency_path_lists() -> None:
    raw = "\n".join(
        [
            "webui\\node_modules\\.pnpm\\@vue+shared@3.5.25\\node_modules\\@vue\\shared\\README.md",
            "webui\\node_modules\\.pnpm\\nanoid@3.3.11\\node_modules\\nanoid\\README.md",
            "webui\\dist\\assets\\index.js",
            "README.md",
            "AGENTS.md",
            "requirements.txt",
        ]
    )

    compressed = compress_observation(raw, ToolBridgeConfig(observation_max_chars=1000))

    assert "dependency/build paths were omitted" in compressed
    assert "node_modules" not in compressed
    assert ".pnpm" not in compressed
    assert "webui\\dist" not in compressed
    assert "README.md" in compressed
    assert "AGENTS.md" in compressed
    assert "requirements.txt" in compressed


def test_compress_observation_reports_when_only_dependency_paths_match() -> None:
    raw = "\n".join(
        [
            "webui\\node_modules\\.pnpm\\@vue+shared@3.5.25\\node_modules\\@vue\\shared\\README.md",
            "webui\\node_modules\\.pnpm\\nanoid@3.3.11\\node_modules\\nanoid\\README.md",
        ]
    )

    compressed = compress_observation(raw, ToolBridgeConfig(observation_max_chars=1000))

    assert "Only dependency/build/cache paths were returned" in compressed
    assert "request a narrower pattern" in compressed
    assert "node_modules" not in compressed


def test_compress_observation_uses_configured_excluded_path_parts() -> None:
    raw = "\n".join(
        [
            "third_party\\sdk\\README.md",
            "vendor_cache\\generated\\notes.md",
            "README.md",
            "docs\\usage.md",
        ]
    )
    config = ToolBridgeConfig(
        observation_max_chars=1000,
        observation_policy=ObservationPolicyConfig(excluded_path_parts=("third_party", "vendor_cache")),
    )

    compressed = compress_observation(raw, config)

    assert "dependency/build paths were omitted" in compressed
    assert "third_party" not in compressed
    assert "vendor_cache" not in compressed
    assert "README.md" in compressed
    assert "docs\\usage.md" in compressed


def test_compress_observation_uses_configured_excluded_path_globs() -> None:
    raw = "\n".join(
        [
            "src\\generated\\client.md",
            "packages\\app\\src\\generated\\schema.md",
            "src\\main.py",
            "tests\\test_gateway.py",
        ]
    )
    config = ToolBridgeConfig(
        observation_max_chars=1000,
        observation_policy=ObservationPolicyConfig(excluded_path_globs=("**/generated/**",)),
    )

    compressed = compress_observation(raw, config)

    assert "dependency/build paths were omitted" in compressed
    assert "generated" not in compressed
    assert "src\\main.py" in compressed
    assert "tests\\test_gateway.py" in compressed


def test_tool_bridge_rewrites_repository_wide_glob_to_shallow_pattern() -> None:
    context = build_context(
        [
            {
                "type": "function",
                "function": {
                    "name": "Glob",
                    "description": "Find files by pattern.",
                    "parameters": {"type": "object", "properties": {"pattern": {"type": "string"}}},
                },
            }
        ],
        ToolBridgeConfig(exposure_policy="all"),
    )

    result = parse_tool_response(
        '```tool_json\n{"calls":[{"id":"call_1","name":"Glob","input":{"pattern":"**/*.md"}}]}\n```',
        context,
    )

    assert result.error is None
    assert [(call.id, call.name, call.input) for call in result.tool_calls] == [
        ("call_1", "Glob", {"pattern": "*.md"})
    ]


def test_tool_bridge_rewrites_repository_wide_all_files_glob_to_directory_list() -> None:
    context = build_context(
        [
            {
                "type": "function",
                "function": {
                    "name": "Glob",
                    "description": "Find files by pattern.",
                    "parameters": {"type": "object", "properties": {"pattern": {"type": "string"}}},
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "LS",
                    "description": "List a directory.",
                    "parameters": {"type": "object", "properties": {"path": {"type": "string"}}},
                },
            },
        ],
        ToolBridgeConfig(exposure_policy="all"),
    )

    result = parse_tool_response(
        '```tool_json\n{"calls":[{"id":"call_1","name":"Glob","input":{"pattern":"**/*"}}]}\n```',
        context,
    )

    assert result.error is None
    assert [(call.id, call.name, call.input) for call in result.tool_calls] == [
        ("call_1", "LS", {"path": "."})
    ]


def test_direct_provider_sanitizes_expensive_glob_without_retry(tmp_path: Path) -> None:
    seen_payloads: list[dict[str, Any]] = []

    class BroadGlobClient:
        def __init__(self, credential: dict[str, Any], http_client: httpx.Client | None = None) -> None:
            self.credential = credential

        def chat_completions(self, payload: dict[str, Any]) -> dict[str, Any]:
            seen_payloads.append(payload)
            return _openai_response(
                '```tool_json\n{"calls":[{"id":"call_1","name":"Glob","input":{"pattern":"**/*.md"}}]}\n```'
            )

    config = GatewayConfig(
        server=ServerConfig(api_key="local-dev-key"),
        upstream=UpstreamConfig(base_url="http://upstream.test/v1", model="web-model"),
        tool_bridge=ToolBridgeConfig(activation_policy="auto", exposure_policy="all"),
    )
    client = TestClient(
        create_app(
            config=config,
            credential_store=_credential_store(tmp_path),
            qwen_client_factory=BroadGlobClient,
            http_client=_not_found_client(),
        )
    )

    response = client.post(
        "/v1/messages",
        headers={**_headers(), "anthropic-version": "2023-06-01"},
        json={
            "model": "qwen-web/qwen3.6-plus",
            "messages": [{"role": "user", "content": "/init"}],
            "tools": [
                {"name": "Glob", "description": "Find files", "input_schema": {"type": "object"}},
                {"name": "Read", "description": "Read files", "input_schema": {"type": "object"}},
            ],
            "max_tokens": 1024,
        },
    )

    assert response.status_code == 200
    assert len(seen_payloads) == 1
    assert response.json()["content"] == [
        {"type": "tool_use", "id": "toolu_call_1", "name": "Glob", "input": {"pattern": "*.md"}}
    ]


def test_anthropic_messages_returns_tool_use_block() -> None:
    seen: dict[str, Any] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["body"] = json.loads(request.content.decode("utf-8"))
        content = '```tool_json\n{"calls":[{"id":"toolu_read","name":"read_file","input":{"path":"README.md"}}]}\n```'
        return httpx.Response(200, json=_openai_response(content), request=request)

    client = _client(handler)

    response = client.post(
        "/v1/messages",
        headers={**_headers(), "anthropic-version": "2023-06-01"},
        json={
            "model": "web-model",
            "system": "You are Claude Code.",
            "messages": [{"role": "user", "content": "read README"}],
            "tools": [{"name": "read_file", "description": "Read a local file", "input_schema": {"type": "object"}}],
            "max_tokens": 1024,
        },
    )

    assert response.status_code == 200
    assert "tools" not in seen["body"]
    assert "tool_choice" not in seen["body"]
    body = response.json()
    assert body["type"] == "message"
    assert body["role"] == "assistant"
    assert body["stop_reason"] == "tool_use"
    assert body["content"] == [{"type": "tool_use", "id": "toolu_read", "name": "read_file", "input": {"path": "README.md"}}]


def test_anthropic_messages_converts_tool_result_to_web_observation() -> None:
    seen: dict[str, Any] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["body"] = json.loads(request.content.decode("utf-8"))
        return httpx.Response(200, json=_openai_response("done"), request=request)

    client = _client(handler)

    response = client.post(
        "/v1/messages",
        headers={**_headers(), "anthropic-version": "2023-06-01"},
        json={
            "model": "web-model",
            "messages": [
                {"role": "assistant", "content": [{"type": "tool_use", "id": "toolu_read", "name": "read_file", "input": {"path": "README.md"}}]},
                {"role": "user", "content": [{"type": "tool_result", "tool_use_id": "toolu_read", "content": "file text"}]},
            ],
            "tools": [{"name": "read_file", "input_schema": {"type": "object"}}],
            "max_tokens": 1024,
        },
    )

    assert response.status_code == 200
    joined = "\n".join(str(m["content"]) for m in seen["body"]["messages"])
    assert "Tool result for read_file" in joined
    assert "toolu_read" in joined
    assert "file text" in joined
    assert response.json()["content"] == [{"type": "text", "text": "done"}]


def test_anthropic_messages_preserves_tool_result_error_state() -> None:
    seen: dict[str, Any] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["body"] = json.loads(request.content.decode("utf-8"))
        return httpx.Response(200, json=_openai_response("done"), request=request)

    client = _client(handler)

    response = client.post(
        "/v1/messages",
        headers={**_headers(), "anthropic-version": "2023-06-01"},
        json={
            "model": "web-model",
            "messages": [
                {"role": "assistant", "content": [{"type": "tool_use", "id": "toolu_fetch", "name": "fetch_url", "input": {"url": "https://example.com"}}]},
                {"role": "user", "content": [{"type": "tool_result", "tool_use_id": "toolu_fetch", "is_error": True, "content": "No data returned"}]},
            ],
            "tools": [{"name": "fetch_url", "input_schema": {"type": "object"}}],
            "max_tokens": 1024,
        },
    )

    assert response.status_code == 200
    joined = "\n".join(str(m["content"]) for m in seen["body"]["messages"])
    assert "Tool result for fetch_url" in joined
    assert "is_error: true" in joined
    assert "different allowed tool or input" in joined


def test_anthropic_tool_result_summarizes_non_text_blocks_without_raw_payload() -> None:
    seen: dict[str, Any] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["body"] = json.loads(request.content.decode("utf-8"))
        return httpx.Response(200, json=_openai_response("done"), request=request)

    client = _client(handler)

    response = client.post(
        "/v1/messages",
        headers={**_headers(), "anthropic-version": "2023-06-01"},
        json={
            "model": "web-model",
            "messages": [
                {"role": "assistant", "content": [{"type": "tool_use", "id": "toolu_screen", "name": "screenshot", "input": {}}]},
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "tool_result",
                            "tool_use_id": "toolu_screen",
                            "content": [
                                {"type": "text", "text": "captured"},
                                {"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": "aGVsbG8="}},
                            ],
                        }
                    ],
                },
            ],
            "tools": [{"name": "screenshot", "input_schema": {"type": "object"}}],
            "max_tokens": 1024,
        },
    )

    assert response.status_code == 200
    joined = "\n".join(str(m["content"]) for m in seen["body"]["messages"])
    assert "captured" in joined
    assert "[image: media_type=image/png, source=base64, data_length=8]" in joined
    assert "aGVsbG8=" not in joined
    assert "{'type':" not in joined


def test_anthropic_messages_streams_text_events() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=_openai_response("plain reply"), request=request)

    client = _client(handler)

    with client.stream(
        "POST",
        "/v1/messages",
        headers={**_headers(), "anthropic-version": "2023-06-01"},
        json={
            "model": "web-model",
            "messages": [{"role": "user", "content": "hello"}],
            "max_tokens": 1024,
            "stream": True,
        },
    ) as response:
        body = response.read().decode("utf-8")

    assert response.status_code == 200
    assert "text/event-stream" in response.headers["content-type"]
    events = _sse_events(body)
    assert [event["_event"] for event in events] == [
        "message_start",
        "content_block_start",
        "content_block_delta",
        "content_block_stop",
        "message_delta",
        "message_stop",
    ]
    assert events[2]["delta"] == {"type": "text_delta", "text": "plain reply"}
    assert events[4]["delta"]["stop_reason"] == "end_turn"


def test_anthropic_messages_accepts_x_api_key_and_passes_common_fields() -> None:
    seen: dict[str, Any] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["body"] = json.loads(request.content.decode("utf-8"))
        return httpx.Response(200, json=_openai_response("ok"), request=request)

    client = _client(handler)

    response = client.post(
        "/v1/messages",
        headers={"x-api-key": "local-dev-key", "anthropic-version": "2023-06-01"},
        json={
            "model": "web-model",
            "system": [{"type": "text", "text": "You are Claude Code.", "cache_control": {"type": "ephemeral"}}],
            "messages": [{"role": "user", "content": [{"type": "text", "text": "hello", "cache_control": {"type": "ephemeral"}}]}],
            "max_tokens": 128,
            "stop_sequences": ["</stop>"],
            "top_p": 0.9,
            "metadata": {"user_id": "local-user"},
            "thinking": {"type": "enabled", "budget_tokens": 1024},
        },
    )

    assert response.status_code == 200
    assert seen["body"]["stop"] == ["</stop>"]
    assert seen["body"]["top_p"] == 0.9
    assert seen["body"]["metadata"] == {"user_id": "local-user"}
    assert "reasoning" not in seen["body"]
    assert "cache_control" not in json.dumps(seen["body"], ensure_ascii=False)


def test_anthropic_count_tokens_returns_estimate_for_claude_code() -> None:
    client = _client(lambda request: httpx.Response(500, request=request))

    response = client.post(
        "/v1/messages/count_tokens",
        headers={"x-api-key": "local-dev-key", "anthropic-version": "2023-06-01"},
        json={
            "model": "qwen-web/qwen3.6-plus",
            "system": "You are Claude Code.",
            "messages": [
                {"role": "user", "content": "hello"},
                {
                    "role": "assistant",
                    "content": [{"type": "tool_use", "id": "toolu_1", "name": "Read", "input": {"file_path": "README.md"}}],
                },
                {"role": "user", "content": [{"type": "tool_result", "tool_use_id": "toolu_1", "content": "file text"}]},
            ],
            "tools": [{"name": "Read", "input_schema": {"type": "object"}}],
        },
    )

    assert response.status_code == 200
    body = response.json()
    assert body["input_tokens"] > 0


def test_tool_bridge_full_exposure_allows_runtime_and_mcp_tools() -> None:
    seen: dict[str, Any] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["body"] = json.loads(request.content.decode("utf-8"))
        return httpx.Response(
            200,
            json=_openai_response(
                '```tool_json\n{"calls":[{"id":"toolu_bash","name":"Bash","input":{"command":"pwd"}},{"id":"toolu_mcp","name":"mcp__repo__search","input":{"query":"gateway"}}]}\n```'
            ),
            request=request,
        )

    config = GatewayConfig(
        server=ServerConfig(api_key="local-dev-key"),
        upstream=UpstreamConfig(base_url="http://upstream.test/v1", model="web-model"),
        tool_bridge=ToolBridgeConfig(exposure_policy="all", max_calls_per_turn=4),
    )
    client = TestClient(create_app(config=config, http_client=httpx.Client(transport=httpx.MockTransport(handler))))

    response = client.post(
        "/v1/messages",
        headers={**_headers(), "anthropic-version": "2023-06-01"},
        json={
            "model": "web-model",
            "messages": [{"role": "user", "content": "run command and search mcp"}],
            "tools": [
                {"name": "Bash", "description": "Run shell commands", "input_schema": {"type": "object"}},
                {"name": "Edit", "description": "Edit files", "input_schema": {"type": "object"}},
                {"name": "mcp__repo__search", "description": "Search repository", "input_schema": {"type": "object"}},
            ],
            "max_tokens": 1024,
        },
    )

    assert response.status_code == 200
    prompt = "\n".join(str(message.get("content", "")) for message in seen["body"]["messages"] if message.get("role") == "system")
    assert '"name": "Bash"' in prompt
    assert '"name": "Edit"' in prompt
    assert '"name": "mcp__repo__search"' in prompt
    body = response.json()
    assert body["stop_reason"] == "tool_use"
    assert [block["name"] for block in body["content"]] == ["Bash", "mcp__repo__search"]


def test_tool_bridge_rewrites_cli_tool_name_to_bash_when_bash_allowed() -> None:
    context = build_context(
        [
            {
                "type": "function",
                "function": {
                    "name": "Bash",
                    "description": "Run shell commands",
                    "parameters": {
                        "type": "object",
                        "properties": {"command": {"type": "string"}},
                        "required": ["command"],
                    },
                },
            }
        ],
        ToolBridgeConfig(exposure_policy="all"),
    )

    result = parse_tool_response(
        '```tool_json\n{"calls":[{"id":"call_1","name":"gh","input":{"command":"repo view Kriswd/web-gateway"}}]}\n```',
        context,
    )

    assert result.error is None
    assert [(call.id, call.name, call.input) for call in result.tool_calls] == [
        ("call_1", "Bash", {"command": "gh repo view Kriswd/web-gateway"})
    ]


def test_tool_bridge_rewrites_terminal_tool_name_to_bash_without_prefixing_wrapper() -> None:
    context = build_context(
        [
            {
                "type": "function",
                "function": {
                    "name": "Bash",
                    "description": "Run shell commands",
                    "parameters": {
                        "type": "object",
                        "properties": {"command": {"type": "string"}},
                        "required": ["command"],
                    },
                },
            }
        ],
        ToolBridgeConfig(exposure_policy="all"),
    )

    result = parse_tool_response(
        '```tool_json\n{"calls":[{"id":"call_1","name":"terminal","input":{"command":"git status --short"}}]}\n```',
        context,
    )

    assert result.error is None
    assert [(call.id, call.name, call.input) for call in result.tool_calls] == [
        ("call_1", "Bash", {"command": "git status --short"})
    ]


def test_tool_bridge_does_not_rewrite_cli_name_without_bash() -> None:
    context = build_context(
        [
            {
                "type": "function",
                "function": {
                    "name": "Read",
                    "description": "Read files",
                    "parameters": {"type": "object"},
                },
            }
        ],
        ToolBridgeConfig(exposure_policy="all"),
    )

    result = parse_tool_response(
        '```tool_json\n{"calls":[{"id":"call_1","name":"gh","input":{"command":"repo view Kriswd/web-gateway"}}]}\n```',
        context,
    )

    assert result.tool_calls == []
    assert result.error is not None
    assert result.error.kind == "unknown_tool"


def test_tool_bridge_all_exposure_does_not_truncate_or_invent_toolsearch() -> None:
    seen: dict[str, Any] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["body"] = json.loads(request.content.decode("utf-8"))
        return httpx.Response(
            200,
            json=_openai_response('```tool_json\n{"calls":[{"id":"toolu_39","name":"Tool39","input":{"value":39}}]}\n```'),
            request=request,
        )

    config = GatewayConfig(
        server=ServerConfig(api_key="local-dev-key"),
        upstream=UpstreamConfig(base_url="http://upstream.test/v1", model="web-model"),
        tool_bridge=ToolBridgeConfig(exposure_policy="all", max_tools_in_prompt=32),
    )
    client = TestClient(create_app(config=config, http_client=httpx.Client(transport=httpx.MockTransport(handler))))
    tools = [{"name": f"Tool{i}", "description": f"Tool {i}", "input_schema": {"type": "object"}} for i in range(40)]

    response = client.post(
        "/v1/messages",
        headers={**_headers(), "anthropic-version": "2023-06-01"},
        json={"model": "web-model", "messages": [{"role": "user", "content": "use the last tool"}], "tools": tools, "max_tokens": 1024},
    )

    assert response.status_code == 200
    prompt = "\n".join(str(message.get("content", "")) for message in seen["body"]["messages"] if message.get("role") == "system")
    assert '"name": "Tool39"' in prompt
    assert "ToolSearch" not in prompt
    assert response.json()["content"] == [{"type": "tool_use", "id": "toolu_39", "name": "Tool39", "input": {"value": 39}}]


def test_tool_bridge_all_exposure_compacts_large_tool_prompt_without_hiding_tools() -> None:
    seen: dict[str, Any] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["body"] = json.loads(request.content.decode("utf-8"))
        return httpx.Response(
            200,
            json=_openai_response('```tool_json\n{"calls":[{"id":"toolu_79","name":"Tool79","input":{"value":79}}]}\n```'),
            request=request,
        )

    config = GatewayConfig(
        server=ServerConfig(api_key="local-dev-key"),
        upstream=UpstreamConfig(base_url="http://upstream.test/v1", model="web-model"),
        tool_bridge=ToolBridgeConfig(exposure_policy="all", tool_prompt_max_chars=6000),
    )
    client = TestClient(create_app(config=config, http_client=httpx.Client(transport=httpx.MockTransport(handler))))
    tools = [
        {
            "name": f"Tool{i}",
            "description": "long description " * 120,
            "input_schema": {"type": "object", "properties": {f"field_{j}": {"type": "string", "description": "x" * 80} for j in range(8)}},
        }
        for i in range(80)
    ]

    response = client.post(
        "/v1/messages",
        headers={**_headers(), "anthropic-version": "2023-06-01"},
        json={"model": "web-model", "messages": [{"role": "user", "content": "use the last tool"}], "tools": tools, "max_tokens": 1024},
    )

    assert response.status_code == 200
    prompt = "\n".join(str(message.get("content", "")) for message in seen["body"]["messages"] if message.get("role") == "system")
    assert len(prompt) <= 7000
    assert "Tool prompt manifest was compacted" in prompt
    assert "Tool79" in prompt
    assert "ToolSearch" not in prompt
    assert response.json()["content"] == [{"type": "tool_use", "id": "toolu_79", "name": "Tool79", "input": {"value": 79}}]


def test_anthropic_tool_bridge_parses_claude_code_tool_summary_text() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        content = (
            "Searched for 15 patterns (ctrl+o to expand)\n\n"
            "Assistant requested tool calls:\n"
            '- Read({"file_path":"E:\\\\ProjectX\\\\webai-gateway\\\\README.md"})\n'
            '- Read({"file_path":"E:\\\\ProjectX\\\\webai-gateway\\\\requirements.txt"})'
        )
        return httpx.Response(200, json=_openai_response(content), request=request)

    config = GatewayConfig(
        server=ServerConfig(api_key="local-dev-key"),
        upstream=UpstreamConfig(base_url="http://upstream.test/v1", model="web-model"),
        tool_bridge=ToolBridgeConfig(exposure_policy="all", max_readonly_calls_per_turn=8),
    )
    client = TestClient(create_app(config=config, http_client=httpx.Client(transport=httpx.MockTransport(handler))))

    response = client.post(
        "/v1/messages",
        headers={**_headers(), "anthropic-version": "2023-06-01"},
        json={
            "model": "web-model",
            "messages": [{"role": "user", "content": "/init"}],
            "tools": [{"name": "Read", "description": "Read files", "input_schema": {"type": "object"}}],
            "max_tokens": 1024,
        },
    )

    assert response.status_code == 200
    body = response.json()
    assert body["stop_reason"] == "tool_use"
    assert [block["name"] for block in body["content"]] == ["Read", "Read"]
    assert body["content"][0]["id"].startswith("toolu_")
    assert body["content"][0]["input"] == {"file_path": "E:\\ProjectX\\webai-gateway\\README.md"}
    assert body["content"][1]["input"] == {"file_path": "E:\\ProjectX\\webai-gateway\\requirements.txt"}


def test_anthropic_tool_bridge_normalizes_provider_search_markup_to_allowed_search_tool() -> None:
    seen: dict[str, Any] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["body"] = json.loads(request.content.decode("utf-8"))
        content = (
            "<search>\n"
            "<query>智谱GLM-5.1网页版体验地址 2026</query>\n"
            "<current_date>2026-04-17</current_date>\n"
            "</search>"
        )
        return httpx.Response(200, json=_openai_response(content), request=request)

    config = GatewayConfig(
        server=ServerConfig(api_key="local-dev-key"),
        upstream=UpstreamConfig(base_url="http://upstream.test/v1", model="web-model"),
        tool_bridge=ToolBridgeConfig(exposure_policy="all"),
    )
    client = TestClient(create_app(config=config, http_client=httpx.Client(transport=httpx.MockTransport(handler))))

    response = client.post(
        "/v1/messages",
        headers={**_headers(), "anthropic-version": "2023-06-01"},
        json={
            "model": "web-model",
            "messages": [{"role": "user", "content": "联网搜索下，5.1都发布了"}],
            "tools": [
                {
                    "name": "WebSearch",
                    "description": "Search the web",
                    "input_schema": {"type": "object", "properties": {"query": {"type": "string"}}},
                }
            ],
            "max_tokens": 1024,
        },
    )

    assert response.status_code == 200
    prompt = "\n".join(str(message.get("content", "")) for message in seen["body"]["messages"] if message.get("role") == "system")
    assert "provider-native" in prompt
    assert "<search>" in prompt
    body = response.json()
    assert body["stop_reason"] == "tool_use"
    assert body["content"] == [
        {
            "type": "tool_use",
            "id": "toolu_search_1",
            "name": "WebSearch",
            "input": {"query": "智谱GLM-5.1网页版体验地址 2026"},
        }
    ]


def test_anthropic_tool_bridge_returns_text_when_provider_search_markup_has_no_search_tool() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        content = "<search><query>智谱GLM-5.1网页版体验地址 2026</query></search>"
        return httpx.Response(200, json=_openai_response(content), request=request)

    config = GatewayConfig(
        server=ServerConfig(api_key="local-dev-key"),
        upstream=UpstreamConfig(base_url="http://upstream.test/v1", model="web-model"),
        tool_bridge=ToolBridgeConfig(exposure_policy="all"),
    )
    client = TestClient(create_app(config=config, http_client=httpx.Client(transport=httpx.MockTransport(handler))))

    response = client.post(
        "/v1/messages",
        headers={**_headers(), "anthropic-version": "2023-06-01"},
        json={
            "model": "web-model",
            "messages": [{"role": "user", "content": "联网搜索下，5.1都发布了"}],
            "tools": [{"name": "Read", "description": "Read files", "input_schema": {"type": "object"}}],
            "max_tokens": 1024,
        },
    )

    assert response.status_code == 200
    body = response.json()
    assert body["stop_reason"] == "end_turn"
    assert body["content"][0]["type"] == "text"
    assert "模型请求联网搜索" in body["content"][0]["text"]
    assert "当前允许工具中没有可用的搜索工具" in body["content"][0]["text"]
    assert "智谱GLM-5.1网页版体验地址 2026" in body["content"][0]["text"]
    assert "<search>" not in body["content"][0]["text"]


def test_anthropic_tool_use_ids_are_toolu_prefixed_for_claude_code() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json=_openai_response('```tool_json\n{"calls":[{"id":"call_1","name":"Read","input":{"file_path":"README.md"}}]}\n```'),
            request=request,
        )

    config = GatewayConfig(
        server=ServerConfig(api_key="local-dev-key"),
        upstream=UpstreamConfig(base_url="http://upstream.test/v1", model="web-model"),
        tool_bridge=ToolBridgeConfig(exposure_policy="all"),
    )
    client = TestClient(create_app(config=config, http_client=httpx.Client(transport=httpx.MockTransport(handler))))

    response = client.post(
        "/v1/messages",
        headers={**_headers(), "anthropic-version": "2023-06-01"},
        json={
            "model": "web-model",
            "messages": [{"role": "user", "content": "read"}],
            "tools": [{"name": "Read", "input_schema": {"type": "object"}}],
            "max_tokens": 1024,
        },
    )

    assert response.status_code == 200
    body = response.json()
    assert body["content"][0]["type"] == "tool_use"
    assert body["content"][0]["id"].startswith("toolu_")
    assert body["content"][0]["id"] != "call_1"


def test_tool_bridge_safe_exposure_hides_claude_code_write_tools() -> None:
    seen: dict[str, Any] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["body"] = json.loads(request.content.decode("utf-8"))
        return httpx.Response(200, json=_openai_response("plain reply"), request=request)

    client = _client(handler)

    response = client.post(
        "/v1/messages",
        headers={**_headers(), "anthropic-version": "2023-06-01"},
        json={
            "model": "web-model",
            "messages": [{"role": "user", "content": "inspect files"}],
            "tools": [
                {"name": "Read", "description": "Read files", "input_schema": {"type": "object"}},
                {"name": "Glob", "description": "Find files", "input_schema": {"type": "object"}},
                {"name": "Edit", "description": "Edit files", "input_schema": {"type": "object"}},
                {"name": "Write", "description": "Write files", "input_schema": {"type": "object"}},
                {"name": "MultiEdit", "description": "Edit many files", "input_schema": {"type": "object"}},
                {"name": "NotebookEdit", "description": "Edit notebooks", "input_schema": {"type": "object"}},
                {"name": "TodoWrite", "description": "Write todos", "input_schema": {"type": "object"}},
                {"name": "mcp__repo__search", "description": "Search repository", "input_schema": {"type": "object"}},
            ],
            "max_tokens": 1024,
        },
    )

    assert response.status_code == 200
    prompt = "\n".join(str(message.get("content", "")) for message in seen["body"]["messages"] if message.get("role") == "system")
    assert '"name": "Read"' in prompt
    assert '"name": "Glob"' in prompt
    assert '"name": "mcp__repo__search"' in prompt
    for blocked_name in ("Edit", "Write", "MultiEdit", "NotebookEdit", "TodoWrite"):
        assert f'"name": "{blocked_name}"' not in prompt


def test_anthropic_messages_streams_tool_use_events_for_qwen_web(tmp_path: Path) -> None:
    store = CredentialStore(tmp_path / "credentials")
    store.save(
        "qwen",
        {
            "cookie": "qwen_session=session-secret",
            "bearer": "bearer-secret",
            "userAgent": "Chrome Test",
        },
    )
    client = TestClient(
        create_app(
            config=_config(),
            credential_store=store,
            qwen_client_factory=_FakeQwenToolClient,
            http_client=_not_found_client(),
        )
    )

    with client.stream(
        "POST",
        "/v1/messages?beta=true",
        headers={**_headers(), "anthropic-version": "2023-06-01"},
        json={
            "model": "qwen-web/qwen3.6-plus",
            "messages": [{"role": "user", "content": "list files"}],
            "tools": [{"name": "list_local_files", "input_schema": {"type": "object"}}],
            "max_tokens": 1024,
            "stream": True,
        },
    ) as response:
        body = response.read().decode("utf-8")

    assert response.status_code == 200
    events = _sse_events(body)
    assert events[1]["content_block"]["type"] == "tool_use"
    assert events[1]["content_block"]["id"]
    assert events[1]["content_block"]["name"] == "list_local_files"
    assert events[1]["content_block"]["input"] == {}
    assert events[2]["delta"] == {"type": "input_json_delta", "partial_json": '{"path":"LOCAL_WORKSPACE"}'}
    assert events[4]["delta"]["stop_reason"] == "tool_use"


def test_anthropic_messages_streams_batch_readonly_tool_events_for_qwen_web(tmp_path: Path) -> None:
    store = CredentialStore(tmp_path / "credentials")
    store.save(
        "qwen",
        {
            "cookie": "qwen_session=session-secret",
            "bearer": "bearer-secret",
            "userAgent": "Chrome Test",
        },
    )
    client = TestClient(
        create_app(
            config=_config(),
            credential_store=store,
            qwen_client_factory=_FakeQwenBatchToolClient,
            http_client=_not_found_client(),
        )
    )

    with client.stream(
        "POST",
        "/v1/messages",
        headers={**_headers(), "anthropic-version": "2023-06-01"},
        json={
            "model": "qwen-web/qwen3.6-plus",
            "messages": [{"role": "user", "content": "/init"}],
            "tools": [
                {
                    "name": "Glob",
                    "description": "Find files by pattern.",
                    "input_schema": {
                        "type": "object",
                        "properties": {"pattern": {"type": "string"}},
                        "required": ["pattern"],
                    },
                }
            ],
            "max_tokens": 1024,
            "stream": True,
        },
    ) as response:
        body = response.read().decode("utf-8")

    assert response.status_code == 200
    events = _sse_events(body)
    tool_starts = [
        event
        for event in events
        if event["_event"] == "content_block_start" and event["content_block"]["type"] == "tool_use"
    ]
    assert len(tool_starts) == 6
    assert {event["content_block"]["name"] for event in tool_starts} == {"Glob"}
    assert not any(
        event.get("delta", {}).get("type") == "text_delta" and "\"calls\"" in event["delta"].get("text", "")
        for event in events
    )
    assert events[-2]["delta"]["stop_reason"] == "tool_use"


def test_streaming_tool_json_becomes_openai_tool_call_chunk() -> None:
    content = '```tool_json\n{"name":"read_file","args":{"path":"README.md"}}\n```'
    event = json.dumps({"choices": [{"delta": {"content": content}, "finish_reason": "stop"}]})

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            headers={"content-type": "text/event-stream"},
            content=f"data: {event}\n\ndata: [DONE]\n\n".encode("utf-8"),
            request=request,
        )

    client = _client(handler)

    with client.stream(
        "POST",
        "/v1/chat/completions",
        headers=_headers(),
        json={
            "model": "web-model",
            "stream": True,
            "messages": [{"role": "user", "content": "read README"}],
            "tools": [{"type": "function", "function": {"name": "read_file", "parameters": {"type": "object"}}}],
        },
    ) as response:
        chunks = list(response.iter_text())

    assert response.status_code == 200
    joined = "".join(chunks)
    assert "tool_json" not in joined
    assert '"tool_calls"' in joined
    assert '"finish_reason":"tool_calls"' in joined.replace(" ", "")


def test_startup_bat_exists() -> None:
    root = Path(__file__).resolve().parents[1]
    assert (root / "start_webai_gateway.bat").exists()


def test_root_serves_webai2api_native_spa_when_available(tmp_path: Path) -> None:
    native_dist = tmp_path / "webui" / "dist"
    native_dist.mkdir(parents=True)
    (native_dist / "index.html").write_text(
        '<!doctype html><html lang="zh-CN"><head><title>WebAI2API</title></head><body><div id="app"></div></body></html>',
        encoding="utf-8",
    )
    client = TestClient(create_app(config=_config(), native_ui_dir=native_dist, http_client=_not_found_client()))

    response = client.get("/")

    assert response.status_code == 200
    assert "WebAI2API" in response.text
    assert "WebAI Gateway 控制台" not in response.text


def test_assets_serve_webai2api_native_files(tmp_path: Path) -> None:
    native_dist = tmp_path / "webui" / "dist"
    assets_dir = native_dist / "assets"
    assets_dir.mkdir(parents=True)
    (assets_dir / "index.js").write_text("console.log('native webai2api');", encoding="utf-8")
    client = TestClient(create_app(config=_config(), native_ui_dir=native_dist, http_client=_not_found_client()))

    response = client.get("/assets/index.js")

    assert response.status_code == 200
    assert "native webai2api" in response.text
    assert response.headers["content-type"].startswith("text/javascript") or response.headers["content-type"].startswith("application/javascript")


def test_admin_routes_proxy_to_webai2api_sidecar_root(tmp_path: Path) -> None:
    native_dist = tmp_path / "webui" / "dist"
    native_dist.mkdir(parents=True)
    (native_dist / "index.html").write_text("<html>native</html>", encoding="utf-8")
    seen: dict[str, Any] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["method"] = request.method
        seen["url"] = str(request.url)
        seen["authorization"] = request.headers.get("authorization")
        return httpx.Response(200, json={"status": "running"}, request=request)

    client = TestClient(
        create_app(
            config=_config(),
            native_ui_dir=native_dist,
            http_client=httpx.Client(transport=httpx.MockTransport(handler)),
        )
    )

    response = client.get("/admin/status?fresh=1", headers={"Authorization": "Bearer webai2api-token"})

    assert response.status_code == 200
    assert response.json() == {"status": "running"}
    assert seen["method"] == "GET"
    assert seen["url"] == "http://upstream.test/admin/status?fresh=1"
    assert seen["authorization"] == "Bearer webai2api-token"


def test_v1_routes_remain_gateway_owned_when_admin_proxy_exists(tmp_path: Path) -> None:
    native_dist = tmp_path / "webui" / "dist"
    native_dist.mkdir(parents=True)
    (native_dist / "index.html").write_text("<html>native</html>", encoding="utf-8")
    seen_paths: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen_paths.append(request.url.path)
        return httpx.Response(200, json={"object": "list", "data": [{"id": "sidecar-model", "object": "model"}]}, request=request)

    client = TestClient(
        create_app(
            config=_config(),
            native_ui_dir=native_dist,
            http_client=httpx.Client(transport=httpx.MockTransport(handler)),
        )
    )

    response = client.get("/v1/models", headers=_headers())

    assert response.status_code == 200
    assert seen_paths == ["/v1/models"]
    model_ids = {item["id"] for item in response.json()["data"]}
    assert "sidecar-model" in model_ids
    assert "gpt-instant" in model_ids


def test_onboarding_returns_gateway_providers_and_models(tmp_path: Path) -> None:
    store = CredentialStore(tmp_path / "credentials")
    store.save(
        "deepseek-web",
        {
            "cookie": "ds_session_id=session-secret",
            "bearer": "bearer-secret",
            "userAgent": "Chrome Test",
        },
    )

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={"object": "list", "data": [{"id": "sidecar-model", "object": "model"}]},
            request=request,
        )

    client = TestClient(
        create_app(
            config=_config(),
            credential_store=store,
            http_client=httpx.Client(transport=httpx.MockTransport(handler)),
        )
    )

    response = client.get("/api/admin/onboarding")

    assert response.status_code == 200
    body = response.json()
    assert body["gateway"]["baseUrl"] == "/v1"
    assert body["gateway"]["apiKey"] == "local-dev-key"
    assert body["gateway"]["defaultModel"] == "web-model"
    assert body["summary"]["providers"] >= 10
    assert body["summary"]["models"] >= 3
    assert body["summary"]["authorizedDirectProviders"] == 1
    assert body["summary"]["webAI2APIProviders"] >= 1
    assert {item["id"] for item in body["models"]} >= {
        "sidecar-model",
        "deepseek-web/deepseek-chat",
        "gpt-instant",
        "qwen-web/qwen3.6-plus",
    }
    deepseek = next(item for item in body["providers"] if item["id"] == "deepseek-web")
    assert deepseek["credential"]["authorized"] is True
    assert deepseek["modelCount"] >= 2
    assert "session-secret" not in response.text
    assert "bearer-secret" not in response.text


def test_vendored_webai2api_frontend_has_gateway_bridge_page() -> None:
    root = Path(__file__).resolve().parents[1]
    main_js = root / "webui" / "src" / "main.js"
    app_vue = root / "webui" / "src" / "App.vue"
    bridge_vue = root / "webui" / "src" / "components" / "gateway" / "KrisBridge.vue"

    assert main_js.exists()
    assert app_vue.exists()
    assert bridge_vue.exists()
    main_source = main_js.read_text(encoding="utf-8")
    app_source = app_vue.read_text(encoding="utf-8")
    bridge_source = bridge_vue.read_text(encoding="utf-8")
    assert "path: '/'" in main_source
    assert "KrisBridge.vue" in main_source.split("path: '/'", 1)[1].split("}", 1)[0]
    assert "/dashboard" in main_source
    assert "/gateway/kris-bridge" in main_source
    assert "网页登录向导" in app_source
    assert "高级管理" in app_source
    assert "publicRoutes" in app_source
    assert "管理登录" in app_source
    assert "/api/admin/onboarding" in bridge_source
    assert "/api/admin/web-auth/browser/start" in bridge_source
    assert "/admin/restart" in bridge_source
    assert "打开授权浏览器" in bridge_source
    assert "未授权" in bridge_source
    assert "可用模型" in bridge_source
    assert "http://127.0.0.1:8610/v1" in bridge_source
    assert "Claude Code" in bridge_source


def test_admin_root_serves_management_ui() -> None:
    client = _client(lambda request: httpx.Response(200, json={}, request=request))

    response = client.get("/")

    assert response.status_code == 200
    assert 'lang="zh-CN"' in response.text
    assert "<title>WebAI2API</title>" in response.text
    assert "/assets/index.js" in response.text
    assert "WebAI Gateway 控制台" not in response.text


def test_admin_static_script_uses_chinese_ui_messages() -> None:
    client = _client(lambda request: httpx.Response(200, json={}, request=request))

    response = client.get("/static/app.js")

    assert response.status_code == 200
    assert "配置已保存并生效" in response.text
    assert "令牌已复制" in response.text
    assert "请在弹出的浏览器里完成 DeepSeek 登录" in response.text
    assert "打开 WebAI2API 登录管理" in response.text
    assert "WebAI2API 原生界面管理" in response.text


def test_static_management_ui_exposes_observation_policy_controls() -> None:
    client = _client(lambda request: httpx.Response(200, json={}, request=request))

    index_response = client.get("/static/index.html")
    script_response = client.get("/static/app.js")

    assert index_response.status_code == 200
    assert script_response.status_code == 200
    assert "observationPolicyPathParts" in index_response.text
    assert "observationPolicyPathGlobs" in index_response.text
    assert "路径列表压缩" in index_response.text
    assert "excludedPathParts" in script_response.text
    assert "excludedPathGlobs" in script_response.text
    assert "pathListMaxItems" in script_response.text
    assert "providerRuntimeTimeoutSeconds" in index_response.text
    assert "providerRuntimePromptMaxChars" in index_response.text
    assert "providerRuntime" in script_response.text


def test_provider_runtime_timeout_round_trips_config(tmp_path: Path) -> None:
    config_path = tmp_path / "config.json"
    client = TestClient(create_app(config=_config(), config_path=config_path, http_client=_not_found_client()))

    response = client.put(
        "/api/admin/config",
        headers=_headers(),
        json={"providerRuntime": {"requestTimeoutSeconds": 240}},
    )

    assert response.status_code == 200
    body = response.json()
    assert body["providerRuntime"]["requestTimeoutSeconds"] == 240
    assert body["providerRuntime"]["promptMaxChars"] == 12000
    saved = json.loads(config_path.read_text(encoding="utf-8"))
    assert saved["providerRuntime"]["requestTimeoutSeconds"] == 240
    assert saved["providerRuntime"]["promptMaxChars"] == 12000
    health = client.get("/health", headers=_headers()).json()
    assert health["config"]["providerRuntime"]["requestTimeoutSeconds"] == 240
    assert health["config"]["providerRuntime"]["promptMaxChars"] == 12000


def test_admin_config_returns_manageable_local_settings(tmp_path: Path) -> None:
    config_path = tmp_path / "config.json"
    client = TestClient(create_app(config=_config(), config_path=config_path, http_client=_not_found_client()))

    response = client.get("/api/admin/config")

    assert response.status_code == 200
    body = response.json()
    assert body["server"]["apiKey"] == "local-dev-key"
    assert body["upstream"]["baseUrl"] == "http://upstream.test/v1"
    assert body["upstream"]["model"] == "web-model"


def test_admin_config_update_persists_and_reloads_auth_token(tmp_path: Path) -> None:
    config_path = tmp_path / "config.json"
    seen: dict[str, Any] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["path"] = request.url.path
        return httpx.Response(200, json={"object": "list", "data": [{"id": "new-model", "object": "model"}]}, request=request)

    client = TestClient(create_app(config=_config(), config_path=config_path, http_client=httpx.Client(transport=httpx.MockTransport(handler))))

    response = client.put(
        "/api/admin/config",
        json={
            "server": {"apiKey": "new-local-key"},
            "upstream": {
                "baseUrl": "http://changed-upstream.test/v1",
                "apiKey": "upstream-key",
                "model": "new-model",
                "toolMode": "prompt",
            },
        },
    )

    assert response.status_code == 200
    saved = json.loads(config_path.read_text(encoding="utf-8"))
    assert saved["server"]["apiKey"] == "new-local-key"
    assert saved["upstream"]["baseUrl"] == "http://changed-upstream.test/v1"
    assert client.get("/v1/models", headers=_headers()).status_code == 401
    assert client.get("/v1/models", headers={"Authorization": "Bearer new-local-key"}).status_code == 200
    assert seen["path"] == "/v1/models"


def test_tool_bridge_observation_policy_round_trips_config(tmp_path: Path) -> None:
    config_path = tmp_path / "config.json"
    client = TestClient(create_app(config=_config(), config_path=config_path, http_client=_not_found_client()))

    response = client.put(
        "/api/admin/config",
        headers=_headers(),
        json={
            "providerRuntime": {"nativeWebSearchPolicy": "force"},
            "tool_bridge": {
                "activationPolicy": "always",
                "observationPolicy": {
                    "excludedPathParts": ["third_party", "vendor_cache"],
                    "excludedPathGlobs": ["**/generated/**"],
                    "pathListMaxItems": 12,
                }
            }
        },
    )

    assert response.status_code == 200
    body = response.json()
    assert body["providerRuntime"]["nativeWebSearchPolicy"] == "force"
    assert body["tool_bridge"]["activationPolicy"] == "always"
    policy = body["tool_bridge"]["observationPolicy"]
    assert policy["excludedPathParts"] == ["third_party", "vendor_cache"]
    assert policy["excludedPathGlobs"] == ["**/generated/**"]
    assert policy["pathListMaxItems"] == 12

    saved = json.loads(config_path.read_text(encoding="utf-8"))
    assert saved["providerRuntime"]["nativeWebSearchPolicy"] == "force"
    assert saved["tool_bridge"]["activationPolicy"] == "always"
    assert saved["tool_bridge"]["observationPolicy"] == policy
    health = client.get("/health", headers=_headers()).json()
    assert health["config"]["providerRuntime"]["nativeWebSearchPolicy"] == "force"
    assert health["config"]["tool_bridge"]["activationPolicy"] == "always"
    assert health["config"]["tool_bridge"]["observationPolicy"] == policy


def test_admin_token_rotation_persists_and_updates_gateway_auth(tmp_path: Path) -> None:
    config_path = tmp_path / "config.json"
    client = TestClient(create_app(config=_config(), config_path=config_path, http_client=_not_found_client()))

    response = client.post("/api/admin/token/rotate")

    assert response.status_code == 200
    token = response.json()["server"]["apiKey"]
    assert token.startswith("wg_")
    assert len(token) > 20
    saved = json.loads(config_path.read_text(encoding="utf-8"))
    assert saved["server"]["apiKey"] == token
    assert client.get("/v1/models", headers=_headers()).status_code == 401
    assert client.get("/v1/models", headers={"Authorization": f"Bearer {token}"}).status_code == 200


def test_web_auth_providers_list_deepseek_without_secrets(tmp_path: Path) -> None:
    store = CredentialStore(tmp_path / "credentials")
    store.save(
        "deepseek-web",
        {
            "cookie": "ds_session_id=session-secret; HWSID=hws-secret",
            "bearer": "bearer-secret",
            "userAgent": "Chrome Test",
        },
    )
    client = TestClient(create_app(config=_config(), credential_store=store, http_client=_not_found_client()))

    response = client.get("/api/admin/web-auth/providers")

    assert response.status_code == 200
    body = response.json()
    deepseek = next(item for item in body["providers"] if item["id"] == "deepseek-web")
    assert deepseek["name"] == "DeepSeek Web"
    assert deepseek["status"] == "available"
    assert deepseek["credential"]["authorized"] is True
    assert deepseek["credential"]["fields"] == {"cookie": True, "bearer": True, "userAgent": True}
    assert "session-secret" not in response.text
    assert "bearer-secret" not in response.text


def test_web_auth_providers_cover_webai2api_supported_sites(tmp_path: Path) -> None:
    client = TestClient(
        create_app(
            config=_config(),
            credential_store=CredentialStore(tmp_path / "credentials"),
            http_client=_not_found_client(),
        )
    )

    response = client.get("/api/admin/web-auth/providers")

    assert response.status_code == 200
    providers = {item["id"]: item for item in response.json()["providers"]}
    assert {
        "lmarena",
        "gemini-biz",
        "nano-banana-free",
        "zai",
        "gemini",
        "zenmux",
        "chatgpt",
        "qwen",
        "qwen-cn",
        "deepseek-web",
        "sora",
        "google-flow",
        "doubao",
    }.issubset(providers)
    assert providers["lmarena"]["capabilities"] == {"text": True, "image": True, "video": False}
    assert providers["gemini-biz"]["capabilities"] == {"text": True, "image": True, "video": True}
    assert providers["nano-banana-free"]["capabilities"] == {"text": False, "image": True, "video": False}
    assert providers["sora"]["capabilities"] == {"text": False, "image": False, "video": True}
    assert providers["deepseek-web"]["route"] == "direct"
    assert providers["qwen"]["route"] == "direct"
    assert providers["qwen"]["toolBridge"] == "strict"
    assert providers["qwen"]["supportsNativeTools"] is False
    assert providers["qwen"]["preferredProtocol"] == "openai"
    assert providers["qwen-cn"]["route"] == "webai2api"
    assert providers["chatgpt"]["route"] == "webai2api"
    assert "chatgpt_text" in providers["chatgpt"]["adapters"]
    assert providers["qwen"]["loginUrl"] == "https://chat.qwen.ai/"
    assert providers["qwen-cn"]["loginUrl"] == "https://www.qianwen.com/"
    assert providers["qwen"]["capabilities"] == {"text": True, "image": False, "video": False}
    assert "qwen_web" in providers["qwen"]["adapters"]
    assert "qwen_cn_web" in providers["qwen-cn"]["adapters"]
    assert "qwen-web/qwen3.6-max-preview" in providers["qwen"]["models"]
    assert "qwen-web/qwen3.6-plus" in providers["qwen"]["models"]
    assert "qwen-web/qwen3-max" in providers["qwen"]["models"]
    assert "qwen-web/qwen3.6-max" not in providers["qwen"]["models"]
    assert "Qwen3.5-Plus" in providers["qwen-cn"]["models"]


def test_qwen_cookie_only_credential_is_not_authorized() -> None:
    summary = credential_summary(
        "qwen",
        {
            "provider": "qwen",
            "cookie": "visitor_id=visitor; device_id=device",
            "bearer": "",
            "userAgent": "Chrome Test",
            "metadata": {"sessionToken": ""},
            "updatedAt": "2026-04-26T00:00:00+00:00",
        },
    )

    assert summary["authorized"] is False
    assert summary["fields"] == {"cookie": True, "bearer": False, "userAgent": True}


def test_qwen_session_token_credential_is_authorized() -> None:
    summary = credential_summary(
        "qwen",
        {
            "provider": "qwen",
            "cookie": "visitor_id=visitor; qwen_session=session-token",
            "bearer": "",
            "userAgent": "Chrome Test",
            "metadata": {"sessionToken": "session-token"},
            "updatedAt": "2026-04-26T00:00:00+00:00",
        },
    )

    assert summary["authorized"] is True


def test_qwen_store_rejects_visitor_cookie_only_credential(tmp_path: Path) -> None:
    store = CredentialStore(tmp_path / "credentials")

    with pytest.raises(ValueError, match="Qwen.*登录态"):
        store.save(
            "qwen",
            {
                "cookie": "visitor_id=visitor; device_id=device",
                "bearer": "",
                "userAgent": "Chrome Test",
                "metadata": {"sessionToken": ""},
            },
        )

    assert store.get("qwen") is None


def test_onboarding_marks_existing_qwen_cookie_only_file_unauthorized(tmp_path: Path) -> None:
    credential_dir = tmp_path / "credentials"
    credential_dir.mkdir()
    (credential_dir / "qwen.json").write_text(
        json.dumps(
            {
                "provider": "qwen",
                "cookie": "visitor_id=visitor; device_id=device",
                "bearer": "",
                "userAgent": "Chrome Test",
                "metadata": {"sessionToken": ""},
                "updatedAt": "2026-04-26T00:00:00+00:00",
            }
        ),
        encoding="utf-8",
    )
    client = TestClient(
        create_app(
            config=_config(),
            credential_store=CredentialStore(credential_dir),
            http_client=_not_found_client(),
        )
    )

    response = client.get("/api/admin/onboarding")

    assert response.status_code == 200
    qwen = next(item for item in response.json()["providers"] if item["id"] == "qwen")
    assert qwen["credential"]["authorized"] is False
    assert qwen["credential"]["fields"]["cookie"] is True
    assert "visitor" not in response.text


class _FakeQwenContext:
    def __init__(self, cookies: list[dict[str, str]]) -> None:
        self._cookies = cookies

    async def cookies(self, urls):
        return self._cookies


class _FakeQwenPage:
    async def evaluate(self, script: str) -> str:
        return "Chrome Test"


def test_read_qwen_credential_ignores_visitor_cookies() -> None:
    credential = asyncio.run(
        _read_qwen_credential(
            _FakeQwenContext(
                [
                    {"name": "visitor_id", "value": "visitor"},
                    {"name": "device_id", "value": "device"},
                ]
            ),
            _FakeQwenPage(),
        )
    )

    assert credential is None


def test_read_qwen_credential_accepts_bearer_login_state() -> None:
    credential = asyncio.run(
        _read_qwen_credential(
            _FakeQwenContext([{"name": "visitor_id", "value": "visitor"}]),
            _FakeQwenPage(),
            bearer_seen="bearer-token",
        )
    )

    assert credential is not None
    assert credential["bearer"] == "bearer-token"
    assert credential["metadata"]["sessionToken"] == "bearer-token"


def test_read_qwen_coder_credential_uses_coder_origin() -> None:
    context = _FakeQwenContext([{"name": "qwen_session", "value": "coder-session"}])

    credential = asyncio.run(
        _read_qwen_credential(
            context,
            _FakeQwenPage(),
            origins=("https://coder.qwen.ai", "https://qwen.ai"),
        )
    )

    assert credential is not None
    assert credential["metadata"]["sessionToken"] == "coder-session"


def test_qwen_coder_auth_service_captures_browser_login(monkeypatch: pytest.MonkeyPatch) -> None:
    progress: list[str] = []
    seen: dict[str, Any] = {}

    class FakePage:
        def on(self, event: str, handler: Any) -> None:
            seen["event"] = event

        async def goto(self, url: str) -> None:
            seen["goto"] = url

        async def evaluate(self, script: str) -> str:
            return "Chrome Test"

    class FakeContext:
        def __init__(self) -> None:
            self.pages = [FakePage()]

        async def cookies(self, urls: Any) -> list[dict[str, str]]:
            seen["cookie_urls"] = urls
            return [{"name": "qwen_session", "value": "coder-session"}]

    class FakeBrowser:
        def __init__(self) -> None:
            self.contexts = [FakeContext()]

        async def new_context(self) -> FakeContext:
            return FakeContext()

        async def close(self) -> None:
            seen["closed"] = True

    class FakeChromium:
        async def connect_over_cdp(self, cdp_url: str) -> FakeBrowser:
            seen["cdp_url"] = cdp_url
            return FakeBrowser()

    class FakePlaywrightManager:
        async def __aenter__(self) -> Any:
            return types.SimpleNamespace(chromium=FakeChromium())

        async def __aexit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
            return None

    fake_async_api = types.SimpleNamespace(async_playwright=lambda: FakePlaywrightManager())
    monkeypatch.setitem(sys.modules, "playwright", types.SimpleNamespace(async_api=fake_async_api))
    monkeypatch.setitem(sys.modules, "playwright.async_api", fake_async_api)

    credential = asyncio.run(
        DeepSeekWebAuthService().capture(
            "qwen-coder",
            "http://127.0.0.1:9222",
            progress=progress.append,
            timeout_seconds=3,
        )
    )

    assert seen["goto"] == "https://coder.qwen.ai/"
    assert "https://coder.qwen.ai" in seen["cookie_urls"]
    assert credential["metadata"]["sessionToken"] == "coder-session"
    assert credential["userAgent"] == "Chrome Test"
    assert seen["closed"] is True
    assert any("Qwen Coder" in item for item in progress)


class _FakeAuthService:
    async def capture(self, provider_id: str, cdp_url: str, progress):
        progress("已连接授权浏览器，正在等待登录状态")
        return {
            "cookie": "ds_session_id=session-secret; HWSID=hws-secret",
            "bearer": "bearer-secret",
            "userAgent": "Chrome Test",
        }


class _FakeVisitorQwenAuthService:
    async def capture(self, provider_id: str, cdp_url: str, progress):
        progress("等待 Qwen 登录态")
        return {
            "cookie": "visitor_id=visitor; device_id=device",
            "bearer": "",
            "userAgent": "Chrome Test",
            "metadata": {"sessionToken": ""},
        }


def test_qwen_auth_job_does_not_succeed_with_visitor_cookies(tmp_path: Path) -> None:
    store = CredentialStore(tmp_path / "credentials")
    client = TestClient(
        create_app(
            config=_config(),
            credential_store=store,
            web_auth_service=_FakeVisitorQwenAuthService(),
            run_auth_jobs_inline=True,
            http_client=_not_found_client(),
        )
    )

    response = client.post("/api/admin/web-auth/jobs", json={"provider": "qwen", "cdpUrl": "http://127.0.0.1:9222"})

    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "failed"
    assert "Qwen" in body["message"]
    assert store.get("qwen") is None


class _FakeBrowserLauncher:
    def start(self, provider_id: str, cdp_url: str) -> dict[str, Any]:
        return {
            "provider": provider_id,
            "cdpUrl": cdp_url,
            "loginUrl": "https://chat.deepseek.com",
            "started": True,
            "message": "授权浏览器已启动",
        }


def test_web_auth_job_captures_and_persists_credentials(tmp_path: Path) -> None:
    store = CredentialStore(tmp_path / "credentials")
    client = TestClient(
        create_app(
            config=_config(),
            credential_store=store,
            web_auth_service=_FakeAuthService(),
            run_auth_jobs_inline=True,
            http_client=_not_found_client(),
        )
    )

    response = client.post("/api/admin/web-auth/jobs", json={"provider": "deepseek-web", "cdpUrl": "http://127.0.0.1:9222"})

    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "succeeded"
    assert body["credential"]["authorized"] is True
    assert "bearer-secret" not in response.text
    assert store.get("deepseek-web")["bearer"] == "bearer-secret"


def test_web_auth_browser_start_returns_beginner_friendly_action() -> None:
    client = TestClient(create_app(config=_config(), browser_launcher=_FakeBrowserLauncher(), http_client=_not_found_client()))

    response = client.post("/api/admin/web-auth/browser/start", json={"provider": "deepseek-web"})

    assert response.status_code == 200
    body = response.json()
    assert body["started"] is True
    assert body["loginUrl"] == "https://chat.deepseek.com"
    assert "授权浏览器已启动" in body["message"]


class _FakeDeepSeekClient:
    def __init__(self, credential: dict[str, Any], http_client: httpx.Client | None = None) -> None:
        self.credential = credential

    def chat_completions(self, payload: dict[str, Any]) -> dict[str, Any]:
        assert self.credential["bearer"] == "bearer-secret"
        return {
            "id": "deepseek-web-test",
            "object": "chat.completion",
            "model": payload["model"],
            "choices": [
                {
                    "index": 0,
                    "finish_reason": "stop",
                    "message": {"role": "assistant", "content": "来自网页登录模型"},
                }
            ],
        }


class _FakeQwenClient:
    def __init__(self, credential: dict[str, Any], http_client: httpx.Client | None = None) -> None:
        self.credential = credential

    def chat_completions(self, payload: dict[str, Any]) -> dict[str, Any]:
        assert self.credential["cookie"] == "qwen_session=session-secret"
        return {
            "id": "qwen-web-test",
            "object": "chat.completion",
            "model": payload["model"],
            "choices": [
                {
                    "index": 0,
                    "finish_reason": "stop",
                    "message": {"role": "assistant", "content": "来自 Qwen 网页模型"},
                }
            ],
        }


class _FakeQwenCoderClient:
    captured_payload: dict[str, Any] = {}

    def __init__(self, credential: dict[str, Any], http_client: httpx.Client | None = None) -> None:
        self.credential = credential

    def chat_completions(self, payload: dict[str, Any]) -> dict[str, Any]:
        _FakeQwenCoderClient.captured_payload = payload
        assert self.credential["cookie"] == "qwen_session=session-secret"
        return {
            "id": "qwen-coder-test",
            "object": "chat.completion",
            "model": payload["model"],
            "choices": [
                {
                    "index": 0,
                    "finish_reason": "stop",
                    "message": {"role": "assistant", "content": "来自 Qwen Coder 网页模型"},
                }
            ],
        }


class _FakeQwenToolClient:
    def __init__(self, credential: dict[str, Any], http_client: httpx.Client | None = None) -> None:
        self.credential = credential

    def chat_completions(self, payload: dict[str, Any]) -> dict[str, Any]:
        assert payload["model"] == "qwen-web/qwen3.6-plus"
        assert "tools" not in payload
        assert "tool_choice" not in payload
        return _openai_response('```tool_json\n{"name":"list_local_files","args":{"path":"LOCAL_WORKSPACE"}}\n```')


class _FakeQwenBatchToolClient:
    def __init__(self, credential: dict[str, Any], http_client: httpx.Client | None = None) -> None:
        self.credential = credential

    def chat_completions(self, payload: dict[str, Any]) -> dict[str, Any]:
        assert payload["model"] == "qwen-web/qwen3.6-plus"
        assert "tools" not in payload
        assert "tool_choice" not in payload
        calls = [
            {"id": "call_1", "name": "Glob", "input": {"pattern": "*.md"}},
            {"id": "call_2", "name": "Glob", "input": {"pattern": "pyproject.toml"}},
            {"id": "call_3", "name": "Glob", "input": {"pattern": "requirements*.txt"}},
            {"id": "call_4", "name": "Glob", "input": {"pattern": ".cursorrules"}},
            {"id": "call_5", "name": "Glob", "input": {"pattern": ".cursor/rules/**"}},
            {"id": "call_6", "name": "Glob", "input": {"pattern": ".github/copilot-instructions.md"}},
        ]
        return _openai_response(json.dumps({"calls": calls}, separators=(",", ":")))


class _FakeQwenMultimodalClient:
    captured_payload: dict[str, Any] = {}

    def __init__(self, credential: dict[str, Any], http_client: httpx.Client | None = None) -> None:
        self.credential = credential

    def chat_completions(self, payload: dict[str, Any]) -> dict[str, Any]:
        _FakeQwenMultimodalClient.captured_payload = payload
        return _openai_response("我看到了图片")


def test_deepseek_web_chat_uses_saved_credentials(tmp_path: Path) -> None:
    store = CredentialStore(tmp_path / "credentials")
    store.save(
        "deepseek-web",
        {
            "cookie": "ds_session_id=session-secret",
            "bearer": "bearer-secret",
            "userAgent": "Chrome Test",
        },
    )
    client = TestClient(
        create_app(
            config=_config(),
            credential_store=store,
            deepseek_client_factory=_FakeDeepSeekClient,
            http_client=_not_found_client(),
        )
    )

    response = client.post(
        "/v1/chat/completions",
        headers=_headers(),
        json={"model": "deepseek-web/deepseek-chat", "messages": [{"role": "user", "content": "你好"}]},
    )

    assert response.status_code == 200
    assert response.json()["choices"][0]["message"]["content"] == "来自网页登录模型"


def test_qwen_web_client_uses_qwen_chat_api_and_parses_stream() -> None:
    seen: dict[str, Any] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/api/v2/chats/new":
            return httpx.Response(200, json={"data": {"id": "chat-1"}}, request=request)
        if request.url.path == "/api/v2/chat/completions":
            seen["body"] = json.loads(request.content.decode("utf-8"))
            seen["query"] = dict(request.url.params)
            return httpx.Response(
                200,
                text='data: {"choices":[{"delta":{"content":"你好"}}]}\n\ndata: {"output":{"text":"，Qwen"}}\n\ndata: [DONE]\n',
                request=request,
            )
        return httpx.Response(404, request=request)

    client = QwenWebClient(
        {"cookie": "qwen_session=session-secret", "bearer": "bearer-secret", "userAgent": "Chrome Test"},
        http_client=httpx.Client(transport=httpx.MockTransport(handler)),
    )

    response = client.chat_completions(
        {"model": "qwen-web/qwen3.6-plus", "messages": [{"role": "user", "content": "你好"}]}
    )

    assert seen["query"]["chat_id"] == "chat-1"
    assert seen["body"]["model"] == "qwen3.6-plus"
    assert seen["body"]["messages"][0]["feature_config"]["thinking_enabled"] is False
    assert response["choices"][0]["message"]["content"] == "你好，Qwen"


def test_qwen_web_client_raises_on_qwen_success_false_json() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/api/v2/chats/new":
            return httpx.Response(200, json={"data": {"id": "chat-1"}}, request=request)
        if request.url.path == "/api/v2/chat/completions":
            return httpx.Response(
                200,
                json={"success": False, "data": {"code": "Not_Found", "details": "Model not found"}},
                request=request,
            )
        return httpx.Response(404, request=request)

    client = QwenWebClient(
        {"cookie": "qwen_session=session-secret", "bearer": "bearer-secret", "userAgent": "Chrome Test"},
        http_client=httpx.Client(transport=httpx.MockTransport(handler)),
    )

    with pytest.raises(RuntimeError, match="Not_Found.*Model not found"):
        client.chat_completions(
            {"model": "qwen-web/qwen3.6-max", "messages": [{"role": "user", "content": "hello"}]}
        )


def test_qwen_web_helpers_normalize_and_parse_common_stream_shapes() -> None:
    text = "\n".join(
        [
            'data: {"choices":[{"delta":{"content":"A"}}]}',
            'data: {"data":{"content":"B"}}',
            '{"answer":"C"}',
            "data: [DONE]",
        ]
    )

    assert normalize_qwen_model("qwen-web/qwen3.5-plus") == "qwen3.5-plus"
    assert parse_qwen_stream_text(text) == "ABC"


def test_qwen_messages_compacts_oversized_prompt_for_web_provider() -> None:
    huge_system = "system bootstrap\n" + ("skill listing entry\n" * 4000)
    tool_protocol = (
        "You are using WebAI Gateway's strict tool bridge.\n"
        "Available tools: Read, Glob, Write.\n"
        "Required tool-call format:\n```tool_json\n{\"calls\":[]}\n```"
    )
    prompt, files = qwen_messages_to_prompt_and_files(
        [
            {"role": "system", "content": huge_system + "\n\n" + tool_protocol},
            {"role": "user", "content": "Please analyze this codebase and create a CLAUDE.md file."},
        ],
        max_prompt_chars=1200,
    )

    assert files == []
    assert len(prompt) <= 1200
    assert "Prompt content was compacted" in prompt
    assert "WebAI Gateway's strict tool bridge" in prompt
    assert "Please analyze this codebase" in prompt
    assert prompt.count("skill listing entry") < 10


def test_qwen_web_parser_prefers_answer_phase_over_think_phase() -> None:
    text = "\n\n".join(
        [
            'data: {"choices":[{"delta":{"content":"thinking","phase":"think"}}]}',
            'data: {"choices":[{"delta":{"content":"OK","phase":"answer"}}]}',
            "data: [DONE]",
        ]
    )

    assert parse_qwen_stream_text(text) == "OK"


def test_qwen_web_stream_returns_complete_tool_json_before_done() -> None:
    consumed = 0
    tool_text = '```tool_json\n{"calls":[{"id":"call_1","name":"Read","input":{}}]}\n```'

    def lines():
        nonlocal consumed
        consumed += 1
        yield "data: " + json.dumps({"choices": [{"delta": {"content": tool_text}}]})
        consumed += 1
        raise AssertionError("stream should stop once a complete tool_json block is available")

    content = _collect_qwen_stream_lines(lines(), deadline_seconds=30)

    assert consumed == 1
    assert '"name":"Read"' in content


def test_qwen_web_stream_has_wall_clock_deadline_for_heartbeat_streams() -> None:
    ticks = iter([0.0, 31.0])

    with pytest.raises(TimeoutError, match="Qwen Web request exceeded 30s"):
        _collect_qwen_stream_lines(
            ['data: {"choices":[{"delta":{"content":"still thinking","phase":"think"}}]}'],
            deadline_seconds=30,
            monotonic=lambda: next(ticks),
        )


def test_qwen_web_chat_uses_saved_credentials(tmp_path: Path) -> None:
    store = CredentialStore(tmp_path / "credentials")
    store.save(
        "qwen",
        {
            "cookie": "qwen_session=session-secret",
            "bearer": "bearer-secret",
            "userAgent": "Chrome Test",
        },
    )
    client = TestClient(
        create_app(
            config=_config(),
            credential_store=store,
            qwen_client_factory=_FakeQwenClient,
            http_client=_not_found_client(),
        )
    )

    response = client.post(
        "/v1/chat/completions",
        headers=_headers(),
        json={"model": "qwen-web/qwen3.6-plus", "messages": [{"role": "user", "content": "你好"}]},
    )

    assert response.status_code == 200
    assert response.json()["choices"][0]["message"]["content"] == "来自 Qwen 网页模型"


def test_anthropic_messages_routes_qwen_coder_direct_provider(tmp_path: Path) -> None:
    client = TestClient(
        create_app(
            config=_config(),
            credential_store=_qwen_coder_credential_store(tmp_path),
            qwen_coder_client_factory=_FakeQwenCoderClient,
            http_client=_not_found_client(),
        )
    )

    response = client.post(
        "/v1/messages",
        headers={**_headers(), "anthropic-version": "2023-06-01"},
        json={
            "model": "qwen-coder/qwen-coder-plus",
            "messages": [{"role": "user", "content": "hello"}],
            "max_tokens": 1024,
        },
    )

    assert response.status_code == 200
    body = response.json()
    assert body["type"] == "message"
    assert body["content"][0]["type"] == "text"
    assert body["content"][0]["text"] == "来自 Qwen Coder 网页模型"
    assert _FakeQwenCoderClient.captured_payload["model"] == "qwen-coder/qwen-coder-plus"


def test_qwen_coder_provider_does_not_claim_gateway_mcp_support() -> None:
    provider = PROVIDERS["qwen-coder"]

    assert provider.supports_native_tools is False
    assert provider.capabilities.get("mcp") is not True


def test_qwen_coder_legacy_plus_alias_maps_to_web_model() -> None:
    assert normalize_qwen_coder_model("qwen-coder/qwen-coder-plus") == "qwen3-coder-plus"
    assert normalize_qwen_coder_model("qwen-coder/qwen3-coder-plus") == "qwen3-coder-plus"


def test_qwen_coder_client_sends_web_model_alias_for_legacy_plus() -> None:
    seen: dict[str, Any] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path.endswith("/api/v2/chats/new"):
            return httpx.Response(200, json={"data": {"id": "chat-test"}}, request=request)
        if request.url.path.endswith("/api/v2/chat/completions"):
            seen["body"] = json.loads(request.content.decode("utf-8"))
            return httpx.Response(
                200,
                content=b'data: {"choices":[{"delta":{"content":"ok","phase":"answer"}}]}\n\ndata: [DONE]\n\n',
                request=request,
                headers={"content-type": "text/event-stream"},
            )
        return httpx.Response(404, request=request)

    qwen = QwenCoderClient(
        {"cookie": "qwen_session=session-secret", "bearer": "bearer-secret", "userAgent": "Chrome Test"},
        http_client=httpx.Client(transport=httpx.MockTransport(handler)),
    )

    response = qwen.chat_completions(
        {
            "model": "qwen-coder/qwen-coder-plus",
            "messages": [{"role": "user", "content": "hello"}],
        }
    )

    assert response["choices"][0]["message"]["content"] == "ok"
    assert seen["body"]["model"] == "qwen3-coder-plus"
    assert seen["body"]["messages"][0]["models"] == ["qwen3-coder-plus"]


def test_qwen_coder_client_does_not_enable_mcp_by_default() -> None:
    seen: dict[str, Any] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path.endswith("/api/v2/chats/new"):
            return httpx.Response(200, json={"data": {"id": "chat-test"}}, request=request)
        if request.url.path.endswith("/api/v2/chat/completions"):
            seen["body"] = json.loads(request.content.decode("utf-8"))
            return httpx.Response(
                200,
                content=b'data: {"choices":[{"delta":{"content":"ok","phase":"answer"}}]}\n\ndata: [DONE]\n\n',
                request=request,
                headers={"content-type": "text/event-stream"},
            )
        return httpx.Response(404, request=request)

    qwen = QwenCoderClient(
        {"cookie": "qwen_session=session-secret", "bearer": "bearer-secret", "userAgent": "Chrome Test"},
        http_client=httpx.Client(transport=httpx.MockTransport(handler)),
    )

    response = qwen.chat_completions(
        {
            "model": "qwen-coder/qwen-coder-plus",
            "messages": [{"role": "user", "content": "hello"}],
        }
    )

    assert response["choices"][0]["message"]["content"] == "ok"
    feature_config = seen["body"]["messages"][0]["feature_config"]
    assert feature_config.get("mcp_enabled") is not True


def test_qwen_web_chat_passes_configured_provider_timeout(tmp_path: Path) -> None:
    seen: dict[str, Any] = {}

    class CapturingQwenClient:
        def __init__(
            self,
            credential: dict[str, Any],
            http_client: httpx.Client | None = None,
            request_timeout_seconds: float | None = None,
            prompt_max_chars: int | None = None,
        ) -> None:
            seen["timeout"] = request_timeout_seconds
            seen["prompt_max_chars"] = prompt_max_chars

        def chat_completions(self, payload: dict[str, Any]) -> dict[str, Any]:
            return _openai_response("来自 Qwen 网页模型")

    config = GatewayConfig(
        server=ServerConfig(api_key="local-dev-key"),
        upstream=UpstreamConfig(base_url="http://upstream.test/v1", model="web-model"),
        provider_runtime=ProviderRuntimeConfig(request_timeout_seconds=240, prompt_max_chars=30000),
    )
    client = TestClient(
        create_app(
            config=config,
            credential_store=_credential_store(tmp_path),
            qwen_client_factory=CapturingQwenClient,
            http_client=_not_found_client(),
        )
    )

    response = client.post(
        "/v1/chat/completions",
        headers=_headers(),
        json={"model": "qwen-web/qwen3.6-plus", "messages": [{"role": "user", "content": "你好"}]},
    )

    assert response.status_code == 200
    assert seen["timeout"] == 240
    assert seen["prompt_max_chars"] == 30000


def test_qwen_web_timeout_returns_gateway_timeout(tmp_path: Path) -> None:
    class TimeoutQwenClient:
        def __init__(self, credential: dict[str, Any], http_client: httpx.Client | None = None) -> None:
            self.credential = credential
            self.last_diagnostic = {
                "prompt_chars": 12000,
                "prompt_max_chars": 12000,
                "prompt_compacted": True,
                "stream_events": 3,
                "output_chars": 0,
                "think_chars": 512,
            }

        def chat_completions(self, payload: dict[str, Any]) -> dict[str, Any]:
            raise TimeoutError("Qwen Web request exceeded 35s")

    client = TestClient(
        create_app(
            config=_config(),
            credential_store=_credential_store(tmp_path),
            qwen_client_factory=TimeoutQwenClient,
            http_client=_not_found_client(),
        )
    )

    response = client.post(
        "/v1/messages",
        headers={**_headers(), "anthropic-version": "2023-06-01"},
        json={"model": "qwen-web/qwen3.6-plus", "messages": [{"role": "user", "content": "hello"}], "max_tokens": 1024},
    )

    assert response.status_code == 504
    assert response.headers["x-should-retry"] == "false"
    detail = response.json()["detail"]
    assert "Qwen Web 响应超时" in detail
    assert "prompt_chars=12000" in detail
    assert "prompt_compacted=True" in detail
    assert "stream_events=3" in detail
    assert "session-secret" not in detail
    assert "bearer-secret" not in detail


def test_qwen_stream_timeout_carries_sanitized_diagnostics() -> None:
    ticks = iter([0.0, 181.0])

    with pytest.raises(TimeoutError) as raised:
        _collect_qwen_stream_lines(
            ['data: {"choices":[{"delta":{"content":"thinking","phase":"think"}}]}'],
            deadline_seconds=180,
            monotonic=lambda: next(ticks),
        )

    diagnostic = getattr(raised.value, "diagnostic", {})
    assert diagnostic["stream_events"] == 1
    assert diagnostic["think_chars"] == len("thinking")
    assert diagnostic["output_chars"] == 0


def test_qwen_web_bridge_hides_runtime_tools_from_web_prompt(tmp_path: Path) -> None:
    seen: dict[str, Any] = {}

    class CapturingQwenClient:
        def __init__(self, credential: dict[str, Any], http_client: httpx.Client | None = None) -> None:
            self.credential = credential

        def chat_completions(self, payload: dict[str, Any]) -> dict[str, Any]:
            seen["payload"] = payload
            return _openai_response(
                '```tool_json\n{"name":"fetch_url","args":{"url":"https://github.com/Einsia/OpenChronicle"}}\n```'
            )

    store = CredentialStore(tmp_path / "credentials")
    store.save(
        "qwen",
        {
            "cookie": "qwen_session=session-secret",
            "bearer": "bearer-secret",
            "userAgent": "Chrome Test",
        },
    )
    config = GatewayConfig(
        server=ServerConfig(api_key="local-dev-key"),
        upstream=UpstreamConfig(base_url="http://upstream.test/v1", model="web-model"),
        tool_bridge=ToolBridgeConfig(activation_policy="always"),
    )
    client = TestClient(
        create_app(
            config=config,
            credential_store=store,
            qwen_client_factory=CapturingQwenClient,
            http_client=_not_found_client(),
        )
    )

    response = client.post(
        "/v1/chat/completions",
        headers=_headers(),
        json={
            "model": "qwen-web/qwen3.6-max-preview",
            "messages": [{"role": "user", "content": "What is OpenChronicle?"}],
            "tools": [
                {"type": "function", "function": {"name": "terminal", "parameters": {"type": "object"}}},
                {"type": "function", "function": {"name": "write_file", "parameters": {"type": "object"}}},
                {"type": "function", "function": {"name": "fetch_url", "parameters": {"type": "object"}}},
            ],
        },
    )

    assert response.status_code == 200
    payload = seen["payload"]
    assert "tools" not in payload
    prompt = "\n".join(str(message.get("content", "")) for message in payload["messages"] if message.get("role") == "system")
    assert '"name": "fetch_url"' in prompt
    assert '"name": "terminal"' not in prompt
    assert '"name": "write_file"' not in prompt
    choice = response.json()["choices"][0]
    assert choice["finish_reason"] == "tool_calls"
    assert choice["message"]["tool_calls"][0]["function"]["name"] == "fetch_url"


def test_qwen_web_auto_activation_routes_plain_web_search_to_native_provider(tmp_path: Path) -> None:
    seen: dict[str, Any] = {}

    class CapturingQwenClient:
        def __init__(self, credential: dict[str, Any], http_client: httpx.Client | None = None) -> None:
            self.credential = credential

        def chat_completions(self, payload: dict[str, Any]) -> dict[str, Any]:
            seen["payload"] = payload
            return _openai_response("已使用 Qwen 网页模型原生联网能力检索。")

    config = GatewayConfig(
        server=ServerConfig(api_key="local-dev-key"),
        upstream=UpstreamConfig(base_url="http://upstream.test/v1", model="web-model"),
        provider_runtime=ProviderRuntimeConfig(native_web_search_policy="auto"),
        tool_bridge=ToolBridgeConfig(activation_policy="auto", exposure_policy="all"),
    )
    client = TestClient(
        create_app(
            config=config,
            credential_store=_credential_store(tmp_path),
            qwen_client_factory=CapturingQwenClient,
            http_client=_not_found_client(),
        )
    )

    response = client.post(
        "/v1/messages",
        headers={**_headers(), "anthropic-version": "2023-06-01"},
        json={
            "model": "qwen-web/qwen3.6-plus",
            "messages": [{"role": "user", "content": "联网搜索下，GLM-5.1 都发布了，官网体验地址在哪？"}],
            "tools": [
                {"name": "Read", "description": "Read files", "input_schema": {"type": "object"}},
                {"name": "WebSearch", "description": "Search the web", "input_schema": {"type": "object", "properties": {"query": {"type": "string"}}}},
            ],
            "max_tokens": 1024,
        },
    )

    assert response.status_code == 200
    payload = seen["payload"]
    assert payload["_webai_native_web_search"] is True
    assert "tools" not in payload
    assert "tool_choice" not in payload
    prompt = "\n".join(str(message.get("content", "")) for message in payload["messages"])
    assert "WebAI Gateway's strict tool bridge" not in prompt
    assert "不要只回复搜索计划" in prompt
    body = response.json()
    assert body["stop_reason"] == "end_turn"
    assert body["content"] == [{"type": "text", "text": "已使用 Qwen 网页模型原生联网能力检索。"}]


def test_qwen_web_native_search_retries_placeholder_response_for_final_answer(tmp_path: Path) -> None:
    seen_payloads: list[dict[str, Any]] = []

    class PlaceholderQwenClient:
        def __init__(self, credential: dict[str, Any], http_client: httpx.Client | None = None) -> None:
            self.credential = credential

        def chat_completions(self, payload: dict[str, Any]) -> dict[str, Any]:
            seen_payloads.append(payload)
            if len(seen_payloads) == 1:
                return _openai_response("我来帮你搜索 GLM-5.1 的官方体验网页。")
            return _openai_response("https://chatglm.cn/\n依据：这是智谱清言官方网页入口。")

    config = GatewayConfig(
        server=ServerConfig(api_key="local-dev-key"),
        upstream=UpstreamConfig(base_url="http://upstream.test/v1", model="web-model"),
        provider_runtime=ProviderRuntimeConfig(native_web_search_policy="auto"),
        tool_bridge=ToolBridgeConfig(activation_policy="auto", exposure_policy="all"),
    )
    client = TestClient(
        create_app(
            config=config,
            credential_store=_credential_store(tmp_path),
            qwen_client_factory=PlaceholderQwenClient,
            http_client=_not_found_client(),
        )
    )

    response = client.post(
        "/v1/messages",
        headers={**_headers(), "anthropic-version": "2023-06-01"},
        json={
            "model": "qwen-web/qwen3.6-max-preview",
            "messages": [{"role": "user", "content": "联网搜索下 GLM5.1 官方体验网页是哪个"}],
            "tools": [{"name": "WebSearch", "description": "Search the web", "input_schema": {"type": "object"}}],
            "max_tokens": 1024,
        },
    )

    assert response.status_code == 200
    assert len(seen_payloads) == 2
    assert seen_payloads[0]["_webai_native_web_search"] is True
    retry_prompt = "\n".join(str(message.get("content", "")) for message in seen_payloads[1]["messages"])
    assert "上一轮只返回了搜索计划" in retry_prompt
    assert response.json()["content"] == [{"type": "text", "text": "https://chatglm.cn/\n依据：这是智谱清言官方网页入口。"}]


def test_qwen_web_continue_inherits_recent_native_search_context(tmp_path: Path) -> None:
    seen: dict[str, Any] = {}

    class ContinuationQwenClient:
        def __init__(self, credential: dict[str, Any], http_client: httpx.Client | None = None) -> None:
            self.credential = credential

        def chat_completions(self, payload: dict[str, Any]) -> dict[str, Any]:
            seen["payload"] = payload
            return _openai_response("继续补充：官方体验入口仍应优先核对 chatglm.cn 和 bigmodel.cn。")

    config = GatewayConfig(
        server=ServerConfig(api_key="local-dev-key"),
        upstream=UpstreamConfig(base_url="http://upstream.test/v1", model="web-model"),
        provider_runtime=ProviderRuntimeConfig(native_web_search_policy="auto"),
        tool_bridge=ToolBridgeConfig(activation_policy="auto", exposure_policy="all"),
    )
    client = TestClient(
        create_app(
            config=config,
            credential_store=_credential_store(tmp_path),
            qwen_client_factory=ContinuationQwenClient,
            http_client=_not_found_client(),
        )
    )

    response = client.post(
        "/v1/messages",
        headers={**_headers(), "anthropic-version": "2023-06-01"},
        json={
            "model": "qwen-web/qwen3.6-max-preview",
            "messages": [
                {"role": "user", "content": "联网搜索下 GLM5.1 官方体验网页是哪个"},
                {"role": "assistant", "content": [{"type": "text", "text": "https://chatglm.cn/\n依据：智谱清言官方入口。"}]},
                {"role": "user", "content": "继续"},
            ],
            "tools": [{"name": "WebSearch", "description": "Search the web", "input_schema": {"type": "object"}}],
            "max_tokens": 1024,
        },
    )

    assert response.status_code == 200
    payload = seen["payload"]
    assert payload["_webai_native_web_search"] is True
    assert "tools" not in payload
    prompt = "\n".join(str(message.get("content", "")) for message in payload["messages"])
    assert "WebAI Gateway's strict tool bridge" not in prompt
    assert "不要只回复搜索计划" in prompt
    assert response.json()["content"] == [{"type": "text", "text": "继续补充：官方体验入口仍应优先核对 chatglm.cn 和 bigmodel.cn。"}]


def test_qwen_web_native_search_off_keeps_web_tool_bridge_available(tmp_path: Path) -> None:
    seen: dict[str, Any] = {}

    class ToolQwenClient:
        def __init__(self, credential: dict[str, Any], http_client: httpx.Client | None = None) -> None:
            self.credential = credential

        def chat_completions(self, payload: dict[str, Any]) -> dict[str, Any]:
            seen["payload"] = payload
            return _openai_response('```tool_json\n{"calls":[{"id":"toolu_web","name":"WebSearch","input":{"query":"GLM-5.1 官方体验网页"}}]}\n```')

    config = GatewayConfig(
        server=ServerConfig(api_key="local-dev-key"),
        upstream=UpstreamConfig(base_url="http://upstream.test/v1", model="web-model"),
        provider_runtime=ProviderRuntimeConfig(native_web_search_policy="off"),
        tool_bridge=ToolBridgeConfig(activation_policy="auto", exposure_policy="all"),
    )
    client = TestClient(
        create_app(
            config=config,
            credential_store=_credential_store(tmp_path),
            qwen_client_factory=ToolQwenClient,
            http_client=_not_found_client(),
        )
    )

    response = client.post(
        "/v1/messages",
        headers={**_headers(), "anthropic-version": "2023-06-01"},
        json={
            "model": "qwen-web/qwen3.6-max-preview",
            "messages": [{"role": "user", "content": "联网搜索下 GLM5.1 官方体验网页是哪个"}],
            "tools": [{"name": "WebSearch", "description": "Search the web", "input_schema": {"type": "object"}}],
            "max_tokens": 1024,
        },
    )

    assert response.status_code == 200
    prompt = "\n".join(str(message.get("content", "")) for message in seen["payload"]["messages"])
    assert "WebAI Gateway's strict tool bridge" in prompt
    assert seen["payload"].get("_webai_native_web_search") is False
    assert response.json()["content"] == [{"type": "tool_use", "id": "toolu_web", "name": "WebSearch", "input": {"query": "GLM-5.1 官方体验网页"}}]


def test_qwen_web_auto_activation_keeps_tool_bridge_for_local_agent_task(tmp_path: Path) -> None:
    seen: dict[str, Any] = {}

    class CapturingQwenClient:
        def __init__(self, credential: dict[str, Any], http_client: httpx.Client | None = None) -> None:
            self.credential = credential

        def chat_completions(self, payload: dict[str, Any]) -> dict[str, Any]:
            seen["payload"] = payload
            return _openai_response('```tool_json\n{"calls":[{"id":"toolu_read","name":"Read","input":{"file_path":"README.md"}}]}\n```')

    config = GatewayConfig(
        server=ServerConfig(api_key="local-dev-key"),
        upstream=UpstreamConfig(base_url="http://upstream.test/v1", model="web-model"),
        tool_bridge=ToolBridgeConfig(activation_policy="auto", exposure_policy="all"),
    )
    client = TestClient(
        create_app(
            config=config,
            credential_store=_credential_store(tmp_path),
            qwen_client_factory=CapturingQwenClient,
            http_client=_not_found_client(),
        )
    )

    response = client.post(
        "/v1/messages",
        headers={**_headers(), "anthropic-version": "2023-06-01"},
        json={
            "model": "qwen-web/qwen3.6-plus",
            "messages": [{"role": "user", "content": "/init"}],
            "tools": [{"name": "Read", "description": "Read files", "input_schema": {"type": "object"}}],
            "max_tokens": 1024,
        },
    )

    assert response.status_code == 200
    prompt = "\n".join(str(message.get("content", "")) for message in seen["payload"]["messages"])
    assert "WebAI Gateway's strict tool bridge" in prompt
    assert seen["payload"].get("_webai_native_web_search") is not True
    assert response.json()["content"] == [{"type": "tool_use", "id": "toolu_read", "name": "Read", "input": {"file_path": "README.md"}}]


def test_qwen_web_tool_bridge_extracts_embedded_json_tool_call_after_prose(tmp_path: Path) -> None:
    class EmbeddedJsonQwenClient:
        def __init__(self, credential: dict[str, Any], http_client: httpx.Client | None = None) -> None:
            self.credential = credential

        def chat_completions(self, payload: dict[str, Any]) -> dict[str, Any]:
            return _openai_response(
                'Using using-superpowers to structure the implementation plan.\n'
                '{\n'
                '  "calls": [\n'
                '    {\n'
                '      "id": "call_1",\n'
                '      "name": "Skill",\n'
                '      "input": {"skill_name": "using-superpowers"}\n'
                '    }\n'
                '  ]\n'
                '}'
            )

    config = GatewayConfig(
        server=ServerConfig(api_key="local-dev-key"),
        upstream=UpstreamConfig(base_url="http://upstream.test/v1", model="web-model"),
        tool_bridge=ToolBridgeConfig(activation_policy="auto", exposure_policy="all"),
    )
    client = TestClient(
        create_app(
            config=config,
            credential_store=_credential_store(tmp_path),
            qwen_client_factory=EmbeddedJsonQwenClient,
            http_client=_not_found_client(),
        )
    )

    response = client.post(
        "/v1/messages",
        headers={**_headers(), "anthropic-version": "2023-06-01"},
        json={
            "model": "qwen-web/qwen3.6-max-preview",
            "messages": [{"role": "user", "content": "/using-superpowers"}],
            "tools": [
                {
                    "name": "Skill",
                    "description": "Activate an available skill",
                    "input_schema": {
                        "type": "object",
                        "properties": {"skill_name": {"type": "string"}},
                        "required": ["skill_name"],
                    },
                }
            ],
            "max_tokens": 1024,
        },
    )

    assert response.status_code == 200
    assert response.json()["content"] == [
        {"type": "tool_use", "id": "toolu_call_1", "name": "Skill", "input": {"skill_name": "using-superpowers"}}
    ]


def test_qwen_web_tool_bridge_repairs_embedded_unknown_task_tool_call(tmp_path: Path) -> None:
    seen_payloads: list[dict[str, Any]] = []

    class UnknownTaskThenAllowedToolQwenClient:
        def __init__(self, credential: dict[str, Any], http_client: httpx.Client | None = None) -> None:
            self.credential = credential

        def chat_completions(self, payload: dict[str, Any]) -> dict[str, Any]:
            seen_payloads.append(payload)
            if len(seen_payloads) == 1:
                return _openai_response(
                    '● {"calls":[{"id":"call_1","name":"Task","input":{"description":"搜索智谱清言/GLM5支持",'
                    '"prompt":"在当前代码库中搜索是否包含智谱清言","subagent_type":"general-purpose"}}]}'
                )
            return _openai_response(
                '```tool_json\n{"calls":[{"id":"call_2","name":"Read","input":{"file_path":"webai_gateway/web_auth.py"}}]}\n```'
            )

    config = GatewayConfig(
        server=ServerConfig(api_key="local-dev-key"),
        upstream=UpstreamConfig(base_url="http://upstream.test/v1", model="web-model"),
        tool_bridge=ToolBridgeConfig(activation_policy="auto", exposure_policy="all"),
    )
    client = TestClient(
        create_app(
            config=config,
            credential_store=_credential_store(tmp_path),
            qwen_client_factory=UnknownTaskThenAllowedToolQwenClient,
            http_client=_not_found_client(),
        )
    )

    response = client.post(
        "/v1/messages",
        headers={**_headers(), "anthropic-version": "2023-06-01"},
        json={
            "model": "qwen-web/qwen3.6-max-preview",
            "messages": [{"role": "user", "content": "原始 web2api 项目支持智谱清言的网页和 GLM5 模型吗？"}],
            "tools": [{"name": "Read", "description": "Read files", "input_schema": {"type": "object"}}],
            "max_tokens": 1024,
        },
    )

    assert response.status_code == 200
    assert len(seen_payloads) == 2
    retry_prompt = "\n".join(str(message.get("content", "")) for message in seen_payloads[1]["messages"])
    assert "未知工具：Task" in retry_prompt
    assert response.json()["content"] == [
        {"type": "tool_use", "id": "toolu_call_2", "name": "Read", "input": {"file_path": "webai_gateway/web_auth.py"}}
    ]


def test_qwen_web_tool_bridge_recovers_when_unknown_tool_repair_repeats(tmp_path: Path) -> None:
    seen_payloads: list[dict[str, Any]] = []

    class RepeatedUnknownTaskThenAnswerQwenClient:
        def __init__(self, credential: dict[str, Any], http_client: httpx.Client | None = None) -> None:
            self.credential = credential

        def chat_completions(self, payload: dict[str, Any]) -> dict[str, Any]:
            seen_payloads.append(payload)
            if len(seen_payloads) <= 2:
                return _openai_response(
                    '{"calls":[{"id":"call_1","name":"Task","input":{"description":"搜索智谱清言/GLM5支持",'
                    '"prompt":"检查是否支持智谱清言","subagent_type":"general-purpose"}}]}'
                )
            return _openai_response("结论：当前只发现 zAI/zai_is_text 的 glm-4.6 支持，未发现 chatglm.cn 或 GLM-5 网页模型实现。")

    config = GatewayConfig(
        server=ServerConfig(api_key="local-dev-key"),
        upstream=UpstreamConfig(base_url="http://upstream.test/v1", model="web-model"),
        tool_bridge=ToolBridgeConfig(activation_policy="auto", exposure_policy="all"),
    )
    client = TestClient(
        create_app(
            config=config,
            credential_store=_credential_store(tmp_path),
            qwen_client_factory=RepeatedUnknownTaskThenAnswerQwenClient,
            http_client=_not_found_client(),
        )
    )

    response = client.post(
        "/v1/messages",
        headers={**_headers(), "anthropic-version": "2023-06-01"},
        json={
            "model": "qwen-web/qwen3.6-max-preview",
            "messages": [{"role": "user", "content": "原始 web2api 项目支持智谱清言的网页和 GLM5 模型吗？"}],
            "tools": [{"name": "Read", "description": "Read files", "input_schema": {"type": "object"}}],
            "max_tokens": 1024,
        },
    )

    assert response.status_code == 200
    assert len(seen_payloads) == 3
    final_retry_prompt = "\n".join(str(message.get("content", "")) for message in seen_payloads[2]["messages"])
    assert "不要再请求未列出的工具" in final_retry_prompt
    assert "Task" in final_retry_prompt
    assert response.json()["content"] == [
        {"type": "text", "text": "结论：当前只发现 zAI/zai_is_text 的 glm-4.6 支持，未发现 chatglm.cn 或 GLM-5 网页模型实现。"}
    ]


def test_qwen_web_retries_incomplete_prelude_for_direct_answer(tmp_path: Path) -> None:
    seen_payloads: list[dict[str, Any]] = []

    class PreludeThenAnswerQwenClient:
        def __init__(self, credential: dict[str, Any], http_client: httpx.Client | None = None) -> None:
            self.credential = credential

        def chat_completions(self, payload: dict[str, Any]) -> dict[str, Any]:
            seen_payloads.append(payload)
            if len(seen_payloads) == 1:
                return _openai_response("我来设计一个支持 GLM-5 网页授权的实现计划。")
            return _openai_response("完整计划：第一步梳理 provider 能力，第二步实现授权捕获，第三步补充配置和测试。")

    config = GatewayConfig(
        server=ServerConfig(api_key="local-dev-key"),
        upstream=UpstreamConfig(base_url="http://upstream.test/v1", model="web-model"),
        tool_bridge=ToolBridgeConfig(activation_policy="auto", exposure_policy="all"),
    )
    client = TestClient(
        create_app(
            config=config,
            credential_store=_credential_store(tmp_path),
            qwen_client_factory=PreludeThenAnswerQwenClient,
            http_client=_not_found_client(),
        )
    )

    response = client.post(
        "/v1/messages",
        headers={**_headers(), "anthropic-version": "2023-06-01"},
        json={
            "model": "qwen-web/qwen3.6-max-preview",
            "messages": [{"role": "user", "content": "设计 GLM-5 网页授权计划"}],
            "max_tokens": 1024,
        },
    )

    assert response.status_code == 200
    assert len(seen_payloads) == 2
    retry_prompt = "\n".join(str(message.get("content", "")) for message in seen_payloads[1]["messages"])
    assert "只包含准备动作或开头" in retry_prompt
    assert response.json()["content"] == [
        {"type": "text", "text": "完整计划：第一步梳理 provider 能力，第二步实现授权捕获，第三步补充配置和测试。"}
    ]


def test_qwen_web_retries_incomplete_prelude_to_tool_call_when_bridge_active(tmp_path: Path) -> None:
    seen_payloads: list[dict[str, Any]] = []

    class PreludeThenToolQwenClient:
        def __init__(self, credential: dict[str, Any], http_client: httpx.Client | None = None) -> None:
            self.credential = credential

        def chat_completions(self, payload: dict[str, Any]) -> dict[str, Any]:
            seen_payloads.append(payload)
            if len(seen_payloads) == 1:
                return _openai_response("我来设计一个支持 GLM-5 网页授权的实现计划。")
            return _openai_response(
                '```tool_json\n{"calls":[{"id":"call_1","name":"Read","input":{"file_path":"webai_gateway/web_auth.py"}}]}\n```'
            )

    config = GatewayConfig(
        server=ServerConfig(api_key="local-dev-key"),
        upstream=UpstreamConfig(base_url="http://upstream.test/v1", model="web-model"),
        tool_bridge=ToolBridgeConfig(activation_policy="auto", exposure_policy="all"),
    )
    client = TestClient(
        create_app(
            config=config,
            credential_store=_credential_store(tmp_path),
            qwen_client_factory=PreludeThenToolQwenClient,
            http_client=_not_found_client(),
        )
    )

    response = client.post(
        "/v1/messages",
        headers={**_headers(), "anthropic-version": "2023-06-01"},
        json={
            "model": "qwen-web/qwen3.6-max-preview",
            "messages": [{"role": "user", "content": "当前项目支持 GLM-5 网页授权，设计计划"}],
            "tools": [{"name": "Read", "description": "Read files", "input_schema": {"type": "object"}}],
            "max_tokens": 1024,
        },
    )

    assert response.status_code == 200
    assert len(seen_payloads) == 2
    retry_prompt = "\n".join(str(message.get("content", "")) for message in seen_payloads[1]["messages"])
    assert "如果确实需要工具" in retry_prompt
    assert response.json()["content"] == [
        {"type": "tool_use", "id": "toolu_call_1", "name": "Read", "input": {"file_path": "webai_gateway/web_auth.py"}}
    ]


def test_qwen_web_tool_bridge_repairs_deferred_research_without_tool_call(tmp_path: Path) -> None:
    seen_payloads: list[dict[str, Any]] = []

    class DeferredResearchQwenClient:
        def __init__(self, credential: dict[str, Any], http_client: httpx.Client | None = None) -> None:
            self.credential = credential

        def chat_completions(self, payload: dict[str, Any]) -> dict[str, Any]:
            seen_payloads.append(payload)
            if len(seen_payloads) == 1:
                return _openai_response("我来设计一个实现计划。首先让我研究现有的网页授权实现模式。")
            return _openai_response('```tool_json\n{"calls":[{"id":"toolu_read","name":"Read","input":{"file_path":"webai_gateway/web_auth.py"}}]}\n```')

    config = GatewayConfig(
        server=ServerConfig(api_key="local-dev-key"),
        upstream=UpstreamConfig(base_url="http://upstream.test/v1", model="web-model"),
        tool_bridge=ToolBridgeConfig(activation_policy="auto", exposure_policy="all"),
    )
    client = TestClient(
        create_app(
            config=config,
            credential_store=_credential_store(tmp_path),
            qwen_client_factory=DeferredResearchQwenClient,
            http_client=_not_found_client(),
        )
    )

    response = client.post(
        "/v1/messages",
        headers={**_headers(), "anthropic-version": "2023-06-01"},
        json={
            "model": "qwen-web/qwen3.6-max-preview",
            "messages": [{"role": "user", "content": "当前这个项目也需要支持 GLM5 的网页授权，你设计一个计划"}],
            "tools": [
                {"name": "Read", "description": "Read files", "input_schema": {"type": "object"}},
                {"name": "Glob", "description": "Find files", "input_schema": {"type": "object"}},
            ],
            "max_tokens": 1024,
        },
    )

    assert response.status_code == 200
    assert len(seen_payloads) == 2
    retry_prompt = "\n".join(str(message.get("content", "")) for message in seen_payloads[1]["messages"])
    assert "没有发起工具调用" in retry_prompt
    assert response.json()["content"] == [
        {"type": "tool_use", "id": "toolu_read", "name": "Read", "input": {"file_path": "webai_gateway/web_auth.py"}}
    ]


def test_qwen_web_tool_bridge_recovers_when_permission_denial_repair_repeats(tmp_path: Path) -> None:
    seen_payloads: list[dict[str, Any]] = []
    denial = (
        "由于我无法直接访问您的文件系统或执行命令，我将为您提供手动更新GitHub项目的详细步骤。"
        "这是最安全可靠的方式，避免意外数据丢失。"
    )

    class RepeatedPermissionDenialThenToolQwenClient:
        def __init__(self, credential: dict[str, Any], http_client: httpx.Client | None = None) -> None:
            self.credential = credential

        def chat_completions(self, payload: dict[str, Any]) -> dict[str, Any]:
            seen_payloads.append(payload)
            if len(seen_payloads) <= 2:
                return _openai_response(denial)
            return _openai_response(
                '```tool_json\n{"calls":[{"id":"toolu_bash","name":"Bash","input":{"command":"git status --short"}}]}\n```'
            )

    config = GatewayConfig(
        server=ServerConfig(api_key="local-dev-key"),
        upstream=UpstreamConfig(base_url="http://upstream.test/v1", model="web-model"),
        tool_bridge=ToolBridgeConfig(activation_policy="auto", exposure_policy="all"),
    )
    client = TestClient(
        create_app(
            config=config,
            credential_store=_credential_store(tmp_path),
            qwen_client_factory=RepeatedPermissionDenialThenToolQwenClient,
            http_client=_not_found_client(),
        )
    )

    response = client.post(
        "/v1/messages",
        headers={**_headers(), "anthropic-version": "2023-06-01"},
        json={
            "model": "qwen-web/qwen3.6-max-preview",
            "messages": [{"role": "user", "content": "更新项目并推送到 GitHub"}],
            "tools": [
                {"name": "Bash", "description": "Run shell commands", "input_schema": {"type": "object"}},
                {"name": "Read", "description": "Read files", "input_schema": {"type": "object"}},
            ],
            "max_tokens": 1024,
        },
    )

    assert response.status_code == 200
    assert len(seen_payloads) == 3
    final_retry_prompt = "\n".join(str(message.get("content", "")) for message in seen_payloads[2]["messages"])
    assert "manual steps" in final_retry_prompt
    assert "Bash" in final_retry_prompt
    assert response.json()["content"] == [
        {"type": "tool_use", "id": "toolu_bash", "name": "Bash", "input": {"command": "git status --short"}}
    ]


def test_qwen_web_streaming_empty_direct_response_returns_diagnostic_text(tmp_path: Path) -> None:
    class EmptyQwenClient:
        def __init__(self, credential: dict[str, Any], http_client: httpx.Client | None = None) -> None:
            self.credential = credential

        def chat_completions(self, payload: dict[str, Any]) -> dict[str, Any]:
            return _openai_response("")

    config = GatewayConfig(
        server=ServerConfig(api_key="local-dev-key"),
        upstream=UpstreamConfig(base_url="http://upstream.test/v1", model="web-model"),
        provider_runtime=ProviderRuntimeConfig(native_web_search_policy="auto"),
        tool_bridge=ToolBridgeConfig(activation_policy="auto", exposure_policy="all"),
    )
    client = TestClient(
        create_app(
            config=config,
            credential_store=_credential_store(tmp_path),
            qwen_client_factory=EmptyQwenClient,
            http_client=_not_found_client(),
        )
    )

    response = client.post(
        "/v1/messages",
        headers={**_headers(), "anthropic-version": "2023-06-01"},
        json={
            "model": "qwen-web/qwen3.6-plus",
            "stream": True,
            "messages": [{"role": "user", "content": "联网搜索一下最新模型发布信息"}],
            "tools": [{"name": "WebSearch", "description": "Search the web", "input_schema": {"type": "object"}}],
            "max_tokens": 1024,
        },
    )

    assert response.status_code == 200
    assert "上游模型返回了空响应" in response.text


def test_anthropic_qwen_web_accepts_image_blocks_as_multimodal_payload(tmp_path: Path) -> None:
    _FakeQwenMultimodalClient.captured_payload = {}
    client = TestClient(
        create_app(
            config=_config(),
            credential_store=_credential_store(tmp_path),
            qwen_client_factory=_FakeQwenMultimodalClient,
            http_client=_not_found_client(),
        )
    )

    response = client.post(
        "/v1/messages",
        headers={**_headers(), "anthropic-version": "2023-06-01"},
        json={
            "model": "qwen-web/qwen3.6-plus",
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "请描述这张图"},
                        {"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": "aGVsbG8="}},
                    ],
                }
            ],
            "max_tokens": 1024,
        },
    )

    assert response.status_code == 200
    payload = _FakeQwenMultimodalClient.captured_payload
    content = payload["messages"][0]["content"]
    assert content[0] == {"type": "text", "text": "请描述这张图"}
    assert content[1] == {"type": "image_url", "image_url": {"url": "data:image/png;base64,aGVsbG8="}}
    assert response.json()["content"] == [{"type": "text", "text": "我看到了图片"}]


def test_qwen_web_client_rejects_multimodal_without_upload_support() -> None:
    seen: dict[str, Any] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path.endswith("/api/v2/chats/new"):
            return httpx.Response(200, json={"data": {"id": "chat-test"}}, request=request)
        if request.url.path.endswith("/api/v2/chat/completions"):
            seen["body"] = json.loads(request.content.decode("utf-8"))
            return httpx.Response(
                200,
                content=b'data: {"choices":[{"delta":{"content":"ok","phase":"answer"}}]}\n\ndata: [DONE]\n\n',
                request=request,
                headers={"content-type": "text/event-stream"},
            )
        return httpx.Response(404, request=request)

    qwen = QwenWebClient(
        {"cookie": "qwen_session=session-secret", "bearer": "bearer-secret", "userAgent": "Chrome Test"},
        http_client=httpx.Client(transport=httpx.MockTransport(handler)),
    )

    with pytest.raises(RuntimeError, match="Qwen Web .*multimodal"):
        qwen.chat_completions(
            {
                "model": "qwen-web/qwen3.6-plus",
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": "describe"},
                            {"type": "image_url", "image_url": {"url": "data:image/png;base64,aGVsbG8="}},
                        ],
                    }
                ],
            }
        )

    assert "body" not in seen


def test_qwen_web_client_enables_auto_search_when_payload_requests_native_web_search() -> None:
    seen: dict[str, Any] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path.endswith("/api/v2/chats/new"):
            return httpx.Response(200, json={"data": {"id": "chat-test"}}, request=request)
        if request.url.path.endswith("/api/v2/chat/completions"):
            seen["body"] = json.loads(request.content.decode("utf-8"))
            return httpx.Response(
                200,
                content=b'data: {"choices":[{"delta":{"content":"ok","phase":"answer"}}]}\n\ndata: [DONE]\n\n',
                request=request,
                headers={"content-type": "text/event-stream"},
            )
        return httpx.Response(404, request=request)

    qwen = QwenWebClient(
        {"cookie": "qwen_session=session-secret", "bearer": "bearer-secret", "userAgent": "Chrome Test"},
        http_client=httpx.Client(transport=httpx.MockTransport(handler)),
    )

    response = qwen.chat_completions(
        {
            "model": "qwen-web/qwen3.6-plus",
            "messages": [{"role": "user", "content": "联网搜索 GLM-5.1 官网体验地址"}],
            "_webai_native_web_search": True,
        }
    )

    assert response["choices"][0]["message"]["content"] == "ok"
    feature_config = seen["body"]["messages"][0]["feature_config"]
    assert seen["body"]["search"] is True
    assert feature_config["auto_search"] is True
    assert feature_config["thinking_enabled"] is False


def test_qwen_web_bridge_rejects_filtered_runtime_tool_call(tmp_path: Path) -> None:
    class RuntimeToolQwenClient:
        def __init__(self, credential: dict[str, Any], http_client: httpx.Client | None = None) -> None:
            self.credential = credential

        def chat_completions(self, payload: dict[str, Any]) -> dict[str, Any]:
            return _openai_response('```tool_json\n{"name":"terminal","args":{"command":"gh repo view Einsia/OpenChronicle"}}\n```')

    store = CredentialStore(tmp_path / "credentials")
    store.save(
        "qwen",
        {
            "cookie": "qwen_session=session-secret",
            "bearer": "bearer-secret",
            "userAgent": "Chrome Test",
        },
    )
    config = GatewayConfig(
        server=ServerConfig(api_key="local-dev-key"),
        upstream=UpstreamConfig(base_url="http://upstream.test/v1", model="web-model"),
        tool_bridge=ToolBridgeConfig(activation_policy="always"),
    )
    client = TestClient(
        create_app(
            config=config,
            credential_store=store,
            qwen_client_factory=RuntimeToolQwenClient,
            http_client=_not_found_client(),
        )
    )

    response = client.post(
        "/v1/chat/completions",
        headers=_headers(),
        json={
            "model": "qwen-web/qwen3.6-max-preview",
            "messages": [{"role": "user", "content": "What is OpenChronicle?"}],
            "tools": [
                {"type": "function", "function": {"name": "terminal", "parameters": {"type": "object"}}},
                {"type": "function", "function": {"name": "fetch_url", "parameters": {"type": "object"}}},
            ],
        },
    )

    assert response.status_code == 200
    assert response.headers["x-webai-tool-bridge-error"] == "unknown_tool"
    message = response.json()["choices"][0]["message"]
    assert "tool_calls" not in message
    assert message["webai_tool_bridge"]["error"] == "unknown_tool"


def test_qwen_web_streaming_tool_call_returns_openai_tool_chunk(tmp_path: Path) -> None:
    store = CredentialStore(tmp_path / "credentials")
    store.save(
        "qwen",
        {
            "cookie": "qwen_session=session-secret",
            "bearer": "bearer-secret",
            "userAgent": "Chrome Test",
        },
    )
    client = TestClient(
        create_app(
            config=_config(),
            credential_store=store,
            qwen_client_factory=_FakeQwenToolClient,
            http_client=_not_found_client(),
        )
    )

    with client.stream(
        "POST",
        "/v1/chat/completions",
        headers=_headers(),
        json={
            "model": "qwen-web/qwen3.6-plus",
            "stream": True,
            "messages": [{"role": "user", "content": "list files"}],
            "tools": [{"type": "function", "function": {"name": "list_local_files", "parameters": {"type": "object"}}}],
        },
    ) as response:
        body = response.read().decode("utf-8")

    assert response.status_code == 200
    assert "tool_json" not in body
    assert '"tool_calls"' in body
    assert '"finish_reason":"tool_calls"' in body.replace(" ", "")
    assert "list_local_files" in body


def test_deepseek_web_chat_requires_browser_login(tmp_path: Path) -> None:
    client = TestClient(
        create_app(
            config=_config(),
            credential_store=CredentialStore(tmp_path / "credentials"),
            http_client=_not_found_client(),
        )
    )

    response = client.post(
        "/v1/chat/completions",
        headers=_headers(),
        json={"model": "deepseek-web/deepseek-chat", "messages": [{"role": "user", "content": "你好"}]},
    )

    assert response.status_code == 401
    assert "请先在控制台完成 DeepSeek 网页登录授权" in response.json()["detail"]


def test_qwen_web_chat_requires_browser_login(tmp_path: Path) -> None:
    client = TestClient(
        create_app(
            config=_config(),
            credential_store=CredentialStore(tmp_path / "credentials"),
            http_client=_not_found_client(),
        )
    )

    response = client.post(
        "/v1/chat/completions",
        headers=_headers(),
        json={"model": "qwen-web/qwen3.6-max", "messages": [{"role": "user", "content": "你好"}]},
    )

    assert response.status_code == 401
    assert "请先在控制台完成 Qwen 网页登录授权" in response.json()["detail"]


def test_qwen_web_chat_rejects_cookie_only_credentials(tmp_path: Path) -> None:
    credential_dir = tmp_path / "credentials"
    credential_dir.mkdir()
    (credential_dir / "qwen.json").write_text(
        json.dumps(
            {
                "provider": "qwen",
                "cookie": "visitor_id=visitor; device_id=device",
                "bearer": "",
                "userAgent": "Chrome Test",
                "metadata": {"sessionToken": ""},
                "updatedAt": "2026-04-26T00:00:00+00:00",
            }
        ),
        encoding="utf-8",
    )

    def fail_if_called(*args, **kwargs):
        raise AssertionError("Qwen client should not be called with visitor cookies")

    client = TestClient(
        create_app(
            config=_config(),
            credential_store=CredentialStore(credential_dir),
            qwen_client_factory=fail_if_called,
            http_client=_not_found_client(),
        )
    )

    response = client.post(
        "/v1/chat/completions",
        headers=_headers(),
        json={"model": "qwen-web/qwen3.6-max", "messages": [{"role": "user", "content": "你好"}]},
    )

    assert response.status_code == 401
    assert "请先在控制台完成 Qwen 网页登录授权" in response.json()["detail"]
