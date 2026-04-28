from __future__ import annotations

import json
import re
import time
import uuid
from typing import Any

import httpx

from webai_gateway.config import GatewayConfig
from webai_gateway.tool_bridge import (
    BridgeResult,
    ToolBridgeContext,
    build_context,
    build_repair_messages,
    extract_tool_calls,
    parse_tool_response,
    prepare_openai_messages,
    should_enable_native_web_search,
    should_bridge_tools,
    to_openai_tool_calls,
)

EMPTY_ASSISTANT_RESPONSE_TEXT = (
    "上游模型返回了空响应。请重试一次；如果持续出现，请在 Gateway 控制台检查 provider 登录态、"
    "模型可用性和请求历史。"
)
TOOL_BRIDGE_REJECTED_RESPONSE_TEXT = (
    "上游模型请求了当前未允许或无效的工具调用，Gateway 已拒绝该工具 JSON。"
    "请重试，或确认下游客户端已暴露所需工具；Gateway 不会执行未注册工具。"
)
NATIVE_WEB_SEARCH_FINAL_ANSWER_INSTRUCTION = (
    "Provider 原生联网能力已启用。请在这一轮直接完成联网检索并给出最终答案；"
    "不要只回复搜索计划、正在搜索、我来帮你搜索等过程说明。"
    "如果问题询问网址、官网、地址、URL 或 link，请给出直接 URL 和一句依据；"
    "如果无法确认，请明确说明无法确认及原因。"
)
NATIVE_WEB_SEARCH_RETRY_INSTRUCTION = (
    "上一轮只返回了搜索计划或不完整答复，没有给出最终结果。"
    "请现在使用原生联网搜索并直接输出最终答案；不要再说“我来搜索/正在搜索”。"
    "如果问题询问网址、官网、地址、URL 或 link，请给出直接 URL 和一句依据；"
    "如果无法确认，请明确说明无法确认及原因。"
)
_URL_RE = re.compile(r"https?://|www\.", re.IGNORECASE)
_NATIVE_SEARCH_PLACEHOLDER_RE = re.compile(
    r"^\s*(?:好的[，,]?\s*)?"
    r"(?:(?:我(?:来|会|将|可以)?|让我|让我们|帮你|为你|现在|正在).{0,24}(?:联网搜索|搜索|查询|查找|检索)"
    r"|(?:i(?:'ll| will)|i’ll|let me|i can|i am going to|i'm going to).{0,36}(?:search|look up|find|check))",
    re.IGNORECASE,
)
_INCOMPLETE_PRELUDE_RE = re.compile(
    r"^\s*(?:好的[，,。]?\s*)?"
    r"(?:(?:我(?:来|会|将|先)|让我|首先|接下来|现在|正在).{0,120}"
    r"(?:帮你|为你|开始|搜索|查找|查看|读取|研究|分析|设计|制定|梳理|处理|准备|继续|实现|计划)"
    r"|(?:i(?:'ll| will)|i’ll|let me|first|next|i am going to|i'm going to).{0,120}"
    r"(?:help|start|search|look up|inspect|read|research|analy[sz]e|design|draft|prepare|check|continue|plan))",
    re.IGNORECASE | re.DOTALL,
)
_SUBSTANTIVE_ANSWER_MARKER_RE = re.compile(
    r"(\n\s*(?:[-*•]|\d+[.、)]|#{1,6}\s)|```|https?://|第[一二三四五六七八九十]+[阶段步点]|[：:]\s*\n)",
    re.IGNORECASE,
)


def upstream_headers(config: GatewayConfig) -> dict[str, str]:
    headers = {"content-type": "application/json"}
    if config.upstream.api_key:
        headers["authorization"] = f"Bearer {config.upstream.api_key}"
    return headers


def build_upstream_payload(
    body: dict[str, Any],
    config: GatewayConfig,
    *,
    provider_native_web_search: bool = False,
) -> tuple[dict[str, Any], bool, set[str], ToolBridgeContext]:
    payload = dict(body)
    tools = body.get("tools")
    bridge_mode = (config.tool_bridge.mode or config.upstream.tool_mode or "strict").strip().lower()
    native_web_search = provider_native_web_search and should_enable_native_web_search(
        body.get("messages"),
        config.provider_runtime.native_web_search_policy,
    )
    bridge = should_bridge_tools(
        tools,
        bridge_mode,
        activation_policy=config.tool_bridge.activation_policy,
        messages=body.get("messages"),
        tool_choice=body.get("tool_choice"),
        provider_native_web_search=native_web_search,
    )
    model = str(body.get("model") or config.upstream.model or "")
    bridge_context = build_context(tools, config.tool_bridge, mode=bridge_mode, model=model)
    allowed_tools: set[str] = set()
    allowed_tools = bridge_context.allowed_names
    payload["model"] = model
    if provider_native_web_search:
        payload["_webai_native_web_search"] = native_web_search
    if bridge:
        messages = body.get("messages") if isinstance(body.get("messages"), list) else []
        payload["messages"] = prepare_openai_messages(messages, bridge_context)
        payload.pop("tools", None)
        payload.pop("tool_choice", None)
    elif native_web_search:
        payload["messages"] = _with_native_web_search_instruction(payload.get("messages"))
    return payload, bridge, allowed_tools, bridge_context


def should_retry_native_web_search_response(data: dict[str, Any]) -> bool:
    content = _openai_message_content(data).strip()
    if not content or _URL_RE.search(content):
        return False
    if len(content) > 240:
        return False
    return bool(_NATIVE_SEARCH_PLACEHOLDER_RE.search(content))


def should_retry_incomplete_response(data: dict[str, Any]) -> bool:
    if _openai_message_tool_calls(data):
        return False
    content = _openai_message_content(data).strip()
    if not content:
        return False
    if len(content) > 320:
        return False
    if _SUBSTANTIVE_ANSWER_MARKER_RE.search(content):
        return False
    return bool(_INCOMPLETE_PRELUDE_RE.search(content))


def build_native_web_search_retry_payload(payload: dict[str, Any], previous_content: str) -> dict[str, Any]:
    retry = dict(payload)
    messages = payload.get("messages") if isinstance(payload.get("messages"), list) else []
    retry_messages = [dict(message) for message in messages if isinstance(message, dict)]
    previous = previous_content.strip()
    if previous:
        previous = f"\n\n上一轮内容：{previous[:500]}"
    retry_messages.append({"role": "user", "content": NATIVE_WEB_SEARCH_RETRY_INSTRUCTION + previous})
    retry["messages"] = retry_messages
    retry["_webai_native_web_search"] = True
    return retry


def build_incomplete_response_retry_payload(payload: dict[str, Any], previous_content: str, *, bridge: bool) -> dict[str, Any]:
    retry = dict(payload)
    messages = payload.get("messages") if isinstance(payload.get("messages"), list) else []
    retry_messages = [dict(message) for message in messages if isinstance(message, dict)]
    previous = (previous_content or "").strip()
    if previous:
        previous = f"\n\n上一轮回复：{previous[:500]}"
    bridge_instruction = (
        "如果确实需要工具，请只输出一个 fenced tool_json block；如果不需要工具，请直接给完整最终答案。"
        if bridge
        else "请直接给完整最终答案。"
    )
    retry_messages.append(
        {
            "role": "user",
            "content": (
                "上一轮回复只包含准备动作或开头，没有完成用户任务。"
                "不要再说“我来/让我/首先/正在”等准备性话术。"
                f"{bridge_instruction}{previous}"
            ),
        }
    )
    retry["messages"] = retry_messages
    return retry


def build_unknown_tool_recovery_payload(
    payload: dict[str, Any],
    bridge_result: BridgeResult,
    *,
    allowed_tools: set[str],
) -> dict[str, Any]:
    recovered = dict(payload)
    messages = payload.get("messages") if isinstance(payload.get("messages"), list) else []
    retry_messages = [dict(message) for message in messages if isinstance(message, dict)]
    allowed = ", ".join(sorted(allowed_tools)) or "(none)"
    reason = bridge_result.error.message if bridge_result.error else "unknown tool"
    retry_messages.append({"role": "assistant", "content": (bridge_result.raw_content or "")[:4000]})
    retry_messages.append(
        {
            "role": "user",
            "content": (
                f"上一轮工具调用被 Gateway 拒绝：{reason}。\n"
                f"当前真实允许的工具名只有：{allowed}。\n"
                "不要再请求未列出的工具，例如 Task、Agent、Subagent、Skill、Bash 或 shell。"
                "如果确实需要工具，只能使用上面列出的真实工具名并输出一个 fenced tool_json block。"
                "如果这些工具不足以完成任务，请不要输出 JSON，直接基于已有上下文给出诚实、完整的最终回答，并说明限制。"
            ),
        }
    )
    recovered["messages"] = retry_messages
    recovered["stream"] = False
    return recovered


def parse_chat_response(
    data: dict[str, Any],
    *,
    bridge: bool,
    allowed_tools: set[str],
    model: str,
    bridge_context: ToolBridgeContext | None = None,
    return_bridge_result: bool = False,
) -> dict[str, Any] | tuple[dict[str, Any], BridgeResult]:
    choices = data.get("choices") if isinstance(data.get("choices"), list) else []
    choice0 = choices[0] if choices and isinstance(choices[0], dict) else {}
    msg = choice0.get("message") if isinstance(choice0.get("message"), dict) else {}
    content = _as_text(msg.get("content"))
    native_tool_calls = msg.get("tool_calls") if isinstance(msg.get("tool_calls"), list) else []
    if not bridge:
        if not native_tool_calls and not content.strip():
            msg["content"] = EMPTY_ASSISTANT_RESPONSE_TEXT
            content = EMPTY_ASSISTANT_RESPONSE_TEXT
        result = BridgeResult(content=content, tool_calls=[], raw_content=content)
        return (data, result) if return_bridge_result else data
    context = bridge_context or _context_from_allowed_tools(allowed_tools)
    result = parse_tool_response(content, context)
    if not result.tool_calls:
        if not result.error and not result.warning and not result.content.strip():
            msg["content"] = EMPTY_ASSISTANT_RESPONSE_TEXT
            result = BridgeResult(content=EMPTY_ASSISTANT_RESPONSE_TEXT, tool_calls=[], raw_content=result.raw_content)
        if result.error and _looks_like_raw_tool_json(result.raw_content):
            msg["content"] = TOOL_BRIDGE_REJECTED_RESPONSE_TEXT
            result = BridgeResult(
                content=TOOL_BRIDGE_REJECTED_RESPONSE_TEXT,
                tool_calls=[],
                error=result.error,
                warning=result.warning,
                raw_content=result.raw_content,
            )
        if content != result.content:
            msg["content"] = result.content
        if result.error:
            msg["webai_tool_bridge"] = {"error": result.error.kind, "message": result.error.message}
        if result.warning:
            msg["webai_tool_bridge_warning"] = result.warning
        return (data, result) if return_bridge_result else data
    parsed = {
        "id": str(data.get("id") or f"chatcmpl-{uuid.uuid4().hex}"),
        "object": "chat.completion",
        "created": int(data.get("created") or time.time()),
        "model": model,
        "choices": [
            {
                "index": int(choice0.get("index") or 0),
                "finish_reason": "tool_calls",
                "message": {
                    "role": "assistant",
                    "content": result.content,
                    "tool_calls": to_openai_tool_calls(result.tool_calls),
                },
            }
        ],
    }
    return (parsed, result) if return_bridge_result else parsed


def _openai_message_content(data: dict[str, Any]) -> str:
    choices = data.get("choices") if isinstance(data.get("choices"), list) else []
    choice0 = choices[0] if choices and isinstance(choices[0], dict) else {}
    msg = choice0.get("message") if isinstance(choice0.get("message"), dict) else {}
    return _as_text(msg.get("content"))


def _openai_message_tool_calls(data: dict[str, Any]) -> list[Any]:
    choices = data.get("choices") if isinstance(data.get("choices"), list) else []
    choice0 = choices[0] if choices and isinstance(choices[0], dict) else {}
    msg = choice0.get("message") if isinstance(choice0.get("message"), dict) else {}
    return msg.get("tool_calls") if isinstance(msg.get("tool_calls"), list) else []


def _looks_like_raw_tool_json(text: str) -> bool:
    raw = text or ""
    return "tool_json" in raw or '"calls"' in raw or '"name"' in raw and any(key in raw for key in ('"input"', '"args"', '"arguments"'))


def _with_native_web_search_instruction(messages: Any) -> list[dict[str, Any]]:
    source = messages if isinstance(messages, list) else []
    normalized = [dict(message) for message in source if isinstance(message, dict)]
    if any(NATIVE_WEB_SEARCH_FINAL_ANSWER_INSTRUCTION in str(message.get("content") or "") for message in normalized):
        return normalized
    return [{"role": "system", "content": NATIVE_WEB_SEARCH_FINAL_ANSWER_INSTRUCTION}, *normalized]


def parse_sse_text(text: str) -> tuple[str, str]:
    content_parts: list[str] = []
    finish_reason = ""
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
            choices = data.get("choices") if isinstance(data.get("choices"), list) else []
            choice0 = choices[0] if choices and isinstance(choices[0], dict) else {}
            if isinstance(choice0.get("finish_reason"), str):
                finish_reason = choice0["finish_reason"]
            delta = choice0.get("delta") if isinstance(choice0.get("delta"), dict) else {}
            content = delta.get("content")
            if content is not None:
                content_parts.append(_as_text(content))
    return "".join(content_parts), finish_reason


def sse_chunk(payload: dict[str, Any]) -> str:
    return "data: " + json.dumps(payload, ensure_ascii=False, separators=(",", ":")) + "\n\n"


def build_tool_call_sse(content: str, *, allowed_tools: set[str], model: str, bridge_context: ToolBridgeContext | None = None) -> str:
    context = bridge_context or _context_from_allowed_tools(allowed_tools)
    result = parse_tool_response(content, context)
    if result.tool_calls:
        return build_openai_tool_calls_sse(to_openai_tool_calls(result.tool_calls), model=model)
    if result.content:
        normal = {
            "id": f"chatcmpl-{uuid.uuid4().hex}",
            "object": "chat.completion.chunk",
            "created": int(time.time()),
            "model": model,
            "choices": [{"index": 0, "delta": {"content": result.content}, "finish_reason": "stop"}],
        }
        return sse_chunk(normal) + "data: [DONE]\n\n"
    return "data: [DONE]\n\n"


def build_openai_tool_calls_sse(tool_calls: list[dict[str, Any]], *, model: str) -> str:
    chunks = []
    for index, call in enumerate(tool_calls):
        if not isinstance(call, dict):
            continue
        fn = call.get("function") if isinstance(call.get("function"), dict) else {}
        chunks.append(
            {
                "index": index,
                "id": str(call.get("id") or f"call_web_{index + 1}"),
                "type": "function",
                "function": fn,
            }
        )
    if not chunks:
        return "data: [DONE]\n\n"
    first = {
        "id": f"chatcmpl-{uuid.uuid4().hex}",
        "object": "chat.completion.chunk",
        "created": int(time.time()),
        "model": model,
        "choices": [{"index": 0, "delta": {"tool_calls": chunks}, "finish_reason": None}],
    }
    final = {
        "id": f"chatcmpl-{uuid.uuid4().hex}",
        "object": "chat.completion.chunk",
        "created": int(time.time()),
        "model": model,
        "choices": [{"index": 0, "delta": {}, "finish_reason": "tool_calls"}],
    }
    return sse_chunk(first) + sse_chunk(final) + "data: [DONE]\n\n"


def post_upstream(client: httpx.Client, config: GatewayConfig, payload: dict[str, Any]) -> httpx.Response:
    url = config.upstream.base_url.rstrip("/") + "/chat/completions"
    return client.post(url, json=payload, headers=upstream_headers(config))


def build_repair_payload(payload: dict[str, Any], bridge_result: BridgeResult) -> dict[str, Any]:
    repaired = dict(payload)
    messages = payload.get("messages") if isinstance(payload.get("messages"), list) else []
    repaired["messages"] = build_repair_messages(messages, bridge_result.raw_content, bridge_result.error)
    repaired["stream"] = False
    return repaired


def bridge_error_headers(result: BridgeResult | None) -> dict[str, str]:
    if result is None or result.error is None:
        return {}
    return {"x-webai-tool-bridge-error": result.error.kind}


def _context_from_allowed_tools(allowed_tools: set[str]) -> ToolBridgeContext:
    tools = [{"type": "function", "function": {"name": name, "parameters": {"type": "object"}}} for name in sorted(allowed_tools)]
    return build_context(tools)


def _as_text(value: Any) -> str:
    if isinstance(value, str):
        return value
    if value is None:
        return ""
    return str(value)
