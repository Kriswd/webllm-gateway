from __future__ import annotations

import json
import re
import time
import uuid
from dataclasses import replace
from typing import Any
from urllib.parse import urlparse

import httpx

from webai_gateway.config import GatewayConfig
from webai_gateway.assistant_turn import build_assistant_turn
from webai_gateway.model_ids import normalize_model_id
from webai_gateway.prompt_compaction import (
    STATELESS_WEB_API_GUARD,
    compact_role_messages_as_ds2api_history,
    message_entries_for_ds2api_prompt,
)
from webai_gateway.tool_bridge import (
    BridgeResult,
    ToolBridgeContext,
    build_context,
    build_local_repo_preflight_tool_call,
    build_skill_loader_preflight_tool_call,
    build_repair_messages,
    extract_tool_calls,
    looks_like_tool_protocol_output,
    parse_tool_response,
    prefer_local_tools_for_local_agent_task,
    prepare_openai_messages,
    sanitize_leaked_tool_protocol_output,
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
TOOL_BRIDGE_REJECTED_RESPONSE_TITLE = (
    "上游模型请求了当前未允许或无效的工具调用，Gateway 已拒绝该工具 JSON。"
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
RESPONSE_LANGUAGE_POLICY_MARKER = "WebLLM Gateway response language policy"
WEBAI2API_CURRENT_TOOL_CHOICE_POLICY_MARKER = "WebLLM Gateway current turn tool-choice policy"
RESPONSE_LANGUAGE_OFF_VALUES = {"", "off", "none", "false", "disabled"}
RESPONSE_LANGUAGE_RETRY_PREVIOUS_MAX_CHARS = 12000
RESPONSE_LANGUAGE_RETRY_INSTRUCTION = (
    "Gateway response language retry: the previous assistant final answer did not follow the configured response "
    "language policy.\n"
    "Rewrite only the previous assistant final answer in Simplified Chinese (zh-CN). Do not add new facts, do not "
    "browse, do not ask the user a question, and do not call or simulate tools.\n"
    "Keep code blocks, code identifiers, file paths, commands, API paths, protocol names, model IDs, product names, "
    "and quoted source text unchanged when translating them would reduce precision."
)
WEBAI2API_CHATGPT_TEXT_PROMPT_MAX_CHARS = 12000
WEBAI2API_CHATGPT_TEXT_MODELS = {"gpt-instant", "gpt-thinking", "gpt-pro"}
_URL_RE = re.compile(r"https?://|www\.", re.IGNORECASE)
_RESPONSE_LANGUAGE_CODE_FENCE_RE = re.compile(r"```.*?```|~~~.*?~~~", re.DOTALL)
_RESPONSE_LANGUAGE_INLINE_CODE_RE = re.compile(r"`[^`\n]+`")
_RESPONSE_LANGUAGE_ENGLISH_WORD_RE = re.compile(r"\b[A-Za-z][A-Za-z'-]{2,}\b")
_RESPONSE_LANGUAGE_CJK_RE = re.compile(r"[\u4e00-\u9fff]")
_RESPONSE_LANGUAGE_OTHER_LANGUAGE_RE = re.compile(
    r"(?:answer|respond|reply|write|translate|rewrite|summari[sz]e)\s+(?:it\s+)?(?:in|to)\s+"
    r"(?:english|japanese|korean|french|german|spanish|russian|arabic|portuguese|italian)\b",
    re.IGNORECASE,
)
_RESPONSE_LANGUAGE_OTHER_LANGUAGE_ZH_RE = re.compile(
    r"(?:请|麻烦|用|使用|改成|翻译成|输出|回答|回复).{0,12}"
    r"(?:英文|英语|日文|日语|韩文|韩语|法文|法语|德文|德语|西班牙文|西班牙语|俄文|俄语)",
)
_RESPONSE_LANGUAGE_OTHER_LANGUAGE_NEGATIVE_RE = re.compile(
    r"(?:不要|别|禁止|不要用|别用).{0,8}(?:英文|英语|english)", re.IGNORECASE
)
_NATIVE_SEARCH_PLACEHOLDER_RE = re.compile(
    r"^\s*(?:好的[，,]?\s*)?"
    r"(?:(?:我(?:来|会|将|可以)?|让我|让我们|帮你|为你|现在|正在).{0,24}(?:联网搜索|搜索|查询|查找|检索)"
    r"|(?:i(?:'ll| will)|i’ll|let me|i can|i am going to|i'm going to).{0,36}(?:search|look up|find|check))",
    re.IGNORECASE,
)
_RECOVERABLE_LOCAL_AGENT_TASK_RE = re.compile(
    r"\b(?:configure|configuration|setup|set\s+up|setting|settings|permission|permissions|command|"
    r"shell|terminal|bash|cmd|powershell|run|execute|install|deploy|edit|write|modify|change|fix)\b|"
    r"(?:配置|设置|权限|命令|终端|运行|执行|启动|安装|部署|编辑|写入|修改|修复)",
    re.IGNORECASE,
)
_INCOMPLETE_PRELUDE_RE = re.compile(
    r"^\s*(?:好的[，,。]?\s*)?"
    r"(?:(?:我(?:来|会|将|先)|让我|首先|接下来|现在|正在).{0,120}"
    r"(?:帮你|为你|开始|搜索|查找|查看|检查|确认|核对|列出|获取|读取|研究|分析|设计|制定|梳理|处理|准备|继续|实现|计划)"
    r"|(?:i(?:'ll| will)|i’ll|let me|first|next|i am going to|i'm going to).{0,120}"
    r"(?:help|start|search|look up|inspect|read|research|analy[sz]e|design|draft|prepare|check|continue|plan))",
    re.IGNORECASE | re.DOTALL,
)
_LEADING_STATUS_PREFIX_RE = re.compile(
    r"^[\s\ufeff\u200b\ufe0f\u2600-\u27bf\U0001f300-\U0001faff\[\]【】()（）:：\-_*•·]+"
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
    model = normalize_model_id(body.get("model"), config.upstream.model)
    bridge_context = build_context(tools, config.tool_bridge, mode=bridge_mode, model=model, tool_choice=body.get("tool_choice"))
    if bridge and _should_fallback_to_safe_exposure(config.tool_bridge, bridge_context, tools):
        bridge_context = build_context(
            tools,
            replace(config.tool_bridge, exposure_policy="safe"),
            mode=bridge_mode,
            model=model,
            tool_choice=body.get("tool_choice"),
        )
    bridge_context = prefer_local_tools_for_local_agent_task(
        bridge_context,
        body.get("messages"),
        tool_choice=body.get("tool_choice"),
    )
    if bridge and _should_recover_empty_local_agent_tool_context(config, bridge_context, body, tools):
        bridge_context = build_context(
            tools,
            replace(config.tool_bridge, tool_profile="agent"),
            mode=bridge_mode,
            model=model,
            tool_choice=body.get("tool_choice"),
        )
        bridge_context = prefer_local_tools_for_local_agent_task(
            bridge_context,
            body.get("messages"),
            tool_choice=body.get("tool_choice"),
        )
    if bridge and bridge_context.enabled:
        native_web_search = False
    allowed_tools: set[str] = set()
    allowed_tools = bridge_context.allowed_names
    payload["model"] = model
    if provider_native_web_search:
        payload["_webai_native_web_search"] = native_web_search
    if bridge:
        messages = body.get("messages") if isinstance(body.get("messages"), list) else []
        prepared_messages = _with_response_language_instruction(
            prepare_openai_messages(messages, bridge_context),
            config.provider_runtime.response_language,
        )
        if _should_inject_webai2api_current_tool_choice_policy(config, bridge_context):
            prepared_messages = _with_webai2api_current_tool_choice_policy(prepared_messages, bridge_context)
        payload["messages"] = prepared_messages
        payload.pop("tools", None)
        payload.pop("tool_choice", None)
    else:
        if _should_inject_response_language_instruction(config, native_web_search=native_web_search):
            payload["messages"] = _with_response_language_instruction(
                payload.get("messages"),
                config.provider_runtime.response_language,
            )
        else:
            payload["messages"] = [dict(message) for message in payload.get("messages", []) if isinstance(message, dict)]
        if native_web_search:
            payload["messages"] = _with_native_web_search_instruction(payload.get("messages"))
    return payload, bridge, allowed_tools, bridge_context


def _should_inject_response_language_instruction(config: GatewayConfig, *, native_web_search: bool = False) -> bool:
    if native_web_search:
        return True
    return not _looks_like_webai2api_sidecar_url(config.upstream.base_url)


def _should_inject_webai2api_current_tool_choice_policy(
    config: GatewayConfig,
    bridge_context: ToolBridgeContext,
) -> bool:
    if not _looks_like_webai2api_sidecar_url(config.upstream.base_url):
        return False
    policy = getattr(bridge_context, "tool_choice_policy", None)
    is_required = getattr(policy, "is_required", None)
    return bool(bridge_context.enabled and callable(is_required) and is_required())


def _with_webai2api_current_tool_choice_policy(
    messages: Any,
    bridge_context: ToolBridgeContext,
) -> list[dict[str, Any]]:
    normalized = [dict(message) for message in messages if isinstance(message, dict)]
    instruction = _webai2api_current_tool_choice_policy_text(bridge_context)
    if not instruction:
        return normalized
    for index in range(len(normalized) - 1, -1, -1):
        if normalized[index].get("role") != "user":
            continue
        normalized[index] = _append_text_to_message_content(normalized[index], instruction)
        return normalized
    return [*normalized, {"role": "user", "content": instruction}]


def _append_text_to_message_content(message: dict[str, Any], text: str) -> dict[str, Any]:
    out = dict(message)
    content = out.get("content")
    if isinstance(content, str):
        if WEBAI2API_CURRENT_TOOL_CHOICE_POLICY_MARKER in content:
            return out
        out["content"] = f"{content.rstrip()}\n\n{text}".strip()
        return out
    if isinstance(content, list):
        if any(
            isinstance(item, dict)
            and WEBAI2API_CURRENT_TOOL_CHOICE_POLICY_MARKER in str(item.get("text") or item.get("content") or "")
            for item in content
        ):
            return out
        out["content"] = [*content, {"type": "text", "text": text}]
        return out
    existing = _as_text(content).strip()
    out["content"] = f"{existing}\n\n{text}".strip() if existing else text
    return out


def _webai2api_current_tool_choice_policy_text(bridge_context: ToolBridgeContext) -> str:
    policy = bridge_context.tool_choice_policy
    mode = str(getattr(policy, "mode", "") or "").strip().lower()
    allowed_names = sorted(str(name).strip() for name in bridge_context.allowed_names if str(name).strip())
    allowed = ", ".join(allowed_names) or "(none)"
    lines = [
        f"[{WEBAI2API_CURRENT_TOOL_CHOICE_POLICY_MARKER}]",
        "This current turn requires a Gateway tool call encoded as a Gateway JSON instruction for the downstream client, not ChatGPT native tools. "
        "You are not executing tools inside ChatGPT; you are only writing the JSON request that Gateway will route.",
        "Do not answer directly, browse, search, cite web results, ask which tool to use, or output prose.",
        f"Allowed tool names for this turn: {allowed}.",
        "The assistant response must start with ```tool_json and contain exactly one Gateway tool_json block.",
    ]
    if mode == "forced":
        forced_name = str(getattr(policy, "forced_name", "") or "").strip()
        if not forced_name and len(allowed_names) == 1:
            forced_name = allowed_names[0]
        if forced_name:
            lines.append(f"Forced tool name: {forced_name}. Do not call any other tool.")
    elif mode == "required":
        lines.append("Use at least one allowed tool. If only one tool is listed, use that tool.")
    example = _required_tool_choice_recovery_example(set(allowed_names), bridge_context)
    if example:
        lines.append("Required tool_json shape for this turn:")
        lines.append(example.strip())
    return "\n".join(lines)


def _looks_like_webai2api_sidecar_url(base_url: str) -> bool:
    try:
        parsed = urlparse(str(base_url or ""))
    except Exception:
        return False
    host = (parsed.hostname or "").lower()
    if host not in {"127.0.0.1", "localhost", "::1"}:
        return False
    return parsed.port == 8500


def _should_fallback_to_safe_exposure(
    options: ToolBridgeConfig,
    context: ToolBridgeContext,
    tools: Any,
) -> bool:
    if not isinstance(tools, list) or not tools:
        return False
    policy = (options.exposure_policy or "safe").strip().lower()
    if policy not in {"local-agent", "local_agent", "code-agent", "code_agent"}:
        return False
    if context.enabled and context.allowed_names:
        return False
    return True


def _should_recover_empty_local_agent_tool_context(
    config: GatewayConfig,
    context: ToolBridgeContext,
    body: dict[str, Any],
    tools: Any,
) -> bool:
    if context.enabled and context.allowed_names:
        return False
    if not isinstance(tools, list) or not tools:
        return False
    options = config.tool_bridge
    policy = (options.exposure_policy or "safe").strip().lower()
    if policy not in {"local-agent", "local_agent", "code-agent", "code_agent"}:
        return False
    if (options.tool_profile or "auto").strip().lower().replace("_", "-") != "auto":
        return False
    task_text = _latest_recoverable_local_agent_task_text(body.get("messages"))
    if not task_text:
        return False
    return bool(_RECOVERABLE_LOCAL_AGENT_TASK_RE.search(task_text))


def _latest_recoverable_local_agent_task_text(messages: Any) -> str:
    if not isinstance(messages, list):
        return ""
    for message in reversed(messages):
        if not isinstance(message, dict) or str(message.get("role") or "") != "user":
            continue
        text = _as_text(message.get("content")).strip()
        if text:
            return text
    return ""


def build_preflight_chat_response(model: str, bridge_context: ToolBridgeContext) -> dict[str, Any] | None:
    call = build_local_repo_preflight_tool_call(bridge_context)
    if call is None:
        return None
    return _tool_call_preflight_chat_response(model, bridge_context, call)


def build_skill_loader_preflight_chat_response(
    model: str,
    bridge_context: ToolBridgeContext,
) -> dict[str, Any] | None:
    call = build_skill_loader_preflight_tool_call(bridge_context)
    if call is None:
        return None
    return _tool_call_preflight_chat_response(model, bridge_context, call)


def _tool_call_preflight_chat_response(
    model: str,
    bridge_context: ToolBridgeContext,
    call: Any,
) -> dict[str, Any]:
    return {
        "id": f"chatcmpl-{uuid.uuid4().hex}",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": model,
        "choices": [
            {
                "index": 0,
                "finish_reason": "tool_calls",
                "message": {
                    "role": "assistant",
                    "content": "",
                    "tool_calls": to_openai_tool_calls([call], bridge_context),
                },
            }
        ],
    }


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
    if _looks_like_empty_or_garbled_fence_text(content):
        return True
    if len(content) > 320:
        return False
    if _SUBSTANTIVE_ANSWER_MARKER_RE.search(content):
        return False
    return bool(_INCOMPLETE_PRELUDE_RE.search(_strip_leading_status_prefix(content)))


def _looks_like_empty_or_garbled_fence_text(text: str) -> bool:
    stripped = (text or "").strip()
    if stripped in {"```", "~~~"}:
        return True
    if len(stripped) > 80 or not (stripped.startswith("```") or stripped.startswith("~~~")):
        return False
    if re.fullmatch(r"```(?:json|tool_json)?\s*```", stripped, re.IGNORECASE):
        return True
    remainder = re.sub(r"[`\s~]+", "", stripped)
    return bool(remainder and len(remainder) <= 24 and not re.search(r"[A-Za-z0-9_\u4e00-\u9fff]", remainder))


def _strip_leading_status_prefix(text: str) -> str:
    return _LEADING_STATUS_PREFIX_RE.sub("", text or "").lstrip()


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


def build_incomplete_response_retry_payload(
    payload: dict[str, Any],
    previous_content: str,
    *,
    bridge: bool,
    allowed_tools: set[str] | None = None,
) -> dict[str, Any]:
    retry = dict(payload)
    messages = payload.get("messages") if isinstance(payload.get("messages"), list) else []
    retry_messages = [dict(message) for message in messages if isinstance(message, dict)]
    previous = (previous_content or "").strip()
    if previous:
        previous = f"\n\n上一轮回复：{previous[:500]}"
    bridge_instruction = (
        "如果确实需要工具，请只输出一个 fenced ```tool_json block；如果不需要工具，请直接给完整最终答案。"
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
    if bridge and allowed_tools:
        retry_messages = _with_standard_tool_json_formatter_message(
            retry_messages,
            allowed_tools=allowed_tools,
            error_kind="incomplete_response",
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
    examples = _unknown_tool_examples(allowed_tools)
    retry_messages.append({"role": "assistant", "content": (bridge_result.raw_content or "")[:4000]})
    retry_messages.append(
        {
            "role": "user",
            "content": (
                f"上一轮工具调用被 Gateway 拒绝：{reason}。\n"
                f"当前真实允许的工具名只有：{allowed}。\n"
                f"不要再请求未列出的工具，例如 {examples}。"
                "如果确实需要工具，只能使用上面列出的真实工具名并输出一个 fenced ```tool_json block。"
                "如果这些工具不足以完成任务，请不要输出 JSON，直接基于已有上下文给出诚实、完整的最终回答，并说明限制。"
            ),
        }
    )
    recovered["messages"] = retry_messages
    recovered["stream"] = False
    return recovered


def build_tool_refusal_recovery_payload(
    payload: dict[str, Any],
    bridge_result: BridgeResult,
    *,
    allowed_tools: set[str],
    escalation: bool = False,
) -> dict[str, Any]:
    recovered = dict(payload)
    messages = payload.get("messages") if isinstance(payload.get("messages"), list) else []
    retry_messages = [dict(message) for message in messages if isinstance(message, dict)]
    error_kind = bridge_result.error.kind if bridge_result.error else ""
    disabled_tools, effective_allowed_tools = _recovery_tool_visibility(
        allowed_tools,
        error_kind=error_kind,
        raw_content=bridge_result.raw_content,
    )
    allowed = ", ".join(sorted(effective_allowed_tools)) or "(none)"
    reason = bridge_result.error.message if bridge_result.error else "tool refusal without call"
    reason = _sanitize_discovery_tool_guidance(reason, effective_allowed_tools, error_kind=error_kind)
    example = _tool_refusal_recovery_example(effective_allowed_tools, error_kind=error_kind)
    shell_instruction = (
        "Bash is allowed; request Bash only when the task explicitly needs git/gh/shell behavior.\n"
        if "Bash" in effective_allowed_tools
        else "Bash/shell is not in the allowed tools list for this turn. Do not request Bash, shell, terminal, cmd, or powershell.\n"
    )
    non_progress_instruction = _non_progress_recovery_instruction(error_kind, disabled_tools)
    progress_tool_hint = (
        "Prefer Edit/MultiEdit/Write to integrate code changes using the already gathered evidence before claiming completion.\n"
        if disabled_tools
        else "Prefer Read/Grep to inspect existing code or Edit/MultiEdit to integrate new code into existing entry points before claiming completion.\n"
    )
    retry_messages.append({"role": "assistant", "content": (bridge_result.raw_content or "")[:4000]})
    retry_messages = _with_recovery_tool_override_message(
        retry_messages,
        bridge_result,
        allowed_tools=allowed_tools,
    )
    escalation_instruction = ""
    if escalation:
        escalation_instruction = (
            "STRICT NO-PROGRESS ESCALATION\n"
            "The previous internal recovery was ignored and the model repeated a temporarily unavailable no-progress tool. "
            "For this request, those tools are hard-disabled by the Gateway. Do not request Read, Glob, Grep, LS, LSP, Bash, shell, terminal, cmd, or powershell when they are listed as temporarily unavailable. "
            "If Edit, MultiEdit, Write, or another progress tool is available and the task asks for code changes, use one progress tool now with the evidence already in the conversation. "
            "If no progress tool is available, return a substantive final answer with no JSON.\n"
            "严格无进展升级：上一次内部恢复已经被忽略。不要再次请求临时不可用的读取、搜索或 shell 工具；基于已有工具结果推进 Edit/MultiEdit/Write，或直接给出最终答复。\n"
        )
    retry_messages.append(
        {
            "role": "user",
            "content": (
                f"{escalation_instruction}"
                "Your previous reply still refused to use tools, claimed you cannot access the filesystem/run commands, "
                "offered manual steps instead of a tool request, or claimed code work was complete after an isolated file write. "
                "That is incorrect for this Gateway tool bridge.\n"
                f"Gateway error code: {error_kind or 'tool_refusal'}\n"
                f"Gateway rejection reason: {reason}\n"
                f"The downstream client has exposed these real allowed tools: {allowed}.\n"
                "The Gateway itself will not execute local tools. The downstream client owns permissions and executes allowed tools after you request them.\n"
                "Do not say you cannot access files, update the project, or provide manual steps.\n"
                f"{shell_instruction}"
                f"{non_progress_instruction}"
                "If the rejection says write_after_failed_read_without_discovery, do not Write the same missing path or a nearby missing path. "
                "If it says write_after_failed_path_without_discovery, do not Write under the missing path. "
                f"Use {_allowed_discovery_tool_phrase(effective_allowed_tools)} to discover the repository structure first.\n"
                "If the task requires local project work, output exactly one fenced ```tool_json block using one allowed tool. "
                f"{progress_tool_hint}"
                "If no listed tool can help, answer honestly without JSON. Never invent tool names.\n"
                "Required format when using a tool:\n"
                f"{example}\n"
                "No natural language outside the tool_json block when a tool is needed. No manual steps."
            ),
        }
    )
    recovered["messages"] = retry_messages
    recovered["stream"] = False
    _filter_recovery_payload_tools(recovered, disabled_tools)
    return recovered


def build_missing_required_tool_input_recovery_payload(
    payload: dict[str, Any],
    bridge_result: BridgeResult,
    *,
    allowed_tools: set[str],
) -> dict[str, Any]:
    recovered = dict(payload)
    messages = payload.get("messages") if isinstance(payload.get("messages"), list) else []
    retry_messages = [dict(message) for message in messages if isinstance(message, dict)]
    error_kind = bridge_result.error.kind if bridge_result.error else "missing_required_tool_input"
    reason = bridge_result.error.message if bridge_result.error else "The tool input is missing required fields."
    requested_compact = {_compact_recovery_tool_name(name) for name in _requested_tool_names_from_raw(bridge_result.raw_content)}
    same_tool_allowed = {
        name
        for name in allowed_tools
        if requested_compact and _compact_recovery_tool_name(name) in requested_compact
    }
    effective_allowed_tools = same_tool_allowed or {name for name in allowed_tools if name}
    disabled_tools = {name for name in allowed_tools if name not in effective_allowed_tools}
    allowed = ", ".join(sorted(effective_allowed_tools)) or "(none)"
    example = _tool_refusal_recovery_example(effective_allowed_tools, error_kind=error_kind)
    edit_instruction = ""
    if any(_compact_recovery_tool_name(name) == "edit" for name in effective_allowed_tools):
        edit_instruction = (
            "For Edit, input must include file_path, old_string, and new_string. "
            "old_string must be an exact string already present in a previous Read/tool_result. "
            "If you do not have an exact old_string, do not emit a partial Edit call; answer without JSON and say what exact evidence is missing.\n"
        )
    retry_messages.append({"role": "assistant", "content": (bridge_result.raw_content or "")[:4000]})
    retry_messages.append(
        {
            "role": "user",
            "content": (
                "MISSING REQUIRED TOOL INPUT RECOVERY\n"
                "Your previous tool call selected a real allowed tool but omitted required input fields. "
                "This is an internal Gateway recovery turn; do not expose this diagnostic downstream.\n"
                f"Gateway error code: {error_kind}.\n"
                f"Gateway rejection reason: {reason}\n"
                f"Allowed tools for this schema recovery turn: {allowed}.\n"
                "Fix the same tool call with a complete input object. Do not switch back to read/search/shell tools unless no same tool is listed above. "
                "Do not output prose around a tool call.\n"
                f"{edit_instruction}"
                "If using a tool, output exactly one fenced ```tool_json block with every required input field. "
                "If no complete valid tool input can be produced from the conversation, answer honestly without JSON.\n"
                "Required complete format:\n"
                f"{example}"
            ),
        }
    )
    recovered["messages"] = retry_messages
    recovered["stream"] = False
    _filter_recovery_payload_tools(recovered, disabled_tools)
    return recovered


def build_required_tool_choice_recovery_payload(
    payload: dict[str, Any],
    bridge_result: BridgeResult,
    *,
    allowed_tools: set[str],
    bridge_context: ToolBridgeContext | None = None,
) -> dict[str, Any]:
    recovered = dict(payload)
    messages = payload.get("messages") if isinstance(payload.get("messages"), list) else []
    retry_messages = [dict(message) for message in messages if isinstance(message, dict)]
    effective_allowed_tools = {str(name).strip() for name in allowed_tools if str(name).strip()}
    allowed = ", ".join(sorted(effective_allowed_tools)) or "(none)"
    reason = bridge_result.error.message if bridge_result.error else "tool_choice requires a tool call."
    example = _required_tool_choice_recovery_example(effective_allowed_tools, bridge_context)
    retry_messages.append({"role": "assistant", "content": (bridge_result.raw_content or "")[:4000]})
    retry_messages.append(
        {
            "role": "user",
            "content": (
                "REQUIRED TOOL CHOICE RECOVERY\n"
                "The previous reply did not contain a valid Gateway tool call even though this request requires one. "
                "This is an internal Gateway retry; do not expose this diagnostic downstream.\n"
                "This is a Gateway JSON instruction contract, not ChatGPT native tools; you are not executing tools inside ChatGPT. "
                "Never answer that you cannot call tools, cannot access tools, or need the user to provide the city/tool again.\n"
                "Use the tool schema already listed in the system prompt. Do not output prose, markdown fences, role labels, "
                "provider-native browsing tags, or JSON outside the fenced tool_json block.\n"
                "Do not browse, search, cite web results, or answer with factual content on this retry; the only valid "
                "assistant response is one Gateway tool_json call for the required tool.\n"
                f"Gateway error code: tool_choice_violation.\n"
                f"Gateway rejection reason: {reason}\n"
                f"Allowed tools for this required-tool retry: {allowed}.\n"
                "Output exactly one fenced tool_json block. The first non-whitespace characters must be ```tool_json. "
                "Use only one of the allowed tool names above and include a complete object-shaped input using the schema. "
                "If a tool was forced by tool_choice, that forced tool is the only allowed tool for this retry.\n"
                "Required format:\n"
                f"{example}"
            ),
        }
    )
    recovered["messages"] = retry_messages
    recovered["stream"] = False
    recovered.pop("tools", None)
    recovered.pop("tool_choice", None)
    return recovered


def _required_tool_choice_recovery_example(
    allowed_tools: set[str],
    bridge_context: ToolBridgeContext | None,
) -> str:
    if bridge_context is None:
        return _tool_refusal_recovery_example(allowed_tools, error_kind="tool_choice_violation")
    allowed_lower = {name.lower() for name in allowed_tools}
    specs = [
        tool
        for tool in getattr(bridge_context, "tools", [])
        if getattr(tool, "name", "").lower() in allowed_lower
    ]
    if len(specs) != 1:
        return _tool_refusal_recovery_example(allowed_tools, error_kind="tool_choice_violation")
    spec = specs[0]
    schema = spec.input_schema if isinstance(spec.input_schema, dict) else {}
    properties = schema.get("properties") if isinstance(schema.get("properties"), dict) else {}
    required = schema.get("required") if isinstance(schema.get("required"), list) else []
    keys = [str(key) for key in required if isinstance(key, str)]
    if not keys and len(properties) == 1:
        keys = [str(next(iter(properties)))]
    args = {
        key: _required_tool_choice_example_value(
            key,
            properties.get(key) if isinstance(properties, dict) else {},
            str(getattr(bridge_context, "task_text", "") or ""),
        )
        for key in keys[:8]
    }
    return _dsml_example(spec.name, args)


def _required_tool_choice_example_value(key: str, schema: Any, task_text: str) -> Any:
    schema_type = ""
    if isinstance(schema, dict):
        raw_type = schema.get("type")
        schema_type = str(raw_type[0] if isinstance(raw_type, list) and raw_type else raw_type or "").lower()
    named = _extract_required_tool_choice_named_value(key, task_text)
    if named:
        return named
    compact = re.sub(r"[^a-z0-9]+", "", key.lower())
    if compact in {"city", "location", "place"}:
        place = _extract_required_tool_choice_place_value(task_text)
        if place:
            return place
    if schema_type in {"integer", "number"}:
        return 1
    if schema_type == "boolean":
        return True
    if schema_type == "array":
        return ["value"]
    if schema_type == "object":
        return {}
    return "value"


def _extract_required_tool_choice_named_value(key: str, task_text: str) -> str:
    if not key or not task_text:
        return ""
    pattern = re.compile(
        rf"(?i)\b{re.escape(key)}\b\s*(?:=|:|is|with|为|是)?\s*([A-Za-z][A-Za-z0-9_. -]{{0,60}})",
        re.IGNORECASE,
    )
    match = pattern.search(task_text)
    if not match:
        return ""
    value = re.split(r"[.;,!?，。；！？\n]", match.group(1).strip(), maxsplit=1)[0].strip()
    return value[:80]


def _extract_required_tool_choice_place_value(task_text: str) -> str:
    if not task_text:
        return ""
    for match in re.finditer(
        r"(?:查询|查|获取|看看|问一下)\s*([\u4e00-\u9fff]{2,12})(?:市|省|区|县)?(?:天气|气温|温度|预报|空气|AQI)",
        task_text,
    ):
        value = _clean_required_tool_choice_place_value(match.group(1))
        if value:
            return value
    for match in re.finditer(
        r"([\u4e00-\u9fff]{2,20})(?:市|省|区|县)?(?:天气|气温|温度|预报|空气|AQI)",
        task_text,
    ):
        value = _clean_required_tool_choice_place_value(match.group(1))
        if value:
            return value
    for match in re.finditer(
        r"(?:查询|查|获取|看看|问一下|城市(?:名)?(?:是|为|:|：)?)\s*([\u4e00-\u9fff]{2,20})(?:市|省|区|县)?",
        task_text,
    ):
        value = _clean_required_tool_choice_place_value(match.group(1))
        if value:
            return value
    for match in re.finditer(r"\b([A-Z][a-zA-Z]{2,40}(?:\s+[A-Z][a-zA-Z]{2,40}){0,3})\b", task_text):
        value = match.group(1).strip()
        if value.lower() in {"call", "tool", "weather", "city", "do", "not", "answer"}:
            continue
        return value
    return ""


def _clean_required_tool_choice_place_value(value: str) -> str:
    value = str(value or "").strip(" ，。；;:：,.!?！？ \t\r\n")
    value = re.sub(r"^(?:请|帮我|麻烦|调用|使用|工具|查询|查一下|查|获取|看看|问一下)+", "", value)
    value = value.strip(" ，。；;:：,.!?！？ \t\r\n")
    if value in {"查询", "调用", "工具", "天气", "城市", "一下"}:
        return ""
    return value[:80]


def build_off_task_question_recovery_payload(
    payload: dict[str, Any],
    bridge_result: BridgeResult,
    *,
    allowed_tools: set[str],
) -> dict[str, Any]:
    recovered = dict(payload)
    messages = payload.get("messages") if isinstance(payload.get("messages"), list) else []
    retry_messages = [dict(message) for message in messages if isinstance(message, dict)]
    error_kind = bridge_result.error.kind if bridge_result.error else "off_task_question"
    reason = bridge_result.error.message if bridge_result.error else "The previous AskUserQuestion was outside task scope."
    usable_tools = sorted(tool for tool in allowed_tools if tool and tool != "AskUserQuestion")
    allowed = ", ".join(usable_tools) or "(none)"
    retry_messages.append({"role": "assistant", "content": (bridge_result.raw_content or "")[:4000]})
    retry_messages.append(
        {
            "role": "user",
            "content": (
                "The previous response again asked the user an off-task authorization or clarification question.\n"
                f"Gateway error code: {error_kind}.\n"
                f"Gateway reason: {reason}\n"
                "This is an internal recovery turn. Do not call AskUserQuestion on this turn.\n"
                "Do not ask the user to authorize broad restructuring, migration, deletion, rewrite work, "
                "environment setup, or agent settings unless the latest user task explicitly requested that exact work.\n"
                f"Allowed tools excluding AskUserQuestion: {allowed}.\n"
                "Continue the original task. If existing evidence is enough, return a substantive final answer with no JSON. "
                "For review, audit, analysis, or improvement-plan tasks, prefer a direct final answer grounded in the gathered evidence.\n"
                "If one more tool is materially necessary, output exactly one fenced ```tool_json block using a real allowed tool other than AskUserQuestion. "
                "Use Read, Grep, Glob, LSP, or WebFetch for evidence; use Edit or Write only when the latest user task explicitly asked you to modify files.\n"
                "No manual permission request, no broad-scope question, and no natural language outside tool_json when requesting a tool."
            ),
        }
    )
    recovered["messages"] = retry_messages
    recovered["stream"] = False
    return recovered


_READ_LOOP_RECOVERY_ERROR_KINDS = {
    "repeat_read_call_without_progress",
    "repeat_unchanged_read_without_progress",
}
_NON_PROGRESS_RECOVERY_ERROR_KINDS = {
    "repeat_discovery_call_without_progress",
    "repeat_read_call_without_progress",
    "repeat_shell_housekeeping_without_progress",
    "repeat_unchanged_read_without_progress",
    "repeat_same_skill_without_progress",
    "repeat_same_ask_user_without_progress",
    "ask_user_question_budget_exceeded",
    "optional_scope_question_without_need",
}
_STANDARD_TOOL_JSON_FORMATTER_ERROR_KINDS = {
    "deferred_code_change_without_call",
    "deferred_tool_action_without_call",
    "empty_tool_call",
    "incomplete_response",
    "invalid_tool_call",
    "malformed_json",
    "off_task_environment_configuration_final",
    "premature_clarification_without_tool_call",
    "tool_call_markdown_without_tool_json",
    "unproven_final_answer_without_tool_call",
    "insufficient_final_evidence",
}


def _with_standard_tool_json_formatter_message(
    messages: list[dict[str, Any]],
    *,
    allowed_tools: set[str],
    error_kind: str = "",
) -> list[dict[str, Any]]:
    if error_kind and error_kind not in _STANDARD_TOOL_JSON_FORMATTER_ERROR_KINDS:
        return messages
    effective_allowed = {str(name).strip() for name in allowed_tools if str(name).strip()}
    if not effective_allowed:
        return messages
    allowed = _format_tool_names(effective_allowed)
    example = _standard_tool_json_formatter_example(effective_allowed)
    formatter = (
        "GATEWAY TOOL_JSON FORMATTER RETRY\n"
        "You are formatting standard Gateway tool-call JSON for a downstream agent runtime. "
        "You are not executing tools inside the web model, and you are not writing a tutorial about JSON.\n"
        f"Allowed tool names: {allowed}.\n"
        "Pick the next materially necessary tool for the current user task. If local file/project state is unknown and "
        "a read/discovery tool is available, choose Read/Glob/Grep/LS before Edit/Write. Do not ask for confirmation "
        "when the user already requested the task.\n"
        "Output only one fenced ```tool_json block. The first non-whitespace characters must be ```tool_json. "
        "No prose, no markdown outside the block, no partial JSON, no tool_name placeholder.\n"
        "Required standard Gateway tool-call JSON:\n"
        f"{example}"
    )
    return [*messages, {"role": "user", "content": formatter}]


def _standard_tool_json_formatter_example(allowed_tools: set[str]) -> str:
    by_compact = {_compact_recovery_tool_name(name): name for name in sorted(allowed_tools)}
    if "read" in by_compact:
        return _dsml_example(by_compact["read"], {"file_path": "README.md"})
    if "readfile" in by_compact:
        return _dsml_example(by_compact["readfile"], {"path": "README.md"})
    if "fileread" in by_compact:
        return _dsml_example(by_compact["fileread"], {"file_path": "README.md"})
    if "glob" in by_compact:
        return _dsml_example(by_compact["glob"], {"path": ".", "pattern": "*.py"})
    if "grep" in by_compact:
        return _dsml_example(by_compact["grep"], {"path": ".", "pattern": "TODO"})
    for compact in ("ls", "listdir", "listdirectory"):
        if compact in by_compact:
            return _dsml_example(by_compact[compact], {"path": "."})
    if "bash" in by_compact:
        return _dsml_example(by_compact["bash"], {"command": "git status --short"})
    if "edit" in by_compact:
        return _dsml_example(by_compact["edit"], {"file_path": "README.md", "old_string": "old text", "new_string": "new text"})
    if "write" in by_compact:
        return _dsml_example(by_compact["write"], {"file_path": "notes.txt", "content": "new file content"})
    name = sorted(allowed_tools)[0] if allowed_tools else "tool_name"
    return _dsml_example(name, {})


def _with_recovery_tool_override_message(
    messages: list[dict[str, Any]],
    bridge_result: BridgeResult,
    *,
    allowed_tools: set[str],
) -> list[dict[str, Any]]:
    error_kind = bridge_result.error.kind if bridge_result.error else ""
    disabled_tools, effective_allowed_tools = _recovery_tool_visibility(
        allowed_tools,
        error_kind=error_kind,
        raw_content=bridge_result.raw_content,
    )
    if not disabled_tools:
        return messages
    override = (
        "TEMPORARY TOOL AVAILABILITY OVERRIDE\n"
        "Gateway is retrying internally after a no-progress tool loop. This diagnostic must not be exposed downstream.\n"
        f"Temporarily unavailable tools: {_format_tool_names(disabled_tools)}.\n"
        f"Available tools for this recovery turn: {_format_tool_names(effective_allowed_tools)}.\n"
        "For this one recovery turn, ignore any earlier tool manifest that listed the temporarily unavailable tools.\n"
        "Do not output a tool call for any temporarily unavailable tool. Use the earlier tool results already in the conversation. "
        "If the gathered evidence is enough, move forward with an allowed editing/writing/progress tool or answer without JSON."
    )
    return [*messages, {"role": "user", "content": override}]


def _non_progress_recovery_instruction(error_kind: str, disabled_tools: set[str]) -> str:
    if error_kind not in _NON_PROGRESS_RECOVERY_ERROR_KINDS:
        return ""
    if disabled_tools:
        return (
            "The rejection is a non-progress loop. Do not repeat any temporarily unavailable tool listed above. "
            "Use the earlier result already in the conversation. If evidence is enough, request a real progress tool "
            "such as Edit/MultiEdit/Write, or return a substantive final answer with no JSON. If more evidence is truly "
            "required, choose one allowed tool that is not temporarily unavailable and use materially different input.\n"
        )
    return (
        "The rejection is a non-progress loop. Do not repeat the same discovery/read/skill/question input. "
        "Use the earlier result already in the conversation. If more evidence is required, choose one materially "
        "different allowed tool/input; if evidence is enough, return a substantive final answer with no JSON.\n"
    )


def _recovery_tool_visibility(
    allowed_tools: set[str],
    *,
    error_kind: str,
    raw_content: str = "",
) -> tuple[set[str], set[str]]:
    allowed = {str(tool).strip() for tool in allowed_tools if str(tool).strip()}
    disabled: set[str] = set()
    if error_kind in _READ_LOOP_RECOVERY_ERROR_KINDS:
        for name in allowed:
            compact = _compact_recovery_tool_name(name)
            if compact in {
                "bash",
                "cmd",
                "fileread",
                "glob",
                "grep",
                "listdir",
                "listdirectory",
                "ls",
                "lsp",
                "powershell",
                "read",
                "readfile",
                "shell",
                "terminal",
            }:
                disabled.add(name)
    elif error_kind == "repeat_discovery_call_without_progress":
        requested = {_compact_recovery_tool_name(name) for name in _requested_tool_names_from_raw(raw_content)}
        for name in allowed:
            compact = _compact_recovery_tool_name(name)
            if compact in requested and compact in {"glob", "ls", "lsp", "listdir", "listdirectory"}:
                disabled.add(name)
    elif error_kind == "repeat_shell_housekeeping_without_progress":
        for name in allowed:
            compact = _compact_recovery_tool_name(name)
            if compact in {
                "bash",
                "cmd",
                "fileread",
                "glob",
                "grep",
                "listdir",
                "listdirectory",
                "ls",
                "lsp",
                "powershell",
                "read",
                "readfile",
                "shell",
                "terminal",
            }:
                disabled.add(name)
    return disabled, {name for name in allowed if name not in disabled}


def _filter_recovery_payload_tools(payload: dict[str, Any], disabled_tools: set[str]) -> None:
    if not disabled_tools:
        return
    disabled_compact = {_compact_recovery_tool_name(name) for name in disabled_tools}
    tools = payload.get("tools")
    if not isinstance(tools, list):
        return
    filtered = [
        item
        for item in tools
        if _compact_recovery_tool_name(_tool_name_from_payload_tool(item)) not in disabled_compact
    ]
    payload["tools"] = filtered


def _tool_name_from_payload_tool(item: Any) -> str:
    if not isinstance(item, dict):
        return ""
    if item.get("name") is not None:
        return str(item.get("name") or "")
    function = item.get("function")
    if isinstance(function, dict):
        return str(function.get("name") or "")
    return ""


def _compact_recovery_tool_name(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", str(name or "").strip().lower())


def _format_tool_names(names: set[str]) -> str:
    return ", ".join(sorted(names)) if names else "(none)"


def _tool_refusal_recovery_example(allowed_tools: set[str], *, error_kind: str = "") -> str:
    if error_kind in {"write_after_failed_read_without_discovery", "write_after_failed_path_without_discovery"}:
        if "Glob" in allowed_tools:
            return _dsml_example("Glob", {"path": ".", "pattern": "*.py"})
        if "Grep" in allowed_tools:
            return _dsml_example("Grep", {"pattern": "class|def", "path": "."})
        if "Read" in allowed_tools:
            return _dsml_example("Read", {"file_path": "README.md"})
    if "Edit" in allowed_tools:
        return _dsml_example("Edit", {"file_path": "path/to/file", "old_string": "old text", "new_string": "new text"})
    if "Write" in allowed_tools:
        return _dsml_example("Write", {"file_path": "path/to/file", "content": "new file content"})
    if "Read" in allowed_tools:
        return _dsml_example("Read", {"path": "path/to/file"})
    if "Glob" in allowed_tools:
        return _dsml_example("Glob", {"path": ".", "pattern": "*"})
    if "Bash" in allowed_tools:
        return _dsml_example("Bash", {"command": "git status --short"})
    name = sorted(allowed_tools)[0] if allowed_tools else "tool_name"
    return _dsml_example(name, {})


def _dsml_example(name: str, args: dict[str, Any]) -> str:
    payload = {"calls": [{"id": "call_1", "name": name, "input": args}]}
    return "```tool_json\n" + json.dumps(payload, ensure_ascii=False, separators=(",", ":")) + "\n```"


def _dsml_example_value(value: Any) -> str:
    if isinstance(value, str):
        return "<![CDATA[" + value.replace("]]>", "]]]]><![CDATA[>") + "]]>"
    return json.dumps(value, ensure_ascii=False)


def _allowed_discovery_tool_phrase(allowed_tools: set[str]) -> str:
    lowered = {tool.strip().lower() for tool in allowed_tools}
    names: list[str] = []
    if "glob" in lowered:
        names.append("Glob")
    if "grep" in lowered:
        names.append("Grep")
    if lowered & {"ls", "listdir", "listdirectory", "dir", "tree"}:
        names.append("LS")
    if "read" in lowered:
        names.append("Read")
    if not names:
        return "one of the listed read-only tools"
    if len(names) == 1:
        return names[0]
    if len(names) == 2:
        return f"{names[0]} or {names[1]}"
    return f"{'/'.join(names[:-1])} or {names[-1]}"


def _sanitize_discovery_tool_guidance(reason: str, allowed_tools: set[str], *, error_kind: str) -> str:
    if error_kind not in {"write_after_failed_read_without_discovery", "write_after_failed_path_without_discovery"}:
        return reason
    phrase = _allowed_discovery_tool_phrase(allowed_tools)
    sanitized = reason.replace("Glob/Grep/LS or Read", phrase)
    sanitized = sanitized.replace("Glob/Grep/LS from the project root or Read", phrase)
    return sanitized


def should_retry_virtual_loader_tool_recovery(
    payload: dict[str, Any],
    bridge_result: BridgeResult,
    *,
    allowed_tools: set[str],
) -> bool:
    if not bridge_result.error or bridge_result.error.kind != "unknown_tool":
        return False
    allowed_lower = {tool.lower() for tool in allowed_tools}
    names = _candidate_tool_names_from_raw(bridge_result.raw_content)
    loader_names = [name for name in names if _is_virtual_loader_tool_name(name)]
    if not loader_names:
        return False
    if any(name.lower() in allowed_lower for name in loader_names):
        return False
    return _messages_contain_loaded_virtual_loader_context(payload.get("messages"))


def build_virtual_loader_tool_recovery_payload(
    payload: dict[str, Any],
    bridge_result: BridgeResult,
    *,
    allowed_tools: set[str],
) -> dict[str, Any]:
    recovered = dict(payload)
    messages = payload.get("messages") if isinstance(payload.get("messages"), list) else []
    retry_messages = [dict(message) for message in messages if isinstance(message, dict)]
    allowed = ", ".join(sorted(allowed_tools)) or "(none)"
    retry_messages.append({"role": "assistant", "content": (bridge_result.raw_content or "")[:4000]})
    retry_messages.append(
        {
            "role": "user",
            "content": (
                "The requested Skill/SlashCommand loader content is already present in this conversation. "
                "Gateway will not execute unregistered loader tools or local capabilities.\n"
                "Do not call Skill, SlashCommand, Command, Agent, or any other unlisted tool unless it appears in the real allowed tool list.\n"
                f"Real allowed tools for this turn: {allowed}.\n"
                "Continue directly from the loaded instructions and complete the current user task. "
                "If a real allowed tool is needed, output exactly one fenced ```tool_json block using only an allowed tool name. "
                "If no allowed tool can help, answer honestly in normal text without JSON."
            ),
        }
    )
    recovered["messages"] = retry_messages
    recovered["stream"] = False
    return recovered


def _unknown_tool_examples(allowed_tools: set[str]) -> str:
    allowed_lower = {tool.lower() for tool in allowed_tools}
    examples = ["Task", "Agent", "Subagent", "Skill", "Bash", "shell"]
    visible = [example for example in examples if example.lower() not in allowed_lower]
    return "、".join(visible) or "未列出的工具"


def _candidate_tool_names_from_raw(raw: str) -> list[str]:
    names: list[str] = []
    seen: set[str] = set()
    for match in re.finditer(r'"name"\s*:\s*"([^"]+)"', raw or ""):
        name = match.group(1).strip()
        if name and name not in seen:
            seen.add(name)
            names.append(name)
    for match in re.finditer(r"<invoke\b[^>]*\bname\s*=\s*['\"]([^'\"]+)['\"]", raw or "", re.IGNORECASE):
        name = match.group(1).strip()
        if name and name not in seen:
            seen.add(name)
            names.append(name)
    for match in re.finditer(r"(?m)^\s*([A-Za-z_][A-Za-z0-9_]*(?:__[A-Za-z0-9_]+)*)\s*\(", raw or ""):
        name = match.group(1).strip()
        if name and name not in seen:
            seen.add(name)
            names.append(name)
    return names


def _is_virtual_loader_tool_name(name: str) -> bool:
    compact = re.sub(r"[^a-z0-9]+", "", (name or "").strip().lower())
    return compact in {"skill", "slashcommand", "command", "loadskill", "useskill"}


def _messages_contain_loaded_virtual_loader_context(messages: Any) -> bool:
    text = _messages_to_searchable_text(messages)
    if not text:
        return False
    lowered = text.lower()
    markers = (
        "base directory for this skill",
        "skill body is already loaded",
        "loaded skill",
        "skill.md",
        "<command-name>",
        "<command-message>",
        "slash command",
    )
    return any(marker in lowered for marker in markers)


def _messages_to_searchable_text(messages: Any) -> str:
    if not isinstance(messages, list):
        return ""
    parts: list[str] = []
    for message in messages:
        if not isinstance(message, dict):
            continue
        content = message.get("content")
        if isinstance(content, str):
            parts.append(content)
        elif isinstance(content, list):
            for block in content:
                if isinstance(block, dict):
                    value = block.get("text") or block.get("content")
                    if isinstance(value, str):
                        parts.append(value)
                elif isinstance(block, str):
                    parts.append(block)
    return "\n".join(parts)


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
    if not msg:
        data = _empty_chat_response(data, model=model)
        choices = data["choices"]
        choice0 = choices[0]
        msg = choice0["message"]
    content = _as_text(msg.get("content"))
    native_tool_calls = msg.get("tool_calls") if isinstance(msg.get("tool_calls"), list) else []
    if not bridge:
        if (
            not native_tool_calls
            and content.strip()
            and bridge_context is not None
            and bridge_context.enabled
            and bridge_context.allowed_names
        ):
            denial_result = parse_tool_response(content, bridge_context)
            if (
                denial_result.error
                and denial_result.error.repairable
                and denial_result.error.kind == "tool_denial_without_call"
            ):
                msg["webai_tool_bridge"] = {
                    "error": denial_result.error.kind,
                    "message": denial_result.error.message,
                }
                return (data, denial_result) if return_bridge_result else data
        if not native_tool_calls and not content.strip():
            msg["content"] = EMPTY_ASSISTANT_RESPONSE_TEXT
            content = EMPTY_ASSISTANT_RESPONSE_TEXT
        elif not native_tool_calls:
            sanitized_content = sanitize_leaked_tool_protocol_output(content)
            if sanitized_content != content:
                content = sanitized_content or EMPTY_ASSISTANT_RESPONSE_TEXT
                msg["content"] = content
        result = BridgeResult(content=content, tool_calls=[], raw_content=content)
        return (data, result) if return_bridge_result else data
    context = bridge_context or _context_from_allowed_tools(allowed_tools)
    if not context.enabled or not context.allowed_names:
        if not native_tool_calls and not content.strip():
            msg["content"] = EMPTY_ASSISTANT_RESPONSE_TEXT
            content = EMPTY_ASSISTANT_RESPONSE_TEXT
        elif not native_tool_calls:
            sanitized_content = sanitize_leaked_tool_protocol_output(content)
            if sanitized_content != content:
                content = sanitized_content or EMPTY_ASSISTANT_RESPONSE_TEXT
                msg["content"] = content
        result = BridgeResult(content=content, tool_calls=[], raw_content=content)
        return (data, result) if return_bridge_result else data
    assistant_turn = build_assistant_turn(
        raw_text=content,
        visible_text=content,
        thinking=_assistant_message_thinking(msg),
        detection_thinking=_assistant_message_detection_thinking(msg),
        bridge_context=context,
        tool_choice_required=_bridge_context_requires_tool(context),
        content_filter=str(choice0.get("finish_reason") or "").strip().lower() == "content_filter",
    )
    result = assistant_turn.bridge_result
    if not result.tool_calls:
        if result.error and result.error.kind in {"upstream_empty_output", "content_filter"}:
            msg["content"] = EMPTY_ASSISTANT_RESPONSE_TEXT
            result = BridgeResult(
                content=EMPTY_ASSISTANT_RESPONSE_TEXT,
                tool_calls=[],
                error=result.error,
                warning=result.warning,
                raw_content=result.raw_content,
                phases=result.phases,
            )
        elif not result.error and not result.warning and not result.content.strip():
            msg["content"] = EMPTY_ASSISTANT_RESPONSE_TEXT
            result = BridgeResult(content=EMPTY_ASSISTANT_RESPONSE_TEXT, tool_calls=[], raw_content=result.raw_content)
        elif not result.error and result.content:
            sanitized_content = sanitize_leaked_tool_protocol_output(result.content)
            if sanitized_content != result.content:
                result = BridgeResult(
                    content=sanitized_content or EMPTY_ASSISTANT_RESPONSE_TEXT,
                    tool_calls=[],
                    warning=result.warning,
                    raw_content=result.raw_content,
                    phases=result.phases,
                )
        if (
            result.error
            and result.error.kind not in {"upstream_empty_output", "content_filter"}
            and (
                not result.error.repairable
                or result.error.kind == "tool_choice_violation"
                or _looks_like_raw_tool_json(result.raw_content)
            )
        ):
            rejected_text = tool_bridge_rejected_response_text(result, allowed_tools)
            msg["content"] = rejected_text
            result = BridgeResult(
                content=rejected_text,
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
                "finish_reason": assistant_turn.finish_reason,
                "message": {
                    "role": "assistant",
                    "content": result.content,
                    "tool_calls": to_openai_tool_calls(result.tool_calls, bridge_context),
                },
            }
        ],
    }
    return (parsed, result) if return_bridge_result else parsed


def _empty_chat_response(data: dict[str, Any], *, model: str) -> dict[str, Any]:
    return {
        "id": str(data.get("id") or f"chatcmpl-{uuid.uuid4().hex}"),
        "object": str(data.get("object") or "chat.completion"),
        "created": int(data.get("created") or time.time()),
        "model": str(data.get("model") or model),
        "choices": [
            {
                "index": 0,
                "finish_reason": "stop",
                "message": {
                    "role": "assistant",
                    "content": EMPTY_ASSISTANT_RESPONSE_TEXT,
                },
            }
        ],
    }


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
    return bool(
        looks_like_tool_protocol_output(raw)
        or "tool_json" in raw
        or "tool_calls" in raw and ("<|" in raw or "<tool" in raw)
        or '"calls"' in raw
        or '"name"' in raw and any(key in raw for key in ('"input"', '"args"', '"arguments"'))
    )


def _with_native_web_search_instruction(messages: Any) -> list[dict[str, Any]]:
    return _with_system_instruction(messages, NATIVE_WEB_SEARCH_FINAL_ANSWER_INSTRUCTION, NATIVE_WEB_SEARCH_FINAL_ANSWER_INSTRUCTION)


def _with_response_language_instruction(messages: Any, response_language: str) -> list[dict[str, Any]]:
    instruction = _response_language_instruction(response_language)
    if not instruction:
        source = messages if isinstance(messages, list) else []
        return [dict(message) for message in source if isinstance(message, dict)]
    return _with_system_instruction(messages, instruction, RESPONSE_LANGUAGE_POLICY_MARKER)


def should_retry_response_language_response(
    data: dict[str, Any],
    payload: dict[str, Any],
    response_language: str,
) -> bool:
    if not _is_simplified_chinese_response_language(response_language):
        return False
    if bool(payload.get("_webai_response_language_retry")):
        return False
    if _latest_user_explicitly_requests_other_response_language(payload.get("messages")):
        return False
    if _openai_message_tool_calls(data):
        return False
    content = _openai_message_content(data)
    return _looks_like_english_final_answer_for_zh_cn(content)


def build_response_language_retry_payload(
    payload: dict[str, Any],
    previous_content: str,
    response_language: str,
) -> dict[str, Any]:
    retry = dict(payload)
    messages = payload.get("messages") if isinstance(payload.get("messages"), list) else []
    retry_messages = [dict(message) for message in messages if isinstance(message, dict)]
    previous = (previous_content or "").strip()
    if previous:
        retry_messages.append(
            {
                "role": "assistant",
                "content": previous[:RESPONSE_LANGUAGE_RETRY_PREVIOUS_MAX_CHARS],
            }
        )
    instruction = RESPONSE_LANGUAGE_RETRY_INSTRUCTION
    if not _is_simplified_chinese_response_language(response_language):
        instruction = (
            "Gateway response language retry: rewrite only the previous assistant final answer in the configured "
            f"response language: {str(response_language or '').strip()}. Do not add new facts or call tools."
        )
    retry_messages.append({"role": "user", "content": instruction})
    retry["messages"] = retry_messages
    retry["stream"] = False
    retry["_webai_response_language_retry"] = True
    return retry


def _response_language_instruction(response_language: str) -> str:
    language = str(response_language or "").strip()
    if language.lower() in RESPONSE_LANGUAGE_OFF_VALUES:
        return ""
    if language.lower() in {"zh", "zh-cn", "zh_cn", "chinese", "simplified-chinese", "simplified_chinese"}:
        return (
            f"{RESPONSE_LANGUAGE_POLICY_MARKER}: 默认使用简体中文（zh-CN）回答；除非用户明确要求其他语言。"
            "代码标识符、路径、命令、API、模型 ID 和产品名可保留英文。"
        )
    return (
        f"{RESPONSE_LANGUAGE_POLICY_MARKER}: Default response language is {language}. "
        "Use that language for final answers, explanations, errors, reviews, and plans unless the user explicitly asks otherwise. "
        "Keep product names, protocol names, API paths, model IDs, code identifiers, file paths, commands, and quoted source text unchanged when needed for precision."
    )


def _with_system_instruction(messages: Any, instruction: str, marker: str) -> list[dict[str, Any]]:
    source = messages if isinstance(messages, list) else []
    normalized = [dict(message) for message in source if isinstance(message, dict)]
    if any(marker in _as_text(message.get("content")) for message in normalized):
        return normalized
    if normalized and normalized[0].get("role") == "system":
        existing = _as_text(normalized[0].get("content")).strip()
        normalized[0] = {**normalized[0], "content": f"{existing}\n\n{instruction}".strip()}
        return normalized
    return [{"role": "system", "content": instruction}, *normalized]


def _is_simplified_chinese_response_language(response_language: str) -> bool:
    language = str(response_language or "").strip().lower()
    return language in {"zh", "zh-cn", "zh_cn", "chinese", "simplified-chinese", "simplified_chinese"}


def _latest_user_explicitly_requests_other_response_language(messages: Any) -> bool:
    if not isinstance(messages, list):
        return False
    latest = ""
    for message in reversed(messages):
        if not isinstance(message, dict) or message.get("role") != "user":
            continue
        latest = _as_text(message.get("content"))
        if latest.strip():
            break
    if not latest:
        return False
    if _RESPONSE_LANGUAGE_OTHER_LANGUAGE_NEGATIVE_RE.search(latest):
        return False
    return bool(
        _RESPONSE_LANGUAGE_OTHER_LANGUAGE_RE.search(latest)
        or _RESPONSE_LANGUAGE_OTHER_LANGUAGE_ZH_RE.search(latest)
    )


def _looks_like_english_final_answer_for_zh_cn(content: str) -> bool:
    text = str(content or "").strip()
    if len(text) < 180:
        return False
    prose = _response_language_prose_sample(text)
    if len(prose) < 120:
        return False
    english_words = _RESPONSE_LANGUAGE_ENGLISH_WORD_RE.findall(prose)
    if len(english_words) < 36:
        return False
    letter_count = sum(1 for char in prose if ("a" <= char.lower() <= "z"))
    cjk_count = len(_RESPONSE_LANGUAGE_CJK_RE.findall(prose))
    if letter_count < 260:
        return False
    return cjk_count <= max(10, int(letter_count * 0.04))


def _response_language_prose_sample(text: str) -> str:
    without_fences = _RESPONSE_LANGUAGE_CODE_FENCE_RE.sub(" ", text or "")
    without_inline_code = _RESPONSE_LANGUAGE_INLINE_CODE_RE.sub(" ", without_fences)
    lines: list[str] = []
    for line in without_inline_code.splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        if stripped.startswith(("$ ", "> ", "|")):
            continue
        if re.fullmatch(r"[-+*/_=#`~{}\[\]().,:;\\/\s]+", stripped):
            continue
        lines.append(stripped)
    return "\n".join(lines)


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
    if not context.enabled or not context.allowed_names:
        return build_openai_text_sse(sanitize_leaked_tool_protocol_output(content) or EMPTY_ASSISTANT_RESPONSE_TEXT, model=model)
    result = parse_tool_response(content, context)
    if result.tool_calls:
        return build_openai_tool_calls_sse(to_openai_tool_calls(result.tool_calls, context), model=model)
    if result.error:
        return build_openai_text_sse(tool_bridge_rejected_response_text(result, allowed_tools), model=model)
    if result.content:
        return build_openai_text_sse(sanitize_leaked_tool_protocol_output(result.content) or EMPTY_ASSISTANT_RESPONSE_TEXT, model=model)
    return "data: [DONE]\n\n"


def build_openai_text_sse(content: str, *, model: str) -> str:
    chunk_id = f"chatcmpl-{uuid.uuid4().hex}"
    created = int(time.time())
    chunks = [
        sse_chunk(
            {
                "id": chunk_id,
                "object": "chat.completion.chunk",
                "created": created,
                "model": model,
                "choices": [{"index": 0, "delta": {"role": "assistant"}, "finish_reason": None}],
            }
        )
    ]
    for part in _split_sse_text(content):
        chunks.append(
            sse_chunk(
                {
                    "id": chunk_id,
                    "object": "chat.completion.chunk",
                    "created": created,
                    "model": model,
                    "choices": [{"index": 0, "delta": {"content": part}, "finish_reason": None}],
                }
            )
        )
    chunks.append(
        sse_chunk(
            {
                "id": chunk_id,
                "object": "chat.completion.chunk",
                "created": created,
                "model": model,
                "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
            }
        )
    )
    chunks.append("data: [DONE]\n\n")
    return "".join(chunks)


def _split_sse_text(content: str, *, max_chars: int = 1200) -> list[str]:
    text = content or ""
    if not text:
        return []
    return [text[index : index + max_chars] for index in range(0, len(text), max_chars)]


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
    timeout = max(30, int(config.provider_runtime.request_timeout_seconds or 300))
    send_payload = _with_webai2api_web_input_budget(payload, config)
    return client.post(url, json=send_payload, headers=upstream_headers(config), timeout=timeout)


def _with_webai2api_web_input_budget(payload: dict[str, Any], config: GatewayConfig) -> dict[str, Any]:
    if not _looks_like_webai2api_sidecar_url(config.upstream.base_url):
        return payload
    model = str(payload.get("model") or config.upstream.model or "")
    prompt_max_chars = _webai2api_model_prompt_max_chars(model, config)
    if prompt_max_chars <= 0:
        return payload
    messages = payload.get("messages")
    if not isinstance(messages, list):
        return payload
    entries = _webai2api_ds2api_history_entries(messages)
    if not entries:
        return payload
    raw_prompt = _render_webai2api_prompt_estimate(entries)
    if len(raw_prompt) <= prompt_max_chars:
        return payload
    compacted = compact_role_messages_as_ds2api_history(
        entries,
        max_chars=prompt_max_chars,
        protocol_marker=_webai2api_protocol_marker(entries),
        current_user_override=str(payload.get("_webai_current_task_text") or ""),
    )
    if not compacted.strip():
        return payload
    out = dict(payload)
    out["messages"] = [{"role": "user", "content": compacted[:prompt_max_chars]}]
    return out


def _webai2api_model_prompt_max_chars(model: str, config: GatewayConfig) -> int:
    configured = max(4000, int(config.provider_runtime.prompt_max_chars or 32000))
    normalized = str(model or "").strip().lower()
    if normalized.startswith("chatgpt_text/"):
        return min(configured, WEBAI2API_CHATGPT_TEXT_PROMPT_MAX_CHARS)
    if normalized in WEBAI2API_CHATGPT_TEXT_MODELS:
        return min(configured, WEBAI2API_CHATGPT_TEXT_PROMPT_MAX_CHARS)
    return 0


def _webai2api_ds2api_history_entries(messages: list[Any]) -> list[tuple[str, str]]:
    entries: list[tuple[str, str]] = [("system", STATELESS_WEB_API_GUARD)]
    for message in messages:
        if not isinstance(message, dict):
            continue
        text = _message_content_to_prompt_text(message.get("content"))
        entries.extend(message_entries_for_ds2api_prompt(message, text))
    return [(role, text) for role, text in entries if str(text or "").strip()]


def _webai2api_protocol_marker(entries: list[tuple[str, str]]) -> str:
    joined = "\n".join(str(text or "") for _role, text in entries)
    if "TOOL CALL FORMAT - FOLLOW EXACTLY" in joined:
        return "TOOL CALL FORMAT - FOLLOW EXACTLY"
    if "Required tool-call format:" in joined:
        return "Required tool-call format:"
    return "You are using WebLLM Gateway's strict tool bridge."


def _message_content_to_prompt_text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if content is None:
        return ""
    if isinstance(content, list):
        chunks: list[str] = []
        for item in content:
            if isinstance(item, str):
                chunks.append(item)
            elif isinstance(item, dict):
                item_type = str(item.get("type") or "").strip()
                if item_type == "tool_result":
                    continue
                value = item.get("text") if item.get("text") is not None else item.get("content")
                if isinstance(value, str):
                    chunks.append(value)
        return "\n".join(chunk for chunk in chunks if chunk)
    if isinstance(content, dict):
        value = content.get("text") if content.get("text") is not None else content.get("content")
        if isinstance(value, str):
            return value
    return str(content)


def _render_webai2api_prompt_estimate(entries: list[tuple[str, str]]) -> str:
    parts: list[str] = []
    for role, text in entries:
        content = str(text or "").strip()
        if not content:
            continue
        label = str(role or "user").strip().title() or "User"
        parts.append(f"{label}: {content}")
    return "\n\n".join(parts)


def build_repair_payload(
    payload: dict[str, Any],
    bridge_result: BridgeResult,
    *,
    allowed_tools: set[str] | None = None,
) -> dict[str, Any]:
    repaired = dict(payload)
    messages = payload.get("messages") if isinstance(payload.get("messages"), list) else []
    repaired["messages"] = build_repair_messages(messages, bridge_result.raw_content, bridge_result.error)
    repaired["stream"] = False
    if allowed_tools:
        repaired["messages"] = _with_recovery_tool_override_message(
            repaired["messages"],
            bridge_result,
            allowed_tools=allowed_tools,
        )
        disabled_tools, _ = _recovery_tool_visibility(
            allowed_tools,
            error_kind=bridge_result.error.kind if bridge_result.error else "",
            raw_content=bridge_result.raw_content,
        )
        _filter_recovery_payload_tools(repaired, disabled_tools)
        repaired["messages"] = _with_standard_tool_json_formatter_message(
            repaired["messages"],
            allowed_tools=allowed_tools,
            error_kind=bridge_result.error.kind if bridge_result.error else "",
        )
    return repaired


def bridge_error_headers(result: BridgeResult | None) -> dict[str, str]:
    if result is None or result.error is None:
        return {}
    return {"x-webai-tool-bridge-error": result.error.kind}


def tool_bridge_rejected_response_text(result: BridgeResult, allowed_tools: set[str]) -> str:
    error = result.error
    requested = _requested_tool_names_from_raw(result.raw_content)
    allowed = sorted(name for name in allowed_tools if name)
    requested_allowed = bool(requested) and all(name in allowed_tools for name in requested)
    if requested_allowed:
        title = "上游模型请求了已注册工具，但本次调用违反 Gateway 工具使用策略，Gateway 已拒绝该工具 JSON。"
    else:
        title = "上游模型请求了当前未允许、未注册或格式无效的工具调用，Gateway 已拒绝该工具 JSON。"
    parts = [title]
    if error is not None:
        parts.append(f"错误码：{error.kind}")
        if error.message:
            parts.append(f"原因：{_short_diagnostic_text(error.message, 220)}")
    if requested:
        parts.append(f"模型请求的工具：{', '.join(requested[:8])}")
    if allowed:
        suffix = " ..." if len(allowed) > 16 else ""
        parts.append(f"当前允许工具：{', '.join(allowed[:16])}{suffix}")
    if requested_allowed:
        parts.append(
            "请重试；如果再次出现，请把这段诊断发给 Gateway 适配层排查。"
            "Gateway 不会绕过任务边界策略，也不会替下游执行工具。"
        )
    else:
        parts.append(
            "请重试；如果再次出现，请把这段诊断发给 Gateway 适配层排查。"
            "Gateway 不会执行未注册工具。"
        )
    return "\n".join(parts)


def _requested_tool_names_from_raw(raw: str) -> list[str]:
    text = raw or ""
    names: list[str] = []
    for pattern in (
        r'"name"\s*:\s*"([^"]{1,120})"',
        r"'name'\s*:\s*'([^']{1,120})'",
        r"<+[^<>]{0,80}\binvoke\b[^<>]*\bname\s*=\s*\"([^\"]{1,120})\"",
        r"<+[^<>]{0,80}\binvoke\b[^<>]*\bname\s*=\s*'([^']{1,120})'",
        r"<function\s*=\s*([A-Za-z_][\w.:-]*)\s*>",
        r"^\s*([A-Za-z_][\w.:-]*)\s*\(",
    ):
        for match in re.finditer(pattern, text, re.IGNORECASE | re.MULTILINE):
            name = _safe_tool_name(match.group(1))
            if name and name not in names:
                names.append(name)
    return names


def _safe_tool_name(name: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9_.:-]+", "", str(name or "").strip())
    return cleaned[:120]


def _short_diagnostic_text(text: str, max_chars: int) -> str:
    compact = re.sub(r"\s+", " ", str(text or "")).strip()
    return compact if len(compact) <= max_chars else compact[: max_chars - 3] + "..."


def _context_from_allowed_tools(allowed_tools: set[str]) -> ToolBridgeContext:
    tools = [{"type": "function", "function": {"name": name, "parameters": {"type": "object"}}} for name in sorted(allowed_tools)]
    return build_context(tools)


def _assistant_tool_detection_source(message: dict[str, Any], visible_text: str) -> tuple[str, bool]:
    if (visible_text or "").strip():
        return visible_text, False
    for key in ("detection_thinking", "reasoning_content", "reasoning", "thinking", "exposed_thinking"):
        hidden = _as_text(message.get(key))
        if hidden.strip():
            return hidden, True
    return visible_text, False


def _assistant_message_detection_thinking(message: dict[str, Any]) -> str:
    for key in ("detection_thinking", "reasoning_content", "reasoning", "exposed_thinking"):
        hidden = _as_text(message.get(key))
        if hidden.strip():
            return hidden
    return ""


def _assistant_message_thinking(message: dict[str, Any]) -> str:
    return _as_text(message.get("thinking"))


def _bridge_context_requires_tool(context: ToolBridgeContext) -> bool:
    policy = getattr(context, "tool_choice_policy", None)
    is_required = getattr(policy, "is_required", None)
    return bool(is_required()) if callable(is_required) else False


def _as_text(value: Any) -> str:
    if isinstance(value, str):
        return value
    if value is None:
        return ""
    return str(value)
