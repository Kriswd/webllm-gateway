from __future__ import annotations

import json
import time
import uuid
from typing import Any

from webai_gateway.model_ids import normalize_model_id
from webai_gateway.tool_bridge import normalize_anthropic_tools


def anthropic_body_to_openai(
    body: dict[str, Any],
    *,
    tool_call_registry: dict[str, dict[str, Any]] | None = None,
) -> dict[str, Any]:
    messages = body.get("messages")
    if not isinstance(messages, list):
        raise ValueError("Anthropic messages must be a list")
    system = body.get("system")
    out_messages: list[dict[str, Any]] = []
    if system:
        out_messages.append({"role": "system", "content": _anthropic_system_to_text(system)})
    tool_call_names: dict[str, str] = {}
    for message in messages:
        if not isinstance(message, dict):
            raise ValueError("Anthropic message must be an object")
        role = str(message.get("role") or "").strip()
        if role not in {"user", "assistant"}:
            raise ValueError(f"Unsupported Anthropic role: {role}")
        converted = _anthropic_message_to_openai(
            message,
            tool_call_names=tool_call_names,
            tool_call_registry=tool_call_registry,
        )
        out_messages.extend(converted)
    openai_body: dict[str, Any] = {
        "model": normalize_model_id(body.get("model")),
        "messages": out_messages,
        "tools": normalize_anthropic_tools(body.get("tools")),
    }
    if "tool_choice" in body:
        openai_body["tool_choice"] = _anthropic_tool_choice_to_openai(body.get("tool_choice"))
    if "temperature" in body:
        openai_body["temperature"] = body.get("temperature")
    if "top_p" in body:
        openai_body["top_p"] = body.get("top_p")
    if "metadata" in body:
        openai_body["metadata"] = body.get("metadata")
    if "stop_sequences" in body:
        openai_body["stop"] = body.get("stop_sequences")
    if "max_tokens" in body:
        openai_body["max_tokens"] = body.get("max_tokens")
    return openai_body


def anthropic_count_tokens(body: dict[str, Any]) -> dict[str, int]:
    normalized = _strip_cache_control(
        {
            "system": body.get("system"),
            "messages": body.get("messages"),
            "tools": body.get("tools"),
            "tool_choice": body.get("tool_choice"),
        }
    )
    raw = json.dumps(normalized, ensure_ascii=False, separators=(",", ":"))
    return {"input_tokens": max(1, round(len(raw) / 4))}


def openai_response_to_anthropic(data: dict[str, Any], *, original_model: str) -> dict[str, Any]:
    choices = data.get("choices") if isinstance(data.get("choices"), list) else []
    choice0 = choices[0] if choices and isinstance(choices[0], dict) else {}
    message = choice0.get("message") if isinstance(choice0.get("message"), dict) else {}
    content_blocks: list[dict[str, Any]] = []
    tool_calls = message.get("tool_calls") if isinstance(message.get("tool_calls"), list) else []
    if tool_calls:
        for call in tool_calls:
            if not isinstance(call, dict):
                continue
            fn = call.get("function") if isinstance(call.get("function"), dict) else {}
            name = str(fn.get("name") or "").strip()
            if not name:
                continue
            raw_args = fn.get("arguments")
            try:
                args = json.loads(raw_args) if isinstance(raw_args, str) and raw_args.strip() else {}
            except Exception:
                args = {}
            if not isinstance(args, dict):
                args = {}
            content_blocks.append(
                {
                    "type": "tool_use",
                    "id": _anthropic_tool_use_id(call.get("id"), len(content_blocks)),
                    "name": name,
                    "input": args,
                }
            )
        stop_reason = "tool_use"
    else:
        text = _as_text(message.get("content"))
        content_blocks.append({"type": "text", "text": text})
        stop_reason = _finish_reason_to_anthropic(str(choice0.get("finish_reason") or "stop"))
    return {
        "id": str(data.get("id") or f"msg_{uuid.uuid4().hex}"),
        "type": "message",
        "role": "assistant",
        "model": original_model,
        "content": content_blocks,
        "stop_reason": stop_reason,
        "stop_sequence": None,
        "usage": data.get("usage") if isinstance(data.get("usage"), dict) else {"input_tokens": 0, "output_tokens": 0},
        "created": int(data.get("created") or time.time()),
    }


def anthropic_response_to_sse(message: dict[str, Any]) -> str:
    chunks: list[str] = []

    def emit(event_name: str, payload: dict[str, Any]) -> None:
        chunks.append(
            "event: "
            + event_name
            + "\n"
            + "data: "
            + json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
            + "\n\n"
        )

    usage = message.get("usage") if isinstance(message.get("usage"), dict) else {}
    start_message = {
        "id": str(message.get("id") or f"msg_{uuid.uuid4().hex}"),
        "type": "message",
        "role": "assistant",
        "model": str(message.get("model") or ""),
        "content": [],
        "stop_reason": None,
        "stop_sequence": None,
        "usage": {
            "input_tokens": int(usage.get("input_tokens") or 0),
            "output_tokens": 0,
        },
    }
    emit("message_start", {"type": "message_start", "message": start_message})

    blocks = message.get("content") if isinstance(message.get("content"), list) else []
    if not blocks:
        blocks = [{"type": "text", "text": ""}]
    for index, block in enumerate(blocks):
        if not isinstance(block, dict):
            continue
        block_type = str(block.get("type") or "").strip()
        if block_type == "tool_use":
            input_value = block.get("input") if isinstance(block.get("input"), dict) else {}
            emit(
                "content_block_start",
                {
                    "type": "content_block_start",
                    "index": index,
                    "content_block": {
                        "type": "tool_use",
                        "id": str(block.get("id") or f"toolu_{index + 1}"),
                        "name": str(block.get("name") or ""),
                        "input": {},
                    },
                },
            )
            partial_json = json.dumps(input_value, ensure_ascii=False, separators=(",", ":"))
            if partial_json != "{}":
                emit(
                    "content_block_delta",
                    {
                        "type": "content_block_delta",
                        "index": index,
                        "delta": {"type": "input_json_delta", "partial_json": partial_json},
                    },
                )
            emit("content_block_stop", {"type": "content_block_stop", "index": index})
            continue

        text = _as_text(block.get("text"))
        emit(
            "content_block_start",
            {
                "type": "content_block_start",
                "index": index,
                "content_block": {"type": "text", "text": ""},
            },
        )
        if text:
            emit(
                "content_block_delta",
                {
                    "type": "content_block_delta",
                    "index": index,
                    "delta": {"type": "text_delta", "text": text},
                },
            )
        emit("content_block_stop", {"type": "content_block_stop", "index": index})

    emit(
        "message_delta",
        {
            "type": "message_delta",
            "delta": {
                "stop_reason": str(message.get("stop_reason") or "end_turn"),
                "stop_sequence": message.get("stop_sequence"),
            },
            "usage": {"output_tokens": int(usage.get("output_tokens") or 0)},
        },
    )
    emit("message_stop", {"type": "message_stop"})
    return "".join(chunks)


def _anthropic_message_to_openai(
    message: dict[str, Any],
    *,
    tool_call_names: dict[str, str],
    tool_call_registry: dict[str, dict[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    role = str(message.get("role") or "").strip()
    content = message.get("content")
    if isinstance(content, str):
        return [{"role": role, "content": content}]
    if not isinstance(content, list):
        raise ValueError("Anthropic message content must be a string or block list")
    text_parts: list[str] = []
    content_parts: list[dict[str, Any]] = []
    assistant_tool_calls: list[dict[str, Any]] = []
    out: list[dict[str, Any]] = []
    for block in content:
        if not isinstance(block, dict):
            raise ValueError("Anthropic content block must be an object")
        block_type = str(block.get("type") or "").strip()
        if block_type == "text":
            text = _as_text(block.get("text"))
            text_parts.append(text)
            content_parts.append({"type": "text", "text": text})
            continue
        if block_type == "image" and role == "user":
            image_part = _anthropic_image_to_openai(block)
            if image_part:
                content_parts.append(image_part)
            continue
        if block_type == "document" and role == "user":
            document_part = _anthropic_document_to_openai(block)
            if document_part:
                content_parts.append(document_part)
            continue
        if block_type in {"thinking", "redacted_thinking"} and role == "assistant":
            continue
        if block_type == "tool_use" and role == "assistant":
            tool_id = str(block.get("id") or f"toolu_{len(assistant_tool_calls) + 1}")
            name = str(block.get("name") or "").strip()
            if not name:
                raise ValueError("Anthropic tool_use block missing name")
            input_value = block.get("input")
            args = input_value if isinstance(input_value, dict) else {}
            tool_call_names[tool_id] = name
            assistant_tool_calls.append(
                {
                    "id": tool_id,
                    "type": "function",
                    "function": {"name": name, "arguments": json.dumps(args, ensure_ascii=False)},
                }
            )
            continue
        if block_type == "tool_result" and role == "user":
            tool_use_id = str(block.get("tool_use_id") or "").strip()
            if not tool_use_id:
                raise ValueError("Anthropic tool_result block missing tool_use_id")
            registered_call = _registered_tool_call(tool_call_registry, tool_use_id)
            if registered_call and tool_use_id not in tool_call_names:
                out.append(
                    {
                        "role": "assistant",
                        "content": "",
                        "tool_calls": [registered_call],
                    }
                )
                function = registered_call.get("function") if isinstance(registered_call.get("function"), dict) else {}
                tool_call_names[tool_use_id] = str(function.get("name") or "tool")
            content_text = _anthropic_tool_result_to_text(block.get("content"))
            converted_tool = {
                "role": "tool",
                "tool_call_id": tool_use_id,
                "name": tool_call_names.get(tool_use_id, "tool"),
                "content": content_text,
            }
            if isinstance(block.get("is_error"), bool):
                converted_tool["is_error"] = block["is_error"]
            out.append(converted_tool)
            continue
        raise ValueError(f"Unsupported Anthropic content block: {block_type}")
    if role == "assistant" and assistant_tool_calls:
        out.insert(
            0,
            {
                "role": "assistant",
                "content": "\n".join(part for part in text_parts if part).strip(),
                "tool_calls": assistant_tool_calls,
            },
        )
        return out
    if content_parts:
        has_non_text = any(part.get("type") != "text" for part in content_parts)
        content_value: Any = content_parts if has_non_text else "\n".join(part for part in text_parts if part).strip()
        out.insert(0, {"role": role, "content": content_value})
    return out


def _registered_tool_call(registry: dict[str, dict[str, Any]] | None, tool_use_id: str) -> dict[str, Any] | None:
    if not registry or not tool_use_id:
        return None
    raw = registry.get(tool_use_id)
    if not isinstance(raw, dict):
        return None
    function = raw.get("function") if isinstance(raw.get("function"), dict) else {}
    name = str(function.get("name") or "").strip()
    if not name:
        return None
    arguments = function.get("arguments")
    if not isinstance(arguments, str):
        arguments = "{}"
    return {
        "id": str(raw.get("id") or tool_use_id),
        "type": "function",
        "function": {
            "name": name,
            "arguments": arguments,
        },
    }


def _anthropic_system_to_text(system: Any) -> str:
    if isinstance(system, str):
        return system
    if isinstance(system, list):
        parts = []
        for block in system:
            if not isinstance(block, dict) or block.get("type") != "text":
                raise ValueError("Only text system blocks are supported")
            parts.append(_as_text(block.get("text")))
        return "\n".join(parts)
    raise ValueError("Anthropic system must be a string or text block list")


def _anthropic_image_to_openai(block: dict[str, Any]) -> dict[str, Any] | None:
    source = block.get("source") if isinstance(block.get("source"), dict) else {}
    source_type = str(source.get("type") or "").strip()
    if source_type == "base64":
        media_type = str(source.get("media_type") or "image/png")
        data = _as_text(source.get("data"))
        if not data:
            return None
        return {"type": "image_url", "image_url": {"url": f"data:{media_type};base64,{data}"}}
    if source_type == "url":
        url = _as_text(source.get("url"))
        return {"type": "image_url", "image_url": {"url": url}} if url else None
    return None


def _anthropic_document_to_openai(block: dict[str, Any]) -> dict[str, Any] | None:
    source = block.get("source") if isinstance(block.get("source"), dict) else {}
    source_type = str(source.get("type") or "").strip()
    title = str(block.get("title") or block.get("name") or "document")
    media_type = str(source.get("media_type") or "application/octet-stream")
    if source_type == "base64":
        data = _as_text(source.get("data"))
        if not data:
            return None
        return {
            "type": "file",
            "file": {
                "filename": title,
                "file_data": f"data:{media_type};base64,{data}",
            },
        }
    if source_type == "url":
        url = _as_text(source.get("url"))
        if not url:
            return None
        return {"type": "file", "file": {"filename": title, "file_url": url}}
    return None


def _anthropic_tool_choice_to_openai(value: Any) -> Any:
    if not isinstance(value, dict):
        return value
    choice_type = str(value.get("type") or "").strip()
    if choice_type == "auto":
        return "auto"
    if choice_type == "any":
        return "required"
    if choice_type == "tool":
        return {"type": "function", "function": {"name": str(value.get("name") or "")}}
    return value


def _anthropic_tool_use_id(value: Any, index: int) -> str:
    raw = str(value or "").strip()
    if raw.startswith("toolu_"):
        return raw
    safe = "".join(char if char.isalnum() or char in {"_", "-"} else "_" for char in raw).strip("_")
    if not safe:
        safe = str(index + 1)
    return f"toolu_{safe[:64]}"


def _anthropic_tool_result_to_text(value: Any) -> str:
    if isinstance(value, str):
        return value
    if isinstance(value, list):
        parts: list[str] = []
        for item in value:
            text = _anthropic_tool_result_part_to_text(item)
            if text:
                parts.append(text)
        return "\n".join(parts)
    if isinstance(value, dict):
        return _anthropic_tool_result_part_to_text(value)
    return _as_text(value)


def _anthropic_tool_result_part_to_text(value: Any) -> str:
    if not isinstance(value, dict):
        return _as_text(value)
    block_type = str(value.get("type") or "").strip()
    if block_type == "text":
        return _as_text(value.get("text"))
    if block_type in {"image", "document"}:
        return _anthropic_media_block_summary(block_type, value)
    return json.dumps(_redact_tool_result_payload(value), ensure_ascii=False, separators=(",", ":"))


def _anthropic_media_block_summary(block_type: str, value: dict[str, Any]) -> str:
    source = value.get("source") if isinstance(value.get("source"), dict) else {}
    media_type = str(source.get("media_type") or value.get("media_type") or "unknown")
    source_type = str(source.get("type") or value.get("source_type") or "unknown")
    data = source.get("data") if isinstance(source, dict) else None
    data_suffix = f", data_length={len(data)}" if isinstance(data, str) else ""
    if block_type == "document":
        title = str(value.get("title") or value.get("name") or "document")
        return f"[document: title={title}, media_type={media_type}, source={source_type}{data_suffix}]"
    return f"[image: media_type={media_type}, source={source_type}{data_suffix}]"


def _redact_tool_result_payload(value: Any) -> Any:
    if isinstance(value, dict):
        redacted: dict[str, Any] = {}
        for key, item in value.items():
            if key in {"data", "file_data", "base64"} and isinstance(item, str):
                redacted[key] = f"[omitted {len(item)} chars]"
            else:
                redacted[key] = _redact_tool_result_payload(item)
        return redacted
    if isinstance(value, list):
        return [_redact_tool_result_payload(item) for item in value]
    return value


def _strip_cache_control(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: _strip_cache_control(item) for key, item in value.items() if key != "cache_control"}
    if isinstance(value, list):
        return [_strip_cache_control(item) for item in value]
    return value


def _finish_reason_to_anthropic(finish_reason: str) -> str:
    reason = finish_reason.strip().lower()
    if reason == "length":
        return "max_tokens"
    if reason == "tool_calls":
        return "tool_use"
    return "end_turn"


def _as_text(value: Any) -> str:
    if isinstance(value, str):
        return value
    if value is None:
        return ""
    return str(value)
