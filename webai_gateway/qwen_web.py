from __future__ import annotations

import json
import re
import time
import uuid
from collections.abc import Callable, Iterable
from typing import Any

import httpx

from .model_ids import normalize_model_id
from .prompt_compaction import (
    STATELESS_WEB_API_GUARD,
    compact_role_messages_as_ds2api_history,
    compact_web_prompt,
    message_entries_for_ds2api_prompt,
    prompt_preserved_task_state_diagnostics,
    web_prompt_history_role,
)


QWEN_MODEL_PREFIX = "qwen-web/"
QWEN_BASE_URL = "https://chat.qwen.ai"
QWEN_MODELS_PATH = "/api/v2/models/"
DEFAULT_QWEN_REQUEST_TIMEOUT_SECONDS = 300
DEFAULT_QWEN_PROMPT_MAX_CHARS = 32000
QWEN_TOOL_BRIDGE_RUNAWAY_OUTPUT_CHARS = 12000
_QWEN_WEB_METADATA_KEYS = {
    "action",
    "chat_id",
    "event",
    "id",
    "message_id",
    "parent_id",
    "phase",
    "status",
    "step",
    "task",
    "timestamp",
    "title",
    "type",
    "uuid",
}
_QWEN_WEB_METADATA_SIGNAL_KEYS = {"event", "phase", "status", "step", "task", "title", "type"}
_QWEN_WEB_FINAL_CONTENT_KEYS = {"answer", "calls", "content", "message", "text", "tool_calls"}
QWEN_WEB_METADATA_RETRY_INSTRUCTION = (
    "\n\n[WebAI Gateway API mode retry]\n"
    "The previous Qwen Web response contained only workflow metadata, such as {\"title\": ...}, "
    "or did not contain a final answer. Do not output JSON metadata, phase titles, task status, or "
    "artifact-only workflow events. Output the final assistant answer directly. If tool use is required, "
    "output only the fenced tool_json block specified above."
)
_DATA_URL_RE = re.compile(r"^data:(?P<media>[^;,]+);base64,(?P<data>.*)$", re.DOTALL)
_TOOL_JSON_BLOCK_RE = re.compile(r"```tool_json\s*(.*?)```", re.IGNORECASE | re.DOTALL)
_TOOL_CALLS_MARKUP_RE = re.compile(
    r"<\s*(?!/)[^<>]{0,80}tool_calls(?=\s|[|｜>])[^<>]*>.*?</\s*[^<>]{0,80}tool_calls(?=\s|[|｜>])[^<>]*>",
    re.IGNORECASE | re.DOTALL,
)


def is_qwen_web_model(model: Any) -> bool:
    return normalize_model_id(model).startswith(QWEN_MODEL_PREFIX)


def normalize_qwen_model(model: str) -> str:
    model = normalize_model_id(model)
    return model.removeprefix(QWEN_MODEL_PREFIX) or "qwen3.5-plus"


class QwenWebClient:
    def __init__(
        self,
        credential: dict[str, Any],
        http_client: httpx.Client | None = None,
        request_timeout_seconds: float = DEFAULT_QWEN_REQUEST_TIMEOUT_SECONDS,
        prompt_max_chars: int = DEFAULT_QWEN_PROMPT_MAX_CHARS,
    ) -> None:
        self.credential = credential
        self.request_timeout_seconds = max(30.0, float(request_timeout_seconds or DEFAULT_QWEN_REQUEST_TIMEOUT_SECONDS))
        self.prompt_max_chars = max(4000, int(prompt_max_chars or DEFAULT_QWEN_PROMPT_MAX_CHARS))
        self.last_diagnostic: dict[str, Any] = {}
        self._model_alias_cache: dict[str, str] | None = None
        self.http_client = http_client or httpx.Client(timeout=self.request_timeout_seconds, trust_env=False)

    def chat_completions(self, payload: dict[str, Any]) -> dict[str, Any]:
        prompt, files = qwen_messages_to_prompt_and_files(
            payload.get("messages"),
            max_prompt_chars=self.prompt_max_chars,
            current_task_text=str(payload.get("_webai_current_task_text") or ""),
        )
        self.last_diagnostic = {
            "prompt_chars": len(prompt),
            "prompt_max_chars": self.prompt_max_chars,
            "prompt_compacted": "Prompt content was compacted" in prompt,
            "message_count": len(payload.get("messages")) if isinstance(payload.get("messages"), list) else 0,
            "current_task_anchor_chars": len(str(payload.get("_webai_current_task_text") or "")),
            **prompt_preserved_task_state_diagnostics(prompt),
        }
        if not prompt.strip():
            raise ValueError("没有可发送给 Qwen 网页模型的消息")
        if files:
            raise RuntimeError("Qwen Web 直连暂不支持 multimodal 附件上传；请改用 WebAI2API Qwen 适配器或支持多模态的上游。")
        model = str(payload.get("model") or f"{QWEN_MODEL_PREFIX}qwen3.5-plus")
        upstream_model = self.resolve_model_alias(model)
        if upstream_model != normalize_qwen_model(model):
            self.last_diagnostic["model_alias_resolved"] = True
            self.last_diagnostic["upstream_model"] = upstream_model
        chat = self.create_chat_session()
        try:
            content = self.send_chat(
                chat_id=str(chat.get("chatId") or chat.get("chat_id") or chat.get("id") or ""),
                prompt=prompt,
                model=upstream_model,
                files=files,
                enable_web_search=bool(payload.get("_webai_native_web_search")),
            )
            metadata_retry_count = 0
            if _is_qwen_web_non_answer_text(content):
                metadata_retry_count = 1
                self.last_diagnostic["metadata_only_response"] = True
                content = self.send_chat(
                    chat_id=str(chat.get("chatId") or chat.get("chat_id") or chat.get("id") or ""),
                    prompt=prompt.rstrip() + QWEN_WEB_METADATA_RETRY_INSTRUCTION,
                    model=upstream_model,
                    files=files,
                    enable_web_search=bool(payload.get("_webai_native_web_search")),
                )
            self.last_diagnostic["metadata_retry_count"] = metadata_retry_count
            if _is_qwen_web_non_answer_text(content):
                self.last_diagnostic["metadata_retry_succeeded"] = False
                content = ""
            elif metadata_retry_count:
                self.last_diagnostic["metadata_retry_succeeded"] = True
            self.last_diagnostic["output_chars"] = len(content)
            self.last_diagnostic["output_preview"] = content[:500]
        except TimeoutError as exc:
            stream_diagnostic = getattr(exc, "diagnostic", {})
            if isinstance(stream_diagnostic, dict):
                self.last_diagnostic = {**self.last_diagnostic, **stream_diagnostic}
            raise
        return {
            "id": f"chatcmpl-{uuid.uuid4().hex}",
            "object": "chat.completion",
            "created": int(time.time()),
            "model": model,
            "choices": [
                {
                    "index": 0,
                    "finish_reason": "stop",
                    "message": {"role": "assistant", "content": content},
                }
            ],
        }

    def list_models(self) -> list[dict[str, Any]]:
        response = self.http_client.get(
            f"{QWEN_BASE_URL}{QWEN_MODELS_PATH}",
            headers=self.headers(accept="application/json"),
            follow_redirects=True,
        )
        response.raise_for_status()
        _raise_for_qwen_response_error(response)
        return parse_qwen_model_catalog(response.json())

    def resolve_model_alias(self, model: str) -> str:
        requested = normalize_qwen_model(model)
        alias_map = self._live_model_alias_map()
        return alias_map.get(requested, requested)

    def _live_model_alias_map(self) -> dict[str, str]:
        if self._model_alias_cache is not None:
            return self._model_alias_cache
        aliases: dict[str, str] = {}
        try:
            for item in self.list_models():
                if not isinstance(item, dict):
                    continue
                model_id = normalize_qwen_model(str(item.get("id") or ""))
                if not model_id:
                    continue
                upstream_model = normalize_qwen_model(str(item.get("upstream_model") or model_id))
                if upstream_model:
                    aliases[model_id] = upstream_model
                    aliases[upstream_model] = upstream_model
        except Exception:
            aliases = {}
        self._model_alias_cache = aliases
        return aliases

    def create_chat_session(self) -> dict[str, Any]:
        response = self.http_client.post(
            f"{QWEN_BASE_URL}/api/v2/chats/new",
            json={},
            headers=self.headers(accept="application/json"),
        )
        response.raise_for_status()
        _raise_for_qwen_response_error(response)
        data = response.json()
        chat_id = _deep_get(data, ("data", "id")) or data.get("chat_id") or data.get("id") or data.get("chatId")
        if not chat_id:
            raise RuntimeError("Qwen 没有返回可用的会话信息")
        return {"chatId": chat_id, "raw": data}

    def send_chat(
        self,
        *,
        chat_id: str,
        prompt: str,
        model: str,
        files: list[dict[str, Any]] | None = None,
        enable_web_search: bool = False,
    ) -> str:
        fid = str(uuid.uuid4())
        qwen_files = files or []
        if qwen_files:
            raise RuntimeError("Qwen Web 直连暂不支持 multimodal 附件上传；请改用 WebAI2API Qwen 适配器或支持多模态的上游。")
        feature_config = {"thinking_enabled": False, "output_schema": "phase"}
        if enable_web_search:
            feature_config["auto_search"] = True
        request_body = {
            "stream": True,
            "version": "2.1",
            "incremental_output": True,
            "chat_id": chat_id,
            "chat_mode": "normal",
            "model": model,
            "parent_id": None,
            "messages": [
                {
                    "fid": fid,
                    "parentId": None,
                    "childrenIds": [],
                    "role": "user",
                    "content": prompt,
                    "user_action": "chat",
                    "files": qwen_files,
                    "timestamp": int(time.time()),
                    "models": [model],
                    "chat_type": "t2t",
                    "feature_config": feature_config,
                }
            ],
        }
        if enable_web_search:
            request_body["search"] = True
        with self.http_client.stream(
            "POST",
            f"{QWEN_BASE_URL}/api/v2/chat/completions",
            params={"chat_id": chat_id},
            json=request_body,
            headers=self.headers(accept="text/event-stream"),
        ) as response:
            response.raise_for_status()
            if "json" in response.headers.get("content-type", "").lower():
                text = response.read().decode(response.encoding or "utf-8", errors="replace")
                _raise_for_qwen_stream_error(text)
                return parse_qwen_stream_text(text)
            return _collect_qwen_stream_lines(
                response.iter_lines(),
                deadline_seconds=self.request_timeout_seconds,
                max_output_chars_without_tool_json=(
                    QWEN_TOOL_BRIDGE_RUNAWAY_OUTPUT_CHARS if _is_tool_bridge_prompt(prompt) else None
                ),
            )

    def headers(self, *, accept: str) -> dict[str, str]:
        headers = {
            "Cookie": str(self.credential.get("cookie") or ""),
            "User-Agent": str(
                self.credential.get("userAgent")
                or "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
            ),
            "Content-Type": "application/json",
            "Accept": accept,
            "Origin": QWEN_BASE_URL,
            "Referer": f"{QWEN_BASE_URL}/",
        }
        bearer = str(self.credential.get("bearer") or "")
        if bearer:
            headers["Authorization"] = f"Bearer {bearer}"
        return headers


def parse_qwen_model_catalog(data: Any) -> list[dict[str, Any]]:
    models: list[dict[str, Any]] = []
    seen: set[str] = set()
    for item in _qwen_model_catalog_items(data):
        if not isinstance(item, dict):
            continue
        raw_model_id = str(
            item.get("id")
            or item.get("model")
            or item.get("modelId")
            or item.get("model_id")
            or ""
        ).strip()
        if not raw_model_id:
            continue
        normalized_raw = normalize_model_id(raw_model_id).removeprefix(QWEN_MODEL_PREFIX)
        if not normalized_raw or not normalized_raw.lower().startswith("qwen"):
            continue
        name = item.get("name") or item.get("label") or item.get("displayName") or item.get("display_name")
        payload = _qwen_model_catalog_payload(normalized_raw, name, availability_source="qwen_live_catalog")
        if payload["id"] not in seen:
            seen.add(payload["id"])
            models.append(payload)
        alias = _qwen_model_alias_from_name(name)
        if alias and alias != normalized_raw:
            alias_payload = _qwen_model_catalog_payload(
                alias,
                name,
                availability_source="qwen_live_catalog_alias",
                upstream_model=normalized_raw,
            )
            if alias_payload["id"] not in seen:
                seen.add(alias_payload["id"])
                models.append(alias_payload)
    return models


def _qwen_model_catalog_payload(
    model: str,
    name: Any,
    *,
    availability_source: str,
    upstream_model: str | None = None,
) -> dict[str, Any]:
    normalized = normalize_model_id(model).removeprefix(QWEN_MODEL_PREFIX)
    payload: dict[str, Any] = {
        "id": f"{QWEN_MODEL_PREFIX}{normalized}",
        "object": "model",
        "owned_by": "qwen",
        "availability_source": availability_source,
        "type": "text",
        "capabilities": {
            "tool_bridge": True,
            "supports_native_tools": False,
            "preferred_protocol": "openai",
            "text": True,
            "image": False,
            "video": False,
        },
    }
    if isinstance(name, str) and name.strip():
        payload["name"] = name.strip()
    if upstream_model:
        payload["upstream_model"] = normalize_model_id(upstream_model).removeprefix(QWEN_MODEL_PREFIX)
    return payload


def _qwen_model_alias_from_name(name: Any) -> str:
    if not isinstance(name, str):
        return ""
    alias = name.strip().lower()
    if not alias:
        return ""
    alias = re.sub(r"\s+", "-", alias)
    alias = re.sub(r"[^a-z0-9.]+", "-", alias)
    alias = re.sub(r"-{2,}", "-", alias).strip("-")
    return alias if alias.startswith("qwen") else ""


def _qwen_model_catalog_items(data: Any) -> list[Any]:
    if isinstance(data, list):
        return list(data)
    if not isinstance(data, dict):
        return []
    candidates: list[Any] = []
    for key in ("data", "models", "items", "list", "result"):
        if key in data:
            candidates.append(data.get(key))
    for candidate in candidates:
        if isinstance(candidate, list):
            return list(candidate)
        if isinstance(candidate, dict):
            for key in ("data", "models", "items", "list", "result"):
                nested = candidate.get(key)
                if isinstance(nested, list):
                    return list(nested)
    return []


def qwen_messages_to_prompt_and_files(
    messages: Any,
    *,
    max_prompt_chars: int | None = None,
    current_task_text: str | None = None,
) -> tuple[str, list[dict[str, Any]]]:
    if not isinstance(messages, list):
        return "", []
    parts: list[str] = [f"System: {STATELESS_WEB_API_GUARD}"]
    role_entries: list[tuple[str, str]] = [("system", STATELESS_WEB_API_GUARD)]
    files: list[dict[str, Any]] = []
    for message in messages:
        if not isinstance(message, dict):
            continue
        text, message_files = _qwen_content_to_text_and_files(message.get("content"), start_index=len(files) + 1)
        files.extend(message_files)
        for entry_role, entry_text in message_entries_for_ds2api_prompt(message, text):
            if not entry_text:
                continue
            prompt_role = web_prompt_history_role(entry_role, entry_text)
            role_entries.append((prompt_role, entry_text))
            if prompt_role == "system":
                parts.append(f"System: {entry_text}")
            elif prompt_role == "assistant":
                parts.append(f"Assistant: {entry_text}")
            elif prompt_role == "tool":
                parts.append(f"Tool: {entry_text}")
            else:
                parts.append(f"User: {entry_text}")
    prompt = "\n\n".join(parts)
    if max_prompt_chars and len(prompt) > int(max_prompt_chars):
        prompt = compact_role_messages_as_ds2api_history(
            role_entries,
            max_chars=max_prompt_chars,
            current_user_override=current_task_text,
        )
    return prompt, files


def _compact_web_prompt(prompt: str, *, max_chars: int) -> str:
    return compact_web_prompt(prompt, max_chars=max_chars)


def _is_tool_bridge_prompt(prompt: str) -> bool:
    return "WebAI Gateway's strict tool bridge" in (prompt or "")


def _qwen_content_to_text_and_files(content: Any, *, start_index: int) -> tuple[str, list[dict[str, Any]]]:
    if isinstance(content, str):
        return content, []
    if not isinstance(content, list):
        return "" if content is None else str(content), []
    chunks: list[str] = []
    files: list[dict[str, Any]] = []
    next_index = start_index
    for item in content:
        if isinstance(item, dict):
            item_type = str(item.get("type") or "").strip()
            if item_type == "text" and item.get("text") is not None:
                chunks.append(str(item["text"]))
                continue
            if item_type == "image_url":
                file_item = _qwen_file_from_image_url(item.get("image_url"), index=next_index)
                if file_item:
                    files.append(file_item)
                    chunks.append(f"[图片: {file_item.get('name') or 'image'}]")
                    next_index += 1
                continue
            if item_type == "file":
                file_item = _qwen_file_from_openai_file(item.get("file"), index=next_index)
                if file_item:
                    files.append(file_item)
                    chunks.append(f"[文件: {file_item.get('name') or 'document'}]")
                    next_index += 1
                continue
        elif item is not None:
            chunks.append(str(item))
    return "\n".join(chunks), files


def _qwen_file_from_image_url(value: Any, *, index: int) -> dict[str, Any] | None:
    if not isinstance(value, dict):
        return None
    url = str(value.get("url") or "")
    if not url:
        return None
    match = _DATA_URL_RE.match(url)
    if match:
        media_type = match.group("media")
        return {
            "type": "image",
            "media_type": media_type,
            "data": match.group("data"),
            "name": f"image_{index}.{_extension_for_media_type(media_type)}",
        }
    return {"type": "image", "url": url, "name": f"image_{index}"}


def _qwen_file_from_openai_file(value: Any, *, index: int) -> dict[str, Any] | None:
    if not isinstance(value, dict):
        return None
    filename = str(value.get("filename") or f"document_{index}")
    file_data = str(value.get("file_data") or "")
    file_url = str(value.get("file_url") or "")
    match = _DATA_URL_RE.match(file_data)
    if match:
        return {
            "type": "document",
            "media_type": match.group("media"),
            "data": match.group("data"),
            "name": filename,
        }
    if file_url:
        return {"type": "document", "url": file_url, "name": filename}
    return None


def _extension_for_media_type(media_type: str) -> str:
    mapping = {
        "image/jpeg": "jpg",
        "image/jpg": "jpg",
        "image/png": "png",
        "image/webp": "webp",
        "image/gif": "gif",
    }
    return mapping.get(media_type.lower(), "bin")


def parse_qwen_stream_text(text: str) -> str:
    return _collect_qwen_stream_lines((text or "").splitlines(), deadline_seconds=DEFAULT_QWEN_REQUEST_TIMEOUT_SECONDS)


def _collect_qwen_stream_lines(
    lines: Iterable[str | bytes],
    *,
    deadline_seconds: float,
    monotonic: Callable[[], float] = time.monotonic,
    max_output_chars_without_tool_json: int | None = None,
) -> str:
    output: list[str] = []
    think_output: list[str] = []
    stream_events = 0
    json_events = 0
    started_at = monotonic()
    for raw_line in lines:
        line = raw_line.decode("utf-8", errors="replace").strip() if isinstance(raw_line, bytes) else str(raw_line).strip()
        stream_events += 1
        elapsed = monotonic() - started_at
        if elapsed > deadline_seconds:
            exc = TimeoutError(f"Qwen Web request exceeded {deadline_seconds:g}s")
            exc.diagnostic = {  # type: ignore[attr-defined]
                "stream_events": stream_events,
                "json_events": json_events,
                "output_chars": len("".join(output)),
                "think_chars": len("".join(think_output)),
            }
            raise exc
        if not line:
            continue
        data_text = line[5:].strip() if line.startswith("data:") else line
        if not data_text:
            continue
        if data_text == "[DONE]":
            return "".join(output or think_output)
        if not data_text.startswith(("{", "[")):
            continue
        try:
            data = json.loads(data_text)
        except Exception:
            continue
        json_events += 1
        message = _qwen_error_message(data)
        if message:
            raise RuntimeError(f"Qwen API error: {message}")
        _collect_qwen_text(data, output, think_output)
        current = "".join(output or think_output)
        if _has_complete_tool_json(current):
            return current
        if max_output_chars_without_tool_json and len(current) >= max_output_chars_without_tool_json:
            return current
        elapsed = monotonic() - started_at
        if elapsed > deadline_seconds:
            exc = TimeoutError(f"Qwen Web request exceeded {deadline_seconds:g}s")
            exc.diagnostic = {  # type: ignore[attr-defined]
                "stream_events": stream_events,
                "json_events": json_events,
                "output_chars": len("".join(output)),
                "think_chars": len("".join(think_output)),
            }
            raise exc
    return "".join(output or think_output)


def _has_complete_tool_json(text: str) -> bool:
    if _TOOL_CALLS_MARKUP_RE.search(text or ""):
        return True
    for match in _TOOL_JSON_BLOCK_RE.finditer(text or ""):
        try:
            parsed = json.loads(match.group(1))
        except Exception:
            continue
        if isinstance(parsed, (dict, list)):
            return True
    return False


def _is_qwen_web_non_answer_text(text: str) -> bool:
    stripped = (text or "").strip()
    if not stripped:
        return True
    if _has_complete_tool_json(stripped):
        return False
    try:
        parsed = json.loads(stripped)
    except Exception:
        return False
    return _is_qwen_web_metadata_only(parsed)


def _is_qwen_web_metadata_only(value: Any) -> bool:
    if isinstance(value, list):
        return bool(value) and all(_is_qwen_web_metadata_only(item) for item in value)
    if not isinstance(value, dict):
        return False
    keys = {str(key) for key in value}
    if not keys or keys & _QWEN_WEB_FINAL_CONTENT_KEYS:
        return False
    if not keys <= _QWEN_WEB_METADATA_KEYS:
        return False
    return bool(keys & _QWEN_WEB_METADATA_SIGNAL_KEYS)


def _collect_qwen_text(data: Any, output: list[str], think_output: list[str]) -> None:
    if isinstance(data, list):
        for item in data:
            _collect_qwen_text(item, output, think_output)
        return
    if not isinstance(data, dict):
        return
    choices = data.get("choices") if isinstance(data.get("choices"), list) else []
    if choices and isinstance(choices[0], dict):
        delta = choices[0].get("delta") if isinstance(choices[0].get("delta"), dict) else {}
        message = choices[0].get("message") if isinstance(choices[0].get("message"), dict) else {}
        phase = str(delta.get("phase") or message.get("phase") or choices[0].get("phase") or "").lower()
        target = think_output if phase == "think" else output
        for candidate in (delta.get("content"), message.get("content")):
            if isinstance(candidate, str):
                target.append(candidate)
    for path in (
        ("output", "text"),
        ("output", "content"),
        ("data", "text"),
        ("data", "content"),
        ("response", "text"),
        ("response", "content"),
    ):
        value = _deep_get(data, path)
        if isinstance(value, str):
            output.append(value)
    for key in ("content", "text", "answer"):
        value = data.get(key)
        if isinstance(value, str):
            output.append(value)


def _deep_get(data: Any, path: tuple[str, ...]) -> Any:
    current = data
    for key in path:
        if not isinstance(current, dict):
            return None
        current = current.get(key)
    return current


def _raise_for_qwen_response_error(response: httpx.Response) -> None:
    content_type = response.headers.get("content-type", "")
    text = response.text.strip()
    if "json" not in content_type.lower() and not text.startswith(("{", "[")):
        return
    try:
        data = response.json()
    except Exception:
        return
    message = _qwen_error_message(data)
    if message:
        raise RuntimeError(f"Qwen API error: {message}")


def _raise_for_qwen_stream_error(text: str) -> None:
    for raw_line in (text or "").splitlines():
        line = raw_line.strip()
        if not line:
            continue
        data_text = line[5:].strip() if line.startswith("data:") else line
        if not data_text or data_text == "[DONE]" or not data_text.startswith(("{", "[")):
            continue
        try:
            data = json.loads(data_text)
        except Exception:
            continue
        message = _qwen_error_message(data)
        if message:
            raise RuntimeError(f"Qwen API error: {message}")


def _qwen_error_message(data: Any) -> str:
    if isinstance(data, list):
        for item in data:
            message = _qwen_error_message(item)
            if message:
                return message
        return ""
    if not isinstance(data, dict):
        return ""
    error = data.get("error") if isinstance(data.get("error"), dict) else {}
    is_failure = data.get("success") is False or bool(error)
    code = data.get("code") or _deep_get(data, ("data", "code")) or error.get("code")
    detail = (
        data.get("message")
        or data.get("msg")
        or data.get("details")
        or _deep_get(data, ("data", "message"))
        or _deep_get(data, ("data", "msg"))
        or _deep_get(data, ("data", "details"))
        or error.get("message")
    )
    if not is_failure:
        return ""
    parts = [str(item).strip() for item in (code, detail) if str(item or "").strip()]
    return " - ".join(parts) if parts else "unknown error"
