from __future__ import annotations

import json
import re
import time
import uuid
from collections.abc import Callable, Iterable
from typing import Any

import httpx


QWEN_CODER_MODEL_PREFIX = "qwen-coder/"
QWEN_CODER_BASE_URL = "https://coder.qwen.ai"
DEFAULT_QWEN_CODER_REQUEST_TIMEOUT_SECONDS = 300  # 编程任务需要更长超时
QWEN_CODER_MODEL_ALIASES = {
    "qwen-coder-plus": "qwen3-coder-plus",
}
_DATA_URL_RE = re.compile(r"^data:(?P<media>[^;,]+);base64,(?P<data>.*)$", re.DOTALL)
_TOOL_JSON_BLOCK_RE = re.compile(r"```tool_json\s*(.*?)```", re.IGNORECASE | re.DOTALL)


def is_qwen_coder_model(model: Any) -> bool:
    return isinstance(model, str) and model.startswith(QWEN_CODER_MODEL_PREFIX)


def normalize_qwen_coder_model(model: str) -> str:
    normalized = model.removeprefix(QWEN_CODER_MODEL_PREFIX) or "qwen3-coder-plus"
    return QWEN_CODER_MODEL_ALIASES.get(normalized, normalized)


class QwenCoderClient:
    """Qwen Coder 专用客户端，支持编程场景的特殊功能"""
    
    def __init__(
        self,
        credential: dict[str, Any],
        http_client: httpx.Client | None = None,
        request_timeout_seconds: float = DEFAULT_QWEN_CODER_REQUEST_TIMEOUT_SECONDS,
        prompt_max_chars: int = 12000,
        enable_artifacts: bool = True,
        enable_mcp: bool = False,
        enable_thinking: bool = True,
    ) -> None:
        self.credential = credential
        self.request_timeout_seconds = max(60.0, float(request_timeout_seconds or DEFAULT_QWEN_CODER_REQUEST_TIMEOUT_SECONDS))
        self.prompt_max_chars = max(4000, int(prompt_max_chars or 12000))
        self.enable_artifacts = enable_artifacts
        self.enable_mcp = enable_mcp
        self.enable_thinking = enable_thinking
        self.last_diagnostic: dict[str, Any] = {}
        self.http_client = http_client or httpx.Client(timeout=self.request_timeout_seconds, trust_env=False)

    def chat_completions(self, payload: dict[str, Any]) -> dict[str, Any]:
        prompt, files = qwen_coder_messages_to_prompt_and_files(payload.get("messages"), max_prompt_chars=self.prompt_max_chars)
        self.last_diagnostic = {
            "prompt_chars": len(prompt),
            "prompt_max_chars": self.prompt_max_chars,
            "prompt_compacted": "Prompt content was compacted" in prompt,
            "message_count": len(payload.get("messages")) if isinstance(payload.get("messages"), list) else 0,
            "artifacts_enabled": self.enable_artifacts,
            "mcp_enabled": self.enable_mcp,
            "thinking_enabled": self.enable_thinking,
        }
        if not prompt.strip():
            raise ValueError("没有可发送给 Qwen Coder 网页模型的消息")
        if files:
            raise RuntimeError("Qwen Coder Web 直连暂不支持 multimodal 附件上传；请改用 WebAI2API Qwen 适配器或支持多模态的上游。")
        model = str(payload.get("model") or f"{QWEN_CODER_MODEL_PREFIX}qwen-coder-plus")
        chat = self.create_chat_session()
        try:
            content = self.send_chat(
                chat_id=str(chat.get("chatId") or chat.get("chat_id") or chat.get("id") or ""),
                prompt=prompt,
                model=normalize_qwen_coder_model(model),
                files=files,
                enable_web_search=bool(payload.get("_webai_native_web_search")),
            )
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

    def create_chat_session(self) -> dict[str, Any]:
        response = self.http_client.post(
            f"{QWEN_CODER_BASE_URL}/api/v2/chats/new",
            json={},
            headers=self.headers(accept="application/json"),
        )
        response.raise_for_status()
        _raise_for_qwen_coder_response_error(response)
        data = response.json()
        chat_id = _deep_get(data, ("data", "id")) or data.get("chat_id") or data.get("id") or data.get("chatId")
        if not chat_id:
            raise RuntimeError("Qwen Coder 没有返回可用的会话信息")
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
            raise RuntimeError("Qwen Coder Web 直连暂不支持 multimodal 附件上传；请改用 WebAI2API Qwen 适配器或支持多模态的上游。")
        
        # Qwen Coder 特有的 feature 配置
        feature_config = {
            "thinking_enabled": self.enable_thinking,
            "output_schema": "phase",
            "artifacts_enabled": self.enable_artifacts,
        }
        if self.enable_mcp:
            feature_config["mcp_enabled"] = True
        if enable_web_search:
            feature_config["auto_search"] = True
            
        request_body = {
            "stream": True,
            "version": "2.1",
            "incremental_output": True,
            "chat_id": chat_id,
            "chat_mode": "coder",  # Coder 模式
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
            f"{QWEN_CODER_BASE_URL}/api/v2/chat/completions",
            params={"chat_id": chat_id},
            json=request_body,
            headers=self.headers(accept="text/event-stream"),
        ) as response:
            response.raise_for_status()
            if "json" in response.headers.get("content-type", "").lower():
                text = response.read().decode(response.encoding or "utf-8", errors="replace")
                _raise_for_qwen_coder_stream_error(text)
                return parse_qwen_coder_stream_text(text)
            return _collect_qwen_coder_stream_lines(response.iter_lines(), deadline_seconds=self.request_timeout_seconds)

    def headers(self, *, accept: str) -> dict[str, str]:
        headers = {
            "Cookie": str(self.credential.get("cookie") or ""),
            "User-Agent": str(
                self.credential.get("userAgent")
                or "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
            ),
            "Content-Type": "application/json",
            "Accept": accept,
            "Origin": QWEN_CODER_BASE_URL,
            "Referer": f"{QWEN_CODER_BASE_URL}/",
        }
        bearer = str(self.credential.get("bearer") or "")
        if bearer:
            headers["Authorization"] = f"Bearer {bearer}"
        return headers


def qwen_coder_messages_to_prompt_and_files(messages: Any, *, max_prompt_chars: int | None = None) -> tuple[str, list[dict[str, Any]]]:
    if not isinstance(messages, list):
        return "", []
    parts: list[str] = []
    files: list[dict[str, Any]] = []
    for message in messages:
        if not isinstance(message, dict):
            continue
        role = str(message.get("role") or "user")
        text, message_files = _qwen_coder_content_to_text_and_files(message.get("content"), start_index=len(files) + 1)
        files.extend(message_files)
        if not text:
            continue
        if role == "system":
            parts.append(f"System: {text}")
        elif role == "assistant":
            parts.append(f"Assistant: {text}")
        else:
            parts.append(f"User: {text}")
    prompt = "\n\n".join(parts)
    if max_prompt_chars:
        prompt = _compact_web_prompt(prompt, max_chars=max_prompt_chars)
    return prompt, files


def _compact_web_prompt(prompt: str, *, max_chars: int) -> str:
    limit = max(1000, int(max_chars or 12000))
    raw = prompt or ""
    if len(raw) <= limit:
        return raw
    notice = (
        "\n\n[Prompt content was compacted by WebAI Gateway for a web-model prompt budget. "
        f"Original length: {len(raw)} characters. Earlier middle content omitted.]\n\n"
    )
    head_len = min(2000, max(80, limit // 10))
    tail_len = max(200, limit - head_len - len(notice))
    protocol_marker = "You are using WebAI Gateway's strict tool bridge."
    marker_index = raw.find(protocol_marker)
    tail_source = raw[marker_index:] if marker_index >= 0 else raw[-tail_len:]
    if len(tail_source) > tail_len:
        tail_source = tail_source[-tail_len:]
    compacted = raw[:head_len].rstrip() + notice + tail_source.lstrip()
    if len(compacted) > limit:
        compacted = compacted[:limit]
    return compacted


def _qwen_coder_content_to_text_and_files(content: Any, *, start_index: int) -> tuple[str, list[dict[str, Any]]]:
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
                file_item = _qwen_coder_file_from_image_url(item.get("image_url"), index=next_index)
                if file_item:
                    files.append(file_item)
                    chunks.append(f"[图片：{file_item.get('name') or 'image'}]")
                    next_index += 1
                continue
            if item_type == "file":
                file_item = _qwen_coder_file_from_openai_file(item.get("file"), index=next_index)
                if file_item:
                    files.append(file_item)
                    chunks.append(f"[文件：{file_item.get('name') or 'document'}]")
                    next_index += 1
                continue
        elif item is not None:
            chunks.append(str(item))
    return "\n".join(chunks), files


def _qwen_coder_file_from_image_url(value: Any, *, index: int) -> dict[str, Any] | None:
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


def _qwen_coder_file_from_openai_file(value: Any, *, index: int) -> dict[str, Any] | None:
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


def parse_qwen_coder_stream_text(text: str) -> str:
    return _collect_qwen_coder_stream_lines((text or "").splitlines(), deadline_seconds=DEFAULT_QWEN_CODER_REQUEST_TIMEOUT_SECONDS)


def _collect_qwen_coder_stream_lines(
    lines: Iterable[str | bytes],
    *,
    deadline_seconds: float,
    monotonic: Callable[[], float] = time.monotonic,
) -> str:
    output: list[str] = []
    think_output: list[str] = []
    artifact_output: list[str] = []  # 收集 artifacts 输出
    stream_events = 0
    json_events = 0
    started_at = monotonic()
    for raw_line in lines:
        line = raw_line.decode("utf-8", errors="replace").strip() if isinstance(raw_line, bytes) else str(raw_line).strip()
        if not line:
            continue
        stream_events += 1
        data_text = line[5:].strip() if line.startswith("data:") else line
        if not data_text:
            continue
        if data_text == "[DONE]":
            # 优先返回 artifact 内容，其次是普通输出
            return "".join(artifact_output or output or think_output)
        if not data_text.startswith(("{", "[")):
            continue
        try:
            data = json.loads(data_text)
        except Exception:
            continue
        json_events += 1
        message = _qwen_coder_error_message(data)
        if message:
            raise RuntimeError(f"Qwen Coder API error: {message}")
        _collect_qwen_coder_text(data, output, think_output, artifact_output)
        current = "".join(output or think_output or artifact_output)
        if _has_complete_tool_json(current):
            return current
        elapsed = monotonic() - started_at
        if elapsed > deadline_seconds:
            exc = TimeoutError(f"Qwen Coder Web request exceeded {deadline_seconds:g}s")
            exc.diagnostic = {  # type: ignore[attr-defined]
                "stream_events": stream_events,
                "json_events": json_events,
                "output_chars": len("".join(output)),
                "think_chars": len("".join(think_output)),
                "artifact_chars": len("".join(artifact_output)),
            }
            raise exc
    return "".join(artifact_output or output or think_output)


def _has_complete_tool_json(text: str) -> bool:
    for match in _TOOL_JSON_BLOCK_RE.finditer(text or ""):
        try:
            parsed = json.loads(match.group(1))
        except Exception:
            continue
        if isinstance(parsed, (dict, list)):
            return True
    return False


def _collect_qwen_coder_text(data: Any, output: list[str], think_output: list[str], artifact_output: list[str]) -> None:
    if isinstance(data, list):
        for item in data:
            _collect_qwen_coder_text(item, output, think_output, artifact_output)
        return
    if not isinstance(data, dict):
        return
    choices = data.get("choices") if isinstance(data.get("choices"), list) else []
    if choices and isinstance(choices[0], dict):
        delta = choices[0].get("delta") if isinstance(choices[0].get("delta"), dict) else {}
        message = choices[0].get("message") if isinstance(choices[0].get("message"), dict) else {}
        phase = str(delta.get("phase") or message.get("phase") or choices[0].get("phase") or "").lower()
        
        # 处理 artifacts 内容
        artifact = delta.get("artifact") or message.get("artifact") or {}
        if isinstance(artifact, dict):
            artifact_content = artifact.get("content") or artifact.get("code") or artifact.get("text")
            if isinstance(artifact_content, str):
                artifact_output.append(artifact_content)
        
        # 处理 thinking 和普通内容
        target = think_output if phase == "think" else output
        for candidate in (delta.get("content"), message.get("content")):
            if isinstance(candidate, str):
                target.append(candidate)
    
    # 其他路径的内容收集
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


def _raise_for_qwen_coder_response_error(response: httpx.Response) -> None:
    content_type = response.headers.get("content-type", "")
    text = response.text.strip()
    if "json" not in content_type.lower() and not text.startswith(("{", "[")):
        return
    try:
        data = response.json()
    except Exception:
        return
    message = _qwen_coder_error_message(data)
    if message:
        raise RuntimeError(f"Qwen Coder API error: {message}")


def _raise_for_qwen_coder_stream_error(text: str) -> None:
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
        message = _qwen_coder_error_message(data)
        if message:
            raise RuntimeError(f"Qwen Coder API error: {message}")


def _qwen_coder_error_message(data: Any) -> str:
    if isinstance(data, list):
        for item in data:
            message = _qwen_coder_error_message(item)
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
