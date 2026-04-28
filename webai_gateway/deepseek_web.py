from __future__ import annotations

import base64
import hashlib
import json
import time
import uuid
from typing import Any

import httpx


DEEPSEEK_MODEL_PREFIX = "deepseek-web/"
DEEPSEEK_BASE_URL = "https://chat.deepseek.com"


def is_deepseek_web_model(model: Any) -> bool:
    return isinstance(model, str) and model.startswith(DEEPSEEK_MODEL_PREFIX)


def normalize_deepseek_model(model: str) -> str:
    return model.removeprefix(DEEPSEEK_MODEL_PREFIX) or "deepseek-chat"


class DeepSeekWebClient:
    def __init__(self, credential: dict[str, Any], http_client: httpx.Client | None = None) -> None:
        self.credential = credential
        self.http_client = http_client or httpx.Client(timeout=120, trust_env=False)

    def chat_completions(self, payload: dict[str, Any]) -> dict[str, Any]:
        prompt = messages_to_prompt(payload.get("messages"))
        if not prompt.strip():
            raise ValueError("没有可发送给 DeepSeek 网页模型的消息")
        model = str(payload.get("model") or "deepseek-web/deepseek-chat")
        session = self.create_chat_session()
        content = self.send_chat(
            session_id=str(session.get("chat_session_id") or session.get("id") or ""),
            prompt=prompt,
            model=normalize_deepseek_model(model),
        )
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
            f"{DEEPSEEK_BASE_URL}/api/v0/chat_session/create",
            json={},
            headers=self.headers(),
        )
        response.raise_for_status()
        data = response.json()
        session = _deep_get(data, ("data", "biz_data"))
        if not isinstance(session, dict):
            raise RuntimeError("DeepSeek 没有返回可用的会话信息")
        session_id = session.get("id") or session.get("chat_session_id")
        if session_id:
            session["chat_session_id"] = session_id
        return session

    def send_chat(self, *, session_id: str, prompt: str, model: str) -> str:
        target_path = "/api/v0/chat/completion"
        pow_response = self.pow_response(target_path)
        response = self.http_client.post(
            f"{DEEPSEEK_BASE_URL}{target_path}",
            json={
                "chat_session_id": session_id,
                "parent_message_id": None,
                "prompt": prompt,
                "ref_file_ids": [],
                "thinking_enabled": model != "deepseek-chat",
                "search_enabled": True,
                "preempt": False,
            },
            headers={**self.headers(), "x-ds-pow-response": pow_response},
        )
        response.raise_for_status()
        return parse_deepseek_stream_text(response.text)

    def pow_response(self, target_path: str) -> str:
        response = self.http_client.post(
            f"{DEEPSEEK_BASE_URL}/api/v0/chat/create_pow_challenge",
            json={"target_path": target_path},
            headers=self.headers(),
        )
        response.raise_for_status()
        data = response.json()
        challenge = (
            _deep_get(data, ("data", "biz_data", "challenge"))
            or _deep_get(data, ("data", "challenge"))
            or data.get("challenge")
        )
        if not isinstance(challenge, dict):
            raise RuntimeError("DeepSeek 没有返回 PoW challenge")
        algorithm = str(challenge.get("algorithm") or "")
        if algorithm != "sha256":
            raise RuntimeError(f"DeepSeek PoW 算法 {algorithm or 'unknown'} 暂未内置，请暂时使用 WebAI2API 上游模式")
        answer = solve_sha256_pow(challenge)
        payload = {**challenge, "answer": answer, "target_path": target_path}
        return base64.b64encode(json.dumps(payload, separators=(",", ":")).encode("utf-8")).decode("ascii")

    def headers(self) -> dict[str, str]:
        headers = {
            "Cookie": str(self.credential.get("cookie") or ""),
            "User-Agent": str(
                self.credential.get("userAgent")
                or "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
            ),
            "Content-Type": "application/json",
            "Accept": "*/*",
            "Referer": "https://chat.deepseek.com/",
            "Origin": "https://chat.deepseek.com",
            "x-client-platform": "web",
            "x-client-version": "1.7.0",
            "x-app-version": "20241129.1",
            "x-client-locale": "zh_CN",
            "x-client-timezone-offset": "28800",
        }
        bearer = str(self.credential.get("bearer") or "")
        if bearer:
            headers["Authorization"] = f"Bearer {bearer}"
        return headers


def messages_to_prompt(messages: Any) -> str:
    if not isinstance(messages, list):
        return ""
    parts: list[str] = []
    for message in messages:
        if not isinstance(message, dict):
            continue
        role = str(message.get("role") or "user")
        content = message.get("content")
        text = content_to_text(content)
        if not text:
            continue
        if role == "system":
            parts.append(f"System: {text}")
        elif role == "assistant":
            parts.append(f"Assistant: {text}")
        else:
            parts.append(f"User: {text}")
    return "\n\n".join(parts)


def content_to_text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        chunks = []
        for item in content:
            if isinstance(item, dict):
                if item.get("type") == "text" and item.get("text") is not None:
                    chunks.append(str(item["text"]))
            elif item is not None:
                chunks.append(str(item))
        return "\n".join(chunks)
    if content is None:
        return ""
    return str(content)


def solve_sha256_pow(challenge: dict[str, Any]) -> int:
    salt = str(challenge.get("salt") or "")
    target = str(challenge.get("challenge") or "")
    difficulty = int(challenge.get("difficulty") or 0)
    target_difficulty = int(difficulty.bit_length() - 1) if difficulty > 1000 else difficulty
    for nonce in range(1_000_001):
        digest = hashlib.sha256(f"{salt}{target}{nonce}".encode("utf-8")).hexdigest()
        if leading_zero_bits(digest) >= target_difficulty:
            return nonce
    raise TimeoutError("DeepSeek SHA256 PoW 求解超时")


def leading_zero_bits(hex_digest: str) -> int:
    zero_bits = 0
    for char in hex_digest:
        value = int(char, 16)
        if value == 0:
            zero_bits += 4
            continue
        zero_bits += (4 - value.bit_length())
        break
    return zero_bits


def parse_deepseek_stream_text(text: str) -> str:
    output: list[str] = []
    for raw_line in (text or "").splitlines():
        line = raw_line.strip()
        if not line.startswith("data:"):
            continue
        data_text = line[5:].strip()
        if not data_text or data_text == "[DONE]":
            continue
        try:
            data = json.loads(data_text)
        except Exception:
            continue
        _collect_deepseek_text(data, output)
    return "".join(output)


def _collect_deepseek_text(data: Any, output: list[str]) -> None:
    if not isinstance(data, dict):
        return
    value = data.get("v")
    path = str(data.get("p") or "")
    if isinstance(value, str) and (not path or "content" in path or "choices" in path):
        output.append(value)
    if isinstance(value, list):
        for item in value:
            if isinstance(item, dict):
                if isinstance(item.get("content"), str):
                    output.append(item["content"])
                elif isinstance(item.get("v"), str) and ("content" in str(item.get("p") or "") or not item.get("p")):
                    output.append(item["v"])
    if data.get("type") == "text" and isinstance(data.get("content"), str):
        output.append(data["content"])
    fragments = _deep_get(data, ("v", "response", "fragments"))
    if isinstance(fragments, list):
        for fragment in fragments:
            if isinstance(fragment, dict) and isinstance(fragment.get("content"), str):
                output.append(fragment["content"])
    choices = data.get("choices") if isinstance(data.get("choices"), list) else []
    if choices and isinstance(choices[0], dict):
        delta = choices[0].get("delta") if isinstance(choices[0].get("delta"), dict) else {}
        if isinstance(delta.get("content"), str):
            output.append(delta["content"])


def _deep_get(data: Any, path: tuple[str, ...]) -> Any:
    current = data
    for key in path:
        if not isinstance(current, dict):
            return None
        current = current.get(key)
    return current
