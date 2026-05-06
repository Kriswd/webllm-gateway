from __future__ import annotations

import copy
import hashlib
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib.parse import quote, unquote


VALIDATION_CACHE_SECONDS = 6 * 60 * 60
PLAN_TYPES = {"free", "plus", "pro", "team", "unknown"}


def account_now() -> str:
    from datetime import datetime, timezone

    return datetime.now(timezone.utc).isoformat()


def stable_json_hash(value: Any) -> str:
    payload = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def webai2api_account_id(provider_id: str, instance_name: str, worker_name: str) -> str:
    return f"webai2api:{provider_id}:{quote(instance_name, safe='')}:{quote(worker_name, safe='')}"


def direct_account_id(provider_id: str, profile_id: str = "default") -> str:
    return f"direct:{provider_id}:{quote(profile_id, safe='')}"


@dataclass(frozen=True)
class ParsedAccountId:
    source: str
    provider_id: str
    instance_name: str | None = None
    worker_name: str | None = None
    profile_id: str | None = None


def parse_account_id(account_id: str) -> ParsedAccountId | None:
    parts = str(account_id or "").split(":")
    if len(parts) < 3:
        return None
    source, provider_id = parts[0], parts[1]
    if source == "webai2api" and len(parts) == 4:
        return ParsedAccountId(
            source=source,
            provider_id=provider_id,
            instance_name=unquote(parts[2]),
            worker_name=unquote(parts[3]),
        )
    if source == "direct" and len(parts) == 3:
        return ParsedAccountId(source=source, provider_id=provider_id, profile_id=unquote(parts[2]))
    return None


class AccountRegistry:
    """Non-sensitive account metadata and validation cache.

    This registry deliberately stores references and labels only. It must never
    contain cookies, bearer tokens, session tokens, API keys, or browser profile
    contents.
    """

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)

    def load(self) -> dict[str, Any]:
        if not self.path.exists():
            return {"version": 1, "accounts": {}, "current": {}, "sync": {}}
        try:
            data = json.loads(self.path.read_text(encoding="utf-8"))
        except Exception:
            return {"version": 1, "accounts": {}, "current": {}, "sync": {}}
        if not isinstance(data, dict):
            return {"version": 1, "accounts": {}, "current": {}, "sync": {}}
        data.setdefault("version", 1)
        if not isinstance(data.get("accounts"), dict):
            data["accounts"] = {}
        if not isinstance(data.get("current"), dict):
            data["current"] = {}
        if not isinstance(data.get("sync"), dict):
            data["sync"] = {}
        return data

    def save(self, data: dict[str, Any]) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_suffix(self.path.suffix + ".tmp")
        tmp.write_text(json.dumps(_strip_sensitive_metadata(data), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        tmp.replace(self.path)

    def metadata(self, account_id: str) -> dict[str, Any]:
        data = self.load()
        accounts = data.get("accounts") if isinstance(data.get("accounts"), dict) else {}
        item = accounts.get(account_id)
        return item if isinstance(item, dict) else {}

    def upsert_metadata(self, account_id: str, metadata: dict[str, Any]) -> dict[str, Any]:
        data = self.load()
        accounts = data["accounts"]
        existing = accounts.get(account_id) if isinstance(accounts.get(account_id), dict) else {}
        merged = {
            **existing,
            **_public_account_metadata(metadata),
            "updatedAt": account_now(),
        }
        if "createdAt" not in merged:
            merged["createdAt"] = account_now()
        accounts[account_id] = merged
        self.save(data)
        return merged

    def update_metadata(self, account_id: str, patch: dict[str, Any]) -> dict[str, Any]:
        allowed: dict[str, Any] = {}
        if isinstance(patch.get("displayName"), str):
            allowed["displayName"] = patch["displayName"].strip()[:120]
        if isinstance(patch.get("note"), str):
            allowed["note"] = patch["note"].strip()[:500]
        if isinstance(patch.get("planType"), str):
            plan_type = patch["planType"].strip().lower()
            allowed["planType"] = plan_type if plan_type in PLAN_TYPES else "unknown"
        return self.upsert_metadata(account_id, allowed)

    def current_account_id(self, provider_id: str) -> str:
        data = self.load()
        current = data.get("current") if isinstance(data.get("current"), dict) else {}
        value = current.get(provider_id)
        return str(value) if value else ""

    def set_current(self, provider_id: str, account_id: str) -> None:
        data = self.load()
        data["current"][provider_id] = account_id
        self.save(data)

    def validation_for(self, account_id: str) -> dict[str, Any]:
        metadata = self.metadata(account_id)
        validation = metadata.get("validation") if isinstance(metadata.get("validation"), dict) else {}
        return validation

    def fresh_validation(self, account_id: str, model_id: str, *, now: float | None = None) -> dict[str, Any] | None:
        validation = self.validation_for(account_id)
        item = validation.get(model_id)
        if not isinstance(item, dict):
            return None
        checked_epoch = float(item.get("checkedEpoch") or 0)
        if checked_epoch <= 0:
            return None
        if (now if now is not None else time.time()) - checked_epoch > VALIDATION_CACHE_SECONDS:
            return None
        return item

    def save_validation(self, account_id: str, model_id: str, result: dict[str, Any]) -> dict[str, Any]:
        data = self.load()
        accounts = data["accounts"]
        metadata = accounts.get(account_id) if isinstance(accounts.get(account_id), dict) else {}
        validation = metadata.get("validation") if isinstance(metadata.get("validation"), dict) else {}
        validation[model_id] = {
            **_public_validation_result(result),
            "checkedAt": account_now(),
            "checkedEpoch": time.time(),
        }
        metadata["validation"] = validation
        metadata["lastValidatedAt"] = validation[model_id]["checkedAt"]
        metadata["updatedAt"] = account_now()
        metadata.setdefault("createdAt", account_now())
        accounts[account_id] = metadata
        self.save(data)
        return validation[model_id]

    def sync_state(self, provider_id: str) -> dict[str, Any]:
        data = self.load()
        sync = data.get("sync") if isinstance(data.get("sync"), dict) else {}
        state = sync.get(provider_id)
        return copy.deepcopy(state) if isinstance(state, dict) else {}

    def save_sync_state(self, provider_id: str, state: dict[str, Any]) -> None:
        data = self.load()
        data["sync"][provider_id] = copy.deepcopy(state)
        self.save(data)


def _public_account_metadata(metadata: dict[str, Any]) -> dict[str, Any]:
    allowed: dict[str, Any] = {}
    for key in ("displayName", "note", "planType", "source", "providerId", "instanceName", "workerName", "workerType"):
        value = metadata.get(key)
        if isinstance(value, str):
            allowed[key] = value
    if allowed.get("planType") not in PLAN_TYPES:
        allowed["planType"] = "unknown"
    validation = metadata.get("validation")
    if isinstance(validation, dict):
        allowed["validation"] = {
            str(model_id): _public_validation_result(result)
            for model_id, result in validation.items()
            if isinstance(result, dict)
        }
    return allowed


def _public_validation_result(result: dict[str, Any]) -> dict[str, Any]:
    status = str(result.get("status") or ("available" if result.get("ok") else "unavailable")).lower()
    if status not in {"available", "unavailable", "pending"}:
        status = "unavailable"
    out: dict[str, Any] = {
        "status": status,
        "ok": status == "available",
    }
    if isinstance(result.get("message"), str):
        out["message"] = result["message"][:700]
    if isinstance(result.get("checkedAt"), str):
        out["checkedAt"] = result["checkedAt"]
    if isinstance(result.get("checkedEpoch"), (int, float)):
        out["checkedEpoch"] = result["checkedEpoch"]
    return out


def _strip_sensitive_metadata(value: Any) -> Any:
    if isinstance(value, dict):
        cleaned: dict[str, Any] = {}
        for key, item in value.items():
            lowered = str(key).lower()
            if any(token in lowered for token in ("cookie", "bearer", "session", "token", "secret", "api_key", "apikey", "authorization")):
                continue
            cleaned[key] = _strip_sensitive_metadata(item)
        return cleaned
    if isinstance(value, list):
        return [_strip_sensitive_metadata(item) for item in value]
    return value
