from __future__ import annotations

import argparse
from contextlib import contextmanager
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import re
import shutil
import socket
import subprocess
import sys
import time
from typing import Any, Sequence
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen
from urllib.parse import urlsplit, urlunsplit

from webai_gateway.config import GatewayConfig, load_config
from webai_gateway.ds2api_sidecar_config import build_ds2api_sidecar_config


LOCAL_HOSTS = {"127.0.0.1", "localhost", "::1"}
WEBAI2API_START_GRACE_SECONDS = 420.0
RUNTIME_LOCK_STALE_SECONDS = 30.0


def collect_supervisor_status(
    config: GatewayConfig,
    gateway_root: Path,
    *,
    probe_ports: bool = True,
) -> dict[str, Any]:
    state = _load_state(gateway_root)
    gateway_url = _public_base_url(config)
    services = [
        {
            "id": "gateway",
            "label": "WebLLM Gateway",
            "role": "public-api",
            "internal": False,
            "baseUrl": gateway_url,
            "port": int(config.server.port),
            "status": "running",
            "pid": os.getpid(),
        },
        _service_status(
            "webai2api",
            "Web login adapter runtime",
            "web-login-runtime",
            config.upstream.base_url,
            default_port=8500,
            state=state,
            probe_ports=probe_ports,
            api_key=config.upstream.api_key,
            optional=True,
        ),
        _service_status(
            "ds2api",
            "DeepSeek compatibility runtime",
            "deepseek-web-runtime",
            config.provider_runtime.deepseek_ds2api_base_url,
            default_port=9331,
            state=state,
            probe_ports=probe_ports,
            optional=True,
        ),
    ]
    failed = [
        item
        for item in services
        if item.get("internal")
        and item.get("status") in {"failed", "missing", "stopped"}
        and not (item.get("optional") and item.get("status") in {"missing", "stopped"})
    ]
    return {
        "mode": "single-entry",
        "singleEntry": True,
        "publicBaseUrl": gateway_url,
        "publicPort": int(config.server.port),
        "internalRuntimeCount": sum(1 for item in services if item.get("internal")),
        "services": services,
        "failedServices": failed,
        "lastSupervisorRun": state.get("updatedAt") or "",
    }


def ensure_managed_runtimes(config: GatewayConfig, config_path: Path, gateway_root: Path) -> dict[str, Any]:
    with _runtime_start_lock(gateway_root):
        previous_state = _load_state(gateway_root)
        services = {
            "webai2api": _ensure_webai2api(config, gateway_root, previous_state),
            "ds2api": _ensure_ds2api(config, config_path, gateway_root),
        }
        state = {"updatedAt": _utc_timestamp(), "services": services}
        _save_state(gateway_root, state)
        return state


def _service_status(
    service_id: str,
    label: str,
    role: str,
    base_url: str,
    *,
    default_port: int,
    state: dict[str, Any],
    probe_ports: bool,
    api_key: str | None = None,
    optional: bool = False,
) -> dict[str, Any]:
    parsed = _parse_service_url(base_url, default_port)
    saved = state.get("services", {}).get(service_id, {}) if isinstance(state.get("services"), dict) else {}
    status = str(saved.get("status") or "unknown")
    if probe_ports:
        if status == "external" and not parsed["local"]:
            status = "external"
        elif parsed["local"] and parsed["port"] and _port_is_listening(parsed["host"], parsed["port"]):
            status = "running"
        elif service_id == "webai2api" and status in {"started", "starting"} and _service_start_is_recent(
            state, saved, max_age_seconds=WEBAI2API_START_GRACE_SECONDS
        ):
            status = "starting"
        elif status == "started" and _state_is_recent(state, max_age_seconds=90.0):
            status = "starting"
        elif status in {"started", "running", "already-running"}:
            status = "stopped"
        elif status not in {"missing", "failed"}:
            status = "unknown" if not parsed["local"] else "stopped"
    result = {
        "id": service_id,
        "label": label,
        "role": role,
        "internal": True,
        "optional": optional,
        "baseUrl": base_url,
        "host": parsed["host"],
        "port": parsed["port"],
        "status": status,
        "pid": saved.get("pid"),
        "message": _safe_message(saved.get("message")),
    }
    if service_id == "webai2api" and probe_ports and status == "running":
        result.update(_probe_webai2api_api(base_url, api_key or ""))
    return result


def _probe_webai2api_api(base_url: str, api_key: str) -> dict[str, Any]:
    models_url = _openai_models_url(base_url)
    headers = {"Accept": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    request = Request(models_url, headers=headers, method="GET")
    try:
        with urlopen(request, timeout=1.5) as response:
            if 200 <= int(response.status) < 300:
                return {"status": "running", "apiReady": True, "message": "WebAI2API OpenAI API ready"}
            return {
                "status": "failed",
                "apiReady": False,
                "message": f"WebAI2API OpenAI API returned HTTP {int(response.status)}",
            }
    except HTTPError as exc:
        preview = _safe_http_error_preview(exc)
        return {
            "status": "failed",
            "apiReady": False,
            "message": f"WebAI2API OpenAI API returned HTTP {exc.code}: {preview}".strip(),
        }
    except (TimeoutError, URLError, OSError) as exc:
        return {
            "status": "starting",
            "apiReady": False,
            "message": f"WebAI2API OpenAI API probe pending: {_safe_message(str(exc))}",
        }


def _openai_models_url(base_url: str) -> str:
    parsed = urlsplit(base_url or "")
    path = parsed.path or "/v1"
    if path.endswith("/chat/completions"):
        path = path[: -len("/chat/completions")]
    if path.endswith("/images/generations"):
        path = path[: -len("/images/generations")]
    path = path.rstrip("/")
    if not path.endswith("/models"):
        path = f"{path}/models" if path else "/models"
    return urlunsplit((parsed.scheme or "http", parsed.netloc, path, "", ""))


def _safe_http_error_preview(exc: HTTPError) -> str:
    try:
        body = exc.read(512).decode("utf-8", errors="replace")
    except Exception:
        body = ""
    body = re.sub(r"\s+", " ", body).strip()
    return _safe_message(body or str(exc.reason or ""))


def _ensure_webai2api(config: GatewayConfig, gateway_root: Path, previous_state: dict[str, Any] | None = None) -> dict[str, Any]:
    parsed = _parse_service_url(config.upstream.base_url, 8500)
    if not parsed["local"]:
        return {"status": "external", "message": f"WebAI2API runtime is external: {config.upstream.base_url}"}
    if parsed["local"] and parsed["port"] and _port_is_listening(parsed["host"], parsed["port"]):
        return {"status": "already-running", "message": "WebAI2API runtime already listening"}
    previous_state = previous_state if isinstance(previous_state, dict) else {}
    previous_services = previous_state.get("services") if isinstance(previous_state.get("services"), dict) else {}
    previous_service = previous_services.get("webai2api") if isinstance(previous_services.get("webai2api"), dict) else {}
    if str(previous_service.get("status") or "") in {"started", "starting"} and _service_start_is_recent(
        previous_state, previous_service, max_age_seconds=WEBAI2API_START_GRACE_SECONDS
    ):
        return {
            "status": "starting",
            "pid": previous_service.get("pid"),
            "path": previous_service.get("path"),
            "startedAt": previous_service.get("startedAt") or previous_state.get("updatedAt") or _utc_timestamp(),
            "message": "WebAI2API runtime launch is still pending; not started again",
        }
    sidecar_dir = _default_webai2api_sidecar_dir(gateway_root).resolve()
    if not (sidecar_dir / "package.json").exists():
        return {"status": "missing", "path": str(sidecar_dir), "message": f"未找到 WebAI2API runtime：{sidecar_dir}"}
    corepack = shutil.which("corepack.cmd") or shutil.which("corepack")
    if not corepack:
        return {"status": "missing", "path": str(sidecar_dir), "message": "未找到 corepack，请先安装 Node.js/Corepack"}
    log_dir = _log_dir(gateway_root)
    with (log_dir / "webai2api-out.log").open("ab") as stdout, (log_dir / "webai2api-err.log").open("ab") as stderr:
        process = subprocess.Popen(
            [corepack, "pnpm", "start"],
            cwd=sidecar_dir,
            stdin=subprocess.DEVNULL,
            stdout=stdout,
            stderr=stderr,
            creationflags=_creation_flags(),
        )
    return {
        "status": "started",
        "pid": process.pid,
        "path": str(sidecar_dir),
        "startedAt": _utc_timestamp(),
        "message": "WebAI2API runtime started",
    }


def _ensure_ds2api(config: GatewayConfig, config_path: Path, gateway_root: Path) -> dict[str, Any]:
    parsed = _parse_service_url(config.provider_runtime.deepseek_ds2api_base_url, 9331)
    if not parsed["local"]:
        return {"status": "external", "message": f"ds2api runtime is external: {config.provider_runtime.deepseek_ds2api_base_url}"}
    if parsed["local"] and parsed["port"] and _port_is_listening(parsed["host"], parsed["port"]):
        return {"status": "already-running", "message": "ds2api runtime already listening"}
    exe = _default_ds2api_exe(gateway_root).resolve()
    if not exe.exists():
        return {"status": "missing", "path": str(exe), "message": f"未找到 ds2api runtime：{exe}"}
    log_dir = _log_dir(gateway_root)
    env = os.environ.copy()
    env["PORT"] = str(parsed["port"] or 9331)
    env["DS2API_CONFIG_JSON"] = json.dumps(build_ds2api_sidecar_config(config), ensure_ascii=True, separators=(",", ":"))
    env["DS2API_ADMIN_KEY"] = env.get("DS2API_ADMIN_KEY") or "local-dev-admin"
    env["DS2API_CONFIG_PATH"] = str((gateway_root / ".webai-gateway" / "ds2api" / "config.json").resolve())
    Path(env["DS2API_CONFIG_PATH"]).parent.mkdir(parents=True, exist_ok=True)
    with (log_dir / "ds2api-out.log").open("ab") as stdout, (log_dir / "ds2api-err.log").open("ab") as stderr:
        process = subprocess.Popen(
            [str(exe)],
            cwd=gateway_root,
            env=env,
            stdin=subprocess.DEVNULL,
            stdout=stdout,
            stderr=stderr,
            creationflags=_creation_flags(),
        )
    return {"status": "started", "pid": process.pid, "path": str(exe), "message": "ds2api runtime started"}


def _parse_service_url(base_url: str, default_port: int) -> dict[str, Any]:
    parsed = urlsplit(base_url or "")
    host = parsed.hostname or "127.0.0.1"
    port = parsed.port or default_port
    return {"host": host, "port": port, "local": host.lower() in LOCAL_HOSTS}


def _public_base_url(config: GatewayConfig) -> str:
    host = config.server.host
    public_host = "127.0.0.1" if host in {"0.0.0.0", "::"} else host
    return urlunsplit(("http", f"{public_host}:{int(config.server.port)}", "", "", ""))


def _default_webai2api_sidecar_dir(gateway_root: Path) -> Path:
    configured = os.environ.get("WEBAI2API_SIDECAR_DIR")
    if configured and configured.strip():
        return Path(configured).expanduser()
    return gateway_root.parent / "WebAI2API-sidecar"


def _default_ds2api_exe(gateway_root: Path) -> Path:
    configured = os.environ.get("WEBAI_DEEPSEEK_DS2API_EXE")
    if configured and configured.strip():
        return Path(configured).expanduser()
    return gateway_root / ".tmp" / "ds2api" / ".tmp-bin" / "ds2api.exe"


def _port_is_listening(host: str, port: int) -> bool:
    try:
        with socket.create_connection((host, int(port)), timeout=0.25):
            return True
    except OSError:
        return False


def _log_dir(gateway_root: Path) -> Path:
    path = gateway_root / ".webai-gateway" / "logs"
    path.mkdir(parents=True, exist_ok=True)
    return path


def _state_path(gateway_root: Path) -> Path:
    return gateway_root / ".webai-gateway" / "runtime" / "managed-runtimes.json"


@contextmanager
def _runtime_start_lock(gateway_root: Path) -> Any:
    lock_path = _state_path(gateway_root).with_suffix(".lock")
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    deadline = time.monotonic() + RUNTIME_LOCK_STALE_SECONDS
    handle: int | None = None
    while handle is None:
        try:
            handle = os.open(str(lock_path), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        except FileExistsError:
            try:
                stale = (time.time() - lock_path.stat().st_mtime) > RUNTIME_LOCK_STALE_SECONDS
            except OSError:
                stale = False
            if stale:
                try:
                    lock_path.unlink()
                except OSError:
                    pass
                continue
            if time.monotonic() >= deadline:
                raise TimeoutError("Timed out waiting for the runtime startup lock")
            time.sleep(0.05)
    try:
        os.write(handle, str(os.getpid()).encode("ascii"))
        yield
    finally:
        os.close(handle)
        try:
            lock_path.unlink()
        except OSError:
            pass


def _load_state(gateway_root: Path) -> dict[str, Any]:
    path = _state_path(gateway_root)
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    return data if isinstance(data, dict) else {}


def _save_state(gateway_root: Path, state: dict[str, Any]) -> None:
    path = _state_path(gateway_root)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(state, ensure_ascii=False, indent=2), encoding="utf-8")


def _safe_message(value: Any) -> str:
    text = str(value or "")
    for marker in ("cookie", "bearer", "token", "session", "api_key", "apikey"):
        if marker in text.lower():
            return "runtime status message hidden"
    text = re.sub(r"[A-Za-z]:\\[^\s,;]+", "[local path]", text)
    text = re.sub(r"/(?:[^/\s]+/){2,}[^\s,;]+", "[local path]", text)
    return text[:500]


def _creation_flags() -> int:
    return int(getattr(subprocess, "CREATE_NO_WINDOW", 0))


def _utc_timestamp() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def _state_is_recent(state: dict[str, Any], *, max_age_seconds: float) -> bool:
    return _timestamp_is_recent(str(state.get("updatedAt") or ""), max_age_seconds=max_age_seconds)


def _service_start_is_recent(state: dict[str, Any], service: dict[str, Any], *, max_age_seconds: float) -> bool:
    started_at = str(service.get("startedAt") or state.get("updatedAt") or "")
    return _timestamp_is_recent(started_at, max_age_seconds=max_age_seconds)


def _timestamp_is_recent(timestamp: str, *, max_age_seconds: float) -> bool:
    try:
        parsed = datetime.strptime(timestamp, "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=timezone.utc)
    except ValueError:
        return False
    age = (datetime.now(timezone.utc) - parsed).total_seconds()
    return 0 <= age <= max_age_seconds


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Start and inspect WebLLM Gateway internal runtimes.")
    parser.add_argument("--config", default="config.json", help="Path to WebLLM Gateway config.json")
    parser.add_argument("--ensure", action="store_true", help="Start missing internal runtimes if possible")
    args = parser.parse_args(argv)
    config_path = Path(args.config).resolve()
    gateway_root = config_path.parent
    config = load_config(config_path)
    if args.ensure:
        result = ensure_managed_runtimes(config, config_path, gateway_root)
        json.dump(result, sys.stdout, ensure_ascii=False, indent=2)
        print()
        return 0
    json.dump(collect_supervisor_status(config, gateway_root), sys.stdout, ensure_ascii=False, indent=2)
    print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
