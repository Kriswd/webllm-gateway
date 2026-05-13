from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


DEFAULT_OBSERVATION_EXCLUDED_PATH_PARTS: tuple[str, ...] = (
    ".cache",
    ".git",
    ".hg",
    ".mypy_cache",
    ".next",
    ".nuxt",
    ".pnpm",
    ".pytest_cache",
    ".ruff_cache",
    ".svn",
    ".turbo",
    ".venv",
    "__pycache__",
    "bower_components",
    "build",
    "coverage",
    "dist",
    "node_modules",
    "site-packages",
    "target",
    "vendor",
)

DEFAULT_DEEPSEEK_DS2API_ACCOUNT_MAX_INFLIGHT = 2
DEFAULT_DEEPSEEK_DS2API_GLOBAL_MAX_INFLIGHT = 4
DEFAULT_DEEPSEEK_DS2API_BEARER_MAX_INFLIGHT = 1
DEFAULT_DEEPSEEK_DS2API_RATE_LIMIT_COOLDOWN_SECONDS = 6.0
DEFAULT_DEEPSEEK_DS2API_CURRENT_INPUT_FILE_ENABLED = False
DEFAULT_DEEPSEEK_DS2API_CURRENT_INPUT_FILE_MIN_CHARS = 0


@dataclass(frozen=True)
class ServerConfig:
    host: str = "127.0.0.1"
    port: int = 8610
    api_key: str = "local-dev-key"


@dataclass(frozen=True)
class UpstreamConfig:
    base_url: str = "http://127.0.0.1:8500/v1"
    api_key: str = ""
    model: str = "webai2api-model"
    tool_mode: str = "prompt"


@dataclass(frozen=True)
class ObservationPolicyConfig:
    summarize_path_lists: bool = True
    excluded_path_parts: tuple[str, ...] = DEFAULT_OBSERVATION_EXCLUDED_PATH_PARTS
    excluded_path_globs: tuple[str, ...] = ()
    path_list_max_items: int = 80


@dataclass(frozen=True)
class ProviderRuntimeConfig:
    request_timeout_seconds: int = 300
    prompt_max_chars: int = 32000
    native_web_search_policy: str = "auto"
    response_language: str = "zh-CN"
    deepseek_ds2api_base_url: str = "http://127.0.0.1:9331/v1"
    deepseek_ds2api_account_max_inflight: int = DEFAULT_DEEPSEEK_DS2API_ACCOUNT_MAX_INFLIGHT
    deepseek_ds2api_global_max_inflight: int = DEFAULT_DEEPSEEK_DS2API_GLOBAL_MAX_INFLIGHT
    deepseek_ds2api_bearer_max_inflight: int = DEFAULT_DEEPSEEK_DS2API_BEARER_MAX_INFLIGHT
    deepseek_ds2api_rate_limit_cooldown_seconds: float = DEFAULT_DEEPSEEK_DS2API_RATE_LIMIT_COOLDOWN_SECONDS
    deepseek_ds2api_current_input_file_enabled: bool = DEFAULT_DEEPSEEK_DS2API_CURRENT_INPUT_FILE_ENABLED
    deepseek_ds2api_current_input_file_min_chars: int = DEFAULT_DEEPSEEK_DS2API_CURRENT_INPUT_FILE_MIN_CHARS
    qwen_web_backend: str = "direct"
    gpt_thinking_backend: str = "webai2api"


@dataclass(frozen=True)
class ToolBridgeConfig:
    mode: str = "strict"
    activation_policy: str = "auto"
    max_tools_in_prompt: int = 32
    max_calls_per_turn: int = 1
    max_readonly_calls_per_turn: int = 32
    tool_prompt_max_chars: int = 8000
    observation_max_chars: int = 4000
    exposure_policy: str = "safe"
    tool_profile: str = "auto"
    semantic_final_judge: str = "off"
    allowed_tool_names: tuple[str, ...] = ()
    readonly_tool_names: tuple[str, ...] = ()
    write_tool_names: tuple[str, ...] = ()
    shell_tool_names: tuple[str, ...] = ()
    observation_policy: ObservationPolicyConfig = field(default_factory=ObservationPolicyConfig)


@dataclass(frozen=True)
class GatewayConfig:
    server: ServerConfig = ServerConfig()
    upstream: UpstreamConfig = UpstreamConfig()
    provider_runtime: ProviderRuntimeConfig = field(default_factory=ProviderRuntimeConfig)
    tool_bridge: ToolBridgeConfig = field(default_factory=ToolBridgeConfig)


def load_config(path: str | Path = "config.json") -> GatewayConfig:
    p = Path(path)
    if not p.exists():
        return GatewayConfig()
    raw = json.loads(p.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        return GatewayConfig()
    server_raw = raw.get("server") if isinstance(raw.get("server"), dict) else {}
    upstream_raw = raw.get("upstream") if isinstance(raw.get("upstream"), dict) else {}
    provider_runtime_raw = (
        raw.get("providerRuntime")
        if isinstance(raw.get("providerRuntime"), dict)
        else raw.get("provider_runtime") if isinstance(raw.get("provider_runtime"), dict) else {}
    )
    tool_bridge_raw = raw.get("tool_bridge") if isinstance(raw.get("tool_bridge"), dict) else {}
    default_provider_runtime = ProviderRuntimeConfig()
    deepseek_ds2api_account_max_inflight = _bounded_int(
        _raw_first(
            provider_runtime_raw,
            "deepseekDs2apiAccountMaxInflight",
            "deepseek_ds2api_account_max_inflight",
            default=default_provider_runtime.deepseek_ds2api_account_max_inflight,
        ),
        default=default_provider_runtime.deepseek_ds2api_account_max_inflight,
        minimum=1,
        maximum=256,
    )
    deepseek_ds2api_global_max_inflight = max(
        deepseek_ds2api_account_max_inflight,
        _bounded_int(
            _raw_first(
                provider_runtime_raw,
                "deepseekDs2apiGlobalMaxInflight",
                "deepseek_ds2api_global_max_inflight",
                default=default_provider_runtime.deepseek_ds2api_global_max_inflight,
            ),
            default=default_provider_runtime.deepseek_ds2api_global_max_inflight,
            minimum=1,
        ),
    )
    deepseek_ds2api_bearer_max_inflight = _bounded_int(
        _raw_first(
            provider_runtime_raw,
            "deepseekDs2apiBearerMaxInflight",
            "deepseek_ds2api_bearer_max_inflight",
            default=default_provider_runtime.deepseek_ds2api_bearer_max_inflight,
        ),
        default=default_provider_runtime.deepseek_ds2api_bearer_max_inflight,
        minimum=1,
        maximum=256,
    )
    deepseek_ds2api_rate_limit_cooldown_seconds = _bounded_float(
        _raw_first(
            provider_runtime_raw,
            "deepseekDs2apiRateLimitCooldownSeconds",
            "deepseek_ds2api_rate_limit_cooldown_seconds",
            default=default_provider_runtime.deepseek_ds2api_rate_limit_cooldown_seconds,
        ),
        default=default_provider_runtime.deepseek_ds2api_rate_limit_cooldown_seconds,
        minimum=0.0,
        maximum=300.0,
    )
    deepseek_ds2api_current_input_file_enabled = _bool_value(
        _raw_first(
            provider_runtime_raw,
            "deepseekDs2apiCurrentInputFileEnabled",
            "deepseek_ds2api_current_input_file_enabled",
            default=default_provider_runtime.deepseek_ds2api_current_input_file_enabled,
        )
    )
    deepseek_ds2api_current_input_file_min_chars = _bounded_int(
        _raw_first(
            provider_runtime_raw,
            "deepseekDs2apiCurrentInputFileMinChars",
            "deepseek_ds2api_current_input_file_min_chars",
            default=default_provider_runtime.deepseek_ds2api_current_input_file_min_chars,
        ),
        default=default_provider_runtime.deepseek_ds2api_current_input_file_min_chars,
        minimum=0,
        maximum=100_000_000,
    )
    allowed_tool_names = tool_bridge_raw.get("allowedToolNames", tool_bridge_raw.get("allowed_tool_names", ()))
    if not isinstance(allowed_tool_names, list):
        allowed_tool_names = []
    readonly_tool_names = tool_bridge_raw.get("readonlyToolNames", tool_bridge_raw.get("readonly_tool_names", ()))
    write_tool_names = tool_bridge_raw.get("writeToolNames", tool_bridge_raw.get("write_tool_names", ()))
    shell_tool_names = tool_bridge_raw.get("shellToolNames", tool_bridge_raw.get("shell_tool_names", ()))
    return GatewayConfig(
        server=ServerConfig(
            host=str(server_raw.get("host") or "127.0.0.1"),
            port=int(server_raw.get("port") or 8610),
            api_key=str(server_raw.get("apiKey") or server_raw.get("api_key") or "local-dev-key"),
        ),
        upstream=UpstreamConfig(
            base_url=str(upstream_raw.get("baseUrl") or upstream_raw.get("base_url") or "http://127.0.0.1:8500/v1"),
            api_key=str(upstream_raw.get("apiKey") or upstream_raw.get("api_key") or ""),
            model=str(upstream_raw.get("model") or "webai2api-model"),
            tool_mode=str(upstream_raw.get("toolMode") or upstream_raw.get("tool_mode") or "prompt"),
        ),
        provider_runtime=ProviderRuntimeConfig(
            request_timeout_seconds=max(
                30,
                int(provider_runtime_raw.get("requestTimeoutSeconds") or provider_runtime_raw.get("request_timeout_seconds") or 300),
            ),
            prompt_max_chars=max(
                4000,
                int(provider_runtime_raw.get("promptMaxChars") or provider_runtime_raw.get("prompt_max_chars") or 32000),
            ),
            native_web_search_policy=str(
                provider_runtime_raw.get("nativeWebSearchPolicy") or provider_runtime_raw.get("native_web_search_policy") or "auto"
            ),
            response_language=str(
                provider_runtime_raw.get("responseLanguage") or provider_runtime_raw.get("response_language") or "zh-CN"
            ),
            deepseek_ds2api_base_url=str(
                provider_runtime_raw.get("deepseekDs2apiBaseUrl")
                or provider_runtime_raw.get("deepseek_ds2api_base_url")
                or "http://127.0.0.1:9331/v1"
            ),
            deepseek_ds2api_account_max_inflight=deepseek_ds2api_account_max_inflight,
            deepseek_ds2api_global_max_inflight=deepseek_ds2api_global_max_inflight,
            deepseek_ds2api_bearer_max_inflight=deepseek_ds2api_bearer_max_inflight,
            deepseek_ds2api_rate_limit_cooldown_seconds=deepseek_ds2api_rate_limit_cooldown_seconds,
            deepseek_ds2api_current_input_file_enabled=deepseek_ds2api_current_input_file_enabled,
            deepseek_ds2api_current_input_file_min_chars=deepseek_ds2api_current_input_file_min_chars,
            qwen_web_backend=str(
                provider_runtime_raw.get("qwenWebBackend") or provider_runtime_raw.get("qwen_web_backend") or "direct"
            ),
            gpt_thinking_backend=str(
                provider_runtime_raw.get("gptThinkingBackend")
                or provider_runtime_raw.get("gpt_thinking_backend")
                or "webai2api"
            ),
        ),
        tool_bridge=ToolBridgeConfig(
            mode=str(tool_bridge_raw.get("mode") or "strict"),
            activation_policy=str(tool_bridge_raw.get("activationPolicy") or tool_bridge_raw.get("activation_policy") or "auto"),
            max_tools_in_prompt=int(tool_bridge_raw.get("maxToolsInPrompt") or tool_bridge_raw.get("max_tools_in_prompt") or 32),
            max_calls_per_turn=int(tool_bridge_raw.get("maxCallsPerTurn") or tool_bridge_raw.get("max_calls_per_turn") or 1),
            max_readonly_calls_per_turn=int(
                tool_bridge_raw.get("maxReadonlyCallsPerTurn") or tool_bridge_raw.get("max_readonly_calls_per_turn") or 32
            ),
            tool_prompt_max_chars=int(tool_bridge_raw.get("toolPromptMaxChars") or tool_bridge_raw.get("tool_prompt_max_chars") or 8000),
            observation_max_chars=int(tool_bridge_raw.get("observationMaxChars") or tool_bridge_raw.get("observation_max_chars") or 4000),
            exposure_policy=str(tool_bridge_raw.get("exposurePolicy") or tool_bridge_raw.get("exposure_policy") or "safe"),
            tool_profile=str(tool_bridge_raw.get("toolProfile") or tool_bridge_raw.get("tool_profile") or "auto"),
            semantic_final_judge=str(
                tool_bridge_raw.get("semanticFinalJudge") or tool_bridge_raw.get("semantic_final_judge") or "off"
            ),
            allowed_tool_names=tuple(str(name) for name in allowed_tool_names if str(name).strip()),
            readonly_tool_names=_string_tuple(readonly_tool_names),
            write_tool_names=_string_tuple(write_tool_names),
            shell_tool_names=_string_tuple(shell_tool_names),
            observation_policy=_load_observation_policy(tool_bridge_raw),
        ),
    )


def config_to_public(config: GatewayConfig) -> dict[str, Any]:
    return {
        "server": {"host": config.server.host, "port": config.server.port},
        "upstream": {
            "baseUrl": config.upstream.base_url,
            "model": config.upstream.model,
            "toolMode": config.upstream.tool_mode,
        },
        "providerRuntime": {
            "requestTimeoutSeconds": config.provider_runtime.request_timeout_seconds,
            "promptMaxChars": config.provider_runtime.prompt_max_chars,
            "nativeWebSearchPolicy": config.provider_runtime.native_web_search_policy,
            "responseLanguage": config.provider_runtime.response_language,
            "deepseekDs2apiBaseUrl": config.provider_runtime.deepseek_ds2api_base_url,
            "deepseekDs2apiAccountMaxInflight": config.provider_runtime.deepseek_ds2api_account_max_inflight,
            "deepseekDs2apiGlobalMaxInflight": config.provider_runtime.deepseek_ds2api_global_max_inflight,
            "deepseekDs2apiBearerMaxInflight": config.provider_runtime.deepseek_ds2api_bearer_max_inflight,
            "deepseekDs2apiRateLimitCooldownSeconds": config.provider_runtime.deepseek_ds2api_rate_limit_cooldown_seconds,
            "deepseekDs2apiCurrentInputFileEnabled": config.provider_runtime.deepseek_ds2api_current_input_file_enabled,
            "deepseekDs2apiCurrentInputFileMinChars": config.provider_runtime.deepseek_ds2api_current_input_file_min_chars,
            "qwenWebBackend": config.provider_runtime.qwen_web_backend,
            "gptThinkingBackend": config.provider_runtime.gpt_thinking_backend,
        },
        "tool_bridge": {
            "mode": config.tool_bridge.mode,
            "activationPolicy": config.tool_bridge.activation_policy,
            "maxToolsInPrompt": config.tool_bridge.max_tools_in_prompt,
            "maxCallsPerTurn": config.tool_bridge.max_calls_per_turn,
            "maxReadonlyCallsPerTurn": config.tool_bridge.max_readonly_calls_per_turn,
            "toolPromptMaxChars": config.tool_bridge.tool_prompt_max_chars,
            "observationMaxChars": config.tool_bridge.observation_max_chars,
            "exposurePolicy": config.tool_bridge.exposure_policy,
            "toolProfile": config.tool_bridge.tool_profile,
            "semanticFinalJudge": config.tool_bridge.semantic_final_judge,
            "allowedToolNames": list(config.tool_bridge.allowed_tool_names),
            "readonlyToolNames": list(config.tool_bridge.readonly_tool_names),
            "writeToolNames": list(config.tool_bridge.write_tool_names),
            "shellToolNames": list(config.tool_bridge.shell_tool_names),
            "observationPolicy": _observation_policy_to_json(config.tool_bridge.observation_policy),
        },
    }


def config_to_admin(config: GatewayConfig) -> dict[str, Any]:
    return {
        "server": {
            "host": config.server.host,
            "port": config.server.port,
            "apiKey": config.server.api_key,
        },
        "upstream": {
            "baseUrl": config.upstream.base_url,
            "apiKey": config.upstream.api_key,
            "model": config.upstream.model,
            "toolMode": config.upstream.tool_mode,
        },
        "providerRuntime": {
            "requestTimeoutSeconds": config.provider_runtime.request_timeout_seconds,
            "promptMaxChars": config.provider_runtime.prompt_max_chars,
            "nativeWebSearchPolicy": config.provider_runtime.native_web_search_policy,
            "responseLanguage": config.provider_runtime.response_language,
            "deepseekDs2apiBaseUrl": config.provider_runtime.deepseek_ds2api_base_url,
            "deepseekDs2apiAccountMaxInflight": config.provider_runtime.deepseek_ds2api_account_max_inflight,
            "deepseekDs2apiGlobalMaxInflight": config.provider_runtime.deepseek_ds2api_global_max_inflight,
            "deepseekDs2apiBearerMaxInflight": config.provider_runtime.deepseek_ds2api_bearer_max_inflight,
            "deepseekDs2apiRateLimitCooldownSeconds": config.provider_runtime.deepseek_ds2api_rate_limit_cooldown_seconds,
            "deepseekDs2apiCurrentInputFileEnabled": config.provider_runtime.deepseek_ds2api_current_input_file_enabled,
            "deepseekDs2apiCurrentInputFileMinChars": config.provider_runtime.deepseek_ds2api_current_input_file_min_chars,
            "qwenWebBackend": config.provider_runtime.qwen_web_backend,
            "gptThinkingBackend": config.provider_runtime.gpt_thinking_backend,
        },
        "tool_bridge": {
            "mode": config.tool_bridge.mode,
            "activationPolicy": config.tool_bridge.activation_policy,
            "maxToolsInPrompt": config.tool_bridge.max_tools_in_prompt,
            "maxCallsPerTurn": config.tool_bridge.max_calls_per_turn,
            "maxReadonlyCallsPerTurn": config.tool_bridge.max_readonly_calls_per_turn,
            "toolPromptMaxChars": config.tool_bridge.tool_prompt_max_chars,
            "observationMaxChars": config.tool_bridge.observation_max_chars,
            "exposurePolicy": config.tool_bridge.exposure_policy,
            "toolProfile": config.tool_bridge.tool_profile,
            "semanticFinalJudge": config.tool_bridge.semantic_final_judge,
            "allowedToolNames": list(config.tool_bridge.allowed_tool_names),
            "readonlyToolNames": list(config.tool_bridge.readonly_tool_names),
            "writeToolNames": list(config.tool_bridge.write_tool_names),
            "shellToolNames": list(config.tool_bridge.shell_tool_names),
            "observationPolicy": _observation_policy_to_json(config.tool_bridge.observation_policy),
        },
    }


def save_config(config: GatewayConfig, path: str | Path = "config.json") -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(p.suffix + ".tmp")
    tmp.write_text(json.dumps(config_to_admin(config), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    tmp.replace(p)


def update_config(config: GatewayConfig, payload: dict[str, Any]) -> GatewayConfig:
    server_raw = payload.get("server") if isinstance(payload.get("server"), dict) else {}
    upstream_raw = payload.get("upstream") if isinstance(payload.get("upstream"), dict) else {}
    provider_runtime_raw = (
        payload.get("providerRuntime")
        if isinstance(payload.get("providerRuntime"), dict)
        else payload.get("provider_runtime") if isinstance(payload.get("provider_runtime"), dict) else {}
    )
    tool_bridge_raw = payload.get("tool_bridge") if isinstance(payload.get("tool_bridge"), dict) else {}
    deepseek_ds2api_account_max_inflight = _bounded_int(
        _raw_first(
            provider_runtime_raw,
            "deepseekDs2apiAccountMaxInflight",
            "deepseek_ds2api_account_max_inflight",
            default=config.provider_runtime.deepseek_ds2api_account_max_inflight,
        ),
        default=config.provider_runtime.deepseek_ds2api_account_max_inflight,
        minimum=1,
        maximum=256,
    )
    deepseek_ds2api_global_max_inflight = max(
        deepseek_ds2api_account_max_inflight,
        _bounded_int(
            _raw_first(
                provider_runtime_raw,
                "deepseekDs2apiGlobalMaxInflight",
                "deepseek_ds2api_global_max_inflight",
                default=config.provider_runtime.deepseek_ds2api_global_max_inflight,
            ),
            default=config.provider_runtime.deepseek_ds2api_global_max_inflight,
            minimum=1,
        ),
    )
    deepseek_ds2api_bearer_max_inflight = _bounded_int(
        _raw_first(
            provider_runtime_raw,
            "deepseekDs2apiBearerMaxInflight",
            "deepseek_ds2api_bearer_max_inflight",
            default=config.provider_runtime.deepseek_ds2api_bearer_max_inflight,
        ),
        default=config.provider_runtime.deepseek_ds2api_bearer_max_inflight,
        minimum=1,
        maximum=256,
    )
    deepseek_ds2api_rate_limit_cooldown_seconds = _bounded_float(
        _raw_first(
            provider_runtime_raw,
            "deepseekDs2apiRateLimitCooldownSeconds",
            "deepseek_ds2api_rate_limit_cooldown_seconds",
            default=config.provider_runtime.deepseek_ds2api_rate_limit_cooldown_seconds,
        ),
        default=config.provider_runtime.deepseek_ds2api_rate_limit_cooldown_seconds,
        minimum=0.0,
        maximum=300.0,
    )
    deepseek_ds2api_current_input_file_enabled = _bool_value(
        _raw_first(
            provider_runtime_raw,
            "deepseekDs2apiCurrentInputFileEnabled",
            "deepseek_ds2api_current_input_file_enabled",
            default=config.provider_runtime.deepseek_ds2api_current_input_file_enabled,
        )
    )
    deepseek_ds2api_current_input_file_min_chars = _bounded_int(
        _raw_first(
            provider_runtime_raw,
            "deepseekDs2apiCurrentInputFileMinChars",
            "deepseek_ds2api_current_input_file_min_chars",
            default=config.provider_runtime.deepseek_ds2api_current_input_file_min_chars,
        ),
        default=config.provider_runtime.deepseek_ds2api_current_input_file_min_chars,
        minimum=0,
        maximum=100_000_000,
    )
    allowed_tool_names = (
        tool_bridge_raw.get("allowedToolNames")
        if "allowedToolNames" in tool_bridge_raw
        else tool_bridge_raw.get("allowed_tool_names", list(config.tool_bridge.allowed_tool_names))
    )
    if not isinstance(allowed_tool_names, list):
        allowed_tool_names = list(config.tool_bridge.allowed_tool_names)
    readonly_tool_names = (
        tool_bridge_raw.get("readonlyToolNames")
        if "readonlyToolNames" in tool_bridge_raw
        else tool_bridge_raw.get("readonly_tool_names", list(config.tool_bridge.readonly_tool_names))
    )
    write_tool_names = (
        tool_bridge_raw.get("writeToolNames")
        if "writeToolNames" in tool_bridge_raw
        else tool_bridge_raw.get("write_tool_names", list(config.tool_bridge.write_tool_names))
    )
    shell_tool_names = (
        tool_bridge_raw.get("shellToolNames")
        if "shellToolNames" in tool_bridge_raw
        else tool_bridge_raw.get("shell_tool_names", list(config.tool_bridge.shell_tool_names))
    )
    return GatewayConfig(
        server=ServerConfig(
            host=str(server_raw.get("host") or config.server.host),
            port=int(server_raw.get("port") or config.server.port),
            api_key=str(server_raw.get("apiKey") if "apiKey" in server_raw else server_raw.get("api_key", config.server.api_key)),
        ),
        upstream=UpstreamConfig(
            base_url=str(upstream_raw.get("baseUrl") or upstream_raw.get("base_url") or config.upstream.base_url),
            api_key=str(upstream_raw.get("apiKey") if "apiKey" in upstream_raw else upstream_raw.get("api_key", config.upstream.api_key)),
            model=str(upstream_raw.get("model") or config.upstream.model),
            tool_mode=str(upstream_raw.get("toolMode") or upstream_raw.get("tool_mode") or config.upstream.tool_mode),
        ),
        provider_runtime=ProviderRuntimeConfig(
            request_timeout_seconds=max(
                30,
                int(
                    provider_runtime_raw.get("requestTimeoutSeconds")
                    if "requestTimeoutSeconds" in provider_runtime_raw
                    else provider_runtime_raw.get("request_timeout_seconds", config.provider_runtime.request_timeout_seconds)
                ),
            ),
            prompt_max_chars=max(
                4000,
                int(
                    provider_runtime_raw.get("promptMaxChars")
                    if "promptMaxChars" in provider_runtime_raw
                    else provider_runtime_raw.get("prompt_max_chars", config.provider_runtime.prompt_max_chars)
                ),
            ),
            native_web_search_policy=str(
                provider_runtime_raw.get("nativeWebSearchPolicy")
                if "nativeWebSearchPolicy" in provider_runtime_raw
                else provider_runtime_raw.get("native_web_search_policy", config.provider_runtime.native_web_search_policy)
            ),
            response_language=str(
                provider_runtime_raw.get("responseLanguage")
                if "responseLanguage" in provider_runtime_raw
                else provider_runtime_raw.get("response_language", config.provider_runtime.response_language)
            ),
            deepseek_ds2api_base_url=str(
                provider_runtime_raw.get("deepseekDs2apiBaseUrl")
                if "deepseekDs2apiBaseUrl" in provider_runtime_raw
                else provider_runtime_raw.get("deepseek_ds2api_base_url", config.provider_runtime.deepseek_ds2api_base_url)
            ),
            deepseek_ds2api_account_max_inflight=deepseek_ds2api_account_max_inflight,
            deepseek_ds2api_global_max_inflight=deepseek_ds2api_global_max_inflight,
            deepseek_ds2api_bearer_max_inflight=deepseek_ds2api_bearer_max_inflight,
            deepseek_ds2api_rate_limit_cooldown_seconds=deepseek_ds2api_rate_limit_cooldown_seconds,
            deepseek_ds2api_current_input_file_enabled=deepseek_ds2api_current_input_file_enabled,
            deepseek_ds2api_current_input_file_min_chars=deepseek_ds2api_current_input_file_min_chars,
            qwen_web_backend=str(
                provider_runtime_raw.get("qwenWebBackend")
                if "qwenWebBackend" in provider_runtime_raw
                else provider_runtime_raw.get("qwen_web_backend", config.provider_runtime.qwen_web_backend)
            ),
            gpt_thinking_backend=str(
                provider_runtime_raw.get("gptThinkingBackend")
                if "gptThinkingBackend" in provider_runtime_raw
                else provider_runtime_raw.get("gpt_thinking_backend", config.provider_runtime.gpt_thinking_backend)
            ),
        ),
        tool_bridge=ToolBridgeConfig(
            mode=str(tool_bridge_raw.get("mode") or config.tool_bridge.mode),
            activation_policy=str(
                tool_bridge_raw.get("activationPolicy")
                if "activationPolicy" in tool_bridge_raw
                else tool_bridge_raw.get("activation_policy", config.tool_bridge.activation_policy)
            ),
            max_tools_in_prompt=int(
                tool_bridge_raw.get("maxToolsInPrompt")
                if "maxToolsInPrompt" in tool_bridge_raw
                else tool_bridge_raw.get("max_tools_in_prompt", config.tool_bridge.max_tools_in_prompt)
            ),
            max_calls_per_turn=int(
                tool_bridge_raw.get("maxCallsPerTurn")
                if "maxCallsPerTurn" in tool_bridge_raw
                else tool_bridge_raw.get("max_calls_per_turn", config.tool_bridge.max_calls_per_turn)
            ),
            max_readonly_calls_per_turn=int(
                tool_bridge_raw.get("maxReadonlyCallsPerTurn")
                if "maxReadonlyCallsPerTurn" in tool_bridge_raw
                else tool_bridge_raw.get("max_readonly_calls_per_turn", config.tool_bridge.max_readonly_calls_per_turn)
            ),
            tool_prompt_max_chars=int(
                tool_bridge_raw.get("toolPromptMaxChars")
                if "toolPromptMaxChars" in tool_bridge_raw
                else tool_bridge_raw.get("tool_prompt_max_chars", config.tool_bridge.tool_prompt_max_chars)
            ),
            observation_max_chars=int(
                tool_bridge_raw.get("observationMaxChars")
                if "observationMaxChars" in tool_bridge_raw
                else tool_bridge_raw.get("observation_max_chars", config.tool_bridge.observation_max_chars)
            ),
            exposure_policy=str(
                tool_bridge_raw.get("exposurePolicy")
                if "exposurePolicy" in tool_bridge_raw
                else tool_bridge_raw.get("exposure_policy", config.tool_bridge.exposure_policy)
            ),
            tool_profile=str(
                tool_bridge_raw.get("toolProfile")
                if "toolProfile" in tool_bridge_raw
                else tool_bridge_raw.get("tool_profile", config.tool_bridge.tool_profile)
            ),
            semantic_final_judge=str(
                tool_bridge_raw.get("semanticFinalJudge")
                if "semanticFinalJudge" in tool_bridge_raw
                else tool_bridge_raw.get("semantic_final_judge", config.tool_bridge.semantic_final_judge)
            ),
            allowed_tool_names=tuple(str(name) for name in allowed_tool_names if str(name).strip()),
            readonly_tool_names=_string_tuple(readonly_tool_names, default=config.tool_bridge.readonly_tool_names),
            write_tool_names=_string_tuple(write_tool_names, default=config.tool_bridge.write_tool_names),
            shell_tool_names=_string_tuple(shell_tool_names, default=config.tool_bridge.shell_tool_names),
            observation_policy=_load_observation_policy(tool_bridge_raw, default=config.tool_bridge.observation_policy),
        ),
    )


def _load_observation_policy(raw: dict[str, Any], default: ObservationPolicyConfig | None = None) -> ObservationPolicyConfig:
    base = default or ObservationPolicyConfig()
    policy_raw = raw.get("observationPolicy") if isinstance(raw.get("observationPolicy"), dict) else raw.get("observation_policy")
    if not isinstance(policy_raw, dict):
        return base
    return ObservationPolicyConfig(
        summarize_path_lists=_bool_value(
            policy_raw.get("summarizePathLists")
            if "summarizePathLists" in policy_raw
            else policy_raw.get("summarize_path_lists", base.summarize_path_lists)
        ),
        excluded_path_parts=_string_tuple(
            policy_raw.get("excludedPathParts")
            if "excludedPathParts" in policy_raw
            else policy_raw.get("excluded_path_parts", base.excluded_path_parts),
            default=base.excluded_path_parts,
        ),
        excluded_path_globs=_string_tuple(
            policy_raw.get("excludedPathGlobs")
            if "excludedPathGlobs" in policy_raw
            else policy_raw.get("excluded_path_globs", base.excluded_path_globs),
            default=base.excluded_path_globs,
        ),
        path_list_max_items=max(
            1,
            int(
                policy_raw.get("pathListMaxItems")
                if "pathListMaxItems" in policy_raw
                else policy_raw.get("path_list_max_items", base.path_list_max_items)
            ),
        ),
    )


def _observation_policy_to_json(policy: ObservationPolicyConfig) -> dict[str, Any]:
    return {
        "summarizePathLists": policy.summarize_path_lists,
        "excludedPathParts": list(policy.excluded_path_parts),
        "excludedPathGlobs": list(policy.excluded_path_globs),
        "pathListMaxItems": policy.path_list_max_items,
    }


def _string_tuple(value: Any, *, default: tuple[str, ...] = ()) -> tuple[str, ...]:
    if value is None:
        return default
    if isinstance(value, str):
        items = value.replace(",", "\n").splitlines()
    elif isinstance(value, (list, tuple)):
        items = value
    else:
        return default
    out: list[str] = []
    seen: set[str] = set()
    for item in items:
        text = str(item).strip()
        if not text or text in seen:
            continue
        seen.add(text)
        out.append(text)
    return tuple(out)


def _raw_first(raw: dict[str, Any], *keys: str, default: Any = None) -> Any:
    for key in keys:
        if key in raw:
            return raw[key]
    return default


def _bounded_int(value: Any, *, default: int, minimum: int, maximum: int | None = None) -> int:
    try:
        result = int(value)
    except (TypeError, ValueError):
        result = int(default)
    result = max(minimum, result)
    if maximum is not None:
        result = min(maximum, result)
    return result


def _bounded_float(value: Any, *, default: float, minimum: float, maximum: float | None = None) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError):
        result = float(default)
    result = max(minimum, result)
    if maximum is not None:
        result = min(maximum, result)
    return result


def _bool_value(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() not in {"0", "false", "no", "off"}
    return bool(value)
