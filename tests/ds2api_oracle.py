from __future__ import annotations

import io
import json
import os
import shutil
import subprocess
import tarfile
import textwrap
from urllib.request import Request, urlopen
from pathlib import Path
from typing import Any

from webai_gateway.ds2api_oracle import DS2API_ORACLE_COMMIT, DS2API_ORACLE_VERSION
DS2API_ORACLE_REPO_URL = "https://github.com/CJackHwang/ds2api"
DS2API_ORACLE_API_URL = "https://api.github.com/repos/CJackHwang/ds2api/commits/main"


DS2API_RUNNER_SOURCE = r'''
package main

import (
	"encoding/json"
	"fmt"
	"os"

	"ds2api/internal/assistantturn"
	"ds2api/internal/promptcompat"
	"ds2api/internal/sse"
	"ds2api/internal/toolcall"
)

type request struct {
	Mode               string `json:"mode"`
	Text               string `json:"text"`
	Thinking           string `json:"thinking"`
	DetectionThinking  string `json:"detectionThinking"`
	Names              []string `json:"names"`
	ToolsRaw           any `json:"toolsRaw"`
	ToolChoice         string `json:"toolChoice"`
	ContentFilter      bool `json:"contentFilter"`
}

type outputError struct {
	Status  int `json:"status"`
	Message string `json:"message"`
	Code    string `json:"code"`
}

type response struct {
	Calls             []toolcall.ParsedToolCall `json:"calls"`
	SawToolCallSyntax bool `json:"sawToolCallSyntax"`
	RejectedByPolicy  bool `json:"rejectedByPolicy"`
	RejectedToolNames []string `json:"rejectedToolNames"`
	OpenAI            []map[string]any `json:"openai,omitempty"`
	Stream            []map[string]any `json:"stream,omitempty"`
	Turn              map[string]any `json:"turn,omitempty"`
}

func main() {
	var req request
	if err := json.NewDecoder(os.Stdin).Decode(&req); err != nil {
		fmt.Fprintf(os.Stderr, "decode request: %v\n", err)
		os.Exit(2)
	}
	var parsed toolcall.ToolCallParseResult
	switch req.Mode {
	case "assistant":
		parsed = toolcall.ParseAssistantToolCallsDetailed(req.Text, req.Thinking, req.Names)
	case "turn":
		policy := promptcompat.DefaultToolChoicePolicy()
		switch req.ToolChoice {
		case "required":
			policy.Mode = promptcompat.ToolChoiceRequired
		case "forced":
			policy.Mode = promptcompat.ToolChoiceForced
		case "none":
			policy.Mode = promptcompat.ToolChoiceNone
		}
		turn := assistantturn.BuildTurnFromCollected(
			sse.CollectResult{
				Text: req.Text,
				Thinking: req.Thinking,
				ToolDetectionThinking: req.DetectionThinking,
				ContentFilter: req.ContentFilter,
			},
			assistantturn.BuildOptions{
				Model: "deepseek-v4-pro",
				Prompt: "",
				ToolNames: req.Names,
				ToolsRaw: req.ToolsRaw,
				ToolChoice: policy,
			},
		)
		final := assistantturn.FinalizeTurn(turn, assistantturn.FinalizeOptions{})
		errPayload := any(nil)
		if final.Error != nil {
			errPayload = outputError{
				Status: final.Error.Status,
				Message: final.Error.Message,
				Code: final.Error.Code,
			}
		}
		resp := response{
			Calls: turn.ToolCalls,
			SawToolCallSyntax: turn.ParsedToolCalls.SawToolCallSyntax,
			RejectedByPolicy: turn.ParsedToolCalls.RejectedByPolicy,
			RejectedToolNames: turn.ParsedToolCalls.RejectedToolNames,
			Turn: map[string]any{
				"text": turn.Text,
				"thinking": turn.Thinking,
				"finishReason": final.FinishReason,
				"hasToolCalls": final.HasToolCalls,
				"hasVisibleText": final.HasVisibleText,
				"hasVisibleOutput": final.HasVisibleOutput,
				"shouldFail": final.ShouldFail,
				"error": errPayload,
			},
		}
		if err := json.NewEncoder(os.Stdout).Encode(resp); err != nil {
			fmt.Fprintf(os.Stderr, "encode response: %v\n", err)
			os.Exit(2)
		}
		return
	default:
		parsed = toolcall.ParseToolCallsDetailed(req.Text, req.Names)
	}
	resp := response{
		Calls: parsed.Calls,
		SawToolCallSyntax: parsed.SawToolCallSyntax,
		RejectedByPolicy: parsed.RejectedByPolicy,
		RejectedToolNames: parsed.RejectedToolNames,
	}
	if req.Mode == "format" {
		resp.OpenAI = toolcall.FormatOpenAIToolCalls(parsed.Calls, req.ToolsRaw)
	}
	if req.Mode == "stream" {
		resp.Stream = toolcall.FormatOpenAIStreamToolCalls(parsed.Calls, req.ToolsRaw)
	}
	if err := json.NewEncoder(os.Stdout).Encode(resp); err != nil {
		fmt.Fprintf(os.Stderr, "encode response: %v\n", err)
		os.Exit(2)
	}
}
'''


def assert_oracle_is_latest() -> None:
    latest = _latest_remote_commit()
    assert latest == DS2API_ORACLE_COMMIT, (
        f"ds2api oracle is stale: expected latest main {latest}, "
        f"but tests pin {DS2API_ORACLE_COMMIT}"
    )


def build_ds2api_runner(workdir: Path) -> Path:
    snapshot = workdir / "ds2api-src"
    _export_reference_snapshot(DS2API_ORACLE_COMMIT, snapshot)
    (workdir / "go.mod").write_text(
        textwrap.dedent(
            f"""
            module ds2api/parityrunner

            go 1.26.0

            require ds2api v0.0.0

            replace ds2api => {snapshot.as_posix()}
            """
        ).strip()
        + "\n",
        encoding="utf-8",
    )
    (workdir / "main.go").write_text(DS2API_RUNNER_SOURCE, encoding="utf-8")
    exe = workdir / ("ds2api-oracle-runner.exe" if os.name == "nt" else "ds2api-oracle-runner")
    subprocess.run(
        [_go_executable(), "build", "-mod=mod", "-o", str(exe), "."],
        cwd=workdir,
        check=True,
        capture_output=True,
        text=True,
        env=_go_test_env(workdir),
    )
    return exe


def _go_executable(*, extra_candidates: list[Path] | None = None) -> str:
    found = shutil.which("go")
    if found:
        return found
    candidates = list(extra_candidates or [])
    goroot = os.environ.get("GOROOT")
    if goroot:
        candidates.append(Path(goroot) / "bin" / ("go.exe" if os.name == "nt" else "go"))
    if os.name == "nt":
        for root in (os.environ.get("ProgramFiles"), os.environ.get("ProgramFiles(x86)"), os.environ.get("LocalAppData")):
            if root:
                candidates.append(Path(root) / "Go" / "bin" / "go.exe")
        candidates.append(Path("C:/Program Files/Go/bin/go.exe"))
    for candidate in candidates:
        if candidate.exists():
            return str(candidate)
    raise FileNotFoundError("Go executable not found; install Go or set GOROOT/PATH before running ds2api oracle tests.")


def run_ds2api_runner(
    runner: Path,
    *,
    text: str,
    names: list[str],
    mode: str = "parse",
    thinking: str = "",
    detection_thinking: str = "",
    tools: list[dict[str, Any]] | None = None,
    tool_choice: str = "auto",
    content_filter: bool = False,
) -> dict[str, Any]:
    request = {
        "mode": mode,
        "text": text,
        "thinking": thinking,
        "detectionThinking": detection_thinking,
        "names": names,
        "toolsRaw": tools if tools is not None else _tools_for_names(names),
        "toolChoice": tool_choice,
        "contentFilter": content_filter,
    }
    completed = subprocess.run(
        [str(runner)],
        input=json.dumps(request, ensure_ascii=False),
        check=True,
        capture_output=True,
        text=True,
    )
    response = json.loads(completed.stdout)
    response["calls"] = response.get("calls") or []
    response["openai"] = response.get("openai") or []
    response["stream"] = response.get("stream") or []
    response["turn"] = response.get("turn") or {}
    return response


def resolve_reference_root(commit: str) -> Path:
    candidates = []
    env_root = os.environ.get("DS2API_REFERENCE_ROOT")
    if env_root:
        candidates.append(Path(env_root))
    candidates.extend(
        [
            Path(r"E:\ProjectX\_reference\ds2api"),
            Path(__file__).resolve().parents[1] / ".tmp" / "ds2api",
        ]
    )
    for candidate in candidates:
        if candidate.exists() and _has_git_commit(candidate, commit):
            return candidate
    formatted = ", ".join(str(path) for path in candidates)
    raise AssertionError(f"ds2api reference commit {commit} not found in: {formatted}")


def _latest_remote_commit() -> str:
    git_error = ""
    try:
        completed = subprocess.run(
            ["git", "ls-remote", DS2API_ORACLE_REPO_URL, "refs/heads/main"],
            check=True,
            capture_output=True,
            text=True,
            env=_git_network_env(),
        )
        latest = completed.stdout.split()[0].strip()
        if latest:
            return latest
    except (OSError, subprocess.CalledProcessError) as exc:
        git_error = str(exc)

    request = Request(DS2API_ORACLE_API_URL, headers={"User-Agent": "webai-gateway-tests"})
    try:
        with urlopen(request, timeout=30) as response:
            payload = json.loads(response.read().decode("utf-8"))
        latest = str(payload.get("sha") or "").strip()
        if latest:
            return latest
    except Exception as exc:  # pragma: no cover - only exercised during network outages.
        raise AssertionError(
            "unable to check ds2api latest commit via git or GitHub API; "
            f"git error: {git_error}; api error: {exc}"
        ) from exc
    raise AssertionError("unable to read ds2api latest commit from GitHub API response")


def _export_reference_snapshot(commit: str, destination: Path) -> None:
    try:
        root = resolve_reference_root(commit)
    except AssertionError:
        archive = _find_or_download_reference_archive(commit)
        _extract_archive_snapshot(archive, destination)
        return
    _export_git_snapshot(root, commit, destination)


def _export_git_snapshot(root: Path, commit: str, destination: Path) -> None:
    destination.mkdir(parents=True, exist_ok=True)
    archive = subprocess.run(
        ["git", "-C", str(root), "archive", "--format=tar", commit],
        check=True,
        capture_output=True,
    )
    with tarfile.open(fileobj=io.BytesIO(archive.stdout), mode="r:") as tar:
        tar.extractall(destination)


def _find_or_download_reference_archive(commit: str) -> Path:
    repo_root = Path(__file__).resolve().parents[1]
    cache_dir = repo_root / ".tmp"
    candidates = [
        cache_dir / f"ds2api-{commit[:8]}.tar.gz",
        cache_dir / f"ds2api-{commit}.tar.gz",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    cache_dir.mkdir(parents=True, exist_ok=True)
    archive = candidates[0]
    request = Request(
        f"https://github.com/CJackHwang/ds2api/archive/{commit}.tar.gz",
        headers={"User-Agent": "webai-gateway-tests"},
    )
    with urlopen(request, timeout=60) as response, archive.open("wb") as handle:
        shutil.copyfileobj(response, handle)
    return archive


def _extract_archive_snapshot(archive: Path, destination: Path) -> None:
    destination.mkdir(parents=True, exist_ok=True)
    destination_root = destination.resolve()
    with tarfile.open(archive, mode="r:gz") as tar:
        for member in tar.getmembers():
            parts = member.name.split("/", 1)
            if len(parts) != 2 or not parts[1]:
                continue
            target = (destination / parts[1]).resolve()
            if not target.is_relative_to(destination_root):
                raise AssertionError(f"unsafe path in ds2api archive: {member.name}")
            if member.isdir():
                target.mkdir(parents=True, exist_ok=True)
                continue
            source = tar.extractfile(member)
            if source is None:
                continue
            target.parent.mkdir(parents=True, exist_ok=True)
            with source, target.open("wb") as handle:
                shutil.copyfileobj(source, handle)


def _has_git_commit(root: Path, commit: str) -> bool:
    completed = subprocess.run(
        ["git", "-C", str(root), "cat-file", "-e", f"{commit}^{{commit}}"],
        capture_output=True,
        text=True,
    )
    return completed.returncode == 0


def _go_test_env(workdir: Path) -> dict[str, str]:
    env = os.environ.copy()
    mod_cache, build_cache = _go_cache_paths(workdir)
    mod_cache.mkdir(parents=True, exist_ok=True)
    build_cache.mkdir(parents=True, exist_ok=True)
    env["GOMODCACHE"] = str(mod_cache)
    env["GOCACHE"] = str(build_cache)
    env.setdefault("GOSUMDB", "off")
    local_go_bin = _local_go_toolchain_bin()
    if local_go_bin:
        env["PATH"] = str(local_go_bin) + os.pathsep + env.get("PATH", "")
    return env


def _go_cache_paths(workdir: Path) -> tuple[Path, Path]:
    env_mod = os.environ.get("GOMODCACHE")
    env_build = os.environ.get("GOCACHE")
    if env_mod and env_build:
        return Path(env_mod), Path(env_build)
    if Path(r"C:\go-mod-cache-ds2api").exists() and Path(r"C:\go-build-cache-ds2api").exists():
        return Path(r"C:\go-mod-cache-ds2api"), Path(r"C:\go-build-cache-ds2api")
    cache_base = Path(os.environ.get("CODEX_GO_TEST_CACHE", r"D:\CodexCache\go"))
    if not cache_base.drive or not Path(cache_base.drive + "\\").exists():
        cache_base = workdir / ".go-cache"
    return cache_base / "mod", cache_base / "build"


def _local_go_toolchain_bin() -> Path | None:
    repo_root = Path(__file__).resolve().parents[1]
    toolchain_root = repo_root / ".tmp" / "go-mod" / "golang.org"
    exe_name = "go.exe" if os.name == "nt" else "go"
    for candidate in toolchain_root.glob("toolchain@*/bin"):
        if (candidate / exe_name).exists():
            return candidate
    return None


def _git_network_env() -> dict[str, str]:
    env = os.environ.copy()
    for key in ("HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY", "http_proxy", "https_proxy", "all_proxy"):
        env.pop(key, None)
    return env


def _tools_for_names(names: list[str]) -> list[dict[str, Any]]:
    return [
        {
            "type": "function",
            "function": {
                "name": name,
                "description": f"{name} tool",
                "parameters": {"type": "object"},
            },
        }
        for name in names
    ]
