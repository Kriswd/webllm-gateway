from pathlib import Path


def test_startup_bat_delegates_internal_runtimes_without_bootstrap_mutation() -> None:
    root = Path(__file__).resolve().parents[1]
    script = (root / "start_webai_gateway.bat").read_text(encoding="utf-8")
    lowered = script.lower()

    assert "webai_gateway.runtime_supervisor" in script
    assert "--ensure" in script
    assert "WEBAI2API_SIDECAR_DIR" not in script
    assert "netstat -ano" not in script
    assert "pnpm','start" not in script
    assert "%%~fI" not in script
    assert "pnpm install" not in lowered
    assert "pnpm run init" not in lowered
