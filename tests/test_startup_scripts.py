from pathlib import Path


def test_startup_bat_starts_webai2api_sidecar_without_bootstrap_mutation() -> None:
    root = Path(__file__).resolve().parents[1]
    script = (root / "start_webai_gateway.bat").read_text(encoding="utf-8")
    lowered = script.lower()

    assert "WEBAI2API_SIDECAR_DIR" in script
    assert "WebAI2API sidecar" in script
    assert ":8500.*LISTENING" in script
    assert "pnpm','start" in script
    assert "%%~fI" not in script
    assert "pnpm install" not in lowered
    assert "pnpm run init" not in lowered
