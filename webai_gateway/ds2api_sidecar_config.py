from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Sequence

from webai_gateway.config import GatewayConfig, load_config


def build_ds2api_sidecar_config(config: GatewayConfig) -> dict[str, Any]:
    account_max_inflight = max(1, min(256, int(config.provider_runtime.deepseek_ds2api_account_max_inflight)))
    global_max_inflight = max(account_max_inflight, int(config.provider_runtime.deepseek_ds2api_global_max_inflight))
    return {
        "keys": [],
        "accounts": [],
        "runtime": {
            "global_max_inflight": global_max_inflight,
            "account_max_inflight": account_max_inflight,
        },
        "current_input_file": {
            "enabled": bool(config.provider_runtime.deepseek_ds2api_current_input_file_enabled),
            "min_chars": max(0, int(config.provider_runtime.deepseek_ds2api_current_input_file_min_chars)),
        },
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Render ds2api sidecar runtime config from WebLLM Gateway config.")
    parser.add_argument("--config", default="config.json", help="Path to WebLLM Gateway config.json")
    args = parser.parse_args(argv)
    config = load_config(Path(args.config))
    print(json.dumps(build_ds2api_sidecar_config(config), ensure_ascii=True, separators=(",", ":")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
