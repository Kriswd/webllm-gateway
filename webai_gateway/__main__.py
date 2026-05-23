from __future__ import annotations

from pathlib import Path
import sys

import uvicorn

from webai_gateway.config import load_config
from webai_gateway.runtime_supervisor import ensure_managed_runtimes


def main() -> None:
    config_path = Path("config.json").resolve()
    config = load_config(config_path)
    try:
        ensure_managed_runtimes(config, config_path, config_path.parent)
    except Exception as exc:
        print(f"内部 runtime supervisor 启动失败，Gateway 将继续启动：{exc}", file=sys.stderr)
    uvicorn.run(
        "webai_gateway.app:app",
        host=config.server.host,
        port=config.server.port,
        reload=False,
    )


if __name__ == "__main__":
    main()
