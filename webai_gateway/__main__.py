from __future__ import annotations

import uvicorn

from webai_gateway.config import load_config


def main() -> None:
    config = load_config()
    uvicorn.run(
        "webai_gateway.app:app",
        host=config.server.host,
        port=config.server.port,
        reload=False,
    )


if __name__ == "__main__":
    main()
