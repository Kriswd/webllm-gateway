# Third-Party Notices

This file summarizes the main third-party projects referenced by or used in
WebLLM Gateway. It is a release aid, not a substitute for checking each upstream
license file before publishing a packaged distribution.

## Optional adapter runtimes

| Project | Purpose | License | Source |
| --- | --- | --- | --- |
| WebAI2API | Optional sidecar runtime for selected web-login providers | MIT | https://github.com/foxhui/WebAI2API |
| ds2api | Optional DeepSeek web runtime and oracle reference | GNU AGPL-3.0 | https://github.com/CJackHwang/ds2api |

WebLLM Gateway does not vendor either runtime in the public source tree. If a
downstream distribution bundles, modifies, or hosts these runtimes, the
downstream distributor must comply with the corresponding upstream license.

## Python service dependencies

| Project | Purpose | License | Source |
| --- | --- | --- | --- |
| FastAPI | HTTP API framework | MIT | https://github.com/fastapi/fastapi |
| Uvicorn | ASGI server | BSD-3-Clause | https://github.com/encode/uvicorn |
| HTTPX / httpcore | HTTP client stack | BSD-3-Clause | https://github.com/encode/httpx |
| Pydantic | Data validation | MIT | https://github.com/pydantic/pydantic |
| Playwright Python | Browser automation used for web login providers | Apache-2.0 | https://github.com/microsoft/playwright-python |
| pytest | Test runner | MIT | https://github.com/pytest-dev/pytest |

## Web UI dependencies

| Project | Purpose | License | Source |
| --- | --- | --- | --- |
| Vue | Frontend framework | MIT | https://github.com/vuejs/core |
| Ant Design Vue | UI component library | MIT | https://github.com/vueComponent/ant-design-vue |
| Pinia | State management | MIT | https://github.com/vuejs/pinia |
| Vite | Frontend build tool | MIT | https://github.com/vitejs/vite |
| @vitejs/plugin-vue | Vue plugin for Vite | MIT | https://github.com/vitejs/vite-plugin-vue |
| @ant-design/icons-vue | Icon components | MIT | https://github.com/ant-design/ant-design-icons |
| noVNC | Browser VNC viewer used by login/runtime diagnostics | MPL-2.0 | https://github.com/novnc/noVNC |

When distributing built `webui/dist` assets, keep this notice with the release
and preserve license notices for bundled frontend dependencies, especially
MPL-2.0 components such as noVNC.
