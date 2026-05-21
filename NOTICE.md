# Notices

WebAI Gateway is released under the MIT License.

The public project tree is intended to contain Gateway source code, tests, UI,
and documentation only. It does not vendor the source code, credentials,
browser profiles, runtime data, or caches of WebAI2API or ds2api. Those projects
are treated as optional external adapter runtimes that can be installed,
started, or contacted by the Gateway when the matching provider is used.

## Optional adapter runtimes

- WebAI2API: optional external runtime by `foxhui`, MIT license.
  Source: https://github.com/foxhui/WebAI2API
- ds2api: optional external runtime and DeepSeek oracle reference by
  `CJackHwang`, GNU AGPL-3.0 license.
  Source: https://github.com/CJackHwang/ds2api

If you redistribute, modify, host, or bundle ds2api, comply with AGPL-3.0 and
provide the corresponding source as required by that license. If you
redistribute a bundled WebAI2API runtime, include its upstream license and
notices.

## Frontend and runtime dependencies

The web UI and Python service use third-party open-source dependencies such as
Vue, Ant Design Vue, Vite, Pinia, FastAPI, Uvicorn, HTTPX, Playwright, and
noVNC. A non-exhaustive license inventory is maintained in
`THIRD_PARTY_NOTICES.md`.

Before publishing a release package that bundles third-party binaries, built
frontend assets, or source code, verify upstream license files again and ship
their notices alongside the release.
