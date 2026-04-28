# WebAI2API Native UI Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make the gateway use WebAI2API's original Vue frontend as the primary user interface while keeping the gateway's `/v1` tool bridge.

**Architecture:** Vendor WebAI2API's `webui` source into this project and add a small gateway page/menu inside the existing Ant Design Vue shell. Python serves the built WebAI2API SPA, proxies `/admin/*` to the configured WebAI2API sidecar root, keeps `/v1/*` on the gateway, and keeps `/api/admin/*` for gateway-local settings.

**Tech Stack:** FastAPI, httpx, Vue 3, Ant Design Vue, Pinia, pytest.

---

### Task 1: Backend Native UI Serving And Admin Proxy

**Files:**
- Modify: `webai_gateway/app.py`
- Test: `tests/test_gateway.py`

- [ ] **Step 1: Write failing tests**

Add tests that prove:
- `/` serves a WebAI2API-style SPA when `webui/dist/index.html` exists.
- `/assets/<file>` serves WebAI2API static assets.
- `/admin/status` proxies to the WebAI2API sidecar root derived from `upstream.base_url`.
- `/v1/models` still uses the gateway model bridge.

- [ ] **Step 2: Run tests and verify failure**

Run: `rtk python -m pytest tests/test_gateway.py::test_root_serves_webai2api_native_spa_when_available tests/test_gateway.py::test_assets_serve_webai2api_native_files tests/test_gateway.py::test_admin_routes_proxy_to_webai2api_sidecar_root tests/test_gateway.py::test_v1_routes_remain_gateway_owned_when_admin_proxy_exists -q`

Expected: fail because the new serving/proxy behavior is not implemented.

- [ ] **Step 3: Implement minimal backend**

Add focused helpers in `webai_gateway/app.py`:
- `_webai2api_ui_dir()`
- `_sidecar_root_from_base_url(base_url)`
- `_proxy_to_sidecar(...)`

Wire:
- `/assets/{path:path}` and `/favicon.png` to WebAI2API build assets when present.
- `/admin` and `/admin/{path:path}` to the sidecar root.
- `/` and SPA fallback paths to WebAI2API `index.html`, falling back to the old control console only if the native UI build is missing.

- [ ] **Step 4: Run tests and verify pass**

Run the focused tests from Step 2.

### Task 2: WebAI2API Frontend Source Integration

**Files:**
- Create/Copy: `webui/`
- Modify: `webui/src/main.js`
- Modify: `webui/src/App.vue`
- Create: `webui/src/components/gateway/KrisBridge.vue`

- [ ] **Step 1: Copy WebAI2API webui source**

Copy `C:\Users\woody\AppData\Local\Temp\WebAI2API\webui` into `E:\ProjectX\webai-gateway\webui`.

- [ ] **Step 2: Add route and menu**

Add route `/gateway/kris-bridge` and menu item `KrisAI / 工具桥` in the existing WebAI2API shell.

- [ ] **Step 3: Add gateway page**

Create `KrisBridge.vue` that fetches `/api/admin/config`, displays:
- Gateway OpenAI base URL: `${location.origin}/v1`
- Gateway token
- Upstream WebAI2API base URL and model
- Tool mode
- Client examples for KrisAI, OpenClaw, Hermes, Claude Code

The page must use Ant Design Vue components and Chinese UI text.

- [ ] **Step 4: Build frontend**

Run: `rtk powershell -NoProfile -Command "cd webui; pnpm install --frozen-lockfile; pnpm build"`

Expected: `webui/dist/index.html` and `webui/dist/assets/*` exist.

### Task 3: Documentation And Verification

**Files:**
- Modify: `README.md`

- [ ] **Step 1: Update docs**

Document that `http://127.0.0.1:8610/` is now the WebAI2API-native UI, `/admin/*` proxies to WebAI2API, `/v1/*` is the gateway tool bridge, and `/api/admin/*` remains gateway-local.

- [ ] **Step 2: Run full verification**

Run:
- `rtk python -m pytest -q`
- browser reload on `http://127.0.0.1:8610/`
- API checks for `/`, `/admin/status`, and `/v1/models`

Expected: tests pass, the WebAI2API-native UI loads, and `/v1/models` still includes gateway catalog models.
