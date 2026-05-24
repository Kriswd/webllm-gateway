# Refined UI Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Apply the selected "纸本编辑部" refined visual system to WebLLM Gateway's primary WebUI and fallback console, with all visible UI typography unified on Source Han Serif SC.

**Architecture:** Keep the existing Vue/Ant Design component structure and Python static fallback. Introduce shared font-face declarations and aligned design tokens in the relevant style blocks instead of changing business logic.

**Tech Stack:** Vue 3, Ant Design Vue, Vite, FastAPI static assets, CSS custom properties.

---

## File Structure

- `webui/public/fonts/NotoSerifSC-VF.ttf`: self-hosted Source Han/Noto Serif SC-compatible font for the built WebUI.
- `webai_gateway/static/fonts/NotoSerifSC-VF.ttf`: same font for fallback static console.
- `webui/src/App.vue`: global font override, app shell, support strip, topbar, drawer/log surface styles.
- `webui/src/components/gateway/KrisBridge.vue`: public onboarding page refined tokens, hero, panels, provider/account cards, models, media, config blocks.
- `webai_gateway/static/styles.css`: fallback console refined tokens, typography, panels, controls, output blocks.
- `webui/dist/*`: generated production bundle after `pnpm build`.

## Tasks

### Task 1: Add Self-Hosted Font Assets

- [ ] Copy `C:\Windows\Fonts\NotoSerifSC-VF.ttf` to `webui/public/fonts/NotoSerifSC-VF.ttf`.
- [ ] Copy `C:\Windows\Fonts\NotoSerifSC-VF.ttf` to `webai_gateway/static/fonts/NotoSerifSC-VF.ttf`.
- [ ] Verify both files exist and are not empty.

### Task 2: Refine App Shell

- [ ] Add `@font-face` and global font overrides in `webui/src/App.vue`.
- [ ] Replace the support strip colors with paper/accent tokens and keep it compact.
- [ ] Replace the topbar and shell background with refined paper surfaces and subtle borders.
- [ ] Ensure Ant Design buttons, inputs, selects, drawers, modals, tags, tables, and typography inherit the same font.

### Task 3: Refine Public Gateway Page

- [ ] Replace `KrisBridge.vue` local tokens with the selected refined palette.
- [ ] Change the hero from dark-brand block to paper-white editorial panel.
- [ ] Update provider rows, account cards, tables, empty states, image preview, and code/log blocks to use the same tokens.
- [ ] Keep existing copy, actions, and data bindings unchanged.

### Task 4: Refine Fallback Console

- [ ] Add the same font-face and token palette to `webai_gateway/static/styles.css`.
- [ ] Update sidebar, panel, metrics, auth, forms, buttons, outputs, and toast styling.
- [ ] Remove remaining sans-serif and monospace font-family declarations.

### Task 5: Build and Verify

- [ ] Run `cd webui; pnpm build`.
- [ ] Run relevant backend tests that cover frontend source expectations.
- [ ] Restart Gateway and verify `/health`.
- [ ] Use the browser to capture desktop and mobile screenshots of the updated UI.
- [ ] Push local `main` to `origin/main`.
