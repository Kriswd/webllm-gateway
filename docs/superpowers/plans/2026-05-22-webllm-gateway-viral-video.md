# WebLLM Gateway Viral Video Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Produce verified vertical and horizontal HyperFrames marketing videos that explain WebLLM Gateway to Agent tool-chain users in plain Chinese.

**Architecture:** Keep the video source in a dedicated HyperFrames project under `docs/assets/videos/webllm-gateway-viral/`. Share the same real screenshots, narration script, visual identity, and scene vocabulary across a `vertical` and a `horizontal` composition, then render separate MP4s for distribution.

**Tech Stack:** HyperFrames HTML compositions, GSAP timelines, HyperFrames CLI lint/validate/inspect/render, Kokoro TTS when Mandarin voice support is available, local WebLLM Gateway screenshots.

---

## File Structure

- Create `docs/assets/videos/webllm-gateway-viral/DESIGN.md` for palette, type and motion rules.
- Create `docs/assets/videos/webllm-gateway-viral/script-zh.txt` for narration and subtitle source.
- Create `docs/assets/videos/webllm-gateway-viral/assets/` for screenshots and generated audio.
- Create `docs/assets/videos/webllm-gateway-viral/vertical/` as the `1080x1920` HyperFrames composition.
- Create `docs/assets/videos/webllm-gateway-viral/horizontal/` as the `1920x1080` HyperFrames composition.
- Create rendered MP4 outputs under `docs/assets/videos/webllm-gateway-viral/renders/`.
- Update `.gitignore` only for bulky HyperFrames caches and generated transient files if the CLI creates them.

### Task 1: Scaffold the HyperFrames project and assets

**Files:**
- Create: `docs/assets/videos/webllm-gateway-viral/DESIGN.md`
- Create: `docs/assets/videos/webllm-gateway-viral/script-zh.txt`
- Create: `docs/assets/videos/webllm-gateway-viral/assets/`

- [ ] **Step 1: Run environment check**

Run:

```powershell
npx hyperframes doctor
```

Expected: Node, Chrome and FFmpeg checks report usable.

- [ ] **Step 2: Scaffold vertical and horizontal compositions**

Run:

```powershell
npx hyperframes init docs/assets/videos/webllm-gateway-viral/vertical --non-interactive
npx hyperframes init docs/assets/videos/webllm-gateway-viral/horizontal --non-interactive
```

Expected: both directories contain HyperFrames source files.

- [ ] **Step 3: Capture real Gateway screenshots**

Capture from local `http://127.0.0.1:8610/`:

- home hero with Qwen 3.7 tag;
- model/provider panel;
- GitHub README screenshot if needed.

Expected: screenshots live in the shared `assets/` directory and do not expose cookies, bearer tokens, API keys beyond the public local example.

- [ ] **Step 4: Write visual identity and narration script**

Create `DESIGN.md` with warm near-black, rust orange, warm white and proof green palette. Create `script-zh.txt` with the approved first-person hook and plain-language bridge explanation.

### Task 2: Build vertical viral composition

**Files:**
- Create/Modify: `docs/assets/videos/webllm-gateway-viral/vertical/index.html`
- Reuse: `docs/assets/videos/webllm-gateway-viral/assets/*`

- [ ] **Step 1: Build static scene layouts**

Lay out six timed scenes with large Chinese captions:

1. hook;
2. web-chat vs Agent work;
3. Gateway reveal;
4. real task proof;
5. plain-language bridge explanation;
6. GitHub CTA.

Expected: every scene is readable at `1080x1920` before motion.

- [ ] **Step 2: Add deterministic motion and transitions**

Use synchronous GSAP timelines, per-scene entrances and a consistent high-energy transition family. Keep final-scene fade only.

- [ ] **Step 3: Add narration or caption-only fallback**

Generate Mandarin narration with HyperFrames TTS if local Mandarin voice dependencies work. If TTS is blocked, keep caption timing and render a polished no-voice draft that can be reviewed immediately.

- [ ] **Step 4: Verify vertical composition**

Run:

```powershell
npx hyperframes lint docs/assets/videos/webllm-gateway-viral/vertical
npx hyperframes validate docs/assets/videos/webllm-gateway-viral/vertical
npx hyperframes inspect docs/assets/videos/webllm-gateway-viral/vertical --samples 15
```

Expected: no blocking lint/layout/contrast issue.

### Task 3: Build horizontal explainer composition

**Files:**
- Create/Modify: `docs/assets/videos/webllm-gateway-viral/horizontal/index.html`
- Reuse: `docs/assets/videos/webllm-gateway-viral/assets/*`

- [ ] **Step 1: Adapt the same story to wider evidence frames**

Use the same hook and visual identity, add wider screenshot treatments for:

- supported Agent clients;
- local Gateway UI;
- bridge explanation;
- permission boundary.

- [ ] **Step 2: Add motion and transitions**

Reuse the vertical visual language without merely scaling the vertical layout.

- [ ] **Step 3: Verify horizontal composition**

Run:

```powershell
npx hyperframes lint docs/assets/videos/webllm-gateway-viral/horizontal
npx hyperframes validate docs/assets/videos/webllm-gateway-viral/horizontal
npx hyperframes inspect docs/assets/videos/webllm-gateway-viral/horizontal --samples 15
```

Expected: no blocking lint/layout/contrast issue.

### Task 4: Render and distribute

**Files:**
- Create: `docs/assets/videos/webllm-gateway-viral/renders/webllm-gateway-viral-vertical.mp4`
- Create: `docs/assets/videos/webllm-gateway-viral/renders/webllm-gateway-viral-horizontal.mp4`

- [ ] **Step 1: Render draft videos**

Run:

```powershell
npx hyperframes render docs/assets/videos/webllm-gateway-viral/vertical --quality draft --output ../renders/webllm-gateway-viral-vertical-draft.mp4
npx hyperframes render docs/assets/videos/webllm-gateway-viral/horizontal --quality draft --output ../renders/webllm-gateway-viral-horizontal-draft.mp4
```

Expected: playable review MP4s.

- [ ] **Step 2: Render final videos after review**

Run:

```powershell
npx hyperframes render docs/assets/videos/webllm-gateway-viral/vertical --quality high --output ../renders/webllm-gateway-viral-vertical.mp4
npx hyperframes render docs/assets/videos/webllm-gateway-viral/horizontal --quality high --output ../renders/webllm-gateway-viral-horizontal.mp4
```

Expected: final MP4s exist and play.

- [ ] **Step 3: Produce distribution recommendation**

Report:

- recommended first release platforms;
- which cut to use on each platform;
- title/caption hook to pair with the video;
- any platform requiring an English or shorter recut.
