# WebLLM Gateway Viral Video

这套 HyperFrames 素材用于 WebLLM Gateway 开源推广，主叙事是：

> 把本来只能在网页里聊天的 AI，接进会用工具干活的 Agent 链路。

## 成片

- 竖屏主片：`renders/webllm-gateway-viral-vertical.mp4`
  - 画幅：`1080x1920`
  - 适合：短视频平台、微信生态和社群转发。
- 横屏说明版：`renders/webllm-gateway-viral-horizontal.mp4`
  - 画幅：`1920x1080`
  - 适合：B站、YouTube、GitHub 说明和开发者社群。

## 源文件

- 视觉规则：`DESIGN.md`
- 竖屏口播：`script-vertical-zh.txt`
- 横屏口播：`script-horizontal-zh.txt`
- 共用素材：`assets/`
- HyperFrames 合成：`vertical/` 与 `horizontal/`

## 本地重渲

```powershell
cd docs/assets/videos/webllm-gateway-viral/vertical
npx hyperframes lint --verbose
npx hyperframes render --quality high --strict --output ..\renders\webllm-gateway-viral-vertical.mp4
```

```powershell
cd docs/assets/videos/webllm-gateway-viral/horizontal
npx hyperframes lint --verbose
npx hyperframes render --quality high --strict --output ..\renders\webllm-gateway-viral-horizontal.mp4
```

抽帧检查图和 draft MP4 默认只作为本地验收产物保留，不纳入仓库。
