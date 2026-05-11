# Media Generation

Gateway 提供 OpenAI-compatible 的图片和视频包装接口。媒体生成依赖 WebAI2API sidecar 的网页登录能力；Gateway 只负责协议包装、鉴权、错误透传和短期缓存。

## Image generation

Endpoint:

```text
POST /v1/images/generations
```

推荐模型：

```text
gpt-image-2
```

兼容模型：

```text
gpt-image-1.5
chatgpt/gpt-image-2
chatgpt/gpt-image-1.5
```

限制：

- 当前只支持 `n=1`。
- `response_format` 支持 `url` 和 `b64_json`。
- `url` 返回的是 data URI，不是公网文件地址。
- 需要 ChatGPT/WebAI2API 登录态可用。

PowerShell 示例：

```powershell
$key = (Get-Content -Raw config.json | ConvertFrom-Json).server.apiKey
$body = @{
  model = "gpt-image-2"
  prompt = "一只蓝色玻璃质感的未来感小机器人，产品摄影风格"
  n = 1
  response_format = "b64_json"
} | ConvertTo-Json

$result = Invoke-RestMethod `
  -Uri "http://127.0.0.1:8610/v1/images/generations" `
  -Method Post `
  -Headers @{ Authorization = "Bearer $key" } `
  -ContentType "application/json" `
  -Body $body

[IO.File]::WriteAllBytes("output.png", [Convert]::FromBase64String($result.data[0].b64_json))
```

## Video generation

Endpoint:

```text
POST /v1/videos
GET /v1/videos/{video_id}
GET /v1/videos/{video_id}/content
```

推荐模型：

```text
sora-2
```

视频内容会短期保存在 Gateway 内存缓存中；重启服务后缓存会丢失。需要长期保存时，下游客户端应在生成后立即下载 `/content`。

## Frontend smoke test

首页“图片生成测试”区域可以直接调用 `/v1/images/generations`。它用于验证网页登录态和 Gateway 包装是否可用，不替代完整的图片工作台。
