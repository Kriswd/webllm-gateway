# WebLLM Gateway 开源推广发布包

这份发布包的主线调整为：不要把复杂编程任务作为主要价值，重点讲 Qwen 3.7 MAX 网页模型接入小龙虾和 Hermes，适合轻中度工具调用、养虾养马和节省部分 API Token。

## 核心定位

WebLLM Gateway 是一个网页登录模型 API 网关。它把 Qwen Web、DeepSeek Web 和已验证的网页登录模型包装成 OpenAI / Anthropic 兼容、可工具调用的标准 API，让小龙虾（OpenClaw）、Hermes 等客户端可以用网页账号跑轻中度任务。

## B站标题

```text
我把Qwen 3.7 MAX网页模型接入了小龙虾和hermes，网页模型也能调用工具干活了？
```

## 一句话版本

把 Qwen 3.7 MAX 网页账号接进小龙虾和 Hermes，让网页模型也能参与轻中度工具调用，适合养虾养马、资料整理、简单自动化和低成本试错。

## 30 秒版本

我开源了一个小项目 WebLLM Gateway。

它不是要把网页模型吹成万能编程模型，也不建议拿来硬跑复杂工程长链路。它更适合把 Qwen 3.7 MAX 这类网页登录模型，接进小龙虾、Hermes 等兼容客户端，处理资料整理、信息查询、简单工具调用、批量小任务。

流程很简单：打开授权浏览器登录网页账号，Gateway 自动检测可用模型，然后复制 API 地址、Key 和模型 ID 到客户端里。客户端负责执行工具和权限确认，Gateway 只负责把“网页模型文本”和“标准工具调用协议”互相翻译。

核心价值：够用的轻任务少烧一点 API Token。

## GitHub 仓库描述

```text
WebLLM Gateway：把 Qwen 3.7 MAX 等网页模型接入小龙虾、Hermes 等兼容客户端，支持轻中度工具调用
```

## 推荐 GitHub Topics

```text
tool-calling
openai-compatible
anthropic-compatible
openclaw
hermes
qwen
qwen37
deepseek
web-llm
ai-gateway
llm-gateway
webai2api
ds2api
```

## B站简介

```text
这次不吹复杂编程任务，重点是低成本轻量工具调用：
Qwen 3.7 MAX 网页模型 -> WebLLM Gateway -> 小龙虾 / Hermes。

适合资料整理、信息查询、简单工具调用、养虾养马、低成本试错，用网页登录账号省一部分 API Token。
复杂工程、长链路高可靠编程任务别硬上，还是交给更稳的模型和官方 API。

项目地址：
https://github.com/Kriswd/webllm-gateway

仅供学习研究，请遵守平台规则和法律法规。
```

## B站标签

```text
Qwen
通义千问
Qwen 3.7 MAX
小龙虾
Hermes
网页模型
工具调用
省Token
开源项目
大模型
```

## 微信群 / 社群文案

我把 WebLLM Gateway 开源了。

一句话：把 Qwen 3.7 MAX 这类网页登录模型，接进小龙虾和 Hermes，让网页模型也能参与工具调用。

但边界先说清楚：它不是万能编程模型，不建议硬跑复杂工程长链路。更适合资料整理、信息查询、简单自动化、养虾养马这类轻中度任务。客户端负责执行工具和权限确认，Gateway 只负责协议翻译。

适合想低成本试错、省一点 API Token、折腾网页模型工具调用的人试试。

项目地址：
https://github.com/Kriswd/webllm-gateway

## 朋友圈文案

最近把 WebLLM Gateway 开源了。

这次定位想说清楚一点：它不是拿网页模型去硬刚复杂编程任务，而是把 Qwen 3.7 MAX 这类网页登录模型接进小龙虾、Hermes，让网页模型也能做轻中度工具调用。

比如资料整理、信息查询、简单自动化、养虾养马这类活，很多时候够用就行，还能少烧一点 API Token。真正执行工具和权限确认的仍然是客户端，Gateway 只是把网页登录模型和标准 OpenAI / Anthropic 工具协议互相翻译。

项目地址：
https://github.com/Kriswd/webllm-gateway

仅供学习研究使用，记得遵守平台规则和相关法律法规。

## X / Twitter 文案

```text
I open-sourced WebLLM Gateway.

It turns Qwen 3.7 MAX and other web-login LLM accounts into OpenAI/Anthropic-compatible APIs for lightweight tool calling in clients like OpenClaw and Hermes.

Not positioned as a heavy coding-agent replacement. The sweet spot is practical, lightweight tasks where web accounts can save API tokens.

https://github.com/Kriswd/webllm-gateway
```

## 60 秒 demo 脚本

1. 打开 Gateway 首页，展示 `Qwen 3.7 MAX 已调通`。
2. 点开 Qwen 平台，展示已授权和模型列表。
3. 复制 `base_url`、`api_key`、`model`。
4. 打开小龙虾或 Hermes 客户端配置页。
5. 填入 `http://127.0.0.1:8610/v1`、`local-dev-key`、`qwen-web/qwen3.7-max-preview`。
6. 下发轻任务：整理一段资料、查询信息、调用一个简单工具。
7. 展示客户端触发工具调用。
8. 展示工具结果回传后，模型继续整理答案。
9. 收尾：复杂工程别硬上，轻中度任务和低成本试错更适合。

## 首批投放渠道

- 小龙虾、Hermes、Qwen / DeepSeek 网页模型折腾群。
- AI 工具、低成本用量、Token 节省相关社群。
- GitHub Trending / 开源项目推荐社群。
- V2EX、掘金、知乎想法、即刻、小红书、B站动态。
- 给 WebAI2API、ds2api、OpenClaw、Hermes 相关讨论区发“兼容/接入经验”，少讲广告，多讲真实 demo。

## 冷启动 Checklist

- GitHub topics 已配置。
- README 首屏能在 10 秒内看懂“小龙虾 / Hermes + Qwen 3.7 MAX + 轻量工具调用”的价值。
- 首页截图能看到 Qwen 3.7 和授权流程。
- 至少准备 1 个真实轻量工具调用录屏。
- 发帖时附上 demo 文档，不只贴仓库链接。
- 置顶一个 issue：欢迎反馈小龙虾 / Hermes 接入问题。
- 每收到一个用户报错，都沉淀成 FAQ 或 troubleshooting 文档。

## 风险提示

传播时不要承诺“永久免费”“永不封号”“绕过限制”。建议统一表述为：本项目仅做本地学习、研究和技术验证，使用者需要遵守对应网站和模型服务条款。
