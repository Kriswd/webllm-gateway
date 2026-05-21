# Qwen Coder 集成方案

## 1. 背景与目标

### 1.1 背景
Qwen 官方推出了专门针对编程场景优化的网页版助手 **Qwen Coder** (https://coder.qwen.ai/)。
相比通用版 Qwen，Coder 版本具备以下核心优势：
- **深度代码能力**：专为代码生成、调试、重构和优化设计。
- **特有功能支持**：
  - **Artifacts**：实时预览和编辑生成的代码工件（前端页面、组件等）。
  - **MCP (Model Context Protocol)**：标准化的工具调用协议，便于连接本地开发环境。
  - **Thinking 模式**：更长的深度推理过程，适合解决复杂算法和架构问题。
  - **项目上下文**：支持上传整个项目文件夹，理解文件间依赖关系。

### 1.2 目标
在 WebLLM Gateway 中快速集成 Qwen Coder，使其能够：
1. 被现有客户端（KrisAI, Claude Code 等）无缝调用。
2. 保留基础对话和联网搜索能力。
3. **重点支持**：编程特有的 Artifacts 展示、MCP 工具调用及长文本代码处理。

---

## 2. 技术调研结论

### 2.1 架构兼容性
经过分析，Qwen Coder 与现有的 Qwen Web (`chat.qwen.ai`) 共享底层架构：
- **认证机制**：共用 Cookie/Token 体系。
- **API 端点**：核心对话接口结构相似，主要通过 `model` 参数或特定的 `source` 标记区分。
- **通信协议**：同样基于 SSE (Server-Sent Events) 流式传输。

### 2.2 差异化特征
| 特性 | Qwen Web (通用) | Qwen Coder (编程) | 集成策略 |
| :--- | :--- | :--- | :--- |
| **默认模型** | Qwen-Max / Plus | Qwen-Coder / Qwen-2.5-Coder | 配置指定模型别名 |
| **系统提示词** | 通用助手人设 | 资深工程师人设 (System Prompt 注入) | 动态注入 System Prompt |
| **工具调用** | 标准搜索/绘图 | MCP 协议、代码解释器 | 扩展 ToolBridge 解析器 |
| **输出格式** | 普通文本 + Markdown | 包含 Artifact 块、Diff 视图 | 新增响应后处理器 |
| **超时需求** | 60-120 秒 | 300 秒+ (复杂代码生成) | 动态调整超时配置 |

---

## 3. 实施方案

### 3.1 总体架构
采用 **"复用核心 + 插件化扩展"** 的策略。
- **核心层**：复用现有的 `qwen_web.py` 中的登录、会话保持、基础请求发送逻辑。
- **适配层**：新建 `qwen_coder_adapter.py`，专门处理 Coder 特有的协议字段和响应解析。
- **路由层**：在网关路由中增加对 `qwen-coder/` 前缀模型的识别。

### 3.2 详细步骤

#### 步骤一：配置更新 (`config.json`)
新增 `qwen_coder` 平台配置，定义专属参数。

```json
{
  "platforms": [
    {
      "name": "qwen_coder",
      "type": "web",
      "base_url": "https://coder.qwen.ai",
      "login_url": "https://login.qwen.ai",
      "models": ["qwen-coder-plus", "qwen-2.5-coder"],
      "features": ["artifacts", "mcp", "project_context"],
      "timeout": 300,
      "system_prompt_template": "You are an expert programming assistant specialized in code generation, debugging, and software architecture. Prefer concise, efficient code."
    }
  ]
}
```

#### 步骤二：核心适配器开发 (`adapters/qwen_coder_adapter.py`)
继承或组合现有 Qwen 适配器，重写关键方法：

1.  **请求构造**：
    - 自动附加编程专用的 System Prompt。
    - 若检测到 `tools` 参数，优先转换为 MCP 格式而非通用 Function Call 格式。
2.  **响应流解析**：
    - 拦截 SSE 数据，识别 `artifact` 类型的数据块。
    - 将 Artifact 内容封装为标准的 Markdown 代码块或自定义 JSON 结构返回给客户端。
3.  **错误处理**：
    - 针对代码生成常见的 "Context Limit" 或 "Execution Timeout" 做特殊重试策略。

#### 步骤三：路由与模型映射 (`router.py`)
- 添加规则：当请求模型名为 `qwen-coder-*` 时，自动路由至 `qwen_coder` 平台。
- 实现模型别名映射，例如将 `qwen-coder-max` 映射到实际的网页模型 ID。

#### 步骤四：ToolBridgeV2 增强
- **MCP 支持**：在 ToolBridge 中增加 MCP Server 的连接逻辑，允许 Qwen Coder 调用本地文件系统、终端命令。
- **Artifact 渲染**：对于前端代码生成，支持返回包含 `html_preview_url` 的元数据，方便前端直接渲染预览。

---

## 4. 接口定义示例

### 4.1 调用示例 (OpenAI 兼容格式)
客户端无需修改代码，只需更换模型名称即可触发 Coder 模式。

```bash
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "qwen-coder-plus",
    "messages": [
      {"role": "system", "content": "Use MCP to read the current directory structure."},
      {"role": "user", "content": "Create a React Todo List app with local storage support."}
    ],
    "stream": true
  }'
```

### 4.2 响应增强 (含 Artifact)
网关将自动解析并标准化 Artifact 输出：

```json
{
  "choices": [{
    "delta": {
      "content": "Here is the React app...",
      "artifact": {
        "id": "art_123",
        "type": "text/html",
        "title": "Todo App Preview",
        "code": "<!DOCTYPE html>..."
      }
    }
  }]
}
```

---

## 5. 开发计划与验证

### 5.1 阶段划分
| 阶段 | 任务内容 | 预计工时 | 交付物 |
| :--- | :--- | :--- | :--- |
| **P0** | 基础连通性：登录、简单对话、流式输出 | 2 小时 | 可运行的 `qwen_coder` 基础适配器 |
| **P1** | 编程特性：System Prompt 注入、长超时支持 | 1 小时 | 支持复杂代码生成的配置 |
| **P2** | 高级特性：Artifact 解析、MCP 工具桥接 | 4 小时 | 完整的 ToolBridgeV2 支持 |
| **P3** | 测试与文档：编写用例，更新 README | 1 小时 | 测试报告、更新后的文档 |

### 5.2 验证标准
1.  **基础对话**：能正常回答编程概念问题。
2.  **代码生成**：能生成完整可运行的代码片段，且格式正确。
3.  **大文件处理**：上传 50KB+ 代码文件上下文，不报错且能理解。
4.  **工具调用**：成功触发一次模拟的 MCP 工具调用（如读取文件）。

---

## 6. 风险与应对

- **风险 1：网页结构变更**
  - *应对*：Qwen Coder 处于快速迭代期，需建立定期的 Selector/XPath 健康检查机制。
- **风险 2：反爬策略升级**
  - *应对*：引入更完善的 Cookie 池管理，支持多账号轮询；增加请求头指纹模拟。
- **风险 3：Artifact 解析失败**
  - *应对*：设置降级策略，若无法解析 Artifact 结构，直接以原始 Markdown 代码块返回，保证可用性。

---

## 7. 结论
集成 Qwen Coder 是提升 WebLLM Gateway 编程竞争力的关键一步。鉴于其与现有 Qwen Web 的高度兼容性，采用**增量开发模式**可在 **1 个工作日内** 完成核心功能上线，快速为用户提供专业的代码辅助能力。
