const state = {
  config: null,
  tokenVisible: false,
  providers: [],
  authJobTimer: null,
  requestDiagnostics: [],
  autoResearch: null,
  autoResearchCandidates: [],
};

const $ = (id) => document.getElementById(id);

function showToast(message) {
  const toast = $("toast");
  toast.textContent = message;
  toast.classList.add("show");
  window.clearTimeout(showToast.timer);
  showToast.timer = window.setTimeout(() => toast.classList.remove("show"), 2600);
}

function pretty(data) {
  return JSON.stringify(data, null, 2);
}

function linesFromTextarea(value) {
  return String(value || "")
    .split(/\r?\n|,/)
    .map((item) => item.trim())
    .filter((item, index, items) => item && items.indexOf(item) === index);
}

function gatewayBaseUrl() {
  return `${window.location.origin}/v1`;
}

function webai2apiRootUrl(route = "/") {
  const rawBaseUrl = state.config?.upstream?.baseUrl || $("upstreamBaseUrl")?.value || "http://127.0.0.1:8500/v1";
  const root = new URL(rawBaseUrl, window.location.href);
  root.pathname = root.pathname.replace(/\/v1\/?$/, "/");
  if (!root.pathname.endsWith("/")) {
    root.pathname = `${root.pathname}/`;
  }
  root.search = "";
  root.hash = "";
  return new URL(route.replace(/^\//, ""), root.href).href;
}

function updateWebAI2APILinks() {
  const uiUrl = webai2apiRootUrl("/");
  $("webai2apiUiUrl").textContent = uiUrl;
}

function openWebAI2APIConsole(route = "/") {
  const url = webai2apiRootUrl(route);
  window.open(url, "_blank", "noopener,noreferrer");
  setAuthLog(`已打开 WebAI2API 原生管理台：${url}`);
}

function embedWebAI2APIConsole(route = "/") {
  const url = webai2apiRootUrl(route);
  $("webai2apiFrame").src = url;
  $("webai2apiFrameWrap").classList.remove("is-hidden");
  setAuthLog(`已在当前页面嵌入 WebAI2API 原生管理台：${url}`);
}

function headers() {
  return {
    "Content-Type": "application/json",
    Authorization: `Bearer ${$("serverApiKey").value}`,
  };
}

function setOutput(data) {
  $("testOutput").textContent = typeof data === "string" ? data : pretty(data);
}

function setAuthLog(message) {
  $("authLog").textContent = message;
}

function appendAuthLog(message) {
  $("authLog").textContent = `${$("authLog").textContent}\n${message}`.trim();
}

function selectedProvider() {
  return state.providers.find((item) => item.id === $("authProvider").value);
}

function applyConfig(config) {
  state.config = config;
  $("serverApiKey").value = config.server.apiKey || "";
  $("upstreamBaseUrl").value = config.upstream.baseUrl || "";
  $("upstreamApiKey").value = config.upstream.apiKey || "";
  $("upstreamModel").value = config.upstream.model || "";
  $("toolModeSelect").value = config.upstream.toolMode || "prompt";
  $("providerRuntimeTimeoutSeconds").value = config.providerRuntime?.requestTimeoutSeconds || 300;
  $("providerRuntimePromptMaxChars").value = config.providerRuntime?.promptMaxChars || 32000;
  $("providerRuntimeResponseLanguage").value = config.providerRuntime?.responseLanguage || "zh-CN";
  $("nativeWebSearchPolicySelect").value = config.providerRuntime?.nativeWebSearchPolicy || "auto";
  $("deepseekDs2apiBaseUrl").value = config.providerRuntime?.deepseekDs2apiBaseUrl || "http://127.0.0.1:9331/v1";
  $("deepseekDs2apiAccountMaxInflight").value = config.providerRuntime?.deepseekDs2apiAccountMaxInflight || 2;
  $("deepseekDs2apiGlobalMaxInflight").value = config.providerRuntime?.deepseekDs2apiGlobalMaxInflight || 4;
  $("deepseekDs2apiBearerMaxInflight").value = config.providerRuntime?.deepseekDs2apiBearerMaxInflight || 1;
  $("deepseekDs2apiRateLimitCooldownSeconds").value = config.providerRuntime?.deepseekDs2apiRateLimitCooldownSeconds ?? 6;
  $("deepseekDs2apiCurrentInputFileEnabled").value = config.providerRuntime?.deepseekDs2apiCurrentInputFileEnabled === true ? "true" : "false";
  $("deepseekDs2apiCurrentInputFileMinChars").value = config.providerRuntime?.deepseekDs2apiCurrentInputFileMinChars ?? 0;
  $("qwenDirectMaxInflight").value = config.providerRuntime?.qwenDirectMaxInflight || 1;
  $("qwenDirectMaxQueue").value = config.providerRuntime?.qwenDirectMaxQueue ?? 1;
  $("qwenDirectRateLimitCooldownSeconds").value = config.providerRuntime?.qwenDirectRateLimitCooldownSeconds ?? 6;
  $("qwenWebBackendSelect").value = config.providerRuntime?.qwenWebBackend || "direct";
  $("gptThinkingBackendSelect").value = config.providerRuntime?.gptThinkingBackend || "webai2api";
  $("toolActivationPolicySelect").value = config.tool_bridge?.activationPolicy || "auto";
  $("toolExposurePolicySelect").value = config.tool_bridge?.exposurePolicy || "safe";
  $("semanticFinalJudgeSelect").value = config.tool_bridge?.semanticFinalJudge || "off";
  const observationPolicy = config.tool_bridge?.observationPolicy || {};
  $("observationPolicyPathSummary").value = observationPolicy.summarizePathLists === false ? "false" : "true";
  $("observationPolicyPathParts").value = (observationPolicy.excludedPathParts || []).join("\n");
  $("observationPolicyPathGlobs").value = (observationPolicy.excludedPathGlobs || []).join("\n");
  $("observationPolicyMaxItems").value = observationPolicy.pathListMaxItems || 80;
  $("gatewayUrl").textContent = `${window.location.origin}/v1`;
  $("upstreamUrl").textContent = config.upstream.baseUrl || "未配置";
  $("modelName").textContent = config.upstream.model || "未配置";
  $("toolMode").textContent = `工具模式：${config.upstream.toolMode || "prompt"} · 激活：${config.tool_bridge?.activationPolicy || "auto"}`;
  $("clientBaseUrl").textContent = gatewayBaseUrl();
  $("clientApiKey").textContent = config.server.apiKey || "";
  $("clientModel").textContent = config.upstream.model || "";
  updateWebAI2APILinks();
}

async function loadConfig() {
  const res = await fetch("/api/admin/config");
  if (!res.ok) {
    throw new Error(`管理配置读取失败：HTTP ${res.status}`);
  }
  applyConfig(await res.json());
}

async function refreshStatus() {
  try {
    const health = await fetch("/health").then((res) => res.json());
    $("gatewayStatus").textContent = health.ok ? "在线" : "异常";
    $("gatewayStatus").className = health.ok ? "is-ok" : "is-error";
  } catch (error) {
    $("gatewayStatus").textContent = "离线";
    $("gatewayStatus").className = "is-error";
  }

  try {
    const models = await fetch("/v1/models", { headers: headers() }).then(async (res) => {
      if (!res.ok) throw new Error(`HTTP ${res.status}`);
      return res.json();
    });
    $("upstreamStatus").textContent = `${models.data?.length || 0} 个模型`;
    $("upstreamStatus").className = "is-ok";
  } catch (error) {
    $("upstreamStatus").textContent = error.message;
    $("upstreamStatus").className = "is-warning";
  }
}

function renderToolBridgeSummary(event) {
  const parts = [];
  if (event.errorKind) {
    parts.push(`错误：${event.errorKind}${event.statusCode ? ` / HTTP ${event.statusCode}` : ""}`);
  }
  if (event.responseWarning) {
    parts.push(`响应警告：${event.responseWarning}`);
  }
  if (event.requestLatestUserPreview) {
    parts.push(`用户请求：${event.requestLatestUserPreview}`);
  }
  if (event.requestToolResultCount) {
    parts.push(`工具结果：${event.requestToolResultCount} 条`);
  }
  if (event.requestToolErrorCount) {
    parts.push(`工具错误：${event.requestToolErrorCount} 条${event.requestLatestToolErrorPreview ? ` / ${event.requestLatestToolErrorPreview}` : ""}`);
  }
  if (event.requestToolSchemas?.length) {
    const schemaText = event.requestToolSchemas.slice(0, 3).map((tool) => {
      const required = tool.required?.length ? ` required=${tool.required.join("/")}` : "";
      const props = tool.properties?.length ? ` props=${tool.properties.slice(0, 4).join("/")}` : "";
      return `${tool.name}${required}${props}`;
    }).join("；");
    parts.push(`工具 schema：${schemaText}`);
  }
  if (event.toolBridgeError) {
    const repairText = event.toolBridgeRepairable ? "可修复" : "硬拒绝";
    parts.push(`ToolBridge 错误：${event.toolBridgeError}（${repairText}）`);
  }
  if (event.toolBridgeWarning) {
    parts.push(`ToolBridge 警告：${event.toolBridgeWarning}`);
  }
  if (event.toolBridgeTools?.length) {
    parts.push(`工具调用：${event.toolBridgeTools.join(" / ")}`);
  } else if (event.responseToolNames?.length) {
    parts.push(`工具调用：${event.responseToolNames.join(" / ")}`);
  }
  if (event.toolBridgeControllerState) {
    const reason = event.toolBridgeControllerReason ? ` / ${event.toolBridgeControllerReason}` : "";
    const budget = event.toolBridgeRetryBudget ? ` / ${event.toolBridgeRetryBudget}` : "";
    parts.push(`Controller：${event.toolBridgeControllerState}${reason}${budget}`);
  }
  if (event.semanticFinalJudgeMode) {
    const verdict = event.semanticFinalJudgeVerdict || "unknown";
    const confidence = typeof event.semanticFinalJudgeConfidence === "number" ? ` / ${Math.round(event.semanticFinalJudgeConfidence * 100)}%` : "";
    const reason = event.semanticFinalJudgeReason ? ` / ${event.semanticFinalJudgeReason}` : "";
    parts.push(`Semantic Judge：${event.semanticFinalJudgeMode} / ${verdict}${confidence}${reason}`);
  }
  if (event.providerPromptCompacted === true) {
    parts.push(`Provider Prompt 已压缩：${event.providerPromptChars || 0}/${event.providerPromptMaxChars || 0}`);
  }
  if (event.providerOutputPreview) {
    parts.push(`Provider 输出：${event.providerOutputPreview}`);
  }
  if (event.providerMetadataOnlyResponse === true) {
    parts.push("Provider 返回 metadata-only，已触发恢复路径");
  }
  if (event.responseEmpty === true) {
    parts.push("响应为空");
  }
  if (event.responseContentPreview) {
    parts.push(`响应摘要：${event.responseContentPreview}`);
  }
  return parts.length ? parts.join(" · ") : "未发现工具桥异常";
}

function renderRequestDiagnostics(events) {
  const list = $("requestDiagnosticsList");
  const status = $("requestDiagnosticsStatus");
  state.requestDiagnostics = Array.isArray(events) ? events : [];
  list.innerHTML = "";
  if (!state.requestDiagnostics.length) {
    status.textContent = "暂无请求诊断记录";
    return;
  }
  status.textContent = `最近 ${state.requestDiagnostics.length} 条请求诊断`;
  const recent = state.requestDiagnostics.slice(-8).reverse();
  for (const event of recent) {
    const item = document.createElement("article");
    item.className = (event.toolBridgeError || event.errorKind || event.responseWarning) ? "diagnostic-item is-error" : "diagnostic-item";
    const title = document.createElement("div");
    title.className = "diagnostic-title";
    title.textContent = `${event.endpoint || "请求"} · ${event.route || "unknown"} · ${event.model || "model"}`;
    const meta = document.createElement("div");
    meta.className = "diagnostic-meta";
    const stage = event.kind === "completion_request_started" ? "开始" : (event.responseKind || event.kind || "响应");
    const idText = event.requestFingerprint ? ` · id=${event.requestFingerprint}` : "";
    meta.textContent = `${event.at ? new Date(event.at).toLocaleString() : "未知时间"} · #${event.diagnosticSeq || "-"} · ${stage} · bridge=${event.bridge === true ? "是" : "否"}${idText}`;
    const summary = document.createElement("p");
    summary.textContent = renderToolBridgeSummary(event);
    item.append(title, meta, summary);
    list.append(item);
  }
}

async function loadRequestDiagnostics() {
  const res = await fetch("/api/admin/request-diagnostics");
  if (!res.ok) {
    throw new Error(`请求诊断读取失败：HTTP ${res.status}`);
  }
  const data = await res.json();
  renderRequestDiagnostics(data.events || []);
}

function renderAutoResearchStatus(data) {
  state.autoResearch = data || null;
  const available = data?.available === true;
  const total = Number(data?.total || 0);
  const passed = Number(data?.passed || 0);
  const failed = Number(data?.failed || 0);
  $("autoResearchStatus").textContent = available ? (failed ? "需修复" : "已通过") : "未就绪";
  $("autoResearchStatus").className = available ? (failed ? "is-error" : "is-ok") : "is-warning";
  $("autoResearchMessage").textContent = data?.message || "尚未采集失败样本";
  $("autoResearchPassRate").textContent = total ? `${Math.round((data.passRate || 0) * 1000) / 10}%` : "-";
  $("autoResearchReplayCount").textContent = `${passed}/${total} 个 replay 通过`;
  $("autoResearchKindCount").textContent = String(data?.failureKinds?.length || 0);
  $("autoResearchLatest").textContent = data?.recent?.[0]?.error || "-";
  $("autoResearchFixtureDir").textContent = data?.fixtureDir || "未配置";

  const kinds = $("autoResearchKinds");
  kinds.innerHTML = "";
  if (!data?.failureKinds?.length) {
    kinds.textContent = "暂无失败类型样本";
  } else {
    for (const item of data.failureKinds) {
      const row = document.createElement("div");
      row.className = "auto-research-row";
      const name = document.createElement("strong");
      name.textContent = item.kind;
      const count = document.createElement("span");
      count.textContent = `${item.count} 个`;
      row.append(name, count);
      kinds.append(row);
    }
  }

  const recent = $("autoResearchRecent");
  recent.innerHTML = "";
  if (!data?.recent?.length) {
    recent.textContent = "暂无采集样本";
  } else {
    for (const item of data.recent) {
      const row = document.createElement("article");
      row.className = "auto-research-row stacked";
      const title = document.createElement("strong");
      title.textContent = item.id;
      const meta = document.createElement("span");
      const updatedAt = item.updatedAt ? new Date(item.updatedAt).toLocaleString() : "未知时间";
      meta.textContent = `${item.error || "允许调用"} · ${updatedAt} · ${item.path || ""}`;
      row.append(title, meta);
      recent.append(row);
    }
  }

  $("autoResearchCommands").textContent = [
    "采集新失败样本：",
    data?.collectCommand || "python -m webai_gateway.auto_research collect <claude-jsonl>",
    "",
    "回放验证：",
    data?.reportCommand || "python -m webai_gateway.auto_research report",
  ].join("\n");
}

function renderAutoResearchCandidates(data) {
  state.autoResearchCandidates = Array.isArray(data?.candidates) ? data.candidates : [];
  const list = $("autoResearchCandidates");
  list.innerHTML = "";
  if (!state.autoResearchCandidates.length) {
    list.textContent = "暂无运行时候选失败样本";
    return;
  }
  for (const item of state.autoResearchCandidates.slice(0, 8)) {
    const row = document.createElement("article");
    row.className = "auto-research-row stacked";
    const title = document.createElement("strong");
    title.textContent = `${item.error || "unknown"} · ${item.source || "runtime"}`;
    const meta = document.createElement("span");
    const updatedAt = item.at ? new Date(item.at).toLocaleString() : "未知时间";
    meta.textContent = [updatedAt, item.stage || item.route || "", item.controllerState || ""].filter(Boolean).join(" · ");
    const preview = document.createElement("span");
    preview.textContent = item.preview || "无摘要";
    row.append(title, meta, preview);
    list.append(row);
  }
}

async function loadAutoResearchStatus() {
  const res = await fetch("/api/admin/auto-research/status");
  if (!res.ok) {
    throw new Error(`自我改进状态读取失败：HTTP ${res.status}`);
  }
  renderAutoResearchStatus(await res.json());
}

async function loadAutoResearchCandidates() {
  const res = await fetch("/api/admin/auto-research/candidates");
  if (!res.ok) {
    throw new Error(`候选失败样本读取失败：HTTP ${res.status}`);
  }
  renderAutoResearchCandidates(await res.json());
}

function renderAuthProviders(data) {
  state.providers = data.providers || [];
  const select = $("authProvider");
  select.innerHTML = "";
  for (const provider of state.providers) {
    const option = document.createElement("option");
    option.value = provider.id;
    option.textContent = provider.route === "direct" ? `${provider.name}（本地直连）` : `${provider.name}（WebAI2API）`;
    select.append(option);
  }
  if (!select.value && state.providers[0]) {
    select.value = state.providers[0].id;
  }
  renderSelectedAuthProvider();
}

function renderSelectedAuthProvider() {
  const provider = selectedProvider();
  if (!provider) {
    $("authProviderLabel").textContent = "网页模型";
    $("authStatus").textContent = "未加载";
    $("authUpdatedAt").textContent = "尚未授权";
    $("authBadge").textContent = "未加载";
    return;
  }
  const credential = provider.credential || {};
  const authorized = credential.authorized === true;
  const availableModels = Array.isArray(provider.availableModels) ? provider.availableModels : [];
  const modelCount = Number.isFinite(provider.modelCount) ? provider.modelCount : availableModels.length;
  const availabilityMessage = provider.availabilityMessage || "";
  const adapterText = provider.adapters?.length ? `适配器：${provider.adapters.join(" / ")}` : "";
  const capabilityText = [
    provider.capabilities?.text ? "文本" : "",
    provider.capabilities?.image ? "图片" : "",
    provider.capabilities?.video ? "视频" : "",
  ].filter(Boolean).join(" / ");
  $("authProviderLabel").textContent = provider.name;
  if (provider.route === "direct") {
    const hasModels = modelCount > 0;
    $("authStatus").textContent = authorized ? (hasModels ? "已授权，可用模型已验证" : "已授权，暂无可用模型") : "未授权";
    $("authStatus").className = authorized && hasModels ? "is-ok" : "is-warning";
    $("authUpdatedAt").textContent = availabilityMessage || (credential.updatedAt ? `更新时间：${new Date(credential.updatedAt).toLocaleString()}` : "尚未授权");
    $("authBadge").textContent = authorized ? (hasModels ? "本地可用" : "等待适配") : "待登录";
    $("startAuthButton").textContent = `一键启动 ${provider.name} 授权浏览器`;
    $("captureAuthButton").textContent = "重新捕获登录态";
    $("captureAuthButton").disabled = false;
    $("clearAuthButton").disabled = !authorized;
  } else {
    $("authStatus").textContent = "交给 WebAI2API 原生管理";
    $("authStatus").className = "is-ok";
    $("authUpdatedAt").textContent = `${capabilityText || "网页模型"}，${adapterText || "适配器由 WebAI2API 管理"}`;
    $("authBadge").textContent = "复用 WebAI2API";
    $("startAuthButton").textContent = "打开 WebAI2API 登录管理";
    $("captureAuthButton").textContent = "本地捕获仅支持直连 Provider";
    $("captureAuthButton").disabled = true;
    $("clearAuthButton").disabled = true;
  }
  $("authModelName").textContent = modelCount > 1
    ? `${availableModels[0]} 等 ${modelCount} 个模型`
    : (availableModels[0] || availabilityMessage || "当前没有验证可用的模型");
}

async function loadAuthProviders() {
  const res = await fetch("/api/admin/web-auth/providers");
  if (!res.ok) {
    throw new Error(`授权状态读取失败：HTTP ${res.status}`);
  }
  renderAuthProviders(await res.json());
}

async function startAuthFlow() {
  const provider = $("authProvider").value || "deepseek-web";
  const providerInfo = selectedProvider();
  if (providerInfo?.route !== "direct") {
    openWebAI2APIConsole("/tools/cache");
    appendAuthLog(`${providerInfo?.name || provider} 的网页登录、工作池和登录模式由 WebAI2API 原生界面管理；本网关只负责 OpenAI 兼容反代与工具桥。`);
    return;
  }
  setAuthLog("正在启动授权浏览器...");
  const res = await fetch("/api/admin/web-auth/browser/start", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ provider }),
  });
  const data = await res.json();
  if (!res.ok) {
    throw new Error(data.detail || `启动失败：HTTP ${res.status}`);
  }
  appendAuthLog(data.message || "授权浏览器已启动");
  if (!data.started) {
    showToast(data.message || "需要手动启动浏览器");
    return;
  }
  showToast(`请在弹出的浏览器里完成 ${providerInfo?.name || provider} 登录`);
  await startAuthCapture();
}

async function startAuthCapture() {
  const provider = $("authProvider").value || "deepseek-web";
  const providerInfo = selectedProvider();
  if (providerInfo?.route !== "direct") {
    setAuthLog("该站点由 WebAI2API 上游负责登录态，本网关不在本地保存它的 cookie。");
    return;
  }
  appendAuthLog("正在捕获网页登录态...");
  const res = await fetch("/api/admin/web-auth/jobs", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ provider }),
  });
  const data = await res.json();
  if (!res.ok) {
    throw new Error(data.detail || `捕获失败：HTTP ${res.status}`);
  }
  await handleAuthJob(data);
}

async function handleAuthJob(job) {
  appendAuthLog(job.message || "授权任务已启动");
  window.clearInterval(state.authJobTimer);
  if (job.status !== "running") {
    await finishAuthJob(job);
    return;
  }
  state.authJobTimer = window.setInterval(async () => {
    try {
      const res = await fetch(`/api/admin/web-auth/jobs/${job.id}`);
      const latest = await res.json();
      if (!res.ok) {
        throw new Error(latest.detail || `授权任务读取失败：HTTP ${res.status}`);
      }
      setAuthLog(latest.message || "正在等待网页登录授权");
      if (latest.status !== "running") {
        window.clearInterval(state.authJobTimer);
        await finishAuthJob(latest);
      }
    } catch (error) {
      window.clearInterval(state.authJobTimer);
      appendAuthLog(error.message);
      showToast(error.message);
    }
  }, 1800);
}

async function finishAuthJob(job) {
  if (job.status === "succeeded") {
    appendAuthLog("授权完成。正在刷新已验证可用模型；未通过调用验证的模型不会显示为可用。");
    await loadAuthProviders();
    showToast("网页登录授权已完成");
    return;
  }
  appendAuthLog(job.message || "授权失败");
  showToast(job.message || "授权失败");
}

async function clearAuthCredential() {
  const provider = $("authProvider").value || "deepseek-web";
  if (!window.confirm("确定清除本机保存的网页登录授权吗？")) {
    return;
  }
  const res = await fetch(`/api/admin/web-auth/credentials/${provider}`, { method: "DELETE" });
  const data = await res.json();
  if (!res.ok) {
    throw new Error(data.detail || `清除失败：HTTP ${res.status}`);
  }
  await loadAuthProviders();
  setAuthLog("授权已清除。");
  showToast("授权已清除");
}

async function saveConfig(event) {
  event.preventDefault();
  const payload = {
    server: {
      apiKey: $("serverApiKey").value,
    },
    upstream: {
      baseUrl: $("upstreamBaseUrl").value,
      apiKey: $("upstreamApiKey").value,
      model: $("upstreamModel").value,
      toolMode: $("toolModeSelect").value,
    },
    providerRuntime: {
      ...(state.config?.providerRuntime || {}),
      requestTimeoutSeconds: Number($("providerRuntimeTimeoutSeconds").value) || 300,
      promptMaxChars: Number($("providerRuntimePromptMaxChars").value) || 32000,
      responseLanguage: $("providerRuntimeResponseLanguage").value || "zh-CN",
      nativeWebSearchPolicy: $("nativeWebSearchPolicySelect").value,
      deepseekDs2apiBaseUrl: $("deepseekDs2apiBaseUrl").value || "http://127.0.0.1:9331/v1",
      deepseekDs2apiAccountMaxInflight: Number($("deepseekDs2apiAccountMaxInflight").value) || 2,
      deepseekDs2apiGlobalMaxInflight: Number($("deepseekDs2apiGlobalMaxInflight").value) || 4,
      deepseekDs2apiBearerMaxInflight: Number($("deepseekDs2apiBearerMaxInflight").value) || 1,
      deepseekDs2apiRateLimitCooldownSeconds: Number($("deepseekDs2apiRateLimitCooldownSeconds").value) || 0,
      deepseekDs2apiCurrentInputFileEnabled: $("deepseekDs2apiCurrentInputFileEnabled").value === "true",
      deepseekDs2apiCurrentInputFileMinChars: Number($("deepseekDs2apiCurrentInputFileMinChars").value) || 0,
      qwenDirectMaxInflight: Number($("qwenDirectMaxInflight").value) || 1,
      qwenDirectMaxQueue: Number($("qwenDirectMaxQueue").value) || 0,
      qwenDirectRateLimitCooldownSeconds: Number($("qwenDirectRateLimitCooldownSeconds").value) || 0,
      qwenWebBackend: $("qwenWebBackendSelect").value || "direct",
      gptThinkingBackend: $("gptThinkingBackendSelect").value || "webai2api",
    },
    tool_bridge: {
      ...(state.config?.tool_bridge || {}),
      activationPolicy: $("toolActivationPolicySelect").value,
      exposurePolicy: $("toolExposurePolicySelect").value,
      semanticFinalJudge: $("semanticFinalJudgeSelect").value,
      observationPolicy: {
        ...(state.config?.tool_bridge?.observationPolicy || {}),
        summarizePathLists: $("observationPolicyPathSummary").value !== "false",
        excludedPathParts: linesFromTextarea($("observationPolicyPathParts").value),
        excludedPathGlobs: linesFromTextarea($("observationPolicyPathGlobs").value),
        pathListMaxItems: Number($("observationPolicyMaxItems").value) || 80,
      },
    },
  };
  const res = await fetch("/api/admin/config", {
    method: "PUT",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
  });
  if (!res.ok) {
    throw new Error(`保存失败：HTTP ${res.status}`);
  }
  applyConfig(await res.json());
  await refreshStatus();
  showToast("配置已保存并生效。");
}

async function rotateToken() {
  if (!window.confirm("现在轮换网关 API 令牌吗？已有客户端需要同步更新。")) {
    return;
  }
  const res = await fetch("/api/admin/token/rotate", { method: "POST" });
  if (!res.ok) {
    throw new Error(`轮换失败：HTTP ${res.status}`);
  }
  applyConfig(await res.json());
  await refreshStatus();
  showToast("令牌已轮换，请把新令牌复制到客户端。");
}

async function copyToken() {
  await navigator.clipboard.writeText($("serverApiKey").value);
  showToast("令牌已复制。");
}

function toggleToken() {
  state.tokenVisible = !state.tokenVisible;
  $("serverApiKey").type = state.tokenVisible ? "text" : "password";
  $("toggleTokenButton").textContent = state.tokenVisible ? "隐藏" : "显示";
}

async function testModels() {
  setOutput("正在测试 /v1/models...");
  const res = await fetch("/v1/models", { headers: headers() });
  const data = await res.json();
  setOutput(data);
}

async function testChat() {
  setOutput("正在测试 /v1/chat/completions...");
  const mode = $("testMode").value;
  const provider = selectedProvider();
  const providerModel = provider?.models?.[0];
  const model = providerModel || $("upstreamModel").value;
  const body = {
    model,
    messages: [{ role: "user", content: $("testPrompt").value }],
  };
  if (mode === "tool") {
    body.tools = [
      {
        type: "function",
        function: {
          name: "read_file",
          description: "读取客户端环境中的本地文件。",
          parameters: {
            type: "object",
            properties: {
              path: { type: "string", description: "要读取的文件路径。" },
            },
            required: ["path"],
          },
        },
      },
    ];
  }
  const res = await fetch("/v1/chat/completions", {
    method: "POST",
    headers: headers(),
    body: JSON.stringify(body),
  });
  const data = await res.json();
  setOutput(data);
}

function wireNavigation() {
  const links = Array.from(document.querySelectorAll(".nav-link"));
  links.forEach((link) => {
    link.addEventListener("click", () => {
      links.forEach((item) => item.classList.remove("active"));
      link.classList.add("active");
    });
  });
}

async function boot() {
  wireNavigation();
  $("configForm").addEventListener("submit", (event) => saveConfig(event).catch((error) => showToast(error.message)));
  $("refreshButton").addEventListener("click", () => {
    Promise.all([refreshStatus(), loadAuthProviders(), loadRequestDiagnostics(), loadAutoResearchStatus(), loadAutoResearchCandidates()]).catch((error) => showToast(error.message));
  });
  $("refreshDiagnosticsButton").addEventListener("click", () => loadRequestDiagnostics().catch((error) => showToast(error.message)));
  $("refreshAutoResearchButton").addEventListener("click", () => {
    Promise.all([loadAutoResearchStatus(), loadAutoResearchCandidates()]).catch((error) => showToast(error.message));
  });
  $("rotateTokenButton").addEventListener("click", () => rotateToken().catch((error) => showToast(error.message)));
  $("copyTokenButton").addEventListener("click", () => copyToken().catch((error) => showToast(error.message)));
  $("toggleTokenButton").addEventListener("click", toggleToken);
  $("modelsTestButton").addEventListener("click", () => testModels().catch((error) => setOutput(error.message)));
  $("chatTestButton").addEventListener("click", () => testChat().catch((error) => setOutput(error.message)));
  $("authProvider").addEventListener("change", renderSelectedAuthProvider);
  $("upstreamBaseUrl").addEventListener("input", updateWebAI2APILinks);
  $("openWebai2apiButton").addEventListener("click", () => openWebAI2APIConsole("/"));
  $("openWebai2apiLoginButton").addEventListener("click", () => openWebAI2APIConsole("/tools/cache"));
  $("embedWebai2apiButton").addEventListener("click", () => embedWebAI2APIConsole("/"));
  $("startAuthButton").addEventListener("click", () => startAuthFlow().catch((error) => {
    appendAuthLog(error.message);
    showToast(error.message);
  }));
  $("captureAuthButton").addEventListener("click", () => startAuthCapture().catch((error) => {
    appendAuthLog(error.message);
    showToast(error.message);
  }));
  $("refreshAuthButton").addEventListener("click", () => loadAuthProviders().catch((error) => showToast(error.message)));
  $("clearAuthButton").addEventListener("click", () => clearAuthCredential().catch((error) => showToast(error.message)));
  try {
    await loadConfig();
    await loadAuthProviders();
    await refreshStatus();
    await loadRequestDiagnostics();
    await loadAutoResearchStatus();
    await loadAutoResearchCandidates();
    setOutput("就绪。");
  } catch (error) {
    setOutput(error.message);
    showToast(error.message);
  }
}

boot();
