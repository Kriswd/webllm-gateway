const state = {
  config: null,
  tokenVisible: false,
  providers: [],
  authJobTimer: null,
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
  $("providerRuntimeTimeoutSeconds").value = config.providerRuntime?.requestTimeoutSeconds || 180;
  $("providerRuntimePromptMaxChars").value = config.providerRuntime?.promptMaxChars || 12000;
  $("nativeWebSearchPolicySelect").value = config.providerRuntime?.nativeWebSearchPolicy || "auto";
  $("toolActivationPolicySelect").value = config.tool_bridge?.activationPolicy || "auto";
  $("toolExposurePolicySelect").value = config.tool_bridge?.exposurePolicy || "safe";
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
  const modelCount = provider.models?.length || 0;
  const adapterText = provider.adapters?.length ? `适配器：${provider.adapters.join(" / ")}` : "";
  const capabilityText = [
    provider.capabilities?.text ? "文本" : "",
    provider.capabilities?.image ? "图片" : "",
    provider.capabilities?.video ? "视频" : "",
  ].filter(Boolean).join(" / ");
  $("authProviderLabel").textContent = provider.name;
  if (provider.route === "direct") {
    $("authStatus").textContent = authorized ? "已授权" : "未授权";
    $("authStatus").className = authorized ? "is-ok" : "is-warning";
    $("authUpdatedAt").textContent = credential.updatedAt ? `更新时间：${new Date(credential.updatedAt).toLocaleString()}` : "尚未授权";
    $("authBadge").textContent = authorized ? "本地可用" : "待登录";
    $("startAuthButton").textContent = "一键启动 DeepSeek 授权浏览器";
    $("captureAuthButton").textContent = "重新捕获登录态";
    $("captureAuthButton").disabled = false;
    $("clearAuthButton").disabled = !authorized;
  } else {
    $("authStatus").textContent = "交给 WebAI2API 原生管理";
    $("authStatus").className = "is-ok";
    $("authUpdatedAt").textContent = `${capabilityText || "网页模型"}，${adapterText || "适配器由 WebAI2API 管理"}`;
    $("authBadge").textContent = "复用 WebAI2API";
    $("startAuthButton").textContent = "打开 WebAI2API 登录管理";
    $("captureAuthButton").textContent = "本地捕获仅支持 DeepSeek";
    $("captureAuthButton").disabled = true;
    $("clearAuthButton").disabled = true;
  }
  $("authModelName").textContent = modelCount > 1 ? `${provider.models[0]} 等 ${modelCount} 个模型` : (provider.models?.[0] || `${provider.id}/default`);
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
  showToast("请在弹出的浏览器里完成 DeepSeek 登录");
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
    appendAuthLog("授权完成。现在可以用 deepseek-web/deepseek-chat 作为模型名。");
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
      requestTimeoutSeconds: Number($("providerRuntimeTimeoutSeconds").value) || 180,
      promptMaxChars: Number($("providerRuntimePromptMaxChars").value) || 12000,
      nativeWebSearchPolicy: $("nativeWebSearchPolicySelect").value,
    },
    tool_bridge: {
      ...(state.config?.tool_bridge || {}),
      activationPolicy: $("toolActivationPolicySelect").value,
      exposurePolicy: $("toolExposurePolicySelect").value,
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
    Promise.all([refreshStatus(), loadAuthProviders()]).catch((error) => showToast(error.message));
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
    setOutput("就绪。");
  } catch (error) {
    setOutput(error.message);
    showToast(error.message);
  }
}

boot();
