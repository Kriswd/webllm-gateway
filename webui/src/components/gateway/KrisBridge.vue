<script setup>
import { computed, onMounted, ref, watch } from 'vue';
import { Modal, message } from 'ant-design-vue';
import { useSettingsStore } from '@/stores/settings';
import {
  ApiOutlined,
  AppstoreOutlined,
  CheckCircleOutlined,
  ClockCircleOutlined,
  CopyOutlined,
  EditOutlined,
  ExperimentOutlined,
  LinkOutlined,
  LoginOutlined,
  PlusOutlined,
  ReloadOutlined,
  RocketOutlined,
  SettingOutlined,
  ToolOutlined,
  UserSwitchOutlined,
} from '@ant-design/icons-vue';

const settingsStore = useSettingsStore();

const loading = ref(false);
const actionLoading = ref(false);
const progressVisible = ref(false);
const loadError = ref('');
const actionError = ref('');
const actionLogs = ref([]);
const actionKind = ref('');
const accountActionId = ref('');
const pendingWebAI2APILogin = ref(null);
const selectedProviderId = ref('');
const modelSearch = ref('');
const modelScope = ref('selected');
const imagePrompt = ref('一张干净的产品摄影图：蓝色玻璃质感的 AI 网关设备，白色背景，柔和棚拍光');
const imageModel = ref('gpt-image-2');
const imageGenerating = ref(false);
const imageError = ref('');
const imageResultUrl = ref('');
const cdpUrl = ref('http://127.0.0.1:9222');
const configProfile = ref('cc-switch');
const accountEditOpen = ref(false);
const accountEditSaving = ref(false);
const remoteAuthOpen = ref(false);
const remoteAuthSaving = ref(false);
const remoteAuthUrl = ref('');
const remoteAuthProviderId = ref('');
const accountEditForm = ref({
  id: '',
  displayName: '',
  planType: 'unknown',
  note: '',
});
const diagnosticsOpen = ref(false);

const onboarding = ref({
  gateway: {
    baseUrl: '/v1',
    apiKey: '',
    defaultModel: '',
    toolMode: 'prompt',
    upstreamBaseUrl: '',
  },
  summary: {
    providers: 0,
    models: 0,
    authorizedProviders: 0,
    authorizedDirectProviders: 0,
    webAI2APIProviders: 0,
  },
  providers: [],
  models: [],
  connectionProfiles: [],
  recommendedConnectionProfile: null,
});

const gatewayBaseUrl = computed(() => `${window.location.origin}${onboarding.value.gateway?.baseUrl || '/v1'}`);
const gatewayRootUrl = computed(() => gatewayBaseUrl.value.replace(/\/v1\/?$/, ''));
const gatewayToken = computed(() => onboarding.value.gateway?.apiKey || '');

const providerPriority = ['deepseek-web', 'qwen', 'qwen-coder', 'chatgpt'];
const providers = computed(() => {
  const list = onboarding.value.providers || [];
  return [...list].sort((a, b) => {
    const ai = providerPriority.indexOf(a.id);
    const bi = providerPriority.indexOf(b.id);
    const ap = ai === -1 ? 99 : ai;
    const bp = bi === -1 ? 99 : bi;
    if (ap !== bp) return ap - bp;
    return String(a.name).localeCompare(String(b.name));
  });
});

const selectedProvider = computed(() => {
  return providers.value.find((item) => item.id === selectedProviderId.value) || providers.value[0] || null;
});

const remoteAuthProvider = computed(() => {
  return providers.value.find((item) => item.id === remoteAuthProviderId.value) || selectedProvider.value;
});

const selectedProviderAccounts = computed(() => {
  return Array.isArray(selectedProvider.value?.accounts) ? selectedProvider.value.accounts : [];
});

const currentAccount = computed(() => {
  return selectedProviderAccounts.value.find((account) => account.id === selectedProvider.value?.currentAccountId)
    || selectedProviderAccounts.value.find((account) => account.current)
    || selectedProviderAccounts.value[0]
    || null;
});

const connectionProfiles = computed(() => (
  Array.isArray(onboarding.value.connectionProfiles) ? onboarding.value.connectionProfiles : []
));
const recommendedConnectionProfile = computed(() => (
  onboarding.value.recommendedConnectionProfile || connectionProfiles.value[0] || null
));
const selectedConnectionProfile = computed(() => {
  const provider = selectedProvider.value;
  if (!provider) return recommendedConnectionProfile.value;
  const visibleModels = new Set(providerVisibleModelIds(provider));
  return connectionProfiles.value.find((profile) => (
    profile.providerId === provider.id && visibleModels.has(profile.modelId) && profile.available !== false
  ))
    || connectionProfiles.value.find((profile) => profile.providerId === provider.id && profile.available !== false)
    || connectionProfiles.value.find((profile) => profile.providerId === provider.id)
    || null;
});
const selectedProviderDefaultModel = computed(() => (
  selectedConnectionProfile.value?.modelId || providerAvailableModels(selectedProvider.value)[0] || ''
));
const selectedClientModel = computed(() => (
  selectedProvider.value
    ? (selectedProviderDefaultModel.value || '<当前平台没有验证可用模型>')
    : (recommendedConnectionProfile.value?.modelId || onboarding.value.gateway?.defaultModel || '<模型 ID>')
));

const filteredModels = computed(() => {
  const query = modelSearch.value.trim().toLowerCase();
  return (onboarding.value.models || []).filter((model) => {
    const modelId = String(model.id || '');
    const owner = providerForModel(modelId);
    const modelOwnerId = onboarding.value.models?.find((item) => item.id === modelId)?.owned_by;
    const hasKnownHiddenOwner = !owner && providers.value.some((provider) => provider.id === modelOwnerId);
    if (hasKnownHiddenOwner) return false;
    const inScope = modelScope.value === 'all' || !selectedProvider.value || providerVisibleModelIds(selectedProvider.value).includes(modelId);
    const matchesQuery = !query || modelId.toLowerCase().includes(query) || owner?.name?.toLowerCase().includes(query);
    return inScope && matchesQuery;
  });
});

const imageModelOptions = computed(() => {
  const discovered = (onboarding.value.models || [])
    .filter((model) => {
      const modelId = String(model.id || '').toLowerCase();
      const capabilities = model.capabilities || {};
      return capabilities.image === true
        || model.type === 'image'
        || modelId.includes('image')
        || modelId.includes('gpt-image');
    })
    .map((model) => String(model.id || '').trim())
    .filter(Boolean);
  return [...new Set(['gpt-image-2', 'gpt-image-1.5', ...discovered])]
    .map((value) => ({ label: value, value }));
});

const imageRequestExample = computed(() => JSON.stringify({
  model: imageModel.value || 'gpt-image-2',
  prompt: imagePrompt.value,
  n: 1,
  response_format: 'b64_json',
}, null, 2));

const ccSwitchClientConfig = computed(() => JSON.stringify({
  env: {
    ANTHROPIC_MODEL: selectedClientModel.value,
    ANTHROPIC_BASE_URL: gatewayRootUrl.value,
    ANTHROPIC_API_BASE_URL: gatewayRootUrl.value,
    ANTHROPIC_AUTH_TOKEN: gatewayToken.value || '<网关令牌>',
    ANTHROPIC_DEFAULT_HAIKU_MODEL: selectedClientModel.value,
    ANTHROPIC_DEFAULT_SONNET_MODEL: selectedClientModel.value,
    ANTHROPIC_DEFAULT_OPUS_MODEL: selectedClientModel.value,
  },
}, null, 2));

const openAIClientConfig = computed(() => [
  '# OpenAI-compatible',
  `base_url = ${gatewayBaseUrl.value}`,
  `api_key = ${gatewayToken.value || '<网关令牌>'}`,
  `model = ${selectedClientModel.value}`,
].join('\n'));

const anthropicClientConfig = computed(() => [
  '# Anthropic-compatible',
  `ANTHROPIC_BASE_URL=${gatewayRootUrl.value}`,
  `ANTHROPIC_API_BASE_URL=${gatewayRootUrl.value}`,
  `ANTHROPIC_AUTH_TOKEN=${gatewayToken.value || '<网关令牌>'}`,
  `ANTHROPIC_MODEL=${selectedClientModel.value}`,
  `ANTHROPIC_DEFAULT_HAIKU_MODEL=${selectedClientModel.value}`,
  `ANTHROPIC_DEFAULT_SONNET_MODEL=${selectedClientModel.value}`,
  `ANTHROPIC_DEFAULT_OPUS_MODEL=${selectedClientModel.value}`,
  '',
  `# 推荐平台：${selectedConnectionProfile.value?.providerName || selectedProvider.value?.name || 'Gateway 默认'}`,
].join('\n'));

const clientConfigOptions = [
  { label: 'cc-switch', value: 'cc-switch' },
  { label: 'OpenAI', value: 'openai' },
  { label: 'Anthropic', value: 'anthropic' },
];

const clientConfigTitle = computed(() => {
  if (configProfile.value === 'openai') return 'OpenAI 兼容配置';
  if (configProfile.value === 'anthropic') return 'Anthropic 环境变量';
  return 'cc-switch 专用配置';
});

const clientConfig = computed(() => {
  if (configProfile.value === 'openai') return openAIClientConfig.value;
  if (configProfile.value === 'anthropic') return anthropicClientConfig.value;
  return ccSwitchClientConfig.value;
});

const modelColumns = [
  { title: '模型 ID', dataIndex: 'id', key: 'id' },
  { title: '来源', key: 'provider', width: 180 },
  { title: '能力', key: 'capability', width: 180 },
  { title: '账号可用性', key: 'availability', width: 180 },
  { title: '操作', key: 'action', width: 110 },
];

const stepItems = computed(() => [
  { title: '选择平台', status: selectedProvider.value ? 'finish' : 'process' },
  { title: '完成授权', status: isProviderReady(selectedProvider.value) ? 'finish' : actionLoading.value ? 'process' : 'wait' },
  { title: '复制接入', status: filteredModels.value.length ? 'finish' : 'wait' },
]);

const progressTitle = computed(() => (actionKind.value === 'smoke' ? '接入检测' : '网页登录进度'));
const selectedProviderAccountStatus = computed(() => {
  const provider = selectedProvider.value;
  if (!provider || selectedProviderAccounts.value.length) return null;
  const declaredModelCount = providerDeclaredModelCount(provider);

  if (provider.loginKind !== 'direct') {
    return {
      tone: 'web',
      title: '需要登录',
      description: `${provider.name} 还没有检测到可用账号。点击“打开网页登录授权”，在弹出的窗口里完成登录，回到这里刷新模型即可。`,
      note: declaredModelCount
        ? `本地已有 ${declaredModelCount} 个候选模型配置；授权完成后会自动检测实际可用模型。`
        : '授权完成并刷新后，会显示实际可用账号和模型。',
      action: '不需要手动复制 Cookie 或填写浏览器参数。',
    };
  }

  return {
    tone: 'direct',
    title: '需要授权',
    description: `点击“打开授权浏览器”完成 ${provider.name} 登录后，Gateway 会自动检测账号和模型。`,
    note: '如果刚完成登录但还没捕获，点弹窗里的“重新检测登录态”；普通刷新只会刷新已保存结果。',
    action: '登录信息只保存在本机。',
  };
});
const accountSectionHint = computed(() => {
  if (currentAccount.value) return `当前使用：${currentAccount.value.displayName}`;
  return selectedProviderAccountStatus.value?.title || '当前平台还没有检测到可用账号';
});
const progressStepItems = computed(() => {
  if (actionKind.value === 'smoke') {
    return [{ title: '发起检测' }, { title: '等待响应' }, { title: '完成' }];
  }
  return [
    { title: '打开窗口' },
    { title: '完成登录' },
    { title: '刷新模型' },
  ];
});
const progressCurrent = computed(() => (actionLoading.value ? 1 : actionError.value ? 0 : 2));
const progressStatus = computed(() => (actionError.value ? 'error' : 'process'));

function isProviderReady(provider) {
  if (!provider) return false;
  if (provider.loginKind === 'direct') return Boolean(provider.credential?.authorized);
  if (provider.webAI2APIAuth?.checked) return Boolean(provider.credential?.authorized);
  return providerAccountCount(provider) > 0 || providerModelCount(provider) > 0;
}

function isProviderAuthorized(provider) {
  return Boolean(provider?.credential?.authorized);
}

function providerForModel(modelId) {
  return providers.value.find((provider) => providerVisibleModelIds(provider).includes(modelId))
    || providers.value.find((provider) => provider.id === onboarding.value.models?.find((model) => model.id === modelId)?.owned_by && providerVisibleModelIds(provider).includes(modelId))
    || null;
}

function providerAvailableModels(provider) {
  return Array.isArray(provider?.availableModels) ? provider.availableModels : [];
}

function providerVisibleModelIds(provider) {
  if (!provider) return [];
  const availabilityIds = Object.keys(provider.modelAvailability || {});
  return availabilityIds.length ? availabilityIds : providerAvailableModels(provider);
}

function providerModelCount(provider) {
  if (!provider) return 0;
  if (Number.isFinite(provider.modelCount)) return provider.modelCount;
  return providerAvailableModels(provider).length;
}

function providerAccountCount(provider) {
  return Array.isArray(provider?.accounts) ? provider.accounts.length : 0;
}

function providerDeclaredModelCount(provider) {
  return Array.isArray(provider?.models) ? provider.models.length : 0;
}

function providerSubtitle(provider) {
  const accountCount = providerAccountCount(provider);
  const modelCount = providerModelCount(provider);
  if (provider?.loginKind !== 'direct') {
    if (!accountCount && !modelCount) {
      const declaredModelCount = providerDeclaredModelCount(provider);
      return declaredModelCount
        ? `需要登录 · ${declaredModelCount} 个模型待检测`
        : '需要登录';
    }
    return `${accountCount} 个账号 · ${modelCount} 个可用模型`;
  }
  return `${accountCount} 个账号 · ${modelCount} 个可用模型`;
}

function accountPlanLabel(planType) {
  const labels = {
    free: 'FREE',
    plus: 'PLUS',
    pro: 'PRO',
    team: 'TEAM',
    unknown: '未知权益',
  };
  return labels[String(planType || 'unknown').toLowerCase()] || '未知权益';
}

function accountPlanColor(planType) {
  const colors = {
    free: 'default',
    plus: 'green',
    pro: 'gold',
    team: 'blue',
    unknown: 'default',
  };
  return colors[String(planType || 'unknown').toLowerCase()] || 'default';
}

function accountSourceLabel(account) {
  if (account?.source === 'direct-profile') return '本机授权账号';
  return '网页登录账号';
}

function modelAvailability(modelId) {
  const provider = providerForModel(modelId);
  return provider?.modelAvailability?.[modelId] || { status: 'pending' };
}

function modelAvailabilityTag(modelId) {
  const availability = modelAvailability(modelId);
  const status = availability.status || 'pending';
  if (status === 'available') return { label: '已验证可用', color: 'success' };
  if (status === 'unavailable') return { label: '验证失败', color: 'error' };
  return { label: '待验证', color: 'default' };
}

function accountValidationSummary(account) {
  const validation = account?.validation || {};
  const values = Object.values(validation);
  const available = values.filter((item) => item?.status === 'available').length;
  const unavailable = values.filter((item) => item?.status === 'unavailable').length;
  if (!values.length) return '尚未验证模型';
  return `${available} 个可用，${unavailable} 个失败`;
}

function accountValidationFailures(account) {
  const validation = account?.validation || {};
  return Object.entries(validation)
    .filter(([, item]) => item?.status === 'unavailable')
    .map(([modelId, item]) => ({
      modelId,
      message: item?.message || '模型验证失败，未返回具体原因',
    }));
}

function capabilityLabels(provider) {
  if (!provider) return [];
  const caps = provider.capabilities || {};
  const labels = [];
  if (caps.text) labels.push({ label: '文本', color: 'blue' });
  if (caps.image) labels.push({ label: '图片', color: 'green' });
  if (caps.video) labels.push({ label: '视频', color: 'orange' });
  return labels;
}

function modelCapabilityLabels(model) {
  if (!model) return [];
  const labels = [];
  const modelId = String(model.id || '');
  const type = String(model.type || '').toLowerCase();
  const imagePolicy = String(model.image_policy || model.imagePolicy || '').toLowerCase();
  const caps = model.capabilities || {};

  if (type === 'text') {
    labels.push({ label: '文本', color: 'blue' });
    if (imagePolicy === 'optional' || imagePolicy === 'required') {
      labels.push({ label: '可附图', color: 'green' });
    }
    return labels;
  }

  if (type === 'image' || modelId.includes('image')) {
    labels.push({ label: '图片', color: 'green' });
    return labels;
  }

  if (type === 'video' || modelId.includes('video')) {
    labels.push({ label: '视频', color: 'orange' });
    return labels;
  }

  return capabilityLabels(providerForModel(modelId));
}

function appendLog(text) {
  actionLogs.value = [...actionLogs.value, `${new Date().toLocaleTimeString()}  ${text}`];
}

function sleep(ms) {
  return new Promise((resolve) => setTimeout(resolve, ms));
}

async function loadOnboarding() {
  loading.value = true;
  loadError.value = '';
  try {
    const res = await fetch('/api/admin/onboarding');
    if (!res.ok) throw new Error(`HTTP ${res.status}`);
    const data = await res.json();
    onboarding.value = data;
    if (data.defaultCdpUrl && cdpUrl.value === 'http://127.0.0.1:9222') {
      cdpUrl.value = data.defaultCdpUrl;
    }
    if (!selectedProviderId.value && providers.value.length) {
      selectedProviderId.value = providers.value[0].id;
    }
  } catch (error) {
    loadError.value = error.message || String(error);
  } finally {
    loading.value = false;
  }
}

function resetActionState(kind) {
  actionKind.value = kind;
  actionError.value = '';
  actionLogs.value = [];
  progressVisible.value = true;
}

async function handleStartLogin(options = {}) {
  const provider = selectedProvider.value;
  if (!provider) return;
  if (provider.loginKind === 'direct') {
    await startDirectAuth(provider);
    return;
  }
  const newAccount = Boolean(options.newAccount);
  Modal.confirm({
    title: newAccount ? '添加授权账号？' : '打开网页登录授权？',
    content: newAccount
      ? '这会为新账号打开独立的网页登录窗口。完成登录后回到这里，点击“恢复 API 并刷新”。'
      : '这会打开当前账号的网页登录窗口，用于修复登录态或更新账号权益。完成后回到这里，点击“恢复 API 并刷新”。',
    okText: newAccount ? '添加并授权' : '打开授权窗口',
    cancelText: '取消',
    async onOk() {
      await startWebAI2APILogin(provider, { newAccount });
    },
  });
}

async function startDirectAuth(provider) {
  resetActionState('direct');
  actionLoading.value = true;
  try {
    appendLog(`正在为 ${provider.name} 启动授权浏览器`);
    const browserRes = await fetch(`/api/admin/onboarding/providers/${encodeURIComponent(provider.id)}/login`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ cdpUrl: cdpUrl.value }),
    });
    const browserData = await browserRes.json();
    if (!browserRes.ok) throw new Error(browserData.detail || `HTTP ${browserRes.status}`);
    appendLog(browserData.message || '授权浏览器已启动');
    if (!browserData.started && browserData.loginUrl) {
      window.open(browserData.loginUrl, '_blank', 'noopener,noreferrer');
    }

    await captureDirectAuth(provider);
    message.success(`${provider.name} 授权完成`);
    await loadOnboarding();
  } catch (error) {
    actionError.value = error.message || String(error);
    appendLog(`授权失败：${actionError.value}`);
    message.error(actionError.value);
  } finally {
    actionLoading.value = false;
  }
}

async function captureDirectAuth(provider) {
  appendLog('正在检测网页登录状态');
  const jobRes = await fetch('/api/admin/web-auth/jobs', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ provider: provider.id, cdpUrl: cdpUrl.value }),
  });
  const job = await jobRes.json();
  if (!jobRes.ok) throw new Error(job.detail || `HTTP ${jobRes.status}`);
  await pollAuthJob(job.id);
}

async function retryDirectAuthCapture(provider = selectedProvider.value) {
  if (!provider) return;
  actionKind.value = 'direct';
  actionError.value = '';
  progressVisible.value = true;
  actionLoading.value = true;
  try {
    appendLog(`正在重新检测 ${provider.name} 登录态`);
    await captureDirectAuth(provider);
    message.success(`${provider.name} 授权完成`);
    await loadOnboarding();
  } catch (error) {
    actionError.value = error.message || String(error);
    appendLog(`重新检测登录态失败：${actionError.value}`);
    message.error(actionError.value);
  } finally {
    actionLoading.value = false;
  }
}

function openRemoteAuthModal(provider = selectedProvider.value) {
  if (!provider) return;
  remoteAuthProviderId.value = provider.id;
  remoteAuthUrl.value = '';
  remoteAuthOpen.value = true;
}

async function submitRemoteAuthUrl() {
  const providerId = remoteAuthProvider.value?.id || selectedProvider.value?.id;
  const url = remoteAuthUrl.value.trim();
  if (!providerId) {
    message.warning('请先选择平台');
    return;
  }
  if (!url) {
    message.warning('请粘贴授权 URL');
    return;
  }
  remoteAuthSaving.value = true;
  try {
    const res = await fetch('/api/admin/web-auth/callback-url', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ provider: providerId, url }),
    });
    const data = await res.json().catch(() => ({}));
    if (!res.ok) throw new Error(data.detail || data.message || `HTTP ${res.status}`);
    remoteAuthOpen.value = false;
    message.success(data.message || '远程授权 URL 已导入');
    await loadOnboarding();
  } catch (error) {
    message.error(error.message || String(error));
  } finally {
    remoteAuthSaving.value = false;
  }
}

async function pollAuthJob(jobId) {
  for (let i = 0; i < 150; i += 1) {
    const res = await fetch(`/api/admin/web-auth/jobs/${jobId}`);
    const job = await res.json();
    if (!res.ok) throw new Error(job.detail || `HTTP ${res.status}`);
    if (job.message) appendLog(job.message);
    if (job.status === 'succeeded') return job;
    if (job.status === 'failed') throw new Error(job.message || '授权失败');
    await sleep(1600);
  }
  throw new Error('等待网页登录超时，请确认登录完成后重试');
}

async function startWebAI2APILogin(provider, options = {}) {
  resetActionState('webai2api');
  pendingWebAI2APILogin.value = {
    providerId: provider.id,
    accountId: '',
    workerName: '',
  };
  actionLoading.value = true;
  try {
    const newAccount = Boolean(options.newAccount);
    appendLog(newAccount ? '正在创建新的网页登录授权窗口' : '正在打开网页登录授权窗口');
    const res = await fetch(`/api/admin/onboarding/providers/${encodeURIComponent(provider.id)}/login`, {
      method: 'POST',
      headers: {
        ...settingsStore.getHeaders(),
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({ providerId: provider.id, newAccount }),
    });
    const data = await res.json().catch(() => ({}));
    if (!res.ok || data.success === false) {
      throw new Error(data.message || data.error?.message || `HTTP ${res.status}`);
    }
    if (data.sidecarStarted) {
      appendLog('授权服务已准备好');
    }
    pendingWebAI2APILogin.value = {
      providerId: data.providerId || provider.id,
      accountId: data.accountId || '',
      workerName: data.workerName || '',
    };
    appendLog(data.message || '已进入网页登录授权模式');
    appendLog(`请在打开的窗口里完成 ${provider.name} 登录，然后回到这里点击“恢复 API 并刷新”`);
    const openedTarget = openWebAI2APILoginTarget(data, provider);
    message.success(openedTarget ? '网页登录窗口已打开' : `${data.browserLabel || 'Camoufox'} 登录窗口已打开`);
  } catch (error) {
    pendingWebAI2APILogin.value = null;
    actionError.value = error.message || String(error);
    appendLog(`登录模式启动失败：${actionError.value}`);
    message.error(actionError.value);
  } finally {
    actionLoading.value = false;
  }
}

function openWebAI2APILoginTarget(data, provider) {
  const targetUrl = data.openUrl || data.loginUrl || data.displayUrl || '';
  if (targetUrl) {
    window.open(targetUrl, '_blank', 'noopener,noreferrer');
    return true;
  }
  const browserLabel = data.browserLabel || 'Camoufox';
  appendLog(`${provider.name} 登录窗口已由 ${browserLabel} 打开，请直接在该浏览器窗口完成登录`);
  return false;
}

async function finishWebAI2APILogin({ close = false } = {}) {
  actionLoading.value = true;
  try {
    appendLog('正在恢复 Gateway API 调用模式');
    const res = await fetch('/api/admin/onboarding/login/finish', {
      method: 'POST',
      headers: {
        ...settingsStore.getHeaders(),
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({}),
    });
    const data = await res.json().catch(() => ({}));
    if (!res.ok || data.success === false) {
      throw new Error(data.message || data.detail || data.error?.message || `HTTP ${res.status}`);
    }
    appendLog(data.message || 'Gateway API 已恢复可调用模式');
    const autoValidated = await autoValidateWebAI2APILogin();
    message.success(autoValidated ? '已恢复 API 并自动检测模型' : '已恢复 API，模型状态已刷新');
    if (close) progressVisible.value = false;
  } catch (error) {
    actionError.value = error.message || String(error);
    appendLog(`恢复 API 模式失败：${actionError.value}`);
    message.error(actionError.value);
  } finally {
    actionLoading.value = false;
  }
}

async function autoValidateWebAI2APILogin() {
  await loadOnboarding();
  const login = pendingWebAI2APILogin.value || {};
  const providerId = login.providerId || selectedProvider.value?.id;
  const provider = providers.value.find((item) => item.id === providerId);
  if (!provider) {
    appendLog('已恢复 API，但没有找到刚授权的平台，已刷新当前模型状态');
    pendingWebAI2APILogin.value = null;
    return false;
  }

  selectedProviderId.value = provider.id;
  const accounts = Array.isArray(provider.accounts) ? provider.accounts : [];
  const account = accounts.find((item) => item.id === login.accountId)
    || accounts.find((item) => item.id === provider.currentAccountId)
    || accounts.find((item) => item.current)
    || accounts[0];
  if (!account) {
    appendLog(`${provider.name} 尚未检测到授权账号，模型状态保持待检测`);
    pendingWebAI2APILogin.value = null;
    return false;
  }

  accountActionId.value = `${account.id}:validate`;
  try {
    appendLog(`正在自动检测 ${provider.name} 的模型可用性`);
    const res = await fetch('/api/admin/accounts/validate', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        providerId: provider.id,
        accountId: account.id,
        modelIds: providerVisibleModelIds(provider),
        force: true,
      }),
    });
    const data = await res.json().catch(() => ({}));
    if (!res.ok) throw new Error(data.detail || data.message || `HTTP ${res.status}`);
    const validation = data.validation || {};
    const values = Object.values(validation);
    const okCount = values.filter((item) => item?.status === 'available').length;
    const failed = values.filter((item) => item?.status === 'unavailable').length;
    if (values.length) {
      appendLog(`模型自动检测完成：${okCount}/${values.length} 可用${failed ? `，${failed} 个不可用` : ''}`);
    } else {
      appendLog('模型自动检测完成：当前没有返回可检测模型');
    }
    await loadOnboarding();
    return true;
  } catch (error) {
    const errorMessage = error.message || String(error);
    appendLog(`模型自动检测失败：${errorMessage}`);
    message.warning(errorMessage);
    await loadOnboarding();
    return false;
  } finally {
    accountActionId.value = '';
    pendingWebAI2APILogin.value = null;
  }
}

async function copyText(text, successText) {
  if (!text) {
    message.warning('没有可复制的内容');
    return;
  }
  await navigator.clipboard.writeText(text);
  message.success(successText);
}

async function runImageSmokeTest() {
  const prompt = imagePrompt.value.trim();
  if (!prompt) {
    message.warning('请先输入图片提示词');
    return;
  }
  imageGenerating.value = true;
  imageError.value = '';
  imageResultUrl.value = '';
  try {
    const headers = {
      ...settingsStore.getHeaders(),
      'Content-Type': 'application/json',
    };
    if (!headers.Authorization && gatewayToken.value) {
      headers.Authorization = `Bearer ${gatewayToken.value}`;
    }
    const res = await fetch('/v1/images/generations', {
      method: 'POST',
      headers,
      body: JSON.stringify({
        model: imageModel.value || 'gpt-image-2',
        prompt,
        n: 1,
        response_format: 'b64_json',
      }),
    });
    const data = await res.json().catch(() => ({}));
    if (!res.ok) {
      const detail = data.detail || data.error?.message || `HTTP ${res.status}`;
      throw new Error(String(detail));
    }
    const first = Array.isArray(data.data) ? data.data[0] : null;
    if (first?.b64_json) {
      imageResultUrl.value = `data:image/png;base64,${first.b64_json}`;
    } else if (first?.url) {
      imageResultUrl.value = first.url;
    } else {
      throw new Error('接口已返回，但没有拿到图片内容');
    }
    message.success('图片生成链路可用');
  } catch (error) {
    imageError.value = error.message || '图片生成失败，请先确认 ChatGPT 授权账号可用';
  } finally {
    imageGenerating.value = false;
  }
}

function copyModel(modelId) {
  copyText(modelId, '模型 ID 已复制');
}

function copySelectedDefaultModel() {
  if (!selectedProviderDefaultModel.value) {
    message.warning('当前平台没有验证可用的模型');
    return;
  }
  copyText(selectedProviderDefaultModel.value, '模型 ID 已复制');
}

function smokeResultLabel(id) {
  const labels = {
    models: '模型列表',
    openai_text: 'OpenAI 文本',
    openai_tool_use: 'OpenAI 工具调用',
    anthropic_tool_use: 'Anthropic 工具调用',
    anthropic_tool_result: 'Anthropic 工具结果',
    auth: '授权状态',
    unsupported: '支持范围',
  };
  return labels[id] || id;
}

function smokeResultDetail(item) {
  if (item.message) return item.message;
  const detail = item.detail || {};
  if (detail.model) return detail.model;
  if (detail.name) return `${detail.name} ${JSON.stringify(detail.input || {})}`;
  if (detail.text) return detail.text;
  return '';
}

async function runProviderSmoke() {
  const provider = selectedProvider.value;
  if (!provider) return;
  resetActionState('smoke');
  actionLoading.value = true;
  try {
    appendLog(`正在检测 ${provider.name}：模型、OpenAI 工具调用、Anthropic 工具闭环`);
    const res = await fetch(`/api/admin/provider-smoke/${provider.id}`, { method: 'POST' });
    const data = await res.json().catch(() => ({}));
    if (!res.ok) throw new Error(data.detail || `HTTP ${res.status}`);
    for (const item of data.results || []) {
      const state = item.ok ? '通过' : '失败';
      const detail = smokeResultDetail(item);
      appendLog(`${state} · ${smokeResultLabel(item.id)}${detail ? ` · ${detail}` : ''}`);
    }
    if (!data.ok) {
      actionError.value = `${provider.name} 工具调用检测未全部通过：${data.passed || 0}/${data.total || 0}`;
      message.error(actionError.value);
      return;
    }
    message.success(`${provider.name} 工具调用检测通过`);
  } catch (error) {
    actionError.value = error.message || String(error);
    appendLog(`工具调用检测失败：${actionError.value}`);
    message.error(actionError.value);
  } finally {
    actionLoading.value = false;
  }
}

async function selectAccount(account) {
  const provider = selectedProvider.value;
  if (!provider || !account || account.current) return;
  accountActionId.value = account.id;
  try {
    const res = await fetch('/api/admin/accounts/select', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ providerId: provider.id, accountId: account.id }),
    });
    const data = await res.json().catch(() => ({}));
    if (!res.ok) throw new Error(data.detail || data.message || `HTTP ${res.status}`);
    message.success(`已切换到 ${account.displayName}`);
    await loadOnboarding();
  } catch (error) {
    message.error(error.message || String(error));
  } finally {
    accountActionId.value = '';
  }
}

async function validateAccount(account) {
  const provider = selectedProvider.value;
  if (!provider || !account) return;
  accountActionId.value = `${account.id}:validate`;
  try {
    const res = await fetch('/api/admin/accounts/validate', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        providerId: provider.id,
        accountId: account.id,
        modelIds: providerVisibleModelIds(provider),
        force: true,
      }),
    });
    const data = await res.json().catch(() => ({}));
    if (!res.ok) throw new Error(data.detail || data.message || `HTTP ${res.status}`);
    const validation = data.validation || {};
    const values = Object.values(validation);
    const okCount = values.filter((item) => item?.status === 'available').length;
    const failed = Object.entries(validation).filter(([, item]) => item?.status === 'unavailable');
    if (values.length > 0 && okCount === 0) {
      const firstFailure = failed[0]?.[1];
      message.error(`模型验证失败：${firstFailure?.message || '没有可用模型'}`);
    } else if (failed.length > 0) {
      message.warning(`模型验证完成：${okCount}/${values.length} 可用，${failed.length} 个失败`);
    } else {
      message.success(`模型验证完成：${okCount}/${values.length} 可用`);
    }
    await loadOnboarding();
  } catch (error) {
    message.error(error.message || String(error));
  } finally {
    accountActionId.value = '';
  }
}

function openAccountEditor(account) {
  if (!account) return;
  accountEditForm.value = {
    id: account.id,
    displayName: account.displayName || '',
    planType: account.planType || 'unknown',
    note: account.note || '',
  };
  accountEditOpen.value = true;
}

async function saveAccountEdit() {
  const provider = selectedProvider.value;
  if (!provider || !accountEditForm.value.id) return;
  accountEditSaving.value = true;
  try {
    const res = await fetch(`/api/admin/accounts/${encodeURIComponent(accountEditForm.value.id)}`, {
      method: 'PATCH',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        providerId: provider.id,
        displayName: accountEditForm.value.displayName,
        planType: accountEditForm.value.planType,
        note: accountEditForm.value.note,
      }),
    });
    const data = await res.json().catch(() => ({}));
    if (!res.ok) throw new Error(data.detail || data.message || `HTTP ${res.status}`);
    accountEditOpen.value = false;
    message.success('账号信息已更新');
    await loadOnboarding();
  } catch (error) {
    message.error(error.message || String(error));
  } finally {
    accountEditSaving.value = false;
  }
}

function openAdvanced(path) {
  window.location.href = path;
}

watch(selectedProviderId, () => {
  modelScope.value = 'selected';
});

onMounted(loadOnboarding);
</script>

<template>
  <div class="onboarding-shell">
    <section class="hero-panel">
      <div class="hero-copy">
        <a-tag color="blue">Local WebAI Access</a-tag>
        <h1>把网页账号变成可工具调用的 API，实现养虾养马自由！</h1>
        <p>登录网页账号，自动检测可用模型，支持在 OpenClaw、Hermes、Claude Code、Codex 或其它兼容 OpenAI 和 Anthropic API 的客户端调用。</p>
        <div class="hero-actions">
          <a-button type="primary" size="large" :loading="actionLoading" @click="handleStartLogin">
            <template #icon><LoginOutlined /></template>
            {{ selectedProvider?.loginKind === 'direct' ? '打开授权浏览器' : '登录或修复账号' }}
          </a-button>
          <a-button size="large" @click="loadOnboarding">
            <template #icon><ReloadOutlined /></template>
            刷新模型
          </a-button>
        </div>
      </div>

      <div class="hero-status">
        <a-steps size="small" :items="stepItems" />
        <div class="stat-grid">
          <div class="stat-item">
            <span>授权平台</span>
            <strong>{{ onboarding.summary.providers }}</strong>
          </div>
          <div class="stat-item">
            <span>模型</span>
            <strong>{{ onboarding.summary.models }}</strong>
          </div>
          <div class="stat-item">
            <span>可用账号</span>
            <strong>{{ onboarding.summary.authorizedProviders ?? onboarding.summary.authorizedDirectProviders }}</strong>
          </div>
        </div>
      </div>
    </section>

    <a-alert
      v-if="loadError"
      type="warning"
      show-icon
      :message="`读取向导数据失败：${loadError}`"
      style="margin-bottom: 16px;"
    />

    <a-spin :spinning="loading">
      <div class="workspace-grid">
        <section class="panel provider-panel">
          <div class="panel-heading">
            <div>
              <h2>选择网页登录平台</h2>
              <p>只保留当前验证过的可用入口，先完成一个平台授权即可开始调用。</p>
            </div>
            <AppstoreOutlined />
          </div>

          <div class="provider-list">
            <button
              v-for="provider in providers"
              :key="provider.id"
              class="provider-item"
              :class="{ active: provider.id === selectedProviderId }"
              type="button"
              @click="selectedProviderId = provider.id"
            >
              <span class="provider-main">
                <strong>{{ provider.name }}</strong>
                <small>{{ providerSubtitle(provider) }}</small>
              </span>
              <span class="provider-tags">
                <a-tag v-if="isProviderAuthorized(provider)" color="success">已授权</a-tag>
                <a-tag v-else-if="provider.loginKind === 'direct'" color="warning">未授权</a-tag>
                <a-tag v-else color="warning">需授权</a-tag>
              </span>
            </button>
          </div>
        </section>

        <section class="panel action-panel">
          <div class="panel-heading">
            <div>
              <h2>{{ selectedProvider?.name || '未选择平台' }}</h2>
            </div>
            <CheckCircleOutlined v-if="isProviderReady(selectedProvider)" class="ready-icon" />
            <ClockCircleOutlined v-else class="pending-icon" />
          </div>

          <div v-if="selectedProvider" class="provider-detail">
            <div class="detail-row">
              <span>授权方式</span>
              <strong>{{ selectedProvider.loginKind === 'direct' ? '自动检测本机登录' : '网页登录授权' }}</strong>
            </div>
            <div class="detail-row">
              <span>登录入口</span>
              <a-typography-link :href="selectedProvider.loginUrl" target="_blank">
                {{ selectedProvider.loginUrl }}
              </a-typography-link>
            </div>
            <div class="capability-row">
              <a-tag v-for="cap in capabilityLabels(selectedProvider)" :key="cap.label" :color="cap.color">
                {{ cap.label }}
              </a-tag>
            </div>
            <div class="account-section">
              <div class="section-title">
                <div>
                  <strong>授权账号</strong>
                  <span>{{ accountSectionHint }}</span>
                </div>
                <a-button
                  class="validate-current-button"
                  size="small"
                  :disabled="!currentAccount"
                  :loading="accountActionId === `${currentAccount?.id}:validate`"
                  @click="validateAccount(currentAccount)"
                >
                  <template #icon><ExperimentOutlined /></template>
                  检测模型
                </a-button>
              </div>

              <div v-if="selectedProviderAccounts.length" class="account-grid">
                <article
                  v-for="account in selectedProviderAccounts"
                  :key="account.id"
                  class="account-card"
                  :class="{ current: account.current }"
                >
                  <div class="account-card-head">
                    <div class="account-name">
                      <strong>{{ account.displayName }}</strong>
                      <small>{{ accountSourceLabel(account) }}</small>
                    </div>
                    <a-space :size="4" wrap>
                      <a-tag :color="accountPlanColor(account.planType)">{{ accountPlanLabel(account.planType) }}</a-tag>
                      <a-tag v-if="account.current" color="success">当前</a-tag>
                      <a-tag v-else-if="account.authorized" color="blue">已授权</a-tag>
                      <a-tag v-else color="warning">待登录</a-tag>
                    </a-space>
                  </div>

                  <div class="account-meta">
                    <span>{{ account.availableModelCount || 0 }} 个模型可用</span>
                    <span>{{ accountValidationSummary(account) }}</span>
                    <span v-if="account.lastValidatedAt">最近验证 {{ new Date(account.lastValidatedAt).toLocaleString() }}</span>
                    <div v-if="accountValidationFailures(account).length" class="account-validation-failures">
                      <strong>模型不可用原因</strong>
                      <span
                        v-for="failure in accountValidationFailures(account)"
                        :key="failure.modelId"
                      >
                        {{ failure.modelId }}：{{ failure.message }}
                      </span>
                    </div>
                  </div>

                  <div class="account-actions">
                    <a-button
                      size="small"
                      type="primary"
                      ghost
                      :disabled="account.current"
                      :loading="accountActionId === account.id"
                      @click="selectAccount(account)"
                    >
                      <template #icon><UserSwitchOutlined /></template>
                      设为当前
                    </a-button>
                    <a-button
                      size="small"
                      :loading="accountActionId === `${account.id}:validate`"
                      @click="validateAccount(account)"
                    >
                      <template #icon><ExperimentOutlined /></template>
                      {{ account.current ? '检测模型' : '切换并检测' }}
                    </a-button>
                    <a-button size="small" @click="openAccountEditor(account)">
                      <template #icon><EditOutlined /></template>
                      编辑
                    </a-button>
                  </div>
                </article>
              </div>
              <div
                v-else
                class="empty-account-state"
                :class="`tone-${selectedProviderAccountStatus?.tone || 'neutral'}`"
              >
                <div class="empty-account-icon">
                  <LinkOutlined v-if="selectedProvider?.loginKind !== 'direct'" />
                  <LoginOutlined v-else />
                </div>
                <div class="empty-account-copy">
                  <strong>{{ selectedProviderAccountStatus?.title || '暂无已授权账号' }}</strong>
                  <p>{{ selectedProviderAccountStatus?.description || '登录完成并刷新后会显示在这里。' }}</p>
                  <small v-if="selectedProviderAccountStatus?.note">{{ selectedProviderAccountStatus.note }}</small>
                  <small v-if="selectedProviderAccountStatus?.action">{{ selectedProviderAccountStatus.action }}</small>
                </div>
              </div>
            </div>

            <div class="action-row">
              <a-button type="primary" size="large" :loading="actionLoading" @click="handleStartLogin">
                <template #icon><LoginOutlined /></template>
                {{ selectedProvider.loginKind === 'direct' ? '打开授权浏览器' : '打开网页登录授权' }}
              </a-button>
              <a-button
                v-if="selectedProvider.loginKind === 'direct'"
                size="large"
                :loading="remoteAuthSaving"
                @click="openRemoteAuthModal(selectedProvider)"
              >
                <template #icon><LinkOutlined /></template>
                远程/NAS 授权
              </a-button>
              <a-button
                v-if="selectedProvider.loginKind !== 'direct'"
                size="large"
                :loading="actionLoading"
                @click="handleStartLogin({ newAccount: true })"
              >
                <template #icon><PlusOutlined /></template>
                添加授权账号
              </a-button>
              <a-button size="large" :disabled="!selectedProviderDefaultModel" @click="copySelectedDefaultModel">
                <template #icon><CopyOutlined /></template>
                复制默认模型
              </a-button>
              <a-button
                v-if="selectedProvider.loginKind === 'direct'"
                size="large"
                :loading="actionLoading && actionKind === 'smoke'"
                :disabled="!isProviderReady(selectedProvider)"
                @click="runProviderSmoke"
              >
                <template #icon><RocketOutlined /></template>
                验证接入
              </a-button>
            </div>

            <div class="shortcut-row">
              <a-button type="link" @click="diagnosticsOpen = !diagnosticsOpen">
                <template #icon><ToolOutlined /></template>
                {{ diagnosticsOpen ? '收起高级修复' : '高级修复' }}
              </a-button>
              <div v-if="diagnosticsOpen" class="diagnostic-links">
                <a-button type="link" @click="openAdvanced('/tools/display')">
                  <template #icon><LinkOutlined /></template>
                  授权窗口
                </a-button>
                <a-button type="link" @click="openAdvanced('/tools/cache')">
                  <template #icon><SettingOutlined /></template>
                  登录数据
                </a-button>
                <a-button type="link" @click="openAdvanced('/settings/workers')">
                  <template #icon><ToolOutlined /></template>
                  通道设置
                </a-button>
              </div>
            </div>
          </div>
        </section>
      </div>

      <section class="panel models-panel">
        <div class="panel-heading compact">
          <div>
            <h2>可用模型</h2>
            <p>优先复制已验证可用的模型 ID，填到客户端后即可开始调用。</p>
          </div>
          <a-segmented
            v-model:value="modelScope"
            :options="[
              { label: '当前平台', value: 'selected' },
              { label: '全部模型', value: 'all' },
            ]"
          />
        </div>

        <div class="model-toolbar">
          <a-input-search v-model:value="modelSearch" placeholder="搜索模型或平台" allow-clear />
        </div>

        <a-table
          row-key="id"
          size="small"
          :columns="modelColumns"
          :data-source="filteredModels"
          :pagination="{ pageSize: 8, showSizeChanger: false }"
          :scroll="{ x: 860 }"
        >
          <template #bodyCell="{ column, record }">
            <template v-if="column.key === 'id'">
              <a-typography-text code copyable>{{ record.id }}</a-typography-text>
            </template>
            <template v-else-if="column.key === 'provider'">
              {{ providerForModel(record.id)?.name || record.owned_by || '上游模型' }}
            </template>
            <template v-else-if="column.key === 'capability'">
              <a-space :size="4" wrap>
                <a-tag
                  v-for="cap in modelCapabilityLabels(record)"
                  :key="cap.label"
                  :color="cap.color"
                >
                  {{ cap.label }}
                </a-tag>
              </a-space>
            </template>
            <template v-else-if="column.key === 'availability'">
              <a-space direction="vertical" :size="2">
                <a-tag :color="modelAvailabilityTag(record.id).color">
                  {{ modelAvailabilityTag(record.id).label }}
                </a-tag>
                <small v-if="modelAvailability(record.id).message" class="availability-message">
                  {{ modelAvailability(record.id).message }}
                </small>
              </a-space>
            </template>
            <template v-else-if="column.key === 'action'">
              <a-button size="small" @click="copyModel(record.id)">
                <template #icon><CopyOutlined /></template>
                复制
              </a-button>
            </template>
          </template>
        </a-table>
      </section>

      <section class="panel media-panel">
        <div class="panel-heading compact">
          <div>
            <h2>图片生成测试</h2>
            <p>选择任一已授权的图片模型做一次小图测试，确认 WebAI2API 媒体链路可以生成图片。</p>
          </div>
          <ExperimentOutlined />
        </div>

        <div class="media-test-grid">
          <div class="media-form">
            <label class="field-label" for="image-model-select">模型</label>
            <a-select
              id="image-model-select"
              v-model:value="imageModel"
              :options="imageModelOptions"
              class="full-width"
            />

            <label class="field-label" for="image-prompt-input">提示词</label>
            <a-textarea
              id="image-prompt-input"
              v-model:value="imagePrompt"
              :rows="4"
              placeholder="描述你要生成的图片"
            />

            <div class="action-row">
              <a-button type="primary" :loading="imageGenerating" @click="runImageSmokeTest">
                <template #icon><ExperimentOutlined /></template>
                生成测试图片
              </a-button>
              <a-button @click="copyText(imageRequestExample, '生图请求示例已复制')">
                <template #icon><CopyOutlined /></template>
                复制请求示例
              </a-button>
            </div>
            <a-alert
              v-if="imageError"
              type="error"
              show-icon
              :message="imageError"
            />
          </div>

          <div class="image-preview-card">
            <img v-if="imageResultUrl" :src="imageResultUrl" alt="图片生成测试结果" />
            <div v-else class="image-placeholder">
              <ExperimentOutlined />
              <span>生成后会在这里预览图片</span>
              <small>该测试使用当前授权账号，不需要额外配置。</small>
            </div>
          </div>
        </div>
      </section>

      <section class="panel config-panel">
        <div class="panel-heading compact">
          <div>
            <h2>接入客户端</h2>
            <p>按客户端类型复制配置，填入兼容 OpenAI 或 Anthropic API 的客户端即可调用。</p>
          </div>
          <ApiOutlined />
        </div>

        <div class="code-header">
          <div class="code-title-stack">
            <span>{{ clientConfigTitle }}</span>
            <small v-if="configProfile === 'cc-switch'">默认展示 cc-switch 的 Claude Provider 配置，复制后粘贴到对应 Provider 的 settings_config。</small>
          </div>
          <a-segmented
            v-model:value="configProfile"
            :options="clientConfigOptions"
            size="small"
          />
          <a-button size="small" @click="copyText(clientConfig, `${clientConfigTitle}已复制`)">
            <template #icon><CopyOutlined /></template>
            复制
          </a-button>
        </div>
        <pre class="code-block">{{ clientConfig }}</pre>
      </section>
    </a-spin>

    <a-modal
      v-model:open="accountEditOpen"
      title="编辑账号信息"
      :confirm-loading="accountEditSaving"
      ok-text="保存"
      cancel-text="取消"
      @ok="saveAccountEdit"
    >
      <a-space direction="vertical" style="width: 100%;">
        <div class="field-block">
          <label>账号显示名</label>
          <a-input v-model:value="accountEditForm.displayName" placeholder="例如 GPT Plus 主账号" />
        </div>
        <div class="field-block">
          <label>账号类型</label>
          <a-select
            v-model:value="accountEditForm.planType"
            :options="[
              { label: '未知权益', value: 'unknown' },
              { label: 'Free', value: 'free' },
              { label: 'Plus', value: 'plus' },
              { label: 'Pro', value: 'pro' },
              { label: 'Team', value: 'team' },
            ]"
          />
        </div>
        <div class="field-block">
          <label>备注</label>
          <a-textarea v-model:value="accountEditForm.note" :rows="3" placeholder="仅保存在本机，不保存凭证正文" />
        </div>
      </a-space>
    </a-modal>

    <a-modal
      v-model:open="remoteAuthOpen"
      title="远程/NAS 授权"
      :confirm-loading="remoteAuthSaving"
      ok-text="导入授权 URL"
      cancel-text="取消"
      @ok="submitRemoteAuthUrl"
    >
      <a-space direction="vertical" style="width: 100%;">
        <a-alert
          type="info"
          show-icon
          message="适合 Gateway 部署在 NAS 或服务器的场景"
          description="只有 URL 里带 cookie、bearer 或 session token 时才能直接导入。普通登录页地址不包含登录态，系统会提示你改用远程 CDP 捕获。"
        />
        <div class="field-block">
          <label>当前平台</label>
          <a-typography-text strong>{{ remoteAuthProvider?.name || '未选择平台' }}</a-typography-text>
        </div>
        <div class="field-block">
          <label>授权 URL</label>
          <a-textarea
            v-model:value="remoteAuthUrl"
            :rows="4"
            placeholder="粘贴完成登录后的回调 URL。不要把这个 URL 发给陌生人。"
          />
        </div>
      </a-space>
    </a-modal>

    <a-modal v-model:open="progressVisible" :title="progressTitle" :footer="null" width="620px">
      <a-alert
        v-if="actionError"
        type="error"
        show-icon
        :message="actionError"
        style="margin-bottom: 12px;"
      />
      <a-steps
        size="small"
        :current="progressCurrent"
        :status="progressStatus"
        :items="progressStepItems"
      />
      <div class="log-box">
        <div v-for="line in actionLogs" :key="line">{{ line }}</div>
        <div v-if="!actionLogs.length">等待操作开始。</div>
      </div>
      <div class="modal-actions">
        <a-button
          :loading="actionLoading"
          @click="actionKind === 'webai2api' ? finishWebAI2APILogin() : retryDirectAuthCapture(selectedProvider)"
        >
          <template #icon><ReloadOutlined /></template>
          {{ actionKind === 'webai2api' ? '恢复 API 并刷新' : '重新检测登录态' }}
        </a-button>
        <a-button
          type="primary"
          :loading="actionLoading && actionKind === 'webai2api'"
          @click="actionKind === 'webai2api' ? finishWebAI2APILogin({ close: true }) : (progressVisible = false)"
        >
          {{ actionKind === 'webai2api' ? '恢复并完成' : '完成' }}
        </a-button>
      </div>
    </a-modal>
  </div>
</template>

<style scoped>
.onboarding-shell {
  --brand: #0d766f;
  --brand-strong: #0b4f4a;
  --accent: #2563eb;
  --amber: #b45309;
  --ink: #0f172a;
  --muted: #64748b;
  --line: #d7e0e7;
  --soft: #f4f8f7;
  --surface: #ffffff;
  --warn-soft: #fff8eb;
  color: var(--ink);
}

.hero-panel,
.panel {
  background: var(--surface);
  border: 1px solid var(--line);
  border-radius: 8px;
  box-shadow: 0 18px 42px rgba(15, 23, 42, 0.055);
}

.hero-panel {
  background:
    linear-gradient(90deg, rgba(8, 49, 45, 0.98), rgba(15, 35, 55, 0.98)),
    #0b302e;
  border: 1px solid rgba(15, 118, 110, 0.42);
  display: grid;
  gap: 26px;
  grid-template-columns: minmax(0, 1.1fr) minmax(360px, 0.9fr);
  margin-bottom: 20px;
  overflow: hidden;
  padding: 30px 34px;
  position: relative;
}

.hero-panel::after {
  background-image:
    linear-gradient(rgba(255, 255, 255, 0.055) 1px, transparent 1px),
    linear-gradient(90deg, rgba(255, 255, 255, 0.055) 1px, transparent 1px);
  background-size: 34px 34px;
  content: "";
  inset: 0;
  opacity: 0.42;
  pointer-events: none;
  position: absolute;
}

.hero-copy,
.hero-status {
  position: relative;
  z-index: 1;
}

.hero-copy h1 {
  color: #f8fafc;
  font-size: 34px;
  font-weight: 700;
  letter-spacing: 0;
  line-height: 1.14;
  margin: 12px 0;
  max-width: 760px;
  text-wrap: pretty;
}

.hero-copy p,
.panel-heading p {
  color: var(--muted);
  line-height: 1.65;
  margin: 0;
  text-wrap: pretty;
}

.hero-copy p {
  color: #bed8d5;
  font-size: 15px;
  max-width: 740px;
}

.hero-copy :deep(.ant-tag) {
  background: rgba(236, 253, 245, 0.12);
  border-color: rgba(167, 243, 208, 0.44);
  color: #b7f4dd;
  font-weight: 600;
  margin: 0;
}

.hero-actions,
.action-row,
.shortcut-row,
.modal-actions {
  display: flex;
  flex-wrap: wrap;
  gap: 10px;
}

.hero-actions {
  margin-top: 22px;
}

.hero-actions :deep(.ant-btn) {
  border-radius: 7px;
  height: 40px;
}

.hero-actions :deep(.ant-btn-primary) {
  background: #14b8a6;
  border-color: #14b8a6;
  box-shadow: 0 12px 24px rgba(20, 184, 166, 0.24);
}

.hero-status {
  align-self: center;
  background: rgba(255, 255, 255, 0.08);
  border: 1px solid rgba(226, 232, 240, 0.16);
  border-radius: 8px;
  padding: 18px;
}

.hero-status :deep(.ant-steps-item-title) {
  color: #e2e8f0 !important;
  font-size: 13px;
}

.hero-status :deep(.ant-steps-item-tail::after) {
  background-color: rgba(125, 211, 252, 0.8) !important;
}

.hero-status :deep(.ant-steps-icon) {
  color: #0d766f !important;
}

.stat-grid {
  display: grid;
  gap: 8px;
  grid-template-columns: repeat(3, 1fr);
  margin-top: 18px;
}

.stat-item {
  background: rgba(255, 255, 255, 0.92);
  border: 1px solid rgba(226, 232, 240, 0.8);
  border-radius: 7px;
  padding: 12px 14px;
}

.stat-item span,
.detail-row span,
.field-block label {
  color: var(--muted);
  display: block;
  font-size: 12px;
}

.stat-item strong {
  display: block;
  color: #061a18;
  font-size: 26px;
  line-height: 1;
  margin-top: 4px;
}

.workspace-grid {
  display: grid;
  gap: 18px;
  grid-template-columns: minmax(320px, 0.78fr) minmax(0, 1.22fr);
  margin-bottom: 18px;
}

.panel {
  padding: 20px;
}

.panel-heading {
  align-items: flex-start;
  display: flex;
  gap: 16px;
  justify-content: space-between;
  margin-bottom: 16px;
}

.panel-heading.compact {
  align-items: center;
}

.panel-heading h2 {
  color: #0f172a;
  font-size: 18px;
  letter-spacing: 0;
  line-height: 1.2;
  margin: 0 0 4px;
}

.panel-heading > .anticon {
  color: var(--brand);
  font-size: 22px;
  margin-top: 4px;
}

.ready-icon {
  color: var(--brand) !important;
}

.pending-icon {
  color: #b45309 !important;
}

.provider-list {
  display: grid;
  gap: 9px;
}

.provider-item {
  align-items: center;
  background: #fbfdfc;
  border: 1px solid var(--line);
  border-radius: 7px;
  cursor: pointer;
  display: flex;
  gap: 12px;
  justify-content: space-between;
  min-height: 64px;
  padding: 12px 14px;
  text-align: left;
  transition: border-color 160ms ease, background 160ms ease, box-shadow 160ms ease, transform 160ms ease;
  width: 100%;
}

.provider-item:hover,
.provider-item.active {
  background: #effbf8;
  border-color: #0f9488;
  box-shadow: 0 14px 32px rgba(15, 118, 110, 0.12);
  transform: translateY(-1px);
}

.provider-main {
  display: grid;
  gap: 3px;
  min-width: 0;
}

.provider-main strong,
.provider-main small {
  overflow: hidden;
  text-overflow: ellipsis;
  white-space: nowrap;
}

.provider-main small {
  color: var(--muted);
  font-size: 12px;
}

.provider-tags {
  flex-shrink: 0;
}

.provider-detail {
  display: grid;
  gap: 14px;
}

.detail-row {
  background: #f8fbfa;
  border: 1px solid #dfe9e6;
  border-radius: 7px;
  display: grid;
  gap: 2px;
  padding: 10px 12px;
}

.detail-row strong {
  display: block;
  margin-top: 2px;
}

.capability-row {
  display: flex;
  flex-wrap: wrap;
  gap: 6px;
}

.account-section {
  border-top: 1px solid #e7eef5;
  display: grid;
  gap: 12px;
  padding-top: 12px;
}

.section-title {
  align-items: center;
  display: flex;
  gap: 12px;
  justify-content: space-between;
}

.validate-current-button {
  align-items: center;
  display: inline-flex;
  justify-content: center;
  min-width: 86px;
  white-space: nowrap;
}

.validate-current-button :deep(.ant-btn-icon) {
  align-items: center;
  display: inline-flex;
  line-height: 0;
}

.section-title strong,
.section-title span {
  display: block;
}

.section-title span {
  color: var(--muted);
  font-size: 12px;
  margin-top: 2px;
}

.account-grid {
  display: grid;
  gap: 10px;
  grid-template-columns: repeat(auto-fit, minmax(240px, 1fr));
}

.account-card {
  background: #ffffff;
  border: 1px solid var(--line);
  border-radius: 7px;
  display: grid;
  gap: 12px;
  padding: 12px;
  transition: border-color 160ms ease, box-shadow 160ms ease, transform 160ms ease;
}

.account-card:hover,
.account-card.current {
  border-color: #0f9488;
  box-shadow: 0 12px 30px rgba(17, 24, 39, 0.08);
  transform: translateY(-1px);
}

.account-card.current {
  background: #f4fffb;
}

.account-card-head {
  align-items: flex-start;
  display: flex;
  gap: 10px;
  justify-content: space-between;
}

.account-name {
  display: grid;
  gap: 3px;
  min-width: 0;
}

.account-name strong,
.account-name small {
  overflow: hidden;
  text-overflow: ellipsis;
  white-space: nowrap;
}

.account-name small,
.account-meta {
  color: var(--muted);
  font-size: 12px;
}

.account-meta {
  display: grid;
  gap: 4px;
}

.account-validation-failures {
  background: #fff7f7;
  border: 1px solid #ffd6d6;
  border-radius: 6px;
  color: #9f1239;
  display: grid;
  gap: 4px;
  margin-top: 4px;
  padding: 8px;
  overflow-wrap: anywhere;
}

.account-validation-failures strong {
  color: #7f1d1d;
}

.account-actions {
  display: flex;
  flex-wrap: wrap;
  gap: 8px;
}

.empty-account-state {
  align-items: flex-start;
  background: #f8fbfa;
  border: 1px solid #d7ebe7;
  border-radius: 7px;
  display: flex;
  gap: 12px;
  padding: 14px;
}

.empty-account-state.tone-web {
  background: var(--warn-soft);
  border-color: #fed7aa;
}

.empty-account-state.tone-direct {
  background: #f0f9ff;
  border-color: #bae6fd;
}

.empty-account-icon {
  align-items: center;
  background: #ffffff;
  border: 1px solid var(--line);
  border-radius: 7px;
  color: var(--brand);
  display: inline-flex;
  flex: 0 0 40px;
  font-size: 18px;
  height: 40px;
  justify-content: center;
  width: 40px;
}

.empty-account-copy {
  display: grid;
  gap: 5px;
  min-width: 0;
}

.empty-account-copy strong {
  color: var(--ink);
}

.empty-account-copy p,
.empty-account-copy small {
  color: #475569;
  line-height: 1.55;
  margin: 0;
  text-wrap: pretty;
}

.field-block {
  display: grid;
  gap: 8px;
}

.shortcut-row {
  border-top: 1px solid #e7eef5;
  padding-top: 10px;
}

.models-panel,
.media-panel,
.config-panel {
  margin-bottom: 18px;
}

.models-panel :deep(.ant-table) {
  background: transparent;
  border-radius: 7px;
}

.models-panel :deep(.ant-table-thead > tr > th) {
  background: #f5f8fa;
  color: #334155;
  font-size: 12px;
  font-weight: 700;
}

.models-panel :deep(.ant-table-tbody > tr > td) {
  border-color: #edf2f5;
}

.models-panel :deep(.ant-table-tbody > tr:hover > td) {
  background: #f8fbfa;
}

.models-panel :deep(.ant-typography code) {
  white-space: nowrap;
}

.model-toolbar {
  margin-bottom: 12px;
  max-width: 420px;
}

.media-test-grid {
  display: grid;
  gap: 18px;
  grid-template-columns: minmax(0, 1.1fr) minmax(260px, 0.9fr);
}

.media-form {
  display: grid;
  gap: 10px;
  min-width: 0;
}

.field-label {
  color: #344054;
  font-size: 13px;
  font-weight: 600;
}

.full-width {
  width: 100%;
}

.image-preview-card {
  align-items: center;
  background:
    repeating-linear-gradient(135deg, rgba(15, 118, 110, 0.055) 0 8px, transparent 8px 16px),
    #f8fbfa;
  border: 1px solid #d7ebe7;
  border-radius: 7px;
  display: flex;
  justify-content: center;
  min-height: 260px;
  overflow: hidden;
  padding: 14px;
}

.image-preview-card img {
  border-radius: 8px;
  display: block;
  max-height: 360px;
  max-width: 100%;
  object-fit: contain;
}

.image-placeholder {
  align-items: center;
  color: var(--muted);
  display: grid;
  gap: 8px;
  justify-items: center;
  line-height: 1.6;
  max-width: 320px;
  text-align: center;
}

.image-placeholder .anticon {
  color: var(--brand);
  font-size: 26px;
}

.availability-message {
  color: var(--muted);
  display: block;
  max-width: 260px;
  overflow-wrap: anywhere;
}

.code-header {
  align-items: center;
  display: flex;
  gap: 10px;
  justify-content: space-between;
  margin-bottom: 8px;
  flex-wrap: wrap;
}

.code-title-stack {
  display: grid;
  gap: 2px;
  min-width: 220px;
}

.code-title-stack small {
  color: var(--muted);
  font-size: 12px;
}

.code-block,
.log-box {
  background: #0b1220;
  border: 1px solid #1f2a3d;
  border-radius: 7px;
  color: #dbeafe;
  font-size: 12px;
  line-height: 1.6;
  margin: 0;
  overflow-x: auto;
  padding: 14px;
  white-space: pre-wrap;
  word-break: break-word;
}

.log-box {
  margin-top: 16px;
  max-height: 220px;
  overflow-y: auto;
}

.modal-actions {
  justify-content: flex-end;
  margin-top: 16px;
}

@media (max-width: 980px) {
  .hero-panel,
  .workspace-grid,
  .media-test-grid {
    grid-template-columns: 1fr;
  }
}

@media (max-width: 640px) {
  .hero-panel,
  .panel {
    padding: 16px;
  }

  .hero-panel {
    gap: 14px;
    margin-bottom: 14px;
  }

  .hero-copy h1 {
    font-size: 22px;
    line-height: 1.18;
    margin: 9px 0;
  }

  .hero-copy p {
    font-size: 13px;
    line-height: 1.55;
  }

  .hero-copy :deep(.ant-tag) {
    font-size: 12px;
    line-height: 1.2;
    padding: 3px 7px;
  }

  .hero-actions {
    gap: 8px;
    margin-top: 14px;
  }

  .hero-actions :deep(.ant-btn) {
    font-size: 13px;
    height: 36px;
    padding-inline: 12px;
  }

  .hero-status {
    align-self: stretch;
    padding: 12px;
  }

  .hero-status :deep(.ant-steps) {
    display: none;
  }

  .stat-grid {
    gap: 6px;
    grid-template-columns: repeat(3, minmax(0, 1fr));
    margin-top: 0;
  }

  .stat-item {
    padding: 9px 8px;
  }

  .stat-item span {
    font-size: 11px;
  }

  .stat-item strong {
    font-size: 21px;
    margin-top: 3px;
  }

  .model-toolbar {
    max-width: none;
  }

  .section-title,
  .empty-account-state {
    align-items: flex-start;
    flex-direction: column;
  }
}
</style>
