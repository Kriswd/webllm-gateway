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
const selectedProviderId = ref('');
const selectedWorkerName = ref('');
const modelSearch = ref('');
const modelScope = ref('selected');
const tokenVisible = ref(false);
const cdpUrl = ref('http://127.0.0.1:9222');
const workerOptions = ref([]);
const accountEditOpen = ref(false);
const accountEditSaving = ref(false);
const accountEditForm = ref({
  id: '',
  displayName: '',
  planType: 'unknown',
  note: '',
});

const localExampleBaseUrl = 'http://127.0.0.1:8610/v1';

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
});

const gatewayBaseUrl = computed(() => `${window.location.origin}${onboarding.value.gateway?.baseUrl || '/v1'}`);
const gatewayToken = computed(() => onboarding.value.gateway?.apiKey || '');
const maskedToken = computed(() => {
  if (!gatewayToken.value) return '未配置';
  if (tokenVisible.value) return gatewayToken.value;
  return `${gatewayToken.value.slice(0, 7)}...${gatewayToken.value.slice(-6)}`;
});

const providerPriority = ['deepseek-web', 'qwen', 'qwen-cn', 'chatgpt', 'gemini', 'lmarena', 'doubao', 'zai', 'sora'];
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

const selectedProviderAccounts = computed(() => {
  return Array.isArray(selectedProvider.value?.accounts) ? selectedProvider.value.accounts : [];
});

const currentAccount = computed(() => {
  return selectedProviderAccounts.value.find((account) => account.id === selectedProvider.value?.currentAccountId)
    || selectedProviderAccounts.value.find((account) => account.current)
    || selectedProviderAccounts.value[0]
    || null;
});

const selectedProviderWorkers = computed(() => {
  const provider = selectedProvider.value;
  if (!provider) return [];
  const adapters = new Set(provider.adapters || []);
  const matched = workerOptions.value.filter((worker) => {
    if (adapters.has(worker.type)) return true;
    return (worker.mergeTypes || []).some((type) => adapters.has(type));
  });
  return matched.length ? matched : workerOptions.value;
});

const selectedProviderDefaultModel = computed(() => providerAvailableModels(selectedProvider.value)[0] || '');

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

const clientConfig = computed(() => [
  `base_url = ${gatewayBaseUrl.value}`,
  `api_key = ${gatewayToken.value || '<网关令牌>'}`,
  `model = ${selectedProvider.value ? (selectedProviderDefaultModel.value || '<当前平台没有验证可用模型>') : (onboarding.value.gateway?.defaultModel || '<模型 ID>')}`,
  '',
  '# KrisAI / OpenClaw / Hermes / Claude Code 都可以使用这组 OpenAI 兼容配置',
  '# 工具调用由 WebAI Gateway 转换为网页模型可理解的 prompt 协议',
].join('\n'));

const modelColumns = [
  { title: '模型 ID', dataIndex: 'id', key: 'id' },
  { title: '来源', key: 'provider', width: 180 },
  { title: '能力', key: 'capability', width: 180 },
  { title: '账号可用性', key: 'availability', width: 180 },
  { title: '操作', key: 'action', width: 110 },
];

const stepItems = computed(() => [
  { title: '选择平台', status: selectedProvider.value ? 'finish' : 'process' },
  { title: '网页登录', status: isProviderReady(selectedProvider.value) ? 'finish' : actionLoading.value ? 'process' : 'wait' },
  { title: '使用模型', status: filteredModels.value.length ? 'finish' : 'wait' },
]);

const progressTitle = computed(() => (actionKind.value === 'smoke' ? 'Provider 自检' : '授权进度'));
const progressStepItems = computed(() => {
  if (actionKind.value === 'smoke') {
    return [{ title: '发起' }, { title: '协议闭环' }, { title: '完成' }];
  }
  return [
    { title: '启动' },
    { title: actionKind.value === 'direct' ? '检测登录' : '登录模式' },
    { title: '完成' },
  ];
});
const progressCurrent = computed(() => (actionLoading.value ? 1 : actionError.value ? 0 : 2));
const progressStatus = computed(() => (actionError.value ? 'error' : 'process'));

function isProviderReady(provider) {
  if (!provider) return false;
  if (provider.loginKind === 'direct') return Boolean(provider.credential?.authorized);
  if (provider.webAI2APIAuth?.checked) return Boolean(provider.credential?.authorized);
  return true;
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
  if (account?.source === 'direct-profile') return '网关直连';
  return `${account?.instanceName || 'WebAI2API'} / ${account?.workerName || 'worker'}`;
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

  if (type === 'video' || modelId.includes('sora') || modelId.includes('video')) {
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
    onboarding.value = await res.json();
    if (!selectedProviderId.value && providers.value.length) {
      selectedProviderId.value = providers.value[0].id;
    }
    await loadWorkers();
  } catch (error) {
    loadError.value = error.message || String(error);
  } finally {
    loading.value = false;
  }
}

async function loadWorkers() {
  try {
    const res = await fetch('/admin/config/instances', { headers: settingsStore.getHeaders() });
    if (!res.ok) return;
    const instances = await res.json();
    const workers = [];
    for (const instance of instances || []) {
      for (const worker of instance.workers || []) {
        workers.push({
          label: `${worker.name} · ${worker.type}`,
          value: worker.name,
          name: worker.name,
          type: worker.type,
          mergeTypes: worker.mergeTypes || [],
          instance: instance.name,
        });
      }
    }
    workerOptions.value = workers;
  } catch {
    workerOptions.value = [];
  }
}

function resetActionState(kind) {
  actionKind.value = kind;
  actionError.value = '';
  actionLogs.value = [];
  progressVisible.value = true;
}

async function handleStartLogin() {
  const provider = selectedProvider.value;
  if (!provider) return;
  if (provider.loginKind === 'direct') {
    await startDirectAuth(provider);
    return;
  }
  Modal.confirm({
    title: '进入 WebAI2API 登录模式？',
    content: '这会让 WebAI2API 以网页登录模式重启。重启后请在打开的浏览器或虚拟显示器里完成登录，再回到这里刷新模型。',
    okText: '进入登录模式',
    cancelText: '取消',
    async onOk() {
      await startWebAI2APILogin(provider);
    },
  });
}

async function startDirectAuth(provider) {
  resetActionState('direct');
  actionLoading.value = true;
  try {
    appendLog(`正在为 ${provider.name} 启动授权浏览器`);
    const browserRes = await fetch('/api/admin/web-auth/browser/start', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ provider: provider.id, cdpUrl: cdpUrl.value }),
    });
    const browserData = await browserRes.json();
    if (!browserRes.ok) throw new Error(browserData.detail || `HTTP ${browserRes.status}`);
    appendLog(browserData.message || '授权浏览器已启动');
    if (!browserData.started && browserData.loginUrl) {
      window.open(browserData.loginUrl, '_blank', 'noopener,noreferrer');
    }

    appendLog('正在检测网页登录状态');
    const jobRes = await fetch('/api/admin/web-auth/jobs', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ provider: provider.id, cdpUrl: cdpUrl.value }),
    });
    const job = await jobRes.json();
    if (!jobRes.ok) throw new Error(job.detail || `HTTP ${jobRes.status}`);
    await pollAuthJob(job.id);
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

async function startWebAI2APILogin(provider) {
  resetActionState('webai2api');
  actionLoading.value = true;
  try {
    const workerName = selectedWorkerName.value || undefined;
    appendLog(workerName ? `正在以登录模式重启 Worker：${workerName}` : '正在以登录模式重启 WebAI2API');
    const res = await fetch('/admin/restart', {
      method: 'POST',
      headers: {
        ...settingsStore.getHeaders(),
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({ loginMode: true, workerName }),
    });
    const data = await res.json().catch(() => ({}));
    if (!res.ok || data.success === false) {
      throw new Error(data.message || data.error?.message || `HTTP ${res.status}`);
    }
    appendLog(data.message || 'WebAI2API 已进入登录模式');
    appendLog(`请完成 ${provider.name} 网页登录，然后点击“刷新模型”确认可用模型`);
    message.success('已进入网页登录模式');
    window.open('/tools/display', '_blank', 'noopener,noreferrer');
  } catch (error) {
    actionError.value = error.message || String(error);
    appendLog(`登录模式启动失败：${actionError.value}`);
    message.error(actionError.value);
  } finally {
    actionLoading.value = false;
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
    appendLog(`正在自检 ${provider.name}：模型、OpenAI 工具调用、Anthropic 工具闭环`);
    const res = await fetch(`/api/admin/provider-smoke/${provider.id}`, { method: 'POST' });
    const data = await res.json().catch(() => ({}));
    if (!res.ok) throw new Error(data.detail || `HTTP ${res.status}`);
    for (const item of data.results || []) {
      const state = item.ok ? '通过' : '失败';
      const detail = smokeResultDetail(item);
      appendLog(`${state} · ${smokeResultLabel(item.id)}${detail ? ` · ${detail}` : ''}`);
    }
    if (!data.ok) {
      actionError.value = `${provider.name} 自检未全部通过：${data.passed || 0}/${data.total || 0}`;
      message.error(actionError.value);
      return;
    }
    message.success(`${provider.name} 自检通过`);
  } catch (error) {
    actionError.value = error.message || String(error);
    appendLog(`自检失败：${actionError.value}`);
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
      }),
    });
    const data = await res.json().catch(() => ({}));
    if (!res.ok) throw new Error(data.detail || data.message || `HTTP ${res.status}`);
    const values = Object.values(data.validation || {});
    const okCount = values.filter((item) => item?.status === 'available').length;
    message.success(`模型验证完成：${okCount}/${values.length} 可用`);
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

function rotateToken() {
  Modal.confirm({
    title: '重新生成网关令牌？',
    content: '旧令牌会立即失效，需要同步更新 KrisAI、OpenClaw、Hermes 或 Claude Code 里的 API Key。',
    okText: '重新生成',
    cancelText: '取消',
    async onOk() {
      const res = await fetch('/api/admin/token/rotate', { method: 'POST' });
      const data = await res.json();
      if (!res.ok) throw new Error(data.detail || `HTTP ${res.status}`);
      message.success('网关令牌已更新');
      await loadOnboarding();
    },
  });
}

watch(selectedProviderId, () => {
  selectedWorkerName.value = selectedProviderWorkers.value[0]?.value || '';
  modelScope.value = 'selected';
});

onMounted(loadOnboarding);
</script>

<template>
  <div class="onboarding-shell">
    <section class="hero-panel">
      <div class="hero-copy">
        <a-tag color="blue">WebAI Gateway</a-tag>
        <h1>网页登录向导</h1>
        <p>选择网页模型平台，按提示完成浏览器登录，然后直接复制模型 ID 给 KrisAI 使用。</p>
        <div class="hero-actions">
          <a-button type="primary" size="large" :loading="actionLoading" @click="handleStartLogin">
            <template #icon><LoginOutlined /></template>
            {{ selectedProvider?.loginKind === 'direct' ? '打开授权浏览器' : '进入 WebAI2API 登录模式' }}
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
            <span>平台</span>
            <strong>{{ onboarding.summary.providers }}</strong>
          </div>
          <div class="stat-item">
            <span>模型</span>
            <strong>{{ onboarding.summary.models }}</strong>
          </div>
          <div class="stat-item">
            <span>已授权</span>
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
              <h2>选择平台</h2>
              <p>优先选择你已经有网页登录账号的平台。</p>
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
                <small>{{ providerAccountCount(provider) }} 个账号 · {{ providerModelCount(provider) }} 个可用模型 ID</small>
              </span>
              <span class="provider-tags">
                <a-tag v-if="isProviderAuthorized(provider)" color="success">已授权</a-tag>
                <a-tag v-else-if="provider.loginKind === 'direct'" color="warning">未授权</a-tag>
                <a-tag v-else color="blue">WebAI2API</a-tag>
              </span>
            </button>
          </div>
        </section>

        <section class="panel action-panel">
          <div class="panel-heading">
            <div>
              <h2>{{ selectedProvider?.name || '未选择平台' }}</h2>
              <p>{{ selectedProvider?.description || '请选择一个网页模型平台。' }}</p>
            </div>
            <CheckCircleOutlined v-if="isProviderReady(selectedProvider)" class="ready-icon" />
            <ClockCircleOutlined v-else class="pending-icon" />
          </div>

          <div v-if="selectedProvider" class="provider-detail">
            <div class="detail-row">
              <span>登录方式</span>
              <strong>{{ selectedProvider.loginKind === 'direct' ? '网关自动捕获' : 'WebAI2API 登录模式' }}</strong>
            </div>
            <div class="detail-row">
              <span>网页登录页</span>
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
                  <span>{{ currentAccount ? `当前使用：${currentAccount.displayName}` : '当前平台还没有检测到可用账号' }}</span>
                </div>
                <a-button
                  size="small"
                  :disabled="!currentAccount"
                  :loading="accountActionId === `${currentAccount?.id}:validate`"
                  @click="validateAccount(currentAccount)"
                >
                  <template #icon><ExperimentOutlined /></template>
                  验证模型
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
                    <span>{{ account.availableModelCount || 0 }} 个模型未失败</span>
                    <span>{{ accountValidationSummary(account) }}</span>
                    <span v-if="account.lastValidatedAt">最近验证 {{ new Date(account.lastValidatedAt).toLocaleString() }}</span>
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
                      验证
                    </a-button>
                    <a-button size="small" @click="openAccountEditor(account)">
                      <template #icon><EditOutlined /></template>
                      编辑
                    </a-button>
                  </div>
                </article>
              </div>
              <a-empty v-else image="simple" description="暂无已授权账号。登录完成并刷新后会显示在这里。" />
            </div>

            <a-alert
              v-if="selectedProvider.availabilityMessage"
              type="warning"
              show-icon
              :message="selectedProvider.availabilityMessage"
            />

            <template v-if="selectedProvider.loginKind === 'direct'">
              <div class="field-block">
                <label>授权浏览器调试地址</label>
                <a-input v-model:value="cdpUrl" placeholder="http://127.0.0.1:9222" />
              </div>
              <a-alert
                type="info"
                show-icon
                message="点击后会打开独立浏览器窗口。登录完成后，网关会自动检测并保存本机登录态。"
              />
            </template>

            <template v-else>
              <div class="field-block">
                <label>登录 Worker</label>
                <a-select
                  v-model:value="selectedWorkerName"
                  allow-clear
                  placeholder="自动选择匹配 Worker"
                  :options="selectedProviderWorkers"
                />
              </div>
              <a-alert
                type="info"
                show-icon
                message="WebAI2API 负责这些平台的浏览器缓存。进入登录模式后，在虚拟显示器或弹出的浏览器里登录。"
              />
            </template>

            <div class="action-row">
              <a-button type="primary" size="large" :loading="actionLoading" @click="handleStartLogin">
                <template #icon><LoginOutlined /></template>
                {{ selectedProvider.loginKind === 'direct' ? '打开授权浏览器' : '进入登录模式' }}
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
                运行自检
              </a-button>
            </div>

            <div class="shortcut-row">
              <a-button type="link" @click="openAdvanced('/tools/display')">
                <template #icon><LinkOutlined /></template>
                虚拟显示器
              </a-button>
              <a-button type="link" @click="openAdvanced('/tools/cache')">
                <template #icon><SettingOutlined /></template>
                缓存与重启
              </a-button>
              <a-button type="link" @click="openAdvanced('/settings/workers')">
                <template #icon><ToolOutlined /></template>
                工作池
              </a-button>
            </div>
          </div>
        </section>
      </div>

      <section class="panel models-panel">
        <div class="panel-heading compact">
          <div>
            <h2>可用模型</h2>
            <p>当前账号验证失败的模型会标红；复制已验证或待验证模型 ID 后即可接入客户端。</p>
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

      <section class="panel config-panel">
        <div class="panel-heading compact">
          <div>
            <h2>接入 KrisAI</h2>
            <p>把下面三项填到 KrisAI、OpenClaw、Hermes 或 Claude Code 的 OpenAI 兼容配置里。</p>
          </div>
          <ApiOutlined />
        </div>

        <div class="config-grid">
          <div class="config-item">
            <span>OpenAI 地址</span>
            <a-typography-text copyable>{{ gatewayBaseUrl }}</a-typography-text>
          </div>
          <div class="config-item">
            <span>本机地址</span>
            <a-typography-text copyable>{{ localExampleBaseUrl }}</a-typography-text>
          </div>
          <div class="config-item token-item">
            <span>API Key</span>
            <strong>{{ maskedToken }}</strong>
            <a-space>
              <a-button size="small" @click="tokenVisible = !tokenVisible">{{ tokenVisible ? '隐藏' : '显示' }}</a-button>
              <a-button size="small" @click="copyText(gatewayToken, 'API Key 已复制')">
                <template #icon><CopyOutlined /></template>
                复制
              </a-button>
              <a-button size="small" danger @click="rotateToken">重新生成</a-button>
            </a-space>
          </div>
        </div>

        <div class="code-header">
          <span>客户端配置</span>
          <a-button size="small" @click="copyText(clientConfig, '客户端配置已复制')">
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
          <a-textarea v-model:value="accountEditForm.note" :rows="3" placeholder="仅保存在 Gateway 本地 metadata，不保存凭证正文" />
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
        <a-button @click="loadOnboarding">
          <template #icon><ReloadOutlined /></template>
          刷新模型
        </a-button>
        <a-button type="primary" @click="progressVisible = false">完成</a-button>
      </div>
    </a-modal>
  </div>
</template>

<style scoped>
.onboarding-shell {
  color: #1f2937;
}

.hero-panel,
.panel {
  background: #ffffff;
  border: 1px solid #e5e7eb;
  border-radius: 8px;
}

.hero-panel {
  display: grid;
  gap: 24px;
  grid-template-columns: minmax(0, 1.25fr) minmax(320px, 0.75fr);
  margin-bottom: 16px;
  padding: 28px;
}

.hero-copy h1 {
  font-size: 32px;
  line-height: 1.18;
  margin: 12px 0 10px;
}

.hero-copy p,
.panel-heading p {
  color: #64748b;
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

.hero-status {
  align-self: center;
}

.stat-grid {
  display: grid;
  gap: 10px;
  grid-template-columns: repeat(3, 1fr);
  margin-top: 22px;
}

.stat-item {
  background: #f8fafc;
  border: 1px solid #eef2f7;
  border-radius: 8px;
  padding: 12px;
}

.stat-item span,
.detail-row span,
.config-item span,
.field-block label {
  color: #64748b;
  display: block;
  font-size: 12px;
}

.stat-item strong {
  display: block;
  font-size: 24px;
  margin-top: 4px;
}

.workspace-grid {
  display: grid;
  gap: 16px;
  grid-template-columns: minmax(280px, 0.9fr) minmax(0, 1.1fr);
  margin-bottom: 16px;
}

.panel {
  padding: 20px;
}

.panel-heading {
  align-items: flex-start;
  display: flex;
  gap: 16px;
  justify-content: space-between;
  margin-bottom: 18px;
}

.panel-heading.compact {
  align-items: center;
}

.panel-heading h2 {
  font-size: 20px;
  margin: 0 0 4px;
}

.panel-heading > .anticon {
  color: #1677ff;
  font-size: 22px;
  margin-top: 4px;
}

.ready-icon {
  color: #0f766e !important;
}

.pending-icon {
  color: #b45309 !important;
}

.provider-list {
  display: grid;
  gap: 8px;
}

.provider-item {
  align-items: center;
  background: #ffffff;
  border: 1px solid #e5e7eb;
  border-radius: 8px;
  cursor: pointer;
  display: flex;
  gap: 12px;
  justify-content: space-between;
  padding: 12px;
  text-align: left;
  transition: border-color 160ms ease, background 160ms ease, transform 160ms ease;
  width: 100%;
}

.provider-item:hover,
.provider-item.active {
  background: #f8fbff;
  border-color: #1677ff;
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
  color: #64748b;
}

.provider-tags {
  flex-shrink: 0;
}

.provider-detail {
  display: grid;
  gap: 14px;
}

.detail-row {
  background: #f8fafc;
  border-radius: 8px;
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
  border-top: 1px solid #eef2f7;
  display: grid;
  gap: 12px;
  padding-top: 14px;
}

.section-title {
  align-items: center;
  display: flex;
  gap: 12px;
  justify-content: space-between;
}

.section-title strong,
.section-title span {
  display: block;
}

.section-title span {
  color: #64748b;
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
  border: 1px solid #e5e7eb;
  border-radius: 8px;
  display: grid;
  gap: 12px;
  padding: 12px;
  transition: border-color 160ms ease, box-shadow 160ms ease, transform 160ms ease;
}

.account-card:hover,
.account-card.current {
  border-color: #1677ff;
  box-shadow: 0 8px 24px rgba(15, 23, 42, 0.08);
  transform: translateY(-1px);
}

.account-card.current {
  background: #f8fbff;
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
  color: #64748b;
  font-size: 12px;
}

.account-meta {
  display: grid;
  gap: 4px;
}

.account-actions {
  display: flex;
  flex-wrap: wrap;
  gap: 8px;
}

.field-block {
  display: grid;
  gap: 8px;
}

.shortcut-row {
  border-top: 1px solid #eef2f7;
  padding-top: 10px;
}

.models-panel,
.config-panel {
  margin-bottom: 16px;
}

.model-toolbar {
  margin-bottom: 12px;
  max-width: 420px;
}

.availability-message {
  color: #64748b;
  display: block;
  max-width: 260px;
  overflow-wrap: anywhere;
}

.config-grid {
  display: grid;
  gap: 12px;
  grid-template-columns: repeat(3, minmax(0, 1fr));
  margin-bottom: 16px;
}

.config-item {
  background: #f8fafc;
  border-radius: 8px;
  min-width: 0;
  padding: 12px;
}

.config-item strong,
.config-item :deep(.ant-typography) {
  display: block;
  margin-top: 6px;
  overflow-wrap: anywhere;
}

.token-item {
  display: grid;
  gap: 8px;
}

.code-header {
  align-items: center;
  display: flex;
  justify-content: space-between;
  margin-bottom: 8px;
}

.code-block,
.log-box {
  background: #0f172a;
  border-radius: 8px;
  color: #e5e7eb;
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
  .config-grid {
    grid-template-columns: 1fr;
  }
}

@media (max-width: 640px) {
  .hero-panel,
  .panel {
    padding: 16px;
  }

  .hero-copy h1 {
    font-size: 26px;
  }

  .stat-grid {
    grid-template-columns: 1fr;
  }
}
</style>
