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
  LinkOutlined,
  LoginOutlined,
  ReloadOutlined,
  RocketOutlined,
  SettingOutlined,
  ToolOutlined,
} from '@ant-design/icons-vue';

const settingsStore = useSettingsStore();

const loading = ref(false);
const actionLoading = ref(false);
const progressVisible = ref(false);
const loadError = ref('');
const actionError = ref('');
const actionLogs = ref([]);
const actionKind = ref('');
const selectedProviderId = ref('');
const selectedWorkerName = ref('');
const modelSearch = ref('');
const modelScope = ref('selected');
const tokenVisible = ref(false);
const cdpUrl = ref('http://127.0.0.1:9222');
const workerOptions = ref([]);

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

const filteredModels = computed(() => {
  const query = modelSearch.value.trim().toLowerCase();
  return (onboarding.value.models || []).filter((model) => {
    const modelId = String(model.id || '');
    const owner = providerForModel(modelId);
    const inScope = modelScope.value === 'all' || !selectedProvider.value || owner?.id === selectedProvider.value.id;
    const matchesQuery = !query || modelId.toLowerCase().includes(query) || owner?.name?.toLowerCase().includes(query);
    return inScope && matchesQuery;
  });
});

const clientConfig = computed(() => [
  `base_url = ${gatewayBaseUrl.value}`,
  `api_key = ${gatewayToken.value || '<网关令牌>'}`,
  `model = ${selectedProvider.value?.availableModels?.[0] || onboarding.value.gateway?.defaultModel || '<模型 ID>'}`,
  '',
  '# KrisAI / OpenClaw / Hermes / Claude Code 都可以使用这组 OpenAI 兼容配置',
  '# 工具调用由 WebAI Gateway 转换为网页模型可理解的 prompt 协议',
].join('\n'));

const modelColumns = [
  { title: '模型 ID', dataIndex: 'id', key: 'id' },
  { title: '来源', key: 'provider', width: 180 },
  { title: '能力', key: 'capability', width: 180 },
  { title: '操作', key: 'action', width: 110 },
];

const stepItems = computed(() => [
  { title: '选择平台', status: selectedProvider.value ? 'finish' : 'process' },
  { title: '网页登录', status: isProviderReady(selectedProvider.value) ? 'finish' : actionLoading.value ? 'process' : 'wait' },
  { title: '使用模型', status: filteredModels.value.length ? 'finish' : 'wait' },
]);

function isProviderReady(provider) {
  if (!provider) return false;
  if (provider.loginKind === 'direct') return Boolean(provider.credential?.authorized);
  return true;
}

function providerForModel(modelId) {
  return providers.value.find((provider) => (provider.models || []).includes(modelId))
    || providers.value.find((provider) => provider.id === onboarding.value.models?.find((model) => model.id === modelId)?.owned_by)
    || null;
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
  await navigator.clipboard.writeText(text);
  message.success(successText);
}

function copyModel(modelId) {
  copyText(modelId, '模型 ID 已复制');
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
            <strong>{{ onboarding.summary.authorizedDirectProviders }}</strong>
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
                <small>{{ provider.modelCount || provider.models?.length || 0 }} 个模型</small>
              </span>
              <span class="provider-tags">
                <a-tag v-if="provider.loginKind === 'direct' && provider.credential?.authorized" color="success">已授权</a-tag>
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
              <a-button size="large" @click="copyText(selectedProvider.availableModels?.[0] || selectedProvider.models?.[0] || '', '模型 ID 已复制')">
                <template #icon><CopyOutlined /></template>
                复制默认模型
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
            <p>复制模型 ID 后，填到 KrisAI 或任何 OpenAI 兼容客户端。</p>
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
                  v-for="cap in capabilityLabels(providerForModel(record.id))"
                  :key="cap.label"
                  :color="cap.color"
                >
                  {{ cap.label }}
                </a-tag>
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

    <a-modal v-model:open="progressVisible" title="授权进度" :footer="null" width="620px">
      <a-alert
        v-if="actionError"
        type="error"
        show-icon
        :message="actionError"
        style="margin-bottom: 12px;"
      />
      <a-steps
        size="small"
        :current="actionLoading ? 1 : actionError ? 0 : 2"
        :status="actionError ? 'error' : 'process'"
        :items="[
          { title: '启动' },
          { title: actionKind === 'direct' ? '检测登录' : '登录模式' },
          { title: '完成' },
        ]"
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
