<script setup>
import { onMounted, onUnmounted, ref } from 'vue';
import { Modal, message } from 'ant-design-vue';
import {
  ApiOutlined,
  CheckCircleOutlined,
  CloseCircleOutlined,
  LoadingOutlined,
  PoweroffOutlined,
  ReloadOutlined,
  RocketOutlined,
} from '@ant-design/icons-vue';
import { useSettingsStore } from '@/stores/settings';
import LoginModal from '@/components/auth/LoginModal.vue';

const settingsStore = useSettingsStore();
const publicRoutes = new Set(['/', '/gateway/kris-bridge']);

const isInitializing = ref(true);
const loginVisible = ref(false);
const apiTestDrawer = ref(false);
const modelList = ref([]);
const testModel = ref('');
const testPrompt = ref('Say hello in one word');
const testResult = ref({ models: 'pending', chat: 'pending' });
const testError = ref({ models: '', chat: '' });
const chatText = ref('');

let connectionCheckInterval = null;
let disconnectModalShown = false;

async function loadModelsForTest() {
  testResult.value.models = 'loading';
  testError.value.models = '';
  try {
    const res = await fetch('/v1/models', { headers: settingsStore.getHeaders() });
    const data = await res.json();
    if (!res.ok) throw new Error(data.error?.message || data.detail || `HTTP ${res.status}`);
    modelList.value = Array.isArray(data.data) ? data.data : [];
    if (!testModel.value && modelList.value.length) {
      testModel.value = modelList.value[0].id;
    }
    testResult.value.models = 'success';
  } catch (error) {
    testResult.value.models = 'error';
    testError.value.models = error.message || String(error);
  }
}

async function runChatTest() {
  if (!testModel.value) {
    message.warning('请先选择模型');
    return;
  }
  testResult.value.chat = 'loading';
  testError.value.chat = '';
  chatText.value = '';
  try {
    const res = await fetch('/v1/chat/completions', {
      method: 'POST',
      headers: { ...settingsStore.getHeaders(), 'Content-Type': 'application/json' },
      body: JSON.stringify({
        model: testModel.value,
        messages: [{ role: 'user', content: testPrompt.value }],
        stream: false,
        max_tokens: 64,
      }),
    });
    const data = await res.json();
    if (!res.ok) throw new Error(data.error?.message || data.detail || `HTTP ${res.status}`);
    chatText.value = data.choices?.[0]?.message?.content || JSON.stringify(data);
    testResult.value.chat = 'success';
  } catch (error) {
    testResult.value.chat = 'error';
    testError.value.chat = error.message || String(error);
  }
}

function openApiTestDrawer() {
  apiTestDrawer.value = true;
  loadModelsForTest();
}

function statusTag(status) {
  if (status === 'success') return { color: 'success', text: '成功', icon: CheckCircleOutlined };
  if (status === 'error') return { color: 'error', text: '失败', icon: CloseCircleOutlined };
  if (status === 'loading') return { color: 'processing', text: '检测中', icon: LoadingOutlined };
  return { color: 'default', text: '待检测', icon: ApiOutlined };
}

async function checkConnection() {
  try {
    const res = await fetch('/health', { signal: AbortSignal.timeout(5000) });
    if (!res.ok) throw new Error(`HTTP ${res.status}`);
    if (disconnectModalShown) {
      disconnectModalShown = false;
      Modal.destroyAll();
      window.location.reload();
    }
  } catch {
    if (!disconnectModalShown && !isInitializing.value) {
      disconnectModalShown = true;
      Modal.warning({
        title: 'Gateway 暂时无法连接',
        content: '请检查本机 8610 服务是否仍在运行；连接恢复后页面会自动刷新。',
        okText: '我知道了',
        centered: true,
      });
    }
  }
}

function openLoginModal() {
  settingsStore.setToken('');
  loginVisible.value = true;
}

onMounted(async () => {
  try {
    if (settingsStore.token) {
      const isValid = await settingsStore.checkAuth();
      if (!isValid) settingsStore.setToken('');
    }
  } catch {
    settingsStore.setToken('');
  } finally {
    loginVisible.value = false;
    isInitializing.value = false;
  }

  connectionCheckInterval = setInterval(checkConnection, 5000);
});

onUnmounted(() => {
  if (connectionCheckInterval) clearInterval(connectionCheckInterval);
});
</script>

<template>
  <a-spin
    v-if="isInitializing"
    :spinning="true"
    tip="正在连接 WebAI Gateway..."
    size="large"
    class="boot-screen"
  />

  <div v-else class="app-shell">
    <LoginModal v-model:visible="loginVisible" />

    <header class="topbar">
      <div class="brand">
        <span class="brand-mark"><RocketOutlined /></span>
        <div>
          <strong>WebAI Gateway</strong>
          <small>网页登录向导</small>
        </div>
      </div>
      <a-space wrap>
        <a-button @click="openApiTestDrawer">
          <template #icon><ApiOutlined /></template>
          接口测试
        </a-button>
        <a-button danger @click="openLoginModal">
          <template #icon><PoweroffOutlined /></template>
          控制台令牌
        </a-button>
      </a-space>
    </header>

    <main class="content-shell">
      <router-view />
    </main>

    <a-drawer v-model:open="apiTestDrawer" title="接口测试" placement="right" width="520">
      <a-space direction="vertical" size="large" class="drawer-stack">
        <a-card size="small" title="GET /v1/models">
          <template #extra>
            <a-button size="small" @click="loadModelsForTest">
              <template #icon><ReloadOutlined /></template>
              刷新
            </a-button>
          </template>
          <a-tag :color="statusTag(testResult.models).color">
            <component :is="statusTag(testResult.models).icon" />
            {{ statusTag(testResult.models).text }}
          </a-tag>
          <p v-if="testResult.models === 'success'" class="muted">返回 {{ modelList.length }} 个模型。</p>
          <p v-if="testError.models" class="error-text">{{ testError.models }}</p>
        </a-card>

        <a-card size="small" title="POST /v1/chat/completions">
          <a-space direction="vertical" class="drawer-stack">
            <a-select v-model:value="testModel" show-search placeholder="选择模型">
              <a-select-option v-for="model in modelList" :key="model.id" :value="model.id">
                {{ model.id }}
              </a-select-option>
            </a-select>
            <a-textarea v-model:value="testPrompt" :rows="3" />
            <a-button type="primary" :loading="testResult.chat === 'loading'" @click="runChatTest">
              发送测试请求
            </a-button>
            <a-tag :color="statusTag(testResult.chat).color">
              <component :is="statusTag(testResult.chat).icon" />
              {{ statusTag(testResult.chat).text }}
            </a-tag>
            <pre v-if="chatText" class="result-box">{{ chatText }}</pre>
            <p v-if="testError.chat" class="error-text">{{ testError.chat }}</p>
          </a-space>
        </a-card>
      </a-space>
    </a-drawer>
  </div>
</template>

<style scoped>
.boot-screen {
  align-items: center;
  display: flex;
  height: 100vh;
  justify-content: center;
}

.app-shell {
  background: #f6f8fb;
  color: #172033;
  min-height: 100vh;
}

.topbar {
  align-items: center;
  background: rgba(255, 255, 255, 0.92);
  border-bottom: 1px solid #e6eaf0;
  display: flex;
  gap: 16px;
  justify-content: space-between;
  min-height: 72px;
  padding: 14px 28px;
  position: sticky;
  top: 0;
  z-index: 20;
}

.brand {
  align-items: center;
  display: flex;
  gap: 12px;
  min-width: 0;
}

.brand-mark {
  align-items: center;
  background: #e8f1ff;
  border: 1px solid #bfdbfe;
  border-radius: 8px;
  color: #1677ff;
  display: inline-flex;
  font-size: 20px;
  height: 40px;
  justify-content: center;
  width: 40px;
}

.brand strong,
.brand small {
  display: block;
}

.brand strong {
  font-size: 18px;
}

.brand small,
.muted {
  color: #64748b;
}

.content-shell {
  margin: 0 auto;
  max-width: 1480px;
  padding: 20px;
}

.drawer-stack {
  width: 100%;
}

.result-box {
  background: #0f172a;
  border-radius: 8px;
  color: #e5e7eb;
  margin: 0;
  max-height: 260px;
  overflow: auto;
  padding: 12px;
  white-space: pre-wrap;
  word-break: break-word;
}

.error-text {
  color: #c2410c;
  margin-bottom: 0;
}

@media (max-width: 720px) {
  .topbar {
    align-items: flex-start;
    flex-direction: column;
    padding: 12px 16px;
  }

  .content-shell {
    padding: 12px;
  }
}
</style>
