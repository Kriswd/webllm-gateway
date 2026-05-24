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
    tip="正在连接 WebLLM Gateway..."
    size="large"
    class="boot-screen"
  />

  <div v-else class="app-shell">
    <LoginModal v-model:visible="loginVisible" />

    <a
      class="support-strip"
      href="https://pay.ldxp.cn/shop/FTIWLFHQ"
      target="_blank"
      rel="noopener noreferrer"
      aria-label="打开作者小店"
    >
      <span class="support-badge">支持作者</span>
      <span class="support-text">如果 WebAI Gateway 帮你少踩坑，可以通过作者的小店支持，有稳定可靠有质保的plus成品账号、Claude代充等，欢迎选购。</span>
      <span class="support-link">去看看</span>
    </a>

    <header class="topbar">
      <div class="brand">
        <span class="brand-mark"><RocketOutlined /></span>
        <div>
          <strong>WebLLM Gateway</strong>
          <small>网页登录模型 API</small>
        </div>
      </div>
      <a-space wrap>
        <a-button @click="openApiTestDrawer">
          <template #icon><ApiOutlined /></template>
          接口测试
        </a-button>
        <a-button danger @click="openLoginModal">
          <template #icon><PoweroffOutlined /></template>
          访问令牌
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
@font-face {
  font-display: swap;
  font-family: "Source Han Serif SC";
  font-style: normal;
  font-weight: 200 900;
  src:
    local("Source Han Serif SC"),
    local("Source Han Serif CN"),
    local("Noto Serif SC"),
    local("Noto Serif CJK SC"),
    url("/fonts/NotoSerifSC-VF.ttf") format("truetype");
}

:global(:root) {
  --wg-font-serif: "Source Han Serif SC", "Noto Serif SC", "Noto Serif CJK SC", STSong, SimSun, serif;
  --wg-bg: #f6f4ef;
  --wg-paper: #fffefa;
  --wg-paper-soft: #f2eee5;
  --wg-paper-muted: #ebe5da;
  --wg-ink: #211f1b;
  --wg-muted: #6c675e;
  --wg-quiet: #8b8377;
  --wg-rule: #dfd9cd;
  --wg-accent: #315b55;
  --wg-accent-strong: #244842;
  --wg-accent-soft: #e5ece9;
  --wg-info: #365979;
  --wg-danger: #98514e;
}

:global(html),
:global(body),
:global(#app),
:global(#app *),
:global(.ant-drawer),
:global(.ant-drawer *),
:global(.ant-modal),
:global(.ant-modal *),
:global(.ant-message),
:global(.ant-message *),
:global(.ant-notification),
:global(.ant-notification *) {
  font-family: var(--wg-font-serif) !important;
  letter-spacing: 0;
}

:global(body) {
  background: var(--wg-bg);
  color: var(--wg-ink);
}

:global(.ant-btn) {
  border-radius: 4px;
  box-shadow: none;
  font-weight: 600;
}

:global(.ant-btn-primary) {
  background: var(--wg-accent);
  border-color: var(--wg-accent);
}

:global(.ant-btn-primary:not(:disabled):hover) {
  background: var(--wg-accent-strong);
  border-color: var(--wg-accent-strong);
}

:global(.ant-input),
:global(.ant-select-selector),
:global(.ant-input-affix-wrapper),
:global(.ant-picker),
:global(.ant-segmented),
:global(.ant-card),
:global(.ant-drawer-content),
:global(.ant-modal-content) {
  border-radius: 5px !important;
}

.boot-screen {
  align-items: center;
  display: flex;
  height: 100vh;
  justify-content: center;
}

.app-shell {
  background:
    linear-gradient(180deg, rgba(255, 254, 250, 0.96), rgba(246, 244, 239, 0.98) 360px),
    var(--wg-bg);
  color: var(--wg-ink);
  min-height: 100vh;
  padding-top: 0;
}

.support-strip {
  align-items: center;
  background: #fbfaf7;
  border-bottom: 1px solid var(--wg-rule);
  color: inherit;
  display: flex;
  gap: 10px;
  min-height: 34px;
  padding: 5px 30px;
  text-decoration: none;
  transition: background 160ms ease, border-color 160ms ease, color 160ms ease;
}

.support-strip:hover {
  background: var(--wg-accent-soft);
  border-color: #c9d5d0;
}

.support-badge {
  background: var(--wg-accent-soft);
  border: 1px solid #c9d5d0;
  border-radius: 3px;
  color: var(--wg-accent);
  flex: 0 0 auto;
  font-size: 12px;
  font-weight: 600;
  line-height: 1;
  padding: 4px 8px;
}

.support-text {
  color: var(--wg-ink);
  flex: 1 1 auto;
  font-size: 13px;
  font-weight: 600;
  line-height: 1.25;
  min-width: 0;
  overflow: hidden;
  text-overflow: ellipsis;
  white-space: nowrap;
}

.support-link {
  border: 1px solid #c9d5d0;
  border-radius: 3px;
  color: var(--wg-accent);
  flex: 0 0 auto;
  font-size: 13px;
  font-weight: 600;
  padding: 4px 10px;
}

.support-strip:hover .support-link {
  background: #fffefa;
  color: var(--wg-accent-strong);
}

.topbar {
  align-items: center;
  backdrop-filter: blur(18px);
  background: rgba(255, 254, 250, 0.92);
  border-bottom: 1px solid rgba(223, 217, 205, 0.9);
  display: flex;
  gap: 16px;
  justify-content: space-between;
  min-height: 58px;
  padding: 10px 32px;
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
  background: var(--wg-accent-soft);
  border: 1px solid #c9d5d0;
  border-radius: 4px;
  color: var(--wg-accent);
  display: inline-flex;
  font-size: 18px;
  height: 38px;
  justify-content: center;
  width: 38px;
}

.brand strong,
.brand small {
  display: block;
}

.brand strong {
  font-size: 17px;
  font-weight: 700;
  letter-spacing: 0;
  line-height: 1.15;
}

.brand small {
  font-size: 12px;
  line-height: 1.25;
}

.brand small,
.muted {
  color: var(--wg-muted);
}

.content-shell {
  margin: 0 auto;
  max-width: 1480px;
  padding: 24px 28px 36px;
}

.drawer-stack {
  width: 100%;
}

.result-box {
  background: #201f1c;
  border: 1px solid #3c3831;
  border-radius: 5px;
  color: #f6f4ef;
  margin: 0;
  max-height: 260px;
  overflow: auto;
  padding: 12px;
  white-space: pre-wrap;
  word-break: break-word;
}

.error-text {
  color: var(--wg-danger);
  margin-bottom: 0;
}

@media (max-width: 720px) {
  .support-strip {
    align-items: center;
    flex-wrap: nowrap;
    min-height: 32px;
    padding: 5px 12px;
  }

  .support-text {
    font-size: 12px;
  }

  .support-link {
    display: none;
  }

  .topbar {
    align-items: center;
    gap: 10px;
    min-height: 52px;
    padding: 8px 12px;
  }

  .brand {
    gap: 9px;
  }

  .brand-mark {
    border-radius: 6px;
    font-size: 16px;
    height: 34px;
    width: 34px;
  }

  .brand strong {
    font-size: 15px;
  }

  .brand small {
    font-size: 11px;
  }

  .topbar :deep(.ant-btn) {
    font-size: 13px;
    height: 32px;
    padding-inline: 10px;
  }

  .content-shell {
    padding: 12px;
  }
}
</style>
