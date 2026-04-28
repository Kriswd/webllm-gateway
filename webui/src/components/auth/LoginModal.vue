<script setup>
import { ref } from 'vue';
import { useSettingsStore } from '@/stores/settings';
import { message } from 'ant-design-vue';
import { LockOutlined } from '@ant-design/icons-vue';

const props = defineProps({
    visible: {
        type: Boolean,
        required: true
    }
});

const emit = defineEmits(['update:visible', 'success']);

const settingsStore = useSettingsStore();
const token = ref(settingsStore.token);
const loading = ref(false);

const handleLogin = async () => {
    if (!token.value) {
        message.warning('请输入 Token');
        return;
    }

    loading.value = true;
    try {
        const originalToken = settingsStore.token;
        settingsStore.setToken(token.value);

        const success = await settingsStore.checkAuth();
        if (success) {
            message.success('验证成功');
            emit('success');
            emit('update:visible', false);
        } else {
            message.error('Token 验证失败，请检查是否正确');
            settingsStore.setToken(originalToken);
        }
    } catch (e) {
        message.error('验证过程发生错误');
    } finally {
        loading.value = false;
    }
};
</script>

<template>
    <a-modal :open="visible" title="需要身份验证" :closable="false" :maskClosable="false" :footer="null" width="400px"
        centered>
        <div style="padding: 20px 0;">
            <div style="text-align: center; margin-bottom: 24px;">
                <a-avatar :size="64" style="background-color: #1890ff">
                    <template #icon>
                        <LockOutlined />
                    </template>
                </a-avatar>
                <div style="margin-top: 16px; font-size: 16px; font-weight: 500;">
                    WebAI2API 管理面板
                </div>
                <div style="color: #8c8c8c; margin-top: 8px;">
                    请输入访问 API Token 以继续
                </div>
            </div>

            <a-form layout="vertical">
                <a-form-item label="API Token">
                    <a-input-password v-model:value="token" placeholder="请输入 API Token" size="large"
                        @pressEnter="handleLogin">
                        <template #prefix>
                            <LockOutlined style="color: rgba(0,0,0,.25)" />
                        </template>
                    </a-input-password>
                </a-form-item>

                <a-button type="primary" block size="large" :loading="loading" @click="handleLogin">
                    验证并登录
                </a-button>
            </a-form>
        </div>
    </a-modal>
</template>
