<script setup>
import { ref, onMounted, reactive, computed } from 'vue';
import { useSettingsStore } from '@/stores/settings';
import { message } from 'ant-design-vue';
import { SettingOutlined, AppstoreOutlined } from '@ant-design/icons-vue';

const settingsStore = useSettingsStore();

const drawerVisible = ref(false);
const currentAdapter = ref(null);
const currentConfig = reactive({});

// 模型过滤配置
const modelFilter = reactive({
    mode: 'blacklist',
    list: []
});

// 挂载时获取数据
onMounted(async () => {
    await Promise.all([
        settingsStore.fetchAdaptersMeta(),
        settingsStore.fetchAdapterConfig()
    ]);
});

// 适配器列表
const adapters = computed(() => settingsStore.adaptersMeta);

// 检查模型是否启用
const isModelEnabled = (modelId) => {
    const inList = modelFilter.list.includes(modelId);
    if (modelFilter.mode === 'whitelist') {
        return inList;
    } else {
        return !inList;
    }
};

// 切换模型启用/禁用
const toggleModel = (modelId, enabled) => {
    const idx = modelFilter.list.indexOf(modelId);

    if (modelFilter.mode === 'whitelist') {
        // 白名单模式：启用=加入列表，禁用=移出列表
        if (enabled && idx === -1) {
            modelFilter.list.push(modelId);
        } else if (!enabled && idx !== -1) {
            modelFilter.list.splice(idx, 1);
        }
    } else {
        // 黑名单模式：禁用=加入列表，启用=移出列表
        if (!enabled && idx === -1) {
            modelFilter.list.push(modelId);
        } else if (enabled && idx !== -1) {
            modelFilter.list.splice(idx, 1);
        }
    }
};

// 切换模式时重置列表
const onModeChange = (newMode) => {
    if (newMode !== modelFilter.mode) {
        modelFilter.mode = newMode;
        modelFilter.list = [];
    }
};

// 打开抽屉进行编辑
const handleEdit = (adapter) => {
    currentAdapter.value = adapter;
    // 加载现有配置或默认值
    const existing = settingsStore.adapterConfig[adapter.id] || {};

    // 重置当前配置表单
    Object.keys(currentConfig).forEach(key => delete currentConfig[key]);

    // 使用现有值或schema中的默认值初始化表单
    if (adapter.configSchema) {
        adapter.configSchema.forEach(field => {
            if (existing[field.key] !== undefined) {
                currentConfig[field.key] = existing[field.key];
            } else {
                currentConfig[field.key] = field.default;
            }
        });
    }

    // 初始化模型过滤配置
    const filter = adapter.modelFilter || { mode: 'blacklist', list: [] };
    modelFilter.mode = filter.mode || 'blacklist';
    modelFilter.list = [...(filter.list || [])];

    drawerVisible.value = true;
};

// 保存配置
const handleSave = async () => {
    if (!currentAdapter.value) return;

    const configToSave = {
        [currentAdapter.value.id]: {
            ...currentConfig,
            modelFilter: {
                mode: modelFilter.mode,
                list: [...modelFilter.list]
            }
        }
    };

    const success = await settingsStore.saveAdapterConfig(configToSave);
    if (success) {
        // 更新本地缓存
        const adapter = settingsStore.adaptersMeta.find(a => a.id === currentAdapter.value.id);
        if (adapter) {
            adapter.modelFilter = { mode: modelFilter.mode, list: [...modelFilter.list] };
        }
        drawerVisible.value = false;
    }
};
</script>

<template>
    <a-layout style="background: transparent;">
        <a-card title="适配器管理" :bordered="false">
            <template #extra>
                <a-button type="link" @click="settingsStore.fetchAdaptersMeta">刷新列表</a-button>
            </template>

            <a-list :grid="{ gutter: 16, xs: 1, sm: 2, md: 3, lg: 3, xl: 4, xxl: 4 }" :data-source="adapters">
                <template #renderItem="{ item }">
                    <a-list-item>
                        <a-card hoverable @click="handleEdit(item)" :bodyStyle="{ padding: '12px 16px' }">
                            <div style="display: flex; align-items: center; justify-content: space-between; gap: 8px;">
                                <div style="display: flex; align-items: center; min-width: 0; flex: 1;">
                                    <AppstoreOutlined
                                        style="font-size: 18px; color: #1890ff; margin-right: 8px; flex-shrink: 0;" />
                                    <span
                                        style="font-weight: 600; font-size: 14px; overflow: hidden; text-overflow: ellipsis; white-space: nowrap;">{{
                                            item.id }}</span>
                                </div>
                                <SettingOutlined style="font-size: 16px; color: #8c8c8c; flex-shrink: 0;" />
                            </div>
                        </a-card>
                    </a-list-item>
                </template>
            </a-list>
        </a-card>

        <!-- 配置抽屉 -->
        <a-drawer v-if="currentAdapter" v-model:open="drawerVisible" :title="`配置适配器 - ${currentAdapter.id}`" width="500"
            placement="right">
            <!-- 适配器描述 -->
            <div v-if="currentAdapter.description"
                style="margin-bottom: 16px; padding: 12px; background: #f5f5f5; border-radius: 6px; color: #666; font-size: 13px; line-height: 1.6;">
                {{ currentAdapter.description }}
            </div>

            <!-- 模型管理折叠面板 -->
            <a-collapse v-if="currentAdapter.models && currentAdapter.models.length > 0" style="margin-bottom: 16px;">
                <a-collapse-panel key="models" header="模型管理">
                    <!-- 模式选择 -->
                    <div style="margin-bottom: 12px;">
                        <span style="margin-right: 12px; color: #666;">过滤模式:</span>
                        <a-radio-group :value="modelFilter.mode" @change="e => onModeChange(e.target.value)">
                            <a-radio value="blacklist">黑名单</a-radio>
                            <a-radio value="whitelist">白名单</a-radio>
                        </a-radio-group>
                    </div>
                    <div style="font-size: 12px; color: #999; margin-bottom: 12px;">
                        {{ modelFilter.mode === 'blacklist' ? '关闭的模型将被禁用，其他模型可用' : '仅开启的模型可用，其他模型禁用' }}
                    </div>

                    <!-- 模型列表 -->
                    <div style="max-height: 300px; overflow-y: auto;">
                        <div v-for="modelId in currentAdapter.models" :key="modelId"
                            style="display: flex; align-items: center; justify-content: space-between; padding: 8px 0; border-bottom: 1px solid #f0f0f0;">
                            <span style="font-size: 13px; color: #333;">{{ modelId }}</span>
                            <a-switch :checked="isModelEnabled(modelId)"
                                @change="checked => toggleModel(modelId, checked)" size="small" />
                        </div>
                    </div>
                </a-collapse-panel>
            </a-collapse>

            <!-- 其他配置项 -->
            <div v-if="!currentAdapter.configSchema || currentAdapter.configSchema.length === 0">
                <a-empty v-if="!currentAdapter.models || currentAdapter.models.length === 0" description="该适配器没有可配置项" />
            </div>

            <a-form layout="vertical" v-if="currentAdapter.configSchema && currentAdapter.configSchema.length > 0">
                <template v-for="field in currentAdapter.configSchema" :key="field.key">
                    <a-form-item :label="field.label" :required="field.required">
                        <!-- 字符串输入 -->
                        <a-input v-if="field.type === 'string'" v-model:value="currentConfig[field.key]"
                            :placeholder="field.placeholder" />

                        <!-- 数字输入 -->
                        <a-input-number v-if="field.type === 'number'" v-model:value="currentConfig[field.key]"
                            :min="field.min" :max="field.max" style="width: 100%;" />

                        <!-- 布尔开关 -->
                        <div v-if="field.type === 'boolean'">
                            <a-switch v-model:checked="currentConfig[field.key]" />
                        </div>

                        <!-- 下拉选择 -->
                        <a-select v-if="field.type === 'select'" v-model:value="currentConfig[field.key]"
                            :options="field.options" />

                        <div v-if="field.note" style="font-size: 12px; color: #8c8c8c; margin-top: 4px;">
                            {{ field.note }}
                        </div>
                    </a-form-item>
                </template>
            </a-form>

            <template #footer>
                <div style="text-align: right;">
                    <a-button style="margin-right: 8px" @click="drawerVisible = false">取消</a-button>
                    <a-button type="primary" @click="handleSave">保存配置</a-button>
                </div>
            </template>
        </a-drawer>
    </a-layout>
</template>
