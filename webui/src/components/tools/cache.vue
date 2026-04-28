<script setup>
import { h, ref, onMounted } from 'vue';
import { message } from 'ant-design-vue';
import { useSystemStore } from '@/stores/system';
import { useSettingsStore } from '@/stores/settings';
import {
    PoweroffOutlined,
    CheckCircleOutlined,
    LoadingOutlined,
    ClockCircleOutlined,
    DeleteOutlined,
    FolderOutlined,
    StopOutlined,
    LoginOutlined,
    DownOutlined
} from '@ant-design/icons-vue';

const systemStore = useSystemStore();
const settingsStore = useSettingsStore();

// 重启步骤当前状态
const currentStep = ref(0);
const restarting = ref(false);

// 重启步骤定义
const restartSteps = ref([
    {
        title: '准备重启',
        status: 'wait',
        icon: h(ClockCircleOutlined),
    },
    {
        title: '发送指令',
        status: 'wait',
        icon: h(PoweroffOutlined),
    },
    {
        title: '等待重启',
        status: 'wait',
        icon: h(LoadingOutlined),
    },
    {
        title: '重启完成',
        status: 'wait',
        icon: h(CheckCircleOutlined),
    },
]);

// 实例文件夹抽屉
const instanceDrawerOpen = ref(false);
const selectedFolders = ref([]);

// 实例文件夹列表
const instanceFolders = ref([]);

// 重启弹窗状态
const restartModalVisible = ref(false);

// Workers 列表（用于登录模式选择）
const workers = ref([]);

// 确认重启弹窗
const restartConfirmVisible = ref(false);
const pendingRestartOptions = ref({});

// 获取 workers 列表
const fetchWorkers = async () => {
    try {
        const res = await fetch('/admin/config/instances', {
            headers: settingsStore.getHeaders()
        });
        if (res.ok) {
            const instances = await res.json();
            // 从 instances 中提取所有 workers
            const allWorkers = [];
            for (const inst of instances) {
                for (const w of (inst.workers || [])) {
                    allWorkers.push({ name: w.name, instance: inst.name });
                }
            }
            workers.value = allWorkers;
        }
    } catch (e) {
        console.error('获取 Workers 列表失败', e);
    }
};

// 显示重启确认
const showRestartConfirm = (options = {}) => {
    pendingRestartOptions.value = options;
    restartConfirmVisible.value = true;
};

// 确认重启
const confirmRestart = () => {
    restartConfirmVisible.value = false;
    handleRestart(pendingRestartOptions.value);
};

onMounted(() => {
    fetchWorkers();
});

// 辅助函数：延迟
const sleep = (ms) => new Promise(resolve => setTimeout(resolve, ms));

// 执行重启
const handleRestart = async (options = {}) => {
    restartModalVisible.value = true;
    restarting.value = true;
    currentStep.value = 0;

    // 步骤1: 准备
    restartSteps.value[0].status = 'process';
    await sleep(500);
    restartSteps.value[0].status = 'finish';
    currentStep.value = 1;

    // 步骤2: 发送指令 (调用API)
    restartSteps.value[1].status = 'process';
    try {
        await systemStore.restartService(options);
        restartSteps.value[1].status = 'finish';
        currentStep.value = 2;
    } catch (e) {
        restartSteps.value[1].status = 'error';
        message.error('无法连接到服务器');
        return;
    }

    // 步骤3: 等待服务恢复 (轮询检查)
    restartSteps.value[2].status = 'process';
    // 先等待一小段时间让服务重启
    await sleep(3000);
    let retries = 20;
    while (retries > 0) {
        try {
            await systemStore.fetchStatus();
            if (systemStore.status) {
                break;
            }
        } catch (e) {
            // ignore
        }
        await sleep(2000);
        retries--;
    }
    restartSteps.value[2].status = 'finish';
    currentStep.value = 3;

    // 步骤4: 完成
    restartSteps.value[3].status = 'finish';

    message.success('服务重启成功');

    // 延迟关闭弹窗并重置状态
    setTimeout(() => {
        restartModalVisible.value = false;
        restarting.value = false;
        restartSteps.value.forEach(step => step.status = 'wait');
        currentStep.value = 0;
    }, 1500);
};

// 停止服务
const handleStop = async () => {
    try {
        const success = await systemStore.stopService();
        if (success) {
            message.success('服务已停止');
        }
    } catch (e) {
        message.error('停止服务失败: ' + e.message);
    }
};

// 清理缓存
const handleClearCache = async () => {
    try {
        const res = await fetch('/admin/cache/clear', {
            method: 'POST',
            headers: settingsStore.getHeaders()
        });
        if (res.ok) {
            message.success('缓存文件夹已清理');
        } else {
            message.error('清理失败');
        }
    } catch (e) {
        message.error('请求失败: ' + e.message);
    }
};

// 打开实例文件夹管理抽屉
const handleOpenInstanceDrawer = async () => {
    selectedFolders.value = [];
    instanceDrawerOpen.value = true;
    try {
        const res = await fetch('/admin/data-folders', {
            headers: settingsStore.getHeaders()
        });
        if (res.ok) {
            instanceFolders.value = await res.json();
        }
    } catch (e) {
        message.error('获取文件夹列表失败');
    }
};

// 选中/取消选中文件夹
const handleFolderSelect = (name, checked) => {
    if (checked) {
        if (!selectedFolders.value.includes(name)) {
            selectedFolders.value.push(name);
        }
    } else {
        selectedFolders.value = selectedFolders.value.filter(n => n !== name);
    }
};

// 删除选中的实例数据
const handleDeleteSelectedFolders = async () => {
    if (selectedFolders.value.length === 0) {
        message.warning('请先选择要删除的文件夹');
        return;
    }

    try {
        const res = await fetch('/admin/data-folders/delete', {
            method: 'POST',
            headers: settingsStore.getHeaders(),
            body: JSON.stringify({ folders: selectedFolders.value })
        });

        if (res.ok) {
            message.success(`已删除 ${selectedFolders.value.length} 个实例数据文件夹`);
            // 刷新列表
            await handleOpenInstanceDrawer();
        } else {
            message.error('删除失败');
        }
    } catch (e) {
        message.error('删除请求失败');
    }
};
</script>

<template>
    <a-layout style="background: transparent;">
        <!-- 项目管理板块 -->
        <a-card title="项目管理" :bordered="false" style="width: 100%; margin-bottom: 10px;">
            <div style="display: flex; align-items: center; justify-content: space-between;">
                <div style="display: flex; align-items: center;">
                    <div style="margin-right: 16px;">
                        <div style="font-weight: 600; margin-bottom: 4px;">系统服务控制</div>
                        <div style="font-size: 12px; color: #8c8c8c;">
                            控制后端服务的运行状态 (重启或停止)
                        </div>
                    </div>
                </div>
                <div>
                    <a-space>
                        <!-- 下拉式重启按钮 -->
                        <a-dropdown-button type="primary" size="large" @click="showRestartConfirm()">
                            <PoweroffOutlined />
                            重启
                            <template #overlay>
                                <a-menu>
                                    <a-menu-item key="normal" @click="showRestartConfirm()">
                                        <PoweroffOutlined />
                                        普通重启
                                    </a-menu-item>
                                    <a-menu-divider />
                                    <a-menu-item key="login" @click="showRestartConfirm({ loginMode: true })">
                                        <LoginOutlined />
                                        登录模式重启
                                    </a-menu-item>
                                    <a-sub-menu v-if="workers.length > 1" key="login-worker" title="指定 Worker 登录">
                                        <template #icon>
                                            <LoginOutlined />
                                        </template>
                                        <a-menu-item v-for="worker in workers" :key="worker.name"
                                            @click="showRestartConfirm({ loginMode: true, workerName: worker.name })">
                                            {{ worker.name }}
                                        </a-menu-item>
                                    </a-sub-menu>
                                </a-menu>
                            </template>
                        </a-dropdown-button>

                        <a-popconfirm ok-text="确定" cancel-text="取消" @confirm="handleStop" placement="topRight">
                            <template #title>
                                <div style="width: 240px;">
                                    <div style="font-weight: 500; margin-bottom: 4px;">确定要停止服务吗？</div>
                                    <div style="font-size: 12px; color: #f5222d;">停止后服务将完全终止，需要手动重新启动。</div>
                                </div>
                            </template>
                            <a-button type="primary" danger size="large">
                                <template #icon>
                                    <StopOutlined />
                                </template>
                                停止
                            </a-button>
                        </a-popconfirm>
                    </a-space>
                </div>
            </div>
        </a-card>

        <!-- 重启确认模态框 -->
        <a-modal v-model:open="restartConfirmVisible" title="确认重启" @ok="confirmRestart" ok-text="确定" cancel-text="取消"
            :width="400">
            <div style="padding: 12px 0;">
                <p v-if="!pendingRestartOptions.loginMode">确定要重启服务吗？</p>
                <p v-else-if="pendingRestartOptions.workerName">
                    确定要以<b>登录模式</b>重启服务吗？<br />
                    <span style="color: #1890ff;">仅初始化 Worker: {{ pendingRestartOptions.workerName }}</span>
                </p>
                <p v-else>确定要以<b>登录模式</b>重启服务吗？</p>
            </div>
        </a-modal>

        <!-- 重启进度模态框 -->
        <a-modal v-model:open="restartModalVisible" title="系统服务重启中" :footer="null" :closable="false"
            :maskClosable="false" width="500px">
            <div style="padding: 24px 0;">
                <a-steps :current="currentStep" :items="restartSteps" />
                <div style="text-align: center; margin-top: 24px; color: #8c8c8c;">
                    请稍候，系统正在执行重启操作...
                </div>
            </div>
        </a-modal>

        <!-- 缓存管理板块 -->
        <a-card title="缓存管理" :bordered="false" style="width: 100%;">
            <a-row :gutter="[16, 16]">
                <!-- 清理缓存 -->
                <a-col :xs="24" :md="12">
                    <a-card style="height: 100%;"
                        :body-style="{ display: 'flex', flexDirection: 'column', height: '100%' }">
                        <div style="flex: 1;">
                            <div style="display: flex; align-items: center; margin-bottom: 12px;">
                                <DeleteOutlined style="font-size: 24px; color: #1890ff; margin-right: 8px;" />
                                <div style="font-weight: 600; font-size: 16px;">清理缓存文件夹</div>
                            </div>
                            <div style="font-size: 12px; color: #8c8c8c; margin-bottom: 16px;">
                                清理项目运行过程中可能会遗留的临时缓存文件（如遇到错误时遗留的图片），<br>
                                不会影响用户数据和配置<strong style="color: #ff4d4f;">有任务运行时请勿执行</strong>
                            </div>
                        </div>
                        <a-popconfirm title="确定要清理缓存文件夹吗？" ok-text="确定" cancel-text="取消" @confirm="handleClearCache">
                            <a-button type="primary" block>
                                <template #icon>
                                    <DeleteOutlined />
                                </template>
                                清理缓存
                            </a-button>
                        </a-popconfirm>
                    </a-card>
                </a-col>

                <!-- 删除实例数据 -->
                <a-col :xs="24" :md="12">
                    <a-card style="height: 100%;"
                        :body-style="{ display: 'flex', flexDirection: 'column', height: '100%' }">
                        <div style="flex: 1;">
                            <div style="display: flex; align-items: center; margin-bottom: 12px;">
                                <FolderOutlined style="font-size: 24px; color: #ff4d4f; margin-right: 8px;" />
                                <div style="font-weight: 600; font-size: 16px;">删除实例数据文件夹</div>
                            </div>
                            <div style="font-size: 12px; color: #8c8c8c; margin-bottom: 16px;">
                                删除所有浏览器实例的用户数据文件夹，<br>
                                包括 Cookie、本地存储等，<strong style="color: #ff4d4f;">请谨慎操作</strong>
                            </div>
                        </div>
                        <a-button danger block @click="handleOpenInstanceDrawer">
                            <template #icon>
                                <FolderOutlined />
                            </template>
                            管理实例数据
                        </a-button>
                    </a-card>
                </a-col>
            </a-row>
        </a-card>

        <!-- 实例数据文件夹管理抽屉 -->
        <a-drawer v-model:open="instanceDrawerOpen" title="管理实例数据文件夹" placement="right" width="500">
            <div style="margin-bottom: 16px;">
                <div style="font-size: 12px; color: #8c8c8c; margin-bottom: 12px;">
                    选择要删除的实例数据文件夹，删除后无法恢复，请谨慎操作
                </div>

                <!-- 文件夹列表 -->
                <a-list :data-source="instanceFolders" bordered>
                    <template #renderItem="{ item }">
                        <a-list-item>
                            <a-list-item-meta>
                                <template #title>
                                    <a-checkbox :checked="selectedFolders.includes(item.name)"
                                        @change="e => handleFolderSelect(item.name, e.target.checked)">
                                        {{ item.name }}
                                    </a-checkbox>
                                </template>
                                <template #description>
                                    <div style="font-size: 12px; margin-top: 4px;">
                                        <div>路径: {{ item.path }}</div>
                                        <div>关联实例: {{ item.instance }}</div>
                                        <div>大小: {{ item.size }}</div>
                                    </div>
                                </template>
                            </a-list-item-meta>
                        </a-list-item>
                    </template>
                </a-list>
            </div>

            <!-- 抽屉底部操作按钮 -->
            <template #footer>
                <div style="display: flex; justify-content: space-between; align-items: center;">
                    <div style="font-size: 12px; color: #8c8c8c;">
                        已选择 {{ selectedFolders.length }} 个文件夹
                    </div>
                    <div>
                        <a-button style="margin-right: 8px;" @click="instanceDrawerOpen = false">
                            取消
                        </a-button>
                        <a-popconfirm placement="topRight" ok-text="确定删除" cancel-text="取消"
                            @confirm="handleDeleteSelectedFolders">
                            <template #title>
                                <div style="white-space: nowrap;">
                                    确定要删除选中的 {{ selectedFolders.length }} 个文件夹吗？
                                </div>
                            </template>
                            <a-button type="primary" danger :disabled="selectedFolders.length === 0">
                                删除选中项
                            </a-button>
                        </a-popconfirm>
                    </div>
                </div>
            </template>
        </a-drawer>
    </a-layout>
</template>
