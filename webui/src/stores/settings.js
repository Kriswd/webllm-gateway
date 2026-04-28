import { defineStore } from 'pinia';
import { message, Modal } from 'ant-design-vue';

export const useSettingsStore = defineStore('settings', {
    state: () => ({
        token: localStorage.getItem('admin_token') || '',
        serverConfig: {},
        browserConfig: {},
        workerConfig: [],
        poolConfig: {
            strategy: 'least_busy',
            waitTimeout: 120,
            failover: {
                enabled: false,
                maxRetries: 3,
                imgDlRetry: false,
                imgDlRetryMaxRetries: 2
            }
        },
        adapterConfig: {},
        adaptersMeta: []
    }),

    actions: {
        setToken(token) {
            this.token = token;
            if (token) {
                localStorage.setItem('admin_token', token);
            } else {
                localStorage.removeItem('admin_token');
            }
        },

        getHeaders() {
            const headers = { 'Content-Type': 'application/json' };
            if (this.token) {
                headers['Authorization'] = `Bearer ${this.token}`;
            }
            return headers;
        },

        async checkAuth() {
            try {
                const res = await fetch('/admin/status', {
                    headers: this.getHeaders()
                });
                return res.status !== 401;
            } catch (e) {
                return false;
            }
        },

        // 错误处理辅助函数
        async handleResponse(res, successMsg) {
            let data = {};
            try {
                data = await res.json();
            } catch (e) {
                // 忽略JSON解析错误
            }

            if (res.ok) {
                if (successMsg) message.success(successMsg);
                return { success: true, data };
            } else {
                console.error('Request failed:', res.status, data);
                // 后端返回格式: { error: { message: "..." } } 或 { message: "..." }
                const errorMessage = data.error?.message || data.message || `请求未成功: ${res.status} ${res.statusText}`;
                Modal.error({
                    title: '保存失败',
                    content: errorMessage,
                    okText: '好的'
                });
                return { success: false, data };
            }
        },

        // --- 服务器配置 ---
        async fetchServerConfig() {
            try {
                const res = await fetch('/admin/config/server', { headers: this.getHeaders() });
                if (res.ok) this.serverConfig = await res.json();
            } catch (e) {
                console.error('Fetch server config failed', e);
            }
        },
        async saveServerConfig(config) {
            try {
                const res = await fetch('/admin/config/server', {
                    method: 'POST',
                    headers: this.getHeaders(),
                    body: JSON.stringify(config)
                });
                const result = await this.handleResponse(res, '服务器设置保存成功');
                if (result.success) {
                    this.serverConfig = config;
                    return true;
                }
            } catch (e) {
                Modal.error({ title: '保存失败 (网络异常)', content: e.message });
            }
            return false;
        },

        // --- 浏览器配置 ---
        async fetchBrowserConfig() {
            try {
                const res = await fetch('/admin/config/browser', { headers: this.getHeaders() });
                if (res.ok) this.browserConfig = await res.json();
            } catch (e) {
                console.error('Fetch browser config failed', e);
            }
        },
        async saveBrowserConfig(config) {
            try {
                const res = await fetch('/admin/config/browser', {
                    method: 'POST',
                    headers: this.getHeaders(),
                    body: JSON.stringify(config)
                });
                const result = await this.handleResponse(res, '浏览器设置保存成功');
                if (result.success) {
                    this.browserConfig = config;
                    return true;
                }
            } catch (e) {
                Modal.error({ title: '保存失败 (网络异常)', content: e.message });
            }
            return false;
        },

        // --- 工作实例配置 ---
        async fetchWorkerConfig() {
            try {
                // 端点已更改为 /admin/config/instances
                const res = await fetch('/admin/config/instances', { headers: this.getHeaders() });
                if (res.ok) this.workerConfig = await res.json();
            } catch (e) {
                console.error('Fetch instance configuration failed', e);
            }
        },
        async saveWorkerConfig(config) {
            try {
                // 端点已更改为 /admin/config/instances
                const res = await fetch('/admin/config/instances', {
                    method: 'POST',
                    headers: this.getHeaders(),
                    body: JSON.stringify(config)
                });
                const result = await this.handleResponse(res, '实例配置保存成功');
                if (result.success) {
                    this.workerConfig = config;
                    return true;
                }
            } catch (e) {
                Modal.error({ title: '保存失败 (网络异常)', content: e.message });
            }
            return false;
        },

        // --- 工作池配置 ---
        async fetchPoolConfig() {
            try {
                const res = await fetch('/admin/config/pool', { headers: this.getHeaders() });
                if (res.ok) {
                    const data = await res.json();
                    // 合并以确保结构存在
                    this.poolConfig = {
                        strategy: data.strategy || 'least_busy',
                        waitTimeout: data.waitTimeout ?? 120,
                        failover: {
                            enabled: data.failover?.enabled || false,
                            maxRetries: data.failover?.maxRetries || 3,
                            imgDlRetry: data.failover?.imgDlRetry || false,
                            imgDlRetryMaxRetries: data.failover?.imgDlRetryMaxRetries ?? 2
                        }
                    };
                }
            } catch (e) {
                console.error('Fetch pool config failed', e);
            }
        },
        async savePoolConfig(config) {
            try {
                const res = await fetch('/admin/config/pool', {
                    method: 'POST',
                    headers: this.getHeaders(),
                    body: JSON.stringify(config)
                });
                const result = await this.handleResponse(res, '工作池设置保存成功');
                if (result.success) {
                    this.poolConfig = config;
                    return true;
                }
            } catch (e) {
                Modal.error({ title: '保存失败 (网络异常)', content: e.message });
            }
            return false;
        },

        // --- 适配器配置与元数据 ---
        async fetchAdaptersMeta() {
            try {
                const res = await fetch('/admin/adapters', { headers: this.getHeaders() });
                if (res.ok) this.adaptersMeta = await res.json();
            } catch (e) {
                console.error('Fetch adapters meta failed', e);
            }
        },
        async fetchAdapterConfig() {
            try {
                const res = await fetch('/admin/config/adapters', { headers: this.getHeaders() });
                if (res.ok) this.adapterConfig = await res.json();
            } catch (e) {
                console.error('Fetch adapter config failed', e);
            }
        },
        async saveAdapterConfig(config) {
            try {
                const res = await fetch('/admin/config/adapters', {
                    method: 'POST',
                    headers: this.getHeaders(),
                    body: JSON.stringify(config)
                });
                const result = await this.handleResponse(res, '适配器设置保存成功');
                if (result.success) {
                    // 通过合并更新本地状态
                    this.adapterConfig = { ...this.adapterConfig, ...config };
                    return true;
                }
            } catch (e) {
                Modal.error({ title: '保存失败 (网络异常)', content: e.message });
            }
            return false;
        }
    }
});
