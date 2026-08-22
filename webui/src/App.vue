<template>
  <n-config-provider
    :theme="naiveTheme"
    :theme-overrides="overrides"
    :locale="zhCN"
    :date-locale="dateZhCN"
  >
    <n-message-provider>
      <n-dialog-provider>
        <!-- Naive UI 标准 admin 布局：绝对定位三段式 -->
        <n-layout position="absolute" has-sider>
          <n-layout-sider
            bordered
            class="glass-sider"
            collapse-mode="width"
            :collapsed-width="64"
            :width="208"
            show-trigger="bar"
            :collapsed="collapsed"
            @collapse="collapsed = true"
            @expand="collapsed = false"
          >
            <div class="app-brand" :class="{ 'is-collapsed': collapsed }">
              <button
                class="theme-toggle"
                type="button"
                :title="isDark ? '切换到光模式' : '切换到暗模式'"
                @click="toggleTheme()"
              >
                <Icon :icon="isDark ? 'lucide:sun' : 'lucide:moon'" />
              </button>
              <template v-if="!collapsed">
                <Icon icon="lucide:brain" class="brand-icon" />
                <span class="brand-text">天使的记忆</span>
              </template>
            </div>
            <n-menu
              :collapsed="collapsed"
              :collapsed-width="64"
              :collapsed-icon-size="20"
              :options="menuOptions"
              :value="activeKey"
              @update:value="onMenuSelect"
            />
          </n-layout-sider>

          <n-layout>
            <n-layout-header bordered class="app-header">
              <span class="header-title">{{ currentTitle }}</span>
            </n-layout-header>
            <n-layout-content
              content-style="padding: 24px;"
              :native-scrollbar="false"
            >
              <router-view />
            </n-layout-content>
          </n-layout>
        </n-layout>
      </n-dialog-provider>
    </n-message-provider>
  </n-config-provider>
</template>

<script setup lang="ts">
import { computed, h, onMounted, provide, ref, watchEffect } from 'vue'
import { useRoute, useRouter } from 'vue-router'
import type { MenuOption } from 'naive-ui'
import {
  NConfigProvider,
  NDialogProvider,
  NIcon,
  NLayout,
  NLayoutContent,
  NLayoutHeader,
  NLayoutSider,
  NMenu,
  NMessageProvider,
  darkTheme,
  dateZhCN,
  zhCN,
} from 'naive-ui'
import { Icon } from '@iconify/vue'
import { useBridge } from '@/composables/useBridge'
import { buildThemeOverrides, createThemeApi, themeKey } from './theme'

const router = useRouter()
const route = useRoute()
const { init } = useBridge()

const collapsed = ref(false)

const theme = createThemeApi()
provide(themeKey, theme)
const { isDark, toggle: toggleTheme } = theme

// data-theme 同步到根节点：CSS 变量按 :root[data-theme] 分组，
// 且 NModal 会 teleport 到 body，变量必须挂在 documentElement 上才能覆盖弹窗。
watchEffect(() => {
  document.documentElement.dataset.theme = theme.mode.value
})

const naiveTheme = computed(() => (theme.isDark.value ? darkTheme : null))
const overrides = computed(() => buildThemeOverrides(theme.mode.value))

function renderIcon(icon: string) {
  return () => h(NIcon, null, { default: () => h(Icon, { icon }) })
}

// 从路由表生成菜单（与路由 meta.title / meta.icon 对应）
const menuOptions: MenuOption[] = router
  .getRoutes()
  .filter(r => r.meta?.title)
  .map(r => ({
    label: r.meta!.title as string,
    key: r.path,
    icon: renderIcon((r.meta!.icon as string) || 'lucide:circle'),
  }))

const activeKey = computed(() => route.path)
const currentTitle = computed(() => (route.meta?.title as string) || '')

function onMenuSelect(key: string) {
  if (key !== route.path) {
    router.push(key)
  }
}

onMounted(async () => {
  await init()
})
</script>

<style scoped>
.app-brand {
  display: flex;
  align-items: center;
  gap: 8px;
  padding: 14px 16px;
  font-size: 15px;
  font-weight: 600;
  white-space: nowrap;
  overflow: hidden;
}

.app-brand.is-collapsed {
  flex-direction: column;
  padding: 10px 0;
}

.brand-icon {
  font-size: 22px;
  flex-shrink: 0;
}

.brand-text {
  overflow: hidden;
  text-overflow: ellipsis;
}

.app-header {
  display: flex;
  align-items: center;
  height: 48px;
  padding: 0 20px;
}

.header-title {
  font-size: 15px;
  font-weight: 600;
}
</style>

<style>
/* ---------- Liquid Glass 设计令牌（iOS 26 基准，与 angel 系列插件同款） ---------- */

html,
body,
#app {
  height: 100%;
  margin: 0;
  padding: 0;
  font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', 'PingFang SC',
    'Microsoft YaHei', sans-serif;
}

:root[data-theme='light'] {
  --bg-base: #f2f2f7;
  --glass-thick-bg: rgba(255, 255, 255, 0.66);
  --glass-regular-bg: rgba(255, 255, 255, 0.3);
  --glass-border: rgba(255, 255, 255, 0.65);
  --glass-highlight: rgba(255, 255, 255, 0.9);
  --glass-shadow: 0 8px 32px rgba(0, 0, 0, 0.08);
  --glass-divider: rgba(60, 60, 67, 0.12);
  --accent: #007aff;
  --accent-soft: rgba(0, 122, 255, 0.12);
  --text-1: #1d1d1f;
  --text-2: #6e6e73;
  --text-3: #8e8e93;
  --radius-xs: 8px;
  --radius-sm: 10px;
  --radius-md: 14px;
  --radius-lg: 20px;
  --radius-full: 999px;
}

:root[data-theme='dark'] {
  --bg-base: #0d0d0f;
  --glass-thick-bg: rgba(28, 28, 30, 0.62);
  --glass-regular-bg: rgba(44, 44, 46, 0.32);
  --glass-border: rgba(255, 255, 255, 0.14);
  --glass-highlight: rgba(255, 255, 255, 0.22);
  --glass-shadow: 0 8px 32px rgba(0, 0, 0, 0.45);
  --glass-divider: rgba(84, 84, 88, 0.4);
  --accent: #0a84ff;
  --accent-soft: rgba(10, 132, 255, 0.18);
  --text-1: #f5f5f7;
  --text-2: #aeaeb2;
  --text-3: #8e8e93;
  --radius-xs: 8px;
  --radius-sm: 10px;
  --radius-md: 14px;
  --radius-lg: 20px;
  --radius-full: 999px;
}

body {
  background: var(--bg-base);
}

/* 环境光斑：给玻璃提供可折射的色彩层。固定在背景，不参与交互。 */
body::before {
  content: '';
  position: fixed;
  inset: 0;
  z-index: 0;
  pointer-events: none;
  background:
    radial-gradient(42% 36% at 12% 8%, rgba(10, 132, 255, 0.16), transparent 70%),
    radial-gradient(38% 32% at 92% 16%, rgba(94, 92, 230, 0.13), transparent 70%),
    radial-gradient(46% 40% at 55% 100%, rgba(255, 55, 95, 0.08), transparent 72%);
}

:root[data-theme='dark'] body::before {
  background:
    radial-gradient(42% 36% at 12% 8%, rgba(10, 132, 255, 0.2), transparent 70%),
    radial-gradient(38% 32% at 92% 16%, rgba(191, 90, 242, 0.14), transparent 70%),
    radial-gradient(46% 40% at 55% 100%, rgba(100, 210, 255, 0.07), transparent 72%);
}

/* Liquid Glass · thick：侧栏面板 */
.glass-sider {
  background: var(--glass-thick-bg) !important;
  backdrop-filter: blur(40px) saturate(180%);
  -webkit-backdrop-filter: blur(40px) saturate(180%);
}

/* 左上角主题切换：胶囊玻璃圆钮 */
.theme-toggle {
  flex-shrink: 0;
  width: 34px;
  height: 34px;
  display: inline-flex;
  align-items: center;
  justify-content: center;
  font-size: 17px;
  color: inherit;
  border: 1px solid var(--glass-border);
  border-radius: var(--radius-full);
  background: var(--glass-regular-bg);
  backdrop-filter: blur(24px) saturate(180%);
  -webkit-backdrop-filter: blur(24px) saturate(180%);
  box-shadow:
    inset 0 1px 0 var(--glass-highlight),
    var(--glass-shadow);
  cursor: pointer;
  transition: transform 0.15s ease, background 0.2s ease;
}

.theme-toggle:hover {
  transform: scale(1.06);
}

.theme-toggle:active {
  transform: scale(0.94);
}

.header-title {
  color: var(--text-1);
}

/* default 型按钮的玻璃胶囊折射层 */
.n-button.n-button--default-type {
  backdrop-filter: blur(24px) saturate(180%);
  -webkit-backdrop-filter: blur(24px) saturate(180%);
}

/* 弹窗卡片：Liquid Glass 大圆角 */
.n-card,
.n-modal {
  border-radius: var(--radius-lg);
}

.n-card {
  border: 1px solid var(--glass-border);
  box-shadow:
    inset 0 1px 0 var(--glass-highlight),
    var(--glass-shadow) !important;
}
</style>
