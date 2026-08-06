<template>
  <n-config-provider :theme="darkTheme" :locale="zhCN" :date-locale="dateZhCN">
    <n-message-provider>
      <n-dialog-provider>
        <n-layout has-sider style="height: 100vh">
          <!-- 左侧导航（两段式第一段） -->
          <n-layout-sider
            bordered
            collapse-mode="width"
            :collapsed-width="56"
            :width="200"
            show-trigger="bar"
            :collapsed="collapsed"
            @collapse="collapsed = true"
            @expand="collapsed = false"
          >
            <div class="app-brand">
              <Icon :icon="'lucide:brain'" class="brand-icon" />
              <span v-if="!collapsed" class="brand-text">天使的记忆</span>
            </div>
            <n-menu
              :collapsed="collapsed"
              :collapsed-width="56"
              :collapsed-icon-size="20"
              :options="menuOptions"
              :value="activeKey"
              @update:value="onMenuSelect"
            />
          </n-layout-sider>

          <!-- 主内容区（两段式第二段） -->
          <n-layout>
            <n-layout-header bordered class="app-header">
              <span class="header-title">{{ currentTitle }}</span>
            </n-layout-header>
            <n-layout-content content-style="padding: 20px 24px; overflow: auto;">
              <router-view />
            </n-layout-content>
          </n-layout>
        </n-layout>
      </n-dialog-provider>
    </n-message-provider>
  </n-config-provider>
</template>

<script setup lang="ts">
import { computed, h, ref, onMounted } from 'vue'
import { useRoute, useRouter } from 'vue-router'
import type { MenuOption } from 'naive-ui'
import { NIcon, darkTheme, dateZhCN, zhCN } from 'naive-ui'
import { Icon } from '@iconify/vue'
import { useBridge } from '@/composables/useBridge'

const router = useRouter()
const route = useRoute()
const { init } = useBridge()

const collapsed = ref(false)

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
