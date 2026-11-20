<template>
  <v-app>
    <v-navigation-drawer
      v-model="drawer"
      :rail="rail"
      permanent
      width="240"
      rail-width="56"
      class="memory-drawer"
      color="transparent"
    >
      <div class="d-flex flex-column h-100">
        <div
          class="drawer-head pa-2"
          :class="rail ? 'd-flex flex-column align-center ga-2' : 'd-flex align-center ga-2'"
        >
          <v-btn
            :icon="rail ? 'lucide:panel-left-open' : 'lucide:panel-left-close'"
            variant="text"
            size="small"
            :aria-label="rail ? '展开侧栏' : '收起侧栏'"
            @click="rail = !rail"
          />
          <span v-if="!rail" class="text-subtitle-1 font-weight-medium flex-grow-1 text-truncate">天使的记忆</span>
        </div>

        <v-divider />

        <v-list class="flex-grow-1 overflow-y-auto py-2" nav bg-color="transparent">
          <v-list-item
            v-for="item in menuItems"
            :key="item.path"
            :active="route.path === item.path"
            color="primary"
            @click="onNav(item.path)"
          >
            <template #prepend>
              <v-icon :icon="item.icon" />
            </template>
            <v-list-item-title v-if="!rail">{{ item.title }}</v-list-item-title>
            <v-tooltip v-if="rail" activator="parent" location="end">{{ item.title }}</v-tooltip>
          </v-list-item>
        </v-list>

        <v-divider />

        <v-list nav bg-color="transparent" class="py-2">
          <v-list-item :active="route.path === '/settings'" color="primary" @click="onNav('/settings')">
            <template #prepend>
              <v-icon icon="lucide:settings" />
            </template>
            <v-list-item-title v-if="!rail">插件设置</v-list-item-title>
            <v-tooltip v-if="rail" activator="parent" location="end">插件设置</v-tooltip>
          </v-list-item>
        </v-list>
      </div>
    </v-navigation-drawer>

    <v-main>
      <div class="content-inner">
        <router-view />
      </div>
    </v-main>

    <ToastHost />
    <ConfirmHost />
  </v-app>
</template>

<script setup lang="ts">
import { computed, onMounted, onUnmounted, ref } from 'vue'
import { useRoute, useRouter } from 'vue-router'
import ToastHost from './components/ToastHost.vue'
import ConfirmHost from './components/ConfirmHost.vue'

const router = useRouter()
const route = useRoute()

const rail = ref(false)
const drawer = ref(true)

// 窄屏自动收起为 rail，与 brush/meme 一致
const narrowMql = window.matchMedia('(max-width: 768px)')
const syncRail = () => {
  rail.value = narrowMql.matches
}
onMounted(() => {
  syncRail()
  narrowMql.addEventListener('change', syncRail)
})
onUnmounted(() => {
  narrowMql.removeEventListener('change', syncRail)
})

// 插件设置沉底，不进主菜单
const menuItems = computed(() =>
  router
    .getRoutes()
    .filter((r) => r.meta?.title && r.path !== '/settings')
    .map((r) => ({
      path: r.path,
      title: String(r.meta?.title ?? r.path),
      icon: String(r.meta?.icon ?? 'lucide:circle'),
    })),
)

function onNav(path: string) {
  if (route.path !== path) void router.push(path)
}
</script>

<style scoped>
.content-inner {
  max-width: 960px;
  margin: 0 auto;
  padding: 24px;
}
</style>

<style>
html,
body,
#app {
  height: 100%;
  margin: 0;
  padding: 0;
}

html {
  overscroll-behavior: none;
}

::-webkit-scrollbar {
  width: 10px;
  height: 10px;
}

::-webkit-scrollbar-thumb {
  border-radius: 999px;
  background: rgba(var(--v-border-color), 0.24);
}

::-webkit-scrollbar-thumb:hover {
  background: rgba(var(--v-border-color), 0.4);
}

::-webkit-scrollbar-track {
  background: transparent;
}
</style>
