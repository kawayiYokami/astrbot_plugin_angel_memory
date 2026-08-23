<template>
  <div class="settings-page">
    <!-- 说明卡：独立一张，与下方分组卡平级 -->
    <n-card embedded class="settings-head-card">
      <div class="settings-head">
        <p class="title">插件全局配置</p>
        <p class="desc">
          与 AstrBot 原生插件配置页共用同一份数据；保存后即时生效，无需重启。
          检索模型等 provider 类改动依赖后台初始化的组件实例，需重载插件后生效。
        </p>
      </div>
    </n-card>

    <!-- 分段卡片：SchemaForm 直接平铺，每卡一组 -->
    <SchemaForm
      v-if="loaded"
      :schema="schema"
      v-model:model-value="values"
      :providers="providers"
    />
    <n-spin v-else size="small" style="margin-top: 40px" />

    <!-- 悬浮保存按钮：仅存在草稿变更时浮出 -->
    <Transition name="fab">
      <n-button
        v-if="dirty"
        class="save-fab"
        type="primary"
        round
        size="large"
        :loading="saving"
        @click="save"
      >
        <template #icon><Icon icon="lucide:save" /></template>
        保存更改
      </n-button>
    </Transition>
  </div>
</template>

<script setup lang="ts">
import { computed, onMounted, ref } from 'vue'
import {
  NButton,
  NCard,
  NSpin,
  useMessage,
} from 'naive-ui'
import { Icon } from '@iconify/vue'
import { useBridge } from '@/composables/useBridge'
import SchemaForm from '@/SchemaForm.vue'
import type { SchemaMeta } from '@/schema'

const { apiGet, apiPost } = useBridge()
const message = useMessage()

const schema = ref<Record<string, SchemaMeta>>({})
const values = ref<Record<string, unknown>>({})
const providers = ref<Record<string, string[]>>({})
const loaded = ref(false)
const saving = ref(false)
// 草稿基线：与当前值不一致即有变更，浮出保存按钮
const baseline = ref('')
const dirty = computed(
  () => loaded.value && JSON.stringify(values.value) !== baseline.value,
)

async function load() {
  try {
    const data = await apiGet<{
      schema: Record<string, SchemaMeta>
      values: Record<string, unknown>
      providers: Record<string, string[]>
    }>('plugin_config')
    schema.value = data?.schema || {}
    values.value = data?.values || {}
    providers.value = data?.providers || {}
    baseline.value = JSON.stringify(values.value)
    loaded.value = true
  } catch (e) {
    message.error(`加载插件设置失败: ${(e as Error).message}`)
  }
}

async function save() {
  saving.value = true
  try {
    await apiPost('plugin_config/save', { values: values.value })
    baseline.value = JSON.stringify(values.value)
    message.success('已保存并即时生效')
  } catch (e) {
    message.error(`保存失败: ${(e as Error).message}`)
  } finally {
    saving.value = false
  }
}

onMounted(load)
</script>

<style scoped>
.settings-page {
  max-width: 800px;
  margin: 0 auto;
  display: flex;
  flex-direction: column;
  gap: 12px;
}

.settings-head-card {
  border-radius: var(--radius-lg);
}

.settings-head {
  min-width: 0;
}

.title {
  margin: 0;
  font-size: 15px;
  font-weight: 600;
  color: var(--text-1);
}

.desc {
  margin: 4px 0 0;
  font-size: 12px;
  line-height: 1.6;
  color: var(--text-3);
}

/* 悬浮保存按钮：右下角浮出 */
.save-fab {
  position: fixed;
  right: 28px;
  bottom: 28px;
  z-index: 100;
  box-shadow: var(--glass-shadow);
}

.fab-enter-active,
.fab-leave-active {
  transition: opacity 0.25s ease, transform 0.25s ease;
}

.fab-enter-from,
.fab-leave-to {
  opacity: 0;
  transform: translateY(14px) scale(0.92);
}
</style>
