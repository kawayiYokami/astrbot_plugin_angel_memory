<template>
  <div class="settings-page">
    <!-- 说明卡：独立一张，与下方分组卡平级 -->
    <v-card rounded="md">
      <v-card-text>
        <div class="text-subtitle-1 font-weight-medium">插件全局配置</div>
        <div class="text-caption text-medium-emphasis mt-1" style="line-height: 1.6;">
          与插件配置页共用同一份数据；保存后即时生效，无需重启。
          模型服务商这类改动依赖后台已初始化的组件，改完后需重载插件才生效。
        </div>
      </v-card-text>
    </v-card>

    <!-- 分段卡片：SchemaForm 直接平铺，每卡一组 -->
    <SchemaForm
      v-if="loaded"
      :schema="schema"
      v-model:model-value="values"
      :providers="providers"
      :scope-candidates="scopeCandidates"
    />
    <div v-else class="d-flex justify-center py-10">
      <v-progress-circular indeterminate color="primary" />
    </div>

    <!-- 悬浮保存按钮：仅存在草稿变更时浮出 -->
    <Transition name="fab">
      <v-btn
        v-if="dirty"
        class="save-fab"
        color="primary"
        variant="flat"
        size="large"
        rounded="pill"
        :loading="saving"
        prepend-icon="lucide:save"
        @click="save"
      >
        保存更改
      </v-btn>
    </Transition>
  </div>
</template>

<script setup lang="ts">
import { computed, onMounted, ref } from 'vue'
import { useBridge } from '@/composables/useBridge'
import { toast } from '@/composables/useToast'
import SchemaForm from '@/SchemaForm.vue'
import type { SchemaMeta, ScopeCandidates } from '@/schema'

const { apiGet, apiPost } = useBridge()

const schema = ref<Record<string, SchemaMeta>>({})
const values = ref<Record<string, unknown>>({})
const providers = ref<Record<string, string[]>>({})
const scopeCandidates = ref<ScopeCandidates>({})
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
    toast.error(`加载插件设置失败: ${(e as Error).message}`)
  }
  // 候选名单独立加载：失败只影响下拉选项，不挡住整页配置
  try {
    scopeCandidates.value = (await apiGet<ScopeCandidates>('scope-map/candidates')) || {}
  } catch (e) {
    console.warn('[设置] 分柜候选名单加载失败:', e)
  }
}

async function save() {
  saving.value = true
  try {
    await apiPost('plugin_config/save', { values: values.value })
    baseline.value = JSON.stringify(values.value)
    toast.success('已保存并即时生效')
  } catch (e) {
    toast.error(`保存失败: ${(e as Error).message}`)
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

/* 悬浮保存按钮：右下角浮出 */
.save-fab {
  position: fixed;
  right: 28px;
  bottom: 28px;
  z-index: 100;
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
