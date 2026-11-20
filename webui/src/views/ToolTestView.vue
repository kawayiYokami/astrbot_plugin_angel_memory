<template>
  <div class="d-flex flex-column ga-3">
    <v-card rounded="md">
      <v-card-text class="config-section pb-2">
        <div class="config-title">回忆工具试跑</div>
        <div class="config-hint">手动触发一次回忆，看模型拿到什么。不影响真实对话</div>
      </v-card-text>
      <v-card-text class="pt-0 d-flex flex-column ga-3">
        <v-textarea
          v-model="query"
          label="要回忆什么"
          rows="3"
          auto-grow
          density="compact"
          hide-details
          variant="outlined"
          placeholder="例如：上次说好的生日礼物"
        />
        <div class="d-flex flex-wrap align-center ga-3">
          <v-select
            v-model="groupId"
            :items="groupOptions"
            label="在哪个群聊里回忆"
            clearable
            density="compact"
            hide-details
            variant="outlined"
            style="max-width: 280px;"
          />
          <v-text-field
            v-model="umo"
            label="会话标识（可不填）"
            density="compact"
            hide-details
            variant="outlined"
            placeholder="留空用示例会话"
            style="max-width: 300px;"
          />
          <v-btn color="primary" variant="flat" :loading="loading === 'rag'" :disabled="busy" @click="run('rag')">
            自动回忆
          </v-btn>
          <v-btn variant="outlined" :loading="loading === 'recall'" :disabled="busy" @click="run('recall')">
            主动回忆
          </v-btn>
        </div>
      </v-card-text>
    </v-card>

    <v-alert v-if="errors.length" type="error" variant="tonal">
      <div v-for="(item, index) in errors" :key="index" class="text-body-2">{{ item }}</div>
    </v-alert>

    <pre class="output">{{ text }}</pre>
  </div>
</template>

<script setup lang="ts">
import { computed, onMounted, ref } from 'vue'
import { useBridge } from '@/composables/useBridge'

interface GroupRow {
  group_id: string
  group_name?: string
}

const { apiGet, apiPost } = useBridge()

const query = ref('')
const groupId = ref<string | null>(null)
const umo = ref('')
const groups = ref<GroupRow[]>([])

const loading = ref<'' | 'rag' | 'recall'>('')
const text = ref('')
const errors = ref<string[]>([])

const busy = computed(() => loading.value !== '')

const groupOptions = computed(() =>
  groups.value.map((g) => ({
    title: g.group_name ? `${g.group_name}（${g.group_id}）` : g.group_id,
    value: g.group_id,
  })),
)

async function run(kind: 'rag' | 'recall') {
  loading.value = kind
  text.value = ''
  errors.value = []
  try {
    const endpoint = kind === 'rag' ? 'tool-test/auto-rag' : 'tool-test/active-recall'
    // 桥接把响应体的 data 解包后直接返回
    const resp: any = await apiPost(endpoint, {
      query: query.value.trim(),
      group_id: groupId.value || '',
      umo: umo.value.trim(),
    })
    text.value = resp?.text || ''
    errors.value = Array.isArray(resp?.errors) ? resp.errors.filter(Boolean) : []
  } catch (e: any) {
    errors.value = [`请求失败：${e?.message || String(e)}`]
  } finally {
    loading.value = ''
  }
}

async function loadGroups() {
  try {
    const resp: any = await apiGet('group-knowledge')
    groups.value = resp?.groups || []
  } catch (e: any) {
    errors.value = [`群列表加载失败：${e?.message || String(e)}`]
  }
}

onMounted(() => loadGroups())
</script>

<style scoped>
.output {
  background: rgba(var(--v-theme-on-surface), 0.06);
  border-radius: 8px;
  padding: 14px;
  overflow-x: auto;
  white-space: pre-wrap;
  word-break: break-word;
  font-size: 12px;
  line-height: 1.6;
  margin: 0;
  max-height: 75vh;
  overflow-y: auto;
  min-height: 140px;
}

.config-title {
  font-size: 0.875rem;
  font-weight: 500;
  color: rgb(var(--v-theme-primaryText));
}
.config-hint {
  font-size: 0.75rem;
  color: rgb(var(--v-theme-secondaryText));
  margin-top: 2px;
}
</style>
