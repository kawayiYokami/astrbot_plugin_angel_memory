<template>
  <n-space vertical :size="12">
    <n-input
      v-model:value="query"
      type="textarea"
      :rows="3"
      placeholder="检索词，例如：洛伊"
    />

    <n-space>
      <n-select
        v-model:value="groupId"
        :options="groupOptions"
        placeholder="群号（留空按私聊处理）"
        clearable
        filterable
        style="width: 260px"
      />
      <n-input v-model:value="umo" placeholder="会话 UMO（留空用示例）" style="width: 300px" />
      <n-button type="primary" :loading="loading" :disabled="!query.trim()" @click="run">
        运行
      </n-button>
    </n-space>

    <n-alert v-if="errors.length" type="error" :bordered="false">
      <div v-for="(item, index) in errors" :key="index" class="error-line">{{ item }}</div>
    </n-alert>

    <pre v-if="text" class="output">{{ text }}</pre>
    <pre v-else-if="!loading" class="output muted">结果会直接显示在这里。</pre>
  </n-space>
</template>

<script setup lang="ts">
import { computed, onMounted, ref } from 'vue'
import { NAlert, NButton, NInput, NSelect, NSpace } from 'naive-ui'
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

const loading = ref(false)
const text = ref('')
const errors = ref<string[]>([])

const groupOptions = computed(() =>
  groups.value.map(g => ({
    label: g.group_name ? `${g.group_name}（${g.group_id}）` : g.group_id,
    value: g.group_id,
  })),
)

async function run() {
  loading.value = true
  text.value = ''
  errors.value = []
  try {
    // 桥接会把响应体的 data 解包后直接返回，这里拿到的就是业务对象
    const resp: any = await apiPost('tool-test/run', {
      query: query.value.trim(),
      group_id: groupId.value || '',
      umo: umo.value.trim(),
    })
    text.value = resp?.text || ''
    errors.value = Array.isArray(resp?.errors) ? resp.errors.filter(Boolean) : []
    if (!text.value && !errors.value.length) {
      errors.value = ['接口返回了空内容，也没有错误信息，请检查插件日志。']
    }
  } catch (e: any) {
    errors.value = [`请求失败：${e?.message || String(e)}`]
  } finally {
    loading.value = false
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
  background: rgba(0, 0, 0, 0.35);
  border-radius: 6px;
  padding: 12px;
  overflow-x: auto;
  white-space: pre-wrap;
  word-break: break-word;
  font-size: 12px;
  line-height: 1.6;
  margin: 0;
  max-height: 70vh;
  overflow-y: auto;
}

.muted {
  opacity: 0.6;
}

.error-line {
  font-size: 12px;
  line-height: 1.6;
  word-break: break-word;
}
</style>
