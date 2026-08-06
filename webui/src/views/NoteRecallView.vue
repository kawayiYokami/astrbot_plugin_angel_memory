<template>
  <div>
    <n-card title="读取笔记" embedded class="mb-4">
      <n-space>
        <n-input-number v-model:value="noteShortId" placeholder="note_short_id" :min="0" style="width: 180px" />
        <n-input-number v-model:value="offset" placeholder="offset" :min="1" style="width: 140px" />
        <n-input-number v-model:value="limit" placeholder="limit" :min="1" style="width: 140px" />
        <n-button type="primary" :loading="loading" @click="doRecall">读取</n-button>
      </n-space>
    </n-card>

    <n-alert v-if="error" type="error" class="mb-4">{{ error }}</n-alert>

    <n-card v-if="result" embedded class="mb-4">
      <template #header>
        <n-space align="center" size="small">
          <Icon icon="lucide:file-text" />
          <span>{{ result.source_file_path }}</span>
        </n-space>
      </template>
      <div class="muted mb-2">
        note_short_id={{ result.note_short_id }} | total_lines={{ result.total_lines }} |
        显示行 {{ result.actual_start_line }}-{{ result.actual_end_line }}
      </div>
      <pre class="code-pre">{{ result.content }}</pre>
    </n-card>

    <!-- 笔记文件浏览 -->
    <n-card title="笔记文件浏览" embedded>
      <n-select
        v-model:value="selectedFile"
        :options="fileOptions"
        placeholder="选择文件"
        clearable
        @update:value="loadFileContent"
      />
      <pre v-if="fileContent" class="code-pre mt-3">{{ fileContent }}</pre>
    </n-card>
  </div>
</template>

<script setup lang="ts">
import { computed, ref, onMounted } from 'vue'
import { useBridge } from '@/composables/useBridge'

const { apiGet, apiPost } = useBridge()

const noteShortId = ref<number | null>(null)
const offset = ref<number | null>(null)
const limit = ref<number | null>(200)
const loading = ref(false)
const result = ref<any>(null)
const error = ref('')

// 文件浏览
const files = ref<string[]>([])
const selectedFile = ref<string | null>(null)
const fileContent = ref('')

const fileOptions = computed(() =>
  files.value.map(f => ({ label: f, value: f })),
)

async function doRecall() {
  loading.value = true
  error.value = ''
  result.value = null
  try {
    const data: any = await apiPost('notes/recall', {
      note_short_id: noteShortId.value,
      offset: offset.value,
      limit: limit.value,
    })
    if (data.error) {
      error.value = data.error
    } else {
      result.value = data
    }
  } catch (e: any) {
    error.value = e.message || '读取失败'
  } finally {
    loading.value = false
  }
}

async function loadFiles() {
  try {
    const data: any = await apiGet('notes/files')
    files.value = data.files || []
  } catch (e) {
    /* ignore */
  }
}

async function loadFileContent() {
  if (!selectedFile.value) return
  try {
    const data: any = await apiGet('notes/file-content', { path: selectedFile.value })
    fileContent.value = data.content || ''
  } catch (e) {
    fileContent.value = '加载失败'
  }
}

onMounted(() => loadFiles())
</script>

<style scoped>
.code-pre {
  background: rgba(0, 0, 0, 0.35);
  border-radius: 6px;
  padding: 12px;
  overflow-x: auto;
  white-space: pre-wrap;
  word-break: break-all;
  font-size: 13px;
  margin: 0;
}

.mt-3 {
  margin-top: 12px;
}

.muted {
  opacity: 0.65;
  font-size: 13px;
}
</style>
