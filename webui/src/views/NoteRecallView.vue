<template>
  <div class="d-flex flex-column ga-4">
    <v-card rounded="md">
      <v-card-text class="config-section pb-2">
        <div class="config-title">读取笔记</div>
        <div class="config-hint">按编号取出一段笔记原文，方便确认索引内容是否完整</div>
      </v-card-text>
      <v-card-text class="pt-0 d-flex flex-wrap align-center ga-3">
        <v-text-field
          v-model.number="noteShortId"
          label="笔记编号"
          type="number"
          min="0"
          density="compact"
          hide-details
          variant="outlined"
          style="max-width: 180px;"
        />
        <v-text-field
          v-model.number="offset"
          label="起始行"
          type="number"
          min="1"
          density="compact"
          hide-details
          variant="outlined"
          placeholder="不填从第一行开始"
          style="max-width: 180px;"
        />
        <v-text-field
          v-model.number="limit"
          label="读取行数"
          type="number"
          min="1"
          density="compact"
          hide-details
          variant="outlined"
          style="max-width: 180px;"
        />
        <v-btn color="primary" variant="flat" :loading="loading" @click="doRecall">读取</v-btn>
      </v-card-text>
    </v-card>

    <v-alert v-if="error" type="error" variant="tonal">{{ error }}</v-alert>

    <v-card v-if="result" rounded="md">
      <v-card-text class="config-section pb-2">
        <div class="d-flex align-center ga-2 min-w-0">
          <v-icon icon="lucide:file-text" size="small" />
          <span class="text-body-2 text-truncate" :title="result.source_file_path">{{ result.source_file_path }}</span>
        </div>
        <div class="config-hint">
          编号 {{ result.note_short_id }}　共 {{ result.total_lines }} 行　本次显示第
          {{ result.actual_start_line }}–{{ result.actual_end_line }} 行
        </div>
      </v-card-text>
      <v-card-text class="pt-0">
        <pre class="code-pre">{{ result.content }}</pre>
      </v-card-text>
    </v-card>

    <!-- 笔记文件浏览 -->
    <v-card rounded="md">
      <v-card-text class="config-section pb-2">
        <div class="config-title">笔记文件浏览</div>
        <div class="config-hint">直接看源文件，排查索引与原文的差异</div>
      </v-card-text>
      <v-card-text class="pt-0">
        <v-select
          v-model="selectedFile"
          :items="fileOptions"
          label="选择文件"
          clearable
          density="compact"
          hide-details
          variant="outlined"
          style="max-width: 520px;"
          @update:model-value="loadFileContent"
        />
        <pre v-if="fileContent" class="code-pre mt-3">{{ fileContent }}</pre>
      </v-card-text>
    </v-card>
  </div>
</template>

<script setup lang="ts">
import { computed, ref, onMounted } from 'vue'
import { useBridge } from '@/composables/useBridge'
import { toast } from '@/composables/useToast'

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

const fileOptions = computed(() => files.value.map((f) => ({ title: f, value: f })))

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
    toast.error(`加载笔记文件列表失败: ${(e as Error).message}`)
  }
}

async function loadFileContent() {
  if (!selectedFile.value) {
    fileContent.value = ''
    return
  }
  try {
    const data: any = await apiGet('notes/file-content', { path: selectedFile.value })
    fileContent.value = data.content || ''
  } catch (e) {
    fileContent.value = ''
    toast.error(`加载文件内容失败: ${(e as Error).message}`)
  }
}

onMounted(() => loadFiles())
</script>

<style scoped>
.code-pre {
  background: rgba(var(--v-theme-on-surface), 0.06);
  border-radius: 6px;
  padding: 12px;
  overflow-x: auto;
  white-space: pre-wrap;
  word-break: break-all;
  font-size: 12px;
  line-height: 1.5;
  margin: 0;
  max-height: 50vh;
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
