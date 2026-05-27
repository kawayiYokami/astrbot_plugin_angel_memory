<template>
  <div>
    <h2 class="text-h5 mb-4">📝 笔记读取</h2>

    <v-card class="mb-4">
      <v-card-text>
        <v-row>
          <v-col cols="12" sm="3">
            <v-text-field
              v-model.number="noteShortId"
              label="note_short_id"
              type="number"
              density="compact"
              variant="outlined"
              :min="0"
            />
          </v-col>
          <v-col cols="12" sm="3">
            <v-text-field
              v-model.number="startLine"
              label="start_line"
              type="number"
              density="compact"
              variant="outlined"
              :min="1"
            />
          </v-col>
          <v-col cols="12" sm="3">
            <v-text-field
              v-model.number="endLine"
              label="end_line"
              type="number"
              density="compact"
              variant="outlined"
              :min="1"
            />
          </v-col>
          <v-col cols="12" sm="3">
            <v-btn color="primary" @click="doRecall" :loading="loading" block>读取</v-btn>
          </v-col>
        </v-row>
      </v-card-text>
    </v-card>

    <v-alert v-if="error" type="error" class="mb-4">{{ error }}</v-alert>

    <v-card v-if="result">
      <v-card-title class="d-flex align-center ga-2">
        <v-icon icon="mdi-file-document" />
        {{ result.source_file_path }}
      </v-card-title>
      <v-card-subtitle>
        note_short_id={{ result.note_short_id }} |
        total_lines={{ result.total_lines }} |
        显示行 {{ result.actual_start_line }}-{{ result.actual_end_line }}
      </v-card-subtitle>
      <v-card-text>
        <pre class="pa-3 rounded" style="background: #1a1a2e; overflow-x: auto; white-space: pre-wrap; font-size: 13px;">{{ result.content }}</pre>
      </v-card-text>
    </v-card>

    <!-- 笔记文件浏览 -->
    <v-card class="mt-6">
      <v-card-title>📂 笔记文件浏览</v-card-title>
      <v-card-text>
        <v-select
          v-model="selectedFile"
          :items="files"
          label="选择文件"
          density="compact"
          variant="outlined"
          @update:model-value="loadFileContent"
        />
        <pre v-if="fileContent" class="pa-3 rounded mt-3" style="background: #1a1a2e; overflow-x: auto; white-space: pre-wrap; font-size: 13px;">{{ fileContent }}</pre>
      </v-card-text>
    </v-card>
  </div>
</template>

<script setup lang="ts">
import { ref, onMounted } from 'vue'
import { useBridge } from '@/composables/useBridge'

const { apiGet, apiPost } = useBridge()

const noteShortId = ref(0)
const startLine = ref(1)
const endLine = ref(200)
const loading = ref(false)
const result = ref<any>(null)
const error = ref('')

// 文件浏览
const files = ref<string[]>([])
const selectedFile = ref('')
const fileContent = ref('')

async function doRecall() {
  loading.value = true
  error.value = ''
  result.value = null
  try {
    const data: any = await apiPost('notes/recall', {
      note_short_id: noteShortId.value,
      start_line: startLine.value,
      end_line: endLine.value,
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
  } catch (e) { /* ignore */ }
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
