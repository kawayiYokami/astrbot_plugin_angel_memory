<template>
  <div>
    <h2 class="text-h5 mb-4">🔄 导入导出</h2>

    <v-row>
      <v-col cols="12" md="6">
        <v-card>
          <v-card-title>导出记忆快照</v-card-title>
          <v-card-text>
            <p class="text-body-2 mb-3">导出中央记忆库的完整快照（记录 + 标签 + 关联），用于备份和迁移。</p>
            <v-btn color="primary" @click="doExport" :loading="exporting" prepend-icon="mdi-download">
              生成并下载快照
            </v-btn>
          </v-card-text>
        </v-card>
      </v-col>

      <v-col cols="12" md="6">
        <v-card>
          <v-card-title>导入记忆快照</v-card-title>
          <v-card-text>
            <p class="text-body-2 mb-3">从 JSON 文件导入记忆数据。已存在的记忆将被跳过。</p>
            <v-file-input
              v-model="importFile"
              label="选择 JSON 文件"
              accept=".json"
              density="compact"
              variant="outlined"
              prepend-icon="mdi-upload"
              class="mb-3"
            />
            <v-btn
              color="secondary"
              @click="doImport"
              :loading="importing"
              :disabled="!selectedImportFile"
              prepend-icon="mdi-import"
            >
              执行导入
            </v-btn>
          </v-card-text>
        </v-card>
      </v-col>
    </v-row>

    <v-alert v-if="importResult" :type="importResult.success ? 'success' : 'error'" class="mt-4">
      <template v-if="importResult.success">
        导入完成：新增 {{ importResult.inserted }} / 跳过 {{ importResult.skipped }} / 失败 {{ importResult.failed }}
      </template>
      <template v-else>
        导入失败：{{ importResult.error }}
      </template>
    </v-alert>
  </div>
</template>

<script setup lang="ts">
import { computed, ref } from 'vue'
import { useBridge } from '@/composables/useBridge'

const { apiGet, apiPost, download } = useBridge()

const exporting = ref(false)
const importing = ref(false)
const importFile = ref<File | File[] | null>(null)
const importResult = ref<any>(null)
const selectedImportFile = computed(() => {
  if (Array.isArray(importFile.value)) return importFile.value[0] ?? null
  return importFile.value ?? null
})

async function doExport() {
  exporting.value = true
  try {
    const now = new Date().toISOString().replace(/[:.]/g, '-').slice(0, 19)
    await download('export', {}, `memory_snapshot_${now}.json`)
  } catch (e) {
    console.error('导出失败:', e)
  } finally {
    exporting.value = false
  }
}

async function doImport() {
  const file = selectedImportFile.value
  if (!file) return
  importing.value = true
  importResult.value = null
  try {
    const text = await file.text()
    const payload = JSON.parse(text)
    importResult.value = await apiPost('import', payload)
  } catch (e: any) {
    importResult.value = { success: false, error: e.message || '导入失败' }
  } finally {
    importing.value = false
  }
}
</script>
