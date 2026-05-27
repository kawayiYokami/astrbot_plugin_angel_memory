<template>
  <div>
    <h2 class="text-h5 mb-4">🛠️ 维护状态</h2>

    <v-row v-if="loading">
      <v-col cols="12" class="text-center">
        <v-progress-circular indeterminate color="primary" />
      </v-col>
    </v-row>

    <template v-else>
      <v-card class="mb-4">
        <v-card-title>maintenance_state.json</v-card-title>
        <v-card-text>
          <template v-if="state">
            <pre class="pa-3 rounded" style="background: #1a1a2e; overflow-x: auto; white-space: pre-wrap; font-size: 13px;">{{ JSON.stringify(state, null, 2) }}</pre>
          </template>
          <v-alert v-else type="info" variant="tonal">
            未找到维护状态文件或文件为空。
          </v-alert>
        </v-card-text>
      </v-card>

      <v-card>
        <v-card-title>备份文件</v-card-title>
        <v-card-text>
          <v-data-table
            v-if="backups.length"
            :headers="backupHeaders"
            :items="backups"
            density="compact"
            :items-per-page="10"
          >
            <template v-slot:item.size="{ item }">
              {{ formatSize(item.size) }}
            </template>
            <template v-slot:item.modified_at="{ item }">
              {{ formatTime(item.modified_at) }}
            </template>
            <template v-slot:item.actions="{ item }">
              <v-btn
                icon="mdi-download"
                size="x-small"
                variant="text"
                color="primary"
                @click="downloadBackup(item.name)"
                :loading="downloadingFile === item.name"
              />
            </template>
          </v-data-table>
          <v-alert v-else type="info" variant="tonal">暂无备份文件。</v-alert>
        </v-card-text>
      </v-card>
    </template>
  </div>
</template>

<script setup lang="ts">
import { ref, onMounted } from 'vue'
import { useBridge } from '@/composables/useBridge'

const { apiGet, download } = useBridge()

const loading = ref(true)
const state = ref<any>(null)
const backups = ref<any[]>([])
const downloadingFile = ref('')

const backupHeaders = [
  { title: '文件名', key: 'name' },
  { title: '大小', key: 'size', width: '100px' },
  { title: '修改时间', key: 'modified_at', width: '180px' },
  { title: '操作', key: 'actions', width: '70px', sortable: false },
]

function formatSize(bytes: number): string {
  if (bytes < 1024) return `${bytes} B`
  if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`
  return `${(bytes / 1024 / 1024).toFixed(1)} MB`
}

function formatTime(ts: number): string {
  return new Date(ts * 1000).toLocaleString('zh-CN')
}

async function downloadBackup(filename: string) {
  downloadingFile.value = filename
  try {
    await download('maintenance/download-backup', { filename }, filename)
  } catch (e) {
    console.error('下载备份失败:', e)
  } finally {
    downloadingFile.value = ''
  }
}

onMounted(async () => {
  try {
    const data: any = await apiGet('maintenance')
    state.value = data.state
    backups.value = data.backups || []
  } catch (e) {
    console.error('加载维护状态失败:', e)
  } finally {
    loading.value = false
  }
})
</script>
