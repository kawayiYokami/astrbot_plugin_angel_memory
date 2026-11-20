<template>
  <div class="d-flex flex-column ga-4">
    <div v-if="loading" class="d-flex justify-center py-12">
      <v-progress-circular indeterminate color="primary" />
    </div>

    <template v-else>
      <!-- 维护状态 -->
      <v-card rounded="md">
        <v-card-text class="config-section pb-2">
          <div class="config-title">维护状态</div>
          <div class="config-hint">记忆整理任务上次跑到哪、有没有异常，都记在这里</div>
        </v-card-text>
        <v-card-text class="pt-0">
          <template v-if="stateRows.length">
            <div
              v-for="row in stateRows"
              :key="row.label"
              class="d-flex justify-space-between align-center ga-4 py-2 state-row"
            >
              <span class="text-body-2 text-medium-emphasis text-no-wrap flex-shrink-0">{{ row.label }}</span>
              <span class="text-body-2 text-right">{{ row.value }}</span>
            </div>
          </template>
          <v-empty-state
            v-else
            icon="lucide:wrench"
            title="还没有维护记录"
            text="记忆整理跑过一次后会出现"
          />
        </v-card-text>
      </v-card>

      <!-- 备份文件 -->
      <v-card rounded="md">
        <v-card-text class="config-section pb-2">
          <div class="config-title">备份文件</div>
          <div class="config-hint">整理前自动留的备份，需要时可以下载回来</div>
        </v-card-text>
        <v-card-text class="pt-0">
          <v-data-table
            :headers="backupHeaders"
            :items="backups"
            :items-per-page="10"
            item-value="name"
            density="comfortable"
          >
            <template #item.size="{ item }">
              {{ formatSize(item.size) }}
            </template>
            <template #item.modified_at="{ item }">
              <span class="text-caption text-medium-emphasis">{{ formatTime(item.modified_at) }}</span>
            </template>
            <template #item.actions="{ item }">
              <v-btn
                variant="text"
                size="small"
                color="primary"
                icon="lucide:download"
                :loading="downloadingFile === item.name"
                :aria-label="'下载备份 ' + item.name"
                @click="downloadBackup(item.name)"
              />
            </template>
            <template #no-data>
              <v-empty-state icon="lucide:layers" title="暂无备份文件" text="记忆整理执行前会自动备份" />
            </template>
          </v-data-table>
        </v-card-text>
      </v-card>
    </template>
  </div>
</template>

<script setup lang="ts">
import { computed, ref, onMounted } from 'vue'
import { useBridge } from '@/composables/useBridge'
import { toast } from '@/composables/useToast'
import { formatSize, formatTime } from '@/utils/format'

const { apiGet, download } = useBridge()

const loading = ref(true)
const state = ref<any>(null)
const backups = ref<any[]>([])
const downloadingFile = ref('')

/** 秒级浮点时间戳 → 2026-11-20 04:00（手动拼，避免 zh-CN 默认用斜杠）；空值返回「暂无」 */
function formatStamp(value: unknown): string {
  const num = Number(value)
  if (!Number.isFinite(num) || num <= 0) return '暂无'
  const d = new Date(num * 1000)
  const pad = (n: number) => String(n).padStart(2, '0')
  return `${d.getFullYear()}-${pad(d.getMonth() + 1)}-${pad(d.getDate())} ${pad(d.getHours())}:${pad(d.getMinutes())}`
}

/** 日期串（20261120）→ 2026-11-20；格式不认识就原样显示 */
function formatDayKey(value: unknown): string {
  const text = String(value ?? '').trim()
  if (!/^\d{8}$/.test(text)) return text || '暂无'
  return `${text.slice(0, 4)}-${text.slice(4, 6)}-${text.slice(6, 8)}`
}

// 状态文件里的键名是内部命名，界面只呈现用户看得懂的三行
const stateRows = computed(() => {
  const s = state.value
  if (!s || typeof s !== 'object') return []
  const rows = [
    { label: '上次整理', value: formatStamp(s.last_updated_at) },
    { label: '上次备份', value: formatDayKey(s.daily_json_backup_last_day) },
    { label: '上次清理', value: formatStamp(s.cleanup_last_sleep_at) },
  ]
  // 三项都没值时按「无记录」处理，让空态出来
  return rows.some((row) => row.value !== '暂无') ? rows : []
})

const backupHeaders = [
  { title: '文件名', key: 'name', sortable: false },
  { title: '大小', key: 'size', width: 110, sortable: false },
  { title: '修改时间', key: 'modified_at', width: 180, sortable: false },
  { title: '操作', key: 'actions', width: 80, sortable: false },
]

async function downloadBackup(filename: string) {
  downloadingFile.value = filename
  try {
    await download('maintenance/download-backup', { filename }, filename)
  } catch (e) {
    toast.error(`下载备份失败: ${(e as Error).message}`)
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
    toast.error(`加载维护状态失败: ${(e as Error).message}`)
  } finally {
    loading.value = false
  }
})
</script>

<style scoped>
.state-row + .state-row {
  border-top: 1px solid rgba(var(--v-border-color), 0.12);
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
