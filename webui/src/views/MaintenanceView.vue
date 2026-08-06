<template>
  <div>
    <n-spin :show="loading">
      <template v-if="!loading">
        <!-- 维护状态 JSON -->
        <n-card title="maintenance_state.json" embedded class="mb-4">
          <template v-if="state">
            <pre class="code-block">{{ JSON.stringify(state, null, 2) }}</pre>
          </template>
          <n-empty v-else description="未找到维护状态文件或文件为空" size="small" />
        </n-card>

        <!-- 备份文件 -->
        <n-card title="备份文件" embedded>
          <template v-if="backups.length">
            <n-data-table
              :columns="backupColumns"
              :data="backups"
              :pagination="{ pageSize: 10 }"
              :bordered="false"
              size="small"
            />
          </template>
          <n-empty v-else description="暂无备份文件" size="small" />
        </n-card>
      </template>
    </n-spin>
  </div>
</template>

<script setup lang="ts">
import { h, ref, onMounted } from 'vue'
import type { DataTableColumns } from 'naive-ui'
import { NButton, NIcon, useMessage } from 'naive-ui'
import { Icon } from '@iconify/vue'
import { useBridge } from '@/composables/useBridge'
import { formatSize, formatTime } from '@/utils/format'

const { apiGet, download } = useBridge()
const message = useMessage()

const loading = ref(true)
const state = ref<any>(null)
const backups = ref<any[]>([])
const downloadingFile = ref('')

const backupColumns: DataTableColumns<any> = [
  { title: '文件名', key: 'name' },
  { title: '大小', key: 'size', width: 100, render: row => formatSize(row.size) },
  { title: '修改时间', key: 'modified_at', width: 170, render: row => formatTime(row.modified_at) },
  {
    title: '操作',
    key: 'actions',
    width: 70,
    render: row =>
      h(
        NButton,
        {
          size: 'small',
          quaternary: true,
          type: 'primary',
          loading: downloadingFile.value === row.name,
          onClick: () => downloadBackup(row.name),
        },
        { icon: () => h(NIcon, null, { default: () => h(Icon, { icon: 'lucide:download' }) }) },
      ),
  },
]

async function downloadBackup(filename: string) {
  downloadingFile.value = filename
  try {
    await download('maintenance/download-backup', { filename }, filename)
  } catch (e) {
    message.error(`下载备份失败: ${(e as Error).message}`)
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
    message.error(`加载维护状态失败: ${(e as Error).message}`)
  } finally {
    loading.value = false
  }
})
</script>

<style scoped>
.mb-4 {
  margin-bottom: 16px;
}

.code-block {
  background: #1a1a2e;
  color: #e0e0e0;
  border-radius: 6px;
  padding: 12px;
  overflow-x: auto;
  white-space: pre-wrap;
  font-size: 13px;
  line-height: 1.5;
}
</style>
