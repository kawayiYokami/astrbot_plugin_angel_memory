<template>
  <div>
    <n-grid :cols="2" :x-gap="16" responsive="screen">
      <!-- 导出 -->
      <n-grid-item span="2 m:1">
        <n-card title="导出记忆快照" embedded>
          <p class="desc">导出中央记忆库的完整快照（记录 + 标签 + 关联），用于备份和迁移。</p>
          <n-button type="primary" :loading="exporting" @click="doExport">
            <template #icon><Icon icon="lucide:download" /></template>
            生成并下载快照
          </n-button>
        </n-card>
      </n-grid-item>

      <!-- 导入 -->
      <n-grid-item span="2 m:1">
        <n-card title="导入记忆快照" embedded>
          <p class="desc">从 JSON 文件导入记忆数据。已存在的记忆将被跳过。</p>
          <n-upload
            accept=".json"
            :max="1"
            :default-upload="false"
            :show-file-list="true"
            @change="onFileChange"
          >
            <n-button>
              <template #icon><Icon icon="lucide:upload" /></template>
              选择 JSON 文件
            </n-button>
          </n-upload>
          <n-button
            type="secondary"
            class="mt-3"
            :loading="importing"
            :disabled="!importFile"
            @click="doImport"
          >
            <template #icon><Icon icon="lucide:file-input" /></template>
            执行导入
          </n-button>
        </n-card>
      </n-grid-item>
    </n-grid>

    <n-alert
      v-if="importResult"
      class="mt-4"
      :type="importResult.success ? 'success' : 'error'"
      :title="importResult.success ? '导入完成' : '导入失败'"
    >
      <template v-if="importResult.success">
        新增 {{ importResult.inserted }} / 跳过 {{ importResult.skipped }} / 失败 {{ importResult.failed }}
      </template>
      <template v-else>
        {{ importResult.error }}
      </template>
    </n-alert>
  </div>
</template>

<script setup lang="ts">
import { ref } from 'vue'
import { useBridge } from '@/composables/useBridge'

const { apiPost, download } = useBridge()

const exporting = ref(false)
const importing = ref(false)
const importFile = ref<File | null>(null)
const importResult = ref<any>(null)

function onFileChange(options: { file: { file?: File } }) {
  importFile.value = options.file.file ?? null
  importResult.value = null
}

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
  if (!importFile.value) return
  importing.value = true
  importResult.value = null
  try {
    const text = await importFile.value.text()
    const payload = JSON.parse(text)
    importResult.value = await apiPost('import', payload)
  } catch (e: any) {
    importResult.value = { success: false, error: e.message || '导入失败' }
  } finally {
    importing.value = false
  }
}
</script>

<style scoped>
.desc {
  font-size: 13px;
  opacity: 0.75;
  margin-bottom: 12px;
}

.mt-3 {
  margin-top: 12px;
}
</style>
