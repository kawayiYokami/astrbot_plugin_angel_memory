<template>
  <div class="d-flex flex-column ga-4">
    <v-row>
      <!-- 导出 -->
      <v-col cols="12" md="6">
        <v-card rounded="md" class="h-100">
          <v-card-text class="config-section pb-2">
            <div class="config-title">导出记忆快照</div>
            <div class="config-hint">把整份记忆库（记录、标签、关联）打包成一个文件，用于备份和迁移</div>
          </v-card-text>
          <v-card-text class="pt-0">
            <v-btn color="primary" variant="flat" prepend-icon="lucide:download" :loading="exporting" @click="doExport">
              生成并下载快照
            </v-btn>
          </v-card-text>
        </v-card>
      </v-col>

      <!-- 导入 -->
      <v-col cols="12" md="6">
        <v-card rounded="md" class="h-100">
          <v-card-text class="config-section pb-2">
            <div class="config-title">导入记忆快照</div>
            <div class="config-hint">从快照文件恢复记忆；已经存在的记忆会自动跳过，不会重复写入</div>
          </v-card-text>
          <v-card-text class="pt-0 d-flex flex-column ga-3">
            <v-file-input
              v-model="importFile"
              label="选择快照文件"
              accept=".json"
              prepend-icon="lucide:upload"
              density="compact"
              hide-details
              variant="outlined"
              @update:model-value="importResult = null"
            />
            <div>
              <v-btn
                variant="outlined"
                prepend-icon="lucide:file-input"
                :loading="importing"
                :disabled="!importFile"
                @click="doImport"
              >
                执行导入
              </v-btn>
            </div>
          </v-card-text>
        </v-card>
      </v-col>
    </v-row>

    <v-alert
      v-if="importResult"
      :type="importResult.success ? 'success' : 'error'"
      :title="importResult.success ? '导入完成' : '导入失败'"
      variant="tonal"
    >
      <template v-if="importResult.success">
        新增 {{ importResult.inserted }} 条，跳过 {{ importResult.skipped }} 条，失败 {{ importResult.failed }} 条
      </template>
      <template v-else>
        {{ importResult.error }}
      </template>
    </v-alert>
  </div>
</template>

<script setup lang="ts">
import { ref } from 'vue'
import { useBridge } from '@/composables/useBridge'
import { toast } from '@/composables/useToast'

const { apiPost, download } = useBridge()

const exporting = ref(false)
const importing = ref(false)
const importFile = ref<File | File[] | null>(null)
const importResult = ref<any>(null)

/** v-file-input 单选时给 File，多选时给数组，这里统一成单个文件 */
function pickedFile(): File | null {
  const value = importFile.value
  if (!value) return null
  if (Array.isArray(value)) return value[0] ?? null
  return value
}

async function doExport() {
  exporting.value = true
  try {
    const now = new Date().toISOString().replace(/[:.]/g, '-').slice(0, 19)
    await download('export', {}, `记忆快照_${now}.json`)
    toast.success('快照已开始下载')
  } catch (e) {
    toast.error(`导出失败: ${(e as Error).message}`)
  } finally {
    exporting.value = false
  }
}

async function doImport() {
  const file = pickedFile()
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

<style scoped>
.config-title {
  font-size: 0.875rem;
  font-weight: 500;
  color: rgb(var(--v-theme-primaryText));
}
.config-hint {
  font-size: 0.75rem;
  color: rgb(var(--v-theme-secondaryText));
  margin-top: 2px;
  line-height: 1.6;
}
</style>
