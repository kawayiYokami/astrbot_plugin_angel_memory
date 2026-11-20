<template>
  <div class="d-flex flex-column ga-4">
    <v-card rounded="md">
      <v-card-text class="d-flex flex-wrap align-center ga-3">
        <v-text-field
          v-model="keyword"
          label="搜索文件路径、标题或标签"
          density="compact"
          hide-details
          variant="outlined"
          prepend-inner-icon="lucide:search"
          clearable
          class="flex-grow-1"
          style="min-width: 220px;"
          @keyup.enter="reload"
        />
        <v-btn variant="flat" color="primary" prepend-icon="lucide:search" @click="reload">搜索</v-btn>
      </v-card-text>
    </v-card>

    <v-card rounded="md">
      <v-data-table-server
        :headers="headers"
        :items="items"
        :items-length="total"
        :loading="loading"
        v-model:page="page"
        v-model:items-per-page="pageSize"
        :items-per-page-options="[10, 20, 50, 100]"
        item-value="note_short_id"
        density="comfortable"
        @update:options="onOptions"
      >
        <template #item.source_file_path="{ item }">
          <span class="text-truncate d-inline-block" style="max-width: 420px;" :title="item.source_file_path">
            {{ item.source_file_path }}
          </span>
        </template>
        <template #item.heading="{ item }">
          <span class="text-truncate d-inline-block" style="max-width: 320px;" :title="buildHeading(item)">
            {{ buildHeading(item) }}
          </span>
        </template>
        <template #item.tags_text="{ item }">
          <div class="d-flex flex-wrap ga-1 py-1">
            <v-chip v-for="tag in parseTags(item.tags_text)" :key="tag" size="x-small" variant="tonal">
              {{ tag }}
            </v-chip>
          </div>
        </template>
        <template #item.actions="{ item }">
          <v-btn
            variant="text"
            size="small"
            icon="lucide:eye"
            :aria-label="'查看笔记索引 ' + item.note_short_id"
            @click="showDetail(item)"
          />
        </template>
        <template #no-data>
          <v-empty-state icon="lucide:notebook-text" title="还没有笔记索引" text="导入笔记后会自动建立索引" />
        </template>
      </v-data-table-server>
    </v-card>

    <v-dialog v-model="detailDialog" max-width="640">
      <v-card rounded="md">
        <v-card-title class="text-subtitle-1 font-weight-medium">笔记索引详情</v-card-title>
        <v-card-text>
          <pre class="json-pre">{{ selectedItem ? JSON.stringify(selectedItem, null, 2) : '' }}</pre>
        </v-card-text>
        <v-card-actions class="px-4 pb-3">
          <v-spacer />
          <v-btn variant="text" @click="detailDialog = false">关闭</v-btn>
        </v-card-actions>
      </v-card>
    </v-dialog>
  </div>
</template>

<script setup lang="ts">
import { ref, onMounted } from 'vue'
import { useBridge } from '@/composables/useBridge'
import { toast } from '@/composables/useToast'
import { parseTags } from '@/utils/format'

const { apiGet } = useBridge()

const loading = ref(false)
const items = ref<any[]>([])
const total = ref(0)
const page = ref(1)
const pageSize = ref(20)
const keyword = ref('')

const detailDialog = ref(false)
const selectedItem = ref<any>(null)

const headers = [
  { title: '编号', key: 'note_short_id', width: 90, sortable: false },
  { title: '文件路径', key: 'source_file_path', sortable: false },
  { title: '标题', key: 'heading', sortable: false },
  { title: '标签', key: 'tags_text', width: 220, sortable: false },
  { title: '行数', key: 'total_lines', width: 80, sortable: false },
  { title: '操作', key: 'actions', width: 80, sortable: false },
]

function buildHeading(item: any): string {
  const parts: string[] = []
  for (let i = 1; i <= 6; i++) {
    const hh = item[`heading_h${i}`]
    if (hh) parts.push(hh)
  }
  return parts.join(' / ') || '(无标题)'
}

function showDetail(item: any) {
  selectedItem.value = item
  detailDialog.value = true
}

function reload() {
  page.value = 1
  void loadData()
}

// 分页/页大小变化统一由表格的 options 事件驱动，避免重复请求
function onOptions(options: { page: number; itemsPerPage: number }) {
  page.value = options.page
  pageSize.value = options.itemsPerPage
  void loadData()
}

async function loadData() {
  loading.value = true
  try {
    const data: any = await apiGet('notes', {
      keyword: keyword.value,
      page: page.value,
      page_size: pageSize.value,
    })
    items.value = data.items || []
    total.value = data.total || 0
  } catch (e) {
    toast.error(`加载笔记索引失败: ${(e as Error).message}`)
  } finally {
    loading.value = false
  }
}

onMounted(() => loadData())
</script>

<style scoped>
.json-pre {
  max-height: 60vh;
  overflow: auto;
  background: rgba(var(--v-theme-on-surface), 0.06);
  border-radius: 6px;
  padding: 12px;
  font-size: 12px;
  white-space: pre-wrap;
  word-break: break-all;
  margin: 0;
}
</style>
