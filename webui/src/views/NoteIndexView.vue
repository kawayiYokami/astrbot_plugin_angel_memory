<template>
  <div>
    <n-card embedded>
      <n-space vertical size="medium">
        <n-input
          v-model:value="keyword"
          placeholder="关键词搜索（路径 / 标题 / tags）"
          clearable
          @keyup.enter="loadData"
        >
          <template #suffix>
            <Icon icon="lucide:search" @click="loadData" style="cursor: pointer" />
          </template>
        </n-input>

        <n-data-table
          remote
          :columns="columns"
          :data="items"
          :loading="loading"
          :pagination="pagination"
          :row-key="(row: any) => row.note_short_id"
          @update:page="onPageChange"
          @update:page-size="onPageSizeChange"
        />
      </n-space>
    </n-card>

    <!-- 详情对话框 -->
    <n-modal
      v-model:show="detailDialog"
      preset="card"
      title="笔记索引详情"
      style="width: 640px"
      :bordered="false"
    >
      <pre class="json-pre">{{ selectedItem ? JSON.stringify(selectedItem, null, 2) : '' }}</pre>
    </n-modal>
  </div>
</template>

<script setup lang="ts">
import { h, ref, onMounted, computed } from 'vue'
import type { DataTableColumns } from 'naive-ui'
import { NButton, NCard, NDataTable, NInput, NModal, NSpace, NTag } from 'naive-ui'
import { Icon } from '@iconify/vue'
import { useBridge } from '@/composables/useBridge'
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

function buildHeading(item: any): string {
  const parts = []
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

const pagination = computed(() => ({
  page: page.value,
  pageSize: pageSize.value,
  itemCount: total.value,
  showSizePicker: true,
  pageSizes: [10, 20, 50, 100],
  prefix: ({ itemCount }: { itemCount?: number }) => `共 ${itemCount ?? 0} 条`,
}))

const columns: DataTableColumns<any> = [
  { title: 'Short ID', key: 'note_short_id', width: 90 },
  {
    title: '文件路径',
    key: 'source_file_path',
    ellipsis: { tooltip: true },
  },
  {
    title: '标题',
    key: 'heading',
    ellipsis: { tooltip: true },
    render: row => buildHeading(row),
  },
  {
    title: 'Tags',
    key: 'tags_text',
    width: 220,
    render: row => {
      const tags = parseTags(row.tags_text)
      if (!tags.length) return ''
      return h(
        'div',
        { style: 'display:flex; flex-wrap:wrap; gap:4px;' },
        tags.map(tag =>
          h(NTag, { size: 'small', bordered: false }, { default: () => tag }),
        ),
      )
    },
  },
  { title: '行数', key: 'total_lines', width: 70 },
  {
    title: '操作',
    key: 'actions',
    width: 70,
    render: row =>
      h(
        NButton,
        { size: 'tiny', quaternary: true, onClick: () => showDetail(row) },
        { icon: () => h(Icon, { icon: 'lucide:eye' }) },
      ),
  },
]

function onPageChange(p: number) {
  page.value = p
  loadData()
}

function onPageSizeChange(s: number) {
  pageSize.value = s
  page.value = 1
  loadData()
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
    console.error('加载笔记索引失败:', e)
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
  background: rgba(0, 0, 0, 0.3);
  border-radius: 6px;
  padding: 12px;
  font-size: 12px;
  white-space: pre-wrap;
  word-break: break-all;
}
</style>
