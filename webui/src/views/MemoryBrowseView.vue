<template>
  <div>
    <n-card embedded class="mb-4">
      <n-space>
        <n-select
          v-model:value="scope"
          :options="scopeOptions"
          placeholder="Scope 过滤"
          clearable
          style="width: 200px"
          @update:value="loadData"
        />
        <n-input
          v-model:value="keyword"
          placeholder="关键词搜索（judgment / reasoning / tags）"
          clearable
          style="flex: 1"
          @keyup.enter="loadData"
        >
          <template #suffix>
            <Icon icon="lucide:search" @click="loadData" style="cursor: pointer" />
          </template>
        </n-input>
      </n-space>
    </n-card>

    <n-card embedded>
      <n-data-table
        remote
        :columns="columns"
        :data="items"
        :loading="loading"
        :pagination="pagination"
        :row-key="(row: any) => row.id"
        @update:page="onPageChange"
        @update:page-size="onPageSizeChange"
      />
    </n-card>

    <!-- 详情对话框 -->
    <n-modal
      v-model:show="detailDialog"
      preset="card"
      title="记忆详情"
      style="width: 680px"
      :bordered="false"
    >
      <template v-if="selectedItem">
        <n-descriptions label-placement="left" :column="1" size="small">
          <n-descriptions-item label="ID">{{ selectedItem.id }}</n-descriptions-item>
          <n-descriptions-item label="类型">{{ selectedItem.memory_type }}</n-descriptions-item>
          <n-descriptions-item label="Judgment">{{ selectedItem.judgment }}</n-descriptions-item>
          <n-descriptions-item label="Reasoning">{{ selectedItem.reasoning || '-' }}</n-descriptions-item>
          <n-descriptions-item label="Tags">
            <n-space size="small">
              <n-tag
                v-for="tag in parseTags(selectedItem.tags)"
                :key="tag"
                size="small"
                :bordered="false"
              >
                {{ tag }}
              </n-tag>
            </n-space>
          </n-descriptions-item>
          <n-descriptions-item label="强度">{{ selectedItem.strength }}</n-descriptions-item>
          <n-descriptions-item label="主动">{{ selectedItem.is_active ? '是' : '否' }}</n-descriptions-item>
          <n-descriptions-item label="Scope">{{ selectedItem.memory_scope }}</n-descriptions-item>
          <n-descriptions-item label="创建时间">{{ formatTime(selectedItem.created_at) }}</n-descriptions-item>
          <n-descriptions-item label="更新时间">{{ formatTime(selectedItem.updated_at) }}</n-descriptions-item>
        </n-descriptions>
      </template>
    </n-modal>

    <!-- 删除确认 -->
    <n-modal
      v-model:show="deleteDialog"
      preset="dialog"
      type="warning"
      title="确认删除"
      :positive-text="'删除'"
      negative-text="取消"
      :positive-button-props="{ type: 'error' }"
      @positive-click="doDelete"
    >
      确定要删除这条记忆吗？此操作不可撤销。
      <div v-if="deleteTarget" class="delete-preview">{{ deleteTarget.judgment }}</div>
    </n-modal>
  </div>
</template>

<script setup lang="ts">
import { h, ref, computed, onMounted } from 'vue'
import type { DataTableColumns } from 'naive-ui'
import {
  NButton,
  NCard,
  NDataTable,
  NDescriptions,
  NDescriptionsItem,
  NInput,
  NModal,
  NSelect,
  NSpace,
  NTag,
} from 'naive-ui'
import { Icon } from '@iconify/vue'
import { useBridge } from '@/composables/useBridge'
import { formatTime, parseTags } from '@/utils/format'

const { apiGet, apiPost } = useBridge()

const loading = ref(false)
const items = ref<any[]>([])
const total = ref(0)
const page = ref(1)
const pageSize = ref(20)
const scope = ref<string | null>(null)
const keyword = ref('')
const scopeOptions = ref<{ label: string; value: string }[]>([])

const detailDialog = ref(false)
const selectedItem = ref<any>(null)
const deleteDialog = ref(false)
const deleteTarget = ref<any>(null)
const deleting = ref(false)

function strengthTagType(s: number): 'success' | 'primary' | 'warning' | 'error' {
  if (s >= 80) return 'success'
  if (s >= 50) return 'primary'
  if (s >= 30) return 'warning'
  return 'error'
}

const pagination = computed(() => ({
  page: page.value,
  pageSize: pageSize.value,
  itemCount: total.value,
  showSizePicker: true,
  pageSizes: [10, 20, 50, 100],
  prefix: ({ itemCount }: { itemCount: number }) => `共 ${itemCount} 条`,
}))

const columns: DataTableColumns<any> = [
  { title: '类型', key: 'memory_type', width: 90 },
  {
    title: 'Judgment',
    key: 'judgment',
    ellipsis: { tooltip: true },
  },
  {
    title: 'Tags',
    key: 'tags',
    width: 220,
    render: row => {
      const tags = parseTags(row.tags)
      if (!tags.length) return ''
      return h(
        'div',
        { style: 'display:flex; flex-wrap:wrap; gap:4px;' },
        tags.map(tag =>
          h(NTag, { size: 'small', type: 'primary', bordered: false }, { default: () => tag }),
        ),
      )
    },
  },
  {
    title: '强度',
    key: 'strength',
    width: 80,
    render: row =>
      h(
        NTag,
        { size: 'small', type: strengthTagType(row.strength), bordered: false },
        { default: () => row.strength },
      ),
  },
  {
    title: '主动',
    key: 'is_active',
    width: 60,
    render: row =>
      h(Icon, {
        icon: row.is_active ? 'lucide:star' : 'lucide:star',
        style: { color: row.is_active ? '#f0a020' : '#888', fontSize: '14px' },
      }),
  },
  {
    title: '创建时间',
    key: 'created_at',
    width: 140,
    render: row => formatTime(row.created_at),
  },
  {
    title: '操作',
    key: 'actions',
    width: 90,
    render: row =>
      h('div', { style: 'display:flex; gap:4px;' }, [
        h(
          NButton,
          { size: 'tiny', quaternary: true, onClick: () => showDetail(row) },
          { icon: () => h(Icon, { icon: 'lucide:eye' }) },
        ),
        h(
          NButton,
          { size: 'tiny', quaternary: true, type: 'error', onClick: () => confirmDelete(row) },
          { icon: () => h(Icon, { icon: 'lucide:trash-2' }) },
        ),
      ]),
  },
]

function showDetail(item: any) {
  selectedItem.value = item
  detailDialog.value = true
}

function confirmDelete(item: any) {
  deleteTarget.value = item
  deleteDialog.value = true
}

async function doDelete() {
  if (!deleteTarget.value || deleting.value) return
  deleting.value = true
  try {
    await apiPost('memories/delete', { id: deleteTarget.value.id })
    deleteDialog.value = false
    await loadData()
  } catch (e) {
    console.error('删除失败:', e)
  } finally {
    deleting.value = false
  }
}

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
    const data: any = await apiGet('memories', {
      scope: scope.value || '',
      keyword: keyword.value || '',
      page: page.value,
      page_size: pageSize.value,
    })
    items.value = data.items || []
    total.value = data.total || 0
  } catch (e) {
    console.error('加载记忆失败:', e)
  } finally {
    loading.value = false
  }
}

onMounted(async () => {
  try {
    const ov: any = await apiGet('overview')
    scopeOptions.value = (ov.scopes || []).map((s: string) => ({ label: s, value: s }))
  } catch (e) {
    /* ignore */
  }
  await loadData()
})
</script>

<style scoped>
.delete-preview {
  margin-top: 10px;
  font-size: 13px;
  opacity: 0.65;
}
</style>
