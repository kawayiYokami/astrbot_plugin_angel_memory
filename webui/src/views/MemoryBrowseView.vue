<template>
  <div class="d-flex flex-column ga-4">
    <!-- 过滤条 -->
    <v-card rounded="md">
      <v-card-text class="d-flex flex-wrap align-center ga-3">
        <v-select
          v-model="scope"
          :items="scopeOptions"
          label="分类域"
          clearable
          density="compact"
          hide-details
          variant="outlined"
          style="max-width: 200px;"
          @update:model-value="reload"
        />
        <v-text-field
          v-model="keyword"
          label="搜索内容或标签"
          density="compact"
          hide-details
          variant="outlined"
          prepend-inner-icon="lucide:search"
          clearable
          class="flex-grow-1"
          style="min-width: 200px;"
          @keyup.enter="reload"
        />
        <v-btn variant="flat" color="primary" prepend-icon="lucide:search" @click="reload">搜索</v-btn>
      </v-card-text>
    </v-card>

    <!-- 记忆列表 -->
    <v-card rounded="md">
      <v-data-table-server
        :headers="headers"
        :items="items"
        :items-length="total"
        :loading="loading"
        v-model:page="page"
        v-model:items-per-page="pageSize"
        :items-per-page-options="[10, 20, 50, 100]"
        item-value="id"
        density="comfortable"
        @update:options="onOptions"
      >
        <template #item.memory_type="{ item }">
          <v-chip size="small" variant="tonal">{{ item.memory_type }}</v-chip>
        </template>
        <template #item.judgment="{ item }">
          <span class="text-truncate d-inline-block" style="max-width: 420px;" :title="item.judgment">
            {{ item.judgment }}
          </span>
        </template>
        <template #item.tags="{ item }">
          <div class="d-flex flex-wrap ga-1 py-1">
            <v-chip v-for="tag in parseTags(item.tags)" :key="tag" size="x-small" color="primary" variant="tonal">
              {{ tag }}
            </v-chip>
          </div>
        </template>
        <template #item.strength="{ item }">
          <v-chip size="small" :color="strengthColor(item.strength)" variant="tonal">{{ item.strength }}</v-chip>
        </template>
        <template #item.is_active="{ item }">
          <v-icon
            icon="lucide:star"
            size="16"
            :color="item.is_active ? 'warning' : undefined"
            :style="item.is_active ? '' : 'opacity:.35'"
            :aria-label="item.is_active ? '主动记忆' : '非主动记忆'"
          />
        </template>
        <template #item.created_at="{ item }">
          <span class="text-caption text-medium-emphasis">{{ formatTime(item.created_at) }}</span>
        </template>
        <template #item.actions="{ item }">
          <div class="d-flex ga-1">
            <v-btn
              variant="text"
              size="small"
              icon="lucide:eye"
              :aria-label="'查看记忆 ' + item.id"
              @click="showDetail(item)"
            />
            <v-btn
              variant="text"
              size="small"
              icon="lucide:trash-2"
              color="error"
              :aria-label="'删除记忆 ' + item.id"
              @click="confirmDelete(item)"
            />
          </div>
        </template>
        <template #no-data>
          <v-empty-state icon="lucide:brain" title="没有找到记忆" text="换个关键词或分类域再试" />
        </template>
      </v-data-table-server>
    </v-card>

    <!-- 详情 -->
    <v-dialog v-model="detailDialog" max-width="680">
      <v-card rounded="md">
        <v-card-title class="text-subtitle-1 font-weight-medium">记忆详情</v-card-title>
        <v-card-text v-if="selectedItem" class="d-flex flex-column ga-3">
          <div v-for="row in detailRows" :key="row.label" class="d-flex justify-space-between ga-4">
            <span class="text-body-2 text-medium-emphasis flex-shrink-0">{{ row.label }}</span>
            <span class="text-body-2 text-right">{{ row.value }}</span>
          </div>
          <div class="d-flex justify-space-between ga-4">
            <span class="text-body-2 text-medium-emphasis flex-shrink-0">标签</span>
            <div class="d-flex flex-wrap ga-1 justify-end">
              <v-chip v-for="tag in parseTags(selectedItem.tags)" :key="tag" size="x-small" variant="tonal">
                {{ tag }}
              </v-chip>
              <span v-if="!parseTags(selectedItem.tags).length" class="text-body-2">-</span>
            </div>
          </div>
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
import { ref, computed, onMounted } from 'vue'
import { useBridge } from '@/composables/useBridge'
import { toast } from '@/composables/useToast'
import { confirmDialog } from '@/composables/useConfirm'
import { formatTime, parseTags } from '@/utils/format'

const { apiGet, apiPost } = useBridge()

const loading = ref(false)
const items = ref<any[]>([])
const total = ref(0)
const page = ref(1)
const pageSize = ref(20)
const scope = ref<string | null>(null)
const keyword = ref('')
const scopeOptions = ref<{ title: string; value: string }[]>([])

const detailDialog = ref(false)
const selectedItem = ref<any>(null)

const headers = [
  { title: '类型', key: 'memory_type', width: 100, sortable: false },
  { title: '内容', key: 'judgment', sortable: false },
  { title: '标签', key: 'tags', width: 220, sortable: false },
  { title: '强度', key: 'strength', width: 90, sortable: false },
  { title: '主动', key: 'is_active', width: 80, sortable: false, headerProps: { class: 'text-no-wrap' } },
  { title: '创建时间', key: 'created_at', width: 150, sortable: false },
  { title: '操作', key: 'actions', width: 110, sortable: false },
]

const detailRows = computed(() => {
  const it = selectedItem.value
  if (!it) return []
  return [
    { label: '编号', value: it.id },
    { label: '类型', value: it.memory_type || '-' },
    { label: '内容', value: it.judgment || '-' },
    { label: '说明', value: it.reasoning || '-' },
    { label: '强度', value: it.strength ?? '-' },
    { label: '主动', value: it.is_active ? '是' : '否' },
    { label: '分类域', value: scopeLabel(it.memory_scope) },
    { label: '创建时间', value: formatTime(it.created_at) },
    { label: '更新时间', value: formatTime(it.updated_at) },
  ]
})

// 分类域：只有 public 是内置英文值，其余是用户自己的柜子名
function scopeLabel(value: unknown): string {
  const text = String(value ?? '')
  if (!text) return '-'
  return text.toLowerCase() === 'public' ? '公开' : text
}

function strengthColor(s: number): string {
  if (s >= 80) return 'success'
  if (s >= 50) return 'primary'
  if (s >= 30) return 'warning'
  return 'error'
}

function showDetail(item: any) {
  selectedItem.value = item
  detailDialog.value = true
}

async function confirmDelete(item: any) {
  const ok = await confirmDialog({
    title: '删除这条记忆？',
    text: `删除后无法恢复。\n\n${item.judgment ?? ''}`,
    confirmText: '删除',
    danger: true,
  })
  if (!ok) return
  try {
    await apiPost('memories/delete', { id: item.id })
    toast.success('已删除')
    await loadData()
  } catch (e) {
    toast.error(`删除失败: ${(e as Error).message}`)
  }
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
    const data: any = await apiGet('memories', {
      scope: scope.value || '',
      keyword: keyword.value || '',
      page: page.value,
      page_size: pageSize.value,
    })
    items.value = data.items || []
    total.value = data.total || 0
  } catch (e) {
    toast.error(`加载记忆失败: ${(e as Error).message}`)
  } finally {
    loading.value = false
  }
}

onMounted(async () => {
  try {
    const ov: any = await apiGet('overview')
    scopeOptions.value = (ov.scopes || []).map((s: string) => ({ title: scopeLabel(s), value: s }))
  } catch (e) {
    console.warn('[记忆浏览] 分类域列表加载失败:', e)
  }
  await loadData()
})
</script>
