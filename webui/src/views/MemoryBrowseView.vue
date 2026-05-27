<template>
  <div>
    <h2 class="text-h5 mb-4">🧾 记忆浏览</h2>

    <v-row class="mb-4">
      <v-col cols="12" sm="4">
        <v-select
          v-model="scope"
          :items="scopeOptions"
          label="Scope 过滤"
          density="compact"
          variant="outlined"
          clearable
          @update:model-value="loadData"
        />
      </v-col>
      <v-col cols="12" sm="8">
        <v-text-field
          v-model="keyword"
          label="关键词搜索（judgment / reasoning / tags）"
          density="compact"
          variant="outlined"
          clearable
          append-inner-icon="mdi-magnify"
          @keyup.enter="loadData"
          @click:append-inner="loadData"
        />
      </v-col>
    </v-row>

    <v-card>
      <v-data-table-server
        :headers="headers"
        :items="items"
        :items-length="total"
        :loading="loading"
        :page="page"
        :items-per-page="pageSize"
        @update:page="onPageChange"
        @update:items-per-page="onPageSizeChange"
        density="compact"
      >
        <template v-slot:item.judgment="{ item }">
          <div class="text-truncate" style="max-width: 300px">{{ item.judgment }}</div>
        </template>
        <template v-slot:item.tags="{ item }">
          <div class="d-flex flex-wrap ga-1">
            <v-chip v-for="tag in parseTags(item.tags)" :key="tag" size="x-small" color="primary" variant="tonal">
              {{ tag }}
            </v-chip>
          </div>
        </template>
        <template v-slot:item.strength="{ item }">
          <v-chip :color="strengthColor(item.strength)" size="small" variant="flat">
            {{ item.strength }}
          </v-chip>
        </template>
        <template v-slot:item.is_active="{ item }">
          <v-icon :icon="item.is_active ? 'mdi-star' : 'mdi-star-outline'" :color="item.is_active ? 'amber' : 'grey'" size="small" />
        </template>
        <template v-slot:item.created_at="{ item }">
          <span class="text-caption">{{ formatTime(item.created_at) }}</span>
        </template>
        <template v-slot:item.actions="{ item }">
          <v-btn icon="mdi-eye" size="x-small" variant="text" @click="showDetail(item)" />
          <v-btn icon="mdi-delete" size="x-small" variant="text" color="error" @click="confirmDelete(item)" />
        </template>
      </v-data-table-server>
    </v-card>

    <!-- 详情对话框 -->
    <v-dialog v-model="detailDialog" max-width="700">
      <v-card v-if="selectedItem">
        <v-card-title>记忆详情</v-card-title>
        <v-card-text>
          <v-list density="compact">
            <v-list-item><strong>ID:</strong>&nbsp;{{ selectedItem.id }}</v-list-item>
            <v-list-item><strong>类型:</strong>&nbsp;{{ selectedItem.memory_type }}</v-list-item>
            <v-list-item><strong>Judgment:</strong>&nbsp;{{ selectedItem.judgment }}</v-list-item>
            <v-list-item><strong>Reasoning:</strong>&nbsp;{{ selectedItem.reasoning }}</v-list-item>
            <v-list-item><strong>Tags:</strong>&nbsp;{{ selectedItem.tags }}</v-list-item>
            <v-list-item><strong>Strength:</strong>&nbsp;{{ selectedItem.strength }}</v-list-item>
            <v-list-item><strong>Active:</strong>&nbsp;{{ selectedItem.is_active ? '是' : '否' }}</v-list-item>
            <v-list-item><strong>Scope:</strong>&nbsp;{{ selectedItem.memory_scope }}</v-list-item>
            <v-list-item><strong>创建时间:</strong>&nbsp;{{ formatTime(selectedItem.created_at) }}</v-list-item>
            <v-list-item><strong>更新时间:</strong>&nbsp;{{ formatTime(selectedItem.updated_at) }}</v-list-item>
          </v-list>
        </v-card-text>
        <v-card-actions>
          <v-spacer />
          <v-btn @click="detailDialog = false">关闭</v-btn>
        </v-card-actions>
      </v-card>
    </v-dialog>

    <!-- 删除确认 -->
    <v-dialog v-model="deleteDialog" max-width="400">
      <v-card>
        <v-card-title class="text-error">确认删除</v-card-title>
        <v-card-text>
          确定要删除这条记忆吗？此操作不可撤销。
          <div class="mt-2 text-caption text-grey">{{ deleteTarget?.judgment }}</div>
        </v-card-text>
        <v-card-actions>
          <v-spacer />
          <v-btn @click="deleteDialog = false">取消</v-btn>
          <v-btn color="error" @click="doDelete" :loading="deleting">删除</v-btn>
        </v-card-actions>
      </v-card>
    </v-dialog>
  </div>
</template>

<script setup lang="ts">
import { ref, onMounted } from 'vue'
import { useBridge } from '@/composables/useBridge'

const { apiGet, apiPost } = useBridge()

const loading = ref(false)
const items = ref<any[]>([])
const total = ref(0)
const page = ref(1)
const pageSize = ref(20)
const scope = ref('')
const keyword = ref('')
const scopeOptions = ref<string[]>([])

const detailDialog = ref(false)
const selectedItem = ref<any>(null)
const deleteDialog = ref(false)
const deleteTarget = ref<any>(null)
const deleting = ref(false)

const headers = [
  { title: '类型', key: 'memory_type', width: '80px' },
  { title: 'Judgment', key: 'judgment' },
  { title: 'Tags', key: 'tags', width: '200px' },
  { title: '强度', key: 'strength', width: '70px' },
  { title: '主动', key: 'is_active', width: '50px' },
  { title: '创建时间', key: 'created_at', width: '140px' },
  { title: '操作', key: 'actions', width: '90px', sortable: false },
]

function parseTags(tags: string): string[] {
  if (!tags) return []
  return tags.split(',').map(t => t.trim()).filter(Boolean)
}

function strengthColor(s: number): string {
  if (s >= 80) return 'success'
  if (s >= 50) return 'primary'
  if (s >= 30) return 'warning'
  return 'error'
}

function formatTime(ts: number | null): string {
  if (!ts) return '-'
  let t = Number(ts)
  if (t > 1e11) t /= 1000
  return new Date(t * 1000).toLocaleString('zh-CN')
}

function showDetail(item: any) {
  selectedItem.value = item
  detailDialog.value = true
}

function confirmDelete(item: any) {
  deleteTarget.value = item
  deleteDialog.value = true
}

async function doDelete() {
  if (!deleteTarget.value) return
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
  // 加载 scope 列表
  try {
    const ov: any = await apiGet('overview')
    scopeOptions.value = ov.scopes || []
  } catch (e) { /* ignore */ }
  await loadData()
})
</script>
