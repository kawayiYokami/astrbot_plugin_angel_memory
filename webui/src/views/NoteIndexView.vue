<template>
  <div>
    <h2 class="text-h5 mb-4">🗂️ 笔记索引</h2>

    <v-card>
      <v-card-text>
        <v-text-field
          v-model="keyword"
          label="关键词搜索（路径 / 标题 / tags）"
          density="compact"
          variant="outlined"
          clearable
          append-inner-icon="mdi-magnify"
          class="mb-3"
          @keyup.enter="loadData"
          @click:append-inner="loadData"
        />

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
          <template v-slot:item.source_file_path="{ item }">
            <div class="text-truncate" style="max-width: 200px">{{ item.source_file_path }}</div>
          </template>
          <template v-slot:item.heading="{ item }">
            <div class="text-truncate" style="max-width: 250px">{{ buildHeading(item) }}</div>
          </template>
          <template v-slot:item.tags_text="{ item }">
            <div class="d-flex flex-wrap ga-1">
              <v-chip
                v-for="tag in parseTags(item.tags_text)"
                :key="tag"
                size="x-small"
                color="secondary"
                variant="tonal"
              >
                {{ tag }}
              </v-chip>
            </div>
          </template>
          <template v-slot:item.actions="{ item }">
            <v-btn
              icon="mdi-eye"
              size="x-small"
              variant="text"
              @click="showDetail(item)"
            />
          </template>
        </v-data-table-server>
      </v-card-text>
    </v-card>

    <!-- 详情对话框 -->
    <v-dialog v-model="detailDialog" max-width="600">
      <v-card v-if="selectedItem">
        <v-card-title>笔记索引详情</v-card-title>
        <v-card-text>
          <pre class="text-body-2" style="white-space: pre-wrap">{{ JSON.stringify(selectedItem, null, 2) }}</pre>
        </v-card-text>
        <v-card-actions>
          <v-spacer />
          <v-btn @click="detailDialog = false">关闭</v-btn>
        </v-card-actions>
      </v-card>
    </v-dialog>
  </div>
</template>

<script setup lang="ts">
import { ref, onMounted } from 'vue'
import { useBridge } from '@/composables/useBridge'

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
  { title: 'Short ID', key: 'note_short_id', width: '80px' },
  { title: '文件路径', key: 'source_file_path' },
  { title: '标题', key: 'heading' },
  { title: 'Tags', key: 'tags_text', width: '200px' },
  { title: '行数', key: 'total_lines', width: '60px' },
  { title: '操作', key: 'actions', width: '60px', sortable: false },
]

function buildHeading(item: any): string {
  const parts = []
  for (let i = 1; i <= 6; i++) {
    const h = item[`heading_h${i}`]
    if (h) parts.push(h)
  }
  return parts.join(' / ') || '(无标题)'
}

function parseTags(tags: string): string[] {
  if (!tags) return []
  return tags.split(',').map(t => t.trim()).filter(Boolean)
}

function showDetail(item: any) {
  selectedItem.value = item
  detailDialog.value = true
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
