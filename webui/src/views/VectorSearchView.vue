<template>
  <div>
    <n-card title="向量检索" embedded class="mb-4">
      <n-space>
        <n-select
          v-model:value="collection"
          :options="collectionOptions"
          placeholder="集合"
          style="width: 240px"
        />
        <n-input
          v-model:value="queryText"
          placeholder="输入一句话测试向量召回"
          style="flex: 1"
          @keyup.enter="doSearch"
        />
        <n-input-number v-model:value="topK" :min="1" :max="50" placeholder="Top K" style="width: 100px" />
        <n-button type="primary" :loading="searching" @click="doSearch">检索</n-button>
      </n-space>
    </n-card>

    <!-- 检索结果 -->
    <n-card v-if="results.length" title="检索结果" embedded class="mb-4">
      <n-space vertical size="small">
        <n-card v-for="(item, idx) in results" :key="idx" embedded size="small">
          <n-space align="center" size="small" class="mb-1">
            <n-tag size="small" type="primary" :bordered="false">
              score: {{ item.score?.toFixed(4) }}
            </n-tag>
            <span class="muted">{{ item.id }}</span>
          </n-space>
          <code class="doc-code">{{ item.document }}</code>
        </n-card>
      </n-space>
    </n-card>

    <n-alert v-if="error" type="error" class="mb-4">{{ error }}</n-alert>

    <!-- 原始浏览 -->
    <n-card embedded>
      <template #header>
        <n-space align="center">
          <span>原始浏览</span>
          <n-button size="tiny" quaternary @click="loadBrowse">
            <template #icon><Icon icon="lucide:refresh-cw" /></template>
          </n-button>
        </n-space>
      </template>
      <n-data-table
        remote
        :columns="browseColumns"
        :data="browseItems"
        :loading="browseLoading"
        :pagination="browsePagination"
        :row-key="(row: any) => row.id"
        @update:page="onBrowsePageChange"
      />
    </n-card>
  </div>
</template>

<script setup lang="ts">
import { h, ref, computed, onMounted, watch } from 'vue'
import type { DataTableColumns } from 'naive-ui'
import {
  NAlert,
  NButton,
  NCard,
  NDataTable,
  NInput,
  NInputNumber,
  NSelect,
  NSpace,
  NTag,
} from 'naive-ui'
import { useBridge } from '@/composables/useBridge'

const { apiGet } = useBridge()

const collection = ref('memory_index')
const collections = ref<any[]>([])
const queryText = ref('')
const topK = ref<number | null>(10)
const searching = ref(false)
const results = ref<any[]>([])
const error = ref('')

// 浏览
const browseItems = ref<any[]>([])
const browseTotal = ref(0)
const browsePage = ref(1)
const browseLoading = ref(false)

const collectionOptions = computed(() =>
  collections.value.map(c => ({
    label: `${c.name}（${c.count} 条）`,
    value: c.name,
  })),
)

const browseColumns: DataTableColumns<any> = [
  { title: 'ID', key: 'id', width: 220, ellipsis: { tooltip: true } },
  { title: '内容', key: 'document', ellipsis: { tooltip: true } },
  { title: '维度', key: 'dimension', width: 80 },
]

const browsePagination = computed(() => ({
  page: browsePage.value,
  pageSize: 20,
  itemCount: browseTotal.value,
  prefix: ({ itemCount }: { itemCount: number }) => `共 ${itemCount} 条`,
}))

async function loadCollections() {
  try {
    const data: any = await apiGet('vector/collections')
    collections.value = data.collections || []
    if (
      collections.value.length &&
      !collections.value.find((c: any) => c.name === collection.value)
    ) {
      collection.value = collections.value[0].name
    }
  } catch (e) {
    /* ignore */
  }
}

async function doSearch() {
  if (!queryText.value.trim()) return
  searching.value = true
  error.value = ''
  try {
    const data: any = await apiGet('vector/search', {
      collection: collection.value,
      text: queryText.value,
      top_k: topK.value,
    })
    if (data.error) {
      error.value = data.error
      results.value = []
    } else {
      results.value = data.results || []
    }
  } catch (e: any) {
    error.value = e.message || '检索失败'
  } finally {
    searching.value = false
  }
}

async function loadBrowse() {
  browseLoading.value = true
  try {
    const data: any = await apiGet('vector/browse', {
      collection: collection.value,
      page: browsePage.value,
      page_size: 20,
    })
    browseItems.value = data.items || []
    browseTotal.value = data.total || 0
  } catch (e) {
    console.error('浏览失败:', e)
  } finally {
    browseLoading.value = false
  }
}

function onBrowsePageChange(p: number) {
  browsePage.value = p
  loadBrowse()
}

watch(collection, () => {
  browsePage.value = 1
  loadBrowse()
})

onMounted(async () => {
  await loadCollections()
  await loadBrowse()
})
</script>

<style scoped>
.doc-code {
  display: block;
  font-size: 13px;
  white-space: pre-wrap;
  word-break: break-all;
}

.muted {
  opacity: 0.6;
  font-size: 12px;
}
</style>
