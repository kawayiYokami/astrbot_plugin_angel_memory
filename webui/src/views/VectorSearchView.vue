<template>
  <div>
    <h2 class="text-h5 mb-4">🧭 向量检索</h2>

    <v-card class="mb-4">
      <v-card-text>
        <v-row>
          <v-col cols="12" sm="3">
            <v-select
              v-model="collection"
              :items="collections"
              item-title="name"
              item-value="name"
              label="集合"
              density="compact"
              variant="outlined"
            >
              <template v-slot:item="{ item, props }">
                <v-list-item v-bind="props">
                  <template v-slot:subtitle>{{ item.raw.count }} 条记录</template>
                </v-list-item>
              </template>
            </v-select>
          </v-col>
          <v-col cols="12" sm="6">
            <v-text-field
              v-model="queryText"
              label="向量查询"
              placeholder="输入一句话测试向量召回"
              density="compact"
              variant="outlined"
              @keyup.enter="doSearch"
            />
          </v-col>
          <v-col cols="12" sm="1">
            <v-text-field
              v-model.number="topK"
              label="Top K"
              type="number"
              density="compact"
              variant="outlined"
              :min="1"
              :max="50"
            />
          </v-col>
          <v-col cols="12" sm="2">
            <v-btn color="primary" @click="doSearch" :loading="searching" block>检索</v-btn>
          </v-col>
        </v-row>
      </v-card-text>
    </v-card>

    <!-- 检索结果 -->
    <v-card v-if="results.length">
      <v-card-title>检索结果（{{ results.length }}条）</v-card-title>
      <v-card-text>
        <v-list>
          <v-list-item v-for="(item, idx) in results" :key="idx" class="mb-2">
            <v-card variant="outlined" class="pa-3">
              <div class="d-flex align-center ga-2 mb-1">
                <v-chip size="small" color="primary">score: {{ item.score?.toFixed(4) }}</v-chip>
                <span class="text-caption text-grey">{{ item.id }}</span>
              </div>
              <code class="text-body-2">{{ item.document }}</code>
            </v-card>
          </v-list-item>
        </v-list>
      </v-card-text>
    </v-card>

    <v-alert v-if="error" type="error" class="mt-4">{{ error }}</v-alert>

    <!-- 原始浏览 -->
    <v-card class="mt-4">
      <v-card-title>
        原始浏览
        <v-btn size="small" variant="text" @click="loadBrowse" icon="mdi-refresh" class="ml-2" />
      </v-card-title>
      <v-card-text>
        <v-data-table-server
          :headers="browseHeaders"
          :items="browseItems"
          :items-length="browseTotal"
          :loading="browseLoading"
          :page="browsePage"
          :items-per-page="20"
          @update:page="onBrowsePageChange"
          density="compact"
        >
          <template v-slot:item.document="{ item }">
            <div class="text-truncate" style="max-width: 400px">{{ item.document }}</div>
          </template>
        </v-data-table-server>
      </v-card-text>
    </v-card>
  </div>
</template>

<script setup lang="ts">
import { ref, onMounted, watch } from 'vue'
import { useBridge } from '@/composables/useBridge'

const { apiGet } = useBridge()

const collection = ref('memory_index')
const collections = ref<any[]>([])
const queryText = ref('')
const topK = ref(10)
const searching = ref(false)
const results = ref<any[]>([])
const error = ref('')

// 浏览
const browseItems = ref<any[]>([])
const browseTotal = ref(0)
const browsePage = ref(1)
const browseLoading = ref(false)

const browseHeaders = [
  { title: 'ID', key: 'id', width: '200px' },
  { title: '内容', key: 'document' },
  { title: '维度', key: 'dimension', width: '70px' },
]

async function loadCollections() {
  try {
    const data: any = await apiGet('vector/collections')
    collections.value = data.collections || []
    if (collections.value.length && !collections.value.find((c: any) => c.name === collection.value)) {
      collection.value = collections.value[0].name
    }
  } catch (e) { /* ignore */ }
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
