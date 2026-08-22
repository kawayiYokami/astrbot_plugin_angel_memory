<template>
  <n-space vertical :size="16">
    <!-- 标签命中搜索 -->
    <n-card title="标签命中搜索" embedded>
      <n-space vertical size="medium">
        <n-space>
          <n-input
            v-model:value="hitQuery"
            placeholder="输入查询文本"
            style="flex: 2"
            @keyup.enter="doHitSearch"
          />
          <n-input
            v-model:value="hitScope"
            placeholder="Scope（可空）"
            style="flex: 1"
            @keyup.enter="doHitSearch"
          />
          <n-button type="primary" :loading="hitLoading" @click="doHitSearch">搜索</n-button>
        </n-space>

        <template v-if="hitResult">
          <n-divider />
          <div>
            <strong>命中标签：</strong>
            <template v-if="hitResult.matched_tags?.length">
              <n-tag
                v-for="tag in hitResult.matched_tags"
                :key="tag"
                type="success"
                size="small"
                :bordered="false"
                class="chip-item"
              >
                {{ tag }}
              </n-tag>
            </template>
            <span v-else class="muted">无命中</span>
          </div>

          <template v-if="hitResult.memory_hits?.length">
            <strong>命中记忆（{{ hitResult.memory_hits.length }}条）：</strong>
            <n-space vertical size="small">
              <n-card
                v-for="mem in hitResult.memory_hits"
                :key="mem.id"
                embedded
                size="small"
              >
                <n-space size="small">
                  <n-tag size="small" type="primary" :bordered="false">{{ mem.memory_type }}</n-tag>
                  <n-tag size="small" type="warning" :bordered="false">命中{{ mem.hit_count }}个标签</n-tag>
                  <n-tag size="small" :bordered="false">强度 {{ mem.strength }}</n-tag>
                </n-space>
                <div class="mem-judgment">{{ mem.judgment }}</div>
                <div v-if="mem.reasoning" class="muted mem-reasoning">{{ mem.reasoning }}</div>
              </n-card>
            </n-space>
          </template>
        </template>
      </n-space>
    </n-card>

    <!-- 全局标签列表 -->
    <n-card title="全局标签列表" embedded>
      <n-space vertical size="medium">
        <n-input
          v-model:value="tagKeyword"
          placeholder="筛选标签名"
          clearable
          @keyup.enter="loadTags"
          @clear="loadTags"
        />
        <n-data-table
          :columns="tagColumns"
          :data="tags"
          :loading="tagsLoading"
          :pagination="{ pageSize: 50 }"
          :row-key="(row: any) => row.id"
        />
      </n-space>
    </n-card>
  </n-space>
</template>

<script setup lang="ts">
import { h, ref, onMounted } from 'vue'
import type { DataTableColumns } from 'naive-ui'
import { NButton, NCard, NDataTable, NDivider, NInput, NSpace, NTag } from 'naive-ui'
import { useBridge } from '@/composables/useBridge'

const { apiGet, apiPost } = useBridge()

// 命中搜索
const hitQuery = ref('')
const hitScope = ref('')
const hitLoading = ref(false)
const hitResult = ref<any>(null)

async function doHitSearch() {
  if (!hitQuery.value.trim()) return
  hitLoading.value = true
  try {
    hitResult.value = await apiPost('tags/hit-search', {
      query: hitQuery.value,
      scope: hitScope.value,
      limit: 50,
    })
  } catch (e) {
    console.error('标签命中搜索失败:', e)
  } finally {
    hitLoading.value = false
  }
}

// 标签列表
const tagKeyword = ref('')
const tags = ref<any[]>([])
const tagsLoading = ref(false)

const tagColumns: DataTableColumns<any> = [
  { title: 'ID', key: 'id', width: 70 },
  {
    title: '标签名',
    key: 'name',
    render: row =>
      h(NTag, { size: 'small' }, { default: () => row.name }),
  },
  { title: '记忆引用', key: 'memory_refs', width: 100 },
  { title: '笔记引用', key: 'note_refs', width: 100 },
]

async function loadTags() {
  tagsLoading.value = true
  try {
    const data: any = await apiGet('tags', { keyword: tagKeyword.value, limit: 300 })
    tags.value = data.tags || []
  } catch (e) {
    console.error('加载标签失败:', e)
  } finally {
    tagsLoading.value = false
  }
}

onMounted(() => loadTags())
</script>

<style scoped>
.chip-item {
  margin: 2px 4px 2px 0;
}

.mem-judgment {
  font-weight: 600;
  font-size: 14px;
}

.mem-reasoning {
  font-size: 13px;
  opacity: 0.7;
  margin-top: 2px;
}

.muted {
  opacity: 0.6;
  font-size: 13px;
}
</style>
