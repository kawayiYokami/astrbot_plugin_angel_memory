<template>
  <div>
    <h2 class="text-h5 mb-4">🔖 Tags 调试</h2>

    <!-- 标签命中搜索 -->
    <v-card class="mb-4">
      <v-card-title>标签命中搜索</v-card-title>
      <v-card-text>
        <v-row>
          <v-col cols="12" sm="6">
            <v-text-field
              v-model="hitQuery"
              label="输入查询文本"
              density="compact"
              variant="outlined"
              @keyup.enter="doHitSearch"
            />
          </v-col>
          <v-col cols="12" sm="3">
            <v-text-field
              v-model="hitScope"
              label="Scope（可空）"
              density="compact"
              variant="outlined"
            />
          </v-col>
          <v-col cols="12" sm="3">
            <v-btn color="primary" @click="doHitSearch" :loading="hitLoading" block>搜索</v-btn>
          </v-col>
        </v-row>

        <template v-if="hitResult">
          <v-divider class="my-3" />
          <div class="mb-2">
            <strong>命中标签：</strong>
            <v-chip v-for="tag in hitResult.matched_tags" :key="tag" size="small" color="success" class="ma-1">
              {{ tag }}
            </v-chip>
            <span v-if="!hitResult.matched_tags?.length" class="text-grey">无命中</span>
          </div>

          <div v-if="hitResult.memory_hits?.length">
            <strong>命中记忆（{{ hitResult.memory_hits.length }}条）：</strong>
            <v-list density="compact" class="mt-2">
              <v-list-item v-for="mem in hitResult.memory_hits" :key="mem.id" class="mb-2">
                <v-card variant="outlined" class="pa-2">
                  <div class="d-flex align-center ga-2 mb-1">
                    <v-chip size="x-small" color="primary">{{ mem.memory_type }}</v-chip>
                    <v-chip size="x-small" color="warning">命中{{ mem.hit_count }}个标签</v-chip>
                    <v-chip size="x-small">强度 {{ mem.strength }}</v-chip>
                  </div>
                  <div>{{ mem.judgment }}</div>
                  <div class="text-caption text-grey mt-1">{{ mem.reasoning }}</div>
                </v-card>
              </v-list-item>
            </v-list>
          </div>
        </template>
      </v-card-text>
    </v-card>

    <!-- 全局标签列表 -->
    <v-card>
      <v-card-title>全局标签列表</v-card-title>
      <v-card-text>
        <v-text-field
          v-model="tagKeyword"
          label="筛选标签名"
          density="compact"
          variant="outlined"
          clearable
          class="mb-3"
          @keyup.enter="loadTags"
          @click:clear="tagKeyword = ''; loadTags()"
        />

        <v-data-table
          :headers="tagHeaders"
          :items="tags"
          :loading="tagsLoading"
          density="compact"
          :items-per-page="50"
        >
          <template v-slot:item.name="{ item }">
            <v-chip size="small" variant="tonal">{{ item.name }}</v-chip>
          </template>
        </v-data-table>
      </v-card-text>
    </v-card>
  </div>
</template>

<script setup lang="ts">
import { ref, onMounted } from 'vue'
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

const tagHeaders = [
  { title: 'ID', key: 'id', width: '60px' },
  { title: '标签名', key: 'name' },
  { title: '记忆引用', key: 'memory_refs', width: '100px' },
  { title: '笔记引用', key: 'note_refs', width: '100px' },
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
