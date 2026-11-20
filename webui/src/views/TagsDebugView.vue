<template>
  <div class="d-flex flex-column ga-4">
    <!-- 标签命中搜索 -->
    <v-card rounded="md">
      <v-card-text class="config-section pb-2">
        <div class="config-title">标签命中搜索</div>
        <div class="config-hint">输入一句话，看它会命中哪些标签和记忆</div>
      </v-card-text>
      <v-card-text class="pt-0 d-flex flex-column ga-3">
        <div class="d-flex flex-wrap align-center ga-3">
          <v-text-field
            v-model="hitQuery"
            label="查询文本"
            density="compact"
            hide-details
            variant="outlined"
            class="flex-grow-1"
            style="min-width: 220px;"
            @keyup.enter="doHitSearch"
          />
          <v-text-field
            v-model="hitScope"
            label="分类域（可不填）"
            density="compact"
            hide-details
            variant="outlined"
            style="max-width: 200px;"
            @keyup.enter="doHitSearch"
          />
          <v-btn color="primary" variant="flat" :loading="hitLoading" @click="doHitSearch">搜索</v-btn>
        </div>

        <template v-if="hitResult">
          <v-divider />
          <div class="d-flex flex-wrap align-center ga-2">
            <span class="text-body-2 font-weight-medium">命中标签</span>
            <template v-if="hitResult.matched_tags?.length">
              <v-chip v-for="tag in hitResult.matched_tags" :key="tag" size="small" color="success" variant="tonal">
                {{ tag }}
              </v-chip>
            </template>
            <span v-else class="text-body-2 text-medium-emphasis">没有命中任何标签</span>
          </div>

          <template v-if="hitResult.memory_hits?.length">
            <div class="text-body-2 font-weight-medium">命中记忆（{{ hitResult.memory_hits.length }} 条）</div>
            <v-card v-for="mem in hitResult.memory_hits" :key="mem.id" variant="outlined" rounded="md">
              <v-card-text>
                <div class="d-flex flex-wrap align-center ga-2 mb-1">
                  <v-chip size="x-small" color="primary" variant="tonal">{{ mem.memory_type }}</v-chip>
                  <v-chip size="x-small" color="warning" variant="tonal">命中 {{ mem.hit_count }} 个标签</v-chip>
                  <v-chip size="x-small" variant="tonal">强度 {{ mem.strength }}</v-chip>
                </div>
                <div class="text-body-2 font-weight-medium">{{ mem.judgment }}</div>
                <div v-if="mem.reasoning" class="text-caption text-medium-emphasis mt-1">{{ mem.reasoning }}</div>
              </v-card-text>
            </v-card>
          </template>
        </template>
      </v-card-text>
    </v-card>

    <!-- 全局标签列表 -->
    <v-card rounded="md">
      <v-card-text class="config-section pb-2">
        <div class="config-title">全局标签列表</div>
        <div class="config-hint">所有记忆和笔记用到的标签，以及各自被引用多少次</div>
      </v-card-text>
      <v-card-text class="pt-0">
        <v-text-field
          v-model="tagKeyword"
          label="筛选标签名"
          density="compact"
          hide-details
          variant="outlined"
          prepend-inner-icon="lucide:search"
          clearable
          class="mb-3"
          style="max-width: 360px;"
          @keyup.enter="loadTags"
          @click:clear="loadTags"
        />
        <v-data-table
          :headers="tagHeaders"
          :items="tags"
          :loading="tagsLoading"
          :items-per-page="50"
          item-value="id"
          density="comfortable"
        >
          <template #item.name="{ item }">
            <v-chip size="small" variant="tonal">{{ item.name }}</v-chip>
          </template>
          <template #no-data>
            <v-empty-state icon="lucide:tags" title="还没有标签" text="记忆生成时会自动打标签" />
          </template>
        </v-data-table>
      </v-card-text>
    </v-card>
  </div>
</template>

<script setup lang="ts">
import { ref, onMounted } from 'vue'
import { useBridge } from '@/composables/useBridge'
import { toast } from '@/composables/useToast'

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
    toast.error(`标签命中搜索失败: ${(e as Error).message}`)
  } finally {
    hitLoading.value = false
  }
}

// 标签列表
const tagKeyword = ref('')
const tags = ref<any[]>([])
const tagsLoading = ref(false)

const tagHeaders = [
  { title: '编号', key: 'id', width: 80, sortable: false },
  { title: '标签名', key: 'name', sortable: false },
  { title: '记忆引用', key: 'memory_refs', width: 110, sortable: false },
  { title: '笔记引用', key: 'note_refs', width: 110, sortable: false },
]

async function loadTags() {
  tagsLoading.value = true
  try {
    const data: any = await apiGet('tags', { keyword: tagKeyword.value, limit: 300 })
    tags.value = data.tags || []
  } catch (e) {
    toast.error(`加载标签失败: ${(e as Error).message}`)
  } finally {
    tagsLoading.value = false
  }
}

onMounted(() => loadTags())
</script>

<style scoped>
.config-title {
  font-size: 0.875rem;
  font-weight: 500;
  color: rgb(var(--v-theme-primaryText));
}
.config-hint {
  font-size: 0.75rem;
  color: rgb(var(--v-theme-secondaryText));
  margin-top: 2px;
}
</style>
