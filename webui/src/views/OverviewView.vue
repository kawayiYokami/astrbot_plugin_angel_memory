<template>
  <div>
    <n-spin :show="loading">
      <template v-if="!loading">
        <!-- 统计卡片 -->
        <n-grid :cols="4" :x-gap="16" responsive="screen" item-responsive>
          <n-grid-item span="4 s:2 m:1">
            <n-card class="stat-card" embedded>
              <div class="stat-value">{{ overview.memory_count ?? '-' }}</div>
              <div class="stat-label">记忆总数</div>
            </n-card>
          </n-grid-item>
          <n-grid-item span="4 s:2 m:1">
            <n-card class="stat-card" embedded>
              <div class="stat-value">{{ overview.global_tag_count ?? '-' }}</div>
              <div class="stat-label">全局标签</div>
            </n-card>
          </n-grid-item>
          <n-grid-item span="4 s:2 m:1">
            <n-card class="stat-card" embedded>
              <div class="stat-value">{{ overview.note_index_count ?? '-' }}</div>
              <div class="stat-label">笔记索引</div>
            </n-card>
          </n-grid-item>
          <n-grid-item span="4 s:2 m:1">
            <n-card class="stat-card" embedded>
              <div class="stat-value">
                <Icon
                  :icon="overview.has_providers ? 'lucide:check-circle-2' : 'lucide:alert-triangle'"
                  :style="{ color: overview.has_providers ? '#63e2b7' : '#f0a020', fontSize: '30px' }"
                />
              </div>
              <div class="stat-label">{{ overview.has_providers ? '提供商就绪' : '无提供商' }}</div>
            </n-card>
          </n-grid-item>
        </n-grid>

        <n-grid :cols="2" :x-gap="16" responsive="screen" class="mt-4">
          <!-- 配置信息 -->
          <n-grid-item span="2 m:1">
            <n-card title="配置信息" embedded>
              <n-descriptions label-placement="left" :column="1" size="small">
                <n-descriptions-item label="嵌入提供商">
                  {{ overview.provider_id || '-' }}
                </n-descriptions-item>
                <n-descriptions-item label="LLM 提供商">
                  {{ overview.llm_provider_id || '-' }}
                </n-descriptions-item>
                <n-descriptions-item label="索引目录">
                  <span class="truncate" :title="overview.index_dir">{{ overview.index_dir || '-' }}</span>
                </n-descriptions-item>
                <n-descriptions-item label="向量索引">
                  {{ overview.has_vector_db ? '可用' : '不可用' }}
                </n-descriptions-item>
              </n-descriptions>
            </n-card>
          </n-grid-item>

          <!-- Scope 与向量集合 -->
          <n-grid-item span="2 m:1">
            <n-card title="Scope 列表" embedded>
              <template v-if="overview.scopes?.length">
                <n-tag
                  v-for="scope in overview.scopes"
                  :key="scope"
                  type="primary"
                  size="small"
                  class="chip-item"
                  :bordered="false"
                >
                  {{ scope }}
                </n-tag>
              </template>
              <n-empty v-else description="暂无 scope" size="small" />
            </n-card>

            <n-card v-if="overview.vector_collections?.length" title="向量集合" embedded class="mt-4">
              <n-tag
                v-for="col in overview.vector_collections"
                :key="col"
                type="info"
                size="small"
                class="chip-item"
                :bordered="false"
              >
                {{ col }}
              </n-tag>
            </n-card>
          </n-grid-item>
        </n-grid>
      </template>
    </n-spin>
  </div>
</template>

<script setup lang="ts">
import { ref, onMounted } from 'vue'
import { useBridge } from '@/composables/useBridge'

const { apiGet } = useBridge()

const loading = ref(true)
const overview = ref<Record<string, any>>({})

onMounted(async () => {
  try {
    overview.value = await apiGet('overview')
  } catch (e) {
    console.error('加载总览失败:', e)
  } finally {
    loading.value = false
  }
})
</script>

<style scoped>
.stat-card {
  text-align: center;
}

.stat-value {
  font-size: 28px;
  font-weight: 700;
  line-height: 1.2;
}

.stat-label {
  margin-top: 4px;
  font-size: 13px;
  opacity: 0.65;
}

.chip-item {
  margin: 2px 4px 2px 0;
}

.truncate {
  display: inline-block;
  max-width: 260px;
  overflow: hidden;
  text-overflow: ellipsis;
  white-space: nowrap;
  vertical-align: bottom;
}
</style>
