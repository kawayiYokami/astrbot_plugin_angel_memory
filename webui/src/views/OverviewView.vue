<template>
  <div>
    <n-spin :show="loading">
      <n-space v-if="!loading" vertical :size="16">
        <!-- 统计卡片 -->
        <n-grid :cols="4" :x-gap="16" :y-gap="16" responsive="screen" item-responsive>
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

          <n-grid :cols="2" :x-gap="16" :y-gap="16" responsive="screen">
            <!-- 配置信息 -->
            <n-grid-item span="2 m:1">
            <n-card title="配置信息" embedded>
              <n-descriptions label-placement="left" :column="1">
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
            <n-space vertical :size="16">
              <n-card title="Scope 列表" embedded>
                <template v-if="overview.scopes?.length">
                  <n-tag
                    v-for="scope in overview.scopes"
                    :key="scope"
                    type="primary"
                    class="chip-item"
                    :bordered="false"
                  >
                    {{ scope }}
                  </n-tag>
                </template>
                <n-empty v-else description="暂无 scope" />
              </n-card>

              <n-card v-if="overview.vector_collections?.length" title="向量集合" embedded>
                <n-tag
                  v-for="col in overview.vector_collections"
                  :key="col"
                  type="info"
                  class="chip-item"
                  :bordered="false"
                >
                  {{ col }}
                </n-tag>
              </n-card>
            </n-space>
          </n-grid-item>
        </n-grid>

        <!-- 反思归纳记录 -->
        <n-card title="反思归纳记录（最近10次）" embedded>
          <template #header-extra>
            <n-button quaternary size="small" :loading="traceLoading" @click="loadReflectionTraces">刷新</n-button>
          </template>
          <n-space vertical :size="12">
            <n-alert type="warning" :bordered="false" style="font-size: 12px">
              内存记录，重启插件后清空。提示词含对话原文，仅本机调试用；出问题时可复制整条反馈。
            </n-alert>
            <n-empty
              v-if="!traceRecords.length && !traceLoading"
              description="暂无反思记录，触发几轮对话后自动生成"
            />
            <n-card v-for="r in traceRecords" :key="r.id" size="small" embedded>
              <template #header>
                <n-space align="center" :size="8">
                  <n-tag :type="traceStatusType(r.status)" size="small" :bordered="false">{{
                    traceStatusText(r.status)
                  }}</n-tag>
                  <span class="muted">{{ r.time }}</span>
                  <span
                    class="muted"
                    style="max-width: 240px; overflow: hidden; text-overflow: ellipsis; white-space: nowrap"
                    >{{ r.session_id || '无会话ID' }}</span
                  >
                </n-space>
              </template>
              <template #header-extra>
                <n-button size="tiny" secondary @click="copyTrace(r)">复制</n-button>
              </template>
              <n-collapse>
                <n-collapse-item title="提示词" name="prompt">
                  <pre class="code-block">{{ r.prompt }}</pre>
                </n-collapse-item>
                <n-collapse-item v-if="r.status !== 'pending'" title="LLM 返回" name="response">
                  <n-alert v-if="r.error" type="error" :bordered="false" style="white-space: pre-wrap">{{
                    r.error
                  }}</n-alert>
                  <pre v-if="r.result" class="code-block">{{ JSON.stringify(r.result, null, 2) }}</pre>
                  <div v-if="!r.result && !r.error" class="muted">无返回内容</div>
                </n-collapse-item>
              </n-collapse>
            </n-card>
          </n-space>
        </n-card>
      </n-space>
    </n-spin>
  </div>
</template>

<script setup lang="ts">
import { ref, onMounted } from 'vue'
import {
  NAlert,
  NButton,
  NCard,
  NCollapse,
  NCollapseItem,
  NDescriptions,
  NDescriptionsItem,
  NEmpty,
  NGrid,
  NGridItem,
  NSpace,
  NSpin,
  NTag,
  useMessage,
} from 'naive-ui'
import { useBridge } from '@/composables/useBridge'

const { apiGet } = useBridge()
const message = useMessage()

const loading = ref(true)
const overview = ref<Record<string, any>>({})

// 反思归纳记录
const traceRecords = ref<any[]>([])
const traceLoading = ref(false)

function traceStatusType(status: string) {
  if (status === 'success') return 'success'
  if (status === 'failed') return 'error'
  return 'warning'
}

function traceStatusText(status: string) {
  if (status === 'success') return '成功'
  if (status === 'failed') return '失败'
  return '处理中'
}

async function loadReflectionTraces() {
  traceLoading.value = true
  try {
    const data: any = await apiGet('trace/reflection')
    traceRecords.value = Array.isArray(data && data.records) ? data.records : []
  } catch (e: any) {
    message.error(e.message || '获取反思记录失败')
  } finally {
    traceLoading.value = false
  }
}

async function copyTrace(r: any) {
  try {
    await navigator.clipboard.writeText(JSON.stringify(r, null, 2))
    message.success('已复制整条记录')
  } catch {
    message.error('复制失败，请手动选择文本')
  }
}

onMounted(async () => {
  try {
    overview.value = await apiGet('overview')
  } catch (e) {
    console.error('加载总览失败:', e)
  } finally {
    loading.value = false
  }
  loadReflectionTraces()
})
</script>

<style scoped>
.stat-card {
  text-align: center;
}

.stat-value {
  font-size: 16px;
  font-weight: 700;
  line-height: 1.2;
}

.stat-label {
  margin-top: 4px;
  font-size: 12px;
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

.muted {
  opacity: 0.6;
  font-size: 12px;
}

.code-block {
  background: #1a1a2e;
  color: #e0e0e0;
  border-radius: 6px;
  padding: 12px;
  overflow-x: auto;
  white-space: pre-wrap;
  font-size: 12px;
  line-height: 1.4;
}
</style>
