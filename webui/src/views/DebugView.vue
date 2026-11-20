<template>
  <div class="d-flex flex-column ga-4">
    <v-alert type="info" variant="tonal" density="compact">
      这一页是给上限压力测试用的，所有参数只对本次请求生效，不会改掉插件里的设置。
    </v-alert>

    <!-- 语义向量探针 -->
    <v-card rounded="md">
      <v-card-text class="config-section pb-2">
        <div class="config-title">语义向量试跑</div>
        <div class="config-hint">把多段文字一次性转成向量，看批次大小和超时能不能撑住</div>
      </v-card-text>
      <v-card-text class="pt-0 d-flex flex-column ga-3">
        <div class="d-flex flex-wrap align-center ga-3">
          <v-text-field
            v-model.number="embedBatchSize"
            label="每批条数"
            type="number"
            min="1"
            density="compact"
            hide-details
            variant="outlined"
            style="max-width: 160px;"
          />
          <v-text-field
            v-model.number="embedTimeout"
            label="超时（秒）"
            type="number"
            min="5"
            max="120"
            density="compact"
            hide-details
            variant="outlined"
            style="max-width: 160px;"
          />
          <v-btn variant="text" size="small" @click="embedTimeout = 5">5 秒</v-btn>
          <v-btn variant="text" size="small" @click="embedTimeout = 60">60 秒</v-btn>
          <v-btn
            color="primary"
            variant="flat"
            :loading="embedLoading"
            :disabled="!embedText.trim()"
            @click="doEmbedProbe"
          >
            开始试跑
          </v-btn>
        </div>
        <v-textarea
          v-model="embedText"
          label="每行一条文字"
          rows="5"
          auto-grow
          density="compact"
          hide-details
          variant="outlined"
          placeholder="你好&#10;世界&#10;天使的记忆"
        />

        <template v-if="embedResult">
          <v-alert v-if="embedResult.error" type="error" variant="tonal">{{ embedResult.error }}</v-alert>
          <template v-else>
            <div class="d-flex flex-wrap ga-2">
              <v-chip size="small" color="success" variant="tonal">
                共 {{ embedResult.total }} 条 / {{ embedResult.batches }} 批
              </v-chip>
              <v-chip size="small" variant="tonal">耗时 {{ embedResult.elapsed_ms }} 毫秒</v-chip>
              <v-chip size="small" variant="tonal">平均 {{ embedResult.avg_ms_per_text }} 毫秒/条</v-chip>
              <v-chip size="small" variant="tonal">维度 {{ embedResult.dimension }}</v-chip>
              <v-chip v-if="embedResult.failed_batches" size="small" color="error" variant="tonal">
                失败 {{ embedResult.failed_batches }} 批
              </v-chip>
              <v-chip v-if="embedResult.timed_out_batches" size="small" color="warning" variant="tonal">
                超时 {{ embedResult.timed_out_batches }} 批
              </v-chip>
            </div>
            <div v-if="embedResult.provider" class="text-caption text-medium-emphasis">
              服务商 {{ embedResult.provider.provider_id || '—' }} / 模型
              {{ embedResult.provider.model_name || '—' }}（{{ embedResult.provider.type || '—' }}）
            </div>
            <v-data-table
              :headers="embedHeaders"
              :items="embedBatchDetails"
              :items-per-page="10"
              density="comfortable"
            >
              <template #item.batch_index="{ item }">#{{ item.batch_index }}</template>
              <template #item.dimension="{ item }">{{ item.dimension ?? '—' }}</template>
              <template #item.success="{ item }">
                <v-chip size="x-small" :color="item.success ? 'success' : item.timed_out ? 'warning' : 'error'" variant="tonal">
                  {{ item.success ? '成功' : item.timed_out ? '超时' : '失败' }}
                </v-chip>
              </template>
              <template #item.error="{ item }">
                <span class="text-truncate d-inline-block" style="max-width: 320px;" :title="item.error || ''">
                  {{ item.error || '—' }}
                </span>
              </template>
            </v-data-table>
            <template v-if="embedResult.preview?.length">
              <div class="text-body-2 font-weight-medium">前 3 条向量片段</div>
              <pre class="code-block">{{ JSON.stringify(embedResult.preview, null, 2) }}</pre>
            </template>
          </template>
        </template>
      </v-card-text>
    </v-card>

    <!-- 排序模型探针 -->
    <v-card rounded="md">
      <v-card-text class="config-section pb-2">
        <div class="config-title">排序模型试跑</div>
        <div class="config-hint">给一句话和一批候选，看排序模型能不能按相关度排出来</div>
      </v-card-text>
      <v-card-text class="pt-0 d-flex flex-column ga-3">
        <v-text-field
          v-model="rerankQuery"
          label="要查的内容"
          density="compact"
          hide-details
          variant="outlined"
          placeholder="例如：天使的记忆"
        />
        <v-textarea
          v-model="rerankDocsText"
          label="候选文本（每行一条）"
          rows="5"
          auto-grow
          density="compact"
          hide-details
          variant="outlined"
          placeholder="记忆系统会记住你的偏好&#10;笔记系统会整理你的知识&#10;向量检索负责召回"
        />
        <div class="d-flex flex-wrap align-center ga-3">
          <v-text-field
            v-model.number="rerankTimeout"
            label="超时（秒）"
            type="number"
            min="5"
            max="120"
            density="compact"
            hide-details
            variant="outlined"
            style="max-width: 150px;"
          />
          <v-text-field
            v-model.number="rerankMaxDocs"
            label="候选上限"
            type="number"
            min="1"
            max="200"
            density="compact"
            hide-details
            variant="outlined"
            style="max-width: 150px;"
          />
          <v-text-field
            v-model.number="rerankMaxTokens"
            label="单段上限（Token）"
            type="number"
            min="128"
            max="8192"
            step="128"
            density="compact"
            hide-details
            variant="outlined"
            style="max-width: 180px;"
          />
          <v-btn
            color="primary"
            variant="flat"
            :loading="rerankLoading"
            :disabled="!rerankQuery.trim() || !rerankDocsText.trim()"
            @click="doRerankProbe"
          >
            开始试跑
          </v-btn>
          <span v-if="rerankHasProvider === false" class="text-caption text-medium-emphasis">
            没有配置排序服务商，会自动退回关键词排序
          </span>
        </div>
        <div v-if="rerankMaxDocs !== null && rerankMaxTokens !== null" class="text-caption text-medium-emphasis">
          只对本次生效。常用 64 / 1024，bge-reranker-v2-m3 单对硬上限 8192
        </div>

        <template v-if="rerankResult">
          <v-alert v-if="rerankResult.timed_out" type="warning" variant="tonal">
            排序超时（超过 {{ rerankResult.timeout }} 秒），已退回关键词排序
          </v-alert>
          <v-alert
            v-if="rerankResult.error && !rerankResult.timed_out"
            :type="rerankResult.has_rerank ? 'error' : 'warning'"
            variant="tonal"
          >
            {{ rerankResult.error }}
          </v-alert>
          <template v-if="rerankResult.scores?.length">
            <div class="d-flex flex-wrap ga-2">
              <v-chip size="small" variant="tonal">耗时 {{ rerankResult.elapsed_ms }} 毫秒</v-chip>
              <v-chip size="small" variant="tonal">返回 {{ rerankResult.scores.length }} 条</v-chip>
              <v-chip v-if="rerankResult.provider_id" size="small" variant="tonal">{{ rerankResult.provider_id }}</v-chip>
              <v-chip v-if="rerankResult.original_docs !== undefined" size="small" variant="tonal">
                候选 {{ rerankResult.original_docs }} → {{ rerankResult.kept_docs }}/{{ rerankResult.rerank_max_docs }}
              </v-chip>
              <v-chip v-if="rerankResult.rerank_max_tokens_per_doc" size="small" variant="tonal">
                单段 {{ rerankResult.rerank_max_tokens_per_doc }} Token
              </v-chip>
              <v-chip v-if="rerankResult.doc_truncated" size="small" color="warning" variant="tonal">
                截断 {{ rerankResult.doc_truncated }} 段
              </v-chip>
              <v-chip v-if="rerankResult.query_truncated" size="small" color="warning" variant="tonal">
                查询已截断
              </v-chip>
            </div>
            <v-data-table
              :headers="rerankHeaders"
              :items="rerankScores"
              :items-per-page="20"
              density="comfortable"
            >
              <template #item.score="{ item }">
                {{ typeof item.score === 'number' ? item.score.toFixed(4) : (item.score ?? '—') }}
              </template>
              <template #item.document="{ item }">
                <span class="text-truncate d-inline-block" style="max-width: 420px;" :title="item.document">
                  {{ item.document }}
                </span>
              </template>
            </v-data-table>
          </template>
          <v-empty-state
            v-else-if="!rerankResult.error && !rerankResult.timed_out"
            icon="lucide:flask-conical"
            title="没有排序结果"
            text="可能被截断成空了，试试减少候选或放宽单段上限"
          />
        </template>
      </v-card-text>
    </v-card>
  </div>
</template>

<script setup lang="ts">
import { computed, ref } from 'vue'
import { useBridge } from '@/composables/useBridge'
import { toast } from '@/composables/useToast'
import { confirmDialog } from '@/composables/useConfirm'

const { apiPost } = useBridge()

// 表格数据统一成 any[]，避免模板插槽里 item 被推断成 unknown
const embedBatchDetails = computed<any[]>(() => embedResult.value?.batch_details ?? [])
const rerankScores = computed<any[]>(() => rerankResult.value?.scores ?? [])

// 语义向量
const embedText = ref('')
const embedBatchSize = ref<number | null>(50)
const embedTimeout = ref<number | null>(5)
const embedLoading = ref(false)
const embedResult = ref<any>(null)

const embedHeaders = [
  { title: '批次', key: 'batch_index', width: 80, sortable: false },
  { title: '条数', key: 'batch_size', width: 80, sortable: false },
  { title: '耗时（毫秒）', key: 'elapsed_ms', width: 120, sortable: false },
  { title: '维度', key: 'dimension', width: 90, sortable: false },
  { title: '状态', key: 'success', width: 90, sortable: false },
  { title: '错误', key: 'error', sortable: false },
]

async function doEmbedProbe() {
  const texts = embedText.value.split('\n').map((s) => s.trim()).filter(Boolean)
  if (!texts.length) {
    toast.warning('请先输入至少一条文字')
    return
  }
  const bs = embedBatchSize.value ?? 50
  if (bs > 200) {
    const ok = await confirmDialog({
      title: '批次偏大',
      text: `每批 ${bs} 条比较多，可能触发服务商的限流或拒绝。要继续吗？`,
      confirmText: '继续',
    })
    if (!ok) return
  }
  embedLoading.value = true
  embedResult.value = null
  try {
    const data: any = await apiPost('debug/embedding', {
      texts,
      batch_size: bs,
      timeout: embedTimeout.value ?? 5,
    })
    if (data.error && data.total === undefined) {
      embedResult.value = data
      toast.error(data.error)
    } else {
      embedResult.value = data
      if (data.failed_batches) toast.warning(`有 ${data.failed_batches} 批失败，请看下方明细`)
      else toast.success('试跑完成')
    }
  } catch (e: any) {
    const msg = e.message || '试跑失败'
    embedResult.value = { error: msg }
    toast.error(msg)
  } finally {
    embedLoading.value = false
  }
}

// 排序模型
const rerankQuery = ref('')
const rerankDocsText = ref('')
const rerankTimeout = ref<number | null>(5)
const rerankMaxDocs = ref<number | null>(64)
const rerankMaxTokens = ref<number | null>(1024)
const rerankLoading = ref(false)
const rerankResult = ref<any>(null)
const rerankHasProvider = ref<boolean | null>(null)

const rerankHeaders = [
  { title: '序号', key: 'index', width: 80, sortable: false },
  { title: '相关度', key: 'score', width: 110, sortable: false },
  { title: '候选文本', key: 'document', sortable: false },
]

async function doRerankProbe() {
  if (!rerankQuery.value.trim() || !rerankDocsText.value.trim()) {
    toast.warning('要查的内容和候选文本都要填')
    return
  }
  const documents = rerankDocsText.value.split('\n').map((s) => s.trim()).filter(Boolean)
  rerankLoading.value = true
  rerankResult.value = null
  try {
    const data: any = await apiPost('debug/rerank', {
      query: rerankQuery.value.trim(),
      documents,
      timeout: rerankTimeout.value ?? 5,
      rerank_max_docs: rerankMaxDocs.value ?? 64,
      rerank_max_tokens_per_doc: rerankMaxTokens.value ?? 1024,
    })
    rerankResult.value = data
    rerankHasProvider.value = data.has_rerank ?? null
    if (data.error) {
      if (data.has_rerank === false) toast.warning(data.error)
      else if (data.timed_out) toast.warning(data.error)
      else toast.error(data.error)
    } else {
      toast.success('试跑完成')
    }
  } catch (e: any) {
    const msg = e.message || '试跑失败'
    rerankResult.value = { error: msg, has_rerank: true }
    toast.error(msg)
  } finally {
    rerankLoading.value = false
  }
}
</script>

<style scoped>
.code-block {
  background: rgba(var(--v-theme-on-surface), 0.06);
  border-radius: 8px;
  padding: 12px;
  overflow-x: auto;
  white-space: pre-wrap;
  font-size: 12px;
  line-height: 1.4;
  margin: 0;
  max-height: 40vh;
}

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
