<template>
  <n-space vertical :size="16">
    <n-alert type="info" :bordered="false" style="white-space: pre-wrap"
      >这是功能调试卡，方便测上限。仅对本次请求生效，不改全局配置。写入链路固定 60s 不受这里影响。</n-alert
    >

    <!-- 嵌入探针 -->
    <n-card title="嵌入探针（Embedding）" embedded>
      <n-space vertical :size="12">
        <n-space>
          <n-input-number
            v-model:value="embedBatchSize"
            :min="1"
            placeholder="批次"
            style="width: 140px"
          >
            <template #prefix>批次</template>
          </n-input-number>
          <n-input-number
            v-model:value="embedTimeout"
            :min="5"
            :max="120"
            placeholder="超时(s)"
            style="width: 150px"
          >
            <template #prefix>超时</template>
          </n-input-number>
          <n-button size="small" quaternary @click="embedTimeout = 5">5s</n-button>
          <n-button size="small" quaternary @click="embedTimeout = 60">60s</n-button>
          <n-button type="primary" :loading="embedLoading" :disabled="!embedText.trim()" @click="doEmbedProbe">测试嵌入</n-button>
        </n-space>
        <n-input
          v-model:value="embedText"
          type="textarea"
          :rows="5"
          placeholder="每行一条文本，例如：&#10;你好&#10;世界&#10;天使的记忆"
        />
        <n-space v-if="embedResult" vertical :size="8">
          <n-alert v-if="embedResult.error" type="error" :bordered="false">{{ embedResult.error }}</n-alert>
          <template v-else>
            <n-space size="small" align="center">
              <n-tag type="success" size="small" :bordered="false">总计 {{ embedResult.total }} 条 / {{ embedResult.batches }} 批</n-tag>
              <n-tag size="small" :bordered="false">耗时 {{ embedResult.elapsed_ms }}ms</n-tag>
              <n-tag size="small" :bordered="false">平均 {{ embedResult.avg_ms_per_text }}ms/条</n-tag>
              <n-tag size="small" :bordered="false">维度 {{ embedResult.dimension }}</n-tag>
              <n-tag v-if="embedResult.failed_batches" type="error" size="small" :bordered="false">失败 {{ embedResult.failed_batches }} 批</n-tag>
              <n-tag v-if="embedResult.timed_out_batches" type="warning" size="small" :bordered="false">超时 {{ embedResult.timed_out_batches }} 批</n-tag>
            </n-space>
            <n-space v-if="embedResult.provider" size="small">
              <span class="muted">provider: {{ embedResult.provider.provider_id || '—' }} / {{ embedResult.provider.model_name || '—' }} ({{ embedResult.provider.type || '—' }})</span>
            </n-space>
            <n-data-table
              :columns="embedColumns"
              :data="embedResult.batch_details || []"
              :pagination="false"
              size="small"
              :bordered="false"
            />
            <n-card v-if="embedResult.preview?.length" size="small" embedded title="预览（前 3 条向量片段）">
              <pre class="code-block">{{ JSON.stringify(embedResult.preview, null, 2) }}</pre>
            </n-card>
          </template>
        </n-space>
      </n-space>
    </n-card>

    <!-- 重排探针 -->
    <n-card title="重排探针（Rerank）" embedded>
      <n-space vertical :size="12">
        <n-input v-model:value="rerankQuery" placeholder="query，例如：天使的记忆" />
        <n-input
          v-model:value="rerankDocsText"
          type="textarea"
          :rows="5"
          placeholder="documents，每行一条，例如：&#10;记忆系统会记住你的偏好&#10;笔记系统会整理你的知识&#10;向量检索负责召回"
        />
        <n-space align="center" :wrap="true">
          <n-input-number v-model:value="rerankTimeout" :min="5" :max="120" style="width: 140px">
            <template #prefix>超时</template>
          </n-input-number>
          <n-input-number v-model:value="rerankMaxDocs" :min="1" :max="200" style="width: 150px">
            <template #prefix>候选上限</template>
          </n-input-number>
          <n-input-number v-model:value="rerankMaxTokens" :min="128" :max="8192" :step="128" style="width: 170px">
            <template #prefix>单段Token</template>
          </n-input-number>
          <n-button type="primary" :loading="rerankLoading" :disabled="!rerankQuery.trim() || !rerankDocsText.trim()" @click="doRerankProbe">测试重排</n-button>
          <span v-if="rerankHasProvider === false" class="muted">未配置重排提供商，将走 BM25 降级</span>
        </n-space>
        <n-alert v-if="rerankMaxDocs !== null && rerankMaxTokens !== null" type="default" :bordered="false" style="font-size: 12px; opacity: 0.7">
          仅本次生效，不改全局配置。默认 64 / 1024，bge-reranker-v2-m3 单对硬上限 8192
        </n-alert>
        <n-space v-if="rerankResult" vertical :size="8">
          <n-alert v-if="rerankResult.timed_out" type="warning" :bordered="false">重排超时（>{{ rerankResult.timeout }}s），已降级</n-alert>
          <n-alert v-if="rerankResult.error && !rerankResult.timed_out" :type="rerankResult.has_rerank ? 'error' : 'warning'" :bordered="false">{{ rerankResult.error }}</n-alert>
          <template v-if="rerankResult.scores?.length">
            <n-space size="small" :wrap="true">
              <n-tag size="small" :bordered="false">耗时 {{ rerankResult.elapsed_ms }}ms</n-tag>
              <n-tag size="small" :bordered="false">返回 {{ rerankResult.scores.length }} 条</n-tag>
              <n-tag v-if="rerankResult.provider_id" size="small" :bordered="false">{{ rerankResult.provider_id }}</n-tag>
              <n-tag v-if="rerankResult.original_docs !== undefined" size="small" :bordered="false">候选 {{ rerankResult.original_docs }}→{{ rerankResult.kept_docs }}/{{ rerankResult.rerank_max_docs }}</n-tag>
              <n-tag v-if="rerankResult.rerank_max_tokens_per_doc" size="small" :bordered="false">Token {{ rerankResult.rerank_max_tokens_per_doc }}</n-tag>
              <n-tag v-if="rerankResult.doc_truncated" type="warning" size="small" :bordered="false">截断 {{ rerankResult.doc_truncated }} 段</n-tag>
              <n-tag v-if="rerankResult.query_truncated" type="warning" size="small" :bordered="false">query已截断</n-tag>
            </n-space>
            <n-data-table
              :columns="rerankColumns"
              :data="rerankResult.scores"
              :pagination="false"
              size="small"
              :bordered="false"
            />
          </template>
          <n-empty v-else-if="!rerankResult.error && !rerankResult.timed_out" description="无重排结果（可能降级为空）" />
        </n-space>
      </n-space>
    </n-card>
  </n-space>
</template>

<script setup lang="ts">
import { h, ref } from 'vue'
import type { DataTableColumns } from 'naive-ui'
import {
  NAlert,
  NButton,
  NCard,
  NDataTable,
  NEmpty,
  NInput,
  NInputNumber,
  NSpace,
  NTag,
  useDialog,
  useMessage,
} from 'naive-ui'
import { useBridge } from '@/composables/useBridge'

const { apiPost } = useBridge()
const message = useMessage()
const dialog = useDialog()

// 嵌入
const embedText = ref('')
const embedBatchSize = ref<number | null>(50)
const embedTimeout = ref<number | null>(5)
const embedLoading = ref(false)
const embedResult = ref<any>(null)

const embedColumns: DataTableColumns<any> = [
  { title: '#', key: 'batch_index', width: 60, render: row => `#${row.batch_index}` },
  { title: 'batch', key: 'batch_size', width: 80 },
  { title: '耗时(ms)', key: 'elapsed_ms', width: 100 },
  { title: '维度', key: 'dimension', width: 80, render: row => row.dimension ?? '—' },
  {
    title: '状态',
    key: 'success',
    width: 80,
    render: row => h(NTag, { size: 'small', type: row.success ? 'success' : row.timed_out ? 'warning' : 'error', bordered: false }, { default: () => (row.success ? '成功' : row.timed_out ? '超时' : '失败') }),
  },
  { title: '错误', key: 'error', ellipsis: { tooltip: true }, render: row => row.error || '—' },
]

async function doEmbedProbe() {
  const texts = embedText.value.split('\n').map(s => s.trim()).filter(Boolean)
  if (!texts.length) {
    message.warning('请输入至少一条文本')
    return
  }
  const bs = embedBatchSize.value ?? 50
  if (bs > 200) {
    const ok = await new Promise<boolean>(resolve => {
      dialog.warning({
        title: '批次过大',
        content: `批次 ${bs} 较大，可能触发供应商 400/413/429 限流，是否继续？`,
        positiveText: '继续',
        negativeText: '取消',
        onPositiveClick: () => resolve(true),
        onNegativeClick: () => resolve(false),
        onClose: () => resolve(false),
      })
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
      message.error(data.error)
    } else {
      embedResult.value = data
      if (data.failed_batches) message.warning(`有 ${data.failed_batches} 批失败，请查看明细`)
      else message.success('嵌入探针完成')
    }
  } catch (e: any) {
    const msg = e.message || '嵌入探针失败'
    embedResult.value = { error: msg }
    message.error(msg)
  } finally {
    embedLoading.value = false
  }
}

// 重排
const rerankQuery = ref('')
const rerankDocsText = ref('')
const rerankTimeout = ref<number | null>(5)
const rerankMaxDocs = ref<number | null>(64)
const rerankMaxTokens = ref<number | null>(1024)
const rerankLoading = ref(false)
const rerankResult = ref<any>(null)
const rerankHasProvider = ref<boolean | null>(null)

const rerankColumns: DataTableColumns<any> = [
  { title: 'index', key: 'index', width: 70 },
  { title: 'score', key: 'score', width: 100, render: row => (typeof row.score === 'number' ? row.score.toFixed(4) : String(row.score ?? '—')) },
  { title: 'document', key: 'document', ellipsis: { tooltip: true } },
]

async function doRerankProbe() {
  if (!rerankQuery.value.trim() || !rerankDocsText.value.trim()) {
    message.warning('请填写 query 和 documents')
    return
  }
  const documents = rerankDocsText.value.split('\n').map(s => s.trim()).filter(Boolean)
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
      if (data.has_rerank === false) message.warning(data.error)
      else if (data.timed_out) message.warning(data.error)
      else message.error(data.error)
    } else {
      message.success('重排探针完成')
    }
  } catch (e: any) {
    const msg = e.message || '重排探针失败'
    rerankResult.value = { error: msg, has_rerank: true }
    message.error(msg)
  } finally {
    rerankLoading.value = false
  }
}
</script>

<style scoped>
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
