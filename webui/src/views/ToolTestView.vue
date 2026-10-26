<template>
  <n-space vertical :size="16">
    <n-alert type="info" :bordered="false">
      这里手动跑两条检索链路，用来回答「为什么搜不出内容」。只读，不写记忆、不写笔记、不触发反馈与衰减。
      面板里没有真实的群消息事件，所以会话是替身：人格解析会退化为空，记忆域可能落到会话键或 public，结果以页面显示的命中来源为准。
    </n-alert>

    <n-card title="检索条件" embedded>
      <n-space vertical :size="12">
        <n-input
          v-model:value="query"
          type="textarea"
          :rows="3"
          placeholder="查询词，例如：钟离的护盾机制"
        />
        <n-space>
          <n-select
            v-model:value="groupId"
            :options="groupOptions"
            placeholder="群号（留空按私聊处理）"
            clearable
            filterable
            style="width: 260px"
          />
          <n-input
            v-model:value="umo"
            placeholder="会话 UMO（留空用示例）"
            style="width: 320px"
          />
        </n-space>
        <n-space>
          <n-button
            type="primary"
            :loading="ragLoading"
            :disabled="!query.trim()"
            @click="runAutoRag"
          >
            跑自动 RAG
          </n-button>
          <n-button
            :loading="recallLoading"
            :disabled="!query.trim()"
            @click="runActiveRecall"
          >
            跑主动搜索
          </n-button>
        </n-space>
      </n-space>
    </n-card>

    <n-tabs type="segment" animated>
      <!-- 自动 RAG -->
      <n-tab-pane name="rag" tab="自动 RAG">
        <n-space vertical :size="12">
          <n-alert v-if="ragError" type="error" :bordered="false">{{ ragError }}</n-alert>
          <n-alert v-else-if="ragResult?.error" type="error" :bordered="false">
            {{ ragResult.error }}
          </n-alert>

          <n-card v-if="ragResult" embedded>
            <n-space size="small" align="center">
              <n-tag :bordered="false" type="info">
                记忆域 {{ ragResult.memory_scope }}
              </n-tag>
              <n-tag :bordered="false">来源 {{ ragResult.matched_by }}</n-tag>
              <n-tag :bordered="false">{{ ragResult.elapsed_ms }}ms</n-tag>
              <n-tag :bordered="false" :type="ragResult.memories?.length ? 'success' : 'warning'">
                记忆 {{ ragResult.memories?.length || 0 }} 条
              </n-tag>
              <n-tag :bordered="false" :type="ragResult.notes?.length ? 'success' : 'warning'">
                笔记 {{ ragResult.notes?.length || 0 }} 条
              </n-tag>
            </n-space>

            <div class="meta-grid">
              <div class="meta-row">
                <span class="meta-label">预处理后查询词</span>
                <span class="meta-value">{{ ragResult.processed_query || '（空）' }}</span>
              </div>
              <div class="meta-row">
                <span class="meta-label">自动注入域</span>
                <span class="meta-value">{{ (ragResult.auto_domains || []).join('、') || '（无）' }}</span>
              </div>
              <div class="meta-row">
                <span class="meta-label">笔记召回</span>
                <span class="meta-value">
                  {{ ragResult.note_recall_enabled ? '已开启' : '已关闭' }} / 候选上限
                  {{ ragResult.note_candidate_top_k }} / 注入上限 {{ ragResult.note_inject_top_k }}
                </span>
              </div>
              <div class="meta-row">
                <span class="meta-label">解析会话</span>
                <span class="meta-value">{{ ragResult.conversation_id }}</span>
              </div>
            </div>

            <n-alert
              v-if="!ragResult.auto_domains?.length"
              type="warning"
              :bordered="false"
              style="margin-top: 12px"
            >
              这个群没有标记为「自动」的知识域，笔记侧必然为空。这是配置问题，不是检索坏了。
            </n-alert>
          </n-card>

          <n-card v-if="ragResult?.memories?.length" title="记忆召回" embedded>
            <n-list hoverable clickable>
              <n-list-item v-for="item in ragResult.memories" :key="item.id">
                <n-space vertical :size="4">
                  <n-space size="small" align="center">
                    <n-tag :bordered="false" size="small" type="info">{{ item.memory_type }}</n-tag>
                    <n-tag :bordered="false" size="small">{{ item.memory_scope }}</n-tag>
                    <n-tag :bordered="false" size="small">强度 {{ item.strength }}</n-tag>
                    <n-tag v-if="item.is_active" :bordered="false" size="small" type="success">主动</n-tag>
                  </n-space>
                  <div class="judgment">{{ item.judgment }}</div>
                  <div v-if="item.reasoning" class="muted">{{ item.reasoning }}</div>
                  <div v-if="item.tags?.length" class="muted">标签：{{ item.tags.join('、') }}</div>
                </n-space>
              </n-list-item>
            </n-list>
          </n-card>

          <n-card v-if="ragResult?.notes?.length" title="笔记候选" embedded>
            <n-list hoverable clickable>
              <n-list-item v-for="(note, index) in ragResult.notes" :key="index">
                <n-space vertical :size="4">
                  <n-space size="small" align="center">
                    <n-tag :bordered="false" size="small">{{ note.domain || '未知域' }}</n-tag>
                    <n-tag :bordered="false" size="small">ID {{ note.note_id }}</n-tag>
                    <n-tag :bordered="false" size="small">
                      相似度 {{ (note.similarity || 0).toFixed(3) }}
                    </n-tag>
                    <span class="muted">L{{ note.line_start }}-{{ note.line_end }}</span>
                  </n-space>
                  <div class="path">{{ note.path }}</div>
                  <div class="muted">{{ note.content }}</div>
                </n-space>
              </n-list-item>
            </n-list>
          </n-card>

          <n-card v-if="ragResult?.injection_preview" title="注入预览（笔记块 + 记忆块）" embedded>
            <pre class="code-pre">{{ ragResult.injection_preview }}</pre>
          </n-card>
        </n-space>
      </n-tab-pane>

      <!-- 主动搜索 -->
      <n-tab-pane name="recall" tab="主动搜索">
        <n-space vertical :size="12">
          <n-alert v-if="recallError" type="error" :bordered="false">{{ recallError }}</n-alert>
          <n-alert v-else-if="recallResult?.error" type="error" :bordered="false">
            {{ recallResult.error }}
          </n-alert>

          <n-card v-if="recallResult" embedded>
            <n-space size="small" align="center">
              <n-tag :bordered="false" type="info">记忆域 {{ recallResult.memory_scope }}</n-tag>
              <n-tag :bordered="false">来源 {{ recallResult.matched_by }}</n-tag>
              <n-tag :bordered="false">{{ recallResult.elapsed_ms }}ms</n-tag>
              <n-tag :bordered="false">{{ recallResult.chars || 0 }} 字符</n-tag>
              <n-tag :bordered="false" :type="recallResult.memory_count ? 'success' : 'warning'">
                记忆 {{ recallResult.memory_count ?? '—' }} 条
              </n-tag>
              <n-tag :bordered="false" :type="recallResult.note_file_count ? 'success' : 'warning'">
                笔记 {{ recallResult.note_file_count ?? '—' }} 个文件
              </n-tag>
            </n-space>

            <div class="meta-grid">
              <div class="meta-row">
                <span class="meta-label">可见域</span>
                <span class="meta-value">
                  {{ (recallResult.visible_domains || []).join('、') || '（无）' }}
                </span>
              </div>
            </div>
          </n-card>

          <n-card v-if="recallResult?.raw" title="工具返回原文" embedded>
            <pre class="code-pre">{{ recallResult.raw }}</pre>
          </n-card>
        </n-space>
      </n-tab-pane>
    </n-tabs>
  </n-space>
</template>

<script setup lang="ts">
import { computed, onMounted, ref } from 'vue'
import {
  NAlert,
  NButton,
  NCard,
  NInput,
  NList,
  NListItem,
  NSelect,
  NSpace,
  NTabPane,
  NTabs,
  NTag,
} from 'naive-ui'
import { useBridge } from '@/composables/useBridge'

interface GroupRow {
  group_id: string
  group_name?: string
}

const { apiGet, apiPost } = useBridge()

const query = ref('')
const groupId = ref<string | null>(null)
const umo = ref('')
const groups = ref<GroupRow[]>([])

const ragLoading = ref(false)
const ragResult = ref<any>(null)
const ragError = ref('')

const recallLoading = ref(false)
const recallResult = ref<any>(null)
const recallError = ref('')

const groupOptions = computed(() =>
  groups.value.map(g => ({
    label: g.group_name ? `${g.group_name}（${g.group_id}）` : g.group_id,
    value: g.group_id,
  })),
)

function basePayload() {
  return {
    query: query.value.trim(),
    group_id: groupId.value || '',
    umo: umo.value.trim(),
  }
}

async function runAutoRag() {
  ragLoading.value = true
  ragError.value = ''
  ragResult.value = null
  try {
    const resp: any = await apiPost('tool-test/auto-rag', basePayload())
    if (resp.status !== 'ok') {
      ragError.value = resp.message || '请求失败'
    } else {
      ragResult.value = resp.data
    }
  } catch (e: any) {
    ragError.value = e.message || '跑自动 RAG 失败'
  } finally {
    ragLoading.value = false
  }
}

async function runActiveRecall() {
  recallLoading.value = true
  recallError.value = ''
  recallResult.value = null
  try {
    const resp: any = await apiPost('tool-test/active-recall', basePayload())
    if (resp.status !== 'ok') {
      recallError.value = resp.message || '请求失败'
    } else {
      recallResult.value = resp.data
    }
  } catch (e: any) {
    recallError.value = e.message || '跑主动搜索失败'
  } finally {
    recallLoading.value = false
  }
}

async function loadGroups() {
  try {
    const resp: any = await apiGet('group-knowledge')
    if (resp.status === 'ok') {
      groups.value = resp.data?.groups || []
    }
  } catch (e) {
    /* 群列表拉不到不影响手动输入群号 */
  }
}

onMounted(() => loadGroups())
</script>

<style scoped>
.code-pre {
  background: rgba(0, 0, 0, 0.35);
  border-radius: 6px;
  padding: 12px;
  overflow-x: auto;
  white-space: pre-wrap;
  word-break: break-all;
  font-size: 12px;
  margin: 0;
}

.meta-grid {
  margin-top: 12px;
  display: flex;
  flex-direction: column;
  gap: 6px;
}

.meta-row {
  display: flex;
  gap: 12px;
  font-size: 12px;
  line-height: 1.6;
}

.meta-label {
  min-width: 96px;
  opacity: 0.6;
}

.meta-value {
  word-break: break-all;
}

.judgment {
  font-size: 13px;
  line-height: 1.6;
}

.path {
  font-size: 12px;
  word-break: break-all;
}

.muted {
  opacity: 0.65;
  font-size: 12px;
  line-height: 1.6;
}
</style>
