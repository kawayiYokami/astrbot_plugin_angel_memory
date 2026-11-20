<template>
  <div>
    <div v-if="loading" class="d-flex justify-center py-12">
      <v-progress-circular indeterminate color="primary" />
    </div>

    <div v-else class="d-flex flex-column ga-4">
      <!-- 统计卡片 -->
      <v-row>
        <v-col cols="12" sm="6" md="3">
          <v-card rounded="md">
            <v-card-text class="stat-card text-center">
              <div class="text-h5 font-weight-bold">{{ overview.memory_count ?? '-' }}</div>
              <div class="stat-label text-caption text-medium-emphasis">记忆总数</div>
            </v-card-text>
          </v-card>
        </v-col>
        <v-col cols="12" sm="6" md="3">
          <v-card rounded="md">
            <v-card-text class="stat-card text-center">
              <div class="text-h5 font-weight-bold">{{ overview.global_tag_count ?? '-' }}</div>
              <div class="stat-label text-caption text-medium-emphasis">全局标签</div>
            </v-card-text>
          </v-card>
        </v-col>
        <v-col cols="12" sm="6" md="3">
          <v-card rounded="md">
            <v-card-text class="stat-card text-center">
              <div class="text-h5 font-weight-bold">{{ overview.note_index_count ?? '-' }}</div>
              <div class="stat-label text-caption text-medium-emphasis">笔记索引</div>
            </v-card-text>
          </v-card>
        </v-col>
        <v-col cols="12" sm="6" md="3">
          <v-card rounded="md">
            <v-card-text class="stat-card text-center">
              <div class="text-h5 font-weight-bold">
                <v-icon
                  :icon="overview.has_providers ? 'lucide:circle-check' : 'lucide:triangle-alert'"
                  :color="overview.has_providers ? 'success' : 'warning'"
                  size="30"
                />
              </div>
              <div class="stat-label text-caption text-medium-emphasis">
                {{ overview.has_providers ? '提供商就绪' : '无提供商' }}
              </div>
            </v-card-text>
          </v-card>
        </v-col>
      </v-row>

      <v-row>
        <!-- 配置信息 -->
        <v-col cols="12" md="6">
          <v-card rounded="md">
            <v-card-text class="config-section">
              <div class="config-title">配置信息</div>
            </v-card-text>
            <v-card-text class="pt-0 d-flex flex-column ga-2">
              <div class="d-flex justify-space-between align-center ga-4">
                <span class="text-body-2 text-medium-emphasis text-no-wrap flex-shrink-0">嵌入模型</span>
                <span class="text-body-2 text-truncate min-w-0 text-right">{{ overview.provider_id || '-' }}</span>
              </div>
              <div class="d-flex justify-space-between align-center ga-4">
                <span class="text-body-2 text-medium-emphasis text-no-wrap flex-shrink-0">对话模型</span>
                <span class="text-body-2 text-truncate min-w-0 text-right">{{ overview.llm_provider_id || '-' }}</span>
              </div>
              <div class="d-flex justify-space-between align-center ga-4">
                <span class="text-body-2 text-medium-emphasis text-no-wrap flex-shrink-0">索引目录</span>
                <span class="text-body-2 text-truncate min-w-0 text-right" :title="overview.index_dir">{{ overview.index_dir || '-' }}</span>
              </div>
            </v-card-text>
          </v-card>
        </v-col>

        <!-- Scope 列表 -->
        <v-col cols="12" md="6">
          <v-card rounded="md">
            <v-card-text class="config-section">
              <div class="config-title">记忆分类域</div>
              <div class="config-hint">不同会话的记忆分开存放，互不干扰</div>
            </v-card-text>
            <v-card-text class="pt-0">
              <template v-if="overview.scopes?.length">
                <v-chip
                  v-for="scope in overview.scopes"
                  :key="scope"
                  color="primary"
                  variant="tonal"
                  size="small"
                  class="ma-1"
                >
                  {{ scopeLabel(scope) }}
                </v-chip>
              </template>
              <v-empty-state
                v-else
                icon="lucide:layers"
                title="还没有分类域"
                text="产生第一条记忆后自动出现"
              />
            </v-card-text>
          </v-card>
        </v-col>
      </v-row>

      <!-- 过程追踪记录 -->
      <v-card rounded="md">
        <v-card-text class="config-section">
          <div class="d-flex align-center ga-3">
            <div class="flex-grow-1 min-w-0">
              <div class="config-title">过程追踪记录（最近20次）</div>
            </div>
            <v-btn
              v-if="traceSelected.length"
              size="small"
              color="error"
              variant="tonal"
              :loading="traceDeleting"
              @click="deleteSelectedTraces"
            >
              删除选中（{{ traceSelected.length }}）
            </v-btn>
            <v-btn variant="text" size="small" :loading="traceLoading" prepend-icon="lucide:refresh-cw" @click="loadTraces">
              刷新
            </v-btn>
          </div>
        </v-card-text>
        <v-card-text class="pt-0 d-flex flex-column ga-3">
          <v-alert type="warning" variant="tonal" density="compact" class="text-body-2">
            保存在本机，不会自动清理；需要时勾选后自行删除。提示词含对话原文，仅本机调试用。
          </v-alert>
          <v-empty-state
            v-if="!traceRecords.length && !traceLoading"
            icon="lucide:scroll-text"
            title="暂无记录"
            text="触发几轮对话或等待睡眠后自动生成"
          />
          <v-card v-for="r in traceRecords" :key="r.id" variant="outlined" rounded="md">
            <v-card-text class="pb-1">
              <div class="d-flex align-center ga-2 flex-wrap">
                <v-checkbox
                  v-if="r.status !== 'pending'"
                  :model-value="traceSelected.includes(r.id)"
                  density="compact"
                  hide-details
                  @update:model-value="(checked: boolean | null) => toggleTraceSelect(r.id, checked === true)"
                />
                <v-chip :color="r.type === 'sleep' ? 'info' : undefined" size="small">
                  {{ traceTypeText(r.type) }}
                </v-chip>
                <v-chip :color="traceStatusColor(r.status)" size="small">
                  {{ traceStatusText(r.status) }}
                </v-chip>
                <span class="text-caption text-medium-emphasis">{{ r.time }}</span>
                <template v-if="r.type === 'sleep'">
                  <span class="text-caption text-medium-emphasis text-truncate" style="max-width: 260px;">
                    {{ traceMetaText(r) }}
                  </span>
                </template>
                <template v-else>
                  <v-chip v-if="chatKind(r.session_id)" size="small" variant="tonal">
                    {{ chatKind(r.session_id) }}
                  </v-chip>
                  <span
                    class="text-caption text-medium-emphasis text-truncate"
                    style="max-width: 260px;"
                    :title="r.session_id || ''"
                  >
                    {{ chatShortId(r.session_id) }}
                  </span>
                </template>
                <v-spacer />
                <v-btn variant="text" size="small" prepend-icon="lucide:copy" @click="copyTrace(r)">
                  复制
                </v-btn>
              </div>
            </v-card-text>
            <v-card-text class="pt-1 d-flex flex-column ga-3">
              <div>
                <div class="d-flex align-center justify-space-between ga-2 mb-1">
                  <span class="text-caption font-weight-medium text-medium-emphasis">用户提示词</span>
                  <v-btn-toggle :model-value="promptView(r.id)" density="compact" divided @update:model-value="(v: string) => setPromptView(r.id, v as 'render' | 'raw')">
                    <v-btn value="render" size="x-small">渲染</v-btn>
                    <v-btn value="raw" size="x-small">原文</v-btn>
                  </v-btn-toggle>
                </div>
                <pre v-if="promptView(r.id) === 'raw'" class="code-block">{{ r.prompt || '（无）' }}</pre>
                <div v-else class="md-body" v-html="renderMarkdown(r.prompt)"></div>
              </div>

              <div v-if="r.status !== 'pending'">
                <div class="d-flex align-center justify-space-between ga-2 mb-1">
                  <span class="text-caption font-weight-medium text-medium-emphasis">模型返回</span>
                  <v-btn-toggle :model-value="resultView(r.id)" density="compact" divided @update:model-value="(v: string) => setResultView(r.id, v as 'render' | 'raw')">
                    <v-btn value="render" size="x-small">渲染</v-btn>
                    <v-btn value="raw" size="x-small">原文</v-btn>
                  </v-btn-toggle>
                </div>
                <v-alert v-if="r.error" type="error" variant="tonal" density="compact" style="white-space: pre-wrap">
                  {{ r.error }}
                </v-alert>
                <pre v-else-if="resultView(r.id) === 'raw'" class="code-block">{{ rawResultText(r) }}</pre>
                <ReflectionResult v-else-if="reflectionView(r)" :view="reflectionView(r)!" />
                <div v-else class="md-body" v-html="renderResult(r)"></div>
              </div>

              <div v-if="r.summary">
                <div class="mb-1">
                  <span class="text-caption font-weight-medium text-medium-emphasis">摘要</span>
                </div>
                <div class="md-body" v-html="renderJson(r.summary)"></div>
              </div>
            </v-card-text>
          </v-card>
        </v-card-text>
      </v-card>
    </div>
  </div>
</template>

<script setup lang="ts">
import { ref, onMounted } from 'vue'
import { useBridge } from '@/composables/useBridge'
import { toast } from '@/composables/useToast'
import ReflectionResult from '@/components/ReflectionResult.vue'

const { apiGet, apiPost } = useBridge()

const loading = ref(true)
const overview = ref<Record<string, any>>({})

// 过程追踪记录
const traceRecords = ref<any[]>([])
const traceLoading = ref(false)
const traceDeleting = ref(false)
const traceSelected = ref<string[]>([])
const promptViews = ref<Record<string, 'render' | 'raw'>>({})
const resultViews = ref<Record<string, 'render' | 'raw'>>({})

function promptView(id: string | number) {
  return promptViews.value[String(id)] || 'render'
}

function resultView(id: string | number) {
  return resultViews.value[String(id)] || 'render'
}

function setPromptView(id: string | number, view: 'render' | 'raw') {
  promptViews.value[String(id)] = view
}

function setResultView(id: string | number, view: 'render' | 'raw') {
  resultViews.value[String(id)] = view
}

function escapeHtml(text: string) {
  return text
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;')
    .replace(/"/g, '&quot;')
}

function renderInline(text: string) {
  return escapeHtml(text)
    .replace(/`([^`]+)`/g, '<code>$1</code>')
    .replace(/\*\*([^*]+)\*\*/g, '<strong>$1</strong>')
}

/** 极简 Markdown 渲染：标题、列表、代码块、加粗、行内代码。 */
function renderMarkdown(text: any) {
  const source = String(text ?? '')
  if (!source.trim()) return '<p class="md-empty">（无）</p>'

  const html: string[] = []
  let listType: 'ul' | 'ol' | '' = ''
  let listItems: string[] = []
  let paragraph: string[] = []
  let codeLines: string[] | null = null

  const flushParagraph = () => {
    if (paragraph.length) {
      html.push(`<p>${paragraph.map(renderInline).join('<br>')}</p>`)
      paragraph = []
    }
  }
  const flushList = () => {
    if (listType) {
      html.push(`<${listType}>${listItems.map((item) => `<li>${item}</li>`).join('')}</${listType}>`)
      listType = ''
      listItems = []
    }
  }

  for (const line of source.split(/\r?\n/)) {
    if (codeLines !== null) {
      if (/^\s*```/.test(line)) {
        html.push(`<pre class="md-code">${escapeHtml(codeLines.join('\n'))}</pre>`)
        codeLines = null
      } else {
        codeLines.push(line)
      }
      continue
    }
    if (/^\s*```/.test(line)) {
      flushParagraph()
      flushList()
      codeLines = []
      continue
    }
    if (!line.trim()) {
      flushParagraph()
      flushList()
      continue
    }
    const heading = line.match(/^(#{1,6})\s+(.*)$/)
    if (heading) {
      flushParagraph()
      flushList()
      const level = heading[1].length
      html.push(`<h${level}>${renderInline(heading[2])}</h${level}>`)
      continue
    }
    const bullet = line.match(/^\s*[-*]\s+(.*)$/)
    if (bullet) {
      flushParagraph()
      if (listType !== 'ul') {
        flushList()
        listType = 'ul'
      }
      listItems.push(renderInline(bullet[1]))
      continue
    }
    const ordered = line.match(/^\s*\d+\.\s+(.*)$/)
    if (ordered) {
      flushParagraph()
      if (listType !== 'ol') {
        flushList()
        listType = 'ol'
      }
      listItems.push(renderInline(ordered[1]))
      continue
    }
    if (listType && /^\s{2,}\S/.test(line)) {
      const last = listItems.length - 1
      listItems[last] = `${listItems[last]}<br>${renderInline(line.trim())}`
      continue
    }
    flushList()
    paragraph.push(line)
  }
  flushParagraph()
  flushList()
  if (codeLines !== null) {
    html.push(`<pre class="md-code">${escapeHtml(codeLines.join('\n'))}</pre>`)
  }
  return html.join('')
}

/** 递归渲染 JSON 为纵向键值树：父字段独占一行，子字段缩进在下一行。 */
function renderJson(value: any): string {
  const scalar = (item: any): string => {
    if (item === null || item === undefined) return '<span class="json-null">null</span>'
    if (typeof item === 'string') return `<span class="json-str">${escapeHtml(item)}</span>`
    if (typeof item === 'number' || typeof item === 'boolean') {
      return `<span class="json-num">${escapeHtml(String(item))}</span>`
    }
    return `<span class="json-str">${escapeHtml(String(item))}</span>`
  }

  const isNested = (item: any): boolean => {
    if (item === null || typeof item !== 'object') return false
    return Array.isArray(item) ? item.length > 0 : Object.keys(item).length > 0
  }

  // 层级越深越淡，到第四层封底，再深也不继续变淡
  const DEPTH_OPACITY = [1, 0.88, 0.78, 0.7, 0.64]

  const render = (item: any, depth: number): string => {
    const opacity = DEPTH_OPACITY[Math.min(depth, DEPTH_OPACITY.length - 1)]
    const indent = `padding-left:${depth * 16}px;opacity:${opacity}`
    if (Array.isArray(item)) {
      if (!item.length) return `<div class="json-line" style="${indent}"><span class="json-null">[]</span></div>`
      return item
        .map((child) => {
          const head = `<div class="json-line" style="${indent}"><span class="json-punc">-</span>`
          if (isNested(child)) return `${head}</div>${render(child, depth + 1)}`
          return `${head} ${scalar(child)}</div>`
        })
        .join('')
    }
    if (item !== null && typeof item === 'object') {
      const entries = Object.entries(item)
      if (!entries.length) return `<div class="json-line" style="${indent}"><span class="json-null">{}</span></div>`
      return entries
        .map(([key, child]) => {
          const head =
            `<div class="json-line" style="${indent}">` +
            `<span class="json-key">${escapeHtml(key)}</span><span class="json-punc">:</span>`
          if (isNested(child)) return `${head}</div>${render(child, depth + 1)}`
          return `${head} ${scalar(child)}</div>`
        })
        .join('')
    }
    return `<div class="json-line" style="${indent}">${scalar(item)}</div>`
  }

  return `<div class="json-obj">${render(value, 0)}</div>`
}

type SoulAxis = { key: string; low: string; high: string; value: number; color: string }
type MemoryItemView = {
  type: string
  judgment: string
  reasoning: string
  tags: string[]
  sourceIds: string[]
}
type ReflectionGroup = { title: string; items: MemoryItemView[] }
type ReflectionView = { soul: SoulAxis[]; usefulIds: string[]; groups: ReflectionGroup[] }

const SOUL_AXES = [
  { key: 'joy', low: '悲', high: '喜', color: '#f0a020' },
  { key: 'anger', low: '恐', high: '怒', color: '#d03050' },
  { key: 'surprise', low: '意料中', high: '意外', color: '#2080f0' },
  { key: 'disgust', low: '认可', high: '厌恶', color: '#18a058' },
]

const ACTION_TITLES: Record<string, string> = {
  create: '生成记忆',
  merge: '合并记忆',
  updata: '更新记忆',
  update: '更新记忆',
}

const ACTION_ORDER = ['生成记忆', '合并记忆', '更新记忆']

function toText(value: any): string {
  if (typeof value === 'string') return value
  if (value === null || value === undefined) return ''
  return String(value)
}

function toTextList(value: any): string[] {
  return Array.isArray(value) ? value.map((item) => toText(item)).filter(Boolean) : []
}

/** 把反思返回的 JSON 翻译成可读结构；结构不认识就返回 null，交给原文渲染。 */
function parseReflection(r: any): ReflectionView | null {
  const data = r && r.result
  if (!data || typeof data !== 'object' || Array.isArray(data)) return null

  const soulRaw = data.soul_state
  const feedback = data.feedback_data && typeof data.feedback_data === 'object' ? data.feedback_data : null
  if (!soulRaw && !feedback) return null

  const soul: SoulAxis[] = []
  if (soulRaw && typeof soulRaw === 'object') {
    for (const axis of SOUL_AXES) {
      const raw = Number(soulRaw[axis.key])
      if (!Number.isFinite(raw)) continue
      soul.push({
        key: axis.key,
        low: axis.low,
        high: axis.high,
        value: Math.min(1, Math.max(0, raw)),
        color: axis.color,
      })
    }
  }

  const usefulIds = feedback ? toTextList(feedback.useful_memory_ids) : []

  const buckets: Record<string, MemoryItemView[]> = {}
  const actions = feedback ? feedback.memory_actions : null
  if (Array.isArray(actions)) {
    for (const action of actions) {
      if (!action || typeof action !== 'object') continue
      const title = ACTION_TITLES[toText(action.action).toLowerCase()]
      const memory = action.memory
      if (!title || !memory || typeof memory !== 'object') continue
      if (!buckets[title]) buckets[title] = []
      buckets[title].push({
        type: toText(memory.type) || '未分类',
        judgment: toText(memory.judgment),
        reasoning: toText(memory.reasoning),
        tags: toTextList(memory.tags),
        sourceIds: toTextList(action.source_memory_ids),
      })
    }
  }

  const groups: ReflectionGroup[] = []
  for (const title of ACTION_ORDER) {
    if (buckets[title] && buckets[title].length) groups.push({ title, items: buckets[title] })
  }

  if (!soul.length && !usefulIds.length && !groups.length) return null
  return { soul, usefulIds, groups }
}

const reflectionCache = new WeakMap<object, ReflectionView | null>()

function reflectionView(r: any): ReflectionView | null {
  if (!r || typeof r !== 'object') return null
  if (reflectionCache.has(r)) return reflectionCache.get(r) ?? null
  const view = parseReflection(r)
  reflectionCache.set(r, view)
  return view
}

function rawResultText(r: any) {
  if (r.response) return String(r.response)
  if (r.result !== undefined && r.result !== null) return JSON.stringify(r.result, null, 2)
  return '（无）'
}

function renderResult(r: any) {
  if (r.result !== undefined && r.result !== null) return renderJson(r.result)
  if (r.response) return renderMarkdown(r.response)
  return '<p class="md-empty">无返回内容</p>'
}

function traceTypeText(type: string) {
  return type === 'sleep' ? '睡眠' : '反思'
}

// 记忆分类域：只有 public 是英文内置值，其余是用户自己的柜子名，原样显示
function scopeLabel(scope: string): string {
  if (String(scope).toLowerCase() === 'public') return '公开'
  return String(scope)
}

// 会话标识：冒号分隔的原始结构（如 default:GroupMessage:123）不直接显示，
// 只显示尾号，类型解析成「群聊 / 私聊」标签，原文收进 title tooltip 方便排查。
function chatKind(chatId: string): string {
  if (!chatId) return ''
  const segments = String(chatId).split(':').filter(Boolean)
  const typeSeg = segments.length >= 2 ? segments[segments.length - 2] : ''
  if (/^(FriendMessage|PrivateMessage)$/i.test(typeSeg)) return '私聊'
  if (/^GroupMessage$/i.test(typeSeg)) return '群聊'
  if (/^OtherMessage$/i.test(typeSeg)) return '其他'
  return ''
}

function chatShortId(chatId: string): string {
  if (!chatId) return '无会话ID'
  const segments = String(chatId).split(':').filter(Boolean)
  return segments[segments.length - 1] || String(chatId)
}

function traceMetaText(r: any) {
  if (r.type === 'sleep') {
    const s = r.summary || {}
    return `候选 ${s.candidates ?? '-'}　判删 ${s.deleted ?? '-'}　判留 ${s.promoted ?? '-'}`
  }
  return r.session_id || '无会话ID'
}

function traceStatusColor(status: string) {
  if (status === 'success') return 'success'
  if (status === 'failed') return 'error'
  return 'warning'
}

function traceStatusText(status: string) {
  if (status === 'success') return '成功'
  if (status === 'failed') return '失败'
  return '处理中'
}

function toggleTraceSelect(id: string, checked: boolean) {
  if (checked) {
    if (!traceSelected.value.includes(id)) traceSelected.value.push(id)
  } else {
    traceSelected.value = traceSelected.value.filter((item) => item !== id)
  }
}

async function loadTraces() {
  traceLoading.value = true
  try {
    const data: any = await apiGet('trace/reflection')
    traceRecords.value = Array.isArray(data && data.records) ? data.records : []
    traceSelected.value = traceSelected.value.filter((id) =>
      traceRecords.value.some((r) => r.id === id && r.status !== 'pending'),
    )
  } catch (e: any) {
    toast.error(e.message || '获取过程记录失败')
  } finally {
    traceLoading.value = false
  }
}

async function deleteSelectedTraces() {
  if (!traceSelected.value.length) return
  traceDeleting.value = true
  try {
    const data: any = await apiPost('trace/delete', { ids: traceSelected.value })
    if (data && data.ok) {
      toast.success(`已删除 ${data.deleted ?? 0} 条记录`)
    } else {
      toast.error((data && data.note) || '删除失败')
    }
    traceSelected.value = []
    await loadTraces()
  } catch (e: any) {
    toast.error(e.message || '删除失败')
  } finally {
    traceDeleting.value = false
  }
}

async function copyTrace(r: any) {
  try {
    await navigator.clipboard.writeText(JSON.stringify(r, null, 2))
    toast.success('已复制整条记录')
  } catch {
    toast.error('复制失败，请手动选择文本')
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
  loadTraces()
})
</script>

<style scoped>
.config-title {
  font-size: 0.875rem;
  font-weight: 500;
  color: rgb(var(--v-theme-primaryText));
}

/* 统计卡：内边距收窄给文字让位，label 强制不换行，避免被挤成半截 */
.stat-card {
  padding: 16px !important;
}
.stat-label {
  margin-top: 4px;
  white-space: nowrap;
  overflow: visible;
}
.config-hint {
  font-size: 0.75rem;
  color: rgb(var(--v-theme-secondaryText));
  margin-top: 2px;
}

.code-block {
  background: rgba(var(--v-theme-on-surface), 0.06);
  border-radius: 6px;
  padding: 12px;
  overflow-x: auto;
  white-space: pre-wrap;
  font-size: 12px;
  line-height: 1.4;
}

.md-body {
  font-size: 12px;
  line-height: 1.6;
  word-break: break-word;
}

.md-body :deep(p) {
  margin: 0 0 6px;
}

.md-body :deep(p:last-child) {
  margin-bottom: 0;
}

.md-body :deep(h1),
.md-body :deep(h2),
.md-body :deep(h3),
.md-body :deep(h4),
.md-body :deep(h5),
.md-body :deep(h6) {
  margin: 10px 0 6px;
  font-size: 13px;
  font-weight: 700;
}

.md-body :deep(h1:first-child) {
  margin-top: 0;
}

.md-body :deep(ul),
.md-body :deep(ol) {
  margin: 0 0 6px;
  padding-left: 20px;
}

.md-body :deep(li) {
  margin-bottom: 2px;
}

.md-body :deep(code) {
  background: rgba(127, 127, 127, 0.18);
  border-radius: 3px;
  padding: 0 3px;
  font-size: 12px;
}

.md-body :deep(.md-code) {
  background: rgba(var(--v-theme-on-surface), 0.06);
  border-radius: 6px;
  padding: 10px;
  margin: 0 0 6px;
  overflow-x: auto;
  white-space: pre-wrap;
  font-size: 12px;
  line-height: 1.4;
}

.md-body :deep(.md-code code) {
  background: none;
  padding: 0;
}

.md-empty {
  opacity: 0.6;
}

.md-body :deep(.json-obj) {
  display: block;
}

.md-body :deep(.json-line) {
  line-height: 1.7;
  white-space: pre-wrap;
  word-break: break-word;
}

.md-body :deep(.json-key) {
  font-weight: 600;
}

.md-body :deep(.json-punc) {
  opacity: 0.5;
  margin-right: 4px;
}

.md-body :deep(.json-null) {
  opacity: 0.5;
}

.md-body :deep(.json-num),
.md-body :deep(.json-str) {
  white-space: pre-wrap;
}
</style>
