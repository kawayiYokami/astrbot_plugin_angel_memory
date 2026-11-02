<template>
  <div class="schema-form" :style="cssVars">
    <!-- 分段卡片：每张 n-card 一个分组，标题硬分段，卡内字段软分段 -->
    <n-card
      v-for="group in groups"
      :key="group.key"
      :title="group.title"
      embedded
      class="group-card"
    >
      <template v-if="group.canReset" #header-extra>
        <n-button size="tiny" quaternary @click.stop="resetGroup(group)">
          <template #icon><Icon icon="lucide:rotate-ccw" /></template>
          恢复默认
        </n-button>
      </template>

      <p v-if="group.subtitle" class="group-desc">{{ group.subtitle }}</p>

      <template v-for="row in group.rows" :key="rowKey(row)">
        <!-- 二级子分区（组内嵌套 object）：小标题软分段 -->
        <div v-if="row.kind === 'subsection'" class="subsection-title">{{ row.title }}</div>

        <div v-else class="field-row" :class="{ 'is-block': row.block }">
          <div class="field-main">
            <div class="field-label">{{ row.label }}</div>
            <div v-if="row.hint" class="field-hint">{{ row.hint }}</div>
          </div>
          <div class="field-control">
            <!-- provider 下拉：显式标记或 *_provider_id 且有可选数据 -->
            <n-select
              v-if="isProviderSelect(row.meta, row.path)"
              :value="asString(getIn(row.path))"
              :options="providerOptions(row.path)"
              clearable
              placeholder="选择模型提供商"
              class="control-select"
              @update:value="(v: unknown) => setIn(row.path, v ?? '')"
            />
            <div v-else-if="isScopeMap(row.meta, row.path)" class="control-scope">
              <div class="scope-head">
                <n-radio-group
                  :value="currentScopeMode"
                  size="small"
                  @update:value="(v: string) => setScopeMode(v as ScopeMode)"
                >
                  <n-radio-button value="persona">按人格</n-radio-button>
                  <n-radio-button value="group">按群聊</n-radio-button>
                </n-radio-group>
                <span class="scope-mode-hint">{{ scopeModeHint }}</span>
              </div>
              <n-dynamic-input
                :value="scopeRows(row.path)"
                :on-create="kvCreate"
                class="control-kv"
                @update:value="(rows: unknown[]) => scopeWrite(row.path, rows)"
              >
                <template #default="{ value }">
                  <div class="kv-row">
                    <n-select
                      class="kv-key"
                      filterable
                      :value="value.key || null"
                      :options="scopeKeyOptions(row.path)"
                      :placeholder="scopeKeyPlaceholder"
                      @update:value="(v: string) => { value.key = v ?? ''; scopeSync(row.path) }"
                    />
                    <n-select
                      class="kv-value"
                      filterable
                      tag
                      :value="value.value || null"
                      :options="scopeValueOptions(value.value)"
                      placeholder="柜子名（可输入新建）"
                      @update:value="(v: string) => { value.value = v ?? ''; scopeSync(row.path) }"
                    />
                  </div>
                </template>
              </n-dynamic-input>
            </div>
            <n-dynamic-input
              v-else-if="isKvMap(row.meta, row.path)"
              :value="kvRowsOf(row.path)"
              :on-create="kvCreate"
              class="control-kv"
              @update:value="(rows: unknown[]) => kvWrite(row.path, rows)"
            >
              <template #default="{ value }">
                <div class="kv-row">
                  <n-input
                    class="kv-key"
                    :value="value.key"
                    placeholder="人格名 / 会话ID"
                    @update:value="(v: string) => { value.key = v; kvSync(row.path) }"
                  />
                  <n-input
                    class="kv-value"
                    :value="value.value"
                    placeholder="记忆分类域"
                    @update:value="(v: string) => { value.value = v; kvSync(row.path) }"
                  />
                </div>
              </template>
            </n-dynamic-input>
            <n-switch
              v-else-if="row.meta.type === 'bool'"
              :value="!!getIn(row.path)"
              @update:value="(v: unknown) => setIn(row.path, v)"
            />
            <n-input-number
              v-else-if="row.meta.type === 'int'"
              :value="asNumber(getIn(row.path))"
              :precision="0"
              class="control-number"
              @update:value="(v: unknown) => setIn(row.path, v)"
            />
            <n-input-number
              v-else-if="row.meta.type === 'float'"
              :value="asNumber(getIn(row.path))"
              :step="0.1"
              class="control-number"
              @update:value="(v: unknown) => setIn(row.path, v)"
            />
            <n-dynamic-tags
              v-else-if="row.meta.type === 'list'"
              :value="asList(getIn(row.path))"
              @update:value="(v: unknown) => setIn(row.path, v)"
            />
            <n-input
              v-else-if="row.meta.type === 'text'"
              :value="asString(getIn(row.path))"
              type="textarea"
              :rows="3"
              @update:value="(v: unknown) => setIn(row.path, v)"
            />
            <n-input
              v-else
              :value="asString(getIn(row.path))"
              class="control-text"
              @update:value="(v: unknown) => setIn(row.path, v)"
            />
          </div>
        </div>
      </template>
    </n-card>
  </div>
</template>

<script setup lang="ts">
import { computed } from 'vue'
import {
  NButton,
  NCard,
  NDynamicInput,
  NDynamicTags,
  NInput,
  NInputNumber,
  NRadioButton,
  NRadioGroup,
  NSelect,
  NSwitch,
  useThemeVars,
} from 'naive-ui'
import type { SelectGroupOption, SelectOption } from 'naive-ui'
import { Icon } from '@iconify/vue'
import type { SchemaMeta, ScopeCandidates } from './schema'
import {
  belongsToMode,
  mergeScopeMap,
  normalizeScopeMode,
  pickScopeEntries,
  type ScopeMode,
} from './scopeMap'

interface FieldRow {
  kind: 'field'
  path: string[]
  label: string
  hint?: string
  meta: SchemaMeta
  block: boolean
}

interface SubsectionRow {
  kind: 'subsection'
  title: string
}

type CardRow = FieldRow | SubsectionRow

interface Group {
  key: string
  title: string
  subtitle?: string
  canReset: boolean
  rows: CardRow[]
}

const props = defineProps<{
  schema: Record<string, SchemaMeta>
  modelValue?: Record<string, unknown>
  providers?: Record<string, string[]>
  scopeCandidates?: ScopeCandidates
}>()

const emit = defineEmits<{
  (e: 'update:modelValue', value: Record<string, unknown>): void
}>()

const themeVars = useThemeVars()

// naive 未暴露为 CSS 变量的文字/分隔色，从主题变量注入本组件作用域
const cssVars = computed(() => ({
  '--sf-title': themeVars.value.textColor1,
  '--sf-muted': themeVars.value.textColor3,
  '--sf-divider': themeVars.value.dividerColor,
}))

// 去掉 schema description 里的「[分组] 」前缀，UI 不需要重复标签
function cleanLabel(text: string | undefined): string {
  return (text ?? '').replace(/^\[[^\]]+\]\s*/, '')
}

function buildField(prefix: string[], key: string, meta: SchemaMeta): FieldRow {
  const type = meta.type ?? 'string'
  // 宽控件独占整行：多行文本与标签编辑器
  const block = type === 'text' || type === 'list'
  return {
    kind: 'field',
    path: [...prefix, key],
    label: cleanLabel(meta.description) || key,
    hint: meta.hint,
    meta,
    block,
  }
}

function rowKey(row: CardRow): string {
  return row.kind === 'field' ? row.path.join('.') : row.title
}

// 顶层叶子归入「常规」组置顶；顶层 object 各成一卡；二级 object 成为卡内子分区
const groups = computed<Group[]>(() => {
  const schema = props.schema || {}
  const rootRows: CardRow[] = []
  const sections: Group[] = []

  for (const [key, meta] of Object.entries(schema)) {
    if (!meta || typeof meta !== 'object') continue
    // 明确标记为 hidden 的字段不渲染：该配置另有专门入口（如「群聊知识库」页面）。
    // 值仍留在 modelValue 里随保存原样提交，不因为没渲染而丢失。
    if (meta._special === 'hidden') continue
    if (meta.type === 'object') {
      const rows: CardRow[] = []
      for (const [subKey, subMeta] of Object.entries(meta.items || {})) {
        if (subMeta?.type === 'object') {
          rows.push({ kind: 'subsection', title: cleanLabel(subMeta.description) || subKey })
          for (const [k, m] of Object.entries(subMeta.items || {})) {
            rows.push(buildField([key, subKey], k, m))
          }
        } else {
          rows.push(buildField([key], subKey, subMeta))
        }
      }
      sections.push({
        key,
        title: cleanLabel(meta.description) || key,
        subtitle: meta.hint,
        canReset: false,
        rows,
      })
    } else {
      // 分柜方式由分柜表上方的切换按钮读写，不单独占一行
      if (meta._special === 'scope_mode') continue
      rootRows.push(buildField([], key, meta))
    }
  }

  const result: Group[] = []
  if (rootRows.length) {
    // 「常规」组不提供恢复默认（顶层字段无统一默认语义）
    result.push({ key: '__root__', title: '常规', canReset: false, rows: rootRows })
  }
  for (const s of sections) {
    s.canReset = s.rows.some((r) => r.kind === 'field' && r.meta.default !== undefined)
    result.push(s)
  }
  return result
})

// 恢复默认：组内所有带 default 的字段重置为 schema 默认值；改动后由父级 dirty 检测浮出保存按钮
function resetGroup(group: Group) {
  for (const row of group.rows) {
    if (row.kind === 'field' && row.meta.default !== undefined) {
      setIn(row.path, row.meta.default)
    }
  }
}

function getIn(path: string[]): unknown {
  let cur: unknown = props.modelValue
  for (const k of path) {
    if (cur && typeof cur === 'object') cur = (cur as Record<string, unknown>)[k]
    else return undefined
  }
  return cur
}

function setIn(path: string[], value: unknown) {
  const root: Record<string, unknown> = { ...(props.modelValue || {}) }
  let cur = root
  for (let i = 0; i < path.length - 1; i++) {
    const next = cur[path[i]]
    cur[path[i]] = next && typeof next === 'object' ? { ...(next as Record<string, unknown>) } : {}
    cur = cur[path[i]] as Record<string, unknown>
  }
  cur[path[path.length - 1]] = value
  emit('update:modelValue', root)
}

function asNumber(v: unknown): number | null {
  return typeof v === 'number' ? v : null
}

function asString(v: unknown): string {
  return typeof v === 'string' ? v : ''
}

function asList(v: unknown): string[] {
  return Array.isArray(v) ? v.map(String) : []
}

// ---------- JSON 映射字段：行式键值编辑器替代手写 JSON ----------
interface KvRow {
  key: string
  value: string
}

const kvDrafts = new Map<string, { source: string; rows: KvRow[] }>()

function parseKvObject(v: unknown): Record<string, string> | null {
  if (typeof v !== 'string') return null
  try {
    const parsed = JSON.parse(v) as unknown
    return parsed && typeof parsed === 'object' && !Array.isArray(parsed)
      ? (parsed as Record<string, string>)
      : null
  } catch {
    return null
  }
}

// schema 标记 editor_language=json 且当前值为合法 JSON 对象字符串时启用；解析失败回退文本框
function isKvMap(meta: SchemaMeta, path: string[]): boolean {
  return meta.editor_language === 'json' && parseKvObject(getIn(path)) !== null
}

function kvRowsOf(path: string[]): KvRow[] {
  const id = path.join('.')
  const source = asString(getIn(path))
  let draft = kvDrafts.get(id)
  if (!draft || draft.source !== source) {
    draft = {
      source,
      rows: Object.entries(parseKvObject(getIn(path)) || {}).map(([key, value]) => ({
        key,
        value: String(value),
      })),
    }
    kvDrafts.set(id, draft)
  }
  return draft.rows
}

const kvCreate = (): KvRow => ({ key: '', value: '' })

function kvWrite(path: string[], rows: ReadonlyArray<unknown>) {
  const obj: Record<string, string> = {}
  for (const row of rows) {
    const r = row as Partial<KvRow>
    if (typeof r.key === 'string' && r.key.trim()) obj[r.key.trim()] = String(r.value ?? '')
  }
  const source = JSON.stringify(obj)
  kvDrafts.set(path.join('.'), { source, rows: rows as KvRow[] })
  setIn(path, source)
}

// 行内输入直接 mutate 行对象（不触发 update:value），输入变化后手动序列化写回
function kvSync(path: string[]) {
  const draft = kvDrafts.get(path.join('.'))
  if (draft) kvWrite(path, draft.rows)
}

// ---------- 会话分柜表：方式存在独立配置项里，列表只渲染当前方式那一半 ----------

function isScopeMap(meta: SchemaMeta, path: string[]): boolean {
  return meta._special === 'scope_map' && parseKvObject(getIn(path)) !== null
}

function scopeMapKeys(path: string[]): string[] {
  return Object.keys(parseKvObject(getIn(path)) || {})
}

// 分柜方式字段由 schema 标记定位，不在表单里单独成行（下面这张表自带切换按钮）
const scopeModePath = computed<string[]>(() => {
  const found = Object.entries(props.schema || {}).find(
    ([, meta]) => meta?._special === 'scope_mode',
  )
  return found ? [found[0]] : ['conversation_scope_mode']
})

const currentScopeMode = computed<ScopeMode>(() => normalizeScopeMode(getIn(scopeModePath.value)))

const scopeModeHint = computed(() =>
  currentScopeMode.value === 'group'
    ? '按群号识别会话，只显示会话键；人格键保留不参与匹配'
    : '按人格名识别会话，只显示人格键；群聊键保留不参与匹配',
)

const scopeKeyPlaceholder = computed(() =>
  currentScopeMode.value === 'group' ? '选择群聊' : '选择人格',
)

// 只负责改方式：一条数据都不动，另一半留着
function setScopeMode(next: ScopeMode) {
  setIn(scopeModePath.value, next)
}

// 行数据按「方式 + 当前表内容」缓存，保证输入过程中行对象不被重建
const scopeDrafts = new Map<string, { source: string; rows: KvRow[] }>()

function scopeRows(path: string[]): KvRow[] {
  const id = path.join('.')
  const mode = currentScopeMode.value
  const source = `${mode}|${asString(getIn(path))}`
  let draft = scopeDrafts.get(id)
  if (!draft || draft.source !== source) {
    draft = {
      source,
      rows: pickScopeEntries(parseKvObject(getIn(path)) || {}, mode).map(([key, value]) => ({
        key,
        value: String(value),
      })),
    }
    scopeDrafts.set(id, draft)
  }
  return draft.rows
}

// 写回时把当前方式的可见半边与另一半合并，隐藏侧一个键都不丢
function scopeWrite(path: string[], rows: ReadonlyArray<unknown>) {
  const mode = currentScopeMode.value
  const visible: Record<string, string> = {}
  for (const row of rows) {
    const r = row as Partial<KvRow>
    const key = String(r.key ?? '').trim()
    if (!key || !belongsToMode(key, mode)) continue
    visible[key] = String(r.value ?? '')
  }
  const merged = mergeScopeMap(parseKvObject(getIn(path)) || {}, visible, mode)
  const source = JSON.stringify(merged)
  scopeDrafts.set(path.join('.'), { source: `${mode}|${source}`, rows: rows as KvRow[] })
  setIn(path, source)
}

// 行内输入直接 mutate 行对象（不触发 update:value），变化后手动合并写回
function scopeSync(path: string[]) {
  const draft = scopeDrafts.get(path.join('.'))
  if (draft) scopeWrite(path, draft.rows)
}

function scopeKeyOptions(path: string[]): Array<SelectOption | SelectGroupOption> {
  const candidates = props.scopeCandidates || {}
  const base: SelectOption[] =
    currentScopeMode.value === 'group'
      ? (candidates.groups || []).map((g) => ({
          // 标签带「群」前缀，与「原配置」里裸群号的老键区分开
          label: g.group_name ? `${g.group_name}（${g.group_id}）` : `群 ${g.group_id}`,
          value: g.session_id,
        }))
      : (candidates.personas || []).map((name) => ({ label: name, value: name }))

  // 当前方式的候选之外的键（含裸群号等历史写法）单独成组，保证能看见、能重选
  const known = new Set(base.map((item) => item.value))
  const legacy = scopeMapKeys(path).filter(
    (key) => belongsToMode(key, currentScopeMode.value) && !known.has(key),
  )
  if (!legacy.length) return base
  return [
    ...base,
    {
      type: 'group',
      key: 'legacy',
      label: '原配置（不在当前方式的候选里）',
      children: legacy.map((key) => ({ label: key, value: key })),
    },
  ]
}

function scopeValueOptions(current: string): SelectOption[] {
  const scopes = props.scopeCandidates?.scopes || []
  const names = current && !scopes.includes(current) ? [...scopes, current] : scopes
  return names.map((name) => ({ label: name, value: name }))
}

// provider 字段判定：显式标记，或 string 类型的 *_provider_id 键且对应列表非空
function isProviderSelect(meta: SchemaMeta, path: string[]): boolean {
  if (meta._special === 'select_provider') return true
  if (meta.type && meta.type !== 'string') return false
  const key = path[path.length - 1]
  if (!/^(.*_)?provider_id$/.test(key)) return false
  const p = props.providers || {}
  const ids = /embed/i.test(key) ? p.embedding : /rerank/i.test(key) ? p.rerank : p.chat
  return !!ids?.length
}

function providerOptions(path: string[]) {
  const key = path[path.length - 1]
  const p = props.providers || {}
  const ids = /embed/i.test(key)
    ? p.embedding || []
    : /rerank/i.test(key)
      ? p.rerank || []
      : p.chat || []
  // 留空在不同字段上语义不同：重排留空是复用核心模型，嵌入留空是不做语义检索
  const emptyLabel = /embed/i.test(key)
    ? '(留空，只用关键词查找)'
    : /rerank/i.test(key)
      ? '(留空，复用核心模型)'
      : '(留空)'
  return [{ label: emptyLabel, value: '' }, ...ids.map((id) => ({ label: id, value: id }))]
}
</script>

<style scoped>
.schema-form {
  display: flex;
  flex-direction: column;
  gap: 12px;
}

/* 卡头硬分段：标题区与内容区之间的分隔线 */
.group-card :deep(.n-card-header) {
  border-bottom: 1px solid var(--sf-divider);
  padding-bottom: 12px;
}

.group-subtitle {
  font-size: 12px;
  font-weight: 400;
  color: var(--sf-muted);
}

/* 分组说明：卡体顶部小字，不进卡头（长文案会挤压标题） */
.group-desc {
  margin: 0 0 6px;
  font-size: 12px;
  color: var(--sf-muted);
  line-height: 1.6;
}

/* ---------- 软分段：卡内字段行，标签/hint 居左，控件右贴 ---------- */

.field-row {
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 20px;
  padding: 10px 0;
}

.field-row + .field-row,
.subsection-title + .field-row {
  border-top: 1px solid var(--sf-divider);
}

.field-main {
  flex: 1;
  min-width: 0;
}

.field-label {
  font-size: 14px;
  color: var(--sf-title);
  line-height: 1.45;
}

.field-hint {
  margin-top: 3px;
  font-size: 12px;
  color: var(--sf-muted);
  line-height: 1.55;
}

/* 控件列：不伸展、右贴；具体控件各自定宽 */
.field-control {
  flex-shrink: 0;
  display: flex;
  align-items: center;
}

.control-number {
  width: 150px;
}

.control-select {
  width: 260px;
}

.control-text {
  width: 260px;
}

/* ---------- 宽控件（多行文本/标签编辑）：独占整行 ---------- */

.field-row.is-block {
  flex-direction: column;
  align-items: stretch;
  gap: 10px;
}

.field-row.is-block .field-control {
  width: 100%;
}

/* ---------- JSON 映射编辑器 ---------- */

.control-kv {
  width: 100%;
}

.control-scope {
  width: 100%;
}

/* 分柜方式切换：先选方式，再编辑下方列表 */
.scope-head {
  display: flex;
  align-items: center;
  gap: 10px;
  margin-bottom: 10px;
}

.scope-mode-hint {
  font-size: 12px;
  color: var(--sf-muted);
  line-height: 1.6;
}

.kv-row {
  flex: 1;
  display: flex;
  gap: 8px;
}

.kv-key {
  flex: 1.5;
}

.kv-value {
  flex: 1;
}

/* ---------- 二级子分区标题 ---------- */

.subsection-title {
  padding: 14px 0 6px;
  font-size: 14px;
  font-weight: 600;
  color: var(--sf-title);
}
</style>
