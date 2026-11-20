<template>
  <div class="schema-form">
    <!-- 分段卡片：每张 v-card 一个分组，标题硬分段，卡内字段软分段 -->
    <v-card
      v-for="group in groups"
      :key="group.key"
      rounded="md"
      style="margin-bottom: 16px; padding-bottom: 8px;"
      class="group-card"
    >
      <v-card-text class="config-section" style="padding-bottom: 8px;">
        <div class="d-flex align-center ga-3">
          <div class="flex-grow-1 min-w-0">
            <div class="config-title text-truncate">{{ group.title }}</div>
            <div v-if="group.subtitle" class="config-hint">{{ group.subtitle }}</div>
          </div>
          <v-btn
            v-if="group.canReset"
            variant="text"
            size="small"
            prepend-icon="lucide:rotate-ccw"
            @click="resetGroup(group)"
          >
            恢复默认
          </v-btn>
        </div>
      </v-card-text>

      <div class="object-config">
        <div class="config-item">
          <template v-for="row in group.rows" :key="rowKey(row)">
            <!-- 二级子分区（组内嵌套 object）：小标题软分段 -->
            <div v-if="row.kind === 'subsection'" class="subsection-title">{{ row.title }}</div>

            <template v-else>
              <v-row class="config-row">
                <v-col cols="12" :sm="row.block ? 12 : 6" class="property-info">
                  <v-list-item density="compact">
                    <v-list-item-title class="property-name">{{ row.label }}</v-list-item-title>
                    <v-list-item-subtitle v-if="row.hint" class="property-hint">{{ row.hint }}</v-list-item-subtitle>
                  </v-list-item>
                </v-col>
                <v-col cols="12" :sm="row.block ? 12 : 6" class="config-input">
                  <!-- 服务商选择 -->
                  <v-select
                    v-if="isProviderSelect(row.meta, row.path)"
                    :model-value="asString(getIn(row.path))"
                    :items="providerOptions(row.path)"
                    clearable
                    placeholder="选择模型服务商"
                    variant="outlined"
                    density="compact"
                    hide-details
                    class="config-field"
                    @update:model-value="(v: unknown) => setIn(row.path, v ?? '')"
                  />
                  <!-- 会话分柜表：方式切换 + 行式键值编辑 -->
                  <div v-else-if="isScopeMap(row.meta, row.path)" class="scope-editor">
                    <div class="d-flex align-center ga-2 mb-2 flex-wrap">
                      <v-radio-group
                        :model-value="currentScopeMode"
                        inline
                        density="compact"
                        hide-details
                        @update:model-value="(v: string | null) => setScopeMode((v ?? 'persona') as ScopeMode)"
                      >
                        <v-radio label="按人格" value="persona" />
                        <v-radio label="按群聊" value="group" />
                      </v-radio-group>
                      <span class="property-hint">{{ scopeModeHint }}</span>
                    </div>
                    <div
                      v-for="(kv, kvIdx) in scopeRows(row.path)"
                      :key="kvIdx"
                      class="d-flex align-center ga-2 mb-2"
                    >
                      <v-autocomplete
                        :model-value="kv.key || null"
                        :items="scopeKeyOptions(row.path)"
                        :placeholder="scopeKeyPlaceholder"
                        variant="outlined"
                        density="compact"
                        hide-details
                        class="config-field flex-grow-1"
                        @update:model-value="(v: string) => { kv.key = v ?? ''; scopeSync(row.path) }"
                      />
                      <v-combobox
                        :model-value="kv.value || null"
                        :items="scopeValueOptions(kv.value)"
                        placeholder="柜子名（可输入新建）"
                        variant="outlined"
                        density="compact"
                        hide-details
                        class="config-field flex-grow-1"
                        @update:model-value="(v: unknown) => { kv.value = Array.isArray(v) ? (v[0] ?? '') : (v ?? ''); scopeSync(row.path) }"
                      />
                      <v-btn
                        variant="text"
                        size="small"
                        icon="lucide:x"
                        color="error"
                        :aria-label="'删除第 ' + (kvIdx + 1) + ' 组'"
                        @click="scopeRemove(row.path, kvIdx)"
                      />
                    </div>
                    <v-btn variant="text" size="small" prepend-icon="lucide:plus" @click="scopeAdd(row.path)">
                      添加一组
                    </v-btn>
                  </div>
                  <!-- JSON 映射：行式键值编辑器替代手写 JSON -->
                  <div v-else-if="isKvMap(row.meta, row.path)" class="kv-editor">
                    <div
                      v-for="(kv, kvIdx) in kvRowsOf(row.path)"
                      :key="kvIdx"
                      class="d-flex align-center ga-2 mb-2"
                    >
                      <v-text-field
                        :model-value="kv.key"
                        placeholder="人格名 / 会话 ID"
                        variant="outlined"
                        density="compact"
                        hide-details
                        class="config-field flex-grow-1"
                        @update:model-value="(v: string) => { kv.key = v; kvSync(row.path) }"
                      />
                      <v-text-field
                        :model-value="kv.value"
                        placeholder="记忆分类域"
                        variant="outlined"
                        density="compact"
                        hide-details
                        class="config-field flex-grow-1"
                        @update:model-value="(v: string) => { kv.value = v; kvSync(row.path) }"
                      />
                      <v-btn
                        variant="text"
                        size="small"
                        icon="lucide:x"
                        color="error"
                        :aria-label="'删除第 ' + (kvIdx + 1) + ' 组'"
                        @click="kvRemove(row.path, kvIdx)"
                      />
                    </div>
                    <v-btn variant="text" size="small" prepend-icon="lucide:plus" @click="kvAdd(row.path)">
                      添加一组
                    </v-btn>
                  </div>
                  <v-switch
                    v-else-if="row.meta.type === 'bool'"
                    :model-value="!!getIn(row.path)"
                    color="primary"
                    hide-details
                    density="compact"
                    inset
                    @update:model-value="(v: unknown) => setIn(row.path, v)"
                  />
                  <v-text-field
                    v-else-if="row.meta.type === 'int' || row.meta.type === 'float'"
                    :model-value="asNumber(getIn(row.path))"
                    type="number"
                    :step="row.meta.type === 'float' ? 0.1 : 1"
                    variant="outlined"
                    density="compact"
                    hide-details
                    class="config-field"
                    @update:model-value="(v: unknown) => setIn(row.path, v === '' || v === null || v === undefined ? null : Number(v))"
                  />
                  <v-combobox
                    v-else-if="row.meta.type === 'list'"
                    :model-value="asList(getIn(row.path))"
                    multiple
                    chips
                    closable-chips
                    variant="outlined"
                    density="compact"
                    hide-details
                    class="config-field"
                    @update:model-value="(v: unknown) => setIn(row.path, v)"
                  />
                  <v-textarea
                    v-else-if="row.meta.type === 'text'"
                    :model-value="asString(getIn(row.path))"
                    rows="3"
                    auto-grow
                    variant="outlined"
                    density="compact"
                    hide-details
                    class="config-field"
                    @update:model-value="(v: unknown) => setIn(row.path, v)"
                  />
                  <v-text-field
                    v-else
                    :model-value="asString(getIn(row.path))"
                    variant="outlined"
                    density="compact"
                    hide-details
                    class="config-field"
                    @update:model-value="(v: unknown) => setIn(row.path, v)"
                  />
                </v-col>
              </v-row>
              <v-divider class="config-divider" />
            </template>
          </template>
        </div>
      </div>
    </v-card>
  </div>
</template>

<script setup lang="ts">
import { computed } from 'vue'
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

// 去掉 schema description 里的「[分组] 」前缀，界面不需要重复标签
function cleanLabel(text: string | undefined): string {
  return (text ?? '').replace(/^\[[^\]]+\]\s*/, '')
}

function buildField(prefix: string[], key: string, meta: SchemaMeta): FieldRow {
  const type = meta.type ?? 'string'
  // 宽控件独占整行：多行文本、标签编辑器与键值编辑器
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
    // 明确标记为 hidden 的字段不渲染：该配置另有专门入口。
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
    result.push({ key: '__root__', title: '常规', canReset: false, rows: rootRows })
  }
  for (const s of sections) {
    s.canReset = s.rows.some((r) => r.kind === 'field' && r.meta.default !== undefined)
    result.push(s)
  }
  return result
})

// 恢复默认：组内所有带默认值的字段重置为 schema 默认值；改动后由父级 dirty 检测浮出保存按钮
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

function kvWrite(path: string[], rows: ReadonlyArray<KvRow>) {
  const obj: Record<string, string> = {}
  for (const r of rows) {
    if (typeof r.key === 'string' && r.key.trim()) obj[r.key.trim()] = String(r.value ?? '')
  }
  const source = JSON.stringify(obj)
  kvDrafts.set(path.join('.'), { source, rows: rows as KvRow[] })
  setIn(path, source)
}

// 行内输入直接改行对象，输入变化后手动序列化写回
function kvSync(path: string[]) {
  const draft = kvDrafts.get(path.join('.'))
  if (draft) kvWrite(path, draft.rows)
}

function kvAdd(path: string[]) {
  const rows = kvRowsOf(path)
  kvWrite(path, [...rows, { key: '', value: '' }])
}

function kvRemove(path: string[], idx: number) {
  const rows = kvRowsOf(path)
  kvWrite(path, rows.filter((_, i) => i !== idx))
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
function scopeWrite(path: string[], rows: ReadonlyArray<KvRow>) {
  const mode = currentScopeMode.value
  const visible: Record<string, string> = {}
  for (const r of rows) {
    const key = String(r.key ?? '').trim()
    if (!key || !belongsToMode(key, mode)) continue
    visible[key] = String(r.value ?? '')
  }
  const merged = mergeScopeMap(parseKvObject(getIn(path)) || {}, visible, mode)
  const source = JSON.stringify(merged)
  scopeDrafts.set(path.join('.'), { source: `${mode}|${source}`, rows: rows as KvRow[] })
  setIn(path, source)
}

// 行内输入直接改行对象，变化后手动合并写回
function scopeSync(path: string[]) {
  const draft = scopeDrafts.get(path.join('.'))
  if (draft) scopeWrite(path, draft.rows)
}

function scopeAdd(path: string[]) {
  const rows = scopeRows(path)
  scopeWrite(path, [...rows, { key: '', value: '' }])
}

function scopeRemove(path: string[], idx: number) {
  const rows = scopeRows(path)
  scopeWrite(path, rows.filter((_, i) => i !== idx))
}

function scopeKeyOptions(path: string[]) {
  const candidates = props.scopeCandidates || {}
  const base: Array<{ title: string; value: string }> =
    currentScopeMode.value === 'group'
      ? (candidates.groups || []).map((g) => ({
          // 标签带「群」前缀，与裸群号的老键区分开
          title: g.group_name ? `${g.group_name}（${g.group_id}）` : `群 ${g.group_id}`,
          value: g.session_id,
        }))
      : (candidates.personas || []).map((name) => ({ title: name, value: name }))

  // 当前方式的候选之外的键（含裸群号等历史写法）照常保留，保证能看见、能重选
  const known = new Set(base.map((item) => item.value))
  const legacy = scopeMapKeys(path).filter(
    (key) => belongsToMode(key, currentScopeMode.value) && !known.has(key),
  )
  return [...base, ...legacy.map((key) => ({ title: `原配置：${key}`, value: key }))]
}

function scopeValueOptions(current: string) {
  const scopes = props.scopeCandidates?.scopes || []
  const names = current && !scopes.includes(current) ? [...scopes, current] : scopes
  return names.map((name) => ({ title: name, value: name }))
}

// 服务商字段判定：显式标记，或 string 类型的 *_provider_id 键且对应列表非空
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
  // 留空在不同字段上语义不同：嵌入留空是不做语义检索，重排留空是复用核心模型
  const emptyTitle = /embed/i.test(key)
    ? '(留空，只用关键词查找)'
    : /rerank/i.test(key)
      ? '(留空，复用核心模型)'
      : '(留空)'
  return [{ title: emptyTitle, value: '' }, ...ids.map((id) => ({ title: id, value: id }))]
}
</script>

<style scoped>
.schema-form {
  display: flex;
  flex-direction: column;
}

/* ---- 上游 AstrBotConfigV4 行式范式（照抄数值；颜色用 rgb() 包裹主题三元组） ---- */
.v-application .config-section,
.config-section {
  margin-bottom: 4px;
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
.object-config {
  width: 100%;
}
.config-row {
  margin: 0;
  align-items: center;
  padding: 8px 8px;
  border-radius: 4px;
}
.config-row:hover {
  background-color: rgba(0, 0, 0, 0.03);
}
.property-info {
  padding: 0;
}
.property-name {
  font-size: 0.875rem;
  color: rgb(var(--v-theme-primaryText));
}
.property-hint {
  font-size: 0.75rem;
  color: rgb(var(--v-theme-secondaryText));
  margin-top: 2px;
}
.config-input {
  padding: 4px 8px;
}
.config-field {
  margin-bottom: 0;
}
.config-divider {
  border-color: rgba(0, 0, 0, 0.1);
  margin-left: 24px;
}

/* 键值/分柜编辑器占满右列 */
.kv-editor,
.scope-editor {
  width: 100%;
}

/* 二级子分区标题 */
.subsection-title {
  padding: 14px 16px 6px;
  font-size: 0.875rem;
  font-weight: 500;
  color: rgb(var(--v-theme-primaryText));
}

.group-card {
  border-color: rgba(var(--v-border-color), 0.16);
}
</style>
