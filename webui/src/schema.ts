/** _conf_schema.json 单项元数据（前端渲染用最小集） */
export interface SchemaMeta {
  type?: string
  description?: string
  hint?: string
  items?: Record<string, SchemaMeta>
  _special?: string
  default?: unknown
  editor_language?: string
}
