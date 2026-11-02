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

/** 会话分柜映射的可选来源：人格名 / 群聊会话ID / 柜子名 */
export interface ScopeCandidates {
  personas?: string[]
  groups?: Array<{ group_id: string; group_name: string; session_id: string }>
  scopes?: string[]
}
