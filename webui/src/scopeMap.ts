/**
 * 记忆分柜表的纯逻辑：方式（persona / group）显式存储在独立配置项里，
 * 表本身仍是扁平 map，两类键各自留存；界面只渲染当前方式那一半。
 */

export type ScopeMode = 'persona' | 'group'

export const DEFAULT_SCOPE_MODE: ScopeMode = 'persona'

/** 配置里读到的值可能是 undefined / 历史脏值，统一收敛成两种方式之一。 */
export function normalizeScopeMode(value: unknown): ScopeMode {
  return value === 'group' ? 'group' : DEFAULT_SCOPE_MODE
}

/** 会话ID 是「平台ID:消息类型:会话ID」三段式，人格名不含冒号。 */
export function belongsToMode(key: string, mode: ScopeMode): boolean {
  return key.includes(':') === (mode === 'group')
}

/** 取出当前方式可见的那一半键值。 */
export function pickScopeEntries(
  map: Record<string, string>,
  mode: ScopeMode,
): Array<[string, string]> {
  return Object.entries(map).filter(([key]) => belongsToMode(key, mode))
}

/**
 * 把编辑后的可见半边写回整张表：不属于当前方式的键原样保留，一个都不丢。
 * 合并基准是「当前完整表」——它自己的另一半从未被本函数改动过，
 * 所以连续切换、来回编辑都不会丢键。
 */
export function mergeScopeMap(
  fullMap: Record<string, string>,
  visible: Record<string, string>,
  mode: ScopeMode,
): Record<string, string> {
  const merged: Record<string, string> = {}
  for (const [key, value] of Object.entries(fullMap)) {
    if (!belongsToMode(key, mode)) merged[key] = value
  }
  for (const [key, value] of Object.entries(visible)) {
    merged[key] = value
  }
  return merged
}
