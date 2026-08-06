/**
 * 通用格式化工具：时间戳 / tags 字符串解析。
 * 多个视图复用，避免重复实现。
 */

/** 时间戳（秒或毫秒）→ 本地时间字符串；空值返回 '-' */
export function formatTime(ts: number | null | undefined): string {
  if (!ts) return '-'
  let t = Number(ts)
  if (t > 1e11) t /= 1000
  return new Date(t * 1000).toLocaleString('zh-CN')
}

/** 逗号拼接的 tags 字符串 → 去空数组 */
export function parseTags(tags: string | null | undefined): string[] {
  if (!tags) return []
  return tags
    .split(',')
    .map(t => t.trim())
    .filter(Boolean)
}

/** 文件大小字节 → 可读字符串 */
export function formatSize(bytes: number | null | undefined): string {
  if (!bytes || bytes <= 0) return '-'
  const units = ['B', 'KB', 'MB', 'GB']
  let v = bytes
  let i = 0
  while (v >= 1024 && i < units.length - 1) {
    v /= 1024
    i++
  }
  return `${v.toFixed(v >= 100 ? 0 : 1)} ${units[i]}`
}
