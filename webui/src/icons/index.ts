import { addCollection } from '@iconify/vue'
import { lucideOffline } from './lucide-offline'

let registered = false

/**
 * 注册内置 lucide 图标集合（构建期固化，运行时不发网络请求）。
 * 必须在任何 <Icon icon="lucide:xxx"> 渲染之前调用一次。
 */
export function registerLucideIcons(): void {
  if (registered) return
  registered = true
  addCollection(lucideOffline as never)
}
