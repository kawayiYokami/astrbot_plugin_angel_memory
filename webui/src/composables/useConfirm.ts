/**
 * 二次确认对话框。替代原来的气泡确认：
 * 用 Promise 形式调用，由 ConfirmHost.vue 统一渲染一个 v-dialog。
 */
import { ref } from 'vue'

export interface ConfirmOptions {
  title?: string
  text: string
  confirmText?: string
  cancelText?: string
  /** 危险操作用红色确认按钮 */
  danger?: boolean
}

export interface ConfirmState extends ConfirmOptions {
  id: number
}

const state = ref<ConfirmState | null>(null)
let resolver: ((confirmed: boolean) => void) | null = null
let seq = 0

export function confirmDialog(options: ConfirmOptions): Promise<boolean> {
  // 同一时刻只允许一个确认框，新的请求先结掉旧的
  resolver?.(false)
  state.value = { ...options, id: ++seq }
  return new Promise<boolean>((resolve) => {
    resolver = resolve
  })
}

export function useConfirmState() {
  function settle(confirmed: boolean) {
    state.value = null
    const done = resolver
    resolver = null
    done?.(confirmed)
  }
  return { state, settle }
}
