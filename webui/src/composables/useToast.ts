/**
 * 轻量提示队列。不引 Vuetify 的 snackbar provider 整套，只维护一个队列，
 * 由 ToastHost.vue 逐条用 v-snackbar 呈现。
 */
import { readonly, ref } from 'vue'

export type ToastKind = 'success' | 'error' | 'warning' | 'info'

export interface ToastItem {
  id: number
  kind: ToastKind
  text: string
}

const queue = ref<ToastItem[]>([])
let seq = 0

function push(kind: ToastKind, text: string) {
  queue.value = [...queue.value, { id: ++seq, kind, text }]
}

export const toast = {
  success: (text: string) => push('success', text),
  error: (text: string) => push('error', text),
  warning: (text: string) => push('warning', text),
  info: (text: string) => push('info', text),
}

export function useToastQueue() {
  const items = readonly(queue)
  function shift() {
    queue.value = queue.value.slice(1)
  }
  return { items, shift }
}
