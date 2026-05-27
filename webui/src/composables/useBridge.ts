/**
 * AstrBot Plugin Page Bridge API 封装
 */

interface BridgeContext {
  pluginName: string
  displayName: string
  pageName: string
  pageTitle: string
  locale: string
  i18n: Record<string, unknown>
}

interface Bridge {
  ready(): Promise<BridgeContext>
  getContext(): BridgeContext
  getLocale(): string
  apiGet(endpoint: string, params?: Record<string, unknown>): Promise<unknown>
  apiPost(endpoint: string, body?: unknown): Promise<unknown>
  upload(endpoint: string, file: File): Promise<unknown>
  download(endpoint: string, params?: Record<string, unknown>, filename?: string): Promise<void>
}

declare global {
  interface Window {
    AstrBotPluginPage: Bridge
  }
}

let bridgeReady = false
let bridgeContext: BridgeContext | null = null

function getBridge(): Bridge | null {
  if (typeof window !== 'undefined' && window.AstrBotPluginPage) {
    return window.AstrBotPluginPage
  }
  return null
}

export function useBridge() {
  const bridge = getBridge()

  async function init(): Promise<BridgeContext | null> {
    if (bridgeReady && bridgeContext) return bridgeContext
    if (!bridge) {
      console.warn('[Bridge] AstrBotPluginPage 不可用，可能在独立开发模式下运行')
      return null
    }
    try {
      bridgeContext = await bridge.ready()
      bridgeReady = true
      return bridgeContext
    } catch (e) {
      console.error('[Bridge] 初始化失败:', e)
      return null
    }
  }

  async function apiGet<T = unknown>(endpoint: string, params?: Record<string, unknown>): Promise<T> {
    if (bridge) {
      return (await bridge.apiGet(endpoint, params)) as T
    }
    // 开发模式回退：直接请求本地代理
    const url = new URL(`/api/plug/astrbot_plugin_angel_memory/${endpoint}`, window.location.origin)
    if (params) {
      Object.entries(params).forEach(([k, v]) => {
        if (v !== undefined && v !== null) url.searchParams.set(k, String(v))
      })
    }
    const resp = await fetch(url.toString())
    return resp.json()
  }

  async function apiPost<T = unknown>(endpoint: string, body?: unknown): Promise<T> {
    if (bridge) {
      return (await bridge.apiPost(endpoint, body)) as T
    }
    const url = `/api/plug/astrbot_plugin_angel_memory/${endpoint}`
    const resp = await fetch(url, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(body),
    })
    return resp.json()
  }

  async function upload<T = unknown>(endpoint: string, file: File): Promise<T> {
    if (bridge) {
      return (await bridge.upload(endpoint, file)) as T
    }
    const url = `/api/plug/astrbot_plugin_angel_memory/${endpoint}`
    const formData = new FormData()
    formData.append('file', file)
    const resp = await fetch(url, { method: 'POST', body: formData })
    return resp.json()
  }

  async function download(endpoint: string, params?: Record<string, unknown>, filename?: string): Promise<void> {
    if (bridge) {
      await bridge.download(endpoint, params, filename)
      return
    }
    const url = new URL(`/api/plug/astrbot_plugin_angel_memory/${endpoint}`, window.location.origin)
    if (params) {
      Object.entries(params).forEach(([k, v]) => {
        if (v !== undefined && v !== null) url.searchParams.set(k, String(v))
      })
    }
    const resp = await fetch(url.toString())
    const blob = await resp.blob()
    const a = document.createElement('a')
    a.href = URL.createObjectURL(blob)
    a.download = filename || 'download.json'
    a.click()
    URL.revokeObjectURL(a.href)
  }

  return { init, apiGet, apiPost, upload, download }
}
