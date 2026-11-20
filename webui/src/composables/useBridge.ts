/**
 * AstrBot 插件页 Bridge 封装。
 *
 * 后端路由已带插件名前缀，这里 endpoint 传相对路径（如 "memes"）。
 *
 * 版本兼容：
 * - 4.30 起宿主注入 window.AstrBotPluginView（AstrBotPluginPage 为别名），
 *   context 里带 isDark / themeColors，并支持 onContext 订阅主题变化。
 * - 4.26~4.28 只有 window.AstrBotPluginPage，context 里没有 themeColors，
 *   也没有 onContext。此时主题固定亮色，其余能力照常可用。
 * - 两者都没有（独立开发模式）走直连 fetch 降级。
 */

export interface BridgeContext {
  pluginName?: string
  displayName?: string
  pageName?: string
  pageTitle?: string
  locale?: string
  i18n?: Record<string, unknown>
  isDark?: boolean
  themeColors?: Record<string, string> | null
}

type ContextHandler = (context: BridgeContext) => void

interface Bridge {
  ready(): Promise<BridgeContext>
  getContext?(): BridgeContext
  getLocale?(): string
  onContext?(handler: ContextHandler): (() => void) | void
  apiGet(endpoint: string, params?: Record<string, unknown>): Promise<unknown>
  apiPost(endpoint: string, body?: unknown): Promise<unknown>
  upload(endpoint: string, file: File): Promise<unknown>
  download?(endpoint: string, params?: Record<string, unknown>, filename?: string): Promise<void>
}

declare global {
  interface Window {
    AstrBotPluginView?: Bridge
    AstrBotPluginPage?: Bridge
  }
}

const PLUGIN_NAME = 'astrbot_plugin_angel_memory'

/** bridge.ready() 超时兜底：宿主没回 context 也不能把首帧卡住 */
const READY_TIMEOUT_MS = 4000

let bridgeContext: BridgeContext | null = null
let readyPromise: Promise<BridgeContext | null> | null = null

/** 新版优先、旧版别名兜底；每次调用都重新取，避免模块加载时机早于宿主注入 */
export function getBridge(): Bridge | null {
  if (typeof window === 'undefined') return null
  // 独立打开（无父窗口）时宿主不会响应 postMessage，必须走 fetch 降级
  if (window.parent === window) return null
  return window.AstrBotPluginView ?? window.AstrBotPluginPage ?? null
}

export function hasBridge(): boolean {
  return getBridge() !== null
}

// postMessage 结构化克隆不接受 Vue 的 Proxy（ref/reactive 包装值），过桥前先落为纯 JSON
function plain<T>(value: T): T {
  return value === undefined ? value : (JSON.parse(JSON.stringify(value)) as T)
}

/** 永不 reject、永不无限等待：超时或异常都按「拿不到上下文」处理 */
function settleWithin<T>(promise: Promise<T>, ms: number): Promise<T | null> {
  return new Promise((resolve) => {
    let done = false
    const finish = (value: T | null) => {
      if (done) return
      done = true
      resolve(value)
    }
    const timer = setTimeout(() => {
      console.warn('[Bridge] ready() 超时，按无主题上下文继续渲染')
      finish(null)
    }, ms)
    promise.then(
      (value) => {
        clearTimeout(timer)
        finish(value)
      },
      (error) => {
        clearTimeout(timer)
        console.warn('[Bridge] ready() 失败:', error)
        finish(null)
      },
    )
  })
}

export function useBridge() {
  /** 只初始化一次，重复调用共享同一个结果（含超时降级结果） */
  function init(): Promise<BridgeContext | null> {
    if (readyPromise) return readyPromise
    const bridge = getBridge()
    if (!bridge) {
      console.warn('[Bridge] 未检测到 AstrBotPluginView / AstrBotPluginPage，按独立开发模式运行')
      readyPromise = Promise.resolve(null)
      return readyPromise
    }
    readyPromise = settleWithin(
      Promise.resolve().then(() => bridge.ready()),
      READY_TIMEOUT_MS,
    ).then((context) => {
      bridgeContext = context
      return context
    })
    return readyPromise
  }

  /** 订阅宿主上下文变化（切亮暗、改主题色、切语言都会推）。老版本没有此能力，返回空解绑函数 */
  function onContext(handler: ContextHandler): () => void {
    const bridge = getBridge()
    if (!bridge || typeof bridge.onContext !== 'function') return () => {}
    try {
      const unsubscribe = bridge.onContext(handler)
      return typeof unsubscribe === 'function' ? unsubscribe : () => {}
    } catch (error) {
      console.warn('[Bridge] onContext 订阅失败:', error)
      return () => {}
    }
  }

  function getTheme(): { isDark: boolean; colors: Record<string, string> | null } {
    const context = bridgeContext ?? getBridge()?.getContext?.() ?? null
    const colors =
      context?.themeColors && typeof context.themeColors === 'object'
        ? context.themeColors
        : null
    return { isDark: context?.isDark === true, colors }
  }

  async function apiGet<T = unknown>(endpoint: string, params?: Record<string, unknown>): Promise<T> {
    const bridge = getBridge()
    if (bridge) {
      return (await bridge.apiGet(endpoint, plain(params))) as T
    }
    // 开发模式回退：直接请求本地代理
    const url = new URL(`/api/plug/${PLUGIN_NAME}/${endpoint}`, window.location.origin)
    if (params) {
      Object.entries(params).forEach(([k, v]) => {
        if (v !== undefined && v !== null) url.searchParams.set(k, String(v))
      })
    }
    const resp = await fetch(url.toString())
    const body = await resp.json()
    // 与 bridge 行为一致：成功返回 data，失败抛 message
    if (body?.status === 'error') {
      throw new Error(body.message || '请求失败')
    }
    return (body?.data ?? body) as T
  }

  async function apiPost<T = unknown>(endpoint: string, body?: unknown): Promise<T> {
    const bridge = getBridge()
    if (bridge) {
      return (await bridge.apiPost(endpoint, plain(body))) as T
    }
    const url = `/api/plug/${PLUGIN_NAME}/${endpoint}`
    const resp = await fetch(url, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(body),
    })
    const json = await resp.json()
    // 与 bridge 行为一致：成功返回 data，失败抛 message
    if (json?.status === 'error') {
      throw new Error(json.message || '请求失败')
    }
    return (json?.data ?? json) as T
  }

  async function upload<T = unknown>(endpoint: string, file: File): Promise<T> {
    const bridge = getBridge()
    if (bridge) {
      return (await bridge.upload(endpoint, file)) as T
    }
    const form = new FormData()
    form.append('file', file)
    const resp = await fetch(`/api/plug/${PLUGIN_NAME}/${endpoint}`, {
      method: 'POST',
      body: form,
    })
    const json = await resp.json()
    if (json?.status === 'error') {
      throw new Error(json.message || '上传失败')
    }
    return (json?.data ?? json) as T
  }

  async function download(endpoint: string, params?: Record<string, unknown>, filename?: string): Promise<void> {
    const bridge = getBridge()
    if (bridge?.download) {
      await bridge.download(endpoint, plain(params), filename)
      return
    }
    // 独立模式回退：fetch 取 blob 后用 <a download> 触发保存
    const url = new URL(`/api/plug/${PLUGIN_NAME}/${endpoint}`, window.location.origin)
    if (params) {
      Object.entries(params).forEach(([k, v]) => {
        if (v !== undefined && v !== null) url.searchParams.set(k, String(v))
      })
    }
    const resp = await fetch(url.toString())
    if (!resp.ok) throw new Error(`下载失败: ${resp.status}`)
    const blob = await resp.blob()
    const a = document.createElement('a')
    a.href = URL.createObjectURL(blob)
    a.download = filename || 'download'
    document.body.appendChild(a)
    a.click()
    a.remove()
    setTimeout(() => URL.revokeObjectURL(a.href), 1000)
  }

  return { init, onContext, getTheme, apiGet, apiPost, upload, download }
}
