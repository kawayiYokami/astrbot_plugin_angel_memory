/**
 * 全局主题系统：以 iOS 26 Liquid Glass 为基准。
 * - 光/暗两套 Naive UI themeOverrides（主色 = iOS 系统蓝）
 * - 圆角体系：xs=8 小件 / sm=10 控件 / md=14 卡片 / lg=20 弹窗与面板 / full=胶囊
 * - 玻璃材质变量在 App.vue 的 :root[data-theme] 中定义，供自定义类使用
 */
import type { GlobalThemeOverrides } from 'naive-ui'
import type { ComputedRef, InjectionKey, Ref } from 'vue'
import { computed, ref } from 'vue'

export type ThemeMode = 'light' | 'dark'

const THEME_STORAGE_KEY = 'angel-memory-theme'

export interface ThemeApi {
  mode: Ref<ThemeMode>
  isDark: ComputedRef<boolean>
  toggle: () => void
}

export const themeKey: InjectionKey<ThemeApi> = Symbol('app-theme')

/** 读取持久化主题；默认光。localStorage 不可用（隐私模式等）时静默回退 */
export function loadThemeMode(): ThemeMode {
  try {
    return localStorage.getItem(THEME_STORAGE_KEY) === 'dark' ? 'dark' : 'light'
  } catch {
    return 'light'
  }
}

function saveThemeMode(mode: ThemeMode) {
  try {
    localStorage.setItem(THEME_STORAGE_KEY, mode)
  } catch {
    /* 忽略持久化失败 */
  }
}

export function createThemeApi(): ThemeApi {
  const mode = ref<ThemeMode>(loadThemeMode())
  return {
    mode,
    isDark: computed(() => mode.value === 'dark'),
    toggle: () => {
      mode.value = mode.value === 'light' ? 'dark' : 'light'
      saveThemeMode(mode.value)
    },
  }
}

// iOS 系统蓝：光 #007AFF / 暗 #0A84FF
const brand = {
  light: { base: '#007aff', hover: '#3395ff', pressed: '#0062cc', suppl: '#3395ff' },
  dark: { base: '#0a84ff', hover: '#409cff', pressed: '#0069cc', suppl: '#409cff' },
}

/**
 * 构建对应模式的 Naive UI 覆盖：
 * - 主色与圆角走 common，全部组件统一生效
 * - cardColor 直接给玻璃底色（NCard 与 preset="card" 的 NModal 共用）
 * - Layout/Sider/Menu 底色置透明，露出页面的液态玻璃氛围层
 */
export function buildThemeOverrides(mode: ThemeMode): GlobalThemeOverrides {
  const c = brand[mode]
  const glassCard = mode === 'light' ? 'rgba(255, 255, 255, 0.88)' : 'rgba(44, 44, 46, 0.85)'
  // default 型按钮的玻璃胶囊底（iOS 26：无纯线框按钮）
  const btnGlass =
    mode === 'light'
      ? {
          color: 'rgba(255, 255, 255, 0.55)',
          colorHover: 'rgba(255, 255, 255, 0.78)',
          colorPressed: 'rgba(255, 255, 255, 0.5)',
          colorFocus: 'rgba(255, 255, 255, 0.55)',
          border: '1px solid rgba(255, 255, 255, 0.65)',
          borderHover: '1px solid rgba(255, 255, 255, 0.95)',
          borderPressed: '1px solid rgba(255, 255, 255, 0.5)',
          borderFocus: '1px solid rgba(255, 255, 255, 0.65)',
        }
      : {
          color: 'rgba(44, 44, 46, 0.5)',
          colorHover: 'rgba(72, 72, 76, 0.62)',
          colorPressed: 'rgba(40, 40, 42, 0.55)',
          colorFocus: 'rgba(44, 44, 46, 0.5)',
          border: '1px solid rgba(255, 255, 255, 0.14)',
          borderHover: '1px solid rgba(255, 255, 255, 0.28)',
          borderPressed: '1px solid rgba(255, 255, 255, 0.12)',
          borderFocus: '1px solid rgba(255, 255, 255, 0.14)',
        }
  return {
    common: {
      primaryColor: c.base,
      primaryColorHover: c.hover,
      primaryColorPressed: c.pressed,
      primaryColorSuppl: c.suppl,
      borderRadius: '10px',
      borderRadiusSmall: '8px',
      cardColor: glassCard,
    },
    Button: {
      // iOS 26 bordered 按钮 = 胶囊形，全部按钮统一
      borderRadius: '999px',
      ...btnGlass,
    },
    Card: {
      // embedded 卡片与常规卡同走玻璃底
      colorEmbedded: glassCard,
    },
    Layout: {
      color: 'transparent',
      siderColor: 'transparent',
      headerColor: 'transparent',
      footerColor: 'transparent',
    },
    Menu: {
      color: 'transparent',
    },
  }
}
