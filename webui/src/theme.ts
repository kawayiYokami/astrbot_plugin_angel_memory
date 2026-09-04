/**
 * 全局主题系统：实色卡片基准，对齐 angel_brush。
 * - A 色（边框/分隔线）唯一真相源：tokens.light/dark.divider
 * - 背景/侧栏/卡片三色统一：bgPage / bgSider / bgCard
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

const brand = {
  light: { base: '#007aff', hover: '#3395ff', pressed: '#0062cc', suppl: '#3395ff' },
  dark: { base: '#0a84ff', hover: '#409cff', pressed: '#0069cc', suppl: '#409cff' },
}

// A 色与面色：亮/暗各一套，theme.ts 为唯一真相源，App.vue 的 CSS 变量需与此对齐
const tokens = {
  light: {
    bgPage: '#F5F5F5',
    bgSider: '#ffffff',
    bgCard: '#ffffff',
    divider: 'rgba(60, 60, 67, 0.12)',
  },
  dark: {
    bgPage: '#101014',
    bgSider: '#18181c',
    bgCard: '#18181c',
    divider: 'rgba(84, 84, 88, 0.4)',
  },
} as const

export function buildThemeOverrides(mode: ThemeMode): GlobalThemeOverrides {
  const c = brand[mode]
  const t = tokens[mode]
  return {
    common: {
      primaryColor: c.base,
      primaryColorHover: c.hover,
      primaryColorPressed: c.pressed,
      primaryColorSuppl: c.suppl,
      fontSize: '15px',
      borderRadius: '10px',
      borderRadiusSmall: '8px',
      bodyColor: t.bgPage,
      cardColor: t.bgCard,
      modalColor: t.bgCard,
      popoverColor: t.bgCard,
      dividerColor: t.divider,
      borderColor: t.divider,
    },
    Button: {
      borderRadius: '999px',
    },
    Layout: {
      color: t.bgPage,
      siderColor: t.bgSider,
    },
    Card: {
      color: t.bgCard,
      colorEmbedded: t.bgCard,
      borderColor: t.divider,
    },
  }
}
