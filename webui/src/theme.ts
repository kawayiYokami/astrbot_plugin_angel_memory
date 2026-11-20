/**
 * Vuetify 主题层：与宿主 AstrBot WebUI 同源。
 *
 * 三条规则：
 * 1. 挂载时用「上游 dashboard 亮色配色」作为默认主题同步创建，保证首帧就有完整样式。
 * 2. 拿到宿主的 themeColors 后，用宿主 Vuetify 主题的 colors 整体重建当前主题。
 * 3. 拿不到 themeColors（旧版宿主 / 独立开发模式）就固定亮色，不猜 prefers-color-scheme。
 */
import { defineComponent, h } from 'vue'
import { createVuetify } from 'vuetify'
import { zhHans } from 'vuetify/locale'
import type { ThemeDefinition } from 'vuetify'
import { Icon } from '@iconify/vue'

export const THEME_NAME = 'astrbot'

/**
 * 插件页自己的画布底色：整页（侧栏 + 内容 + 卡片）统一用这一色，
 * 刻意与上游 dashboard 的背景区分开（上游亮色 #ffffff、暗色 #1a1a1a）。
 * 语义色（primary / error / success 等）仍跟随宿主，只有画布用固定档位。
 */
const CANVAS_COLORS = {
  light: '#f5f6f8',
  dark: '#141619',
} as const

/**
 * 卡片表面色：比页面亮一档，靠色差让卡片从页面里浮出来。
 * 参考上游比例（暗色 #1a1a1a → #242424）。
 */
const SURFACE_COLORS = {
  light: '#ffffff',
  dark: '#1e2126',
} as const

/**
 * 上游 4.30 dashboard 亮色主题（dashboard/src/theme/LightTheme.ts）配色。
 * 用作默认主题与降级主题，让插件页在旧版宿主里看起来仍是原生页。
 */
const FALLBACK_LIGHT_COLORS: Record<string, string> = {
  primary: '#3c96ca',
  secondary: '#2f86bd',
  info: '#03c9d7',
  success: '#00c853',
  warning: '#ffc107',
  error: '#f44336',
  background: '#ffffff',
  surface: '#ffffff',
  primaryText: '#1b1c1d',
  secondaryText: '#000000aa',
  border: '#d0d0d0',
  containerBg: '#fffffff4',
  lightprimary: '#eef2f6',
  lightsecondary: '#e8f3fa',
  gray100: '#fafafacc',
}

/** 把 @iconify/vue 的 Icon 接成 Vuetify 图标集组件 */
const IconifyIcon = defineComponent({
  name: 'IconifyIcon',
  props: {
    icon: { type: [String, Object, Function, Array], default: '' },
    tag: { type: String, default: 'i' },
  },
  setup(props) {
    return () => {
      const name = typeof props.icon === 'string' ? props.icon : ''
      if (!name) return null
      return h(Icon, { icon: name.includes(':') ? name : `lucide:${name}` })
    }
  },
})

/** Vuetify 内部用到的 $alias，全部映射到内置的 lucide 图标，避免依赖 @mdi/font */
const ICON_ALIASES: Record<string, string> = {
  collapse: 'chevron-up',
  expand: 'chevron-down',
  dropdown: 'chevron-down',
  close: 'x',
  clear: 'x-circle',
  delete: 'x-circle',
  complete: 'check',
  success: 'circle-check',
  info: 'info',
  warning: 'triangle-alert',
  error: 'circle-alert',
  checkboxOn: 'check',
  checkboxOff: 'square',
  checkboxIndeterminate: 'minus',
  radioOn: 'circle-check',
  radioOff: 'circle',
  plus: 'plus',
  minus: 'minus',
  upload: 'upload',
  file: 'paperclip',
  edit: 'pencil',
  prev: 'chevron-left',
  next: 'chevron-right',
  menu: 'menu',
  sortAsc: 'chevron-up',
  sortDesc: 'chevron-down',
}

export interface HostThemeInput {
  isDark: boolean
  /** 宿主 Vuetify 主题的 colors 对象；缺失表示宿主不提供主题下发 */
  colors?: Record<string, string> | null
}

/**
 * 把宿主主题映射成一份 Vuetify ThemeDefinition。
 * Vuetify 会把这份定义 mergeDeep 进自己的默认亮/暗主题，
 * 所以这里只写需要覆盖的语义色，其余变量（opacity 等）沿用默认。
 */
export function buildThemeDefinition(input: HostThemeInput): ThemeDefinition {
  const hostColors =
    input.colors && typeof input.colors === 'object' && Object.keys(input.colors).length
      ? input.colors
      : null

  // 关键降级：没有 themeColors 就固定亮色，不跟随 isDark
  const dark = hostColors ? input.isDark === true : false
  const src = hostColors ?? FALLBACK_LIGHT_COLORS

  const text = src.primaryText || src['on-surface'] || (dark ? '#e8eaed' : '#1b1c1d')
  // 画布色不再取宿主的 background/surface，整页统一一档，和上游背景区分开
  const canvas = dark ? CANVAS_COLORS.dark : CANVAS_COLORS.light

  return {
    dark,
    colors: {
      ...src,
      primary: src.primary || FALLBACK_LIGHT_COLORS.primary,
      secondary: src.secondary || FALLBACK_LIGHT_COLORS.secondary,
      error: src.error || FALLBACK_LIGHT_COLORS.error,
      info: src.info || FALLBACK_LIGHT_COLORS.info,
      success: src.success || FALLBACK_LIGHT_COLORS.success,
      warning: src.warning || FALLBACK_LIGHT_COLORS.warning,
      // 侧栏、内容区共用页面画布色；卡片用亮一档的表面色，靠色差浮出来
      background: canvas,
      surface: SURFACE_COLORS[dark ? 'dark' : 'light'],
      'on-background': text,
      'on-surface': text,
      // 宿主亮色主题里 on-surface-variant 是 #fff，直接沿用会出现白字，统一按正文色处理
      'on-surface-variant': text,
      'surface-variant': dark ? '#2b2f36' : '#dfe3e8',
      'surface-bright': canvas,
      'surface-light': dark ? '#2b2f36' : '#dfe3e8',
    },
  }
}

export const vuetify = createVuetify({
  // Vuetify 内置文案（分页、排序、空态、每页条数）默认是英文，挂中文包后全站统一
  locale: {
    locale: 'zhHans',
    fallback: 'en',
    messages: { zhHans },
  },
  icons: {
    defaultSet: 'lucide',
    sets: { lucide: { component: IconifyIcon } },
    aliases: ICON_ALIASES,
  },
  theme: {
    defaultTheme: THEME_NAME,
    themes: {
      // 同步创建，首帧即为完整亮色主题；宿主主题稍后覆盖
      [THEME_NAME]: buildThemeDefinition({ isDark: false, colors: null }),
    },
  },
  defaults: {
    VCard: { rounded: 'card' },
    VChip: { rounded: 'pill' },
    VTextField: { variant: 'outlined', density: 'comfortable', hideDetails: 'auto', rounded: 'control' },
    VTextarea: { variant: 'outlined', density: 'comfortable', hideDetails: 'auto' },
    VSelect: { variant: 'outlined', density: 'comfortable', hideDetails: 'auto', rounded: 'control' },
    VAutocomplete: { variant: 'outlined', density: 'comfortable', hideDetails: 'auto' },
    VCombobox: { variant: 'outlined', density: 'comfortable', hideDetails: 'auto' },
    VFileInput: { variant: 'outlined', density: 'comfortable', hideDetails: 'auto' },
    VSwitch: { color: 'primary', hideDetails: 'auto', density: 'comfortable' },
    VCheckbox: { color: 'primary', hideDetails: 'auto', density: 'comfortable' },
    VSnackbar: { rounded: 'lg', elevation: 6 },
    VTooltip: { location: 'top' },
    VList: { density: 'comfortable' },
    VProgressCircular: { color: 'primary' },
  },
})

/**
 * Vuetify 默认亮/暗主题（已含 variables），作为替换 themes 时的非颜色字段来源。
 * 必须在 createVuetify 之后取，此时 parseThemeOptions 已把默认主题 merge 进来。
 */
const THEME_BASES = {
  light: vuetify.theme.themes.value.light,
  dark: vuetify.theme.themes.value.dark,
}

/**
 * 应用宿主主题。Vuetify 的 themes 不是深度响应式，
 * 必须整体替换引用，computedThemes / styles 才会重算并重写主题样式表。
 *
 * 注意：createTheme 会用 mergeDeep 把 Vuetify 默认主题并进自定义主题，
 * 从而补上 variables（border-opacity / emphasis-opacity 等）。
 * 这里整体替换 themes 会绕过那一步，若不把 variables 补回来，
 * genCssVariables 会在 Object.entries(theme.variables) 上抛错，
 * 结果是主题样式表完全不更新（页面停在上一次的主题）。
 */
export function applyHostTheme(input: HostThemeInput): void {
  const definition = buildThemeDefinition(input)
  const base = definition.dark ? THEME_BASES.dark : THEME_BASES.light
  const merged: ThemeDefinition = {
    ...base,
    ...definition,
    colors: { ...base.colors, ...definition.colors },
  }

  const themes = vuetify.theme.themes
  themes.value = { ...themes.value, [THEME_NAME]: merged }
  vuetify.theme.global.name.value = THEME_NAME

  // bridge 自己也会写这两个属性，独立开发模式下靠这里兜底
  const root = document.documentElement
  const mode = definition.dark ? 'dark' : 'light'
  root.dataset.theme = mode
  root.style.colorScheme = mode
}
