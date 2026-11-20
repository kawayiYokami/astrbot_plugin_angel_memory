import { createApp } from 'vue'
import 'vuetify/styles'
import './styles/global.scss'
import { createRouter, createWebHashHistory } from 'vue-router'
import App from './App.vue'
import { routes } from './router'
import { applyHostTheme, vuetify } from './theme'
import { registerLucideIcons } from './icons'
import { useBridge, type BridgeContext } from './composables/useBridge'

// 图标先于任何渲染注册，页面不依赖外网
registerLucideIcons()

// 插件视图跑在 iframe 里，静态资源按真实路径解析，history 刷新会 404，必须用 hash
const router = createRouter({
  history: createWebHashHistory(),
  routes,
})

const app = createApp(App)
app.use(router)
app.use(vuetify)

// 先用同步创建的亮色主题挂载：bridge 慢、超时或不存在都不会白屏
app.mount('#app')

const { init, onContext } = useBridge()

function applyContext(context: BridgeContext | null) {
  const colors =
    context?.themeColors && typeof context.themeColors === 'object' ? context.themeColors : null
  applyHostTheme({ isDark: context?.isDark === true, colors })
}

// 主题对齐放在挂载之后异步做，失败也只是保持亮色
void init().then(applyContext)

// 订阅宿主变化：切亮暗、改主题色都会推新 context。
// 老版本没有 onContext，这里拿到的是空解绑函数，不影响渲染。
onContext(applyContext)
