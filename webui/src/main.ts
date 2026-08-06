import { createApp } from 'vue'
import { createRouter, createWebHashHistory } from 'vue-router'
import { Icon } from '@iconify/vue'

import App from './App.vue'
import { routes } from './router'

const router = createRouter({
  history: createWebHashHistory(),
  routes,
})

const app = createApp(App)
app.use(router)
app.component('Icon', Icon)
app.mount('#app')
