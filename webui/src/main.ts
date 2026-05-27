import { createApp } from 'vue'
import { createRouter, createWebHashHistory } from 'vue-router'
import { createVuetify } from 'vuetify'
import 'vuetify/styles'
import '@mdi/font/css/materialdesignicons.css'

import App from './App.vue'
import { routes } from './router'

const router = createRouter({
  history: createWebHashHistory(),
  routes,
})

const vuetify = createVuetify({
  theme: {
    defaultTheme: 'dark',
    themes: {
      dark: {
        colors: {
          primary: '#7C4DFF',
          secondary: '#448AFF',
          accent: '#FF4081',
          surface: '#1E1E2E',
          background: '#11111B',
        },
      },
      light: {
        colors: {
          primary: '#6200EA',
          secondary: '#2962FF',
          accent: '#FF4081',
        },
      },
    },
  },
})

const app = createApp(App)
app.use(router)
app.use(vuetify)
app.mount('#app')
