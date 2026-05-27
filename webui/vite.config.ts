import { defineConfig } from 'vite'
import vue from '@vitejs/plugin-vue'
import vuetify from 'vite-plugin-vuetify'
import { resolve } from 'path'

export default defineConfig({
  plugins: [
    vue(),
    vuetify({ autoImport: true }),
  ],
  base: './',
  build: {
    outDir: resolve(__dirname, '../pages/memory-dashboard'),
    emptyOutDir: true,
    cssCodeSplit: false,
    rollupOptions: {
      output: {
        // 禁用代码分割，全部打成单文件，避免 iframe CORS 问题
        inlineDynamicImports: true,
        entryFileNames: 'assets/[name]-[hash].js',
        assetFileNames: 'assets/[name]-[hash][extname]',
      },
    },
  },
  resolve: {
    alias: {
      '@': resolve(__dirname, 'src'),
    },
  },
})
