import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

// https://vitejs.dev/config/
export default defineConfig({
  plugins: [react()],
  server: {
    host: '0.0.0.0',
    port: 5174,
    allowedHosts: ['ai-kvm2', 'localhost', '192.168.1.2'],
    proxy: {
      '/api': {
        target: 'http://192.168.1.2:7998',
        changeOrigin: true
      }
    }
  }
})
