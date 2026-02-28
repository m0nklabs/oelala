import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'
import { sentryVitePlugin } from '@sentry/vite-plugin'

// https://vitejs.dev/config/
export default defineConfig({
  build: {
    sourcemap: true, // Required for Sentry source maps
  },
  plugins: [
    react(),
    // Sentry source map upload — only active when SENTRY_AUTH_TOKEN is set
    ...(process.env.SENTRY_AUTH_TOKEN
      ? [
          sentryVitePlugin({
            org: process.env.SENTRY_ORG || 'oelala',
            project: process.env.SENTRY_PROJECT || 'oelala-frontend',
            authToken: process.env.SENTRY_AUTH_TOKEN,
          }),
        ]
      : []),
  ],
  server: {
    host: '0.0.0.0',
    port: 5174,
    allowedHosts: ['ai-kvm2', 'localhost', '192.168.1.2', 'oelala.xyz'],
    proxy: {
      '/api': {
        target: 'http://192.168.1.2:7998',
        changeOrigin: true
      },
      '/comfyui': {
        target: 'http://192.168.1.2:7998',
        changeOrigin: true
      },
      '/media': {
        target: 'http://192.168.1.2:7998',
        changeOrigin: true
      }
    }
  }
})
