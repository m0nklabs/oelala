import { defineConfig } from 'vitest/config'
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
  test: {
    // jsdom, because the hooks under test touch URL and document. The suite runs
    // through `npm test` and through CI's existing `npm test -- --run` step.
    environment: 'jsdom',
    globals: true,
    setupFiles: ['./src/test/setup.js'],
    include: ['src/**/*.{test,spec}.{js,jsx}'],
  },
  optimizeDeps: {
    // Pre-bundle ALL React deps in a single pass to prevent mid-session
    // re-optimization which causes split module instances (dispatcher=null).
    include: [
      'react',
      'react-dom',
      'react-dom/client',
      'react/jsx-runtime',
      'react/jsx-dev-runtime',
      'lucide-react',
      'recharts',
      '@sentry/react',
      '@supabase/supabase-js',
    ],
  },
  server: {
    host: '0.0.0.0',
    port: 5174,
    allowedHosts: ['ai-kvm2', 'localhost', 'localhost', 'oelala.xyz'],
    proxy: {
      '/api': {
        target: 'http://localhost:7998',
        changeOrigin: true
      },
      '/comfyui': {
        target: 'http://localhost:7998',
        changeOrigin: true
      },
      '/media': {
        target: 'http://localhost:7998',
        changeOrigin: true
      }
    }
  }
})
