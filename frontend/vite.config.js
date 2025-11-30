import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

export default defineConfig({
  plugins: [react()],
  server: {
    host: '0.0.0.0',
    port: 3000,
    allowedHosts: [
      'localhost',
      '.ts.net',  // Allow all Tailscale hostnames
      'linux.taild5d988.ts.net'
    ],
    proxy: {
      '/api': {
        target: 'http://localhost:8000',
        changeOrigin: true
      }
    }
  },
  optimizeDeps: {
    include: [
      'react',
      'react-dom',
      'react-router-dom',
      'axios',
      '@xterm/xterm',
      '@xterm/addon-fit',
      '@xterm/addon-web-links',
      'lucide-react',
      'recharts',
      'react-dropzone'
    ],
    force: true
  },
  build: {
    rollupOptions: {
      output: {
        manualChunks: {
          'vendor': ['react', 'react-dom', 'react-router-dom', 'axios'],
          'terminal': ['@xterm/xterm', '@xterm/addon-fit', '@xterm/addon-web-links'],
          'charts': ['recharts'],
          'icons': ['lucide-react']
        }
      }
    }
  }
})
