import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'
export default defineConfig({
  plugins:[react()],
  server:{
    allowedHosts:['surgery-small-habits-recommend.trycloudflare.com'],
    proxy:{'/api':'http://127.0.0.1:8008'},
  },
})
