import { defineConfig } from 'vite';
import react from '@vitejs/plugin-react';

// Dev server proxies API calls to the FastAPI backend (run
// `python -m src.api.app` separately on :8000) so `npm run dev` never
// has to deal with CORS. In production, FastAPI serves this app's built
// `dist/` directly from the same origin, so no proxy is needed there.
export default defineConfig({
  plugins: [react()],
  server: {
    port: 5173,
    proxy: {
      '/health': 'http://localhost:8000',
      '/status': 'http://localhost:8000',
      '/predict': 'http://localhost:8000',
    },
  },
  build: {
    outDir: 'dist',
  },
});
