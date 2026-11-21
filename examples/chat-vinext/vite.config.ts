import { defineConfig } from 'vite';
import vinext from 'vinext';

const coopCoepHeaders = {
  'Cross-Origin-Opener-Policy': 'same-origin',
  'Cross-Origin-Embedder-Policy': 'require-corp',
};

export default defineConfig({
  plugins: [vinext()],
  server: {
    headers: coopCoepHeaders,
  },
  preview: {
    headers: coopCoepHeaders,
  },
});
