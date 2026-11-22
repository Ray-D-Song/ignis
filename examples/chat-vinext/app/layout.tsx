import type { Metadata } from 'next';
import '@fontsource-variable/inter';
import '@fontsource-variable/jetbrains-mono';
import '@xterm/xterm/css/xterm.css';
import './globals.css';

export const metadata: Metadata = {
  title: 'Ignis — Browser-Native LLM Inference Engine',
  description:
    'Browser-native LLM inference engine powered by WebGPU, WebAssembly, and OPFS with an interactive Vite + React WebContainer playground.',
  openGraph: {
    title: 'Ignis — Browser-Native LLM Inference Engine',
    description:
      'Browser-native WebGPU LLM inference engine with WASM FSM structured decoding, CodeMirror 6, and a live Vite + React WebContainer sandbox.',
    type: 'website',
  },
};

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en">
      <head>
        <link
          rel="icon"
          href="data:image/svg+xml,<svg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 32 32'><rect width='32' height='32' rx='8' fill='%2309090b'/><path d='M16 5L22 14L17.5 14L20 27L10 17L14.5 17L16 5Z' fill='%23f97316'/></svg>"
        />
      </head>
      <body>{children}</body>
    </html>
  );
}
