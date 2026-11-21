import type { Metadata } from 'next';
import './globals.css';

export const metadata: Metadata = {
  title: 'Ignis · Browser-Native WebGPU LLM & Full-Omni Playground',
  description:
    'Zero-dependency WebGPU inference engine built from first principles: OPFS streaming, strict W3C WGSL kernels, Rust/WASM JSON Schema FSM, and Full-Omni vision/audio/24 kHz voice.',
  openGraph: {
    title: 'Ignis · Browser-Native WebGPU LLM & Full-Omni Playground',
    description:
      'Run Spark-X2.6-4B (85-134 tok/s), LFM3-Omni-2B (Vision+Audio+24 kHz Voice), and Gemma-4-E4B purely inside your browser via WebGPU + OPFS.',
    type: 'website',
  },
};

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en">
      <head>
        <link
          rel="icon"
          href="data:image/svg+xml,<svg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 32 32'><rect width='32' height='32' rx='8' fill='%23f97316'/><text x='50%25' y='54%25' dominant-baseline='middle' text-anchor='middle' fill='white' font-family='sans-serif' font-weight='bold' font-size='18'>I</text></svg>"
        />
      </head>
      <body>{children}</body>
    </html>
  );
}
