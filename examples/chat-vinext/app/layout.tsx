import type { Metadata } from 'next';
import './globals.css';

export const metadata: Metadata = {
  title: 'Ignis Chat · Spark-X2.6-4B (INT8 LM Head) on WebGPU',
  description: 'Pure in-browser WebGPU LLM inference powered by Ignis',
};

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="zh-CN">
      <head>
        <link
          rel="icon"
          href="data:image/svg+xml,<svg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 32 32'><rect width='32' height='32' rx='8' fill='%23f97316'/><text x='16' y='22' text-anchor='middle' font-size='18' font-weight='800' fill='white' font-family='sans-serif'>I</text></svg>"
        />
      </head>
      <body>{children}</body>
    </html>
  );
}
