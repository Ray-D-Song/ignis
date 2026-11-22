import type { Metadata } from 'next';
import '@fontsource-variable/inter';
import '@fontsource-variable/jetbrains-mono';
import '@xterm/xterm/css/xterm.css';
import './globals.css';

export const metadata: Metadata = {
  title: 'Ignis Code — Pure-Browser Full-Stack Coding Agent IDE',
  description:
    'Run an autonomous Coding Agent harness in your browser with WebGPU local inference, StackBlitz WebContainers, CodeMirror 6 diffs, and a live Vite 6 + React + Node API workspace.',
  openGraph: {
    title: 'Ignis Code — Pure-Browser Full-Stack Coding Agent IDE',
    description:
      'Pure-browser WebGPU coding agent harness with WebContainers Node.js sandbox, schema-locked tool calls, live diffs, and full-stack preview.',
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
