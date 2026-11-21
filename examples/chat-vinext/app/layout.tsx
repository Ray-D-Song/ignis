import type { Metadata } from 'next';
import '@fontsource-variable/inter';
import '@fontsource-variable/jetbrains-mono';
import './globals.css';

export const metadata: Metadata = {
  title: 'Ignis — Browser Agent Runtime',
  description:
    'Run an autonomous local agent with schema-locked tool calls, a sandboxed workspace, and live artifact previews directly inside your browser via WebGPU.',
  openGraph: {
    title: 'Ignis — Browser Agent Runtime',
    description:
      'Autonomous pure-browser WebGPU agent harness with sandboxed JS execution, structured tool calls, and multimodal perception.',
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
