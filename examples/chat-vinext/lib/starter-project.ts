export interface StarterFileSpec {
  path: string;
  content: string;
}

const PACKAGE_JSON = `{
  "name": "ignis-vite-react-app",
  "private": true,
  "version": "0.0.0",
  "type": "module",
  "scripts": {
    "dev": "vite",
    "build": "vite build",
    "preview": "vite preview"
  },
  "dependencies": {
    "react": "^18.3.1",
    "react-dom": "^18.3.1"
  },
  "devDependencies": {
    "@vitejs/plugin-react": "^4.5.2",
    "vite": "^6.3.5"
  }
}
`;

const VITE_CONFIG_JS = `import { defineConfig } from 'vite';
import react from '@vitejs/plugin-react';

export default defineConfig({
  plugins: [react()],
  server: {
    host: true,
    port: 3000,
  },
});
`;

const INDEX_HTML = `<!doctype html>
<html lang="en">
  <head>
    <meta charset="UTF-8" />
    <meta name="viewport" content="width=device-width, initial-scale=1.0" />
    <title>Vite + React</title>
  </head>
  <body>
    <div id="root"></div>
    <script type="module" src="/src/main.jsx"></script>
  </body>
</html>
`;

const MAIN_JSX = `import { StrictMode } from 'react';
import { createRoot } from 'react-dom/client';
import './index.css';
import App from './App.jsx';

createRoot(document.getElementById('root')).render(
  <StrictMode>
    <App />
  </StrictMode>,
);
`;

const APP_JSX = `import { useState } from 'react';
import './App.css';

function ViteLogo() {
  return (
    <svg className="logo" viewBox="0 0 256 257" aria-hidden="true">
      <defs>
        <linearGradient id="viteGradA" x1="-.828%" x2="57.636%" y1="7.652%" y2="78.411%">
          <stop offset="0%" stopColor="#41D1FF" />
          <stop offset="100%" stopColor="#BD34FE" />
        </linearGradient>
        <linearGradient id="viteGradB" x1="43.376%" x2="50.316%" y1="2.242%" y2="89.03%">
          <stop offset="0%" stopColor="#FFEA83" />
          <stop offset="8.333%" stopColor="#FFDD35" />
          <stop offset="100%" stopColor="#FFA800" />
        </linearGradient>
      </defs>
      <path
        fill="url(#viteGradA)"
        d="M255.153 37.938L134.897 252.976c-2.483 4.44-8.862 4.466-11.382.048L.875 37.958c-2.746-4.814 1.371-10.646 6.827-9.67l120.385 21.517a6.537 6.537 0 0 0 2.322-.004l117.867-21.483c5.438-.991 9.574 4.796 6.877 9.62Z"
      />
      <path
        fill="url(#viteGradB)"
        d="M185.432.063L96.44 17.501a3.268 3.268 0 0 0-2.634 3.014l-5.474 92.456a3.268 3.268 0 0 0 3.997 3.378l24.777-5.718c2.318-.535 4.413 1.507 3.936 3.838l-7.361 36.047c-.495 2.426 1.782 4.5 4.151 3.78l15.304-4.649c2.372-.72 4.652 1.36 4.15 3.788l-11.698 56.621c-.732 3.542 3.979 5.473 5.943 2.437l1.313-2.028l72.516-144.72c1.215-2.423-.88-5.186-3.54-4.672l-25.505 4.922c-2.396.462-4.435-1.77-3.759-4.114l16.646-57.705c.677-2.35-1.37-4.583-3.769-4.113Z"
      />
    </svg>
  );
}

function ReactLogo() {
  return (
    <svg className="logo react" viewBox="-11.5 -10.23174 23 20.46348" aria-hidden="true">
      <circle cx="0" cy="0" r="2.05" fill="#61dafb" />
      <g stroke="#61dafb" strokeWidth="1" fill="none">
        <ellipse rx="11" ry="4.2" />
        <ellipse rx="11" ry="4.2" transform="rotate(60)" />
        <ellipse rx="11" ry="4.2" transform="rotate(120)" />
      </g>
    </svg>
  );
}

export default function App() {
  const [count, setCount] = useState(0);

  return (
    <div className="vite-root">
      <div>
        <a href="https://vite.dev" target="_blank" rel="noreferrer">
          <ViteLogo />
        </a>
        <a href="https://react.dev" target="_blank" rel="noreferrer">
          <ReactLogo />
        </a>
      </div>
      <h1>Vite + React</h1>
      <div className="card">
        <button type="button" onClick={() => setCount((c) => c + 1)}>
          count is {count}
        </button>
        <p>
          Edit <code>src/App.jsx</code> and save to test HMR
        </p>
      </div>
      <p className="read-the-docs">
        Click on the Vite and React logos to learn more
      </p>
    </div>
  );
}
`;

const APP_CSS = `.vite-root {
  max-width: 1280px;
  margin: 0 auto;
  padding: 2rem;
  text-align: center;
}

.logo {
  height: 6em;
  width: 6em;
  padding: 1.5em;
  will-change: filter;
  transition: filter 300ms;
}

.logo:hover {
  filter: drop-shadow(0 0 2em #646cffaa);
}

.logo.react:hover {
  filter: drop-shadow(0 0 2em #61dafbaa);
}

@keyframes logo-spin {
  from {
    transform: rotate(0deg);
  }
  to {
    transform: rotate(360deg);
  }
}

@media (prefers-reduced-motion: no-preference) {
  a:nth-of-type(2) .logo {
    animation: logo-spin infinite 20s linear;
  }
}

.card {
  padding: 2em;
}

.read-the-docs {
  color: #888;
}
`;

const INDEX_CSS = `:root {
  font-family: Inter, system-ui, Avenir, Helvetica, Arial, sans-serif;
  line-height: 1.5;
  font-weight: 400;

  color-scheme: light dark;
  color: rgba(255, 255, 255, 0.87);
  background-color: #242424;

  font-synthesis: none;
  text-rendering: optimizeLegibility;
  -webkit-font-smoothing: antialiased;
  -moz-osx-font-smoothing: grayscale;
}

a {
  font-weight: 500;
  color: #646cff;
  text-decoration: inherit;
}

a:hover {
  color: #535bf2;
}

body {
  margin: 0;
  display: flex;
  place-items: center;
  min-width: 320px;
  min-height: 100vh;
}

h1 {
  font-size: 3.2em;
  line-height: 1.1;
}

button {
  border-radius: 8px;
  border: 1px solid transparent;
  padding: 0.6em 1.2em;
  font-size: 1em;
  font-weight: 500;
  font-family: inherit;
  background-color: #1a1a1a;
  color: inherit;
  cursor: pointer;
  transition: border-color 0.25s;
}

button:hover {
  border-color: #646cff;
}

button:focus,
button:focus-visible {
  outline: 4px auto -webkit-focus-ring-color;
}

@media (prefers-color-scheme: light) {
  :root {
    color: #213547;
    background-color: #ffffff;
  }
  a:hover {
    color: #747bff;
  }
  button {
    background-color: #f9f9f9;
  }
}
`;

export const STARTER_PROJECT_FILES: StarterFileSpec[] = [
  { path: 'package.json', content: PACKAGE_JSON },
  { path: 'vite.config.js', content: VITE_CONFIG_JS },
  { path: 'index.html', content: INDEX_HTML },
  { path: 'src/main.jsx', content: MAIN_JSX },
  { path: 'src/App.jsx', content: APP_JSX },
  { path: 'src/App.css', content: APP_CSS },
  { path: 'src/index.css', content: INDEX_CSS },
];

export type WebContainerTree = Record<string, unknown>;

export function toWebContainerTree(files: Array<{ path: string; content: string }>): WebContainerTree {
  const root: Record<string, unknown> = {};
  for (const f of files) {
    const parts = f.path.replace(/^\/+/, '').split('/').filter(Boolean);
    let cursor = root;
    for (let i = 0; i < parts.length; i++) {
      const seg = parts[i];
      if (i === parts.length - 1) {
        cursor[seg] = { file: { contents: f.content } };
      } else {
        if (!cursor[seg] || typeof (cursor[seg] as { directory?: unknown }).directory !== 'object') {
          cursor[seg] = { directory: {} };
        }
        cursor = (cursor[seg] as { directory: Record<string, unknown> }).directory;
      }
    }
  }
  return root;
}
