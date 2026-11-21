import fs from 'node:fs';
import path from 'node:path';
import { execFileSync } from 'node:child_process';

const appRoot = path.resolve(import.meta.dirname, '..');
const clientOutDir = path.join(appRoot, 'dist', 'client');
const outDir = path.join(appRoot, 'out');
const vinextCli = path.join(appRoot, 'node_modules', 'vinext', 'dist', 'cli.js');

// 1. Sync runtime/ and kernels/ -> public/ignis/
execFileSync(process.execPath, [path.join(appRoot, 'scripts', 'sync-ignis.mjs')], {
  cwd: appRoot,
  stdio: 'inherit',
});

// 2. Clean out/ and build static export via vinext
fs.rmSync(outDir, { recursive: true, force: true });
execFileSync(process.execPath, [vinextCli, 'build'], {
  cwd: appRoot,
  stdio: 'inherit',
  env: {
    ...process.env,
    IGNIS_STATIC_EXPORT: '1',
  },
});

if (!fs.existsSync(path.join(clientOutDir, 'index.html'))) {
  throw new Error(`[build-pages] vinext static export did not produce ${path.join(clientOutDir, 'index.html')}`);
}

// 3. Mirror vinext@1.4.6 static client output (dist/client/) into out/ for Cloudflare Pages
fs.cpSync(clientOutDir, outDir, { recursive: true });

// 4. Ensure _headers is present in out/
const srcHeaders = path.join(appRoot, 'public', '_headers');
const dstHeaders = path.join(outDir, '_headers');
if (fs.existsSync(srcHeaders) && !fs.existsSync(dstHeaders)) {
  fs.copyFileSync(srcHeaders, dstHeaders);
}

// 5. Assert critical static export outputs exist
const required = [
  'index.html',
  '_headers',
  'ignis/runtime/ignis-worker.mjs',
  'ignis/runtime/ignis-engine.mjs',
  'ignis/runtime/wasm/ignis_tokenizer.wasm',
  'ignis/kernels/codegen.mjs',
  'samples/demo-shapes.png',
  'samples/demo-tone.wav',
];

for (const rel of required) {
  const full = path.join(outDir, rel);
  if (!fs.existsSync(full)) {
    throw new Error(`[build-pages] Expected static export artifact missing: ${full}`);
  }
}

console.log(`[build-pages] Static Cloudflare Pages export verified -> ${outDir}`);
