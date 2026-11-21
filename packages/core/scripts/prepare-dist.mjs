import fs from 'node:fs';
import path from 'node:path';

const pkgRoot = path.resolve(import.meta.dirname, '..');
const repoRoot = path.resolve(pkgRoot, '../..');
const distDir = path.join(pkgRoot, 'dist');

function copyRecursive(src, dst) {
  fs.mkdirSync(dst, { recursive: true });
  for (const entry of fs.readdirSync(src, { withFileTypes: true })) {
    const sPath = path.join(src, entry.name);
    const dPath = path.join(dst, entry.name);
    if (entry.isDirectory()) {
      copyRecursive(sPath, dPath);
    } else {
      fs.copyFileSync(sPath, dPath);
    }
  }
}

function rewriteRelativeImports(filePath) {
  const src = fs.readFileSync(filePath, 'utf-8');
  const rewritten = src
    .replaceAll('../../runtime/', './runtime/')
    .replaceAll('../../kernels/', './kernels/')
    .replaceAll('../../kernels', './kernels');
  fs.writeFileSync(filePath, rewritten, 'utf-8');
}

function assertNoParentTraversal(dir) {
  const escapedRoot = `${distDir}${path.sep}`;
  const relTraversalRe = /(['"`])((?:\.\.\/)+[^'"`\s]+)\1/g;
  for (const entry of fs.readdirSync(dir, { withFileTypes: true })) {
    const p = path.join(dir, entry.name);
    if (entry.isDirectory()) {
      assertNoParentTraversal(p);
    } else if (entry.name.endsWith('.mjs') || entry.name.endsWith('.ts')) {
      const content = fs.readFileSync(p, 'utf-8');
      for (const match of content.matchAll(relTraversalRe)) {
        const relSpec = match[2];
        const resolved = path.resolve(path.dirname(p), relSpec);
        if (!resolved.startsWith(escapedRoot) && resolved !== distDir) {
          throw new Error(`Package-escaping relative path "${relSpec}" found in ${p}`);
        }
      }
    }
  }
}

fs.rmSync(distDir, { recursive: true, force: true });
fs.mkdirSync(distDir, { recursive: true });

copyRecursive(path.join(repoRoot, 'runtime'), path.join(distDir, 'runtime'));
copyRecursive(path.join(repoRoot, 'kernels'), path.join(distDir, 'kernels'));

for (const fname of ['index.mjs', 'index.d.ts', 'openai-client.mjs']) {
  const src = path.join(pkgRoot, fname);
  const dst = path.join(distDir, fname);
  fs.copyFileSync(src, dst);
  rewriteRelativeImports(dst);
}

const rootLicense = path.join(repoRoot, 'LICENSE');
if (fs.existsSync(rootLicense)) {
  fs.copyFileSync(rootLicense, path.join(pkgRoot, 'LICENSE'));
}

assertNoParentTraversal(distDir);
console.log('[ignis-webgpu] Built self-contained dist/ successfully');
