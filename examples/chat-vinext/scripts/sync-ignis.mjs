import fs from 'node:fs';
import path from 'node:path';

const appRoot = path.resolve(import.meta.dirname, '..');
const ignisRoot = path.resolve(appRoot, '../..');
const destRoot = path.join(appRoot, 'public', 'ignis');

function copyDirClean(src, dst) {
  fs.mkdirSync(dst, { recursive: true });
  for (const entry of fs.readdirSync(src, { withFileTypes: true })) {
    const sPath = path.join(src, entry.name);
    const dPath = path.join(dst, entry.name);
    if (entry.isDirectory()) {
      copyDirClean(sPath, dPath);
    } else {
      fs.copyFileSync(sPath, dPath);
    }
  }
}

fs.rmSync(destRoot, { recursive: true, force: true });
copyDirClean(path.join(ignisRoot, 'runtime'), path.join(destRoot, 'runtime'));
copyDirClean(path.join(ignisRoot, 'kernels'), path.join(destRoot, 'kernels'));
console.log('[sync-ignis] synced runtime/ and kernels/ -> public/ignis/');
