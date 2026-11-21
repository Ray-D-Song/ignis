import fs from 'node:fs';
import path from 'node:path';

const pkgRoot = path.resolve(import.meta.dirname, '..');
const repoRoot = path.resolve(pkgRoot, '../..');
const srcPackerDir = path.join(repoRoot, 'packer');
const outPyDir = path.join(pkgRoot, 'py');

const FILES = [
  'ignis_pack.py',
  'blueprints.py',
  'quant.py',
  'pack_multimodal.py',
  'verify_pack.py',
];

fs.rmSync(outPyDir, { recursive: true, force: true });
fs.mkdirSync(outPyDir, { recursive: true });

for (const f of FILES) {
  const src = path.join(srcPackerDir, f);
  if (!fs.existsSync(src)) {
    throw new Error(`Missing packer source file: ${src}`);
  }
  fs.copyFileSync(src, path.join(outPyDir, f));
}

fs.writeFileSync(
  path.join(outPyDir, 'requirements.txt'),
  ['numpy>=1.26', 'safetensors>=0.4', 'torch>=2.3', 'transformers>=4.48'].join('\n') + '\n',
  'utf-8'
);

const rootLicense = path.join(repoRoot, 'LICENSE');
if (fs.existsSync(rootLicense)) {
  fs.copyFileSync(rootLicense, path.join(pkgRoot, 'LICENSE'));
}

console.log('[ignis-webgpu-packer] Built self-contained py/ bundle successfully');
