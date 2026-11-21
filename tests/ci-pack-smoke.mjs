import fs from 'node:fs';
import os from 'node:os';
import path from 'node:path';
import { execFileSync } from 'node:child_process';
import { pathToFileURL } from 'node:url';

const ROOT_DIR = path.resolve(import.meta.dirname, '..');

function runNpm(args, cwd) {
  const cmd = process.platform === 'win32' ? 'npm.cmd' : 'npm';
  return execFileSync(cmd, args, { cwd, encoding: 'utf-8', shell: process.platform === 'win32' });
}

function extractTarball(tgzPath, destDir) {
  fs.mkdirSync(destDir, { recursive: true });
  execFileSync('tar', ['-xzf', tgzPath, '-C', destDir], { stdio: 'inherit' });
}

async function main() {
  const tmpDir = fs.mkdtempSync(path.join(os.tmpdir(), 'ignis-pack-smoke-'));
  try {
    const coreDir = path.join(ROOT_DIR, 'packages', 'core');
    const packerDir = path.join(ROOT_DIR, 'packages', 'packer');

    const coreOut = runNpm(['pack', '--pack-destination', tmpDir], coreDir).trim().split(/\r?\n/).pop().trim();
    const packerOut = runNpm(['pack', '--pack-destination', tmpDir], packerDir).trim().split(/\r?\n/).pop().trim();

    const coreExt = path.join(tmpDir, 'core-unpacked');
    const packerExt = path.join(tmpDir, 'packer-unpacked');
    extractTarball(path.join(tmpDir, coreOut), coreExt);
    extractTarball(path.join(tmpDir, packerOut), packerExt);

    const corePkgRoot = path.join(coreExt, 'package');
    const mustCoreFiles = [
      'LICENSE',
      'README.md',
      'dist/index.mjs',
      'dist/index.d.ts',
      'dist/openai-client.mjs',
      'dist/runtime/ignis-worker.mjs',
      'dist/runtime/ignis-engine.mjs',
      'dist/runtime/device-profile.mjs',
      'dist/runtime/speculative.mjs',
      'dist/runtime/wasm/ignis_tokenizer.wasm',
      'dist/kernels/codegen.mjs',
      'dist/kernels/multimodal-codegen.mjs',
    ];
    for (const rel of mustCoreFiles) {
      if (!fs.existsSync(path.join(corePkgRoot, rel))) {
        throw new Error(`ignis-webgpu tarball missing ${rel}`);
      }
    }

    // Import packaged entry completely isolated outside repo root
    const mod = await import(pathToFileURL(path.join(corePkgRoot, 'dist/index.mjs')).href);
    const expectedExports = [
      'IgnisOpenAI',
      'buildIgnisChatPrompt',
      'IgnisEngine',
      'IgnisClient',
      'IgnisTokenizer',
      'IgnisFsm',
      'detectDeviceProfile',
      'NgramLookupSpeculator',
      'DraftTargetSpeculator',
      'getIgnisAssetUrls',
    ];
    for (const sym of expectedExports) {
      if (!(sym in mod)) {
        throw new Error(`ignis-webgpu tarball missing export ${sym}`);
      }
    }
    const urls = mod.getIgnisAssetUrls();
    if (!urls.workerUrl.endsWith('/dist/runtime/ignis-worker.mjs') || !urls.wasmUrl.endsWith('/dist/runtime/wasm/ignis_tokenizer.wasm')) {
      throw new Error(`Unexpected getIgnisAssetUrls() output: ${JSON.stringify(urls)}`);
    }

    const packerPkgRoot = path.join(packerExt, 'package');
    const mustPackerFiles = [
      'LICENSE',
      'README.md',
      'bin/ignis-pack.mjs',
      'py/ignis_pack.py',
      'py/blueprints.py',
      'py/quant.py',
      'py/pack_multimodal.py',
      'py/requirements.txt',
    ];
    for (const rel of mustPackerFiles) {
      if (!fs.existsSync(path.join(packerPkgRoot, rel))) {
        throw new Error(`ignis-webgpu-packer tarball missing ${rel}`);
      }
    }

    const helpOut = execFileSync(process.execPath, [path.join(packerPkgRoot, 'bin/ignis-pack.mjs'), '--help'], {
      cwd: tmpDir,
      encoding: 'utf-8',
    });
    if (!helpOut.includes('--weight-bits') || !helpOut.includes('--lm-head-dtype')) {
      throw new Error(`ignis-pack --help output invalid:\n${helpOut}`);
    }

    console.log(`[PASS] Self-contained npm tarball smoke checks passed (${coreOut} + ${packerOut})`);
  } finally {
    fs.rmSync(tmpDir, { recursive: true, force: true });
  }
}

main().catch((err) => {
  console.error(err);
  process.exit(1);
});
