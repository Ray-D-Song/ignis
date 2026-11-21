#!/usr/bin/env node
import fs from 'node:fs';
import path from 'node:path';
import { spawnSync } from 'node:child_process';

const pkgRoot = path.resolve(import.meta.dirname, '..');
const repoRoot = path.resolve(pkgRoot, '../..');

const packagedScript = path.join(pkgRoot, 'py', 'ignis_pack.py');
const monorepoScript = path.join(repoRoot, 'packer', 'ignis_pack.py');
const packerScript = fs.existsSync(packagedScript) ? packagedScript : monorepoScript;
const reqPath = path.join(pkgRoot, 'py', 'requirements.txt');

const venvWin = path.join(repoRoot, 'packer', '.venv', 'Scripts', 'python.exe');
const venvUnix = path.join(repoRoot, 'packer', '.venv', 'bin', 'python');

function findPython() {
  if (process.env.IGNIS_PYTHON) {
    return { cmd: process.env.IGNIS_PYTHON, prefixArgs: [] };
  }
  const venvCandidates = [
    path.join(repoRoot, 'packer', '.venv'),
    path.join(process.cwd(), 'packer', '.venv'),
    path.join(process.cwd(), '.venv'),
  ];
  for (const venvDir of venvCandidates) {
    const exe =
      process.platform === 'win32'
        ? path.join(venvDir, 'Scripts', 'python.exe')
        : path.join(venvDir, 'bin', 'python');
    if (fs.existsSync(exe)) {
      return { cmd: exe, prefixArgs: [] };
    }
  }
  const sysCandidates = [
    ['python3', []],
    ['python', []],
    ...(process.platform === 'win32' ? [['py', ['-3']]] : []),
  ];
  for (const [cmd, prefixArgs] of sysCandidates) {
    const chk = spawnSync(cmd, [...prefixArgs, '--version'], { stdio: 'ignore' });
    if (chk.status === 0 && !chk.error) {
      return { cmd, prefixArgs };
    }
  }
  return null;
}

const py = findPython();
if (!py) {
  console.error('[ignis-pack] Could not find a Python 3 interpreter. Install Python 3 or set IGNIS_PYTHON.');
  process.exit(1);
}
const args = process.argv.slice(2);
const res = spawnSync(py.cmd, [...py.prefixArgs, packerScript, ...args], {
  stdio: 'inherit',
  cwd: process.cwd(),
});

if (res.status !== 0 && res.status !== null && !args.includes('-h') && !args.includes('--help')) {
  const hint = fs.existsSync(reqPath)
    ? `pip install -r "${reqPath}"`
    : 'pip install numpy safetensors torch transformers';
  console.error(`\n[ignis-pack] Python dependencies missing? Run: ${hint}`);
}

process.exit(res.status ?? 1);
