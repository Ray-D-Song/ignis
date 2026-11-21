#!/usr/bin/env node
import fs from 'node:fs';
import path from 'node:path';
import { spawnSync } from 'node:child_process';

const pkgRoot = path.resolve(import.meta.dirname, '..');
const repoRoot = path.resolve(pkgRoot, '../..');
const packerScript = path.join(repoRoot, 'packer', 'ignis_pack.py');

const venvWin = path.join(repoRoot, 'packer', '.venv', 'Scripts', 'python.exe');
const venvUnix = path.join(repoRoot, 'packer', '.venv', 'bin', 'python');
const pythonExe = process.env.IGNIS_PYTHON
  || (fs.existsSync(venvWin) ? venvWin : fs.existsSync(venvUnix) ? venvUnix : 'python');

const args = process.argv.slice(2);
const res = spawnSync(pythonExe, [packerScript, ...args], {
  stdio: 'inherit',
  cwd: process.cwd(),
});
process.exit(res.status ?? 1);
