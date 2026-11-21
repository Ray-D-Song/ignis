import fs from 'node:fs';
import os from 'node:os';
import path from 'node:path';

/**
 * Resolve WebGPU-capable Chromium/Edge/Thorium browser executable paths across Windows, macOS, and Linux
 * without any machine-specific usernames or hardcoded local drive paths.
 */
export function resolveAllBrowserExecutables() {
  if (process.env.IGNIS_BROWSER) {
    return [process.env.IGNIS_BROWSER];
  }

  const platform = process.platform;
  const home = os.homedir();
  const candidates = [];

  if (platform === 'win32') {
    const localApp = process.env.LOCALAPPDATA || path.join(home, 'AppData', 'Local');
    const progFiles = process.env.PROGRAMFILES || 'C:\\Program Files';
    const progFilesX86 = process.env['PROGRAMFILES(X86)'] || 'C:\\Program Files (x86)';
    candidates.push(
      path.join(localApp, 'Thorium', 'Application', 'thorium.exe'),
      path.join(progFilesX86, 'Microsoft', 'Edge', 'Application', 'msedge.exe'),
      path.join(progFiles, 'Microsoft', 'Edge', 'Application', 'msedge.exe'),
      path.join(progFiles, 'Google', 'Chrome', 'Application', 'chrome.exe'),
      path.join(localApp, 'Google', 'Chrome', 'Application', 'chrome.exe'),
    );
  } else if (platform === 'darwin') {
    candidates.push(
      '/Applications/Thorium.app/Contents/MacOS/Thorium',
      '/Applications/Google Chrome Canary.app/Contents/MacOS/Google Chrome Canary',
      '/Applications/Google Chrome.app/Contents/MacOS/Google Chrome',
      '/Applications/Microsoft Edge.app/Contents/MacOS/Microsoft Edge',
    );
  } else {
    candidates.push(
      '/usr/bin/thorium-browser',
      '/usr/bin/google-chrome-unstable',
      '/usr/bin/google-chrome',
      '/usr/bin/chromium',
      '/usr/bin/chromium-browser',
      '/usr/bin/microsoft-edge',
    );
  }

  return candidates.filter((p) => fs.existsSync(p));
}

export function resolveBrowserExecutable() {
  const found = resolveAllBrowserExecutables();
  return found.length > 0 ? found[0] : undefined;
}

export function resolvePythonExecutable(rootDir) {
  if (process.env.IGNIS_PYTHON) return process.env.IGNIS_PYTHON;
  const venvWin = path.join(rootDir, 'packer', '.venv', 'Scripts', 'python.exe');
  if (fs.existsSync(venvWin)) return venvWin;
  const venvUnix = path.join(rootDir, 'packer', '.venv', 'bin', 'python');
  if (fs.existsSync(venvUnix)) return venvUnix;
  return process.platform === 'win32' ? 'python' : 'python3';
}
