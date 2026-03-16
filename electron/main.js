/**
 * Cerebro Desktop — Native Architecture (v4.0)
 *
 * No Docker. No containers. No Defender exclusions.
 * Backend + Redis run as bundled native processes.
 * Install → License → Launch. Like any normal app.
 */

const { app, BrowserWindow, ipcMain, dialog, shell, session } = require('electron');
const path = require('path');
const fs = require('fs');
const net = require('net');
const { spawn, execFile } = require('child_process');
const { autoUpdater } = require('electron-updater');
const { NativeManager, CEREBRO_DIR, MEMORY_DIR, LOG_DIR } = require('./native-manager');
const { loadPortConfig, savePortConfig, getBackendUrl } = require('./port-config');
const { LicenseManager } = require('./license-manager');
const { createTray, updateTrayStatus } = require('./tray');

const portConfig = loadPortConfig();

process.on('uncaughtException', (err) => {
  console.error('[Main] Uncaught exception:', err);
});
process.on('unhandledRejection', (reason) => {
  console.error('[Main] Unhandled rejection:', reason);
});

// Linux GPU handling
if (process.platform === 'linux') {
  app.commandLine.appendSwitch('ignore-gpu-blocklist');
  app.commandLine.appendSwitch('enable-gpu-rasterization');
  app.commandLine.appendSwitch('disable-software-rasterizer');
}

// Windows DPI fix
if (process.platform === 'win32') {
  app.commandLine.appendSwitch('high-dpi-support', '1');
  app.commandLine.appendSwitch('enable-use-zoom-for-dsf', 'false');
}

const isDev = process.env.CEREBRO_DEV === '1';
const nativeManager = new NativeManager();
nativeManager.setPortConfig(portConfig);

const BACKEND_URL = `http://127.0.0.1:${nativeManager.backendPort}`;

let licenseManager;
try {
  licenseManager = new LicenseManager();
} catch (err) {
  console.error('[Main] LicenseManager init failed, retrying:', err.message);
  try {
    licenseManager = new LicenseManager();
  } catch (err2) {
    console.error('[Main] LicenseManager init failed permanently:', err2.message);
    licenseManager = null;
  }
}

// Prevent multiple instances
const gotTheLock = app.requestSingleInstanceLock();
if (!gotTheLock) {
  app.quit();
  process.exit(0);
}

let mainWindow = null;
let splashWindow = null;
let tray = null;
let isQuitting = false;
let electronUpdateReady = false;
let isAutoUpdating = false;
let licenseRefreshTimer = null;
let cdpChromeProcess = null;

// --- Chrome CDP for browser agent access ---
const CDP_PORT = 9222;

const CHROME_PATHS_WIN = [
  path.join(process.env['PROGRAMFILES'] || 'C:\\Program Files', 'Google', 'Chrome', 'Application', 'chrome.exe'),
  path.join(process.env['PROGRAMFILES(X86)'] || 'C:\\Program Files (x86)', 'Google', 'Chrome', 'Application', 'chrome.exe'),
  path.join(process.env.LOCALAPPDATA || '', 'Google', 'Chrome', 'Application', 'chrome.exe'),
];

const CHROME_PATHS_LINUX = [
  '/usr/bin/google-chrome',
  '/usr/bin/google-chrome-stable',
  '/usr/bin/chromium',
  '/usr/bin/chromium-browser',
];

function findChromePath() {
  const paths = process.platform === 'win32' ? CHROME_PATHS_WIN : CHROME_PATHS_LINUX;
  for (const p of paths) {
    if (fs.existsSync(p)) return p;
  }
  return null;
}

function isCdpAvailable() {
  return new Promise((resolve) => {
    const socket = new net.Socket();
    socket.setTimeout(2000);
    socket.once('connect', () => { socket.destroy(); resolve(true); });
    socket.once('error', () => { socket.destroy(); resolve(false); });
    socket.once('timeout', () => { socket.destroy(); resolve(false); });
    socket.connect(CDP_PORT, '127.0.0.1');
  });
}

async function ensureChromeWithCDP() {
  if (await isCdpAvailable()) {
    console.log('[CDP] Chrome already listening on port', CDP_PORT);
    return null;
  }

  const chromePath = findChromePath();
  if (!chromePath) {
    console.log('[CDP] Chrome not found, browser agents will not have browser access');
    return null;
  }

  const cdpProfileDir = path.join(app.getPath('userData'), 'cerebro-chrome-cdp');
  fs.mkdirSync(cdpProfileDir, { recursive: true });

  const args = [
    `--remote-debugging-port=${CDP_PORT}`,
    `--remote-allow-origins=*`,
    `--user-data-dir=${cdpProfileDir}`,
    '--no-first-run',
    '--no-default-browser-check',
  ];

  console.log(`[CDP] Launching Chrome: ${chromePath}`);
  const proc = spawn(chromePath, args, { stdio: 'ignore', detached: false });

  proc.on('error', (err) => console.error('[CDP] Failed:', err.message));
  proc.on('exit', (code) => {
    console.log(`[CDP] Chrome exited (code ${code})`);
    if (cdpChromeProcess === proc) cdpChromeProcess = null;
  });

  for (let i = 0; i < 20; i++) {
    if (await isCdpAvailable()) {
      console.log('[CDP] Ready on port', CDP_PORT);
      return proc;
    }
    await new Promise(r => setTimeout(r, 500));
  }

  console.error('[CDP] Chrome launched but not responding');
  try { proc.kill(); } catch (_) {}
  return null;
}

// --- Window management ---

function createSplashWindow() {
  splashWindow = new BrowserWindow({
    width: 400,
    height: 300,
    frame: false,
    transparent: true,
    resizable: false,
    alwaysOnTop: true,
    skipTaskbar: true,
    webPreferences: { nodeIntegration: false, contextIsolation: true },
  });
  splashWindow.loadFile(path.join(__dirname, 'splash.html'));
  splashWindow.center();
}

function createMainWindow() {
  mainWindow = new BrowserWindow({
    width: 1200,
    height: 800,
    minWidth: 800,
    minHeight: 600,
    show: false,
    icon: path.join(__dirname, '..', 'assets', 'icon.png'),
    title: 'Cerebro',
    webPreferences: {
      preload: path.join(__dirname, 'preload.js'),
      nodeIntegration: false,
      contextIsolation: true,
      webSecurity: true,
    },
  });

  mainWindow.on('close', (e) => {
    if (!isQuitting) {
      e.preventDefault();
      mainWindow.hide();
    }
  });

  mainWindow.on('closed', () => { mainWindow = null; });

  mainWindow.webContents.on('render-process-gone', (_event, details) => {
    console.error('[Main] Renderer crashed:', details.reason);
    setTimeout(() => {
      if (mainWindow && !mainWindow.isDestroyed()) mainWindow.loadURL(BACKEND_URL);
    }, 2000);
  });

  return mainWindow;
}

function updateSplashStatus(message) {
  if (splashWindow && !splashWindow.isDestroyed()) {
    splashWindow.webContents.executeJavaScript(
      `document.getElementById('status').textContent = ${JSON.stringify(message)}`
    ).catch(() => {});
  }
}

async function checkLicense() {
  if (isDev) return true;
  if (!licenseManager) return false;
  updateSplashStatus('Checking license...');
  const result = await licenseManager.revalidate();
  if (result.valid) {
    console.log(`[Main] License valid (plan: ${result.plan}${result.offline ? ', offline' : ''})`);
    return true;
  }
  console.log(`[Main] License invalid: ${result.reason}`);
  return false;
}

// --- MCP Config ---
const MCP_CONFIG_URL = 'https://raw.githubusercontent.com/Professor-Low/Cerebro/main/config/mcp-desktop.json';

async function fetchMcpConfig() {
  const https = require('https');
  return new Promise((resolve, reject) => {
    const req = https.get(MCP_CONFIG_URL, { timeout: 5000 }, (res) => {
      if (res.statusCode !== 200) { reject(new Error(`HTTP ${res.statusCode}`)); res.resume(); return; }
      let data = '';
      res.on('data', (chunk) => { data += chunk; });
      res.on('end', () => { try { resolve(JSON.parse(data)); } catch (e) { reject(e); } });
    });
    req.on('error', reject);
    req.on('timeout', () => { req.destroy(); reject(new Error('Timeout')); });
  });
}

function validateRemoteMcpConfig(raw) {
  if (!raw || typeof raw !== 'object' || Array.isArray(raw)) return { mcpServers: {} };

  const ALLOWED_SERVER_NAMES = new Set(['cerebro']);
  const ALLOWED_COMMANDS = new Set(['docker', 'cerebro']);
  const ALLOWED_IMAGE_PREFIX = 'ghcr.io/professor-low/';
  const DANGEROUS_DOCKER_FLAGS = ['--privileged', '--pid=host', '--network=host', '--cap-add'];

  const homedir = require('os').homedir();
  const SAFE_PREFIXES = [homedir];
  if (process.env.APPDATA) SAFE_PREFIXES.push(process.env.APPDATA);
  if (process.env.LOCALAPPDATA) SAFE_PREFIXES.push(process.env.LOCALAPPDATA);

  const validated = { mcpServers: {} };

  if (raw.mcpServers && typeof raw.mcpServers === 'object') {
    for (const [name, serverConfig] of Object.entries(raw.mcpServers)) {
      if (!ALLOWED_SERVER_NAMES.has(name)) continue;
      if (!serverConfig || typeof serverConfig !== 'object') continue;

      const cmd = serverConfig.command;
      if (typeof cmd !== 'string') continue;

      const isFullPathCerebro = cmd.toLowerCase().endsWith('cerebro.exe') || cmd.toLowerCase().endsWith('cerebro');
      const isAllowedShort = ALLOWED_COMMANDS.has(cmd);
      if (!isAllowedShort && !isFullPathCerebro) continue;

      if (isFullPathCerebro && !isAllowedShort) {
        const resolved = path.resolve(cmd);
        const inSafeDir = SAFE_PREFIXES.some(prefix =>
          resolved.toLowerCase().startsWith(prefix.toLowerCase() + path.sep) ||
          resolved.toLowerCase().startsWith(prefix.toLowerCase() + '/')
        );
        if (!inSafeDir) continue;
      }

      const args = serverConfig.args;
      if (args !== undefined) {
        if (!Array.isArray(args) || !args.every(a => typeof a === 'string')) continue;
        if ((isFullPathCerebro || cmd === 'cerebro') && (args.length !== 1 || args[0] !== 'serve')) continue;
        if (cmd === 'docker') {
          if (args.some(arg => DANGEROUS_DOCKER_FLAGS.some(flag => arg === flag || arg.startsWith(flag + '=')))) continue;
          const runIdx = args.indexOf('run');
          if (runIdx !== -1) {
            const imageArg = args.find((a, i) => i > runIdx && !a.startsWith('-') && a.includes('/') && !a.includes('='));
            if (imageArg && !imageArg.startsWith(ALLOWED_IMAGE_PREFIX)) continue;
          }
        }
      }

      const clean = { command: cmd };
      if (args) clean.args = [...args];
      if (serverConfig.env && typeof serverConfig.env === 'object' && !Array.isArray(serverConfig.env)) {
        clean.env = Object.fromEntries(Object.entries(serverConfig.env).filter(([k, v]) => typeof k === 'string' && typeof v === 'string'));
      } else {
        clean.env = {};
      }
      validated.mcpServers[name] = clean;
    }
  }
  return validated;
}

function getNativeMcpConfig() {
  const config = loadMemoryConfig();
  const storagePath = config.storagePath || MEMORY_DIR;

  // In native mode, prefer the native cerebro pip package if installed
  const nativePath = findNativeCerebro();
  if (nativePath) {
    return {
      mcpServers: {
        cerebro: {
          command: nativePath,
          args: ['serve'],
          env: {
            CEREBRO_DATA_DIR: storagePath,
            CEREBRO_STANDALONE: '1',
          },
        },
      },
    };
  }

  // Fallback: try 'cerebro' on PATH
  return {
    mcpServers: {
      cerebro: {
        command: 'cerebro',
        args: ['serve'],
        env: {
          CEREBRO_DATA_DIR: storagePath,
          CEREBRO_STANDALONE: '1',
        },
      },
    },
  };
}

async function writeMcpConfig(mcpServers) {
  const configPath = path.join(require('os').homedir(), '.claude.json');
  let config = {};
  try {
    const existing = fs.readFileSync(configPath, 'utf8');
    config = JSON.parse(existing);
  } catch (e) {
    if (e.code !== 'ENOENT') throw new Error('Cannot update MCP config: ~/.claude.json is not valid JSON');
  }
  if (!config.mcpServers) config.mcpServers = {};
  for (const [name, serverConfig] of Object.entries(mcpServers)) {
    config.mcpServers[name] = serverConfig;
    console.log(`[MCP] Set server ${name}`);
  }
  fs.writeFileSync(configPath, JSON.stringify(config, null, 2), 'utf8');
  console.log(`[MCP] Config written to ${configPath}`);
  return configPath;
}

async function refreshMcpConfigSilently() {
  try {
    const rawRemote = await fetchMcpConfig();
    const remote = validateRemoteMcpConfig(rawRemote);
    const dynamicConfig = getNativeMcpConfig();
    if (remote.mcpServers) remote.mcpServers.cerebro = dynamicConfig.mcpServers.cerebro;
    await writeMcpConfig(remote.mcpServers);
    console.log('[MCP] Config refreshed');
  } catch (err) {
    console.log(`[MCP] Refresh skipped: ${err.message}`);
  }
}

// --- License refresh timer ---

function startLicenseRefreshTimer() {
  if (licenseRefreshTimer) clearInterval(licenseRefreshTimer);
  licenseRefreshTimer = setInterval(async () => {
    console.log('[Main] Periodic license check...');
    const result = await licenseManager.revalidate();
    if (!result.valid) {
      console.log(`[Main] License no longer valid: ${result.reason}`);
      if (mainWindow && !mainWindow.isDestroyed()) {
        mainWindow.webContents.send('license-expired', { reason: result.reason, cancelAtPeriodEnd: result.cancelAtPeriodEnd });
      }
      setTimeout(() => {
        if (mainWindow && !mainWindow.isDestroyed()) {
          showSetupWizard();
          mainWindow.webContents.once('did-finish-load', () => {
            mainWindow.webContents.send('license-failure', { valid: false, reason: result.reason, cancelAtPeriodEnd: result.cancelAtPeriodEnd });
          });
        }
      }, 3000);
      clearInterval(licenseRefreshTimer);
      licenseRefreshTimer = null;
    }
  }, 4 * 60 * 60 * 1000);
}

function showSetupWizard() {
  if (splashWindow) { splashWindow.close(); splashWindow = null; }
  mainWindow.loadFile(path.join(__dirname, 'activation.html'));
  mainWindow.show();
  mainWindow.focus();
}

// --- Start the native stack ---

async function startNativeStack() {
  try {
    updateSplashStatus('Starting Cerebro...');
    await nativeManager.startStack((progress) => {
      updateSplashStatus(progress.message || 'Starting...');
    });

    // Verify storage
    let storageHealth = null;
    try {
      storageHealth = await nativeManager.verifyStorageMount();
      if (storageHealth.warnings.length > 0) {
        console.warn('[Main] Storage warnings:', storageHealth.warnings);
      }
    } catch (err) {
      console.warn('[Main] Storage health check failed:', err.message);
    }

    return { ok: true, storageHealth };
  } catch (err) {
    console.error('[Main] Native stack failed:', err.message);
    return { ok: false, error: err.message };
  }
}

// --- Load frontend ---

async function loadFrontend() {
  updateSplashStatus('Loading Cerebro...');

  let retries = 0;
  const maxRetries = 15;

  mainWindow.loadURL(BACKEND_URL);

  mainWindow.webContents.on('did-finish-load', () => {
    if (splashWindow) { splashWindow.close(); splashWindow = null; }
    mainWindow.show();
    mainWindow.focus();

    // Check for updates in background
    checkForUpdatesQuietly();
    setTimeout(checkForUpdatesQuietly, 60 * 1000);
    setTimeout(checkForUpdatesQuietly, 5 * 60 * 1000);
    setInterval(checkForUpdatesQuietly, 30 * 60 * 1000);
  });

  mainWindow.webContents.on('did-fail-load', (_event, errorCode, errorDescription) => {
    retries++;
    console.error(`[Main] Frontend load failed (${retries}/${maxRetries}): ${errorCode} ${errorDescription}`);
    if (retries < maxRetries) {
      setTimeout(() => { if (mainWindow) mainWindow.loadURL(BACKEND_URL); }, 2000);
    } else {
      console.error('[Main] Frontend failed after max retries');
      if (mainWindow) showSetupWizard();
    }
  });
}

// --- Update checking ---

async function checkForUpdatesQuietly() {
  // Check Electron app updates
  try {
    await autoUpdater.checkForUpdates();
  } catch (err) {
    console.log('[Main] Update check failed:', err.message);
  }

  // Fallback: GitHub API version check
  try {
    const https = require('https');
    const latestVersion = await new Promise((resolve, reject) => {
      const req = https.get('https://api.github.com/repos/Professor-Low/Cerebro-life-desktop/releases/latest', {
        headers: { 'User-Agent': 'Cerebro-Desktop/' + app.getVersion() },
        timeout: 10000,
      }, (res) => {
        let body = '';
        res.on('data', (chunk) => { body += chunk; });
        res.on('end', () => {
          try { const data = JSON.parse(body); resolve(data.tag_name ? data.tag_name.replace(/^v/, '') : null); }
          catch { resolve(null); }
        });
      });
      req.on('error', reject);
      req.on('timeout', () => { req.destroy(); reject(new Error('timeout')); });
    });

    if (latestVersion && latestVersion !== app.getVersion()) {
      const current = app.getVersion().split('.').map(Number);
      const latest = latestVersion.split('.').map(Number);
      let isNewer = false;
      for (let i = 0; i < 3; i++) {
        if ((latest[i] || 0) > (current[i] || 0)) { isNewer = true; break; }
        if ((latest[i] || 0) < (current[i] || 0)) break;
      }
      if (isNewer && mainWindow && !mainWindow.isDestroyed()) {
        console.log(`[Main] Update available (${app.getVersion()} → ${latestVersion})`);
        mainWindow.webContents.executeJavaScript(`
          if (typeof window.__cerebroShowUpdateBanner === 'function') {
            window.__cerebroShowUpdateBanner('electron');
          }
        `).catch(() => {});
      }
    }
  } catch (err) {
    console.log('[Main] GitHub version check failed:', err.message);
  }
}

// --- Linux desktop integration ---

function installDesktopIntegration() {
  if (process.platform !== 'linux') return;
  const homeDir = require('os').homedir();
  const markerPath = path.join(app.getPath('userData'), '.desktop-integrated');
  if (fs.existsSync(markerPath)) return;

  try {
    const iconSrc = path.join(__dirname, '..', 'assets', 'icon-256.png');
    const iconFile = fs.existsSync(iconSrc) ? iconSrc : path.join(__dirname, '..', 'assets', 'icon.png');
    if (!fs.existsSync(iconFile)) return;

    const iconDestDir = path.join(homeDir, '.local', 'share', 'icons', 'hicolor', '256x256', 'apps');
    fs.mkdirSync(iconDestDir, { recursive: true });
    fs.copyFileSync(iconFile, path.join(iconDestDir, 'cerebro.png'));

    const desktopDir = path.join(homeDir, '.local', 'share', 'applications');
    fs.mkdirSync(desktopDir, { recursive: true });

    const appImagePath = process.env.APPIMAGE || process.execPath;
    const desktopEntry = [
      '[Desktop Entry]', 'Name=Cerebro', 'Comment=Your AI, Everywhere',
      `Exec="${appImagePath}" %U`, 'Icon=cerebro', 'Type=Application',
      'Categories=Utility;ArtificialIntelligence;Science;',
      'StartupWMClass=cerebro-desktop', 'Terminal=false',
    ].join('\n') + '\n';

    fs.writeFileSync(path.join(desktopDir, 'cerebro.desktop'), desktopEntry);
    try { require('child_process').execFileSync('update-desktop-database', [desktopDir], { timeout: 5000 }); } catch (_) {}
    try { require('child_process').execFileSync('gtk-update-icon-cache', ['-f', '-t', path.join(homeDir, '.local', 'share', 'icons', 'hicolor')], { timeout: 5000 }); } catch (_) {}

    fs.writeFileSync(markerPath, new Date().toISOString());
    console.log('[Main] Desktop integration installed');
  } catch (err) {
    console.error('[Main] Desktop integration failed:', err.message);
  }
}

// --- App lifecycle ---

app.on('second-instance', () => {
  if (mainWindow) {
    if (mainWindow.isMinimized()) mainWindow.restore();
    mainWindow.show();
    mainWindow.focus();
  }
});

app.whenReady().then(async () => {
  console.log('[Main] Cerebro Desktop starting (Native architecture v4.0)');

  installDesktopIntegration();

  // Configure auto-updater
  autoUpdater.autoDownload = true;
  autoUpdater.autoInstallOnAppQuit = true;
  autoUpdater.logger = null;

  autoUpdater.on('update-downloaded', (info) => {
    console.log(`[Main] Update downloaded: v${info.version}`);
    electronUpdateReady = true;
    if (mainWindow && !mainWindow.isDestroyed()) {
      mainWindow.webContents.executeJavaScript(`
        if (typeof window.__cerebroShowUpdateBanner === 'function') {
          window.__cerebroShowUpdateBanner('electron');
        }
      `).catch(() => {});
    }
  });

  autoUpdater.on('error', (err) => {
    console.log('[Main] Auto-updater error:', err.message);
  });

  // Grant audio permissions
  session.defaultSession.setPermissionRequestHandler((webContents, permission, callback) => {
    callback(['media', 'audioCapture', 'mediaKeySystem'].includes(permission));
  });
  session.defaultSession.setPermissionCheckHandler((_webContents, permission) => {
    return ['media', 'audioCapture', 'mediaKeySystem'].includes(permission);
  });

  createSplashWindow();
  setTimeout(() => {
    if (splashWindow && !splashWindow.isDestroyed()) {
      console.error('[Main] Splash timeout');
      splashWindow.close();
      splashWindow = null;
      if (mainWindow) showSetupWizard();
    }
  }, 90000);  // 90s timeout for native startup (model loading)
  createMainWindow();

  // System tray
  const trayIconPath = path.join(__dirname, '..', 'assets', 'tray-icon.png');
  tray = createTray(mainWindow, trayIconPath);

  // License gate
  const licensed = await checkLicense();
  if (!licensed) {
    const licenseStatus = licenseManager ? licenseManager.getStatus() : { valid: false, reason: 'init_failed' };
    showSetupWizard();
    mainWindow.webContents.once('did-finish-load', () => {
      mainWindow.webContents.send('license-failure', licenseStatus);
    });
    return;
  }

  // Start native stack
  const result = await startNativeStack();
  if (!result.ok) {
    console.error('[Main] Stack failed:', result.error);
    showSetupWizard();
    return;
  }

  await loadFrontend();
  updateTrayStatus(tray, 'running');
  startLicenseRefreshTimer();
  refreshMcpConfigSilently().catch(err => console.warn('[MCP] Refresh failed:', err.message));

  if (result.storageHealth && result.storageHealth.warnings.length > 0) {
    setTimeout(() => {
      if (mainWindow) mainWindow.webContents.send('storage-health-warning', result.storageHealth);
    }, 3000);
  }
});

app.on('window-all-closed', () => {
  // Stay in tray
});

app.on('activate', () => {
  if (mainWindow) { mainWindow.show(); mainWindow.focus(); }
});

app.on('before-quit', async (e) => {
  if (!isQuitting) {
    isQuitting = true;
    e.preventDefault();

    // Write shutdown marker
    try {
      fs.writeFileSync(path.join(CEREBRO_DIR, '.last-shutdown'), new Date().toISOString());
    } catch (_) {}

    if (!isAutoUpdating) {
      updateTrayStatus(tray, 'stopping');
      await nativeManager.stopStack();
    }

    if (cdpChromeProcess) {
      try { cdpChromeProcess.kill(); } catch (_) {}
      cdpChromeProcess = null;
    }

    app.quit();
  }
});

// ============================================================
// IPC Handlers
// ============================================================

// App info
ipcMain.handle('get-app-version', () => app.getVersion());
ipcMain.handle('get-edition', () => 'native');

// Port configuration
ipcMain.handle('get-port-config', () => loadPortConfig());
ipcMain.handle('set-port-config', (_e, cfg) => {
  savePortConfig(cfg);
  return { success: true, note: 'Restart required' };
});

// License
ipcMain.handle('get-license-status', () => licenseManager ? licenseManager.getStatus() : { valid: false, reason: 'init_failed' });
ipcMain.handle('activate-license', async (_event, key) => {
  const result = await licenseManager.activate(key);
  if (result.success) console.log(`[Main] License activated (plan: ${result.plan})`);
  return result;
});
ipcMain.handle('refresh-license', async () => {
  const result = await licenseManager.revalidate();
  return result;
});

// Backend status (replaces Docker status)
ipcMain.handle('check-docker', async () => {
  // Compatibility: return that "Docker" (native backend) is available
  return {
    installed: true,
    running: nativeManager.isRunning(),
    compose: true,
    edition: 'native',
  };
});

ipcMain.handle('check-claude-code', async () => nativeManager.isClaudeInstalled());
ipcMain.handle('check-wsl', async () => ({ installed: true }));

// These are no-ops in native mode (compatibility with activation.html)
ipcMain.handle('install-docker', async () => ({ success: true }));
ipcMain.handle('start-docker-daemon', async () => ({ success: true }));
ipcMain.handle('setup-docker', async () => {
  try {
    await nativeManager.startStack((p) => {
      if (mainWindow) mainWindow.webContents.send('docker-start-progress', p);
    });
    return { success: true };
  } catch (err) {
    return { success: false, error: err.message };
  }
});

ipcMain.handle('pull-images', async () => {
  // No images to pull in native mode — backend is bundled
  if (mainWindow) {
    mainWindow.webContents.send('pull-progress', { stage: 'done', message: 'Backend ready', percent: 100 });
  }
  return { success: true };
});

ipcMain.handle('start-stack', async () => {
  try {
    await nativeManager.startStack();
    return { success: true };
  } catch (err) {
    return { success: false, error: err.message };
  }
});

ipcMain.handle('stop-stack', async () => {
  try {
    await nativeManager.stopStack();
    return { success: true };
  } catch (err) {
    return { success: false, error: err.message };
  }
});

ipcMain.handle('get-docker-status', async () => ({
  running: nativeManager.isRunning(),
  edition: 'native',
  containers: nativeManager.isRunning() ? [
    { name: 'cerebro-backend', status: 'running', health: 'healthy' },
    { name: 'cerebro-redis', status: 'running', health: 'healthy' },
  ] : [],
}));

ipcMain.handle('get-docker-logs', async () => nativeManager.getLogs(200));

// Updates — native mode only has Electron updates
ipcMain.handle('check-for-updates', async () => {
  try {
    await autoUpdater.checkForUpdates();
    return { updateAvailable: electronUpdateReady };
  } catch (err) {
    return { updateAvailable: false, error: err.message };
  }
});

ipcMain.handle('apply-update', async () => {
  try {
    if (electronUpdateReady) {
      isAutoUpdating = true;
      autoUpdater.quitAndInstall(true, true);
      return { success: true };
    }

    // Check and download
    await autoUpdater.checkForUpdates();
    return { success: true, message: 'Update check started' };
  } catch (err) {
    return { success: false, error: err.message };
  }
});

// MCP
ipcMain.handle('configure-mcp', async () => {
  try {
    let mcpConfig;
    try {
      const rawConfig = await fetchMcpConfig();
      mcpConfig = validateRemoteMcpConfig(rawConfig);
    } catch {
      mcpConfig = getNativeMcpConfig();
    }
    const nativeConfig = getNativeMcpConfig();
    if (mcpConfig.mcpServers) mcpConfig.mcpServers.cerebro = nativeConfig.mcpServers.cerebro;
    const configPath = await writeMcpConfig(mcpConfig.mcpServers);
    return { success: true, configPath };
  } catch (err) {
    return { success: false, error: err.message };
  }
});

// Kokoro TTS — not bundled in native mode (optional separate install)
ipcMain.handle('install-kokoro-tts', async () => {
  return { success: false, error: 'Voice engine is available as an optional download. Visit cerebro.dev/voice for setup instructions.' };
});

// Wizard completion
ipcMain.handle('wizard-complete', async () => {
  try {
    if (!nativeManager.isRunning()) {
      const result = await startNativeStack();
      if (!result.ok) return { success: false, error: result.error };
    }
    await loadFrontend();
    updateTrayStatus(tray, 'running');
    nativeManager.markSetupComplete();
    return { success: true };
  } catch (err) {
    return { success: false, error: err.message };
  }
});

ipcMain.handle('get-setup-status', async () => {
  const claudeStatus = await nativeManager.isClaudeInstalled().catch(() => ({ installed: false }));
  return {
    licensed: licenseManager ? licenseManager.getStatus().valid : false,
    dockerInstalled: true,   // Always true — no Docker needed
    dockerRunning: nativeManager.isRunning(),
    claudeInstalled: claudeStatus.installed,
    setupComplete: nativeManager.isSetupComplete(),
    stackRunning: nativeManager.isRunning(),
    edition: 'native',
  };
});

// Settings
ipcMain.handle('toggle-autostart', (_event, enabled) => {
  app.setLoginItemSettings({ openAtLogin: enabled, path: process.env.APPIMAGE || app.getPath('exe') });
  return app.getLoginItemSettings().openAtLogin;
});
ipcMain.handle('get-autostart', () => app.getLoginItemSettings().openAtLogin);
ipcMain.handle('enable-autostart', () => {
  app.setLoginItemSettings({ openAtLogin: true, path: process.env.APPIMAGE || app.getPath('exe') });
  return true;
});

// File access
ipcMain.handle('get-file-access-config', () => nativeManager.loadFileAccessConfig());
ipcMain.handle('save-file-access-config', async (_event, config) => {
  try {
    nativeManager.saveFileAccessConfig(config);
    return { success: true };
  } catch (err) {
    return { success: false, error: err.message };
  }
});
ipcMain.handle('get-file-access-presets', () => nativeManager.getPresetMounts());
ipcMain.handle('browse-folder', async () => {
  try {
    const win = BrowserWindow.getFocusedWindow() || mainWindow;
    const result = await dialog.showOpenDialog(win, { properties: ['openDirectory'], title: 'Select folder' });
    if (result.canceled) return { canceled: true };
    return { canceled: false, path: result.filePaths[0] };
  } catch (err) {
    return { canceled: true, error: err.message };
  }
});

// Restart stack (native version)
ipcMain.handle('restart-docker-stack', async () => {
  try {
    await nativeManager.stopStack();
    await nativeManager.startStack();
    return { success: true };
  } catch (err) {
    return { success: false, error: err.message };
  }
});

// Restart & setup state
ipcMain.handle('needs-restart', async () => false);
ipcMain.handle('save-setup-state', (_event, state) => { nativeManager.saveSetupState(state); return true; });
ipcMain.handle('load-setup-state', () => nativeManager.loadSetupState());
ipcMain.handle('clear-setup-state', () => { nativeManager.clearSetupState(); return true; });

// Chrome CDP
ipcMain.handle('launch-chrome-cdp', async () => {
  try {
    if (await isCdpAvailable()) return { success: true, alreadyRunning: true };
    cdpChromeProcess = await ensureChromeWithCDP();
    if (cdpChromeProcess || await isCdpAvailable()) return { success: true };
    return { success: false, error: 'Chrome failed to start' };
  } catch (err) {
    return { success: false, error: err.message };
  }
});

ipcMain.handle('stop-chrome-cdp', async () => {
  try {
    if (cdpChromeProcess) {
      try { cdpChromeProcess.kill(); } catch (_) {}
      cdpChromeProcess = null;
      return { success: true };
    }
    return { success: true, message: 'Chrome was not running' };
  } catch (err) {
    return { success: false, error: err.message };
  }
});

// Claude credentials
ipcMain.handle('check-claude-credentials', () => nativeManager.checkClaudeCredentials());
ipcMain.handle('refresh-claude-credentials', () => nativeManager.refreshClaudeCredentials());
ipcMain.handle('silent-refresh-oauth', async () => nativeManager.silentRefreshOAuthToken());

// Claude login flow
let _authWindow = null;

ipcMain.handle('launch-claude-login', async () => {
  try {
    if (_authWindow && !_authWindow.isDestroyed()) _authWindow.close();

    const env = { ...process.env };
    if (process.platform !== 'win32') env.BROWSER = 'echo';

    return new Promise((resolve) => {
      const proc = spawn('claude', ['auth', 'login'], { env, shell: true, stdio: ['ignore', 'pipe', 'pipe'] });
      let urlFound = false;
      const urlRegex = /https:\/\/[^\s"'<>]+/g;

      function handleOutput(chunk) {
        const text = chunk.toString();
        if (!urlFound) {
          const matches = text.match(urlRegex);
          if (matches) {
            const authUrl = matches.find(u => u.includes('claude.ai') || u.includes('anthropic.com') || u.includes('oauth')) || matches[0];
            if (authUrl) {
              urlFound = true;
              _authWindow = new BrowserWindow({
                width: 500, height: 700, title: 'Cerebro — Claude Authentication',
                parent: mainWindow, modal: true, autoHideMenuBar: true,
                webPreferences: { nodeIntegration: false, contextIsolation: true },
              });
              _authWindow.loadURL(authUrl);
              _authWindow.on('closed', () => { _authWindow = null; });
              resolve({ success: true, message: 'Login window opened.' });
            }
          }
        }
      }

      proc.stdout.on('data', handleOutput);
      proc.stderr.on('data', handleOutput);

      setTimeout(() => {
        if (!urlFound) resolve({ success: true, message: 'Login opened in browser.' });
      }, 8000);

      proc.on('error', (err) => {
        if (!urlFound) resolve({ success: false, error: err.message });
      });

      // Poll for credentials
      let attempts = 0;
      const watcher = setInterval(() => {
        attempts++;
        const status = nativeManager.checkClaudeCredentials();
        if (status.valid && status.expiresIn > 30) {
          clearInterval(watcher);
          if (mainWindow) mainWindow.webContents.send('credentials-refreshed', { expiresIn: status.expiresIn, message: 'Login successful' });
          if (_authWindow && !_authWindow.isDestroyed()) _authWindow.close();
          try { proc.kill(); } catch (_) {}
        }
        if (attempts >= 60) {
          clearInterval(watcher);
          try { proc.kill(); } catch (_) {}
        }
      }, 5000);
    });
  } catch (err) {
    return { success: false, error: err.message };
  }
});

// --- Memory Storage ---
const MEMORY_CONFIG_FILE = path.join(CEREBRO_DIR, 'memory-config.json');

function loadMemoryConfig() {
  try {
    if (fs.existsSync(MEMORY_CONFIG_FILE)) {
      const cfg = JSON.parse(fs.readFileSync(MEMORY_CONFIG_FILE, 'utf-8'));
      cfg.source = 'file';
      return cfg;
    }
  } catch {}

  const envPath = process.env.CEREBRO_DATA_DIR;
  if (envPath) {
    try {
      if (fs.existsSync(envPath)) {
        const config = { storagePath: envPath, source: 'env', envVar: 'CEREBRO_DATA_DIR' };
        fs.mkdirSync(path.dirname(MEMORY_CONFIG_FILE), { recursive: true });
        fs.writeFileSync(MEMORY_CONFIG_FILE, JSON.stringify({ storagePath: envPath }, null, 2));
        return config;
      }
    } catch {}
  }

  return { storagePath: null, source: 'default' };
}

function saveMemoryConfig(config) {
  const { source, envVar, ...persistent } = config;
  fs.writeFileSync(MEMORY_CONFIG_FILE, JSON.stringify(persistent, null, 2));
}

function validatePath(inputPath, opts = {}) {
  const { allowNull = false, mustExist = false } = opts;
  if (inputPath === null || inputPath === undefined) {
    if (allowNull) return null;
    throw new Error('Path is required');
  }
  if (typeof inputPath !== 'string' || inputPath.trim() === '') throw new Error('Path must be non-empty');
  if (inputPath.includes('..')) throw new Error('Path traversal not allowed');

  const resolved = path.resolve(inputPath);
  const BLOCKED_WIN = ['C:\\Windows', 'C:\\Program Files', 'C:\\Program Files (x86)'];
  const BLOCKED_UNIX = ['/etc', '/var', '/usr', '/root', '/boot', '/sbin', '/bin', '/lib', '/proc', '/sys', '/dev'];

  if (process.platform === 'win32') {
    for (const prefix of BLOCKED_WIN) {
      if (resolved.toLowerCase().startsWith(prefix.toLowerCase())) throw new Error(`Blocked: ${prefix}`);
    }
  } else {
    for (const prefix of BLOCKED_UNIX) {
      if (resolved === prefix || resolved.startsWith(prefix + '/')) throw new Error(`Blocked: ${prefix}`);
    }
  }
  if (resolved === '/') throw new Error('Root not allowed');

  if (mustExist && !fs.existsSync(resolved)) throw new Error(`Not found: ${resolved}`);
  return resolved;
}

function findNativeCerebro() {
  if (process.platform !== 'win32') return null;
  const appData = process.env.APPDATA || '';
  const localAppData = process.env.LOCALAPPDATA || '';
  for (const pyVer of ['313', '312', '311']) {
    const candidate = path.join(appData, 'Python', `Python${pyVer}`, 'Scripts', 'cerebro.exe');
    if (fs.existsSync(candidate)) return candidate;
  }
  for (const pyVer of ['313', '312', '311']) {
    const candidate = path.join(localAppData, 'Programs', 'Python', `Python${pyVer}`, 'Scripts', 'cerebro.exe');
    if (fs.existsSync(candidate)) return candidate;
  }
  try {
    const output = require('child_process').execFileSync('where', ['cerebro.exe'], { encoding: 'utf-8', timeout: 5000 });
    const firstLine = output.trim().split(/\r?\n/)[0];
    if (firstLine && fs.existsSync(firstLine)) return firstLine;
  } catch {}
  return null;
}

ipcMain.handle('get-memory-config', () => {
  const cfg = loadMemoryConfig();
  return { storagePath: cfg.storagePath, source: cfg.source || 'default', envVar: cfg.envVar || null, localMirrorPath: cfg.localMirrorPath || null };
});

ipcMain.handle('browse-storage-folder', async () => {
  try {
    const win = BrowserWindow.getFocusedWindow() || mainWindow;
    const result = await dialog.showOpenDialog(win, { properties: ['openDirectory'], title: 'Select storage location' });
    if (result.canceled) return { canceled: true };
    return { canceled: false, path: result.filePaths[0] };
  } catch (err) {
    return { canceled: true, error: err.message };
  }
});

ipcMain.handle('set-storage-path', async (_event, newPath) => {
  try {
    const validatedPath = validatePath(newPath, { allowNull: true });
    const config = loadMemoryConfig();
    config.storagePath = validatedPath;
    saveMemoryConfig(config);
    const mcpConfig = getNativeMcpConfig();
    await writeMcpConfig(mcpConfig.mcpServers);
    return { success: true };
  } catch (err) {
    return { success: false, error: err.message };
  }
});

ipcMain.handle('get-storage-health', async () => {
  try {
    return await nativeManager.verifyStorageMount();
  } catch (err) {
    return { healthy: true, warnings: [`Check failed: ${err.message}`] };
  }
});

ipcMain.handle('get-storage-stats', async (_event, folderPath) => {
  try {
    const targetPath = folderPath ? validatePath(folderPath, { mustExist: true }) : CEREBRO_DIR;
    let totalSize = 0, fileCount = 0;
    function walkDir(dir, depth = 0) {
      if (depth > 10) return;
      try {
        for (const entry of fs.readdirSync(dir, { withFileTypes: true })) {
          if (entry.isSymbolicLink()) continue;
          const full = path.join(dir, entry.name);
          try {
            if (entry.isDirectory()) walkDir(full, depth + 1);
            else if (entry.isFile()) { fileCount++; totalSize += fs.statSync(full).size; }
          } catch {}
        }
      } catch {}
    }
    walkDir(targetPath);
    let humanSize;
    if (totalSize < 1024) humanSize = totalSize + ' B';
    else if (totalSize < 1048576) humanSize = (totalSize / 1024).toFixed(1) + ' KB';
    else if (totalSize < 1073741824) humanSize = (totalSize / 1048576).toFixed(1) + ' MB';
    else humanSize = (totalSize / 1073741824).toFixed(2) + ' GB';
    return { totalSize, humanSize, fileCount };
  } catch (err) {
    return { totalSize: 0, humanSize: '0 B', fileCount: 0, error: err.message };
  }
});

// Simplified storage handlers (no Docker volume operations needed)
ipcMain.handle('setup-local-mirror', async () => ({ success: true, filesCopied: 0, errors: [] }));
ipcMain.handle('sync-storage-mirror', async () => ({ success: true, pulled: 0, pushed: 0, errors: [] }));
ipcMain.handle('set-sync-interval', async (_event, minutes) => ({ success: true, intervalMinutes: minutes }));
ipcMain.handle('scan-merge-preview', async () => ({ folders: [], totalNew: 0, totalSkipped: 0, sourceSize: 0, humanSize: '0 B' }));
ipcMain.handle('merge-storage', async () => ({ success: true, copied: 0, skipped: 0, errors: [] }));
ipcMain.handle('migrate-storage', async (_event, destPath) => {
  try {
    const validated = validatePath(destPath);
    fs.mkdirSync(validated, { recursive: true });
    // Copy from current memory dir to new location
    if (process.platform === 'win32') {
      execFile('xcopy', [MEMORY_DIR, validated, '/E', '/I', '/H', '/Y'], { timeout: 120000 });
    } else {
      execFile('cp', ['-r', MEMORY_DIR + '/.', validated], { timeout: 120000 });
    }
    return { success: true };
  } catch (err) {
    return { success: false, error: err.message };
  }
});

ipcMain.handle('restart-computer', () => {
  if (process.platform === 'win32') {
    execFile('shutdown', ['/r', '/t', '5']);
  } else {
    execFile('loginctl', ['reboot']);
  }
  return true;
});
