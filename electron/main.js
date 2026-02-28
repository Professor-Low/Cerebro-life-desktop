const { app, BrowserWindow, ipcMain, dialog, shell, session } = require('electron');
const path = require('path');
const fs = require('fs');
const net = require('net');
const { spawn, execFile } = require('child_process');
const { autoUpdater } = require('electron-updater');
const { DockerManager } = require('./docker-manager');
const { LicenseManager } = require('./license-manager');
const { createTray, updateTrayStatus } = require('./tray');

process.on('uncaughtException', (err) => {
  console.error('[Main] Uncaught exception:', err);
});
process.on('unhandledRejection', (reason) => {
  console.error('[Main] Unhandled rejection:', reason);
});

// Linux GPU handling: use real GPU if available, avoid SwiftShader fallback
if (process.platform === 'linux') {
  // Force Chromium to use the actual GPU (NVIDIA/Intel) instead of SwiftShader blocklist fallback
  app.commandLine.appendSwitch('ignore-gpu-blocklist');
  app.commandLine.appendSwitch('enable-gpu-rasterization');
  // Disable SwiftShader specifically — if no real GPU, prefer basic software rendering over SwiftShader
  app.commandLine.appendSwitch('disable-software-rasterizer');
}

// Windows DPI / rendering fix — prevents blurry text on scaled displays
if (process.platform === 'win32') {
  app.commandLine.appendSwitch('high-dpi-support', '1');
  app.commandLine.appendSwitch('enable-use-zoom-for-dsf', 'false');
}

const isDev = process.env.CEREBRO_DEV === '1';
const dockerManager = new DockerManager();
const licenseManager = new LicenseManager();

// Prevent multiple instances — focus existing window if second copy is launched
const gotTheLock = app.requestSingleInstanceLock();
if (!gotTheLock) {
  // Another instance already has the lock — quit immediately
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

// --- Chrome CDP for Docker browser access ---

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
    socket.once('connect', () => {
      socket.destroy();
      resolve(true);
    });
    socket.once('error', () => {
      socket.destroy();
      resolve(false);
    });
    socket.once('timeout', () => {
      socket.destroy();
      resolve(false);
    });
    socket.connect(CDP_PORT, '127.0.0.1');
  });
}

async function ensureChromeWithCDP() {
  // Check if CDP is already available (user's Chrome or previous launch)
  if (await isCdpAvailable()) {
    console.log('[CDP] Chrome already listening on port', CDP_PORT);
    return null;
  }

  const chromePath = findChromePath();
  if (!chromePath) {
    console.log('[CDP] Chrome not found, Docker agents will not have browser access');
    return null;
  }

  const cdpProfileDir = path.join(app.getPath('userData'), 'cerebro-chrome-cdp');
  if (!fs.existsSync(cdpProfileDir)) {
    fs.mkdirSync(cdpProfileDir, { recursive: true });
  }

  const args = [
    `--remote-debugging-port=${CDP_PORT}`,
    `--remote-allow-origins=*`,
    `--user-data-dir=${cdpProfileDir}`,
    '--no-first-run',
    '--no-default-browser-check',
  ];

  console.log(`[CDP] Launching Chrome for CDP: ${chromePath}`);
  const proc = spawn(chromePath, args, {
    stdio: 'ignore',
    detached: false,
  });

  proc.on('error', (err) => {
    console.error('[CDP] Failed to launch Chrome:', err.message);
  });

  proc.on('exit', (code) => {
    console.log(`[CDP] Chrome CDP process exited (code ${code})`);
    if (cdpChromeProcess === proc) cdpChromeProcess = null;
  });

  // Wait up to 10 seconds for CDP to become available
  for (let i = 0; i < 20; i++) {
    if (await isCdpAvailable()) {
      console.log('[CDP] Chrome CDP ready on port', CDP_PORT);
      return proc;
    }
    await new Promise(r => setTimeout(r, 500));
  }

  console.error('[CDP] Chrome launched but CDP not responding after 10s');
  try { proc.kill(); } catch (_) {}
  return null;
}

function createSplashWindow() {
  splashWindow = new BrowserWindow({
    width: 400,
    height: 300,
    frame: false,
    transparent: true,
    resizable: false,
    alwaysOnTop: true,
    skipTaskbar: true,
    webPreferences: {
      nodeIntegration: false,
      contextIsolation: true,
    },
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

  mainWindow.on('closed', () => {
    mainWindow = null;
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

  updateSplashStatus('Checking license...');
  const result = await licenseManager.revalidate();
  if (result.valid) {
    console.log(`[Main] License valid (plan: ${result.plan}${result.offline ? ', offline mode' : ''})`);
    return true;
  }

  console.log(`[Main] License invalid: ${result.reason}`);
  return false;
}

// --- Dynamic MCP config from Cerebro repo ---
const MCP_CONFIG_URL = 'https://raw.githubusercontent.com/Professor-Low/Cerebro/main/config/mcp-desktop.json';

async function fetchMcpConfig() {
  const https = require('https');
  return new Promise((resolve, reject) => {
    const req = https.get(MCP_CONFIG_URL, { timeout: 5000 }, (res) => {
      if (res.statusCode !== 200) {
        reject(new Error(`HTTP ${res.statusCode}`));
        res.resume();
        return;
      }
      let data = '';
      res.on('data', (chunk) => { data += chunk; });
      res.on('end', () => {
        try {
          resolve(JSON.parse(data));
        } catch (e) {
          reject(e);
        }
      });
    });
    req.on('error', reject);
    req.on('timeout', () => { req.destroy(); reject(new Error('Timeout')); });
  });
}

/**
 * Validates remote MCP config before it touches ~/.claude.json.
 * Drops unknown servers, blocks dangerous commands/images/flags.
 */
function validateRemoteMcpConfig(raw) {
  if (!raw || typeof raw !== 'object' || Array.isArray(raw)) {
    console.warn('[MCP] Remote config is not a valid object, using empty');
    return { mcpServers: {} };
  }

  const ALLOWED_SERVER_NAMES = new Set(['cerebro']);
  const ALLOWED_COMMANDS = new Set(['docker', 'cerebro']);
  const ALLOWED_IMAGE_PREFIX = 'ghcr.io/professor-low/';
  const DANGEROUS_DOCKER_FLAGS = ['--privileged', '--pid=host', '--network=host', '--cap-add'];

  const validated = { mcpServers: {} };

  if (raw.mcpServers && typeof raw.mcpServers === 'object') {
    for (const [name, serverConfig] of Object.entries(raw.mcpServers)) {
      // Only allow whitelisted server names
      if (!ALLOWED_SERVER_NAMES.has(name)) {
        console.warn(`[MCP] Dropped unknown server "${name}" from remote config`);
        continue;
      }

      if (!serverConfig || typeof serverConfig !== 'object' || Array.isArray(serverConfig)) {
        console.warn(`[MCP] Dropped server "${name}": invalid config object`);
        continue;
      }

      // Validate command
      const cmd = serverConfig.command;
      if (typeof cmd !== 'string' || !ALLOWED_COMMANDS.has(cmd)) {
        console.warn(`[MCP] Dropped server "${name}": disallowed command "${cmd}"`);
        continue;
      }

      // Validate args is an array of strings
      const args = serverConfig.args;
      if (args !== undefined) {
        if (!Array.isArray(args) || !args.every(a => typeof a === 'string')) {
          console.warn(`[MCP] Dropped server "${name}": args must be array of strings`);
          continue;
        }

        // For docker commands, validate image and flags
        if (cmd === 'docker') {
          // Check for dangerous Docker flags
          const hasBlockedFlag = args.some(arg =>
            DANGEROUS_DOCKER_FLAGS.some(flag => arg === flag || arg.startsWith(flag + '='))
          );
          if (hasBlockedFlag) {
            console.warn(`[MCP] Dropped server "${name}": contains dangerous Docker flag`);
            continue;
          }

          // Find the image argument (first arg that looks like an image name, after 'run')
          const runIdx = args.indexOf('run');
          if (runIdx !== -1) {
            // Image is typically the first non-flag arg after 'run' and option args
            const imageArg = args.find((a, i) =>
              i > runIdx && !a.startsWith('-') && a.includes('/') && !a.includes('=')
            );
            if (imageArg && !imageArg.startsWith(ALLOWED_IMAGE_PREFIX)) {
              console.warn(`[MCP] Dropped server "${name}": image "${imageArg}" not from allowed prefix`);
              continue;
            }
          }
        }
      }

      // Build a clean validated entry (defense against prototype pollution)
      const clean = { command: cmd };
      if (args) clean.args = [...args];
      if (serverConfig.env && typeof serverConfig.env === 'object' && !Array.isArray(serverConfig.env)) {
        clean.env = Object.fromEntries(
          Object.entries(serverConfig.env).filter(([k, v]) => typeof k === 'string' && typeof v === 'string')
        );
      } else {
        clean.env = {};
      }

      validated.mcpServers[name] = clean;
    }
  }

  return validated;
}

function getHardcodedMcpConfig() {
  return {
    mcpServers: {
      cerebro: {
        command: 'docker',
        args: [
          'run', '--rm', '-i',
          '-v', 'cerebro_cerebro-data:/data/memory',
          '-e', 'CEREBRO_DATA_DIR=/data/memory',
          '-e', 'CEREBRO_STANDALONE=1',
          'ghcr.io/professor-low/cerebro-memory:latest',
          'cerebro', 'serve',
        ],
        env: {},
      },
    },
  };
}

async function writeMcpConfig(mcpServers) {
  // Claude Code stores MCP servers in ~/.claude.json under the "mcpServers" key.
  // Verified: `claude mcp add-json` writes to this file, and `claude mcp remove`
  // reports "File modified: ~/.claude.json".
  // We do a direct read/modify/write to avoid cmd.exe mangling JSON quotes.
  const os = require('os');
  const configPath = path.join(os.homedir(), '.claude.json');

  let config = {};
  try {
    const existing = fs.readFileSync(configPath, 'utf8');
    config = JSON.parse(existing);
  } catch (e) {
    if (e.code !== 'ENOENT') {
      console.error('[MCP] Cannot parse ~/.claude.json:', e.message);
      throw new Error('Cannot update MCP config: ~/.claude.json is not valid JSON');
    }
    // File doesn't exist yet (new user without Claude Code) — start fresh
  }

  if (!config.mcpServers) config.mcpServers = {};

  for (const [name, serverConfig] of Object.entries(mcpServers)) {
    // Preserve existing cerebro config if it has a custom volume mount
    if (name === 'cerebro' && config.mcpServers.cerebro) {
      const existingArgs = config.mcpServers.cerebro.args || [];
      const newArgs = serverConfig.args || [];
      const existingVolume = existingArgs.find(a => typeof a === 'string' && a.includes(':/data/memory'));
      const newVolume = newArgs.find(a => typeof a === 'string' && a.includes(':/data/memory'));
      if (existingVolume && newVolume && existingVolume !== newVolume) {
        console.log(`[MCP] Preserved existing cerebro config with custom mount: ${existingVolume}`);
        continue;
      }
    }
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
    // Inject current storage path so refresh doesn't overwrite custom mounts
    const dynamicConfig = getDynamicMcpConfig();
    if (remote.mcpServers) {
      remote.mcpServers.cerebro = dynamicConfig.mcpServers.cerebro;
    }
    await writeMcpConfig(remote.mcpServers);
    console.log('[MCP] Config refreshed from remote (storage-aware)');
  } catch (err) {
    console.log(`[MCP] Background refresh skipped: ${err.message}`);
  }
}

function startLicenseRefreshTimer() {
  if (licenseRefreshTimer) clearInterval(licenseRefreshTimer);

  // Re-check license every 4 hours while app is running
  licenseRefreshTimer = setInterval(async () => {
    console.log('[Main] Periodic license refresh check...');
    const result = await licenseManager.revalidate();

    if (!result.valid) {
      console.log(`[Main] License no longer valid: ${result.reason}`);

      // Notify the renderer that license died mid-session
      if (mainWindow && !mainWindow.isDestroyed()) {
        mainWindow.webContents.send('license-expired', {
          reason: result.reason,
          cancelAtPeriodEnd: result.cancelAtPeriodEnd,
        });
      }

      // Give the renderer 3 seconds to show a message, then lock the app
      setTimeout(() => {
        if (mainWindow && !mainWindow.isDestroyed()) {
          showSetupWizard();
          // Send the failure reason to the activation page once it loads
          mainWindow.webContents.once('did-finish-load', () => {
            mainWindow.webContents.send('license-failure', {
              valid: false,
              reason: result.reason,
              cancelAtPeriodEnd: result.cancelAtPeriodEnd,
            });
          });
        }
      }, 3000);

      clearInterval(licenseRefreshTimer);
      licenseRefreshTimer = null;
    } else if (result.offline) {
      console.log('[Main] License valid (offline mode)');
    }
  }, 4 * 60 * 60 * 1000); // 4 hours
}

function showSetupWizard() {
  if (splashWindow) {
    splashWindow.close();
    splashWindow = null;
  }

  mainWindow.loadFile(path.join(__dirname, 'activation.html'));
  mainWindow.show();
  mainWindow.focus();
}

async function ensureDefenderExclusion() {
  if (process.platform !== 'win32') return;

  // Check marker — but re-verify if marker is older than 7 days or says "failed"
  const markerPath = path.join(require('os').homedir(), '.cerebro', '.defender-excluded');
  let needsCheck = true;
  try {
    if (fs.existsSync(markerPath)) {
      const content = fs.readFileSync(markerPath, 'utf-8').trim();
      if (content.toLowerCase() === 'failed') {
        needsCheck = true; // Previous attempt failed — must retry
      } else {
        // Check if marker is fresh (less than 7 days old)
        const markerDate = new Date(content);
        const ageMs = Date.now() - markerDate.getTime();
        const sevenDays = 7 * 24 * 60 * 60 * 1000;
        if (!isNaN(markerDate.getTime()) && ageMs < sevenDays) {
          return; // Marker is fresh and valid — skip check
        }
        console.log('[Main] Defender exclusion marker is stale, re-verifying...');
      }
    }
  } catch {}

  if (needsCheck) {
    // Verify actual exclusion state (runs elevated check)
    const excluded = await dockerManager.isDefenderExcluded().catch(() => false);
    if (excluded) {
      // Refresh marker timestamp
      try { fs.writeFileSync(markerPath, new Date().toISOString()); } catch {}
      return;
    }
  }

  console.log('[Main] Defender exclusion not found — requesting elevation to add it');
  updateSplashStatus('Configuring Windows Defender...');
  const result = await dockerManager.addDefenderExclusion();
  if (!result.success) {
    console.warn('[Main] Defender exclusion failed (non-fatal):', result.error);
  }
}

async function startDockerStack() {
  updateSplashStatus('Starting Docker containers...');

  // Auto-detect memory config from env vars before writing compose
  // (ensures bind mount is set up on first run if CEREBRO_DATA_DIR is set)
  loadMemoryConfig();

  // Defender exclusion now runs at app startup (before license gate), so no
  // need to duplicate it here. The marker file prevents repeated UAC prompts.

  // 1. Check if Docker is installed — if not, wizard handles install
  const installed = await dockerManager.isDockerInstalled();
  if (!installed) {
    console.log('[Main] Docker not installed, deferring to wizard');
    return { ok: false, error: 'Docker is not installed' };
  }

  // 2. Check if Docker daemon is running — auto-start if not
  const running = await dockerManager.isDockerRunning();
  if (!running) {
    updateSplashStatus('Starting Docker...');
    const startResult = await dockerManager.startDockerDaemon((p) => {
      updateSplashStatus(p.message || 'Starting Docker...');
    });
    if (!startResult.success) {
      console.error('[Main] Failed to start Docker daemon:', startResult.error);
      return { ok: false, error: `Failed to start Docker daemon: ${startResult.error}` };
    }
  }

  // 3. Launch Chrome with CDP so Docker containers can access the browser
  updateSplashStatus('Starting browser for agents...');
  try {
    cdpChromeProcess = await ensureChromeWithCDP();
  } catch (err) {
    console.error('[Main] Chrome CDP launch failed (non-fatal):', err.message);
  }

  // 4. Write/refresh config (always refresh compose to fix volume mounts)
  updateSplashStatus('Writing configuration...');
  await dockerManager.writeComposeFile();
  dockerManager.writeEnvFile();
  await dockerManager.setClaudeCliPath();

  // 5. Start the compose stack
  try {
    updateSplashStatus('Starting containers...');
    await dockerManager.startStack();
    return { ok: true };
  } catch (err) {
    console.error('[Main] Docker stack failed to start:', err.message);
    const result = { ok: false, error: err.message };
    if (err.portConflict) {
      result.portConflict = true;
      result.portConflictReason = err.portConflictReason;
    }
    return result;
  }
}

async function loadFrontend() {
  updateSplashStatus('Loading Cerebro...');

  let retries = 0;
  const maxRetries = 15;

  mainWindow.loadURL('http://localhost:61000');

  mainWindow.webContents.on('did-finish-load', () => {
    if (splashWindow) {
      splashWindow.close();
      splashWindow = null;
    }
    mainWindow.show();
    mainWindow.focus();

    // Check for updates in the background — aggressive early, then every 30 min.
    // Early checks catch releases published right after the user launched the app.
    checkForUpdatesQuietly();
    setTimeout(checkForUpdatesQuietly, 60 * 1000);       // 1 min after launch
    setTimeout(checkForUpdatesQuietly, 5 * 60 * 1000);   // 5 min after launch
    setInterval(checkForUpdatesQuietly, 30 * 60 * 1000); // then every 30 min
  });

  mainWindow.webContents.on('did-fail-load', (_event, errorCode, errorDescription) => {
    retries++;
    console.error(`[Main] Frontend failed to load (attempt ${retries}/${maxRetries}): ${errorCode} ${errorDescription}`);
    if (retries < maxRetries) {
      setTimeout(() => {
        if (mainWindow) mainWindow.loadURL('http://localhost:61000');
      }, 2000);
    } else {
      console.error('[Main] Frontend failed to load after max retries, showing wizard');
      if (mainWindow) showSetupWizard();
    }
  });
}

async function checkForUpdatesQuietly() {
  let bannerShown = false;

  // Check Docker image updates (existing behavior)
  try {
    const result = await dockerManager.checkForUpdates();
    if (result.updateAvailable && mainWindow && !mainWindow.isDestroyed()) {
      mainWindow.webContents.executeJavaScript(`
        if (typeof window.__cerebroShowUpdateBanner === 'function') {
          window.__cerebroShowUpdateBanner('docker');
        }
      `).catch(() => {});
      bannerShown = true;
    }
  } catch (err) {
    console.log('[Main] Docker update check failed:', err.message);
  }

  // Check Electron app updates (auto-downloads if available)
  try {
    await autoUpdater.checkForUpdates();
  } catch (err) {
    console.log('[Main] Electron update check failed:', err.message);
  }

  // Fallback: compare app version against latest GitHub release.
  // This catches cases where both docker manifest inspect and electron-updater
  // fail silently (e.g. Docker experimental features disabled, AV blocking downloads).
  if (!bannerShown) {
    try {
      const https = require('https');
      const latestVersion = await new Promise((resolve, reject) => {
        const req = https.get('https://api.github.com/repos/Professor-Low/Cerebro-life-desktop/releases/latest', {
          headers: { 'User-Agent': 'Cerebro-Desktop/' + app.getVersion() },
          timeout: 10000,
        }, (res) => {
          if (res.statusCode === 302 || res.statusCode === 301) {
            return reject(new Error('redirect'));
          }
          let body = '';
          res.on('data', (chunk) => { body += chunk; });
          res.on('end', () => {
            try {
              const data = JSON.parse(body);
              resolve(data.tag_name ? data.tag_name.replace(/^v/, '') : null);
            } catch { resolve(null); }
          });
        });
        req.on('error', reject);
        req.on('timeout', () => { req.destroy(); reject(new Error('timeout')); });
      });

      if (latestVersion && latestVersion !== app.getVersion()) {
        // Simple semver comparison: split on dots, compare numerically
        const current = app.getVersion().split('.').map(Number);
        const latest = latestVersion.split('.').map(Number);
        let isNewer = false;
        for (let i = 0; i < 3; i++) {
          if ((latest[i] || 0) > (current[i] || 0)) { isNewer = true; break; }
          if ((latest[i] || 0) < (current[i] || 0)) break;
        }
        if (isNewer && mainWindow && !mainWindow.isDestroyed()) {
          console.log(`[Main] GitHub API fallback: update available (${app.getVersion()} → ${latestVersion})`);
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
}

// --- Linux desktop integration (first launch) ---
function installDesktopIntegration() {
  if (process.platform !== 'linux') return;

  const os = require('os');
  const homeDir = os.homedir();
  const markerPath = path.join(app.getPath('userData'), '.desktop-integrated');

  if (fs.existsSync(markerPath)) return;

  try {
    const iconSrc = path.join(__dirname, '..', 'assets', 'icon-256.png');
    const iconFile = fs.existsSync(iconSrc) ? iconSrc : path.join(__dirname, '..', 'assets', 'icon.png');

    if (!fs.existsSync(iconFile)) {
      console.log('[Main] No icon file found, skipping desktop integration');
      return;
    }

    const iconDestDir = path.join(homeDir, '.local', 'share', 'icons', 'hicolor', '256x256', 'apps');
    fs.mkdirSync(iconDestDir, { recursive: true });
    fs.copyFileSync(iconFile, path.join(iconDestDir, 'cerebro.png'));

    const desktopDir = path.join(homeDir, '.local', 'share', 'applications');
    fs.mkdirSync(desktopDir, { recursive: true });

    const appImagePath = process.env.APPIMAGE || process.execPath;
    const desktopEntry = [
      '[Desktop Entry]',
      'Name=Cerebro',
      'Comment=Your AI, Everywhere',
      `Exec="${appImagePath}" %U`,
      'Icon=cerebro',
      'Type=Application',
      'Categories=Utility;ArtificialIntelligence;Science;',
      'StartupWMClass=cerebro-desktop',
      'Terminal=false',
    ].join('\n') + '\n';

    fs.writeFileSync(path.join(desktopDir, 'cerebro.desktop'), desktopEntry);

    const { execFileSync } = require('child_process');
    try {
      execFileSync('update-desktop-database', [desktopDir], { timeout: 5000 });
    } catch (_) { /* not critical */ }
    try {
      execFileSync('gtk-update-icon-cache', ['-f', '-t', path.join(homeDir, '.local', 'share', 'icons', 'hicolor')], { timeout: 5000 });
    } catch (_) { /* not critical */ }

    fs.writeFileSync(markerPath, new Date().toISOString());
    console.log('[Main] Desktop integration installed');
  } catch (err) {
    console.error('[Main] Desktop integration failed:', err.message);
  }
}

async function shutdown() {
  console.log('[Main] Shutting down...');
  await dockerManager.stopStack();
}

// Focus existing window when a second instance is launched
app.on('second-instance', () => {
  if (mainWindow) {
    if (mainWindow.isMinimized()) mainWindow.restore();
    mainWindow.show();
    mainWindow.focus();
  }
});

// App lifecycle
app.whenReady().then(async () => {
  console.log('[Main] Cerebro Desktop starting (Docker architecture)');

  // ── STEP 0: Windows Defender exclusion ──────────────────────────
  // MUST run before ANY network activity, Docker calls, or Chrome CDP.
  // Defender's behavioral detection (Behavior:Win32/LummaStealer.CER!MTB)
  // triggers on network/Docker patterns and kills the process within seconds.
  // The NSIS installer also adds exclusions at install time, but this covers
  // upgrades from older versions and manual installs.
  try {
    await ensureDefenderExclusion();
  } catch (err) {
    console.warn('[Main] Defender exclusion failed (non-fatal):', err.message);
  }

  installDesktopIntegration();

  // Configure electron auto-updater (AFTER Defender exclusion)
  autoUpdater.autoDownload = true;
  autoUpdater.autoInstallOnAppQuit = true;
  autoUpdater.logger = null;

  autoUpdater.on('update-downloaded', (info) => {
    console.log(`[Main] Electron update downloaded: v${info.version}`);
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
    console.log('[Main] Auto-updater error (non-fatal):', err.message);
  });

  // Grant microphone & audio permissions so voice/STT works without prompts
  session.defaultSession.setPermissionRequestHandler((webContents, permission, callback) => {
    const allowed = ['media', 'audioCapture', 'mediaKeySystem'];
    callback(allowed.includes(permission));
  });
  session.defaultSession.setPermissionCheckHandler((_webContents, permission) => {
    const allowed = ['media', 'audioCapture', 'mediaKeySystem'];
    return allowed.includes(permission);
  });

  createSplashWindow();
  setTimeout(() => {
    if (splashWindow && !splashWindow.isDestroyed()) {
      console.error('[Main] Splash timeout — startup took too long');
      splashWindow.close();
      splashWindow = null;
      if (mainWindow) showSetupWizard();
    }
  }, 60000);
  createMainWindow();

  // Create system tray
  const trayIconPath = path.join(__dirname, '..', 'assets', 'tray-icon.png');
  tray = createTray(mainWindow, trayIconPath);

  // Post-restart resume: check for saved setup state before license gate
  const savedState = dockerManager.loadSetupState();
  if (savedState && savedState.step === 'needs-setup') {
    console.log('[Main] Resuming setup after restart');
    showSetupWizard();
    // Signal the activation page to jump to the setup step once loaded
    mainWindow.webContents.on('did-finish-load', () => {
      mainWindow.webContents.send('resume-after-restart', savedState);
    });
    return;
  }

  // License gate
  const licensed = await checkLicense();
  if (!licensed) {
    const licenseStatus = licenseManager.getStatus();
    showSetupWizard();
    // Tell the activation page WHY the user is here (expired vs first-time)
    mainWindow.webContents.once('did-finish-load', () => {
      mainWindow.webContents.send('license-failure', licenseStatus);
    });
    return;
  }

  // Forward credential events to renderer
  dockerManager.on('credentials-expired', (data) => {
    if (mainWindow) mainWindow.webContents.send('credentials-expired', data);
  });
  dockerManager.on('credentials-refreshed', (data) => {
    if (mainWindow) mainWindow.webContents.send('credentials-refreshed', data);
  });

  // Returning user: start Docker stack and load frontend
  const dockerResult = await startDockerStack();
  if (!dockerResult.ok) {
    console.error('[Main] Docker start failed:', dockerResult.error);
    showSetupWizard();
    // If it's a port conflict, send the detailed error to the activation page
    if (dockerResult.portConflict) {
      mainWindow.webContents.once('did-finish-load', () => {
        mainWindow.webContents.send('port-conflict', {
          reason: dockerResult.portConflictReason,
          error: dockerResult.error,
        });
      });
    }
    return;
  }

  await loadFrontend();
  updateTrayStatus(tray, 'running');
  startLicenseRefreshTimer();
  refreshMcpConfigSilently();
});

app.on('window-all-closed', () => {
  // Don't quit - we stay in system tray
});

app.on('activate', () => {
  if (mainWindow) {
    mainWindow.show();
    mainWindow.focus();
  }
});

app.on('before-quit', async (e) => {
  if (!isQuitting) {
    isQuitting = true;
    e.preventDefault();
    if (!isAutoUpdating) {
      updateTrayStatus(tray, 'stopping');
      await shutdown();
    }
    // Kill CDP Chrome process if we launched it
    if (cdpChromeProcess) {
      try { cdpChromeProcess.kill(); } catch (_) {}
      cdpChromeProcess = null;
    }
    app.quit();
  }
});

// --- IPC handlers ---

// App info
ipcMain.handle('get-app-version', () => app.getVersion());
ipcMain.handle('get-edition', () => 'docker');

// License
ipcMain.handle('get-license-status', () => licenseManager.getStatus());
ipcMain.handle('activate-license', async (_event, key) => {
  const result = await licenseManager.activate(key);
  if (result.success) {
    console.log(`[Main] License activated (plan: ${result.plan})`);
  }
  return result;
});

ipcMain.handle('refresh-license', async () => {
  const result = await licenseManager.revalidate();
  if (result.valid) {
    console.log(`[Main] License refresh OK (plan: ${result.plan})`);
  } else {
    console.log(`[Main] License refresh failed: ${result.reason}`);
  }
  return result;
});

// Docker status
ipcMain.handle('check-docker', async () => {
  const installed = await dockerManager.isDockerInstalled();
  if (!installed) return { installed: false, running: false };
  const running = await dockerManager.isDockerRunning();
  return { installed, running };
});

ipcMain.handle('check-claude-code', async () => {
  return dockerManager.isClaudeInstalled();
});

ipcMain.handle('get-docker-status', async () => {
  return dockerManager.getStatus();
});

ipcMain.handle('get-docker-logs', async () => {
  return dockerManager.getLogs();
});

// Docker install & daemon management
ipcMain.handle('install-docker', async () => {
  try {
    const result = await dockerManager.installDocker((progress) => {
      if (mainWindow) {
        mainWindow.webContents.send('docker-install-progress', progress);
      }
    });
    return result;
  } catch (err) {
    return { success: false, error: err.message };
  }
});

ipcMain.handle('check-wsl', async () => {
  return dockerManager.checkWslAvailable();
});

ipcMain.handle('start-docker-daemon', async () => {
  try {
    const result = await dockerManager.startDockerDaemon((progress) => {
      if (mainWindow) {
        mainWindow.webContents.send('docker-start-progress', progress);
      }
    });
    return result;
  } catch (err) {
    return { success: false, error: err.message };
  }
});

// Setup & lifecycle
ipcMain.handle('setup-docker', async () => {
  try {
    await dockerManager.writeComposeFile();
    dockerManager.writeEnvFile();
    await dockerManager.setClaudeCliPath();
    return { success: true };
  } catch (err) {
    return { success: false, error: err.message };
  }
});

ipcMain.handle('pull-images', async () => {
  try {
    await dockerManager.pullImages((progress) => {
      if (mainWindow) {
        mainWindow.webContents.send('pull-progress', progress);
      }
    });
    return { success: true };
  } catch (err) {
    return { success: false, error: err.message };
  }
});

ipcMain.handle('install-kokoro-tts', async () => {
  try {
    await dockerManager.installKokoroTts((progress) => {
      if (mainWindow) {
        mainWindow.webContents.send('kokoro-install-progress', progress);
      }
    });
    return { success: true };
  } catch (err) {
    return { success: false, error: err.message };
  }
});

ipcMain.handle('start-stack', async () => {
  try {
    await dockerManager.startStack();
    return { success: true };
  } catch (err) {
    return { success: false, error: err.message };
  }
});

ipcMain.handle('stop-stack', async () => {
  try {
    await dockerManager.stopStack();
    return { success: true };
  } catch (err) {
    return { success: false, error: err.message };
  }
});

// Updates
ipcMain.handle('check-for-updates', async () => {
  // Check Docker image updates
  const dockerResult = await dockerManager.checkForUpdates().catch(() => ({ updateAvailable: false }));
  if (dockerResult.updateAvailable) return dockerResult;

  // Check Electron app updates via autoUpdater
  try {
    const checkResult = await autoUpdater.checkForUpdates();
    if (electronUpdateReady) return { updateAvailable: true, type: 'electron' };
    // If download is in progress, wait briefly for it
    if (checkResult && checkResult.downloadPromise) {
      await Promise.race([checkResult.downloadPromise, new Promise(r => setTimeout(r, 5000))]);
      if (electronUpdateReady) return { updateAvailable: true, type: 'electron' };
    }
  } catch {}

  // Fallback: compare version against latest GitHub release
  try {
    const https = require('https');
    const latestVersion = await new Promise((resolve, reject) => {
      const req = https.get('https://api.github.com/repos/Professor-Low/Cerebro-life-desktop/releases/latest', {
        headers: { 'User-Agent': 'Cerebro-Desktop/' + app.getVersion() },
        timeout: 10000,
      }, (res) => {
        if (res.statusCode === 302 || res.statusCode === 301) return reject(new Error('redirect'));
        let body = '';
        res.on('data', (chunk) => { body += chunk; });
        res.on('end', () => {
          try { resolve(JSON.parse(body).tag_name?.replace(/^v/, '') || null); }
          catch { resolve(null); }
        });
      });
      req.on('error', reject);
      req.on('timeout', () => { req.destroy(); reject(new Error('timeout')); });
    });

    if (latestVersion) {
      const current = app.getVersion().split('.').map(Number);
      const latest = latestVersion.split('.').map(Number);
      for (let i = 0; i < 3; i++) {
        if ((latest[i] || 0) > (current[i] || 0)) return { updateAvailable: true, type: 'github', version: latestVersion };
        if ((latest[i] || 0) < (current[i] || 0)) break;
      }
    }
  } catch {}

  return { updateAvailable: false };
});

ipcMain.handle('apply-update', async () => {
  try {
    // Best-effort Defender exclusion — never block the update.
    // The NSIS installer (installer.nsh) handles Defender exclusions during
    // installation, so quitAndInstall() will handle it properly. We attempt
    // it here as a courtesy but NEVER gate the update on it — blocking the
    // update is worse than a potential Defender issue (which the installer fixes).
    if (process.platform === 'win32') {
      try {
        await ensureDefenderExclusion();
      } catch (e) {
        console.warn('[Update] Defender exclusion attempt failed (non-blocking):', e.message);
      }
    }

    const dockerResult = await dockerManager.checkForUpdates();
    const needsDocker = dockerResult.updateAvailable;

    // Re-check for pending Electron updates (electronUpdateReady resets on restart)
    if (!electronUpdateReady) {
      try {
        const checkResult = await autoUpdater.checkForUpdates();
        if (checkResult && checkResult.downloadPromise) {
          await checkResult.downloadPromise;
        }
        // Give the update-downloaded event a moment to fire
        await new Promise(r => setTimeout(r, 2000));
      } catch {}
    }
    const needsElectron = electronUpdateReady;

    // Set update suppression timestamp BEFORE navigating away from the frontend.
    // The frontend's own localStorage.setItem never runs because we replace the page
    // with a splash screen, destroying the JS context that was awaiting the IPC result.
    if (mainWindow && !mainWindow.isDestroyed()) {
      try {
        await mainWindow.webContents.executeJavaScript(
          `try{localStorage.setItem('cerebro_last_update_ts',String(Date.now()))}catch(e){}`
        );
      } catch {}
    }

    // Show splash screen
    if (mainWindow) {
      mainWindow.loadURL('data:text/html,' + encodeURIComponent(`<!DOCTYPE html>
<html><head><style>
  * { margin:0; padding:0; box-sizing:border-box; }
  body { background:#0a0a0f; color:#e2e8f0; font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif;
    display:flex; flex-direction:column; align-items:center; justify-content:center; height:100vh;
    overflow:hidden; }
  .update-container { text-align:center; width:340px; }
  .logo-ring { width:64px; height:64px; margin:0 auto 28px; position:relative; }
  .logo-ring::before { content:''; position:absolute; inset:0; border-radius:50%;
    border:2.5px solid rgba(139,92,246,0.15); }
  .logo-ring::after { content:''; position:absolute; inset:0; border-radius:50%;
    border:2.5px solid transparent; border-top-color:#8b5cf6; border-right-color:#a78bfa;
    animation:spin 1.2s cubic-bezier(0.45,0.05,0.55,0.95) infinite; }
  @keyframes spin { to { transform:rotate(360deg); } }
  .logo-dot { position:absolute; top:50%; left:50%; transform:translate(-50%,-50%);
    width:20px; height:20px; background:linear-gradient(135deg,#8b5cf6,#a78bfa);
    border-radius:50%; box-shadow:0 0 20px rgba(139,92,246,0.4); }
  h2 { color:#e2e8f0; font-size:1.15rem; font-weight:600; margin-bottom:6px; letter-spacing:-0.01em; }
  #status { color:#94a3b8; font-size:0.8rem; margin-bottom:20px; min-height:1.2em; }
  .bar-track { width:100%; height:6px; background:rgba(139,92,246,0.1); border-radius:3px;
    overflow:hidden; position:relative; }
  .bar-fill { height:100%; width:0%; border-radius:3px;
    background:linear-gradient(90deg,#7c3aed,#8b5cf6,#a78bfa);
    transition:width 0.6s cubic-bezier(0.25,0.46,0.45,0.94);
    box-shadow:0 0 12px rgba(139,92,246,0.3); position:relative; }
  .bar-fill::after { content:''; position:absolute; top:0; left:0; right:0; bottom:0;
    background:linear-gradient(90deg,transparent,rgba(255,255,255,0.15),transparent);
    animation:shimmer 1.8s ease-in-out infinite; }
  @keyframes shimmer { 0%{transform:translateX(-100%)} 100%{transform:translateX(100%)} }
  #percent { color:#a78bfa; font-size:0.75rem; margin-top:8px; font-variant-numeric:tabular-nums; }
</style></head><body>
  <div class="update-container">
    <div class="logo-ring"><div class="logo-dot"></div></div>
    <h2>Updating Cerebro</h2>
    <p id="status">Preparing update...</p>
    <div class="bar-track"><div class="bar-fill" id="bar"></div></div>
    <p id="percent"></p>
  </div>
</body></html>`));
    }

    // Helper to update the splash screen progress bar + status
    const updateSplash = (message, percent) => {
      if (!mainWindow || mainWindow.isDestroyed()) return;
      try {
        mainWindow.webContents.executeJavaScript(`
          var s=document.getElementById('status');
          var b=document.getElementById('bar');
          var p=document.getElementById('percent');
          if(s)s.textContent=${JSON.stringify(message)};
          if(b)b.style.width='${Math.min(100, Math.max(0, percent))}%';
          if(p)p.textContent=${percent > 0 ? JSON.stringify(Math.round(percent) + '%') : "''"};
        `).catch(() => {});
      } catch {}
    };

    // Step 1: Docker update (if needed)
    if (needsDocker) {
      await dockerManager.applyUpdate((progress) => {
        // Map stages to clean messages + progress percentages
        const stage = progress.stage || '';
        const msg = progress.message || '';
        if (stage === 'pulling_core') {
          // 0-50%: core image download
          const pct = progress.percent != null ? progress.percent * 0.5 : 10;
          updateSplash('Downloading core services...', pct);
        } else if (stage === 'pulling_optional') {
          // 50-70%: optional image download
          const pct = 50 + (progress.percent != null ? progress.percent * 0.2 : 5);
          updateSplash('Downloading voice engine...', pct);
        } else if (stage === 'pulling_optional_skip') {
          updateSplash('Voice engine skipped', 70);
        } else if (stage === 'stopping') {
          updateSplash('Stopping services...', 75);
        } else if (stage === 'starting') {
          updateSplash('Starting services...', 85);
        } else if (stage === 'done') {
          updateSplash('Update complete!', 100);
        } else {
          // Fallback — still don't show raw Docker output
          updateSplash('Updating...', 5);
        }
      });
      // Re-sync frontend and env after Docker update to ensure latest files are mounted
      try { dockerManager.writeEnvFile(); } catch {}
    }

    // Step 2: Electron update (if needed) — quits and reinstalls silently
    if (needsElectron) {
      updateSplash('Installing app update...', 95);
      await new Promise(r => setTimeout(r, 1000));
      isAutoUpdating = true;  // Skip Docker shutdown in before-quit
      autoUpdater.quitAndInstall(true, true);  // silent install, force relaunch
      return { success: true };
    }

    // Docker-only update — reload frontend and re-inject suppression timestamp
    if (mainWindow && !mainWindow.isDestroyed()) {
      mainWindow.loadURL('http://localhost:61000');
      mainWindow.webContents.once('did-finish-load', () => {
        if (mainWindow && !mainWindow.isDestroyed()) {
          mainWindow.webContents.executeJavaScript(
            `try{localStorage.setItem('cerebro_last_update_ts',String(Date.now()))}catch(e){}`
          ).catch(() => {});
        }
      });
    }
    return { success: true };
  } catch (err) {
    if (mainWindow && !mainWindow.isDestroyed()) {
      mainWindow.loadURL('http://localhost:61000');
    }
    return { success: false, error: err.message };
  }
});

// MCP configuration (for Claude Code integration)
// Fetches config dynamically from the Cerebro repo; falls back to hardcoded.
ipcMain.handle('configure-mcp', async () => {
  try {
    let mcpConfig;
    let source;

    try {
      const rawConfig = await fetchMcpConfig();
      mcpConfig = validateRemoteMcpConfig(rawConfig);
      source = 'remote';
    } catch (fetchErr) {
      console.log(`[MCP] Remote fetch failed (${fetchErr.message}), using dynamic config`);
      mcpConfig = getDynamicMcpConfig();
      source = 'dynamic';
    }

    // Always inject the current storage path into the cerebro MCP server entry
    const dynamicConfig = getDynamicMcpConfig();
    if (mcpConfig.mcpServers) {
      mcpConfig.mcpServers.cerebro = dynamicConfig.mcpServers.cerebro;
    }

    const configPath = await writeMcpConfig(mcpConfig.mcpServers);
    console.log(`[MCP] Config written from ${source} source (storage-aware)`);
    return { success: true, configPath };
  } catch (err) {
    return { success: false, error: err.message };
  }
});

// Wizard completion: start stack and load frontend
ipcMain.handle('wizard-complete', async () => {
  try {
    const result = await startDockerStack();
    if (!result.ok) {
      const response = { success: false, error: result.error || 'Failed to start Docker stack' };
      if (result.portConflict) {
        response.portConflict = true;
        response.portConflictReason = result.portConflictReason;
      }
      return response;
    }
    await loadFrontend();
    updateTrayStatus(tray, 'running');
    return { success: true };
  } catch (err) {
    return { success: false, error: err.message };
  }
});

ipcMain.handle('get-setup-status', async () => {
  const dockerStatus = await dockerManager.isDockerRunning().catch(() => false);
  const claudeStatus = await dockerManager.isClaudeInstalled().catch(() => ({ installed: false }));

  return {
    licensed: licenseManager.getStatus().valid,
    dockerInstalled: await dockerManager.isDockerInstalled().catch(() => false),
    dockerRunning: dockerStatus,
    claudeInstalled: claudeStatus.installed,
    setupComplete: dockerManager.isSetupComplete(),
    stackRunning: dockerManager.isRunning(),
  };
});

// Settings
ipcMain.handle('toggle-autostart', (_event, enabled) => {
  app.setLoginItemSettings({
    openAtLogin: enabled,
    path: process.env.APPIMAGE || app.getPath('exe'),
  });
  return app.getLoginItemSettings().openAtLogin;
});

ipcMain.handle('get-autostart', () => {
  return app.getLoginItemSettings().openAtLogin;
});

ipcMain.handle('enable-autostart', () => {
  app.setLoginItemSettings({
    openAtLogin: true,
    path: process.env.APPIMAGE || app.getPath('exe'),
  });
  return true;
});

// Claude credentials
ipcMain.handle('check-claude-credentials', () => {
  return dockerManager.checkClaudeCredentials();
});

ipcMain.handle('refresh-claude-credentials', () => {
  return dockerManager.refreshClaudeCredentials();
});

ipcMain.handle('silent-refresh-oauth', async () => {
  return dockerManager.silentRefreshOAuthToken();
});

// Launch Claude CLI login — captures OAuth URL and opens in-app browser window
let _authWindow = null;

ipcMain.handle('launch-claude-login', async () => {
  try {
    // Close any existing auth window
    if (_authWindow && !_authWindow.isDestroyed()) _authWindow.close();

    // Spawn `claude auth login` with BROWSER env trick to capture the URL
    // On Unix: BROWSER=echo prints the URL to stdout instead of opening a browser
    // On Windows: we parse stdout/stderr for the URL pattern as fallback
    const env = { ...process.env };
    if (process.platform !== 'win32') {
      env.BROWSER = 'echo';
    }

    return new Promise((resolve) => {
      const proc = spawn('claude', ['auth', 'login'], {
        env,
        shell: true,
        stdio: ['ignore', 'pipe', 'pipe'],
      });

      let output = '';
      let urlFound = false;
      const urlRegex = /https:\/\/[^\s"'<>]+/g;

      function handleOutput(chunk) {
        const text = chunk.toString();
        output += text;
        console.log('[Claude Auth]', text.trim());

        // Look for OAuth URL in output
        if (!urlFound) {
          const matches = text.match(urlRegex);
          if (matches) {
            // Find the auth/OAuth URL (prefer claude.ai or anthropic URLs)
            const authUrl = matches.find(u =>
              u.includes('claude.ai') || u.includes('anthropic.com') || u.includes('oauth')
            ) || matches[0];

            if (authUrl) {
              urlFound = true;
              openAuthWindow(authUrl);
              resolve({ success: true, message: 'Login window opened inside Cerebro.' });
            }
          }
        }
      }

      proc.stdout.on('data', handleOutput);
      proc.stderr.on('data', handleOutput);

      // If CLI opens browser directly (Windows), we still poll for credentials
      setTimeout(() => {
        if (!urlFound) {
          // Fallback: CLI probably opened browser directly, still poll for completion
          console.log('[Claude Auth] No URL captured, CLI may have opened browser directly');
          resolve({ success: true, message: 'Login opened — complete authentication in the browser.' });
        }
      }, 8000);

      proc.on('error', (err) => {
        console.error('[Claude Auth] Process error:', err.message);
        if (!urlFound) {
          resolve({ success: false, error: 'Could not start claude auth login: ' + err.message });
        }
      });

      proc.on('close', (code) => {
        console.log('[Claude Auth] Process exited with code', code);
        // Check if credentials were written successfully
        setTimeout(() => {
          const status = dockerManager.checkClaudeCredentials();
          if (status.valid) {
            const refreshResult = dockerManager.refreshClaudeCredentials();
            if (mainWindow) {
              mainWindow.webContents.send('credentials-refreshed', {
                expiresIn: refreshResult.expiresIn || status.expiresIn,
                message: 'Login successful — credentials refreshed',
              });
            }
            if (_authWindow && !_authWindow.isDestroyed()) _authWindow.close();
          }
        }, 1000);
      });

      // Poll for fresh credentials while login is in progress (5s intervals, 5 min max)
      let attempts = 0;
      const maxAttempts = 60;
      const watcher = setInterval(() => {
        attempts++;
        const status = dockerManager.checkClaudeCredentials();
        if (status.valid && status.expiresIn > 30) {
          clearInterval(watcher);
          const refreshResult = dockerManager.refreshClaudeCredentials();
          if (mainWindow) {
            mainWindow.webContents.send('credentials-refreshed', {
              expiresIn: refreshResult.expiresIn || status.expiresIn,
              message: 'Login successful — credentials refreshed',
            });
          }
          // Close auth window and kill CLI process
          if (_authWindow && !_authWindow.isDestroyed()) _authWindow.close();
          try { proc.kill(); } catch (_) {}
        }
        if (attempts >= maxAttempts) {
          clearInterval(watcher);
          try { proc.kill(); } catch (_) {}
        }
      }, 5000);
    });
  } catch (err) {
    return { success: false, error: err.message };
  }
});

function openAuthWindow(url) {
  _authWindow = new BrowserWindow({
    width: 500,
    height: 700,
    title: 'Cerebro — Claude Authentication',
    parent: mainWindow,
    modal: true,
    autoHideMenuBar: true,
    webPreferences: {
      nodeIntegration: false,
      contextIsolation: true,
    },
  });

  _authWindow.loadURL(url);

  _authWindow.on('closed', () => {
    _authWindow = null;
  });

  // Watch for successful redirects back to localhost (OAuth callback)
  _authWindow.webContents.on('will-redirect', (_event, redirectUrl) => {
    if (redirectUrl.includes('localhost') || redirectUrl.includes('127.0.0.1')) {
      console.log('[Claude Auth] OAuth callback detected, login completing...');
    }
  });
}

// Chrome CDP launch (called by frontend when user clicks "Launch Chrome")
ipcMain.handle('launch-chrome-cdp', async () => {
  try {
    if (await isCdpAvailable()) {
      return { success: true, alreadyRunning: true };
    }
    cdpChromeProcess = await ensureChromeWithCDP();
    if (cdpChromeProcess || await isCdpAvailable()) {
      return { success: true };
    }
    return { success: false, error: 'Chrome failed to start' };
  } catch (err) {
    return { success: false, error: err.message };
  }
});

// Chrome CDP stop (called by frontend when user clicks browser orb to stop)
ipcMain.handle('stop-chrome-cdp', async () => {
  try {
    if (cdpChromeProcess) {
      try { cdpChromeProcess.kill(); } catch (_) {}
      cdpChromeProcess = null;
      return { success: true };
    }
    // If we didn't launch it but CDP is still available, try to kill via PID
    if (await isCdpAvailable()) {
      try {
        const { execFileSync } = require('child_process');
        // Use netstat to find PID on CDP port, then taskkill
        const result = execFileSync('cmd', ['/c', 'netstat -ano | findstr :9222 | findstr LISTENING'], { encoding: 'utf-8', timeout: 3000 });
        const lines = result.trim().split('\n');
        const pids = new Set();
        for (const line of lines) {
          const parts = line.trim().split(/\s+/);
          const pid = parts[parts.length - 1];
          if (pid && pid !== '0') pids.add(pid);
        }
        for (const pid of pids) {
          try { execFileSync('taskkill', ['/F', '/PID', pid], { timeout: 3000 }); } catch (_) {}
        }
        return { success: true };
      } catch (e) {
        return { success: false, error: 'Could not find Chrome CDP process: ' + e.message };
      }
    }
    return { success: true, message: 'Chrome was not running' };
  } catch (err) {
    return { success: false, error: err.message };
  }
});

// Restart & setup state
ipcMain.handle('needs-restart', async () => {
  return dockerManager.checkNeedsRestart();
});

ipcMain.handle('save-setup-state', (_event, state) => {
  dockerManager.saveSetupState(state);
  return true;
});

ipcMain.handle('load-setup-state', () => {
  return dockerManager.loadSetupState();
});

ipcMain.handle('clear-setup-state', () => {
  dockerManager.clearSetupState();
  return true;
});

// File access settings
ipcMain.handle('get-file-access-config', () => {
  return dockerManager.loadFileAccessConfig();
});

ipcMain.handle('save-file-access-config', async (_event, config) => {
  try {
    dockerManager.saveFileAccessConfig(config);
    await dockerManager.writeComposeFile();
    return { success: true };
  } catch (err) {
    return { success: false, error: err.message };
  }
});

ipcMain.handle('get-file-access-presets', () => {
  return dockerManager.getPresetMounts();
});

ipcMain.handle('browse-folder', async () => {
  try {
    const win = BrowserWindow.getFocusedWindow() || mainWindow;
    const result = await dialog.showOpenDialog(win, {
      properties: ['openDirectory'],
      title: 'Select folder to share with Cerebro',
    });
    if (result.canceled || !result.filePaths.length) return { canceled: true };
    return { canceled: false, path: result.filePaths[0] };
  } catch (err) {
    return { canceled: true, error: err.message };
  }
});

ipcMain.handle('restart-docker-stack', async () => {
  try {
    await dockerManager.stopStack();
    await dockerManager.writeComposeFile();
    await dockerManager.startStack();
    return { success: true };
  } catch (err) {
    return { success: false, error: err.message };
  }
});

// ---- Path Validation ----

/**
 * Validates a user-supplied path before any filesystem or Docker operations.
 * Throws on invalid/dangerous paths. Returns the resolved absolute path.
 * Options:
 *   allowNull: true  → returns null if input is null (for "reset to default")
 *   mustExist: true  → checks that path exists and resolves symlinks
 */
function validatePath(inputPath, opts = {}) {
  const { allowNull = false, mustExist = false } = opts;

  if (inputPath === null || inputPath === undefined) {
    if (allowNull) return null;
    throw new Error('Path is required');
  }

  if (typeof inputPath !== 'string' || inputPath.trim() === '') {
    throw new Error('Path must be a non-empty string');
  }

  // Reject raw traversal sequences before resolving
  if (inputPath.includes('..')) {
    throw new Error('Path traversal ("..") is not allowed');
  }

  const resolved = path.resolve(inputPath);

  // Block system-critical directories (case-insensitive on Windows)
  const BLOCKED_PREFIXES_WIN = [
    'C:\\Windows', 'C:\\Program Files', 'C:\\Program Files (x86)',
    'C:\\ProgramData\\ssh', 'C:\\System Volume Information',
  ];
  const BLOCKED_PREFIXES_UNIX = [
    '/etc', '/var', '/usr', '/root', '/boot', '/sbin', '/bin', '/lib',
    '/proc', '/sys', '/dev',
  ];
  const BLOCKED_EXACT = ['/'];

  const check = process.platform === 'win32' ? resolved.toLowerCase() : resolved;

  if (process.platform === 'win32') {
    for (const prefix of BLOCKED_PREFIXES_WIN) {
      if (check.startsWith(prefix.toLowerCase()) || check === prefix.toLowerCase()) {
        throw new Error(`Access to system directory "${prefix}" is blocked`);
      }
    }
  } else {
    for (const prefix of BLOCKED_PREFIXES_UNIX) {
      if (check === prefix || check.startsWith(prefix + '/')) {
        throw new Error(`Access to system directory "${prefix}" is blocked`);
      }
    }
  }

  if (BLOCKED_EXACT.includes(check)) {
    throw new Error('Access to root directory is blocked');
  }

  // If mustExist, verify existence and resolve symlinks to check real target
  if (mustExist) {
    if (!fs.existsSync(resolved)) {
      throw new Error(`Path does not exist: ${resolved}`);
    }
    try {
      const realPath = fs.realpathSync(resolved);
      const realCheck = process.platform === 'win32' ? realPath.toLowerCase() : realPath;
      if (process.platform === 'win32') {
        for (const prefix of BLOCKED_PREFIXES_WIN) {
          if (realCheck.startsWith(prefix.toLowerCase())) {
            throw new Error(`Symlink target resolves to blocked directory "${prefix}"`);
          }
        }
      } else {
        for (const prefix of BLOCKED_PREFIXES_UNIX) {
          if (realCheck === prefix || realCheck.startsWith(prefix + '/')) {
            throw new Error(`Symlink target resolves to blocked directory "${prefix}"`);
          }
        }
        if (BLOCKED_EXACT.includes(realCheck)) {
          throw new Error('Symlink target resolves to root directory');
        }
      }
    } catch (e) {
      if (e.message.includes('blocked') || e.message.includes('Symlink')) throw e;
      // realpathSync failed for other reasons (broken symlink, etc.) — allow if path itself exists
    }
  }

  return resolved;
}

// ---- Memory Storage Config ----
const MEMORY_CONFIG_FILE = path.join(require('os').homedir(), '.cerebro', 'memory-config.json');

function loadMemoryConfig() {
  try {
    if (fs.existsSync(MEMORY_CONFIG_FILE)) {
      const cfg = JSON.parse(fs.readFileSync(MEMORY_CONFIG_FILE, 'utf-8'));
      cfg.source = 'file';
      return cfg;
    }
  } catch (e) {
    console.error('[Main] Failed to load memory config:', e.message);
  }

  // Auto-detect from CEREBRO_DATA_DIR env var
  const envPath = process.env.CEREBRO_DATA_DIR;
  if (envPath) {
    try {
      if (fs.existsSync(envPath)) {
        const config = { storagePath: envPath, source: 'env', envVar: 'CEREBRO_DATA_DIR' };
        // Auto-create config file so Docker compose picks up the mount
        fs.mkdirSync(path.dirname(MEMORY_CONFIG_FILE), { recursive: true });
        fs.writeFileSync(MEMORY_CONFIG_FILE, JSON.stringify({ storagePath: envPath }, null, 2));
        console.log(`[Main] Auto-created memory config from CEREBRO_DATA_DIR: ${envPath}`);
        return config;
      } else {
        console.warn(`[Main] CEREBRO_DATA_DIR set to "${envPath}" but path does not exist — skipping`);
      }
    } catch (e) {
      console.error('[Main] Failed to auto-create memory config from env:', e.message);
    }
  }

  return { storagePath: null, source: 'default' };
}

function saveMemoryConfig(config) {
  fs.writeFileSync(MEMORY_CONFIG_FILE, JSON.stringify(config, null, 2));
}

function getDynamicMcpConfig() {
  const config = loadMemoryConfig();
  const volumeArg = config.storagePath
    ? `${config.storagePath.replace(/\\/g, '/')}:/data/memory`
    : 'cerebro_cerebro-data:/data/memory';

  return {
    mcpServers: {
      cerebro: {
        command: 'docker',
        args: [
          'run', '--rm', '-i',
          '-v', volumeArg,
          '-e', 'CEREBRO_DATA_DIR=/data/memory',
          '-e', 'CEREBRO_STANDALONE=1',
          'ghcr.io/professor-low/cerebro-memory:latest',
          'cerebro', 'serve',
        ],
        env: {},
      },
    },
  };
}

ipcMain.handle('get-memory-config', () => {
  const cfg = loadMemoryConfig();
  // Return source and envVar so frontend can show provenance
  return { storagePath: cfg.storagePath, source: cfg.source || 'default', envVar: cfg.envVar || null };
});

ipcMain.handle('browse-storage-folder', async () => {
  try {
    const win = BrowserWindow.getFocusedWindow() || mainWindow;
    const result = await dialog.showOpenDialog(win, {
      properties: ['openDirectory'],
      title: 'Select memory storage location',
    });
    if (result.canceled || !result.filePaths.length) return { canceled: true };
    return { canceled: false, path: result.filePaths[0] };
  } catch (err) {
    return { canceled: true, error: err.message };
  }
});

ipcMain.handle('set-storage-path', async (_event, newPath) => {
  try {
    const validatedPath = validatePath(newPath, { allowNull: true });
    const config = loadMemoryConfig();
    config.storagePath = validatedPath; // null means Docker volume default
    saveMemoryConfig(config);
    // Rewrite compose file with new mount
    await dockerManager.writeComposeFile();
    // Sync MCP config so Claude Code uses the same storage path
    const mcpConfig = getDynamicMcpConfig();
    await writeMcpConfig(mcpConfig.mcpServers);
    console.log(`[MCP] Config synced with storage path: ${newPath || 'Docker volume (default)'}`);
    return { success: true };
  } catch (err) {
    return { success: false, error: err.message };
  }
});

ipcMain.handle('get-storage-stats', async (_event, folderPath) => {
  try {
    const targetPath = folderPath
      ? validatePath(folderPath, { mustExist: true })
      : path.join(require('os').homedir(), '.cerebro');
    let totalSize = 0;
    let fileCount = 0;

    const MAX_DEPTH = 10;
    function walkDir(dir, depth = 0) {
      if (depth > MAX_DEPTH) return;
      try {
        const entries = fs.readdirSync(dir, { withFileTypes: true });
        for (const entry of entries) {
          const fullPath = path.join(dir, entry.name);
          try {
            if (entry.isSymbolicLink()) continue; // skip symlinks
            if (entry.isDirectory()) {
              walkDir(fullPath, depth + 1);
            } else if (entry.isFile()) {
              fileCount++;
              totalSize += fs.statSync(fullPath).size;
            }
          } catch (e) { /* skip inaccessible files */ }
        }
      } catch (e) { /* skip inaccessible dirs */ }
    }

    walkDir(targetPath);

    let humanSize;
    if (totalSize < 1024) humanSize = totalSize + ' B';
    else if (totalSize < 1024 * 1024) humanSize = (totalSize / 1024).toFixed(1) + ' KB';
    else if (totalSize < 1024 * 1024 * 1024) humanSize = (totalSize / (1024 * 1024)).toFixed(1) + ' MB';
    else humanSize = (totalSize / (1024 * 1024 * 1024)).toFixed(2) + ' GB';

    return { totalSize, humanSize, fileCount };
  } catch (err) {
    return { totalSize: 0, humanSize: '0 B', fileCount: 0, error: err.message };
  }
});

ipcMain.handle('scan-merge-preview', async (_event, sourcePath) => {
  try {
    validatePath(sourcePath, { mustExist: true });
    const config = loadMemoryConfig();
    const destPath = config.storagePath;

    // Skip these folders/files during merge
    const SKIP_DIRS = new Set(['embeddings', 'indexes', 'cache', '__pycache__']);
    const SKIP_FILES = new Set(['keyword_index.db', 'quick_facts.json']);
    const SKIP_EXTS = new Set(['.lock']);

    const MAX_DEPTH = 10;
    function listFilesRecursive(dir, base, depth = 0) {
      if (depth > MAX_DEPTH) return [];
      const results = [];
      try {
        const entries = fs.readdirSync(dir, { withFileTypes: true });
        for (const entry of entries) {
          if (entry.isSymbolicLink()) continue; // skip symlinks
          const rel = base ? base + '/' + entry.name : entry.name;
          if (entry.isDirectory()) {
            if (SKIP_DIRS.has(entry.name)) continue;
            results.push(...listFilesRecursive(path.join(dir, entry.name), rel, depth + 1));
          } else if (entry.isFile()) {
            if (SKIP_FILES.has(entry.name) || SKIP_EXTS.has(path.extname(entry.name))) continue;
            results.push(rel);
          }
        }
      } catch (e) { /* skip inaccessible */ }
      return results;
    }

    // Get source file list
    const sourceFiles = new Set(listFilesRecursive(sourcePath, ''));

    // Get destination file list
    let destFiles;
    if (!destPath) {
      // Docker volume — list files via docker exec
      const { execFile: ef } = require('child_process');
      const listing = await new Promise((resolve, reject) => {
        ef('docker', ['exec', 'cerebro-backend-1', 'find', '/data/memory', '-type', 'f', '-printf', '%P\\n'], { timeout: 30000 }, (err, stdout) => {
          if (err) reject(new Error('Docker listing failed: ' + err.message));
          else resolve(stdout);
        });
      });
      destFiles = new Set(listing.trim().split('\n').filter(Boolean).map(f => f.replace(/\\/g, '/')));
    } else {
      destFiles = new Set(listFilesRecursive(destPath, ''));
    }

    // Build per-folder summary
    const folderMap = {};
    for (const f of sourceFiles) {
      const folder = f.includes('/') ? f.split('/')[0] : '(root)';
      if (!folderMap[folder]) folderMap[folder] = { name: folder, newFiles: 0, existingFiles: 0, skipped: 0, size: 0 };
      if (destFiles.has(f)) {
        folderMap[folder].existingFiles++;
      } else {
        folderMap[folder].newFiles++;
        try { folderMap[folder].size += fs.statSync(path.join(sourcePath, f)).size; } catch (e) { /* skip */ }
      }
    }

    const folders = Object.values(folderMap).sort((a, b) => b.newFiles - a.newFiles);
    const totalNew = folders.reduce((s, f) => s + f.newFiles, 0);
    const totalSkipped = folders.reduce((s, f) => s + f.existingFiles, 0);

    // Total size from per-folder sizes
    const sourceSize = folders.reduce((s, f) => s + f.size, 0);

    let humanSize;
    if (sourceSize < 1024) humanSize = sourceSize + ' B';
    else if (sourceSize < 1024 * 1024) humanSize = (sourceSize / 1024).toFixed(1) + ' KB';
    else if (sourceSize < 1024 * 1024 * 1024) humanSize = (sourceSize / (1024 * 1024)).toFixed(1) + ' MB';
    else humanSize = (sourceSize / (1024 * 1024 * 1024)).toFixed(2) + ' GB';

    return { folders, totalNew, totalSkipped, sourceSize, humanSize };
  } catch (err) {
    return { error: err.message };
  }
});

ipcMain.handle('merge-storage', async (_event, sourcePath, selectedFolders) => {
  try {
    validatePath(sourcePath, { mustExist: true });

    // Validate selectedFolders: no path separators or traversal
    if (Array.isArray(selectedFolders)) {
      for (const folder of selectedFolders) {
        if (typeof folder !== 'string' || folder.includes('/') || folder.includes('\\') || folder.includes('..')) {
          throw new Error(`Invalid folder name: "${folder}"`);
        }
      }
    }

    const config = loadMemoryConfig();
    const destPath = config.storagePath;

    const SKIP_DIRS = new Set(['embeddings', 'indexes', 'cache', '__pycache__']);
    const SKIP_FILES = new Set(['keyword_index.db', 'quick_facts.json']);
    const SKIP_EXTS = new Set(['.lock']);

    const MAX_DEPTH = 10;
    function listFilesRecursive(dir, base, depth = 0) {
      if (depth > MAX_DEPTH) return [];
      const results = [];
      try {
        const entries = fs.readdirSync(dir, { withFileTypes: true });
        for (const entry of entries) {
          if (entry.isSymbolicLink()) continue; // skip symlinks
          const rel = base ? base + '/' + entry.name : entry.name;
          if (entry.isDirectory()) {
            if (SKIP_DIRS.has(entry.name)) continue;
            results.push(...listFilesRecursive(path.join(dir, entry.name), rel, depth + 1));
          } else if (entry.isFile()) {
            if (SKIP_FILES.has(entry.name) || SKIP_EXTS.has(path.extname(entry.name))) continue;
            results.push(rel);
          }
        }
      } catch (e) { /* skip inaccessible */ }
      return results;
    }

    let sourceFiles = listFilesRecursive(sourcePath, '');

    // Filter by selected folders if provided
    if (Array.isArray(selectedFolders) && selectedFolders.length > 0) {
      const folderSet = new Set(selectedFolders);
      sourceFiles = sourceFiles.filter(f => {
        const folder = f.includes('/') ? f.split('/')[0] : '(root)';
        return folderSet.has(folder);
      });
    }

    let copied = 0;
    let skipped = 0;
    const errors = [];

    if (!destPath) {
      // Destination is Docker volume — use docker cp per file via temp staging
      const { execFile: ef } = require('child_process');
      const execPromise = (cmd, args, opts) => new Promise((resolve, reject) => {
        ef(cmd, args, opts, (err, stdout) => {
          if (err) reject(err);
          else resolve(stdout);
        });
      });

      // Get existing files in Docker volume
      const listing = await execPromise('docker', ['exec', 'cerebro-backend-1', 'find', '/data/memory', '-type', 'f', '-printf', '%P\\n'], { timeout: 30000 });
      const destFiles = new Set(listing.trim().split('\n').filter(Boolean).map(f => f.replace(/\\/g, '/')));

      for (const relFile of sourceFiles) {
        // Validate relFile has no traversal before docker cp
        if (relFile.includes('..')) {
          errors.push({ file: relFile, error: 'Path traversal blocked' });
          continue;
        }
        if (destFiles.has(relFile)) {
          skipped++;
          continue;
        }
        try {
          // Ensure directory exists in container
          const dirInContainer = '/data/memory/' + relFile.split('/').slice(0, -1).join('/');
          if (dirInContainer !== '/data/memory/') {
            await execPromise('docker', ['exec', 'cerebro-backend-1', 'mkdir', '-p', dirInContainer], { timeout: 10000 });
          }
          // Copy file into container
          const srcFull = path.join(sourcePath, relFile);
          await execPromise('docker', ['cp', srcFull, 'cerebro-backend-1:/data/memory/' + relFile], { timeout: 30000 });
          copied++;
        } catch (e) {
          errors.push({ file: relFile, error: e.message });
        }
      }
    } else {
      // Destination is a local folder
      const destFiles = new Set(listFilesRecursive(destPath, ''));

      for (const relFile of sourceFiles) {
        if (destFiles.has(relFile)) {
          skipped++;
          continue;
        }
        try {
          const srcFull = path.join(sourcePath, relFile);
          const dstFull = path.join(destPath, relFile);
          // Ensure parent directory exists
          fs.mkdirSync(path.dirname(dstFull), { recursive: true });
          fs.copyFileSync(srcFull, dstFull);
          copied++;
        } catch (e) {
          errors.push({ file: relFile, error: e.message });
        }
      }
    }

    return { success: true, copied, skipped, errors };
  } catch (err) {
    return { success: false, copied: 0, skipped: 0, error: err.message, errors: [] };
  }
});

ipcMain.handle('migrate-storage', async (_event, destPath) => {
  try {
    const validatedDest = validatePath(destPath);
    const config = loadMemoryConfig();
    const sourcePath = config.storagePath;

    // Ensure destination exists
    fs.mkdirSync(validatedDest, { recursive: true });

    if (!sourcePath) {
      // Extract from Docker volume
      const { execFile: ef } = require('child_process');
      await new Promise((resolve, reject) => {
        ef('docker', ['cp', 'cerebro-backend-1:/data/memory/.', validatedDest], { timeout: 120000 }, (err) => {
          if (err) reject(new Error('Docker copy failed: ' + err.message));
          else resolve();
        });
      });
    } else {
      // Copy from custom path to new path
      const { execFile: ef } = require('child_process');
      if (process.platform === 'win32') {
        await new Promise((resolve, reject) => {
          ef('xcopy', [sourcePath, validatedDest, '/E', '/I', '/H', '/Y'], { timeout: 120000 }, (err) => {
            if (err) reject(new Error('Copy failed: ' + err.message));
            else resolve();
          });
        });
      } else {
        await new Promise((resolve, reject) => {
          ef('cp', ['-r', sourcePath + '/.', validatedDest], { timeout: 120000 }, (err) => {
            if (err) reject(new Error('Copy failed: ' + err.message));
            else resolve();
          });
        });
      }
    }

    return { success: true };
  } catch (err) {
    return { success: false, error: err.message };
  }
});

ipcMain.handle('restart-computer', () => {
  const { execFile } = require('child_process');
  if (process.platform === 'win32') {
    execFile('shutdown', ['/r', '/t', '5'], (err) => {
      if (err) console.error('[Main] Restart failed:', err.message);
    });
  } else {
    execFile('loginctl', ['reboot'], (err) => {
      if (err) console.error('[Main] Restart failed:', err.message);
    });
  }
  return true;
});
